#include "simt-step/Conversion/SimtStepToStructured.h"
#include "simt-step/Conversion/StructuredCFGBuilder.h"

#include "simt-step/Dialect/SimtStructured/StructuredDialect.h"
#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>

using namespace mlir;

namespace simt::conversion {

namespace {

struct BlockControlInfo {
  Block *mergeTarget = nullptr;
  Block *continueTarget = nullptr;
  bool pushMask = false;
  bool popMask = false;
};

static DenseMap<Block *, BlockControlInfo> blockControlInfo;

static void createCondBranch(OpBuilder &builder, Location loc, Value condition,
                             Block *trueBlock, ValueRange trueOperands,
                             Block *falseBlock, ValueRange falseOperands) {
  OperationState state(loc, cf::CondBranchOp::getOperationName());
  state.addOperands(condition);
  state.addOperands(trueOperands);
  state.addOperands(falseOperands);
  state.addSuccessors({trueBlock, falseBlock});
  SmallVector<int32_t, 3> segments = {
      1, static_cast<int32_t>(trueOperands.size()),
      static_cast<int32_t>(falseOperands.size())};
  state.addAttribute("operand_segment_sizes",
                     builder.getDenseI32ArrayAttr(segments));
  builder.create(state);
}

static void eraseTerminatorIfPresent(Block *block) {
  if (block && !block->empty() && block->back().hasTrait<OpTrait::IsTerminator>())
    block->back().erase();
}

static LogicalResult lowerIfToCFG(simt::dialect::IfOp ifOp) {
  Location loc = ifOp.getLoc();
  Block *parentBlock = ifOp->getBlock();

  Block *afterBlock = parentBlock->splitBlock(ifOp.getOperation());
  SmallVector<Location> resultLocs(ifOp.getNumResults(), loc);
  afterBlock->addArguments(ifOp.getResultTypes(), resultLocs);

  Region &parentRegion = *parentBlock->getParent();
  Block *thenBlock = new Block();
  parentRegion.getBlocks().insert(afterBlock->getIterator(), thenBlock);

  Block *elseBlock = nullptr;
  bool hasElse = !ifOp.getElseRegion().empty();
  if (!hasElse && ifOp.getNumResults() != 0)
    return ifOp.emitError("if without else must not return values"), failure();

  if (hasElse) {
    elseBlock = new Block();
    parentRegion.getBlocks().insert(afterBlock->getIterator(), elseBlock);
  }

  eraseTerminatorIfPresent(parentBlock);
  OpBuilder condBuilder(parentBlock, parentBlock->end());
  Value condition = ifOp.getCondition();

  Block *falseDest = hasElse ? elseBlock : afterBlock;
  ValueRange falseOperands = hasElse ? ValueRange{} : ValueRange{};
  createCondBranch(condBuilder, loc, condition, thenBlock, ValueRange{},
                   falseDest, falseOperands);

  auto &headerInfo = blockControlInfo[parentBlock];
  headerInfo.mergeTarget = afterBlock;
  headerInfo.pushMask = true;
  headerInfo.popMask = true;

  IRMapping thenMapping;
  OpBuilder thenBuilder(thenBlock, thenBlock->begin());
  for (Operation &op : ifOp.getThenRegion().front()) {
    if (auto yield = dyn_cast<simt::dialect::YieldOp>(&op)) {
      SmallVector<Value> results;
      results.reserve(yield.getNumOperands());
      for (Value operand : yield.getOperands())
        results.push_back(thenMapping.lookup(operand));
      thenBuilder.create<cf::BranchOp>(loc, afterBlock, results);
      continue;
    }
    Operation *cloned = thenBuilder.clone(op, thenMapping);
    for (auto [orig, repl] : llvm::zip(op.getResults(), cloned->getResults()))
      thenMapping.map(orig, repl);
  }

  if (hasElse) {
    IRMapping elseMapping;
    OpBuilder elseBuilder(elseBlock, elseBlock->begin());
    for (Operation &op : ifOp.getElseRegion().front()) {
      if (auto yield = dyn_cast<simt::dialect::YieldOp>(&op)) {
        SmallVector<Value> results;
        results.reserve(yield.getNumOperands());
        for (Value operand : yield.getOperands())
          results.push_back(elseMapping.lookup(operand));
        elseBuilder.create<cf::BranchOp>(loc, afterBlock, results);
        continue;
      }
      Operation *cloned = elseBuilder.clone(op, elseMapping);
      for (auto [orig, repl] : llvm::zip(op.getResults(), cloned->getResults()))
        elseMapping.map(orig, repl);
    }
  }

  ifOp.replaceAllUsesWith(afterBlock->getArguments());
  ifOp.erase();
  return success();
}

static LogicalResult lowerLoopToCFG(simt::dialect::LoopOp loopOp) {
  Location loc = loopOp.getLoc();
  Block *parentBlock = loopOp->getBlock();

  Block *afterBlock = parentBlock->splitBlock(loopOp.getOperation());
  SmallVector<Location> resultLocs(loopOp.getNumResults(), loc);
  afterBlock->addArguments(loopOp.getResultTypes(), resultLocs);

  Region &parentRegion = *parentBlock->getParent();

  Block *prepareRegionBlock = &loopOp.getPrepareRegion().front();
  Block *bodyRegionBlock = &loopOp.getBodyRegion().front();

  Block *headerBlock = new Block();
  parentRegion.getBlocks().insert(afterBlock->getIterator(), headerBlock);
  for (BlockArgument arg : prepareRegionBlock->getArguments())
    headerBlock->addArgument(arg.getType(), loc);

  Block *bodyBlock = new Block();
  parentRegion.getBlocks().insert(afterBlock->getIterator(), bodyBlock);
  for (BlockArgument arg : bodyRegionBlock->getArguments())
    bodyBlock->addArgument(arg.getType(), loc);

  eraseTerminatorIfPresent(parentBlock);
  OpBuilder entryBuilder(parentBlock, parentBlock->end());
  entryBuilder.create<cf::BranchOp>(loc, headerBlock, loopOp.getInits());

  auto &headerInfo = blockControlInfo[headerBlock];
  headerInfo.mergeTarget = afterBlock;
  headerInfo.continueTarget = headerBlock;
  headerInfo.pushMask = true;
  headerInfo.popMask = true;

  auto &bodyInfo = blockControlInfo[bodyBlock];
  bodyInfo.continueTarget = headerBlock;
  bodyInfo.pushMask = true;
  bodyInfo.popMask = true;

  IRMapping prepareMapping;
  for (auto [orig, repl] : llvm::zip(prepareRegionBlock->getArguments(),
                                     headerBlock->getArguments()))
    prepareMapping.map(orig, repl);

  OpBuilder headerBuilder(headerBlock, headerBlock->begin());
  for (Operation &op : *prepareRegionBlock) {
    if (auto cond = dyn_cast<simt::dialect::ConditionOp>(&op)) {
      Value condition = prepareMapping.lookup(cond.getCondition());
      SmallVector<Value> trueOperands;
      SmallVector<Value> falseOperands;
      trueOperands.reserve(cond.getForwarded().size());
      falseOperands.reserve(cond.getForwarded().size());
      for (Value operand : cond.getForwarded()) {
        Value mapped = prepareMapping.lookup(operand);
        trueOperands.push_back(mapped);
        falseOperands.push_back(mapped);
      }

      if (trueOperands.size() != bodyBlock->getNumArguments())
        return cond.emitError("condition forwarded count must match body arguments"),
               failure();
      if (falseOperands.size() != afterBlock->getNumArguments())
        return cond.emitError("condition forwarded count must match loop results"),
               failure();

      createCondBranch(headerBuilder, loc, condition, bodyBlock,
                       ValueRange(trueOperands), afterBlock,
                       ValueRange(falseOperands));
      continue;
    }

    Operation *cloned = headerBuilder.clone(op, prepareMapping);
    for (auto [orig, repl] : llvm::zip(op.getResults(), cloned->getResults()))
      prepareMapping.map(orig, repl);
  }

  IRMapping bodyMapping;
  for (auto [orig, repl] : llvm::zip(bodyRegionBlock->getArguments(),
                                     bodyBlock->getArguments()))
    bodyMapping.map(orig, repl);

  OpBuilder bodyBuilder(bodyBlock, bodyBlock->begin());
  for (Operation &op : *bodyRegionBlock) {
    if (auto cont = dyn_cast<simt::dialect::ContinueOp>(&op)) {
      SmallVector<Value> operands;
      operands.reserve(cont.getNumOperands());
      for (Value operand : cont.getOperands())
        operands.push_back(bodyMapping.lookup(operand));
      bodyBuilder.create<cf::BranchOp>(loc, headerBlock, operands);
      continue;
    }

    if (auto brk = dyn_cast<simt::dialect::BreakOp>(&op)) {
      SmallVector<Value> operands;
      operands.reserve(brk.getNumOperands());
      for (Value operand : brk.getOperands())
        operands.push_back(bodyMapping.lookup(operand));
      bodyBuilder.create<cf::BranchOp>(loc, afterBlock, operands);
      continue;
    }

    if (auto yield = dyn_cast<simt::dialect::YieldOp>(&op)) {
      SmallVector<Value> operands;
      operands.reserve(yield.getNumOperands());
      for (Value operand : yield.getOperands())
        operands.push_back(bodyMapping.lookup(operand));
      bodyBuilder.create<cf::BranchOp>(loc, headerBlock, operands);
      continue;
    }

    Operation *cloned = bodyBuilder.clone(op, bodyMapping);
    for (auto [orig, repl] : llvm::zip(op.getResults(), cloned->getResults()))
      bodyMapping.map(orig, repl);
  }

  loopOp.replaceAllUsesWith(afterBlock->getArguments());
  loopOp.erase();
  return success();
}
static LogicalResult lowerSwitchToCFG(simt::dialect::SwitchOp switchOp) {
  Location loc = switchOp.getLoc();
  Block *parentBlock = switchOp->getBlock();

  Block *afterBlock = parentBlock->splitBlock(switchOp);
  unsigned numResults = switchOp.getNumResults();
  SmallVector<Location> resultLocs(numResults, loc);

  SmallVector<Type> carriedTypes;
  SmallVector<Location> carriedLocs;
  for (BlockArgument arg : parentBlock->getArguments()) {
    carriedTypes.push_back(arg.getType());
    carriedLocs.push_back(arg.getLoc());
  }
  unsigned numCarried = carriedTypes.size();

  auto addCarriedArgs = [&](Block *block) {
    if (numCarried == 0)
      return;
    block->addArguments(carriedTypes, carriedLocs);
  };

  auto replaceCarriedUses = [&](Block *block) {
    if (numCarried == 0)
      return;
    auto carriedArgs = block->getArguments().take_front(numCarried);
    for (auto [index, parentArg] : llvm::enumerate(parentBlock->getArguments()))
      parentArg.replaceUsesWithIf(carriedArgs[index], [&](OpOperand &use) {
        return use.getOwner()->getBlock() == block;
      });
  };

  addCarriedArgs(afterBlock);
  afterBlock->addArguments(switchOp.getResultTypes(), resultLocs);
  replaceCarriedUses(afterBlock);

  Region &parentRegion = *parentBlock->getParent();
  Block *headerBlock = new Block();
  parentRegion.getBlocks().insert(afterBlock->getIterator(), headerBlock);
  addCarriedArgs(headerBlock);
  headerBlock->addArguments(switchOp.getResultTypes(), resultLocs);
  replaceCarriedUses(headerBlock);

  Block *exitBlock = new Block();
  parentRegion.getBlocks().insert(afterBlock->getIterator(), exitBlock);
  addCarriedArgs(exitBlock);
  exitBlock->addArguments(switchOp.getResultTypes(), resultLocs);
  replaceCarriedUses(exitBlock);

  SmallVector<DenseMap<Block *, Value>> blockValueForResult(numResults);
  DenseMap<Block *, SmallVector<Value>> blockPayload;
  auto recordPayloadForArgs = [&](Block *block) {
    if (numResults == 0)
      return;
    SmallVector<Value> payload;
    payload.reserve(numResults);
    for (Value arg : block->getArguments().drop_front(numCarried))
      payload.push_back(arg);
    blockPayload[block] = payload;
    for (auto [idx, val] : llvm::enumerate(payload))
      blockValueForResult[idx][block] = val;
  };

  recordPayloadForArgs(afterBlock);
  recordPayloadForArgs(exitBlock);

  SmallVector<Block *, 4> caseBlocks;
  SmallVector<Block *, 4> fallthroughBlocks;
  Region &cases = switchOp.getCaseBody();
  for (Block &origCaseInit : cases) {
    (void)origCaseInit;
    Block *caseBlock = new Block();
    parentRegion.getBlocks().insert(exitBlock->getIterator(), caseBlock);
    addCarriedArgs(caseBlock);
    caseBlock->addArguments(switchOp.getResultTypes(), resultLocs);
    caseBlocks.push_back(caseBlock);
    replaceCarriedUses(caseBlock);
    recordPayloadForArgs(caseBlock);
  }

  auto &headerInfo = blockControlInfo[headerBlock];
  headerInfo.mergeTarget = exitBlock;
  headerInfo.pushMask = true;

  auto &exitInfo = blockControlInfo[exitBlock];
  exitInfo.popMask = true;

  for (Block *caseBlock : caseBlocks) {
    auto &info = blockControlInfo[caseBlock];
    info.popMask = true;
    info.mergeTarget = exitBlock;
  }

  eraseTerminatorIfPresent(parentBlock);
  OpBuilder entryBuilder(parentBlock, parentBlock->end());
  SmallVector<Value> headerOperands;
  headerOperands.reserve(numCarried + numResults);
  headerOperands.append(parentBlock->args_begin(), parentBlock->args_end());
  for (Value value : switchOp.getInitialValues()) {
    if (auto arg = mlir::dyn_cast<BlockArgument>(value)) {
      if (arg.getOwner() == afterBlock)
        value = parentBlock->getArgument(arg.getArgNumber());
    }
    headerOperands.push_back(value);
  }
  entryBuilder.create<cf::BranchOp>(loc, headerBlock, headerOperands);

  OpBuilder headerBuilder(headerBlock, headerBlock->begin());
  if (caseBlocks.empty()) {
    headerBuilder.create<cf::BranchOp>(loc, exitBlock, headerBlock->getArguments());
  } else {
    headerBuilder.create<cf::BranchOp>(loc, caseBlocks.front(), headerBlock->getArguments());
  }

  OpBuilder exitBuilder(exitBlock, exitBlock->begin());
  exitBuilder.create<cf::BranchOp>(loc, afterBlock, exitBlock->getArguments());

  if (!caseBlocks.empty()) {
    unsigned numResults = switchOp.getNumResults();
    unsigned payloadCount = numResults >= 3 ? numResults - 3 : 0;

    auto cloneCase = [&](Block &origCase, Block *newCase, Block *nextCase) {
      IRMapping mapping;
      auto newCaseResultArgs = newCase->getArguments().drop_front(numCarried);
      for (auto [origArg, newArg] : llvm::zip(origCase.getArguments(),
                                              newCaseResultArgs))
        mapping.map(origArg, newArg);

      OpBuilder caseBuilder(newCase, newCase->begin());
      for (Operation &op : origCase.without_terminator()) {
        Operation *cloned = caseBuilder.clone(op, mapping);
        for (auto [origRes, newRes] : llvm::zip(op.getResults(), cloned->getResults()))
          mapping.map(origRes, newRes);
      }

      auto yieldOp = cast<simt::dialect::YieldOp>(origCase.getTerminator());
      SmallVector<Value> yieldValues;
      yieldValues.reserve(numResults);
      for (Value operand : yieldOp.getResults())
        yieldValues.push_back(mapping.lookup(operand));

      Value fallthrough = numResults > payloadCount + 1 ? yieldValues[payloadCount + 1]
                                                        : Value();
      Value switchDone = numResults > payloadCount + 2 ? yieldValues[payloadCount + 2]
                                                       : Value();

      OpBuilder termBuilder(newCase, newCase->end());
      Value constFalse = termBuilder.create<mlir::arith::ConstantIntOp>(loc, 0, 1);
      Value fallthroughCond = constFalse;
      if (fallthrough && switchDone && nextCase != exitBlock) {
        Value notDone = termBuilder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::eq, switchDone, constFalse);
        fallthroughCond = termBuilder.create<mlir::arith::AndIOp>(loc, fallthrough, notDone);
      }

      SmallVector<Value, 8> branchOperands;
      if (numCarried != 0) {
        auto carriedArgs = newCase->getArguments().take_front(numCarried);
        branchOperands.append(carriedArgs.begin(), carriedArgs.end());
      }
      branchOperands.append(yieldValues.begin(), yieldValues.end());
      Block *fallthroughBlock = nullptr;
      if (nextCase != exitBlock) {
        fallthroughBlock = new Block();
        parentRegion.getBlocks().insert(nextCase->getIterator(), fallthroughBlock);
        addCarriedArgs(fallthroughBlock);
        fallthroughBlock->addArguments(switchOp.getResultTypes(), resultLocs);

        auto &fallthroughInfo = blockControlInfo[fallthroughBlock];
        fallthroughInfo.pushMask = true;
        fallthroughInfo.mergeTarget = exitBlock;
        recordPayloadForArgs(fallthroughBlock);
        fallthroughBlocks.push_back(fallthroughBlock);
      } else {
        fallthroughCond = constFalse;
      }

      Block *trueDest = fallthroughBlock ? fallthroughBlock : exitBlock;
      createCondBranch(termBuilder, loc, fallthroughCond, trueDest, branchOperands,
                       exitBlock, branchOperands);

      if (fallthroughBlock) {
        OpBuilder ftBuilder(fallthroughBlock, fallthroughBlock->begin());
        ftBuilder.create<cf::BranchOp>(loc, nextCase, fallthroughBlock->getArguments());
        replaceCarriedUses(fallthroughBlock);
      }

      replaceCarriedUses(newCase);
      if (numResults != 0) {
        SmallVector<Value> payload(yieldValues.begin(), yieldValues.end());
        blockPayload[newCase] = payload;
        for (auto [idx, val] : llvm::enumerate(payload))
          blockValueForResult[idx][newCase] = val;
      }
    };

    for (auto [index, origCase] : llvm::enumerate(cases)) {
      Block *nextCase = (index + 1 < caseBlocks.size()) ? caseBlocks[index + 1]
                                                        : exitBlock;
      cloneCase(origCase, caseBlocks[index], nextCase);
    }
  }

  ValueRange initialValues = switchOp.getInitialValues();
  if (numResults != 0) {
    SmallVector<Value> payload(initialValues.begin(), initialValues.end());
    blockPayload[parentBlock] = payload;
    for (auto [idx, val] : llvm::enumerate(payload))
      blockValueForResult[idx][parentBlock] = val;
  }

  auto appendMissingDestOperands = [&](Block *target) {
    unsigned expected = target->getNumArguments();
    if (expected == numCarried)
      return;

    for (Block *pred : target->getPredecessors()) {
      Operation *terminator = pred->getTerminator();
      auto appendRange = [&](MutableOperandRange destOperands) {
        if (destOperands.size() == expected)
          return;
        assert(destOperands.size() == numCarried &&
               "branch missing carried operands during switch lowering");

        llvm::errs() << "Switch lowering: adjusting edge " << pred << " -> "
                     << target << " (" << destOperands.size() << " -> "
                     << expected << ")\n";

        auto payloadIt = blockPayload.find(pred);
        if (payloadIt == blockPayload.end()) {
          SmallVector<Value> inferredPayload;
          inferredPayload.reserve(numResults);
          for (unsigned idx = 0; idx < numResults; ++idx) {
            auto &valueMap = blockValueForResult[idx];
            if (auto it = valueMap.find(pred); it != valueMap.end())
              inferredPayload.push_back(it->second);
            else
              inferredPayload.push_back(afterBlock->getArgument(numCarried + idx));
          }
          payloadIt = blockPayload.try_emplace(pred, inferredPayload).first;
        }

        SmallVector<Value> newOperands;
        newOperands.reserve(destOperands.size() + payloadIt->second.size());
        for (OpOperand &operand : destOperands)
          newOperands.push_back(operand.get());
        newOperands.append(payloadIt->second.begin(), payloadIt->second.end());
        destOperands.assign(newOperands);

        SmallVector<Value> updatedPayload(newOperands.begin() + numCarried,
                                          newOperands.end());
        blockPayload[pred] = updatedPayload;
        for (auto [idx, val] : llvm::enumerate(updatedPayload))
          blockValueForResult[idx][pred] = val;

        llvm::errs() << "  payload size now " << updatedPayload.size() << "\n";
      };

      if (auto branch = dyn_cast<cf::BranchOp>(terminator)) {
        if (branch.getDest() == target)
          appendRange(branch.getDestOperandsMutable());
        continue;
      }

      if (auto cond = dyn_cast<cf::CondBranchOp>(terminator)) {
        if (cond.getTrueDest() == target)
          appendRange(cond.getTrueDestOperandsMutable());
        if (cond.getFalseDest() == target)
          appendRange(cond.getFalseDestOperandsMutable());
        continue;
      }
    }
  };

  SmallVector<Block *, 8> operandTargets;
  operandTargets.push_back(exitBlock);
  operandTargets.push_back(afterBlock);
  operandTargets.append(caseBlocks.begin(), caseBlocks.end());
  operandTargets.append(fallthroughBlocks.begin(), fallthroughBlocks.end());

  llvm::DenseSet<Block *> visitedTargets;
  for (Block *target : operandTargets)
    if (visitedTargets.insert(target).second)
      appendMissingDestOperands(target);

  if (numResults != 0) {
    for (auto [idx, result] : llvm::enumerate(switchOp.getResults())) {
      Value fallback = afterBlock->getArgument(numCarried + idx);
      auto &valueMap = blockValueForResult[idx];
      for (auto &use : llvm::make_early_inc_range(result.getUses())) {
        Block *useBlock = use.getOwner()->getBlock();
        Value replacement = fallback;
        if (auto it = valueMap.find(useBlock); it != valueMap.end())
          replacement = it->second;
        else
          llvm::errs() << "Switch result " << idx
                       << " using fallback in block " << useBlock << "\n";
        use.set(replacement);
      }
    }
  }

  switchOp.erase();
  return success();
}


static LogicalResult lowerStructuredControlToCFG(func::FuncOp func) {
  blockControlInfo.clear();
  bool progress = true;
  while (progress) {
    progress = false;

    SmallVector<simt::dialect::IfOp, 8> ifOps;
    func.walk([&](simt::dialect::IfOp ifOp) { ifOps.push_back(ifOp); });
    for (auto ifOp : llvm::reverse(ifOps)) {
      if (failed(lowerIfToCFG(ifOp)))
        return failure();
      progress = true;
    }

    SmallVector<simt::dialect::LoopOp, 8> loopOps;
    func.walk([&](simt::dialect::LoopOp loopOp) { loopOps.push_back(loopOp); });
    for (auto loopOp : llvm::reverse(loopOps)) {
      if (failed(lowerLoopToCFG(loopOp)))
        return failure();
      progress = true;
    }

    SmallVector<simt::dialect::SwitchOp, 8> switchOps;
    func.walk([&](simt::dialect::SwitchOp switchOp) { switchOps.push_back(switchOp); });
    for (auto switchOp : llvm::reverse(switchOps)) {
      if (failed(lowerSwitchToCFG(switchOp))) {
        switchOp.emitError("failed to lower simt.switch to CFG");
        return failure();
      }
      progress = true;
    }
  }

  return success();
}

struct SimtStepToStructuredPass
    : public PassWrapper<SimtStepToStructuredPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimtStepToStructuredPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<simt::structured::SimtStructDialect, cf::ControlFlowDialect>();
  }

  StringRef getArgument() const final { return "simt-step-to-structured"; }
  StringRef getDescription() const final {
    return "Lower simt_step operations into the structured SIMT dialect";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    StructuredCFGBuilder builder(func);
    if (failed(builder.build())) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<Pass> createSimtStepToStructuredPass() {
  return std::make_unique<SimtStepToStructuredPass>();
}

void registerSimtStepToStructuredPass() {
  PassRegistration<SimtStepToStructuredPass>();
}

} // namespace simt::conversion
