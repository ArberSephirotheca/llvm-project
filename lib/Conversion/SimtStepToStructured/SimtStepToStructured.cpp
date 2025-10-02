#include "simt-step/Conversion/SimtStepToStructured.h"

#include "simt-step/Dialect/SimtStructured/StructuredDialect.h"
#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>

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

    if (failed(lowerStructuredControlToCFG(func))) {
      signalPassFailure();
      return;
    }

    if (func.getBody().empty())
      return;

    Block &entryBlock = func.front();

    SmallVector<Block *> originalBlocks;
    DenseMap<Block *, SmallVector<Operation *>> blockOriginalOps;
    for (Block &block : func) {
      originalBlocks.push_back(&block);
      for (Operation &op : block)
        blockOriginalOps[&block].push_back(&op);
    }

    OpBuilder topBuilder(&entryBlock, entryBlock.begin());
    auto loc = func.getLoc();

    IRMapping mapper;
    DenseMap<Block *, simt::structured::BlockOp> blockMapping;
    DenseMap<Block *, mlir::Value> blockActiveMask;

    for (auto [index, block] : llvm::enumerate(originalBlocks)) {
      std::string name = (index == 0) ? std::string("entry")
                                      : ("block" + std::to_string(index));
      auto symName = topBuilder.getStringAttr(name);
      auto blockOp = topBuilder.create<simt::structured::BlockOp>(
          loc, symName, mlir::Value(), FlatSymbolRefAttr(), FlatSymbolRefAttr(),
          simt::structured::ReconvergencePolicyAttr());
      topBuilder.setInsertionPointAfter(blockOp);

      mlir::Region &region = blockOp.getBody();
      if (region.empty())
        region.emplaceBlock();
      Block &structuredBlock = region.front();
      for (BlockArgument arg : block->getArguments()) {
        auto newArg = structuredBlock.addArgument(arg.getType(), arg.getLoc());
        mapper.map(arg, newArg);
      }

      blockMapping[block] = blockOp;
    }

    auto getOrCreateMask = [&](Block *origBlock, Block &structuredBlock,
                               mlir::Location loc) -> mlir::Value {
      auto it = blockActiveMask.find(origBlock);
      if (it != blockActiveMask.end())
        return it->second;

      OpBuilder maskBuilder = OpBuilder::atBlockBegin(&structuredBlock);
      auto mask = maskBuilder
                      .create<simt::dialect::ActiveMaskOp>(loc,
                                                          maskBuilder.getI64Type())
                      .getResult();
      blockActiveMask[origBlock] = mask;
      return mask;
    };

    SmallVector<Value> functionReturnValues;
    bool hasFunctionReturn = false;

    for (Block *origBlock : originalBlocks) {
      simt::structured::BlockOp blockOp = blockMapping[origBlock];
      Block &structuredBlock = blockOp.getBody().front();

      BlockControlInfo meta;
      bool hasMeta = false;
      if (auto it = blockControlInfo.find(origBlock); it != blockControlInfo.end()) {
        meta = it->second;
        hasMeta = true;
      }

      FlatSymbolRefAttr mergeAttr;
      if (hasMeta && meta.mergeTarget)
        mergeAttr = FlatSymbolRefAttr::get(
            blockOp.getContext(),
            blockMapping[meta.mergeTarget].getSymNameAttr().getValue());
      FlatSymbolRefAttr continueAttr;
      if (hasMeta && meta.continueTarget)
        continueAttr = FlatSymbolRefAttr::get(
            blockOp.getContext(),
            blockMapping[meta.continueTarget].getSymNameAttr().getValue());

      if (mergeAttr)
        blockOp.setMergeTargetAttr(mergeAttr);
      else
        blockOp->removeAttr(blockOp.getMergeTargetAttrName());

      if (continueAttr)
        blockOp.setContinueTargetAttr(continueAttr);
      else
        blockOp->removeAttr(blockOp.getContinueTargetAttrName());

      Value blockMask = getOrCreateMask(origBlock, structuredBlock, loc);

      Operation *lastInserted = nullptr;
      if (hasMeta && meta.pushMask) {
        Operation *maskDef = blockMask.getDefiningOp();
        OpBuilder pushBuilder(&structuredBlock, structuredBlock.begin());
        if (maskDef && maskDef->getBlock() == &structuredBlock)
          pushBuilder.setInsertionPointAfter(maskDef);
        else
          pushBuilder.setInsertionPointToStart(&structuredBlock);

        auto pushOp = pushBuilder.create<simt::structured::MaskPushOp>(
            loc, blockMask, mergeAttr, continueAttr, IntegerAttr());
        lastInserted = pushOp.getOperation();
      }

      OpBuilder cloneBuilder(&structuredBlock, structuredBlock.begin());
      if (lastInserted)
        cloneBuilder.setInsertionPointAfter(lastInserted);
      else
        cloneBuilder.setInsertionPointToStart(&structuredBlock);

      for (Operation *op : blockOriginalOps[origBlock]) {
        if (isa<cf::BranchOp, cf::CondBranchOp, func::ReturnOp>(op))
          continue;

        Operation *cloned = cloneBuilder.clone(*op, mapper);
        for (auto [index, result] : llvm::enumerate(op->getResults()))
          mapper.map(result, cloned->getResult(index));

        if (auto maskOp = dyn_cast<simt::dialect::ActiveMaskOp>(op))
          blockActiveMask[origBlock] = cloned->getResult(0);
      }

      Operation *terminator = origBlock->getTerminator();
      Location termLoc = terminator->getLoc();

      auto mapValues = [&](ValueRange operands,
                           SmallVectorImpl<Value> &mapped) -> bool {
        for (Value operand : operands) {
          Value mappedValue = mapper.lookupOrNull(operand);
          if (!mappedValue) {
            terminator->emitError()
                << "unmapped operand during SimtStepToStructured lowering";
            return false;
          }
          mapped.push_back(mappedValue);
        }
        return true;
      };

      auto emitMaskPop = [&](OpBuilder &builder) {
        if (hasMeta && meta.popMask)
          (void)builder.create<simt::structured::MaskPopOp>(termLoc,
                                                           blockMask.getType());
      };

      if (auto branch = dyn_cast<cf::BranchOp>(terminator)) {
        SmallVector<Value> destOperands;
        if (!mapValues(branch.getDestOperands(), destOperands)) {
          signalPassFailure();
          return;
        }

        simt::structured::BlockOp destBlockOp = blockMapping[branch.getDest()];
        auto targetAttr = FlatSymbolRefAttr::get(destBlockOp.getContext(),
                                                 destBlockOp.getSymName());

        OpBuilder termBuilder(&structuredBlock, structuredBlock.end());
        emitMaskPop(termBuilder);
        termBuilder.create<simt::structured::BranchOp>(termLoc, blockMask,
                                                       targetAttr, destOperands);
      } else if (auto cond = dyn_cast<cf::CondBranchOp>(terminator)) {
        SmallVector<Value> trueOperands;
        SmallVector<Value> falseOperands;
        if (!mapValues(cond.getTrueDestOperands(), trueOperands) ||
            !mapValues(cond.getFalseDestOperands(), falseOperands)) {
          signalPassFailure();
          return;
        }

        Value condition = mapper.lookup(cond.getCondition());

        simt::structured::BlockOp trueBlockOp = blockMapping[cond.getTrueDest()];
        simt::structured::BlockOp falseBlockOp = blockMapping[cond.getFalseDest()];

        auto trueTarget = FlatSymbolRefAttr::get(trueBlockOp.getContext(),
                                                 trueBlockOp.getSymName());
        auto falseTarget = FlatSymbolRefAttr::get(falseBlockOp.getContext(),
                                                  falseBlockOp.getSymName());

        OpBuilder termBuilder(&structuredBlock, structuredBlock.end());
        emitMaskPop(termBuilder);
        termBuilder.create<simt::structured::CondBranchOp>(
            termLoc, condition, blockMask, blockMask, trueTarget, falseTarget,
            trueOperands, falseOperands, FlatSymbolRefAttr(),
            simt::structured::ReconvergencePolicyAttr());
      } else if (auto ret = dyn_cast<func::ReturnOp>(terminator)) {
        SmallVector<Value> returnValues;
        if (!mapValues(ret.getOperands(), returnValues)) {
          signalPassFailure();
          return;
        }
        OpBuilder termBuilder(&structuredBlock, structuredBlock.end());
        emitMaskPop(termBuilder);
        termBuilder.create<simt::structured::ReturnOp>(termLoc, returnValues);

        if (!hasFunctionReturn) {
          functionReturnValues = returnValues;
          hasFunctionReturn = true;
        } else {
          if (functionReturnValues.size() != returnValues.size()) {
            ret.emitError()
                << "mismatched return arity across SimtStepToStructured lowering";
            signalPassFailure();
            return;
          }
          for (auto [expected, current] : llvm::zip(functionReturnValues,
                                                    returnValues)) {
            if (expected.getType() != current.getType()) {
              ret.emitError()
                  << "mismatched return type across SimtStepToStructured lowering";
              signalPassFailure();
              return;
            }
          }
        }
      } else {
        terminator->emitError()
            << "unsupported terminator in SimtStepToStructured lowering";
        signalPassFailure();
        return;
      }
    }

    for (Block *origBlock : originalBlocks) {
      for (Operation *op : blockOriginalOps[origBlock])
        op->erase();
      if (origBlock != &entryBlock)
        origBlock->erase();
    }

    OpBuilder retBuilder(&entryBlock, entryBlock.end());
    if (hasFunctionReturn)
      retBuilder.create<func::ReturnOp>(loc, functionReturnValues);
    else
      retBuilder.create<func::ReturnOp>(loc);
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
