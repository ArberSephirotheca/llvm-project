#include "simt-step/Conversion/StructuredCFGBuilder.h"

#include "simt-step/Dialect/SimtStructured/StructuredDialect.h"
#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"

#include <llvm/Support/Casting.h>

#include <llvm/ADT/StringRef.h>

using namespace mlir;

namespace simt::conversion {

namespace {

constexpr llvm::StringLiteral kUnimplementedMsg(
    "StructuredCFGBuilder skeleton reached. Implement the new builder.");

static LogicalResult signalUnimplemented(FunctionOpInterface func) {
  func.emitError(kUnimplementedMsg);
  return failure();
}

} // namespace

struct StructuredCFGBuilder::BlockInfo {
  mlir::Block *original = nullptr;
  SmallVector<mlir::Type, 4> carriedTypes;
  SmallVector<mlir::Value, 4> payloadSeed;
  SmallVector<mlir::BlockArgument, 4> blockArgs;
  SmallVector<mlir::Operation *, 4> controlOps;

  simt::structured::BlockOp structuredOp;
  mlir::Block *structuredBody = nullptr;

  mlir::Block *mergeTarget = nullptr;
  mlir::Block *continueTarget = nullptr;

  bool requestsMaskPush = false;
  bool requestsMaskPop = false;
};

struct StructuredCFGBuilder::EdgeInfo {
  BlockInfo *source = nullptr;
  BlockInfo *dest = nullptr;
  SmallVector<mlir::Value, 8> payload;
  SmallVector<mlir::Value, 4> maskValues;
  enum Kind { Plain, ConditionalTrue, ConditionalFalse, LoopBackEdge } kind =
      Plain;
};

struct StructuredCFGBuilder::IfInfo {
  mlir::Operation *op = nullptr;
  BlockInfo *parent = nullptr;
  BlockInfo *thenBlock = nullptr;
  BlockInfo *elseBlock = nullptr;
};

struct StructuredCFGBuilder::LoopInfo {
  mlir::Operation *op = nullptr;
  BlockInfo *parent = nullptr;
  BlockInfo *prepareBlock = nullptr;
  BlockInfo *bodyBlock = nullptr;
  SmallVector<mlir::Value, 4> forwardedToBody;
  SmallVector<mlir::Value, 4> forwardedToExit;
};

struct StructuredCFGBuilder::SwitchInfo {
  mlir::Operation *op = nullptr;
  BlockInfo *parent = nullptr;
  SmallVector<BlockInfo *, 4> caseBlocks;
  BlockInfo *defaultBlock = nullptr;
  SmallVector<int64_t, 8> caseValues;
};

StructuredCFGBuilder::StructuredCFGBuilder(FunctionOpInterface func)
    : func(func) {}

LogicalResult StructuredCFGBuilder::build() {
  mapper = std::make_unique<IRMapping>();
  domInfo = std::make_unique<DominanceInfo>(func);

  collectOriginalBlocks();

  if (blockOrder.empty())
    return success();

  // TODO: implement the staged builder pipeline described in
  // docs/structured_cfg_builder_plan.md once analysis and payload propagation
  // logic lands.
  if (failed(analyseBlocks()))
    return failure();
  if (failed(computePayloads()))
    return failure();
  if (failed(enumerateEdges()))
    return failure();
  if (failed(emitStructuredBlocks()))
    return failure();
  if (failed(cleanupOriginalCFG()))
    return failure();

  return success();
}

void StructuredCFGBuilder::collectOriginalBlocks() {
  blockOrder.clear();
  if (!func)
    return;

  SmallVector<Region *, 8> worklist;
  for (Region &region : func->getOperation()->getRegions())
    worklist.push_back(&region);

  while (!worklist.empty()) {
    Region *region = worklist.pop_back_val();
    if (!region)
      continue;
    for (Block &block : *region) {
      blockOrder.push_back(&block);
      for (Operation &nestedOp : block)
        for (Region &nestedRegion : nestedOp.getRegions())
          if (!nestedRegion.empty())
            worklist.push_back(&nestedRegion);
    }
  }
}

StructuredCFGBuilder::BlockInfo &
StructuredCFGBuilder::getOrCreateBlockInfo(mlir::Block *block) {
  auto [it, inserted] = blockInfos.try_emplace(block);
  BlockInfo &info = it->second;
  if (inserted) {
    info.original = block;
    info.carriedTypes.reserve(block->getNumArguments());
    info.blockArgs.reserve(block->getNumArguments());
    for (mlir::BlockArgument arg : block->getArguments()) {
      info.carriedTypes.push_back(arg.getType());
      info.blockArgs.push_back(arg);
    }
  }
  return info;
}

StructuredCFGBuilder::BlockInfo *
StructuredCFGBuilder::lookupBlockInfo(mlir::Block *block) {
  if (auto it = blockInfos.find(block); it != blockInfos.end())
    return &it->second;
  return nullptr;
}

const StructuredCFGBuilder::BlockInfo *
StructuredCFGBuilder::lookupBlockInfo(mlir::Block *block) const {
  if (auto it = blockInfos.find(block); it != blockInfos.end())
    return &it->second;
  return nullptr;
}

LogicalResult StructuredCFGBuilder::analyseBlocks() {
  blockInfos.clear();
  ifInfos.clear();
  loopInfos.clear();
  switchInfos.clear();

  for (mlir::Block *block : blockOrder)
    (void)getOrCreateBlockInfo(block);

  for (mlir::Block *block : blockOrder) {
    BlockInfo &info = getOrCreateBlockInfo(block);
    info.mergeTarget = nullptr;
    info.continueTarget = nullptr;
    info.requestsMaskPush = false;
    info.requestsMaskPop = false;
    info.payloadSeed.clear();
    info.controlOps.clear();

    for (mlir::Operation &op : *block) {
      if (auto ifOp = llvm::dyn_cast<simt::dialect::IfOp>(&op)) {
        info.controlOps.push_back(&op);
        if (failed(analyseIfOp(info, &op)))
          return failure();
        continue;
      }
      if (auto loopOp = llvm::dyn_cast<simt::dialect::LoopOp>(&op)) {
        (void)loopOp;
        info.controlOps.push_back(&op);
        if (failed(analyseLoopOp(info, &op)))
          return failure();
        continue;
      }
      if (auto switchOp = llvm::dyn_cast<simt::dialect::SwitchOp>(&op)) {
        (void)switchOp;
        info.controlOps.push_back(&op);
        if (failed(analyseSwitchOp(info, &op)))
          return failure();
        continue;
      }
    }
  }

  return success();
}

LogicalResult StructuredCFGBuilder::computePayloads() {
  // Seed every block with its formal arguments so later stages know the
  // expected tuple shape even before payload propagation is finished.
  for (mlir::Block *block : blockOrder) {
    BlockInfo &info = getOrCreateBlockInfo(block);
    info.payloadSeed.clear();
    info.payloadSeed.append(info.blockArgs.begin(), info.blockArgs.end());
  }

  // Loop headers receive their initial payload from the `simt.loop` operands.
  for (const auto &entry : loopInfos) {
    mlir::Operation *op = entry.first;
    const LoopInfo &loopInfo = entry.second;
    auto loopOp = llvm::dyn_cast_or_null<simt::dialect::LoopOp>(op);
    if (!loopOp)
      continue;

    if (!loopInfo.prepareBlock)
      continue;

    BlockInfo &prepareInfo = *loopInfo.prepareBlock;
    mlir::ValueRange inits = loopOp.getInits();
    if (inits.size() != prepareInfo.blockArgs.size()) {
      op->emitOpError("loop init arity must match prepare block arguments");
      return failure();
    }
    prepareInfo.payloadSeed.assign(inits.begin(), inits.end());

    if (loopInfo.bodyBlock) {
      BlockInfo &bodyInfo = *loopInfo.bodyBlock;
      bodyInfo.payloadSeed.assign(bodyInfo.blockArgs.begin(),
                                  bodyInfo.blockArgs.end());

      if (prepareInfo.original) {
        if (auto *term = prepareInfo.original->getTerminator()) {
          if (auto cond = llvm::dyn_cast<simt::dialect::ConditionOp>(term)) {
            mlir::ValueRange forwarded = cond.getForwarded();
            if (forwarded.size() == bodyInfo.blockArgs.size())
              bodyInfo.payloadSeed.assign(forwarded.begin(), forwarded.end());
            loopInfo.forwardedToBody.assign(forwarded.begin(), forwarded.end());
            loopInfo.forwardedToExit.assign(forwarded.begin(), forwarded.end());
          }
        }
      }
    }
  }

  // Switch headers inherit their initial payload directly from the op.
  for (const auto &entry : switchInfos) {
    mlir::Operation *op = entry.first;
    const SwitchInfo &switchInfo = entry.second;
    auto switchOp = llvm::dyn_cast_or_null<simt::dialect::SwitchOp>(op);
    if (!switchOp)
      continue;

    if (!switchInfo.parent)
      continue;

    BlockInfo &parentInfo = *switchInfo.parent;
    parentInfo.payloadSeed.assign(parentInfo.blockArgs.begin(),
                                  parentInfo.blockArgs.end());
    mlir::ValueRange initialValues = switchOp.getInitialValues();
    parentInfo.payloadSeed.append(initialValues.begin(), initialValues.end());

    for (BlockInfo *caseInfo : switchInfo.caseBlocks) {
      if (!caseInfo)
        continue;
      caseInfo->payloadSeed.assign(caseInfo->blockArgs.begin(),
                                   caseInfo->blockArgs.end());
      if (caseInfo->original && !caseInfo->original->empty())
        if (auto yield = llvm::dyn_cast<simt::dialect::YieldOp>(
                caseInfo->original->getTerminator()))
          caseInfo->payloadSeed.assign(yield.getResults().begin(),
                                       yield.getResults().end());
    }
    if (switchInfo.defaultBlock) {
      switchInfo.defaultBlock->payloadSeed.assign(
          switchInfo.defaultBlock->blockArgs.begin(),
          switchInfo.defaultBlock->blockArgs.end());
      if (switchInfo.defaultBlock->original &&
          !switchInfo.defaultBlock->original->empty())
        if (auto yield = llvm::dyn_cast<simt::dialect::YieldOp>(
                switchInfo.defaultBlock->original->getTerminator()))
          switchInfo.defaultBlock->payloadSeed.assign(
              yield.getResults().begin(), yield.getResults().end());
    }
  }

  // TODO: propagate payloads through yields/terminators to reach a fixed point
  // once edge enumeration is implemented.
  return success();
}

LogicalResult StructuredCFGBuilder::enumerateEdges() {
  edges.clear();

  for (mlir::Block *block : blockOrder) {
    BlockInfo &info = getOrCreateBlockInfo(block);

    for (mlir::Operation *op : info.controlOps) {
      if (!op)
        continue;

      if (auto ifIt = ifInfos.find(op); ifIt != ifInfos.end()) {
        const IfInfo &ifInfo = ifIt->second;

        EdgeInfo thenEdge;
        thenEdge.source = &info;
        thenEdge.dest = ifInfo.thenBlock;
        thenEdge.kind = EdgeInfo::ConditionalTrue;
        if (thenEdge.dest && failed(ensurePayloadShape(thenEdge)))
          return failure();
        edges.push_back(std::move(thenEdge));

        EdgeInfo elseEdge;
        elseEdge.source = &info;
        elseEdge.dest = ifInfo.elseBlock;
        elseEdge.kind = EdgeInfo::ConditionalFalse;
        if (elseEdge.dest && failed(ensurePayloadShape(elseEdge)))
          return failure();
        edges.push_back(std::move(elseEdge));
        continue;
      }

      if (auto loopIt = loopInfos.find(op); loopIt != loopInfos.end()) {
        const LoopInfo &loopInfo = loopIt->second;

        EdgeInfo entryEdge;
        entryEdge.source = &info;
        entryEdge.dest = loopInfo.prepareBlock;
        entryEdge.kind = EdgeInfo::Plain;
        if (entryEdge.dest && failed(ensurePayloadShape(entryEdge)))
          return failure();
        edges.push_back(std::move(entryEdge));

        if (loopInfo.prepareBlock && loopInfo.prepareBlock->original) {
          if (auto *term = loopInfo.prepareBlock->original->getTerminator()) {
            if (auto cond = llvm::dyn_cast<simt::dialect::ConditionOp>(term)) {
              EdgeInfo trueEdge;
              trueEdge.source = loopInfo.prepareBlock;
              trueEdge.dest = loopInfo.bodyBlock;
              trueEdge.kind = EdgeInfo::ConditionalTrue;
              if (!loopInfo.forwardedToBody.empty())
                trueEdge.payload.assign(loopInfo.forwardedToBody.begin(),
                                         loopInfo.forwardedToBody.end());
              if (trueEdge.dest && failed(ensurePayloadShape(trueEdge)))
                return failure();
              edges.push_back(std::move(trueEdge));

              EdgeInfo falseEdge;
              falseEdge.source = loopInfo.prepareBlock;
              falseEdge.dest = loopInfo.parent;
              falseEdge.kind = EdgeInfo::ConditionalFalse;
              if (!loopInfo.forwardedToExit.empty())
                falseEdge.payload.assign(loopInfo.forwardedToExit.begin(),
                                          loopInfo.forwardedToExit.end());
              if (falseEdge.dest && failed(ensurePayloadShape(falseEdge)))
                return failure();
              edges.push_back(std::move(falseEdge));
            }
          }
        }

        if (loopInfo.bodyBlock && loopInfo.bodyBlock->original) {
          for (Operation &nested : *loopInfo.bodyBlock->original) {
            if (auto cont = llvm::dyn_cast<simt::dialect::ContinueOp>(&nested)) {
              (void)cont;
              EdgeInfo backEdge;
              backEdge.source = loopInfo.bodyBlock;
              backEdge.dest = loopInfo.prepareBlock;
              backEdge.kind = EdgeInfo::LoopBackEdge;
              backEdge.payload.assign(cont.getOperands().begin(),
                                      cont.getOperands().end());
              if (backEdge.dest && failed(ensurePayloadShape(backEdge)))
                return failure();
              edges.push_back(std::move(backEdge));
              continue;
            }
            if (llvm::isa<simt::dialect::BreakOp>(&nested) ||
                llvm::isa<simt::dialect::YieldOp>(&nested)) {
              EdgeInfo exitEdge;
              exitEdge.source = loopInfo.bodyBlock;
              exitEdge.dest = loopInfo.parent;
              exitEdge.kind = EdgeInfo::Plain;
              if (auto breakOp = llvm::dyn_cast<simt::dialect::BreakOp>(&nested))
                exitEdge.payload.assign(breakOp->getOperands().begin(),
                                        breakOp->getOperands().end());
              else if (auto yieldOp = llvm::dyn_cast<simt::dialect::YieldOp>(&nested))
                exitEdge.payload.assign(yieldOp->getOperands().begin(),
                                        yieldOp->getOperands().end());
              if (exitEdge.dest && failed(ensurePayloadShape(exitEdge)))
                return failure();
              edges.push_back(std::move(exitEdge));
              continue;
            }
          }
        }

        continue;
      }

      if (auto switchIt = switchInfos.find(op); switchIt != switchInfos.end()) {
        const SwitchInfo &switchInfo = switchIt->second;

        for (BlockInfo *caseInfo : switchInfo.caseBlocks) {
          EdgeInfo caseEdge;
          caseEdge.source = &info;
          caseEdge.dest = caseInfo;
          caseEdge.kind = EdgeInfo::Plain;
          if (caseEdge.dest && failed(ensurePayloadShape(caseEdge)))
            return failure();
          edges.push_back(std::move(caseEdge));

          if (caseInfo && caseInfo->original)
            if (auto term = caseInfo->original->getTerminator())
              if (auto yield = llvm::dyn_cast<simt::dialect::YieldOp>(term)) {
                EdgeInfo exitEdge;
                exitEdge.source = caseInfo;
                exitEdge.dest = switchInfo.parent;
                exitEdge.kind = EdgeInfo::Plain;
                exitEdge.payload.assign(yield.getResults().begin(),
                                        yield.getResults().end());
                if (exitEdge.dest && failed(ensurePayloadShape(exitEdge)))
                  return failure();
                edges.push_back(std::move(exitEdge));
              }
        }

        if (switchInfo.defaultBlock) {
          EdgeInfo defaultEdge;
          defaultEdge.source = &info;
          defaultEdge.dest = switchInfo.defaultBlock;
          defaultEdge.kind = EdgeInfo::Plain;
          if (failed(ensurePayloadShape(defaultEdge)))
            return failure();
          edges.push_back(std::move(defaultEdge));

          if (switchInfo.defaultBlock->original)
            if (auto term = switchInfo.defaultBlock->original->getTerminator())
              if (auto yield = llvm::dyn_cast<simt::dialect::YieldOp>(term)) {
                EdgeInfo exitEdge;
                exitEdge.source = switchInfo.defaultBlock;
                exitEdge.dest = switchInfo.parent;
                exitEdge.kind = EdgeInfo::Plain;
                exitEdge.payload.assign(yield.getResults().begin(),
                                         yield.getResults().end());
                if (exitEdge.dest && failed(ensurePayloadShape(exitEdge)))
                  return failure();
                edges.push_back(std::move(exitEdge));
              }
        }

        continue;
      }
    }
  }

  return success();
}

LogicalResult StructuredCFGBuilder::emitStructuredBlocks() {
  // TODO: create `simt_struct.block` ops mirroring `blockOrder`, clone bodies,
  // and materialise mask stack management before forwarding payloads.
  return signalUnimplemented(func);
}

LogicalResult StructuredCFGBuilder::cleanupOriginalCFG() {
  // TODO: erase the legacy CFG blocks once the structured region is fully
  // populated.
  return signalUnimplemented(func);
}

LogicalResult StructuredCFGBuilder::emitStructuredBlock(BlockInfo &info) {
  (void)info;
  // TODO: populate a single structured block, including carried arguments and
  // mask entry prologue.
  return signalUnimplemented(func);
}

LogicalResult StructuredCFGBuilder::emitStructuredTerminator(
    BlockInfo &source, const EdgeInfo &edge) {
  (void)source;
  (void)edge;
  // TODO: create `branch`/`cond_branch` terminators with the payload tuple.
  return signalUnimplemented(func);
}

LogicalResult StructuredCFGBuilder::analyseIfOp(BlockInfo &header,
                                                 mlir::Operation *op) {
  auto ifOp = llvm::dyn_cast<simt::dialect::IfOp>(op);
  if (!ifOp) {
    op->emitOpError("expected simt_step.if operation during analysis");
    return failure();
  }

  IfInfo &info = ifInfos[op];
  info.op = op;
  info.parent = &header;

  header.requestsMaskPush = true;
  header.requestsMaskPop = true;

  mlir::Region &thenRegion = ifOp.getThenRegion();
  if (thenRegion.empty()) {
    op->emitOpError("then region must contain a block");
    return failure();
  }

  BlockInfo &thenInfo = getOrCreateBlockInfo(&thenRegion.front());
  info.thenBlock = &thenInfo;

  mlir::Region &elseRegion = ifOp.getElseRegion();
  if (!elseRegion.empty()) {
    BlockInfo &elseInfo = getOrCreateBlockInfo(&elseRegion.front());
    info.elseBlock = &elseInfo;
  }

  return success();
}

LogicalResult StructuredCFGBuilder::analyseLoopOp(BlockInfo &header,
                                                   mlir::Operation *op) {
  auto loopOp = llvm::dyn_cast<simt::dialect::LoopOp>(op);
  if (!loopOp) {
    op->emitOpError("expected simt_step.loop operation during analysis");
    return failure();
  }

  LoopInfo &info = loopInfos[op];
  info.op = op;
  info.parent = &header;

  mlir::Region &prepareRegion = loopOp.getPrepareRegion();
  mlir::Region &bodyRegion = loopOp.getBodyRegion();
  if (prepareRegion.empty() || bodyRegion.empty()) {
    op->emitOpError("loop regions must contain a single block");
    return failure();
  }

  BlockInfo &prepareInfo = getOrCreateBlockInfo(&prepareRegion.front());
  BlockInfo &bodyInfo = getOrCreateBlockInfo(&bodyRegion.front());
  info.prepareBlock = &prepareInfo;
  info.bodyBlock = &bodyInfo;

  prepareInfo.requestsMaskPush = true;
  prepareInfo.requestsMaskPop = true;
  prepareInfo.continueTarget = prepareInfo.original;

  bodyInfo.requestsMaskPush = true;
  bodyInfo.requestsMaskPop = true;
  bodyInfo.continueTarget = prepareInfo.original;

  return success();
}

LogicalResult StructuredCFGBuilder::analyseSwitchOp(BlockInfo &header,
                                                     mlir::Operation *op) {
  auto switchOp = llvm::dyn_cast<simt::dialect::SwitchOp>(op);
  if (!switchOp) {
    op->emitOpError("expected simt_step.switch operation during analysis");
    return failure();
  }

  SwitchInfo &info = switchInfos[op];
  info.op = op;
  info.parent = &header;
  info.caseBlocks.clear();
  info.caseValues.clear();
  info.defaultBlock = nullptr;

  header.requestsMaskPush = true;

  auto caseValuesAttr = switchOp.getCaseValuesAttr();
  if (!caseValuesAttr) {
    op->emitOpError("missing case_values attribute");
    return failure();
  }
  auto caseValues = caseValuesAttr.asArrayRef();

  mlir::Region &caseRegion = switchOp.getCaseBody();
  if (caseRegion.empty()) {
    op->emitOpError("switch body region must not be empty");
    return failure();
  }

  if (caseRegion.getBlocks().size() != caseValues.size()) {
    op->emitOpError("case_values count must match number of case blocks");
    return failure();
  }

  info.caseValues.append(caseValues.begin(), caseValues.end());

  unsigned blockIndex = 0;
  for (mlir::Block &caseBlock : caseRegion) {
    BlockInfo &caseInfo = getOrCreateBlockInfo(&caseBlock);
    caseInfo.requestsMaskPop = true;

    if (blockIndex + 1 == caseValues.size())
      info.defaultBlock = &caseInfo;
    else
      info.caseBlocks.push_back(&caseInfo);

    ++blockIndex;
  }

  return success();
}

LogicalResult StructuredCFGBuilder::ensurePayloadShape(EdgeInfo &edge) {
  if (!edge.dest)
    return success();

  if (!edge.payload.empty()) {
    // Trust the caller-provided payload, but sanity-check width if we already
    // know the destination seed tuple.
    if (!edge.dest->payloadSeed.empty() &&
        edge.dest->payloadSeed.size() != edge.payload.size()) {
      func.emitError("edge payload arity does not match destination seed");
      return failure();
    }
    return success();
  }

  SmallVector<mlir::Value, 8> expected;
  if (!edge.dest->payloadSeed.empty())
    expected.append(edge.dest->payloadSeed.begin(), edge.dest->payloadSeed.end());
  else
    expected.append(edge.dest->blockArgs.begin(), edge.dest->blockArgs.end());

  edge.payload.assign(expected.begin(), expected.end());
  return success();
}

LogicalResult StructuredCFGBuilder::propagatePayload(
    BlockInfo &source, BlockInfo &dest, llvm::ArrayRef<mlir::Value> values) {
  (void)source;
  if (values.size() != dest.blockArgs.size()) {
    func.emitError("payload arity mismatch while propagating values");
    return failure();
  }

  dest.payloadSeed.assign(values.begin(), values.end());
  return success();
}

LogicalResult StructuredCFGBuilder::materialiseMaskEntry(BlockInfo &info) {
  (void)info;
  // TODO: insert `mask_push`/`mask_merge` ops for the structured block.
  return signalUnimplemented(func);
}

LogicalResult StructuredCFGBuilder::materialiseMaskExit(BlockInfo &info) {
  (void)info;
  // TODO: add `mask_pop` when a block represents a merge point.
  return signalUnimplemented(func);
}

} // namespace simt::conversion
