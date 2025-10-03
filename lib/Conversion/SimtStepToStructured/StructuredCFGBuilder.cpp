#include "simt-step/Conversion/StructuredCFGBuilder.h"

#include "simt-step/Dialect/SimtStructured/StructuredDialect.h"
#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"

#include <llvm/Support/Casting.h>

#include <llvm/ADT/STLExtras.h>
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
  mlir::BlockArgument structuredMaskArg;
  mlir::Value currentMask;
  SmallVector<mlir::BlockArgument, 4> structuredArgs;

  mlir::Block *mergeTarget = nullptr;
  mlir::Block *continueTarget = nullptr;

  bool requestsMaskPush = false;
  bool requestsMaskPop = false;
  std::string symbolName;
  SmallVector<const EdgeInfo *, 4> outgoingEdges;
  Operation *originalTerminator = nullptr;
};

struct StructuredCFGBuilder::EdgeInfo {
  BlockInfo *source = nullptr;
  BlockInfo *dest = nullptr;
  SmallVector<mlir::Value, 8> payload;
  SmallVector<mlir::Value, 4> maskValues;
  enum Kind { Plain, ConditionalTrue, ConditionalFalse, LoopBackEdge } kind =
      Plain;
  mlir::Value condition;
  mlir::Value switchDoneFlag;
  bool isSwitchFallthrough = false;
};

struct StructuredCFGBuilder::IfInfo {
  mlir::Operation *op = nullptr;
  BlockInfo *parent = nullptr;
  BlockInfo *thenBlock = nullptr;
  BlockInfo *elseBlock = nullptr;
  mlir::Value condition;
};

struct StructuredCFGBuilder::LoopInfo {
  mlir::Operation *op = nullptr;
  BlockInfo *parent = nullptr;
  BlockInfo *prepareBlock = nullptr;
  BlockInfo *bodyBlock = nullptr;
  mlir::Value condition;
  SmallVector<mlir::Value, 4> forwardedToBody;
  SmallVector<mlir::Value, 4> forwardedToExit;
};

struct StructuredCFGBuilder::SwitchInfo {
  mlir::Operation *op = nullptr;
  BlockInfo *parent = nullptr;
  SmallVector<BlockInfo *, 4> caseBlocks;
  BlockInfo *defaultBlock = nullptr;
  SmallVector<int64_t, 8> caseValues;
  unsigned payloadCount = 0;
  bool hasControlFlags = false;
  struct CaseRecord {
    BlockInfo *block = nullptr;
    BlockInfo *nextCase = nullptr;
    mlir::Value matchSeen;
    mlir::Value fallthrough;
    mlir::Value switchDone;
  };
  SmallVector<CaseRecord, 4> caseRecords;
  CaseRecord defaultRecord;
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

  for (auto [index, block] : llvm::enumerate(blockOrder)) {
    BlockInfo &info = getOrCreateBlockInfo(block);
    info.mergeTarget = nullptr;
    info.continueTarget = nullptr;
    info.requestsMaskPush = false;
    info.requestsMaskPop = false;
    info.payloadSeed.clear();
    info.controlOps.clear();
    info.outgoingEdges.clear();
    info.structuredOp = nullptr;
    info.structuredBody = nullptr;
    info.structuredMaskArg = nullptr;
    info.currentMask = nullptr;
    info.structuredArgs.clear();
    info.originalTerminator = nullptr;
    if (index == 0)
      info.symbolName = "entry";
    else
      info.symbolName = ("block" + std::to_string(index));

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

    switchInfo.caseRecords.clear();
    switchInfo.caseRecords.reserve(switchInfo.caseBlocks.size());
    for (auto [index, caseInfo] : llvm::enumerate(switchInfo.caseBlocks)) {
      SwitchInfo::CaseRecord record;
      record.block = caseInfo;
      record.nextCase = nullptr;
      if (index + 1 < switchInfo.caseBlocks.size())
        record.nextCase = switchInfo.caseBlocks[index + 1];
      else
        record.nextCase = switchInfo.defaultBlock;

      if (switchInfo.hasControlFlags && caseInfo && caseInfo->original &&
          !caseInfo->original->empty()) {
        if (auto yield = llvm::dyn_cast<simt::dialect::YieldOp>(
                caseInfo->original->getTerminator())) {
          auto results = yield.getResults();
          if (results.size() >= switchInfo.payloadCount + 3) {
            record.matchSeen = results[switchInfo.payloadCount + 0];
            record.fallthrough = results[switchInfo.payloadCount + 1];
            record.switchDone = results[switchInfo.payloadCount + 2];
          }
        }
      }
      switchInfo.caseRecords.push_back(record);
    }

    switchInfo.defaultRecord = SwitchInfo::CaseRecord();
    switchInfo.defaultRecord.block = switchInfo.defaultBlock;
    if (switchInfo.hasControlFlags && switchInfo.defaultBlock &&
        switchInfo.defaultBlock->original &&
        !switchInfo.defaultBlock->original->empty()) {
      if (auto yield = llvm::dyn_cast<simt::dialect::YieldOp>(
              switchInfo.defaultBlock->original->getTerminator())) {
        auto results = yield.getResults();
        if (results.size() >= switchInfo.payloadCount + 3) {
          switchInfo.defaultRecord.matchSeen =
              results[switchInfo.payloadCount + 0];
          switchInfo.defaultRecord.fallthrough =
              results[switchInfo.payloadCount + 1];
          switchInfo.defaultRecord.switchDone =
              results[switchInfo.payloadCount + 2];
        }
      }
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
        thenEdge.condition = ifInfo.condition;
        if (thenEdge.dest && failed(ensurePayloadShape(thenEdge)))
          return failure();
        edges.push_back(std::move(thenEdge));
        info.outgoingEdges.push_back(&edges.back());

        EdgeInfo elseEdge;
        elseEdge.source = &info;
        elseEdge.dest = ifInfo.elseBlock;
        elseEdge.kind = EdgeInfo::ConditionalFalse;
        elseEdge.condition = ifInfo.condition;
        if (elseEdge.dest && failed(ensurePayloadShape(elseEdge)))
          return failure();
        edges.push_back(std::move(elseEdge));
        info.outgoingEdges.push_back(&edges.back());
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
        info.outgoingEdges.push_back(&edges.back());

        if (loopInfo.prepareBlock && loopInfo.prepareBlock->original) {
          if (auto *term = loopInfo.prepareBlock->original->getTerminator()) {
            if (auto cond = llvm::dyn_cast<simt::dialect::ConditionOp>(term)) {
              loopInfo.condition = cond.getCondition();
              EdgeInfo trueEdge;
              trueEdge.source = loopInfo.prepareBlock;
              trueEdge.dest = loopInfo.bodyBlock;
              trueEdge.kind = EdgeInfo::ConditionalTrue;
              if (!loopInfo.forwardedToBody.empty())
                trueEdge.payload.assign(loopInfo.forwardedToBody.begin(),
                                         loopInfo.forwardedToBody.end());
              trueEdge.condition = cond.getCondition();
              if (trueEdge.dest && failed(ensurePayloadShape(trueEdge)))
                return failure();
              edges.push_back(std::move(trueEdge));
              loopInfo.prepareBlock->outgoingEdges.push_back(&edges.back());

              EdgeInfo falseEdge;
              falseEdge.source = loopInfo.prepareBlock;
              falseEdge.dest = loopInfo.parent;
              falseEdge.kind = EdgeInfo::ConditionalFalse;
              if (!loopInfo.forwardedToExit.empty())
                falseEdge.payload.assign(loopInfo.forwardedToExit.begin(),
                                          loopInfo.forwardedToExit.end());
              falseEdge.condition = cond.getCondition();
              if (falseEdge.dest && failed(ensurePayloadShape(falseEdge)))
                return failure();
              edges.push_back(std::move(falseEdge));
              loopInfo.prepareBlock->outgoingEdges.push_back(&edges.back());
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
              loopInfo.bodyBlock->outgoingEdges.push_back(&edges.back());
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
              loopInfo.bodyBlock->outgoingEdges.push_back(&edges.back());
              continue;
            }
          }
        }

        continue;
      }

      if (auto switchIt = switchInfos.find(op); switchIt != switchInfos.end()) {
        const SwitchInfo &switchInfo = switchIt->second;

        for (const auto &record : switchInfo.caseRecords) {
          BlockInfo *caseInfo = record.block;
          EdgeInfo caseEntry;
          caseEntry.source = &info;
          caseEntry.dest = caseInfo;
          caseEntry.kind = EdgeInfo::Plain;
          if (caseEntry.dest && failed(ensurePayloadShape(caseEntry)))
            return failure();
          edges.push_back(std::move(caseEntry));
          info.outgoingEdges.push_back(&edges.back());

          auto emitExitEdge = [&](EdgeInfo::Kind kind, BlockInfo *dest)
                                  -> LogicalResult {
            EdgeInfo exitEdge;
            exitEdge.source = caseInfo;
            exitEdge.dest = dest;
            exitEdge.kind = kind;
            if (caseInfo)
              exitEdge.payload.assign(caseInfo->payloadSeed.begin(),
                                      caseInfo->payloadSeed.end());
            if (kind != EdgeInfo::Plain) {
              exitEdge.isSwitchFallthrough = true;
              exitEdge.condition = record.fallthrough;
              exitEdge.switchDoneFlag = record.switchDone;
            }
            if (exitEdge.dest && failed(ensurePayloadShape(exitEdge)))
              return failure();
            edges.push_back(std::move(exitEdge));
            if (caseInfo)
              caseInfo->outgoingEdges.push_back(&edges.back());
            return success();
          };

          bool canFallthrough = record.nextCase != nullptr &&
                                switchInfo.hasControlFlags && record.fallthrough &&
                                record.switchDone;

          if (canFallthrough) {
            EdgeInfo fallEdge;
            fallEdge.source = caseInfo;
            fallEdge.dest = record.nextCase;
            fallEdge.kind = EdgeInfo::ConditionalTrue;
            fallEdge.condition = record.fallthrough;
            fallEdge.switchDoneFlag = record.switchDone;
            fallEdge.isSwitchFallthrough = true;
            if (caseInfo)
              fallEdge.payload.assign(caseInfo->payloadSeed.begin(),
                                      caseInfo->payloadSeed.end());
            if (fallEdge.dest && failed(ensurePayloadShape(fallEdge)))
              return failure();
            edges.push_back(std::move(fallEdge));
            if (caseInfo)
              caseInfo->outgoingEdges.push_back(&edges.back());

            if (failed(emitExitEdge(EdgeInfo::ConditionalFalse, switchInfo.parent)))
              return failure();
          } else {
            if (failed(emitExitEdge(EdgeInfo::Plain, switchInfo.parent)))
              return failure();
          }
        }

        if (switchInfo.defaultBlock) {
          EdgeInfo defaultEntry;
          defaultEntry.source = &info;
          defaultEntry.dest = switchInfo.defaultBlock;
          defaultEntry.kind = EdgeInfo::Plain;
          if (failed(ensurePayloadShape(defaultEntry)))
            return failure();
          edges.push_back(std::move(defaultEntry));
          info.outgoingEdges.push_back(&edges.back());

          if (switchInfo.defaultBlock) {
            EdgeInfo exitEdge;
            exitEdge.source = switchInfo.defaultBlock;
            exitEdge.dest = switchInfo.parent;
            exitEdge.kind = EdgeInfo::Plain;
            if (switchInfo.defaultBlock)
              exitEdge.payload.assign(switchInfo.defaultBlock->payloadSeed.begin(),
                                       switchInfo.defaultBlock->payloadSeed.end());
            if (exitEdge.dest && failed(ensurePayloadShape(exitEdge)))
              return failure();
            edges.push_back(std::move(exitEdge));
            switchInfo.defaultBlock->outgoingEdges.push_back(&edges.back());
          }
        }

        continue;
      }
    }
  }

  return success();
}

LogicalResult StructuredCFGBuilder::emitStructuredBlocks() {
  if (blockOrder.empty())
    return success();

  OpBuilder builder(func);
  mapper->clear();

  SmallVector<BlockInfo *, 16> orderedInfos;
  orderedInfos.reserve(blockOrder.size());
  for (mlir::Block *origBlock : blockOrder)
    orderedInfos.push_back(&blockInfos[origBlock]);

  Block *entryBlock = blockOrder.front();
  builder.setInsertionPointToStart(entryBlock);

  // First pass: create structured block ops and map block arguments.
  for (BlockInfo *info : orderedInfos) {
    auto symAttr = builder.getStringAttr(info->symbolName);
    auto blockOp = builder.create<simt::structured::BlockOp>(symAttr,
                                                             mlir::Value());
    info->structuredOp = blockOp;
    info->structuredBody = &blockOp.getBody().front();
    info->structuredMaskArg = blockOp.getMaskArgument();
    info->currentMask = info->structuredMaskArg;
    info->structuredArgs.clear();
    info->originalTerminator = info->original ? info->original->getTerminator()
                                              : nullptr;

    if (info->original) {
      for (mlir::BlockArgument origArg : info->original->getArguments()) {
        auto newArg = info->structuredBody->addArgument(origArg.getType(),
                                                        origArg.getLoc());
        info->structuredArgs.push_back(newArg);
        mapper->map(origArg, newArg);
      }
    }

    builder.setInsertionPointAfter(blockOp);
  }

  // Second pass: set attributes and clone bodies (excluding terminators).
  for (BlockInfo *info : orderedInfos) {
    auto blockOp = info->structuredOp;
    if (!blockOp)
      continue;

    auto setSymbolAttr = [&](StringRef attrName, BlockInfo *target) {
      if (!target || !target->structuredOp)
        return;
      auto ref = builder.getSymbolRefAttr(target->structuredOp.getSymName());
      blockOp->setAttr(attrName, ref);
    };

    if (info->mergeTarget)
      setSymbolAttr(blockOp.getMergeTargetAttrName(),
                    &blockInfos[info->mergeTarget]);
    else
      blockOp->removeAttr(blockOp.getMergeTargetAttrName());

    if (info->continueTarget)
      setSymbolAttr(blockOp.getContinueTargetAttrName(),
                    &blockInfos[info->continueTarget]);
    else
      blockOp->removeAttr(blockOp.getContinueTargetAttrName());

    if (failed(materialiseMaskEntry(*info)))
      return failure();

    OpBuilder bodyBuilder(info->structuredBody,
                          info->structuredBody->begin());
    if (info->original) {
      for (Operation &op : info->original->without_terminator()) {
        if (isa<cf::BranchOp, cf::CondBranchOp, func::ReturnOp>(&op))
          continue;
        if (auto active = dyn_cast<simt::dialect::ActiveMaskOp>(&op)) {
          mapper->map(active.getResult(), info->currentMask);
          continue;
        }
        Operation *cloned = bodyBuilder.clone(op, *mapper);
        mapper->map(op.getResults(), cloned->getResults());
      }
    }

    if (failed(emitStructuredTerminator(*info)))
      return failure();
  }

  return success();
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

LogicalResult StructuredCFGBuilder::emitStructuredTerminator(BlockInfo &source) {
  if (!source.structuredBody)
    return success();

  Operation *origTerm = source.originalTerminator;
  OpBuilder builder(source.structuredBody, source.structuredBody->end());
  Location loc = origTerm ? origTerm->getLoc() : func.getLoc();

  auto mapValue = [&](Value v, StringRef context) -> std::optional<Value> {
    if (!v)
      return Value();
    if (Value mapped = mapper->lookupOrNull(v))
      return mapped;
    if (origTerm)
      origTerm->emitError() << "unable to map value in " << context;
    return std::nullopt;
  };

  auto mapValues = [&](ArrayRef<Value> vals, SmallVectorImpl<Value> &out,
                       StringRef context) -> LogicalResult {
    for (Value v : vals) {
      auto mapped = mapValue(v, context);
      if (!mapped)
        return failure();
      out.push_back(*mapped);
    }
    return success();
  };

  if (auto ret = dyn_cast_or_null<func::ReturnOp>(origTerm)) {
    SmallVector<Value> results;
    if (failed(mapValues(ret.getOperands(), results, "return")))
      return failure();
    builder.create<simt::structured::ReturnOp>(loc, results);
    return success();
  }

  if (source.outgoingEdges.empty()) {
    if (origTerm)
      origTerm->emitError("structured terminator requires recorded edges");
    return failure();
  }

  const EdgeInfo *trueEdge = nullptr;
  const EdgeInfo *falseEdge = nullptr;
  SmallVector<const EdgeInfo *, 4> plainEdges;

  for (const EdgeInfo *edge : source.outgoingEdges) {
    switch (edge->kind) {
    case EdgeInfo::ConditionalTrue:
      trueEdge = edge;
      break;
    case EdgeInfo::ConditionalFalse:
      falseEdge = edge;
      break;
    case EdgeInfo::LoopBackEdge:
    case EdgeInfo::Plain:
      plainEdges.push_back(edge);
      break;
    }
  }

  auto buildBranch = [&](const EdgeInfo *edge) -> LogicalResult {
    if (!edge || !edge->dest || !edge->dest->structuredOp) {
      if (origTerm)
        origTerm->emitError("branch edge missing destination");
      return failure();
    }
    SmallVector<Value> operands;
    if (failed(mapValues(edge->payload, operands, "branch payload")))
      return failure();
    Value mask = source.currentMask ? source.currentMask : source.structuredMaskArg;
    auto targetAttr = builder.getSymbolRefAttr(
        edge->dest->structuredOp.getSymName());
    builder.create<simt::structured::BranchOp>(loc, mask, targetAttr,
                                               operands);
    return success();
  };

  if (trueEdge && falseEdge) {
    auto cond = mapValue(trueEdge->condition, "cond branch condition");
    if (!cond) {
      if (origTerm)
        origTerm->emitError("conditional edge missing condition value");
      return failure();
    }

    SmallVector<Value> truePayload;
    SmallVector<Value> falsePayload;
    if (failed(mapValues(trueEdge->payload, truePayload,
                         "cond true payload")) ||
        failed(mapValues(falseEdge->payload, falsePayload,
                         "cond false payload")))
      return failure();

    auto trueTarget = trueEdge->dest && trueEdge->dest->structuredOp
                          ? builder.getSymbolRefAttr(
                                trueEdge->dest->structuredOp.getSymName())
                          : FlatSymbolRefAttr();
    auto falseTarget = falseEdge->dest && falseEdge->dest->structuredOp
                           ? builder.getSymbolRefAttr(
                                 falseEdge->dest->structuredOp.getSymName())
                           : FlatSymbolRefAttr();

    Value trueMask = source.currentMask ? source.currentMask
                                        : source.structuredMaskArg;
    Value falseMask = trueMask;

    builder.create<simt::structured::CondBranchOp>(
        loc, *cond, trueMask, falseMask, trueTarget, falseTarget,
        truePayload, falsePayload, FlatSymbolRefAttr(),
        simt::structured::ReconvergencePolicyAttr());
    return success();
  }

  if (plainEdges.size() == 1)
    return buildBranch(plainEdges.front());

  if (origTerm)
    origTerm->emitError("unsupported structured terminator pattern");
  return failure();
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

  info.condition = ifOp.getCondition();

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

  info.condition = nullptr;

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
  info.caseRecords.clear();
  info.defaultRecord = SwitchInfo::CaseRecord();

  unsigned numResults = switchOp.getNumResults();
  info.hasControlFlags = numResults >= 3;
  info.payloadCount = info.hasControlFlags ? numResults - 3 : numResults;

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
  if (!info.structuredBody)
    return success();

  OpBuilder builder(info.structuredBody, info.structuredBody->begin());
  Location loc = func.getLoc();
  if (info.original && !info.original->empty())
    loc = info.original->front().getLoc();

  Value mask = info.currentMask ? info.currentMask : info.structuredMaskArg;
  Operation *lastOp = nullptr;

  if (info.requestsMaskPop) {
    auto pop = builder.create<simt::structured::MaskPopOp>(loc,
                                                           mask.getType());
    lastOp = pop.getOperation();
    auto merge = builder.create<simt::structured::MaskMergeOp>(
        loc, mask.getType(), info.structuredMaskArg);
    mask = merge.getMerged();
    lastOp = merge.getOperation();
  }

  auto getTargetAttr = [&](mlir::Block *target) -> FlatSymbolRefAttr {
    if (!target)
      return FlatSymbolRefAttr();
    if (BlockInfo *targetInfo = lookupBlockInfo(target))
      if (targetInfo->structuredOp)
        return builder.getSymbolRefAttr(targetInfo->structuredOp.getSymName());
    return FlatSymbolRefAttr();
  };

  if (info.requestsMaskPush) {
    if (lastOp)
      builder.setInsertionPointAfter(lastOp);
    else
      builder.setInsertionPointToStart(info.structuredBody);
    auto mergeAttr = getTargetAttr(info.mergeTarget);
    auto continueAttr = getTargetAttr(info.continueTarget);
    builder.create<simt::structured::MaskPushOp>(loc, mask, mergeAttr,
                                                 continueAttr, IntegerAttr());
  }

  info.currentMask = mask;
  return success();
}

LogicalResult StructuredCFGBuilder::materialiseMaskExit(BlockInfo &info) {
  (void)info;
  return success();
}

} // namespace simt::conversion
