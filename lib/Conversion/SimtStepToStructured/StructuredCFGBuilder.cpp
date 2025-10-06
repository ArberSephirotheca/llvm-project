
#include "simt-step/Dialect/SimtStructured/StructuredDialect.h"
#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"
#include "simt-step/Conversion/StructuredCFGBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
#include <llvm/Support/ErrorHandling.h>

#include <cassert>

#include <limits>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/DenseSet.h>

using namespace mlir;

namespace simt::conversion {

namespace {

constexpr llvm::StringLiteral kUnimplementedMsg(
    "StructuredCFGBuilder skeleton reached. Implement the new builder.");

[[maybe_unused]] static LogicalResult
signalUnimplemented(FunctionOpInterface func) {
  func.emitError(kUnimplementedMsg);
  return failure();
}

} // namespace

StructuredCFGBuilder::StructuredCFGBuilder(FunctionOpInterface func)
    : func(func) {}

LogicalResult StructuredCFGBuilder::build() {
  mapper = std::make_unique<IRMapping>();
  domInfo = std::make_unique<DominanceInfo>(func);
  functionReturnValues.clear();
  hasFunctionReturn = false;
  structuredOpsInOrder.clear();

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
  if (failed(stabilisePayloadSeeds()))
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
  for (Region &region : func.getOperation()->getRegions())
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
    info.structuredOp = simt::structured::BlockOp();
    info.structuredBody = nullptr;
    info.structuredMaskArg = nullptr;
    info.currentMask = nullptr;
    info.structuredArgs.clear();
    info.payloadArgs.clear();
    info.payloadBlockArgOffset = 0;
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
    info.payloadBlockArgOffset = info.payloadSeed.size() >= info.blockArgs.size()
                                     ? info.payloadSeed.size() -
                                           info.blockArgs.size()
                                     : 0;
  }

  // Loop headers receive their initial payload from the `simt.loop` operands.
  for (auto &entry : loopInfos) {
    mlir::Operation *op = entry.first;
    LoopInfo &loopInfo = entry.second;
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
    prepareInfo.payloadBlockArgOffset = prepareInfo.payloadSeed.size() >=
                                                prepareInfo.blockArgs.size()
                                            ? prepareInfo.payloadSeed.size() -
                                                  prepareInfo.blockArgs.size()
                                            : 0;

    if (loopInfo.bodyBlock) {
      BlockInfo &bodyInfo = *loopInfo.bodyBlock;
      bodyInfo.payloadSeed.assign(bodyInfo.blockArgs.begin(),
                                  bodyInfo.blockArgs.end());
      bodyInfo.payloadBlockArgOffset = bodyInfo.payloadSeed.size() >=
                                               bodyInfo.blockArgs.size()
                                           ? bodyInfo.payloadSeed.size() -
                                                 bodyInfo.blockArgs.size()
                                           : 0;

      if (prepareInfo.original) {
        if (auto *term = prepareInfo.original->getTerminator()) {
          if (auto cond = llvm::dyn_cast<simt::dialect::ConditionOp>(term)) {
            mlir::ValueRange forwarded = cond.getForwarded();
            if (forwarded.size() == bodyInfo.blockArgs.size())
              bodyInfo.payloadSeed.assign(forwarded.begin(), forwarded.end());
            bodyInfo.payloadBlockArgOffset = bodyInfo.payloadSeed.size() >=
                                                     bodyInfo.blockArgs.size()
                                                 ? bodyInfo.payloadSeed.size() -
                                                       bodyInfo.blockArgs.size()
                                                 : 0;
            loopInfo.forwardedToBody.clear();
            loopInfo.forwardedToBody.append(forwarded.begin(), forwarded.end());
            loopInfo.forwardedToExit.clear();
            loopInfo.forwardedToExit.append(forwarded.begin(), forwarded.end());
          }
        }
      }
    }

    if (!loopInfo.forwardedToBody.empty() && loopInfo.bodyBlock)
      if (failed(propagatePayload(*loopInfo.prepareBlock, *loopInfo.bodyBlock,
                                  loopInfo.forwardedToBody)))
        return failure();

    if (!loopInfo.forwardedToExit.empty() && loopInfo.mergeBlock)
      if (failed(propagatePayload(*loopInfo.prepareBlock, *loopInfo.mergeBlock,
                                  loopInfo.forwardedToExit)))
        return failure();
  }

  // Capture if-yield payloads so nested selections propagate tuples to their
  // merge blocks.
  for (auto &entry : ifInfos) {
    IfInfo &ifInfo = entry.second;
    auto ifOp = llvm::dyn_cast_or_null<simt::dialect::IfOp>(ifInfo.op);
    if (!ifOp)
      continue;
    ifInfo.thenYieldValues.clear();
    ifInfo.elseYieldValues.clear();
    ifInfo.thenYieldOp = nullptr;
    ifInfo.elseYieldOp = nullptr;
    ifInfo.elseImplicitYield = false;

    auto recordYield = [&](Region &region,
                           llvm::SmallVector<mlir::Value, 4> &out,
                           mlir::Operation *&yieldOp) {
      yieldOp = nullptr;
      out.clear();
      for (Block &block : region) {
        Operation *terminator = block.getTerminator();
        if (auto yield = llvm::dyn_cast<simt::dialect::YieldOp>(terminator)) {
          out.assign(yield.getOperands().begin(), yield.getOperands().end());
          yieldOp = yield;
          return;
        }
      }
    };

    recordYield(ifOp.getThenRegion(), ifInfo.thenYieldValues,
                ifInfo.thenYieldOp);
    if (!ifOp.getElseRegion().empty())
      recordYield(ifOp.getElseRegion(), ifInfo.elseYieldValues,
                  ifInfo.elseYieldOp);
    else if (!ifInfo.mergeArgs.empty())
      ifInfo.elseImplicitYield = true;

    SmallVector<mlir::Value, 8> parentPayload;
    if (ifInfo.parent) {
      parentPayload.assign(ifInfo.parent->payloadSeed.begin(),
                           ifInfo.parent->payloadSeed.end());
      if (parentPayload.empty())
        parentPayload.assign(ifInfo.parent->blockArgs.begin(),
                             ifInfo.parent->blockArgs.end());
    }

    if (ifInfo.thenBlock) {
      ifInfo.thenBlock->payloadSeed.assign(parentPayload.begin(),
                                           parentPayload.end());
      ifInfo.thenBlock->payloadBlockArgOffset = parentPayload.size();
    }
    if (ifInfo.elseBlock) {
      ifInfo.elseBlock->payloadSeed.assign(parentPayload.begin(),
                                           parentPayload.end());
      ifInfo.elseBlock->payloadBlockArgOffset = parentPayload.size();
    }

    if (BlockInfo *mergeBlock = ifInfo.mergeBlock) {
      SmallVector<mlir::Value, 8> combined;
      combined.append(parentPayload.begin(), parentPayload.end());
      combined.append(mergeBlock->blockArgs.begin(),
                      mergeBlock->blockArgs.end());
      mergeBlock->payloadSeed.assign(combined.begin(), combined.end());
      mergeBlock->payloadBlockArgOffset = parentPayload.size();
    }
  }

  // Switch headers inherit their initial payload directly from the op.
  for (auto &entry : switchInfos) {
    mlir::Operation *op = entry.first;
    SwitchInfo &switchInfo = entry.second;
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
    parentInfo.payloadBlockArgOffset = parentInfo.payloadSeed.size() >=
                                               parentInfo.blockArgs.size()
                                           ? parentInfo.payloadSeed.size() -
                                                 parentInfo.blockArgs.size()
                                           : 0;

    switchInfo.carriedCount = 0;
    for (BlockInfo *caseInfo : switchInfo.caseBlocks) {
      if (!caseInfo)
        continue;
      caseInfo->payloadSeed.assign(caseInfo->blockArgs.begin(),
                                   caseInfo->blockArgs.end());
      caseInfo->payloadBlockArgOffset = caseInfo->payloadSeed.size() >=
                                                caseInfo->blockArgs.size()
                                            ? caseInfo->payloadSeed.size() -
                                                  caseInfo->blockArgs.size()
                                            : 0;
      if (caseInfo->original && !caseInfo->original->empty())
        if (auto yield = llvm::dyn_cast<simt::dialect::YieldOp>(
                caseInfo->original->getTerminator()))
          caseInfo->payloadSeed.assign(yield.getResults().begin(),
                                       yield.getResults().end());
      caseInfo->payloadBlockArgOffset = caseInfo->payloadSeed.size() >=
                                                caseInfo->blockArgs.size()
                                            ? caseInfo->payloadSeed.size() -
                                                  caseInfo->blockArgs.size()
                                            : 0;
    }
    if (switchInfo.defaultBlock) {
      switchInfo.defaultBlock->payloadSeed.assign(
          switchInfo.defaultBlock->blockArgs.begin(),
          switchInfo.defaultBlock->blockArgs.end());
      switchInfo.defaultBlock->payloadBlockArgOffset =
          switchInfo.defaultBlock->payloadSeed.size() >=
                  switchInfo.defaultBlock->blockArgs.size()
              ? switchInfo.defaultBlock->payloadSeed.size() -
                    switchInfo.defaultBlock->blockArgs.size()
              : 0;
      if (switchInfo.defaultBlock->original &&
          !switchInfo.defaultBlock->original->empty())
        if (auto yield = llvm::dyn_cast<simt::dialect::YieldOp>(
                switchInfo.defaultBlock->original->getTerminator()))
          switchInfo.defaultBlock->payloadSeed.assign(
              yield.getResults().begin(), yield.getResults().end());
      switchInfo.defaultBlock->payloadBlockArgOffset =
          switchInfo.defaultBlock->payloadSeed.size() >=
                  switchInfo.defaultBlock->blockArgs.size()
              ? switchInfo.defaultBlock->payloadSeed.size() -
                    switchInfo.defaultBlock->blockArgs.size()
              : 0;
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

      if (caseInfo && caseInfo->original) {
        auto args = caseInfo->original->getArguments();
        for (unsigned i = 0; i < switchInfo.carriedCount && i < args.size(); ++i)
          record.carriedValues.push_back(args[i]);
      }
      if (switchInfo.hasControlFlags && caseInfo && caseInfo->original &&
          !caseInfo->original->empty()) {
        if (auto yield = llvm::dyn_cast<simt::dialect::YieldOp>(
                caseInfo->original->getTerminator())) {
          auto results = yield.getResults();
          if (results.size() >= switchInfo.payloadCount + 3) {
            record.matchSeen = results[switchInfo.payloadCount + 0];
            record.fallthrough = results[switchInfo.payloadCount + 1];
            record.switchDone = results[switchInfo.payloadCount + 2];
            record.payloadValues.append(results.begin(),
                                        results.begin() + switchInfo.payloadCount);
            record.controlValues.append(results.begin() + switchInfo.payloadCount,
                                        results.end());
          } else {
            record.payloadValues.append(results.begin(), results.end());
          }
        } else if (caseInfo) {
          auto args = caseInfo->original->getArguments();
          unsigned payloadStart = switchInfo.carriedCount;
          unsigned payloadEnd = std::min<unsigned>(args.size(),
                                                   payloadStart + switchInfo.payloadCount);
          record.payloadValues.append(args.begin() + payloadStart,
                                      args.begin() + payloadEnd);
        }
      }
      if (record.payloadValues.empty() && caseInfo && caseInfo->original) {
        auto args = caseInfo->original->getArguments();
        unsigned payloadStart = switchInfo.carriedCount;
        unsigned payloadEnd = std::min<unsigned>(args.size(),
                                                 payloadStart + switchInfo.payloadCount);
        record.payloadValues.append(args.begin() + payloadStart,
                                    args.begin() + payloadEnd);
      }
      switchInfo.caseRecords.push_back(record);
    }

    switchInfo.defaultRecord = SwitchInfo::CaseRecord();
    switchInfo.defaultRecord.block = switchInfo.defaultBlock;
    if (switchInfo.defaultBlock && switchInfo.defaultBlock->original) {
      auto args = switchInfo.defaultBlock->original->getArguments();
      for (unsigned i = 0; i < switchInfo.carriedCount && i < args.size(); ++i)
        switchInfo.defaultRecord.carriedValues.push_back(args[i]);
    }
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
          switchInfo.defaultRecord.payloadValues.append(
              results.begin(), results.begin() + switchInfo.payloadCount);
          switchInfo.defaultRecord.controlValues.append(
              results.begin() + switchInfo.payloadCount, results.end());
        }
      }
    }
    if (switchInfo.defaultRecord.payloadValues.empty() &&
        switchInfo.defaultBlock && switchInfo.defaultBlock->original) {
      auto args = switchInfo.defaultBlock->original->getArguments();
      for (unsigned i = 0; i < switchInfo.carriedCount && i < args.size(); ++i)
        switchInfo.defaultRecord.carriedValues.push_back(args[i]);
      unsigned payloadStart = switchInfo.carriedCount;
      unsigned payloadEnd = std::min<unsigned>(args.size(),
                                               payloadStart + switchInfo.payloadCount);
      switchInfo.defaultRecord.payloadValues.append(args.begin() + payloadStart,
                                                    args.begin() + payloadEnd);
    }

    auto buildCombinedPayload = [&](SwitchInfo::CaseRecord &record,
                                    bool includeControl) {
      llvm::SmallVector<mlir::Value, 8> combined;
      combined.append(record.carriedValues.begin(), record.carriedValues.end());
      combined.append(record.payloadValues.begin(), record.payloadValues.end());
      if (includeControl)
        combined.append(record.controlValues.begin(), record.controlValues.end());
      return combined;
    };

    auto propagateIfPossible = [&](BlockInfo *source, BlockInfo *dest,
                                   llvm::ArrayRef<mlir::Value> values) -> LogicalResult {
      if (!dest || values.empty())
        return success();
      if (!source)
        source = dest;
      return propagatePayload(*source, *dest, values);
    };

    for (auto &record : switchInfo.caseRecords) {
      bool includeControl = switchInfo.hasControlFlags;
      auto combined = buildCombinedPayload(record, includeControl);
      if (failed(propagateIfPossible(switchInfo.parent, record.block, combined)))
        return failure();
      if (record.nextCase)
        if (failed(propagateIfPossible(record.block, record.nextCase, combined)))
          return failure();
    }

    auto defaultCombined = buildCombinedPayload(switchInfo.defaultRecord,
                                                switchInfo.hasControlFlags);
    if (failed(propagateIfPossible(switchInfo.parent, switchInfo.defaultBlock,
                                   defaultCombined)))
      return failure();
  }

  // TODO: tighten propagation by iterating until convergence once back-edge
  // payload materialisation lands.
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

        BlockInfo *mergeDest = ifInfo.mergeBlock;

        EdgeInfo thenEdge;
        thenEdge.source = &info;
        thenEdge.dest = ifInfo.thenBlock;
        thenEdge.kind = EdgeInfo::ConditionalTrue;
        thenEdge.condition = ifInfo.condition;
        thenEdge.origin = op;
        if (thenEdge.dest && failed(ensurePayloadShape(thenEdge)))
          return failure();
        edges.push_back(std::move(thenEdge));
        info.outgoingEdges.push_back(edges.size() - 1);

        EdgeInfo elseEdge;
        elseEdge.source = &info;
        elseEdge.dest = ifInfo.elseBlock ? ifInfo.elseBlock : mergeDest;
        elseEdge.kind = EdgeInfo::ConditionalFalse;
        elseEdge.condition = ifInfo.condition;
        elseEdge.origin = op;
        if (elseEdge.dest && failed(ensurePayloadShape(elseEdge)))
          return failure();
        edges.push_back(std::move(elseEdge));
        info.outgoingEdges.push_back(edges.size() - 1);

        auto addMergeEdge = [&](BlockInfo *source,
                                llvm::ArrayRef<mlir::Value> values,
                                mlir::Operation *origin) -> LogicalResult {
          if ((!source && !origin) || !mergeDest)
            return success();

          BlockInfo *edgeSource = source;
          if (origin && origin->getBlock())
            if (BlockInfo *originInfo = lookupBlockInfo(origin->getBlock()))
              edgeSource = originInfo;
          if (!edgeSource)
            return success();

          EdgeInfo mergeEdge;
          mergeEdge.source = edgeSource;
          mergeEdge.dest = mergeDest;
          mergeEdge.kind = EdgeInfo::Plain;
          mergeEdge.origin = origin;

          SmallVector<mlir::Value, 8> payload;
          if (ifInfo.parent)
            payload.append(ifInfo.parent->payloadSeed.begin(),
                           ifInfo.parent->payloadSeed.end());
          if (!values.empty())
            payload.append(values.begin(), values.end());

          if (mergeDest && payload.size() < mergeDest->payloadSeed.size()) {
            auto begin = mergeDest->payloadSeed.begin() + payload.size();
            payload.append(begin, mergeDest->payloadSeed.end());
          }

          if (payload.empty() && mergeDest)
            payload.append(mergeDest->payloadSeed.begin(),
                           mergeDest->payloadSeed.end());

          mergeEdge.payload = std::move(payload);
          if (failed(ensurePayloadShape(mergeEdge)))
            return failure();

          edges.push_back(std::move(mergeEdge));
          edgeSource->outgoingEdges.push_back(edges.size() - 1);
          return success();
        };

        if (failed(addMergeEdge(ifInfo.thenBlock, ifInfo.thenYieldValues,
                                 ifInfo.thenYieldOp)))
          return failure();
        if (ifInfo.elseBlock) {
          if (failed(addMergeEdge(ifInfo.elseBlock, ifInfo.elseYieldValues,
                                   ifInfo.elseYieldOp)))
            return failure();
        }
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
        info.outgoingEdges.push_back(edges.size() - 1);

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
              trueEdge.condition = cond.getCondition();
              if (trueEdge.dest && failed(ensurePayloadShape(trueEdge)))
                return failure();
              edges.push_back(std::move(trueEdge));
              loopInfo.prepareBlock->outgoingEdges.push_back(edges.size() - 1);

              EdgeInfo falseEdge;
              falseEdge.source = loopInfo.prepareBlock;
              falseEdge.dest = loopInfo.mergeBlock;
              falseEdge.kind = EdgeInfo::ConditionalFalse;
              if (!loopInfo.forwardedToExit.empty())
                falseEdge.payload.assign(loopInfo.forwardedToExit.begin(),
                                          loopInfo.forwardedToExit.end());
              falseEdge.condition = cond.getCondition();
              if (falseEdge.dest && failed(ensurePayloadShape(falseEdge)))
                return failure();
              edges.push_back(std::move(falseEdge));
              loopInfo.prepareBlock->outgoingEdges.push_back(edges.size() - 1);
            }
          }
        }

        if (loopInfo.bodyBlock && loopInfo.bodyBlock->original) {
          for (Operation &nested : *loopInfo.bodyBlock->original) {
            if (auto cont = llvm::dyn_cast<simt::dialect::ContinueOp>(&nested)) {
              EdgeInfo backEdge;
              backEdge.source = loopInfo.bodyBlock;
              backEdge.dest = loopInfo.prepareBlock;
              backEdge.kind = EdgeInfo::LoopBackEdge;
              backEdge.origin = cont;
              backEdge.payload.assign(cont.getOperands().begin(),
                                      cont.getOperands().end());
              edges.push_back(std::move(backEdge));
              if (edges.back().dest && failed(ensurePayloadShape(edges.back())))
                return failure();
              loopInfo.bodyBlock->outgoingEdges.push_back(edges.size() - 1);
              continue;
            }
            if (auto breakOp = llvm::dyn_cast<simt::dialect::BreakOp>(&nested)) {
              EdgeInfo exitEdge;
              exitEdge.source = loopInfo.bodyBlock;
              exitEdge.dest = loopInfo.mergeBlock;
              exitEdge.kind = EdgeInfo::Plain;
              exitEdge.origin = breakOp;
              exitEdge.payload.assign(breakOp->getOperands().begin(),
                                      breakOp->getOperands().end());
              edges.push_back(std::move(exitEdge));
              if (edges.back().dest && failed(ensurePayloadShape(edges.back())))
                return failure();
              loopInfo.bodyBlock->outgoingEdges.push_back(edges.size() - 1);
              continue;
            }
            if (auto yieldOp = llvm::dyn_cast<simt::dialect::YieldOp>(&nested)) {
              EdgeInfo contEdge;
              contEdge.source = loopInfo.bodyBlock;
              contEdge.dest = loopInfo.prepareBlock;
              contEdge.kind = EdgeInfo::LoopBackEdge;
              contEdge.origin = yieldOp;
              contEdge.payload.assign(yieldOp->getOperands().begin(),
                                      yieldOp->getOperands().end());
              edges.push_back(std::move(contEdge));
              if (edges.back().dest && failed(ensurePayloadShape(edges.back())))
                return failure();
              loopInfo.bodyBlock->outgoingEdges.push_back(edges.size() - 1);
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
          info.outgoingEdges.push_back(edges.size() - 1);

          mlir::Operation *yieldOp = nullptr;
          if (caseInfo && caseInfo->original)
            yieldOp = caseInfo->original->getTerminator();

          auto emitExitEdge = [&](EdgeInfo::Kind kind, BlockInfo *dest)
                                  -> LogicalResult {
            EdgeInfo exitEdge;
            exitEdge.source = caseInfo;
            exitEdge.dest = dest;
            exitEdge.kind = kind;
            exitEdge.payload.append(record.carriedValues.begin(),
                                     record.carriedValues.end());
            exitEdge.payload.append(record.payloadValues.begin(),
                                     record.payloadValues.end());
            if (kind != EdgeInfo::Plain) {
              exitEdge.isSwitchFallthrough = true;
              exitEdge.condition = record.fallthrough;
              exitEdge.switchDoneFlag = record.switchDone;
            }
            exitEdge.origin = yieldOp;
            if (exitEdge.dest && failed(ensurePayloadShape(exitEdge)))
              return failure();
           edges.push_back(std::move(exitEdge));
           if (caseInfo)
              caseInfo->outgoingEdges.push_back(edges.size() - 1);
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
            fallEdge.payload.append(record.carriedValues.begin(),
                                     record.carriedValues.end());
            fallEdge.payload.append(record.payloadValues.begin(),
                                     record.payloadValues.end());
            fallEdge.payload.append(record.controlValues.begin(),
                                     record.controlValues.end());
            if (fallEdge.dest && failed(ensurePayloadShape(fallEdge)))
              return failure();
           edges.push_back(std::move(fallEdge));
            if (caseInfo)
              caseInfo->outgoingEdges.push_back(edges.size() - 1);

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
          info.outgoingEdges.push_back(edges.size() - 1);

          if (switchInfo.defaultBlock) {
            EdgeInfo exitEdge;
            exitEdge.source = switchInfo.defaultBlock;
            exitEdge.dest = switchInfo.parent;
            exitEdge.kind = EdgeInfo::Plain;
            exitEdge.payload.append(
                switchInfo.defaultRecord.carriedValues.begin(),
                switchInfo.defaultRecord.carriedValues.end());
            exitEdge.payload.append(
                switchInfo.defaultRecord.payloadValues.begin(),
                switchInfo.defaultRecord.payloadValues.end());
            if (exitEdge.dest && failed(ensurePayloadShape(exitEdge)))
              return failure();
           edges.push_back(std::move(exitEdge));
            if (switchInfo.defaultBlock)
              switchInfo.defaultBlock->outgoingEdges.push_back(edges.size() - 1);
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

  structuredOpsInOrder.clear();

  Block *entryBlock = blockOrder.front();
  builder.setInsertionPointToStart(entryBlock);

  unsigned autoNameCounter = orderedInfos.size();

  // First pass: create structured block ops and map block arguments.
  for (BlockInfo *info : orderedInfos) {
    if (info->symbolName.empty()) {
      if (info == orderedInfos.front())
        info->symbolName = "entry";
      else
        info->symbolName =
            ("block" + std::to_string(autoNameCounter++));
    }
    auto symAttr = builder.getStringAttr(info->symbolName);
    auto blockOp = builder.create<simt::structured::BlockOp>(func.getLoc(),
                                                             symAttr,
                                                             mlir::Value());
    info->structuredOp = blockOp;
    info->structuredBody = &blockOp.getBody().front();
    info->structuredMaskArg = blockOp.getMaskArgument();
    info->currentMask = info->structuredMaskArg;
    info->structuredArgs.clear();
    info->payloadArgs.clear();
    info->originalTerminator = info->original ? info->original->getTerminator()
                                              : nullptr;
    structuredOpsInOrder.push_back(blockOp);

    if (info->original) {
      for (mlir::BlockArgument origArg : info->original->getArguments()) {
        auto newArg = info->structuredBody->addArgument(origArg.getType(),
                                                        origArg.getLoc());
        info->structuredArgs.push_back(newArg);
        mapper->map(origArg, newArg);
      }
    }

    info->payloadArgs.reserve(info->payloadSeed.size());
    for (auto [index, seed] : llvm::enumerate(info->payloadSeed)) {
      if (index < info->structuredArgs.size()) {
        info->payloadArgs.push_back(info->structuredArgs[index]);
        continue;
      }
      if (!seed) {
        info->payloadArgs.push_back(mlir::BlockArgument());
        continue;
      }
      auto arg = info->structuredBody->addArgument(seed.getType(), func.getLoc());
      info->payloadArgs.push_back(arg);
    }

    builder.setInsertionPointAfter(blockOp);
  }

  // Second pass: populate each structured block.
  for (BlockInfo *info : orderedInfos)
    if (failed(emitStructuredBlock(*info)))
      return failure();

  return success();
}

LogicalResult StructuredCFGBuilder::stabilisePayloadSeeds() {
  if (edges.empty())
    return success();

  auto mergePayload = [&](BlockInfo &dest, llvm::ArrayRef<mlir::Value> incoming,
                          bool &changed) -> LogicalResult {
    if (incoming.size() != dest.payloadSeed.size()) {
      func.emitError("payload arity mismatch while merging seeds");
      return failure();
    }
    for (auto [index, value] : llvm::enumerate(incoming)) {
      if (!value)
        continue;
      mlir::Value current = dest.payloadSeed[index];
      mlir::Value placeholder;
      if (index >= dest.payloadBlockArgOffset) {
        unsigned argIndex = index - dest.payloadBlockArgOffset;
        if (argIndex < dest.blockArgs.size())
          placeholder = dest.blockArgs[argIndex];
      }
      if (!current || current == placeholder) {
        if (current != value) {
          dest.payloadSeed[index] = value;
          changed = true;
        }
        continue;
      }
      if (value == current || value == placeholder)
        continue;
      if (placeholder) {
        if (current != placeholder) {
          dest.payloadSeed[index] = placeholder;
          changed = true;
        }
        continue;
      }
      auto diag = func.emitError("conflicting payload values for block");
      if (!dest.symbolName.empty())
        diag << " '" << dest.symbolName << "'";
      diag << " at index " << index;
      if (current) {
        diag << ", current=" << current;
      }
      if (value) {
        diag << ", incoming=" << value;
      }
      return failure();
    }
    return success();
  };

  llvm::SmallVector<BlockInfo *, 16> worklist;
  llvm::DenseSet<BlockInfo *> queued;
  auto enqueue = [&](BlockInfo *info) {
    if (!info)
      return;
    if (queued.insert(info).second)
      worklist.push_back(info);
  };

  for (EdgeInfo &edge : edges)
    if (edge.dest && !edge.payload.empty())
      enqueue(edge.dest);

  while (!worklist.empty()) {
    BlockInfo *dest = worklist.pop_back_val();
    queued.erase(dest);

    bool changed = false;
    for (EdgeInfo &edge : edges) {
      if (edge.dest != dest || edge.payload.empty())
        continue;
      if (failed(mergePayload(*dest, edge.payload, changed)))
        return failure();
    }

    if (changed)
      for (unsigned edgeIndex : dest->outgoingEdges) {
        EdgeInfo &edge = edges[edgeIndex];
        if (edge.dest)
          enqueue(edge.dest);
      }
  }

  for (EdgeInfo &edge : edges)
    if (edge.dest && edge.payload.empty())
      if (failed(ensurePayloadShape(edge)))
        return failure();

  return success();
}

LogicalResult StructuredCFGBuilder::cleanupOriginalCFG() {
  if (structuredOpsInOrder.empty())
    return success();

  SmallVector<Type, 8> entryArgTypes;
  SmallVector<Location, 8> entryArgLocs;
  if (!func.getFunctionBody().empty()) {
    Block &oldEntry = func.getFunctionBody().front();
    entryArgTypes.reserve(oldEntry.getNumArguments());
    entryArgLocs.reserve(oldEntry.getNumArguments());
    for (BlockArgument arg : oldEntry.getArguments()) {
      entryArgTypes.push_back(arg.getType());
      entryArgLocs.push_back(arg.getLoc());
    }
  } else {
    auto argTypes = func.getArgumentTypes();
    entryArgTypes.append(argTypes.begin(), argTypes.end());
    entryArgLocs.resize(entryArgTypes.size(), func.getLoc());
  }

  // Detach structured ops from their current blocks so we can rebuild the
  // function body from scratch.
  for (simt::structured::BlockOp blockOp : structuredOpsInOrder)
    if (blockOp)
      blockOp->remove();

  // Drop all existing blocks in the function.
  mlir::Region &body = func.getFunctionBody();
  while (!body.empty())
    body.front().erase();

  // Recreate a single entry block to host the structured blocks.
  mlir::Block *entry = &body.emplaceBlock();

  for (auto [index, type] : llvm::enumerate(entryArgTypes)) {
    Location loc = index < entryArgLocs.size() ? entryArgLocs[index]
                                               : func.getLoc();
    entry->addArgument(type, loc);
  }

  OpBuilder builder(entry, entry->begin());
  for (simt::structured::BlockOp blockOp : structuredOpsInOrder)
    if (blockOp)
      builder.insert(blockOp);

  builder.setInsertionPointToEnd(entry);
  if (func.getNumResults() == 0)
    builder.create<func::ReturnOp>(func.getLoc());

  return success();
}

LogicalResult StructuredCFGBuilder::emitStructuredBlock(BlockInfo &info) {
  simt::structured::BlockOp blockOp = info.structuredOp;
  if (!blockOp)
    return success();

  auto setSymbolAttr = [&](StringRef attrName, BlockInfo *target) {
    if (!target || !target->structuredOp)
      return;
    auto ref = FlatSymbolRefAttr::get(func.getContext(),
                                      target->structuredOp.getSymName());
    blockOp->setAttr(attrName, ref);
  };

  if (info.mergeTarget)
    setSymbolAttr(blockOp.getMergeTargetAttrName(),
                  lookupBlockInfo(info.mergeTarget));
  else
    blockOp->removeAttr(blockOp.getMergeTargetAttrName());

  if (info.continueTarget)
    setSymbolAttr(blockOp.getContinueTargetAttrName(),
                  lookupBlockInfo(info.continueTarget));
  else
    blockOp->removeAttr(blockOp.getContinueTargetAttrName());

  if (failed(materialiseMaskEntry(info)))
    return failure();

  OpBuilder bodyBuilder(info.structuredBody, info.structuredBody->begin());

  for (auto [index, seed] : llvm::enumerate(info.payloadSeed)) {
    if (!seed)
      continue;
    if (index >= info.payloadArgs.size())
      continue;
    mlir::BlockArgument arg = info.payloadArgs[index];
    if (!arg)
      continue;
    mapper->map(seed, arg);
  }

  auto mapValueInline = [&](Value v, Operation *contextOp)
                            -> std::optional<Value> {
    if (!v)
      return Value();
    if (Value mapped = mapper->lookupOrNull(v))
      return mapped;
    if (contextOp)
      contextOp->emitError("unable to remap operand during structured lowering");
    return std::nullopt;
  };
  auto mapValuesInline = [&](ValueRange vals, Operation *contextOp,
                             SmallVectorImpl<Value> &out) -> LogicalResult {
    for (Value v : vals) {
      auto mapped = mapValueInline(v, contextOp);
      if (!mapped)
        return failure();
      out.push_back(*mapped);
    }
    return success();
  };

  if (info.original) {
    for (Operation &op : info.original->without_terminator()) {
      if (isa<simt::structured::BlockOp>(&op))
        continue;
      if (auto nestedIf = dyn_cast<simt::dialect::IfOp>(&op)) {
        if (auto it = ifInfos.find(nestedIf.getOperation());
            it != ifInfos.end()) {
          IfInfo &nestedInfo = it->second;
          ensureIfMergeBlock(nestedInfo, nestedIf.getOperation());
          OpBuilder::InsertionGuard guard(bodyBuilder);
          bodyBuilder.setInsertionPointToEnd(info.structuredBody);
          if (failed(emitStructuredIf(info, nestedInfo, bodyBuilder)))
            return failure();
          for (auto [index, result] : llvm::enumerate(nestedIf.getResults()))
            mapper->map(result, nestedInfo.mergeArgs[index]);
          continue;
        }
      }
      if (auto cont = dyn_cast<simt::dialect::ContinueOp>(&op)) {
        SmallVector<unsigned, 2> edgesForContinue;
        auto &outgoing = info.outgoingEdges;
        for (auto it = outgoing.begin(); it != outgoing.end();) {
          EdgeInfo &edge = edges[*it];
          if (edge.origin == cont) {
            edgesForContinue.push_back(*it);
            it = outgoing.erase(it);
          } else {
            ++it;
          }
        }
        if (edgesForContinue.empty()) {
          cont.emitError("missing structured continue edge");
          return failure();
        }
        for (unsigned edgeIndex : edgesForContinue) {
          EdgeInfo &edge = edges[edgeIndex];
          if (!edge.dest || !edge.dest->structuredOp) {
            cont.emitError("missing destination for structured continue");
            return failure();
          }
          SmallVector<Value> operands;
          if (failed(mapValuesInline(edge.payload, cont, operands)))
            return failure();
          Value mask = info.currentMask ? info.currentMask : info.structuredMaskArg;
          if (info.requestsMaskPush) {
            auto pop = bodyBuilder.create<simt::structured::MaskPopOp>(
                cont.getLoc(), mask.getType());
            mask = pop.getResult();
            info.currentMask = mask;
          }
          auto targetAttr = FlatSymbolRefAttr::get(
              func.getContext(), edge.dest->structuredOp.getSymName());
          bodyBuilder.create<simt::structured::BranchOp>(cont.getLoc(), mask,
                                                         targetAttr, operands);
        }
        continue;
      }
      if (auto brk = dyn_cast<simt::dialect::BreakOp>(&op)) {
        SmallVector<unsigned, 2> edgesForBreak;
        auto &outgoing = info.outgoingEdges;
        for (auto it = outgoing.begin(); it != outgoing.end();) {
          EdgeInfo &edge = edges[*it];
          if (edge.origin == brk) {
            edgesForBreak.push_back(*it);
            it = outgoing.erase(it);
          } else {
            ++it;
          }
        }
        if (edgesForBreak.empty()) {
          brk.emitError("missing structured break edge");
          return failure();
        }
        for (unsigned edgeIndex : edgesForBreak) {
          EdgeInfo &edge = edges[edgeIndex];
          if (!edge.dest || !edge.dest->structuredOp) {
            brk.emitError("missing destination for structured break");
            return failure();
          }
          SmallVector<Value> operands;
          if (failed(mapValuesInline(edge.payload, brk, operands)))
            return failure();
          Value mask = info.currentMask ? info.currentMask : info.structuredMaskArg;
          if (info.requestsMaskPush) {
            auto pop = bodyBuilder.create<simt::structured::MaskPopOp>(
                brk.getLoc(), mask.getType());
            mask = pop.getResult();
            info.currentMask = mask;
          }
          auto targetAttr = FlatSymbolRefAttr::get(
              func.getContext(), edge.dest->structuredOp.getSymName());
          bodyBuilder.create<simt::structured::BranchOp>(brk.getLoc(), mask,
                                                         targetAttr, operands);
        }
        continue;
      }
      if (isa<cf::BranchOp, cf::CondBranchOp, func::ReturnOp>(&op))
        continue;
      if (auto active = dyn_cast<simt::dialect::ActiveMaskOp>(&op)) {
        mapper->map(active.getResult(), info.currentMask);
        continue;
      }
      Operation *cloned = bodyBuilder.clone(op, *mapper);
      mapper->map(op.getResults(), cloned->getResults());
    }
  }
  if (!info.structuredBody->empty() &&
      info.structuredBody->back().hasTrait<OpTrait::IsTerminator>())
    return success();

  return emitStructuredTerminator(info);
}

LogicalResult StructuredCFGBuilder::emitStructuredIf(BlockInfo &header,
                                                     IfInfo &info,
                                                     OpBuilder &builder) {
  constexpr unsigned kInvalidEdge = std::numeric_limits<unsigned>::max();
  unsigned trueEdgeIdx = kInvalidEdge;
  unsigned falseEdgeIdx = kInvalidEdge;

  for (unsigned edgeIndex : header.outgoingEdges) {
    EdgeInfo &edge = edges[edgeIndex];
    if (edge.origin != info.op)
      continue;
    if (edge.kind == EdgeInfo::ConditionalTrue)
      trueEdgeIdx = edgeIndex;
    if (edge.kind == EdgeInfo::ConditionalFalse)
      falseEdgeIdx = edgeIndex;
  }

  if (trueEdgeIdx == kInvalidEdge || falseEdgeIdx == kInvalidEdge) {
    if (info.op)
      info.op->emitError("missing structured edges for if lowering");
    return failure();
  }

  EdgeInfo &trueEdge = edges[trueEdgeIdx];
  EdgeInfo &falseEdge = edges[falseEdgeIdx];

  auto getTargetAttr = [&](BlockInfo *block) -> FlatSymbolRefAttr {
    if (!block || !block->structuredOp)
      return FlatSymbolRefAttr();
    return FlatSymbolRefAttr::get(builder.getContext(),
                                  block->structuredOp.getSymName());
  };

  auto mapValue = [&](Value v, StringRef context) -> std::optional<Value> {
    if (!v)
      return Value();
    if (Value mapped = mapper->lookupOrNull(v))
      return mapped;
    if (info.op)
      info.op->emitError() << "unable to map value in " << context;
    return std::nullopt;
  };

  auto mapPayload = [&](EdgeInfo &edge, StringRef context,
                        llvm::SmallVectorImpl<Value> &out) -> LogicalResult {
    for (Value v : edge.payload) {
      auto mapped = mapValue(v, context);
      if (!mapped)
        return failure();
      out.push_back(*mapped);
    }
    return success();
  };

  SmallVector<Value> truePayload;
  if (failed(mapPayload(trueEdge, "if true payload", truePayload)))
    return failure();

  SmallVector<Value> falsePayload;
  if (failed(mapPayload(falseEdge, "if false payload", falsePayload)))
    return failure();

  auto mappedCond = mapValue(info.condition, "if condition");
  if (!mappedCond)
    return failure();

  Value mask = header.currentMask ? header.currentMask : header.structuredMaskArg;

  builder.create<simt::structured::CondBranchOp>(
      info.op ? info.op->getLoc() : header.structuredOp.getLoc(), *mappedCond,
      mask, mask, getTargetAttr(trueEdge.dest), getTargetAttr(falseEdge.dest),
      truePayload, falsePayload, FlatSymbolRefAttr(),
      simt::structured::ReconvergencePolicyAttr());

  auto removeEdgeFromHeader = [&](unsigned index) {
    if (index == kInvalidEdge)
      return;
    auto it = llvm::find(header.outgoingEdges, index);
    if (it != header.outgoingEdges.end())
      header.outgoingEdges.erase(it);
  };

  removeEdgeFromHeader(trueEdgeIdx);
  removeEdgeFromHeader(falseEdgeIdx);

  return success();
}

LogicalResult StructuredCFGBuilder::emitStructuredTerminator(BlockInfo &source) {
  if (!source.structuredBody)
    return success();

  if (failed(materialiseMaskExit(source)))
    return failure();

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

  auto mapValues = [&](mlir::ValueRange vals, SmallVectorImpl<Value> &out,
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

    if (!hasFunctionReturn) {
      functionReturnValues.assign(results.begin(), results.end());
      hasFunctionReturn = true;
    } else {
      if (functionReturnValues.size() != results.size()) {
        ret.emitError("inconsistent return arity in structured lowering");
        return failure();
      }
      for (auto [lhs, rhs] : llvm::zip(functionReturnValues, results))
        if (lhs.getType() != rhs.getType()) {
          ret.emitError("inconsistent return types in structured lowering");
          return failure();
        }
    }
    return success();
  }

  auto getTargetAttr = [&](mlir::Block *target) -> FlatSymbolRefAttr {
    if (!target)
      return FlatSymbolRefAttr();
    if (BlockInfo *targetInfo = lookupBlockInfo(target))
      if (targetInfo->structuredOp)
        return FlatSymbolRefAttr::get(
            builder.getContext(),
            llvm::cast<simt::structured::BlockOp>(targetInfo->structuredOp)
                .getSymName());
    return FlatSymbolRefAttr();
  };

  if (source.outgoingEdges.empty()) {
    if (!origTerm)
      return success();

    Value mask = source.currentMask ? source.currentMask : source.structuredMaskArg;

    if (auto branch = dyn_cast<cf::BranchOp>(origTerm)) {
      SmallVector<Value> destOperands;
      if (failed(mapValues(branch.getDestOperands(), destOperands,
                           "branch payload")))
        return failure();
      auto targetAttr = getTargetAttr(branch.getDest());
      builder.create<simt::structured::BranchOp>(loc, mask, targetAttr,
                                                 destOperands);
      return success();
    }

    if (auto cond = dyn_cast<cf::CondBranchOp>(origTerm)) {
      SmallVector<Value> trueOperands;
      SmallVector<Value> falseOperands;
      if (failed(mapValues(cond.getTrueDestOperands(), trueOperands,
                           "true payload")) ||
          failed(mapValues(cond.getFalseDestOperands(), falseOperands,
                           "false payload")))
        return failure();
      auto condition = mapValue(cond.getCondition(), "condition");
      if (!condition)
        return failure();

      auto trueTarget = getTargetAttr(cond.getTrueDest());
      auto falseTarget = getTargetAttr(cond.getFalseDest());
      builder.create<simt::structured::CondBranchOp>(
          loc, *condition, mask, mask, trueTarget, falseTarget, trueOperands,
          falseOperands, FlatSymbolRefAttr(),
          simt::structured::ReconvergencePolicyAttr());
      return success();
    }

    origTerm->emitError("unable to synthesize structured terminator");
    return failure();
  }

  constexpr unsigned kInvalidEdge = std::numeric_limits<unsigned>::max();
  unsigned trueEdgeIdx = kInvalidEdge;
  unsigned falseEdgeIdx = kInvalidEdge;
  SmallVector<unsigned, 4> plainEdgeIdxs;

  for (unsigned edgeIndex : source.outgoingEdges) {
    EdgeInfo &edge = edges[edgeIndex];
    switch (edge.kind) {
    case EdgeInfo::ConditionalTrue:
      trueEdgeIdx = edgeIndex;
      break;
    case EdgeInfo::ConditionalFalse:
      falseEdgeIdx = edgeIndex;
      break;
    case EdgeInfo::LoopBackEdge:
    case EdgeInfo::Plain:
      plainEdgeIdxs.push_back(edgeIndex);
      break;
    }
  }

  auto buildBranch = [&](unsigned edgeIndex) -> LogicalResult {
    if (edgeIndex == kInvalidEdge)
      return failure();
    EdgeInfo &edge = edges[edgeIndex];
    if (!edge.dest || !edge.dest->structuredOp) {
      if (origTerm)
        origTerm->emitError("branch edge missing destination");
      return failure();
    }
    SmallVector<Value> operands;
    if (failed(mapValues(edge.payload, operands, "branch payload")))
      return failure();
    Value mask = source.currentMask ? source.currentMask : source.structuredMaskArg;
    auto targetAttr = FlatSymbolRefAttr::get(builder.getContext(),
                                             edge.dest->structuredOp.getSymName());
    builder.create<simt::structured::BranchOp>(loc, mask, targetAttr,
                                               operands);
    return success();
  };

  if (trueEdgeIdx != kInvalidEdge && falseEdgeIdx != kInvalidEdge) {
    EdgeInfo &trueEdge = edges[trueEdgeIdx];
    EdgeInfo &falseEdge = edges[falseEdgeIdx];
    auto cond = mapValue(trueEdge.condition, "cond branch condition");
    if (!cond) {
      if (origTerm)
        origTerm->emitError("conditional edge missing condition value");
      return failure();
    }

    Value finalCond = *cond;
    if (trueEdge.isSwitchFallthrough) {
      auto mappedDoneOpt = mapValue(trueEdge.switchDoneFlag,
                                    "switch done flag");
      if (!mappedDoneOpt)
        return failure();
      Value doneVal = *mappedDoneOpt;
      if (!doneVal) {
        if (origTerm)
          origTerm->emitError("switch fallthrough missing done flag");
        return failure();
      }
      auto constFalse = builder.create<arith::ConstantIntOp>(loc, 0, 1);
      Value notDone = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, doneVal, constFalse);
      finalCond = builder.create<arith::AndIOp>(loc, *cond, notDone);
    }

    SmallVector<Value> truePayload;
    SmallVector<Value> falsePayload;
    if (failed(mapValues(trueEdge.payload, truePayload,
                         "cond true payload")) ||
        failed(mapValues(falseEdge.payload, falsePayload,
                         "cond false payload")))
      return failure();

    auto trueTarget = trueEdge.dest && trueEdge.dest->structuredOp
                          ? FlatSymbolRefAttr::get(
                                builder.getContext(),
                                trueEdge.dest->structuredOp.getSymName())
                          : FlatSymbolRefAttr();
    auto falseTarget = falseEdge.dest && falseEdge.dest->structuredOp
                           ? FlatSymbolRefAttr::get(
                                 builder.getContext(),
                                 falseEdge.dest->structuredOp.getSymName())
                           : FlatSymbolRefAttr();

    Value trueMask = source.currentMask ? source.currentMask
                                        : source.structuredMaskArg;
    Value falseMask = trueMask;

    builder.create<simt::structured::CondBranchOp>(
        loc, finalCond, trueMask, falseMask, trueTarget, falseTarget,
        truePayload, falsePayload, FlatSymbolRefAttr(),
        simt::structured::ReconvergencePolicyAttr());
    return success();
  }

  if (plainEdgeIdxs.size() == 1)
    return buildBranch(plainEdgeIdxs.front());

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
  info.mergeBlock = &header;
  info.mergeBlock = &header;
  info.resultTypes.clear();
  info.results.clear();
  info.resultTypes.reserve(ifOp.getNumResults());
  info.results.reserve(ifOp.getNumResults());
  for (mlir::Value result : ifOp.getResults()) {
    info.resultTypes.push_back(result.getType());
    info.results.push_back(result);
  }

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
  } else {
    info.elseBlock = nullptr;
  }

  BlockInfo &mergeInfo = ensureIfMergeBlock(info, op);
  info.mergeBlock = &mergeInfo;

  mlir::Block *mergeTargetBlock = mergeInfo.original;
  if (mergeTargetBlock) {
    thenInfo.mergeTarget = mergeTargetBlock;
    if (info.elseBlock)
      info.elseBlock->mergeTarget = mergeTargetBlock;
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
  info.mergeBlock = nullptr;
  info.mergeArgs.clear();

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

  BlockInfo &mergeInfo = ensureLoopMergeBlock(info, op);

  prepareInfo.mergeTarget = mergeInfo.original;
  bodyInfo.mergeTarget = mergeInfo.original;

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
    if (!edge.dest->payloadSeed.empty()) {
      if (edge.payload.size() > edge.dest->payloadSeed.size()) {
        func.emitError()
            << "edge payload arity exceeds destination seed for block '"
            << edge.dest->symbolName << "' (expected "
            << edge.dest->payloadSeed.size() << ", actual "
            << edge.payload.size() << ")";
        return failure();
      }
      if (edge.payload.size() < edge.dest->payloadSeed.size()) {
        edge.payload.append(edge.dest->payloadSeed.begin() + edge.payload.size(),
                            edge.dest->payloadSeed.end());
      }
    } else {
      edge.dest->payloadSeed.assign(edge.payload.begin(), edge.payload.end());
      edge.dest->payloadBlockArgOffset =
          edge.dest->payloadSeed.size() >= edge.dest->blockArgs.size()
              ? edge.dest->payloadSeed.size() - edge.dest->blockArgs.size()
              : 0;
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
  dest.payloadBlockArgOffset = dest.payloadSeed.size() >= dest.blockArgs.size()
                                   ? dest.payloadSeed.size() -
                                         dest.blockArgs.size()
                                   : 0;
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
        return FlatSymbolRefAttr::get(builder.getContext(),
                                      targetInfo->structuredOp.getSymName());
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


StructuredCFGBuilder::BlockInfo &
StructuredCFGBuilder::ensureLoopMergeBlock(LoopInfo &loopInfo,
                                           mlir::Operation *loopOperation) {
  auto loopOp = llvm::dyn_cast<simt::dialect::LoopOp>(loopOperation);
  assert(loopOp && "expected simt_step.loop operation");

  mlir::Block *mergeBlock = nullptr;
  if (!loopInfo.mergeBlock) {
    mlir::Block *parentBlock = loopOp->getBlock();
    auto splitPoint = std::next(loopOp->getIterator());
    mergeBlock = parentBlock->splitBlock(splitPoint);

    BlockInfo &mergeInfo = getOrCreateBlockInfo(mergeBlock);
    if (mergeInfo.symbolName.empty()) {
      if (loopInfo.parent)
        mergeInfo.symbolName =
            loopInfo.parent->symbolName + std::string(".merge");
      else
        mergeInfo.symbolName = "block" + std::to_string(blockInfos.size());
    }

    auto orderIt = llvm::find(blockOrder, parentBlock);
    if (orderIt != blockOrder.end())
      blockOrder.insert(orderIt + 1, mergeBlock);
    else
      blockOrder.push_back(mergeBlock);

    loopInfo.mergeBlock = &mergeInfo;
  } else {
    mergeBlock = loopInfo.mergeBlock->original;
  }

  BlockInfo &mergeInfo = *loopInfo.mergeBlock;
  if (!mergeBlock)
    mergeBlock = mergeInfo.original;

  if (!mergeBlock)
    llvm::report_fatal_error("loop merge block must exist after splitting");

  unsigned numResults = loopOp.getNumResults();
  unsigned currentArgs = mergeBlock->getNumArguments();
  if (currentArgs > numResults) {
    loopOperation->emitOpError(
        "loop merge block already has more arguments than loop results");
    llvm::report_fatal_error("invalid loop merge configuration");
  }

  Location loopLoc = loopOp.getLoc();
  while (mergeBlock->getNumArguments() < numResults)
    (void)mergeBlock->addArgument(loopOp.getResultTypes()
                                      [mergeBlock->getNumArguments()],
                                  loopLoc);

  if (mergeBlock->getNumArguments() != numResults) {
    loopOperation->emitOpError(
        "loop merge block argument count must match loop results");
    llvm::report_fatal_error("mismatched loop merge arity");
  }

  mergeInfo.blockArgs.clear();
  mergeInfo.carriedTypes.clear();
  for (mlir::BlockArgument arg : mergeBlock->getArguments()) {
    mergeInfo.blockArgs.push_back(arg);
    mergeInfo.carriedTypes.push_back(arg.getType());
  }
  mergeInfo.payloadSeed.assign(mergeInfo.blockArgs.begin(),
                               mergeInfo.blockArgs.end());
  mergeInfo.payloadBlockArgOffset = mergeInfo.payloadSeed.size() >=
                                            mergeInfo.blockArgs.size()
                                        ? mergeInfo.payloadSeed.size() -
                                              mergeInfo.blockArgs.size()
                                        : 0;
  mergeInfo.requestsMaskPop = true;
  mergeInfo.requestsMaskPush = false;
  mergeInfo.mergeTarget = nullptr;
  mergeInfo.continueTarget = nullptr;

  loopInfo.mergeArgs.clear();
  for (auto [index, result] : llvm::enumerate(loopOp.getResults())) {
    mlir::BlockArgument arg = mergeBlock->getArgument(index);
    loopInfo.mergeArgs.push_back(arg);
    if (mapper)
      mapper->map(result, arg);
    result.replaceAllUsesWith(arg);
  }

  mergeInfo.original = mergeBlock;

  return mergeInfo;
}

StructuredCFGBuilder::BlockInfo &
StructuredCFGBuilder::ensureIfMergeBlock(IfInfo &ifInfo,
                                         mlir::Operation *ifOperation) {
  auto ifOp = llvm::dyn_cast<simt::dialect::IfOp>(ifOperation);
  assert(ifOp && "expected simt_step.if operation");

  mlir::Block *parentBlock = ifOp->getBlock();
  mlir::Block *mergeBlock = nullptr;

  if (ifInfo.mergeBlock && ifInfo.mergeBlock->original &&
      ifInfo.mergeBlock->original != parentBlock) {
    mergeBlock = ifInfo.mergeBlock->original;
  } else {
    auto splitPoint = std::next(ifOp->getIterator());
    mergeBlock = parentBlock->splitBlock(splitPoint);

    BlockInfo &mergeInfo = getOrCreateBlockInfo(mergeBlock);
    if (mergeInfo.symbolName.empty()) {
      mergeInfo.symbolName = ifInfo.parent && !ifInfo.parent->symbolName.empty()
                                 ? (ifInfo.parent->symbolName + ".merge")
                                 : ("block" + std::to_string(blockInfos.size()));
    }

    auto orderIt = llvm::find(blockOrder, parentBlock);
    if (orderIt != blockOrder.end())
      blockOrder.insert(orderIt + 1, mergeBlock);
    else
      blockOrder.push_back(mergeBlock);

    ifInfo.mergeBlock = &mergeInfo;
  }

  BlockInfo &mergeInfo = *ifInfo.mergeBlock;
  if (!mergeBlock)
    mergeBlock = mergeInfo.original;

  if (!mergeBlock)
    llvm::report_fatal_error("if merge block must exist after splitting");

  // Ensure block arguments mirror the if results.
  unsigned numResults = ifOp.getNumResults();
  while (mergeBlock->getNumArguments() < numResults)
    (void)mergeBlock->addArgument(ifOp.getResultTypes()
                                      [mergeBlock->getNumArguments()],
                                  ifOp.getLoc());

  mergeInfo.blockArgs.clear();
  mergeInfo.carriedTypes.clear();
  mergeInfo.payloadSeed.clear();
  for (mlir::BlockArgument arg : mergeBlock->getArguments()) {
    mergeInfo.blockArgs.push_back(arg);
    mergeInfo.carriedTypes.push_back(arg.getType());
    mergeInfo.payloadSeed.push_back(arg);
  }

  ifInfo.mergeArgs.clear();
  for (mlir::BlockArgument arg : mergeBlock->getArguments())
    ifInfo.mergeArgs.push_back(arg);

  // Redirect SSA uses of the if results to the merge block arguments.
  for (auto [idx, result] : llvm::enumerate(ifOp.getResults())) {
    if (idx < mergeBlock->getNumArguments())
      result.replaceAllUsesWith(mergeBlock->getArgument(idx));
  }

  ifInfo.results.clear();
  for (mlir::BlockArgument arg : mergeBlock->getArguments())
    ifInfo.results.push_back(arg);

  return mergeInfo;
}

LogicalResult StructuredCFGBuilder::materialiseMaskExit(BlockInfo &info) {
  if (!info.structuredBody)
    return success();

  if (!info.requestsMaskPush)
    return success();

  Location loc = info.originalTerminator ? info.originalTerminator->getLoc()
                                          : func.getLoc();
  Value current = info.currentMask ? info.currentMask : info.structuredMaskArg;
  if (!current)
    return success();

  OpBuilder builder(info.structuredBody, info.structuredBody->end());
  auto pop = builder.create<simt::structured::MaskPopOp>(loc, current.getType());
  info.currentMask = pop.getResult();
  return success();
}

} // namespace simt::conversion
