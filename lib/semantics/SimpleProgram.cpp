#include "simt-step/semantics/SimpleProgram.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <bit>
#include <iterator>
#include <utility>

#include <mlir/IR/Block.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OpDefinition.h>

#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>

namespace simt::semantics {

namespace {
using StepType = SimpleProgramRunner::StepType;
using StateType = SimpleProgramRunner::StateType;
using ValueType = SimpleProgramRunner::ValueType;
}

llvm::Error SimpleProgramRunner::runBlock(mlir::Block *block,
                                          SemanticsContext context) {
    auto &state = interpreter_.state();
    StateType newState;
    state = std::move(newState);

    if (block->empty())
        return llvm::Error::success();

    constexpr WaveId wave = 0;
    // Default to 32 active lanes if the caller does not supply a mask.
    // Honor requested mask; default to four lanes if unspecified for testing.
    std::uint64_t laneMask =
        context.activeMask ? context.activeMask : ((1ull << 4) - 1ull);

    auto &waveCtx = state.waves[wave];
    waveCtx.currentMask = 0;

    DynamicBlockKey entryKey{block, 0};
    auto &dynamicBlock = waveCtx.blocks[entryKey];
    dynamicBlock.block = block;
    dynamicBlock.sequenceId = 0;

    dynamicBlock.expectedMask = laneMask;
    dynamicBlock.activeMask = laneMask;
    dynamicBlock.completedMask = 0;

    waveCtx.currentMask = laneMask;

    std::uint64_t tmpMask = laneMask;
    while (tmpMask) {
        unsigned lane = std::countr_zero(tmpMask);
        tmpMask &= tmpMask - 1;
        auto &laneCtx = waveCtx.lanes[lane];
        laneCtx.values.clear();
        laneCtx.hasReturned = false;
        laneCtx.returnValue.reset();
        laneCtx.phase = decltype(laneCtx.phase)::Running;
        laneCtx.currentBlock = entryKey;

        SemanticsContext laneContext = context;
        laneContext.activeMask = laneMask;
        laneContext.laneId = lane;

        StepType initialStep =
            buildStepForIterator(entryKey, block, block->begin(), laneContext, lane);
        interpreter_.enqueue(wave, entryKey, lane, std::move(initialStep));
    }

    if (llvm::Error err = interpreter_.run())
        return err;

    return llvm::Error::success();
}

SimpleProgramRunner::StepType
SimpleProgramRunner::buildStepForIterator(const DynamicBlockKey &key,
                                          mlir::Block *block,
                                          mlir::Block::iterator it,
                                          SemanticsContext context,
                                          LaneId lane) {
    if (it == block->end())
        return StepType::halt();

    context.laneId = lane;
    if (auto waveIt = interpreter_.state().waves.find(0);
        waveIt != interpreter_.state().waves.end()) {
        auto blockIt = waveIt->second.blocks.find(key);
        if (blockIt != waveIt->second.blocks.end()) {
            context.activeMask = blockIt->second.activeMask;
            context.valueEnv = &blockIt->second.valueEnvs[lane];
        }
    }

    if (auto loopOp = llvm::dyn_cast<simt::dialect::LoopOp>(&*it)) {
        if (!enableLoopDispatch_)
            goto generic_dispatch;
        auto &interpState = interpreter_.state();
        auto waveIt = interpState.waves.find(0);
        if (waveIt == interpState.waves.end())
            return StepType::halt();
        auto &waveCtx = waveIt->second;
        auto parentBlockIt = waveCtx.blocks.find(key);
        if (parentBlockIt == waveCtx.blocks.end())
            return StepType::halt();
        auto &parentBlock = parentBlockIt->second;
        std::uint64_t laneBit = 1ull << lane;
        if ((parentBlock.activeMask & laneBit) == 0)
            return StepType::halt();

        std::uint64_t activeMask = parentBlock.activeMask;
        if (activeMask == 0)
            return StepType::halt();
        std::uint64_t parentExpected =
            parentBlock.expectedMask ? parentBlock.expectedMask : activeMask;

        mlir::Block *prepareBlock = &loopOp.getPrepareRegion().front();
        mlir::Block *bodyBlock = &loopOp.getBodyRegion().front();

        std::uint32_t baseSeq = key.sequenceId + 1;
        DynamicBlockKey prepKey{prepareBlock, baseSeq};
        DynamicBlockKey bodyKey{bodyBlock, baseSeq + 1};

        auto &prepareCtx = waveCtx.blocks[prepKey];
        prepareCtx.block = prepareBlock;
        prepareCtx.sequenceId = prepKey.sequenceId;
        prepareCtx.expectedMask = parentExpected;
        prepareCtx.activeMask = activeMask;
        prepareCtx.completedMask = 0;
        prepareCtx.loopOp = loopOp.getOperation();
        prepareCtx.isLoopPrepare = true;
        prepareCtx.isLoopBody = false;

        auto &bodyCtx = waveCtx.blocks[bodyKey];
        bodyCtx.block = bodyBlock;
        bodyCtx.sequenceId = bodyKey.sequenceId;
        bodyCtx.expectedMask = parentExpected;
        bodyCtx.activeMask = 0;
        bodyCtx.completedMask = 0;
        bodyCtx.loopOp = loopOp.getOperation();
        bodyCtx.isLoopPrepare = false;
        bodyCtx.isLoopBody = true;

        auto nextIt = std::next(it);
        StepType parentCont = StepType::continueWith(
            [this, key, block, nextIt, context, lane]() mutable -> StepType {
                return buildStepForIterator(key, block, nextIt, context, lane);
            });
        parentBlock.continuations[lane] = parentCont;

        MergeStackEntry<ValueType, StepType> entry;
        entry.parent = key;
        entry.pendingChildren.push_back(prepKey);
        entry.childMasks.push_back(activeMask);
        entry.expectedMask = parentExpected;
        entry.completedMask = 0;
        entry.loopFrame.emplace();
        auto &loopFrame = *entry.loopFrame;
        loopFrame.loopOp = loopOp.getOperation();
        loopFrame.prepareKey = prepKey;
        loopFrame.bodyKey = bodyKey;
        loopFrame.nextSequenceId = bodyKey.sequenceId + 1;

        auto inits = loopOp.getInits();
        llvm::ArrayRef<mlir::BlockArgument> prepArgs = prepareBlock->getArguments();
        std::uint64_t mask = activeMask;
        while (mask) {
            unsigned l = std::countr_zero(mask);
            mask &= mask - 1;
            SemanticsContext laneCtx = context;
            laneCtx.laneId = l;
            laneCtx.activeMask = activeMask;
            auto &tuple = loopFrame.carried[l];
            tuple.clear();
            tuple.reserve(inits.size());
            for (mlir::Value init : inits) {
                auto valueOrErr = semantics_.evaluateValue(init, laneCtx);
                if (!valueOrErr) {
                    llvm::consumeError(valueOrErr.takeError());
                    tuple.push_back(ValueType{});
                } else {
                    tuple.push_back(*valueOrErr);
                }
            }

            auto &env = prepareCtx.valueEnvs[l];
            env.clear();
            for (auto indexed : llvm::enumerate(prepArgs)) {
                if (indexed.index() < tuple.size())
                    env[indexed.value()] = tuple[indexed.index()];
            }

            SemanticsContext childContext = context;
            childContext.activeMask = activeMask;
            childContext.laneId = l;
            StepType childStep = buildStepForIterator(
                prepKey, prepareBlock, prepareBlock->begin(), childContext, l);
            interpreter_.enqueue(0, prepKey, l, std::move(childStep));
        }

        parentBlock.activeMask &= ~activeMask;
        waveCtx.mergeStack.push_back(std::move(entry));
        return StepType::halt();
    }

generic_dispatch:
    if (auto ifOp = llvm::dyn_cast<simt::dialect::IfOp>(&*it)) {
        auto &interpState = interpreter_.state();
        auto waveIt = interpState.waves.find(0);
        if (waveIt == interpState.waves.end())
            return StepType::halt();
        auto &waveCtx = waveIt->second;
        DynamicBlockKey parentKey = key;
        auto parentBlockIt = waveCtx.blocks.find(parentKey);
        if (parentBlockIt == waveCtx.blocks.end())
            return StepType::halt();
        auto &parentBlock = parentBlockIt->second;

        std::uint64_t trueMask = 0;
        std::uint64_t falseMask = 0;
        // Evaluate condition per active lane in this block.
        std::uint64_t activeMask = parentBlock.activeMask;
        while (activeMask) {
            unsigned laneId = std::countr_zero(activeMask);
            activeMask &= activeMask - 1;
            SemanticsContext condContext = context;
            condContext.laneId = laneId;
            condContext.activeMask = parentBlock.activeMask;
            auto condOrErr = evaluateBool(ifOp.getCondition(), condContext);
            if (!condOrErr) {
                llvm::consumeError(condOrErr.takeError());
                continue;
            }
            if (*condOrErr)
                trueMask |= (1ull << laneId);
            else
                falseMask |= (1ull << laneId);
        }

        // If there is no else region, lanes falling through rejoin the parent
        // immediately rather than spawning a child block.
        if (ifOp.getElseRegion().empty())
            falseMask = 0;

        auto nextIt = std::next(it);
        // Parent continuation to resume after reconvergence.
        StepType parentCont = StepType::continueWith(
            [this, key, block, nextIt, context, lane]() mutable -> StepType {
                return buildStepForIterator(key, block, nextIt, context, lane);
            });
        parentBlock.continuations[lane] = parentCont;

        // Create child dynamic blocks.
        auto makeChildKey = [&](mlir::Block *b, std::uint32_t iter) {
            return DynamicBlockKey{b, iter};
        };
        std::uint32_t baseSeq = parentKey.sequenceId + 1;

        std::optional<DynamicBlockKey> thenKey;
        std::optional<DynamicBlockKey> elseKey;

        // Use parent.expectedMask to conservatively seed child expected masks.
        std::uint64_t parentExpected = parentBlock.expectedMask ? parentBlock.expectedMask
                                                                 : parentBlock.activeMask;
        std::uint64_t childTrueExpected = parentExpected & trueMask;
        std::uint64_t childFalseExpected = parentExpected & falseMask;

        if (trueMask) {
            DynamicBlockKey key =
                makeChildKey(&ifOp.getThenRegion().front(), baseSeq);
            thenKey = key;
            auto &childBlock = waveCtx.blocks[key];
            childBlock.block = key.block;
            childBlock.sequenceId = key.sequenceId;
            childBlock.expectedMask = childTrueExpected ? childTrueExpected : trueMask;
            childBlock.activeMask = trueMask;
            childBlock.completedMask = 0;
        }

        if (falseMask && !ifOp.getElseRegion().empty()) {
            DynamicBlockKey key =
                makeChildKey(&ifOp.getElseRegion().front(), baseSeq + 1);
            elseKey = key;
            auto &childBlock = waveCtx.blocks[key];
            childBlock.block = key.block;
            childBlock.sequenceId = key.sequenceId;
            childBlock.expectedMask =
                childFalseExpected ? childFalseExpected : falseMask;
            childBlock.activeMask = falseMask;
            childBlock.completedMask = 0;
        }

        // Push merge entry.
        MergeStackEntry<SemValue, StepType> entry;
        entry.parent = parentKey;
        if (thenKey) {
            entry.pendingChildren.push_back(*thenKey);
            entry.childMasks.push_back(trueMask);
        }
        if (elseKey) {
            entry.pendingChildren.push_back(*elseKey);
            entry.childMasks.push_back(falseMask);
        }
        entry.expectedMask = (childTrueExpected ? childTrueExpected : trueMask) |
                             (childFalseExpected ? childFalseExpected : falseMask);
        entry.completedMask = 0;
        if (entry.expectedMask != 0)
            waveCtx.mergeStack.push_back(entry);

        // Remove only dispatched lanes from the parent; others keep executing.
        parentBlock.activeMask &= ~(trueMask | falseMask);

        // Enqueue child continuations per lane.
        if (thenKey) {
            std::uint64_t mask = trueMask;
            while (mask) {
                unsigned l = std::countr_zero(mask);
                mask &= mask - 1;
                SemanticsContext laneCtx = context;
                laneCtx.activeMask = trueMask;
                laneCtx.laneId = l;
                auto *childBlock = const_cast<mlir::Block *>(thenKey->block);
                StepType childStep =
                    buildStepForIterator(*thenKey, childBlock, childBlock->begin(),
                                         laneCtx, l);
                interpreter_.enqueue(0, *thenKey, l, std::move(childStep));
                // This lane will not participate in other children; clear it from
                // their expected masks.
                if (elseKey) {
                    waveCtx.blocks[*elseKey].expectedMask &= ~(1ull << l);
                }
            }
        }
        if (elseKey) {
            std::uint64_t mask = falseMask;
            while (mask) {
                unsigned l = std::countr_zero(mask);
                mask &= mask - 1;
                SemanticsContext laneCtx = context;
                laneCtx.activeMask = falseMask;
                laneCtx.laneId = l;
                auto *childBlock = const_cast<mlir::Block *>(elseKey->block);
                StepType childStep =
                    buildStepForIterator(*elseKey, childBlock, childBlock->begin(),
                                         laneCtx, l);
                interpreter_.enqueue(0, *elseKey, l, std::move(childStep));
                // This lane will not participate in other children; clear it from
                // their expected masks.
                if (thenKey) {
                    waveCtx.blocks[*thenKey].expectedMask &= ~(1ull << l);
                }
            }
        }

        // Stop current lane; scheduler will pick up children.
        return StepType::halt();
    }

    return interpreter_.makeNextOp(0, key, block, it, context, lane);
}

llvm::Expected<bool> SimpleProgramRunner::evaluateBool(mlir::Value value,
                                                       SemanticsContext &context) {
    auto semv = semantics_.evaluateValue(value, context);
    if (!semv)
        return semv.takeError();
    return semv->asBool();
}

llvm::Error SimpleProgramRunner::handleIfOp(simt::dialect::IfOp ifOp,
                                            SemanticsContext context) {
    auto condOrErr = evaluateBool(ifOp.getCondition(), context);
    if (!condOrErr)
        return condOrErr.takeError();

    bool cond = *condOrErr;
    if (cond) {
        return runBlock(&ifOp.getThenRegion().front(), context);
    }

    if (!ifOp.getElseRegion().empty())
        return runBlock(&ifOp.getElseRegion().front(), context);

    return llvm::Error::success();
}

} // namespace simt::semantics
