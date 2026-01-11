#include "simt-step/semantics/SimpleProgram.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <bit>
#include <iterator>
#include <utility>

#include <mlir/IR/Block.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OpDefinition.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>

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

    // Default to 32 active lanes if the caller does not supply a mask.
    // Honor requested mask; default to four lanes if unspecified for testing.
    std::uint64_t globalMask =
        context.activeMask ? context.activeMask : ((1ull << 4) - 1ull);
    std::uint32_t subgroupWidth =
        context.subgroupWidth ? context.subgroupWidth : 8;
    if (subgroupWidth == 0)
        subgroupWidth = 1;
    if (subgroupWidth > 64)
        subgroupWidth = 64;

    llvm::DenseMap<WaveId, std::uint64_t> waveMasks;
    llvm::SmallVector<WaveId, 8> waveIds;
    std::uint64_t tmpMask = globalMask;
    while (tmpMask) {
        unsigned globalLane = std::countr_zero(tmpMask);
        tmpMask &= tmpMask - 1;
        WaveId wave = static_cast<WaveId>(globalLane / subgroupWidth);
        unsigned lane = globalLane % subgroupWidth;
        auto it = waveMasks.find(wave);
        if (it == waveMasks.end()) {
            waveMasks[wave] = 0;
            waveIds.push_back(wave);
            it = waveMasks.find(wave);
        }
        it->second |= (1ull << lane);
    }
    llvm::sort(waveIds);

    DynamicBlockKey entryKey{block, 0};
    for (WaveId wave : waveIds) {
        std::uint64_t laneMask = waveMasks[wave];
        auto &waveCtx = state.waves[wave];
        waveCtx.waveId = wave;
        waveCtx.subgroupWidth = subgroupWidth;
        waveCtx.policy = context.policy;
        waveCtx.currentMask = laneMask;

        auto &dynamicBlock = waveCtx.blocks[entryKey];
        dynamicBlock.block = block;
        dynamicBlock.sequenceId = 0;

        dynamicBlock.expectedMask = laneMask;
        dynamicBlock.activeMask = laneMask;
        dynamicBlock.completedMask = 0;

        std::uint64_t localMask = laneMask;
        while (localMask) {
            unsigned lane = std::countr_zero(localMask);
            localMask &= localMask - 1;
            auto &laneCtx = waveCtx.lanes[lane];
            laneCtx.values.clear();
            laneCtx.hasReturned = false;
            laneCtx.returnValue.reset();
            laneCtx.phase = decltype(laneCtx.phase)::Running;
            laneCtx.currentBlock = entryKey;
            laneCtx.callStack.clear();

            SemanticsContext laneContext = context;
            laneContext.activeMask = laneMask;
            laneContext.expectedMask = laneMask;
            laneContext.laneId = lane;
            laneContext.waveId = wave;
            laneContext.subgroupWidth = subgroupWidth;

            StepType initialStep = buildStepForIterator(
                wave, entryKey, block, block->begin(), laneContext, lane);
            interpreter_.enqueue(wave, entryKey, lane, std::move(initialStep));
        }
    }

    if (llvm::Error err = interpreter_.run())
        return err;

    return llvm::Error::success();
}

SimpleProgramRunner::StepType
SimpleProgramRunner::buildStepForIterator(WaveId wave,
                                          const DynamicBlockKey &key,
                                          mlir::Block *block,
                                          mlir::Block::iterator it,
                                          SemanticsContext context,
                                          LaneId lane) {
    if (it == block->end())
        return StepType::halt();

    context.laneId = lane;
    context.waveId = wave;
    if (auto waveIt = interpreter_.state().waves.find(wave);
        waveIt != interpreter_.state().waves.end()) {
        auto blockIt = waveIt->second.blocks.find(key);
        if (blockIt != waveIt->second.blocks.end()) {
            context.activeMask = blockIt->second.activeMask;
            context.expectedMask = blockIt->second.expectedMask;
            context.valueEnv = &blockIt->second.valueEnvs[lane];
        }
    }

    return interpreter_.makeNextOp(wave, key, block, it, context, lane);
}

} // namespace simt::semantics
