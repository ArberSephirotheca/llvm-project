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
        laneCtx.callStack.clear();

        SemanticsContext laneContext = context;
        laneContext.activeMask = laneMask;
        laneContext.expectedMask = laneMask;
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
            context.expectedMask = blockIt->second.expectedMask;
            context.valueEnv = &blockIt->second.valueEnvs[lane];
        }
    }

    return interpreter_.makeNextOp(0, key, block, it, context, lane);
}

} // namespace simt::semantics
