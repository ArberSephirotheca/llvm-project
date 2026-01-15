#include "simt-step/semantics/SimpleProgram.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <bit>
#include <iterator>
#include <utility>

#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>

#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>

#include <vector>

namespace simt::semantics {

namespace {
using StepType = SimpleProgramRunner::StepType;
using StateType = SimpleProgramRunner::StateType;
using ValueType = SimpleProgramRunner::ValueType;

static SemValue castInitValue(mlir::Type elementType, int64_t value) {
    if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(elementType)) {
        if (intTy.getWidth() <= 32)
            return SemValue::fromInt32(static_cast<int32_t>(value));
        return SemValue::fromInt64(value);
    }
    if (mlir::isa<mlir::IndexType>(elementType))
        return SemValue::fromInt64(value);
    if (mlir::isa<mlir::FloatType>(elementType))
        return SemValue::fromFloat(static_cast<float>(value));
    llvm::report_fatal_error("unsupported buffer element type");
}

static mlir::func::FuncOp resolveEntryFunction(mlir::Operation &op,
                                               llvm::StringRef entryName) {
    if (auto func = mlir::dyn_cast<mlir::func::FuncOp>(op))
        return func;
    if (auto module = mlir::dyn_cast<mlir::ModuleOp>(op))
        return module.lookupSymbol<mlir::func::FuncOp>(entryName);
    if (auto module = op.getParentOfType<mlir::ModuleOp>())
        return module.lookupSymbol<mlir::func::FuncOp>(entryName);
    return {};
}
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

    // Guard against unfinished lanes or pending synchronization.
    for (const auto &waveIt : state.waves) {
        const auto &waveCtx = waveIt.second;
        if (!waveCtx.collectives.empty() || !waveCtx.syncPoints.empty()) {
            return llvm::make_error<llvm::StringError>(
                "runBlock: wave left pending collectives or sync points",
                llvm::inconvertibleErrorCode());
        }
        for (const auto &laneIt : waveCtx.lanes) {
            if (!laneIt.second.hasReturned) {
                return llvm::make_error<llvm::StringError>(
                    "runBlock: lane did not return",
                    llvm::inconvertibleErrorCode());
            }
        }
    }

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

mlir::LogicalResult runOperationToBuffer(
    mlir::Operation &op,
    unsigned bufferArgIndex,
    std::vector<int64_t> &buffer,
    const RunOperationOptions &options,
    llvm::ArrayRef<BufferInitEntry> initEntries) {
    mlir::func::FuncOp func = resolveEntryFunction(op, options.entry);
    if (!func) {
        llvm::errs() << "runOperationToBuffer: missing entry @"
                     << options.entry << "\n";
        return mlir::failure();
    }
    if (bufferArgIndex >= func.getNumArguments()) {
        llvm::errs() << "runOperationToBuffer: buffer arg out of range\n";
        return mlir::failure();
    }
    mlir::Value bufferArg = func.getArgument(bufferArgIndex);
    auto bufferType =
        mlir::dyn_cast<simt::dialect::ResourceType>(bufferArg.getType());
    if (!bufferType) {
        llvm::errs() << "runOperationToBuffer: arg " << bufferArgIndex
                     << " is not a resource\n";
        return mlir::failure();
    }

    SimpleSemantics::clearMemory();
    auto &memMutable = SimpleSemantics::memoryMutable();
    if (options.bufferSize > 0) {
        for (auto arg : func.getArguments()) {
            auto resTy =
                mlir::dyn_cast<simt::dialect::ResourceType>(arg.getType());
            if (!resTy)
                continue;
            auto fillValue = castInitValue(resTy.getElementType(),
                                           options.fillValue);
            for (int64_t i = 0; i < options.bufferSize; ++i)
                memMutable[arg][i] = fillValue;
        }
    }

    for (const auto &entry : initEntries) {
        if (entry.argIndex >= func.getNumArguments()) {
            llvm::errs() << "runOperationToBuffer: init arg out of range\n";
            return mlir::failure();
        }
        mlir::Value arg = func.getArgument(entry.argIndex);
        auto resTy =
            mlir::dyn_cast<simt::dialect::ResourceType>(arg.getType());
        if (!resTy) {
            llvm::errs() << "runOperationToBuffer: init arg is not a resource\n";
            return mlir::failure();
        }
        memMutable[arg][entry.index] =
            castInitValue(resTy.getElementType(), entry.value);
    }

    ExecutionPolicy defaultPolicy;
    SemanticsContext semaCtx;
    unsigned width =
        std::min<unsigned>(64, std::max<unsigned>(1, options.lanes));
    semaCtx.activeMask =
        width >= 64 ? ~0ull : ((1ull << static_cast<std::uint64_t>(width)) - 1ull);
    semaCtx.subgroupWidth = std::max<unsigned>(1, options.subgroupWidth);
    semaCtx.policy = options.policy ? options.policy : &defaultPolicy;

    SimpleProgramRunner runner;
    if (options.trace)
        runner.setTraceSink(options.trace);

    auto &entry = func.getBody().front();
    if (llvm::Error err = runner.runBlock(&entry, semaCtx)) {
        llvm::errs() << "runOperationToBuffer: run failed: "
                     << llvm::toString(std::move(err)) << "\n";
        return mlir::failure();
    }

    buffer.clear();
    int64_t outSize = options.bufferSize;
    const auto &mem = SimpleSemantics::memory();
    auto memIt = mem.find(bufferArg);
    if (outSize <= 0) {
        int64_t maxIndex = -1;
        if (memIt != mem.end()) {
            for (const auto &kv : memIt->second) {
                if (kv.first > maxIndex)
                    maxIndex = kv.first;
            }
        }
        outSize = maxIndex + 1;
    }
    if (outSize <= 0)
        return mlir::success();

    buffer.assign(static_cast<size_t>(outSize), options.fillValue);
    if (memIt != mem.end()) {
        for (const auto &kv : memIt->second) {
            if (kv.first < 0 || kv.first >= outSize)
                continue;
            buffer[static_cast<size_t>(kv.first)] = kv.second.asInt64();
        }
    }

    return mlir::success();
}

} // namespace simt::semantics
