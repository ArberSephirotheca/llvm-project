#pragma once

#include "simt-step/semantics/Effects.h"
#include "simt-step/semantics/ExecutionState.h"
#include "simt-step/semantics/SemanticsContext.h"
#include "simt-step/semantics/Trace.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <functional>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <queue>
#include <string>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/SymbolTable.h>

namespace mlir {
class Operation;
} // namespace mlir

namespace simt::semantics {

inline bool EnableCPSDebugLogs = false;

inline std::string formatMaskBits(std::uint64_t mask, unsigned width) {
    std::string s;
    s.reserve(width + 2);
    s.append("0b");
    for (int i = static_cast<int>(width) - 1; i >= 0; --i) {
        s.push_back((mask & (1ull << i)) ? '1' : '0');
    }
    return s;
}

inline const char *blockKindLabel(DynamicBlockKind kind) {
    switch (kind) {
    case DynamicBlockKind::Plain:
        return "plain";
    case DynamicBlockKind::IfThen:
        return "if.then";
    case DynamicBlockKind::IfElse:
        return "if.else";
    case DynamicBlockKind::SwitchCase:
        return "switch.case";
    case DynamicBlockKind::SwitchDefault:
        return "switch.default";
    case DynamicBlockKind::LoopPrepare:
        return "loop.prepare";
    case DynamicBlockKind::LoopBody:
        return "loop.body";
    }
    return "unknown";
}

template <typename ValueT, typename StepT>
inline void logMergeStackState(const WaveContext<ValueT, StepT> &waveCtx) {
    auto fmt = [&](std::uint64_t m) { return formatMaskBits(m, 32); };
    llvm::errs() << "[CPS] MergeStack size=" << waveCtx.mergeStack.size() << "\n";
    for (std::size_t idx = 0; idx < waveCtx.mergeStack.size(); ++idx) {
        const auto &entry = waveCtx.mergeStack[idx];
        llvm::errs() << "  [" << idx << "] parent=" << entry.parent.block
                     << " seq=" << entry.parent.sequenceId
                     << " expected=0b" << fmt(entry.expectedMask)
                     << " completed=0b" << fmt(entry.completedMask)
                     << " children=" << entry.pendingChildren.size()
                     << (entry.loopFrame ? " (loop)" : "") << "\n";
        for (std::size_t ci = 0; ci < entry.pendingChildren.size(); ++ci) {
            llvm::errs() << "      child[" << ci << "]=" << entry.pendingChildren[ci].block
                         << " seq=" << entry.pendingChildren[ci].sequenceId
                         << " mask=0b" << fmt(entry.childMasks[ci]) << "\n";
        }
    }
}

/// Continuation-Passing Style control primitive returned by interpreter steps.
template <typename ValueT>
class Step {
public:
    struct Halt {};

    struct Continue {
        std::function<Step()> next;
    };

    struct Produce {
        ValueT value;
    };

    struct Suspend {
        Effect effect;
        std::function<Step()> resume;
    };

    using State = std::variant<Halt, Continue, Produce, Suspend>;

    Step() : state_(Halt{}) {}

    explicit Step(Halt halt) : state_(std::move(halt)) {}
    explicit Step(Continue cont) : state_(std::move(cont)) {}
    explicit Step(Produce prod) : state_(std::move(prod)) {}
    explicit Step(Suspend susp) : state_(std::move(susp)) {}

    static Step halt() { return Step(Halt{}); }

    static Step continueWith(std::function<Step()> next) {
        return Step(Continue{std::move(next)});
    }

    static Step produce(ValueT value) {
        return Step(Produce{std::move(value)});
    }

    static Step suspend(Effect effect, std::function<Step()> resume) {
        return Step(Suspend{std::move(effect), std::move(resume)});
    }

    bool isHalt() const { return std::holds_alternative<Halt>(state_); }
    bool isContinue() const { return std::holds_alternative<Continue>(state_); }
    bool isProduce() const { return std::holds_alternative<Produce>(state_); }
    bool isSuspend() const { return std::holds_alternative<Suspend>(state_); }

    const State &state() const { return state_; }
    State takeState() && { return std::move(state_); }

private:
    State state_;
};

/// Minimal tagless interface wrapper. Semantic implementations are expected to
/// provide the aliases below.
template <typename Impl>
struct SimtStepSemanticsAdaptor {
    using ValueType = typename Impl::ValueType;
    using StepType = Step<ValueType>;

    StepType eval(Impl &impl, mlir::Operation *op, SemanticsContext &context) {
        if constexpr (requires { impl.evalOperation(op, context); }) {
            return impl.evalOperation(op, context);
        } else {
            return impl.eval(op, context);
        }
    }
};

/// High-level interpreter shell that delegates to a semantics implementation
/// and exposes a CPS stepping API.
template <typename SemanticsT>
class SimtStepExecutor {
public:
    using ValueType = typename SemanticsT::ValueType;
    using StepType = Step<ValueType>;

    explicit SimtStepExecutor(SemanticsT semantics)
        : semantics_(std::move(semantics)) {}

    StepType step(mlir::Operation *op, SemanticsContext &context) {
        return adaptor_.eval(semantics_, op, context);
    }

    SemanticsT &semantics() { return semantics_; }
    const SemanticsT &semantics() const { return semantics_; }

private:
    SemanticsT semantics_;
    SimtStepSemanticsAdaptor<SemanticsT> adaptor_;
};

/// Template interpreter harness that drives CPS-style semantics.
template <typename SemanticsT>
class CPSInterpreter {
public:
    using ValueType = typename SemanticsT::ValueType;
    using StepType = Step<ValueType>;
    using StateType = InterpreterState<ValueType, StepType>;

    explicit CPSInterpreter(SemanticsT semantics)
        : semantics_(std::move(semantics)) {}

    void setTraceSink(TraceSink *sink) { traceSink_ = sink; }

    StateType &state() { return state_; }
    const StateType &state() const { return state_; }

    /// Enqueue an initial continuation for the given wave/block/lane triple.
    void enqueue(WaveId wave, const DynamicBlockKey &block, LaneId lane,
                 StepType step) {
        ensureWaveBlock(wave, block, lane);
        if (EnableCPSDebugLogs) {
            llvm::errs() << "[CPS] enqueue lane=" << lane
                         << " block=" << block.block
                         << " seq=" << block.sequenceId << "\n";
            dumpReadyQueue();
            dumpContinuations();
        }
        state_.readyQueue.push(
            ReadyContinuation<ValueType, StepType>{wave, block, lane, std::move(step)});
    }

    void dumpReadyQueue() const {
        if (!EnableCPSDebugLogs)
            return;
        llvm::errs() << "[CPS] ReadyQueue size=" << state_.readyQueue.size() << "\n";
        std::queue<ReadyContinuation<ValueType, StepType>> tmp = state_.readyQueue;
        std::size_t idx = 0;
        while (!tmp.empty()) {
            const auto &item = tmp.front();
            llvm::errs() << "  [" << idx++ << "] wave=" << item.wave
                         << " block=" << item.block.block
                         << " seq=" << item.block.sequenceId
                         << " lane=" << item.lane << "\n";
            tmp.pop();
        }
    }

    void dumpContinuations() const {
        if (!EnableCPSDebugLogs)
            return;
        for (const auto &wavePair : state_.waves) {
            WaveId w = wavePair.first;
            const auto &waveCtx = wavePair.second;
            llvm::errs() << "[CPS] Continuations for wave " << w << "\n";
            for (const auto &blockPair : waveCtx.blocks) {
                const auto &key = blockPair.first;
                const auto &blk = blockPair.second;
                if (blk.continuations.empty())
                    continue;
                llvm::errs() << "  block=" << key.block
                             << " seq=" << key.sequenceId
                             << " lanes:";
                for (const auto &c : blk.continuations)
                    llvm::errs() << " " << c.first;
                llvm::errs() << "\n";
            }
        }
    }

    /// Execute a single ready continuation if available.
    llvm::Error runOne() {
        if (state_.readyQueue.empty())
            return llvm::Error::success();
        auto item = std::move(state_.readyQueue.front());
        state_.readyQueue.pop();
        return processReady(std::move(item));
    }

    /// Run until there are no ready continuations left.
    llvm::Error run() {
        while (!state_.readyQueue.empty()) {
            if (llvm::Error err = runOne())
                return err;
        }
        return llvm::Error::success();
    }

    /// Build a continuation that executes the operation at `it` for the given
    /// wave/block/lane and chains to the next iterator.
    StepType makeNextOp(WaveId wave,
                        const DynamicBlockKey &key,
                        mlir::Block *block,
                        mlir::Block::iterator it,
                        SemanticsContext context,
                        LaneId lane) {
        // Defer execution: return a continuation that will run this op when invoked.
        return StepType::continueWith(
            [this, wave, key, block, it, context, lane]() mutable -> StepType {
                if (it == block->end())
                    return StepType::halt();

                SemanticsContext ctx = context;
                ctx.laneId = lane;
                WaveContext<ValueType, StepType> *waveCtx = nullptr;
                DynamicBlock<ValueType, StepType> *blockCtx = nullptr;
                if (auto waveIt = state_.waves.find(wave); waveIt != state_.waves.end()) {
                    waveCtx = &waveIt->second;
                    if (!waveCtx->policy && ctx.policy)
                        waveCtx->policy = ctx.policy;
                    if (!ctx.policy && waveCtx->policy)
                        ctx.policy = waveCtx->policy;
                    if (auto *blk = getBlock(*waveCtx, key)) {
                        blockCtx = blk;
                        ctx.activeMask = blk->activeMask;
                        ctx.expectedMask =
                            blk->expectedMask ? blk->expectedMask : blk->activeMask;
                        auto envIt = blk->valueEnvs.find(lane);
                        if (envIt != blk->valueEnvs.end())
                            ctx.valueEnv = &envIt->second;
                    }
                }
                const std::uint32_t blockSeq = key.sequenceId;
                const void *blockPtr = key.block;
                const char *blockKind =
                    blockCtx ? blockKindLabel(blockCtx->kind) : "unknown";
                std::optional<std::uint32_t> blockIter =
                    blockCtx ? blockCtx->loopIteration : std::nullopt;

                if (auto handled =
                        handleLoopSplit(wave, key, block, it, ctx, lane))
                    return *handled;

                if (auto handled =
                        handleSwitchSplit(wave, key, block, it, ctx, lane))
                    return *handled;

                if (auto handled = handleLoopPrepareTerminator(
                        wave, key, block, it, ctx, lane))
                    return *handled;

                if (auto handled =
                        handleLoopYield(wave, key, block, it, ctx, lane))
                    return *handled;

                if (auto handled = handleLoopContinue(
                        wave, key, block, it, ctx, lane))
                    return *handled;

                if (auto handled =
                        handleSwitchYield(wave, key, block, it, ctx, lane))
                    return *handled;

                if (auto handled =
                        handleBreak(wave, key, block, it, ctx, lane))
                    return *handled;

                if (auto handled =
                        handleIfYield(wave, key, block, it, ctx, lane))
                    return *handled;

                if (auto handled =
                        handleIfSplit(wave, key, block, it, ctx, lane))
                    return *handled;

                if (auto handled =
                        handleCallOp(wave, key, block, it, ctx, lane))
                    return *handled;

                // Mark return as terminal for this lane so we don't resume parents.
                if (auto retOp = llvm::dyn_cast<mlir::func::ReturnOp>(&*it)) {
                    auto waveIt = state_.waves.find(wave);
                    if (waveIt != state_.waves.end()) {
                        auto &waveCtx = waveIt->second;
                        auto &laneCtx = waveCtx.lanes[lane];
                        if (!laneCtx.callStack.empty()) {
                            auto frame = std::move(laneCtx.callStack.back());
                            laneCtx.callStack.pop_back();
                            if (auto *blockCtx = getBlock(waveCtx, key))
                                blockCtx->activeMask &= ~(1ull << lane);
                            if (frame.results.size() != retOp.getNumOperands())
                                llvm::report_fatal_error(
                                    "call return value count mismatch");
                            auto *callerBlockCtx = getBlock(waveCtx, frame.callerKey);
                            if (!callerBlockCtx)
                                llvm::report_fatal_error(
                                    "call return missing caller block");
                            if (!frame.results.empty()) {
                                auto valOrErr =
                                    evaluateValue(waveCtx, key, retOp.getOperand(0),
                                                  lane, ctx.activeMask, ctx.expectedMask);
                                if (!valOrErr)
                                    llvm::report_fatal_error(
                                        "call return value evaluation failed");
                                callerBlockCtx->valueEnvs[lane][frame.results[0]] =
                                    *valOrErr;
                            }
                            callerBlockCtx->activeMask |= (1ull << lane);
                            laneCtx.phase =
                                LaneContext<ValueType, StepType>::Phase::Running;
                            laneCtx.hasReturned = false;
                            laneCtx.returnValue.reset();
                            laneCtx.currentBlock = frame.callerKey;
                            SemanticsContext resumeCtx;
                            resumeCtx.laneId = lane;
                            resumeCtx.policy = ctx.policy;
                            resumeCtx.overrideMode.reset();
                            return StepType::continueWith(
                                [this, wave, frame = std::move(frame), lane,
                                 resumeCtx]() mutable -> StepType {
                                    return makeNextOp(wave, frame.callerKey,
                                                      frame.callerBlock,
                                                      frame.resumeIt, resumeCtx,
                                                      lane);
                                });
                        }
                        laneCtx.phase =
                            LaneContext<ValueType, StepType>::Phase::Completed;
                        laneCtx.hasReturned = true;
                    }
                    if (traceSink_) {
                        std::uint64_t expectedMask =
                            ctx.expectedMask ? ctx.expectedMask : ctx.activeMask;
                        traceSink_->onReturn(wave, lane,
                                             retOp.getNumOperands() > 0,
                                             ctx.activeMask, expectedMask,
                                             blockSeq, blockPtr, blockKind, blockIter);
                    }
                }

                if (EnableCPSDebugLogs) {
                    llvm::errs() << "[CPS] eval lane=" << lane
                                 << " block=" << block
                                 << " seq=" << key.sequenceId
                                 << " op=" << it->getName().getStringRef() << "\n";
                }
                if (traceSink_) {
                    std::uint64_t expectedMask =
                        ctx.expectedMask ? ctx.expectedMask : ctx.activeMask;
                    traceSink_->onStepBegin(
                        wave, lane, it->getName().getStringRef().str(),
                        ctx.activeMask, expectedMask,
                        blockSeq, blockPtr, blockKind, blockIter);
                }

                StepType current = adaptor_.eval(semantics_, &*it, ctx);
                mlir::Block::iterator nextIt = std::next(it);
                bool isTerminator = it->hasTrait<mlir::OpTrait::IsTerminator>();
                const bool hasNext = nextIt != block->end();

                while (true) {
                    typename StepType::State stateVariant = std::move(current).takeState();

                    if (auto *cont =
                            std::get_if<typename StepType::Continue>(&stateVariant)) {
                        if (!cont->next)
                            return StepType::halt();
                        current = cont->next();
                        continue;
                    }

                    if (auto *suspend =
                            std::get_if<typename StepType::Suspend>(&stateVariant)) {
                        Effect effect = std::move(suspend->effect);
                        auto resume = std::move(suspend->resume);
                        if (waveCtx && executionModeForOp(&*it, ctx) ==
                                           ExecutionMode::Collective) {
                            if (isMemoryOp(&*it)) {
                                auto *collective =
                                    effect.template get_if<CollectiveEffect>();
                                if (!collective)
                                    llvm::report_fatal_error(
                                        "collective memory op: missing collective effect");
                                std::uint32_t token =
                                    collective->token.value_or(
                                        collective->operation);
                                waveCtx->collectiveTokenToOp[token] = &*it;
                                auto &syncPoint = waveCtx->collectives[token];
                                auto idxOrErr =
                                    evaluateValue(*waveCtx, key, it->getOperand(1),
                                                  lane, ctx.activeMask,
                                                  ctx.expectedMask);
                                if (!idxOrErr) {
                                    llvm::consumeError(idxOrErr.takeError());
                                    llvm::report_fatal_error(
                                        "collective memory op: failed to evaluate index");
                                }
                                syncPoint.memoryIndices[lane] = std::move(*idxOrErr);

                                if (isBufferStore(&*it)) {
                                    auto valOrErr =
                                        evaluateValue(*waveCtx, key,
                                                      it->getOperand(2), lane,
                                                      ctx.activeMask,
                                                      ctx.expectedMask);
                                    if (!valOrErr) {
                                        llvm::consumeError(valOrErr.takeError());
                                        llvm::report_fatal_error(
                                            "collective memory op: failed to evaluate value");
                                    }
                                    syncPoint.memoryValues[lane] =
                                        std::move(*valOrErr);
                                    resume = []() mutable -> StepType {
                                        return StepType::halt();
                                    };
                                } else {
                                    resume = [this, wave, token, lane]()
                                                 mutable -> StepType {
                                        auto waveIt = state_.waves.find(wave);
                                        if (waveIt == state_.waves.end())
                                            llvm::report_fatal_error(
                                                "collective memory resume: missing wave context");
                                        auto &waveCtx = waveIt->second;
                                        auto syncIt = waveCtx.collectives.find(token);
                                        if (syncIt == waveCtx.collectives.end())
                                            llvm::report_fatal_error(
                                                "collective memory resume: missing sync point");
                                        auto &syncPoint = syncIt->second;
                                        auto resultIt = syncPoint.results.find(lane);
                                        if (resultIt == syncPoint.results.end())
                                            llvm::report_fatal_error(
                                                "collective memory resume: missing lane result");
                                        ValueType result = resultIt->second;
                                        syncPoint.results.erase(resultIt);
                                        syncPoint.continuations.erase(lane);
                                        if (syncPoint.results.empty()) {
                                            waveCtx.collectives.erase(syncIt);
                                            waveCtx.collectiveTokenToOp.erase(token);
                                        }
                                        return StepType::produce(std::move(result));
                                    };
                                }
                            }
                        }
                        if (waveCtx && isWaveOp(&*it) &&
                            executionModeForOp(&*it, ctx) == ExecutionMode::Collective) {
                            if (!blockCtx)
                                llvm::report_fatal_error(
                                    "collective wave op: missing block context");
                            auto *collective =
                                effect.template get_if<CollectiveEffect>();
                            if (!collective)
                                llvm::report_fatal_error(
                                    "collective wave op: missing collective effect");
                            if (it->getNumOperands() != 1)
                                llvm::report_fatal_error(
                                    "collective wave op: expected one operand");
                            auto predOrErr =
                                evaluateValue(*waveCtx, key, it->getOperand(0),
                                              lane, ctx.activeMask,
                                              ctx.expectedMask);
                            if (!predOrErr) {
                                llvm::consumeError(predOrErr.takeError());
                                llvm::report_fatal_error(
                                    "collective wave op: failed to evaluate operand");
                            }
                            std::uint32_t token =
                                collective->token.value_or(collective->operation);
                            waveCtx->collectiveTokenToOp[token] = &*it;
                            auto &syncPoint = waveCtx->collectives[token];
                            syncPoint.operands[lane] = std::move(*predOrErr);
                            resume = [this, wave, token, lane]() mutable -> StepType {
                                auto waveIt = state_.waves.find(wave);
                                if (waveIt == state_.waves.end())
                                    llvm::report_fatal_error(
                                        "collective wave resume: missing wave context");
                                auto &waveCtx = waveIt->second;
                                auto syncIt = waveCtx.collectives.find(token);
                                if (syncIt == waveCtx.collectives.end())
                                    llvm::report_fatal_error(
                                        "collective wave resume: missing sync point");
                                auto &syncPoint = syncIt->second;
                                auto resultIt = syncPoint.results.find(lane);
                                if (resultIt == syncPoint.results.end())
                                    llvm::report_fatal_error(
                                        "collective wave resume: missing lane result");
                                ValueType result = resultIt->second;
                                syncPoint.results.erase(resultIt);
                                syncPoint.continuations.erase(lane);
                                if (syncPoint.results.empty()) {
                                    waveCtx.collectives.erase(syncIt);
                                    waveCtx.collectiveTokenToOp.erase(token);
                                }
                                return StepType::produce(std::move(result));
                            };
                        }
                        if (traceSink_) {
                            std::uint64_t expectedMask =
                                ctx.expectedMask ? ctx.expectedMask : ctx.activeMask;
                            traceSink_->onSuspend(
                                wave, lane, effect, ctx.activeMask, expectedMask,
                                blockSeq, blockPtr, blockKind, blockIter);
                        }

                        std::function<StepType(StepType)> handleResumed;
                        handleResumed = [this, wave, key, block, nextIt, ctx, lane,
                                         isTerminator, hasNext, waveCtx, blockCtx,
                                         blockSeq, blockPtr, blockKind, blockIter,
                                         op = &*it, &handleResumed](StepType current)
                                         mutable -> StepType {
                            if (traceSink_) {
                                std::uint64_t expectedMask =
                                    ctx.expectedMask ? ctx.expectedMask : ctx.activeMask;
                                traceSink_->onResume(
                                    wave, lane, ctx.activeMask, expectedMask,
                                    blockSeq, blockPtr, blockKind, blockIter);
                            }
                            while (true) {
                                auto resumedState = std::move(current).takeState();
                                if (auto *cont =
                                        std::get_if<typename StepType::Continue>(
                                            &resumedState)) {
                                    if (!cont->next)
                                        return StepType::halt();
                                    current = cont->next();
                                    continue;
                                }
                                if (auto *susp =
                                        std::get_if<typename StepType::Suspend>(
                                            &resumedState)) {
                                    Effect eff = std::move(susp->effect);
                                    auto innerResume = std::move(susp->resume);
                                    auto chained = [innerResume = std::move(innerResume),
                                                    &handleResumed]() mutable -> StepType {
                                        return handleResumed(innerResume());
                                    };
                                    return StepType::suspend(std::move(eff), std::move(chained));
                                }
                                if (auto *prod =
                                        std::get_if<typename StepType::Produce>(
                                            &resumedState)) {
                                    if (blockCtx && op->getNumResults() == 1)
                                        blockCtx->valueEnvs[lane][op->getResult(0)] =
                                            prod->value;
                                    if (!isTerminator && hasNext) {
                                        return StepType::continueWith(
                                            [this, wave, key, block, nextIt, ctx, lane]()
                                            mutable -> StepType {
                                                return makeNextOp(wave, key, block, nextIt,
                                                                  ctx, lane);
                                            });
                                    }
                                    if (isTerminator && waveCtx)
                                        handleReconvergence(wave, *waveCtx, key, lane);
                                    return StepType::produce(std::move(prod->value));
                                }
                                if (std::holds_alternative<typename StepType::Halt>(
                                        resumedState)) {
                                    if (!isTerminator && hasNext) {
                                        return StepType::continueWith(
                                            [this, wave, key, block, nextIt, ctx, lane]()
                                            mutable -> StepType {
                                                return makeNextOp(wave, key, block, nextIt,
                                                                  ctx, lane);
                                            });
                                    }
                                    if (isTerminator && waveCtx)
                                        handleReconvergence(wave, *waveCtx, key, lane);
                                    return StepType::halt();
                                }
                                if (!isTerminator && hasNext) {
                                    return StepType::continueWith(
                                        [this, wave, key, block, nextIt, ctx, lane]()
                                        mutable -> StepType {
                                            return makeNextOp(wave, key, block, nextIt,
                                                              ctx, lane);
                                        });
                                }
                                return StepType::halt();
                            }
                        };

                        return StepType::suspend(
                            std::move(effect),
                            [handleResumed = std::move(handleResumed),
                             resume = std::move(resume)]() mutable -> StepType {
                                return handleResumed(resume());
                            });
                    }

                    if (auto *prod =
                            std::get_if<typename StepType::Produce>(&stateVariant)) {
                        if (blockCtx && it->getNumResults() == 1)
                            blockCtx->valueEnvs[lane][it->getResult(0)] = prod->value;
                        if (!isTerminator && hasNext) {
                            return StepType::continueWith(
                                [this, wave, key, block, nextIt, ctx, lane]() mutable
                                -> StepType {
                                    return makeNextOp(wave, key, block, nextIt, ctx, lane);
                                });
                        }
                        if (isTerminator && waveCtx)
                            handleReconvergence(wave, *waveCtx, key, lane);
                        return StepType::produce(std::move(prod->value));
                    }

                    if (std::holds_alternative<typename StepType::Halt>(stateVariant)) {
                        if (!isTerminator && hasNext) {
                            return StepType::continueWith(
                                [this, wave, key, block, nextIt, ctx, lane]() mutable
                                -> StepType {
                                    return makeNextOp(wave, key, block, nextIt, ctx, lane);
                                });
                        }
                        if (isTerminator && waveCtx)
                            handleReconvergence(wave, *waveCtx, key, lane);
                        return StepType::halt();
                    }

                    if (!isTerminator && hasNext) {
                        return StepType::continueWith(
                            [this, wave, key, block, nextIt, ctx, lane]() mutable -> StepType {
                                return makeNextOp(wave, key, block, nextIt, ctx, lane);
                            });
                    }

                    return StepType::halt();
                }
            });
    }

private:
    using IfDecisionMap = llvm::DenseMap<LaneId, bool>;
    using SwitchDecisionMap = llvm::DenseMap<LaneId, std::int64_t>;

    static bool isControlFlowOp(mlir::Operation *op) {
        return llvm::isa<simt::dialect::IfOp, simt::dialect::LoopOp,
                         simt::dialect::SwitchOp>(op);
    }

    static bool isMemoryOp(mlir::Operation *op) {
        auto name = op->getName().getStringRef();
        return name == "simt_step.buffer.load" || name == "simt_step.buffer.store";
    }

    static bool isBufferLoad(mlir::Operation *op) {
        return op->getName().getStringRef() == "simt_step.buffer.load";
    }

    static bool isBufferStore(mlir::Operation *op) {
        return op->getName().getStringRef() == "simt_step.buffer.store";
    }

    static bool isWaveOp(mlir::Operation *op) {
        return op->hasTrait<simt::dialect::SimtWave>();
    }

    static bool valueToBool(const ValueType &value) {
        if constexpr (std::is_same_v<ValueType, SemValue>)
            return value.asBool();
        llvm::report_fatal_error("collective wave op: unsupported value type");
        return false;
    }

    static ValueType makeInt32Value(std::int32_t value) {
        if constexpr (std::is_same_v<ValueType, SemValue>)
            return SemValue::fromInt32(value);
        llvm::report_fatal_error("collective wave op: unsupported value type");
        return ValueType();
    }

    static auto &memoryMutable() {
        if constexpr (requires { SemanticsT::memoryMutable(); }) {
            return SemanticsT::memoryMutable();
        } else {
            llvm::report_fatal_error(
                "collective memory op: semantics does not expose memory");
        }
    }

    void computeWaveCollectiveResults(
        const mlir::Operation *op,
        CollectiveSyncPoint<ValueType, StepType> &syncPoint) {
        if (syncPoint.expectedMask == 0)
            llvm::report_fatal_error(
                "collective wave op: missing expected mask");
        if (!llvm::isa<simt::dialect::WaveCountBitsOp>(op))
            llvm::report_fatal_error(
                "collective wave op: unsupported operation");
        std::uint64_t predMask = 0;
        std::uint64_t mask = syncPoint.expectedMask;
        while (mask) {
            unsigned lane = std::countr_zero(mask);
            mask &= mask - 1;
            auto operandIt = syncPoint.operands.find(lane);
            if (operandIt == syncPoint.operands.end())
                llvm::report_fatal_error(
                    "collective wave op: missing operand value");
            if (valueToBool(operandIt->second))
                predMask |= (1ull << lane);
        }
        std::int32_t count =
            static_cast<std::int32_t>(std::popcount(predMask));
        mask = syncPoint.expectedMask;
        while (mask) {
            unsigned lane = std::countr_zero(mask);
            mask &= mask - 1;
            syncPoint.results[lane] = makeInt32Value(count);
        }
        syncPoint.operands.clear();
    }

    bool computeMemoryCollectiveResults(
        const mlir::Operation *op,
        CollectiveSyncPoint<ValueType, StepType> &syncPoint) {
        if (!isMemoryOp(const_cast<mlir::Operation *>(op)))
            llvm::report_fatal_error(
                "collective memory op: unsupported operation");
        if (syncPoint.expectedMask == 0)
            llvm::report_fatal_error(
                "collective memory op: missing expected mask");

        auto &mem = memoryMutable();
        auto *mutableOp = const_cast<mlir::Operation *>(op);
        mlir::Value res = mutableOp->getOperand(0);
        if (isBufferLoad(const_cast<mlir::Operation *>(op))) {
            std::uint64_t mask = syncPoint.expectedMask;
            while (mask) {
                unsigned lane = std::countr_zero(mask);
                mask &= mask - 1;
                auto idxIt = syncPoint.memoryIndices.find(lane);
                if (idxIt == syncPoint.memoryIndices.end())
                    llvm::report_fatal_error(
                        "collective memory load: missing index");
                int64_t idx = idxIt->second.asInt64();
                auto resIt = mem.find(res);
                if (resIt == mem.end())
                    llvm::report_fatal_error("buffer.load: missing value at index");
                auto valIt = resIt->second.find(idx);
                if (valIt == resIt->second.end())
                    llvm::report_fatal_error("buffer.load: missing value at index");
                syncPoint.results[lane] = valIt->second;
            }
            return true;
        }

        if (isBufferStore(const_cast<mlir::Operation *>(op))) {
            std::uint64_t mask = syncPoint.expectedMask;
            // Apply stores in lane order to keep conflicts deterministic.
            while (mask) {
                unsigned lane = std::countr_zero(mask);
                mask &= mask - 1;
                auto idxIt = syncPoint.memoryIndices.find(lane);
                if (idxIt == syncPoint.memoryIndices.end())
                    llvm::report_fatal_error(
                        "collective memory store: missing index");
                auto valIt = syncPoint.memoryValues.find(lane);
                if (valIt == syncPoint.memoryValues.end())
                    llvm::report_fatal_error(
                        "collective memory store: missing value");
                int64_t idx = idxIt->second.asInt64();
                mem[res][idx] = valIt->second;
            }
            return false;
        }

        llvm::report_fatal_error("collective memory op: unsupported operation");
    }

    ExecutionMode executionModeForOp(mlir::Operation *op,
                                     const SemanticsContext &context) const {
        if (context.overrideMode)
            return *context.overrideMode;
        if (context.policy) {
            auto name = op->getName().getStringRef();
            auto it = context.policy->overrides.find(name);
            if (it != context.policy->overrides.end())
                return it->second;
            if (isControlFlowOp(op))
                return context.policy->controlFlow;
            if (isWaveOp(op))
                return context.policy->waveOps;
            if (isMemoryOp(op))
                return context.policy->memoryOps;
        }
        if (isWaveOp(op))
            return ExecutionMode::Collective;
        return ExecutionMode::Independent;
    }

    std::optional<StepType> gateControlFlowOp(WaveId wave,
                                              const DynamicBlockKey &key,
                                              mlir::Block *block,
                                              mlir::Block::iterator it,
                                              SemanticsContext context,
                                              LaneId lane) {
        ExecutionMode mode = executionModeForOp(&*it, context);
        if (mode == ExecutionMode::Collective)
            return gateControlFlow(wave, key, block, it, context, lane);
        if (mode == ExecutionMode::Synchronous)
            return gateSynchronousOp(wave, key, block, it, context, lane);
        return std::nullopt;
    }

    std::optional<StepType> gateControlFlow(WaveId wave,
                                            const DynamicBlockKey &key,
                                            mlir::Block *block,
                                            mlir::Block::iterator it,
                                            SemanticsContext context,
                                            LaneId lane) {
        auto waveIt = state_.waves.find(wave);
        if (waveIt == state_.waves.end())
            llvm::report_fatal_error("collective-cf: missing wave context");
        auto &waveCtx = waveIt->second;
        auto *blockCtx = getBlock(waveCtx, key);
        if (!blockCtx)
            llvm::report_fatal_error("collective-cf: missing block context");

        auto *op = &*it;
        std::uint64_t laneBit = 1ull << lane;
        auto readyIt = blockCtx->controlReadyMask.find(op);
        if (readyIt != blockCtx->controlReadyMask.end()) {
            if (readyIt->second & laneBit) {
                readyIt->second &= ~laneBit;
                if (readyIt->second == 0)
                    blockCtx->controlReadyMask.erase(readyIt);
                return std::nullopt;
            }
        }

        std::uint64_t expected =
            context.expectedMask ? context.expectedMask : context.activeMask;
        if (expected == 0)
            expected = laneBit;

        std::uint32_t token = 0;
        auto tokenIt = blockCtx->controlTokens.find(op);
        if (tokenIt == blockCtx->controlTokens.end()) {
            token = waveCtx.nextControlToken++;
            blockCtx->controlTokens[op] = token;
        } else {
            token = tokenIt->second;
        }
        waveCtx.controlTokenToOp[token] = op;

        CollectiveEffect effect;
        effect.operation = 0;
        effect.activeMask = expected;
        effect.token = token;

        if (traceSink_) {
            traceSink_->onSuspend(
                wave, lane, Effect(effect), context.activeMask, expected,
                key.sequenceId, key.block, blockKindLabel(blockCtx->kind),
                blockCtx->loopIteration);
        }

        return StepType::suspend(
            Effect(std::move(effect)),
            [this, wave, key, block, it, lane, context]() mutable -> StepType {
                SemanticsContext resumeCtx;
                resumeCtx.laneId = lane;
                resumeCtx.policy = context.policy;
                return makeNextOp(wave, key, block, it, resumeCtx, lane);
            });
    }

    std::optional<StepType> gateSynchronousOp(WaveId wave,
                                              const DynamicBlockKey &key,
                                              mlir::Block *block,
                                              mlir::Block::iterator it,
                                              SemanticsContext context,
                                              LaneId lane) {
        auto waveIt = state_.waves.find(wave);
        if (waveIt == state_.waves.end())
            llvm::report_fatal_error("sync-cf: missing wave context");
        auto &waveCtx = waveIt->second;
        auto *blockCtx = getBlock(waveCtx, key);
        if (!blockCtx)
            llvm::report_fatal_error("sync-cf: missing block context");

        auto *op = &*it;
        std::uint64_t laneBit = 1ull << lane;
        auto readyIt = blockCtx->controlReadyMask.find(op);
        if (readyIt != blockCtx->controlReadyMask.end()) {
            if (readyIt->second & laneBit) {
                readyIt->second &= ~laneBit;
                if (readyIt->second == 0)
                    blockCtx->controlReadyMask.erase(readyIt);
                return std::nullopt;
            }
        }

        std::uint64_t expected =
            context.expectedMask ? context.expectedMask : context.activeMask;
        if (expected == 0)
            expected = laneBit;

        std::uint32_t token = 0;
        auto tokenIt = blockCtx->controlTokens.find(op);
        if (tokenIt == blockCtx->controlTokens.end()) {
            token = waveCtx.nextControlToken++;
            blockCtx->controlTokens[op] = token;
        } else {
            token = tokenIt->second;
        }
        waveCtx.syncTokenToOp[token] = op;

        SynchronizationEffect effect;
        effect.operation = 0;
        effect.activeMask = expected;
        effect.token = token;

        if (traceSink_) {
            traceSink_->onSuspend(
                wave, lane, Effect(effect), context.activeMask, expected,
                key.sequenceId, key.block, blockKindLabel(blockCtx->kind),
                blockCtx->loopIteration);
        }

        return StepType::suspend(
            Effect(std::move(effect)),
            [this, wave, key, block, it, lane, context]() mutable -> StepType {
                SemanticsContext resumeCtx;
                resumeCtx.laneId = lane;
                resumeCtx.policy = context.policy;
                return makeNextOp(wave, key, block, it, resumeCtx, lane);
            });
    }

    void handleControlFlowCollective(WaveId wave, const DynamicBlockKey &key,
                                     mlir::Operation *op,
                                     std::uint64_t expectedMask) {
        auto waveIt = state_.waves.find(wave);
        if (waveIt == state_.waves.end())
            llvm::report_fatal_error("collective-cf: missing wave context");
        auto &waveCtx = waveIt->second;
        auto *blockCtx = getBlock(waveCtx, key);
        if (!blockCtx)
            llvm::report_fatal_error("collective-cf: missing block context");

        mlir::Block *block = const_cast<mlir::Block *>(key.block);
        auto it = op->getIterator();

        std::uint64_t evalExpected =
            expectedMask ? expectedMask
                         : (blockCtx->expectedMask ? blockCtx->expectedMask
                                                   : blockCtx->activeMask);
        std::uint64_t evalActive =
            expectedMask ? expectedMask : blockCtx->activeMask;
        if (blockCtx->expectedMask == 0)
            blockCtx->expectedMask = evalExpected;

        llvm::DenseMap<LaneId, bool> ifDecisions;
        llvm::DenseMap<LaneId, std::int64_t> switchDecisions;
        if (auto ifOp = llvm::dyn_cast<simt::dialect::IfOp>(op)) {
            std::uint64_t mask = evalExpected;
            while (mask) {
                LaneId lane = static_cast<LaneId>(std::countr_zero(mask));
                mask &= mask - 1;
                auto condOrErr = evaluateBool(waveCtx, key, ifOp.getCondition(),
                                              lane, evalActive, evalExpected);
                bool takeThen = false;
                if (condOrErr)
                    takeThen = *condOrErr;
                else
                    llvm::consumeError(condOrErr.takeError());
                ifDecisions[lane] = takeThen;
            }
        } else if (auto switchOp = llvm::dyn_cast<simt::dialect::SwitchOp>(op)) {
            std::uint64_t mask = evalExpected;
            while (mask) {
                LaneId lane = static_cast<LaneId>(std::countr_zero(mask));
                mask &= mask - 1;
                auto selectorOrErr =
                    evaluateValue(waveCtx, key, switchOp.getSelector(), lane,
                                  evalActive, evalExpected);
                std::int64_t selectorValue = 0;
                if (selectorOrErr)
                    selectorValue = selectorOrErr->asInt64();
                else
                    llvm::consumeError(selectorOrErr.takeError());
                switchDecisions[lane] = selectorValue;
            }
        }

        auto dispatchLane = [&](LaneId lane) {
            blockCtx->activeMask |= (1ull << lane);
            SemanticsContext laneCtx;
            laneCtx.activeMask = evalActive;
            laneCtx.expectedMask = evalExpected;
            laneCtx.laneId = lane;
            laneCtx.policy = waveCtx.policy;
            laneCtx.overrideMode = ExecutionMode::Independent;
            laneCtx.suppressStepTrace = true;
            auto envIt = blockCtx->valueEnvs.find(lane);
            if (envIt != blockCtx->valueEnvs.end())
                laneCtx.valueEnv = &envIt->second;

            if (traceSink_ && !laneCtx.suppressStepTrace) {
                traceSink_->onResume(
                    wave, lane, evalActive, evalExpected, key.sequenceId, key.block,
                    blockKindLabel(blockCtx->kind), blockCtx->loopIteration);
            }

            if (llvm::isa<simt::dialect::IfOp>(op)) {
                (void)handleIfSplit(wave, key, block, it, laneCtx, lane,
                                    &ifDecisions);
                return;
            }
            if (llvm::isa<simt::dialect::LoopOp>(op)) {
                (void)handleLoopSplit(wave, key, block, it, laneCtx, lane);
                return;
            }
            if (llvm::isa<simt::dialect::SwitchOp>(op)) {
                (void)handleSwitchSplit(wave, key, block, it, laneCtx, lane,
                                        &switchDecisions);
                return;
            }

            llvm::report_fatal_error("collective-cf: unsupported control op");
        };

        std::uint64_t mask = evalExpected;
        while (mask) {
            LaneId lane = static_cast<LaneId>(std::countr_zero(mask));
            mask &= mask - 1;
            dispatchLane(lane);
        }
    }

    std::optional<StepType> handleLoopSplit(WaveId wave,
                                            const DynamicBlockKey &key,
                                            mlir::Block *block,
                                            mlir::Block::iterator it,
                                            SemanticsContext context,
                                            LaneId lane) {
        auto loopOp = llvm::dyn_cast<simt::dialect::LoopOp>(&*it);
        if (!loopOp)
            return std::nullopt;

        auto waveIt = state_.waves.find(wave);
        if (waveIt == state_.waves.end())
            llvm::report_fatal_error("handleLoopSplit: missing wave context");
        auto &waveCtx = waveIt->second;
        auto parentBlockIt = waveCtx.blocks.find(key);
        if (parentBlockIt == waveCtx.blocks.end())
            llvm::report_fatal_error("handleLoopSplit: missing parent block context");
        auto &parentBlock = parentBlockIt->second;
        std::uint64_t laneBit = 1ull << lane;
        // if ((parentBlock.activeMask & laneBit) == 0)
        //     return StepType::halt();

        // std::uint64_t activeMask = parentBlock.activeMask;
        // if (activeMask == 0)
        //     return StepType::halt();
        std::uint64_t parentExpected = parentBlock.expectedMask;

        if (auto gated = gateControlFlowOp(wave, key, block, it, context, lane))
            return gated;
        if (traceSink_ && !context.suppressStepTrace) {
            std::uint64_t expectedMask =
                context.expectedMask ? context.expectedMask : context.activeMask;
            if (expectedMask == 0)
                expectedMask = laneBit;
            traceSink_->onStepBegin(
                wave, lane, it->getName().getStringRef().str(),
                context.activeMask, expectedMask, key.sequenceId, key.block,
                blockKindLabel(parentBlock.kind), parentBlock.loopIteration);
        }

        if (EnableCPSDebugLogs) {
            auto fmt = [&](std::uint64_t m) { return formatMaskBits(m, 32); };
            llvm::errs() << "[CPS] handleLoopSplit lane=" << lane
                         << " parent=" << key.block << " seq=" << key.sequenceId
                         << " active=" << fmt(parentBlock.activeMask)
                         << " expected=" << fmt(parentBlock.expectedMask)
                         << "\n";
        }

        mlir::Block *prepareBlock = &loopOp.getPrepareRegion().front();
        mlir::Block *bodyBlock = &loopOp.getBodyRegion().front();

        std::uint32_t baseSeq = key.sequenceId + 1;
        DynamicBlockKey prepKey{prepareBlock, baseSeq};
        DynamicBlockKey bodyKey{bodyBlock, baseSeq + 1};

        auto &prepareCtx = waveCtx.blocks[prepKey];
        prepareCtx.block = prepareBlock;
        prepareCtx.sequenceId = prepKey.sequenceId;
        prepareCtx.parentKey = key;
        if (prepareCtx.expectedMask == 0)
            prepareCtx.expectedMask = parentExpected;
        prepareCtx.activeMask |= laneBit;
        prepareCtx.completedMask &= ~laneBit;
        prepareCtx.loopOp = loopOp.getOperation();
        prepareCtx.switchOp = nullptr;
        prepareCtx.isLoopPrepare = true;
        prepareCtx.isLoopBody = false;
        prepareCtx.loopIteration = 0;
        prepareCtx.kind = DynamicBlockKind::LoopPrepare;
        assert(!(prepareCtx.loopOp && prepareCtx.switchOp) &&
               "dynamic block cannot have both loopOp and switchOp");

        auto &bodyCtx = waveCtx.blocks[bodyKey];
        bodyCtx.block = bodyBlock;
        bodyCtx.sequenceId = bodyKey.sequenceId;
        bodyCtx.parentKey = prepKey;
        if (bodyCtx.expectedMask == 0)
            bodyCtx.expectedMask = parentExpected;
        bodyCtx.activeMask &= ~laneBit;
        bodyCtx.completedMask &= ~laneBit;
        bodyCtx.loopOp = loopOp.getOperation();
        bodyCtx.switchOp = nullptr;
        bodyCtx.ifOp = nullptr;
        bodyCtx.isLoopPrepare = false;
        bodyCtx.isLoopBody = true;
        bodyCtx.loopIteration = 0;
        bodyCtx.kind = DynamicBlockKind::LoopBody;
        assert(!(bodyCtx.loopOp && bodyCtx.switchOp) &&
               "dynamic block cannot have both loopOp and switchOp");

        auto nextIt = std::next(it);
        SemanticsContext parentContext = context;
        parentContext.overrideMode.reset();
        parentContext.suppressStepTrace = false;
        StepType parentCont = StepType::continueWith(
            [this, wave, key, block, nextIt, parentContext, lane]() mutable
            -> StepType {
                return makeNextOp(wave, key, block, nextIt, parentContext, lane);
            });
        // Store for later reconvergence; do not enqueue until the lane returns.
        parentBlock.continuations[lane] = parentCont;

        auto findEntry = [&](WaveContext<ValueType, StepType> &ctx,
                             const DynamicBlockKey &parentKey,
                             const mlir::Operation *loop) {
            for (auto it = ctx.mergeStack.rbegin(); it != ctx.mergeStack.rend(); ++it) {
                if (it->parent == parentKey && it->loopFrame &&
                    it->loopFrame->loopOp == loop)
                    return &*it;
            }
            return static_cast<MergeStackEntry<ValueType, StepType> *>(nullptr);
        };
        MergeStackEntry<ValueType, StepType> *entry =
            findEntry(waveCtx, key, loopOp.getOperation());
        if (!entry) {
            MergeStackEntry<ValueType, StepType> newEntry;
            newEntry.parent = key;
            newEntry.loopFrame.emplace();
            newEntry.loopFrame->loopOp = loopOp.getOperation();
            newEntry.loopFrame->prepareKey = prepKey;
            newEntry.loopFrame->bodyKey = bodyKey;
            waveCtx.mergeStack.push_back(std::move(newEntry));
            entry = &waveCtx.mergeStack.back();
            if (EnableCPSDebugLogs) {
                llvm::errs() << "[CPS] push merge (loop) parent=" << key.block
                             << " seq=" << key.sequenceId << "\n";
                logMergeStackState<ValueType, StepType>(waveCtx);
            }
        }
        auto &loopFrame = *entry->loopFrame;
        if (!llvm::is_contained(entry->pendingChildren, prepKey)) {
            entry->pendingChildren.push_back(prepKey);
            entry->childMasks.push_back(0);
        }
        if (!llvm::is_contained(entry->pendingChildren, bodyKey)) {
            entry->pendingChildren.push_back(bodyKey);
            entry->childMasks.push_back(0);
        }
        entry->expectedMask |= (parentExpected ? (parentExpected & laneBit) : laneBit);

        auto inits = loopOp.getInits();
        llvm::ArrayRef<mlir::BlockArgument> prepArgs = prepareBlock->getArguments();
        auto &tuple = loopFrame.carried[lane];
        tuple.clear();
        tuple.reserve(inits.size());
        std::uint64_t evalActive =
            context.activeMask ? context.activeMask : parentBlock.activeMask;
        std::uint64_t evalExpected =
            context.expectedMask
                ? context.expectedMask
                : (parentBlock.expectedMask ? parentBlock.expectedMask
                                            : parentBlock.activeMask);
        for (mlir::Value init : inits) {
            auto valueOrErr =
                evaluateValue(waveCtx, key, init, lane, evalActive, evalExpected);
            if (!valueOrErr)
                llvm::report_fatal_error("handleLoopSplit: failed to evaluate init");
            tuple.push_back(*valueOrErr);
        }
        loopFrame.laneNextSeq[lane] = bodyKey.sequenceId + 1;

        auto &env = prepareCtx.valueEnvs[lane];
        for (auto indexed : llvm::enumerate(prepArgs)) {
            if (indexed.index() < tuple.size())
                env[indexed.value()] = tuple[indexed.index()];
        }

        SemanticsContext childContext = context;
        childContext.overrideMode.reset();
        childContext.suppressStepTrace = false;
        childContext.activeMask = prepareCtx.activeMask;
        childContext.laneId = lane;
        StepType childStep = makeNextOp(wave, prepKey, prepareBlock,
                                        prepareBlock->begin(), childContext, lane);
        enqueue(wave, prepKey, lane, std::move(childStep));

        parentBlock.activeMask &= ~laneBit;
        return StepType::halt();
    }

    std::optional<StepType> handleSwitchSplit(WaveId wave,
                                              const DynamicBlockKey &key,
                                              mlir::Block *block,
                                              mlir::Block::iterator it,
                                              SemanticsContext context,
                                              LaneId lane,
                                              const SwitchDecisionMap *decisions = nullptr) {
        auto switchOp = llvm::dyn_cast<simt::dialect::SwitchOp>(&*it);
        if (!switchOp)
            return std::nullopt;

        auto waveIt = state_.waves.find(wave);
        if (waveIt == state_.waves.end())
            llvm::report_fatal_error("handleSwitchSplit: missing wave context");
        auto &waveCtx = waveIt->second;
        auto parentBlockIt = waveCtx.blocks.find(key);
        if (parentBlockIt == waveCtx.blocks.end())
            llvm::report_fatal_error("handleSwitchSplit: missing parent block");
        auto &parentBlock = parentBlockIt->second;
        std::uint64_t laneBit = 1ull << lane;
        if ((parentBlock.activeMask & laneBit) == 0)
            parentBlock.activeMask |= laneBit;

        std::uint64_t parentExpected =
            parentBlock.expectedMask ? parentBlock.expectedMask : parentBlock.activeMask;

        if (auto gated = gateControlFlowOp(wave, key, block, it, context, lane))
            return gated;
        if (traceSink_ && !context.suppressStepTrace) {
            std::uint64_t expectedMask =
                context.expectedMask ? context.expectedMask : context.activeMask;
            if (expectedMask == 0)
                expectedMask = laneBit;
            traceSink_->onStepBegin(
                wave, lane, it->getName().getStringRef().str(),
                context.activeMask, expectedMask, key.sequenceId, key.block,
                blockKindLabel(parentBlock.kind), parentBlock.loopIteration);
        }

        mlir::Region &caseRegion = switchOp.getCaseBody();
        unsigned numBlocks = static_cast<unsigned>(std::distance(caseRegion.begin(),
                                                                 caseRegion.end()));
        llvm::SmallVector<mlir::Block *, 4> caseBlocks;
        caseBlocks.reserve(numBlocks);
        for (mlir::Block &b : caseRegion)
            caseBlocks.push_back(&b);
        if (caseBlocks.empty())
            llvm::report_fatal_error("handleSwitchSplit: missing case blocks");

        auto caseValues = switchOp.getCaseValues();
        if (caseValues.size() + 1 != numBlocks)
            llvm::report_fatal_error("handleSwitchSplit: case_values size mismatch");
        auto defaultIndexAttr = switchOp.getDefaultIndexAttr();
        if (!defaultIndexAttr)
            llvm::report_fatal_error("handleSwitchSplit: missing default_index attr");
        int64_t defaultIndex = defaultIndexAttr.getInt();
        if (defaultIndex < 0 ||
            static_cast<std::size_t>(defaultIndex) >= caseBlocks.size())
            llvm::report_fatal_error("handleSwitchSplit: default_index out of range");
        llvm::SmallVector<unsigned, 4> caseValueBlocks;
        caseValueBlocks.reserve(caseValues.size());
        for (unsigned idx = 0; idx < caseBlocks.size(); ++idx) {
            if (idx == static_cast<unsigned>(defaultIndex))
                continue;
            caseValueBlocks.push_back(idx);
        }
        if (caseValueBlocks.size() != caseValues.size())
            llvm::report_fatal_error("handleSwitchSplit: case_values mapping mismatch");

        auto nextIt = std::next(it);
        SemanticsContext parentContext = context;
        parentContext.overrideMode.reset();
        parentContext.suppressStepTrace = false;
        StepType parentCont = StepType::continueWith(
            [this, wave, key, block, nextIt, parentContext, lane]() mutable
            -> StepType {
                return makeNextOp(wave, key, block, nextIt, parentContext, lane);
            });
        // Store for later reconvergence; do not enqueue until the lane returns.
        parentBlock.continuations[lane] = parentCont;

        auto findEntry = [&](WaveContext<ValueType, StepType> &ctx,
                             const DynamicBlockKey &parentKey) {
            for (auto it = ctx.mergeStack.rbegin(); it != ctx.mergeStack.rend(); ++it) {
                if (!it->loopFrame && it->parent == parentKey)
                    return &*it;
            }
            return static_cast<MergeStackEntry<ValueType, StepType> *>(nullptr);
        };
        MergeStackEntry<ValueType, StepType> *entry = findEntry(waveCtx, key);
        if (!entry) {
            MergeStackEntry<ValueType, StepType> newEntry;
            newEntry.parent = key;
            waveCtx.mergeStack.push_back(std::move(newEntry));
            entry = &waveCtx.mergeStack.back();
            if (EnableCPSDebugLogs) {
                llvm::errs() << "[CPS] push merge (switch) parent=" << key.block
                             << " seq=" << key.sequenceId << "\n";
                logMergeStackState<ValueType, StepType>(waveCtx);
            }
        }

        std::uint64_t laneMask =
            parentExpected ? (parentExpected & laneBit) : laneBit;

        // C-like switch with explicit fallthrough: pick a case by selector, then
        // allow fallthrough to subsequent cases when the terminator requests it.
        std::uint64_t evalActive =
            context.activeMask ? context.activeMask : parentBlock.activeMask;
        std::uint64_t evalExpected =
            context.expectedMask
                ? context.expectedMask
                : (parentBlock.expectedMask ? parentBlock.expectedMask
                                            : parentBlock.activeMask);
        std::int64_t selectorValue = 0;
        if (decisions) {
            auto decisionIt = decisions->find(lane);
            if (decisionIt == decisions->end())
                llvm::report_fatal_error(
                    "handleSwitchSplit: missing selector decision");
            selectorValue = decisionIt->second;
        } else {
            auto selectorOrErr =
                evaluateValue(waveCtx, key, switchOp.getSelector(), lane,
                              evalActive, evalExpected);
            if (selectorOrErr)
                selectorValue = selectorOrErr->asInt64();
            else
                llvm::consumeError(selectorOrErr.takeError());
        }

        unsigned caseIdx = static_cast<unsigned>(defaultIndex);
        for (auto indexed : llvm::enumerate(caseValues)) {
            if (indexed.value() == selectorValue) {
                if (indexed.index() >= caseValueBlocks.size())
                    llvm::report_fatal_error(
                        "handleSwitchSplit: case_values mapping overflow");
                caseIdx = caseValueBlocks[indexed.index()];
                break;
            }
        }
        if (caseIdx >= numBlocks)
            llvm::report_fatal_error("handleSwitchSplit: target block not found");
        llvm::SmallVector<bool, 4> caseFallthrough;
        caseFallthrough.reserve(numBlocks);
        for (unsigned idx = 0; idx < numBlocks; ++idx) {
            mlir::Block *caseBlock = caseBlocks[idx];
            if (caseBlock->empty())
                llvm::report_fatal_error("handleSwitchSplit: missing switch yield");
            auto yield =
                llvm::dyn_cast<simt::dialect::YieldOp>(caseBlock->back());
            if (!yield)
                llvm::report_fatal_error("handleSwitchSplit: missing switch yield");
            auto attr = yield->getAttrOfType<mlir::BoolAttr>("fallthrough");
            if (!attr)
                llvm::report_fatal_error("handleSwitchSplit: missing fallthrough attr");
            bool fall = attr.getValue();
            caseFallthrough.push_back(fall);
        }

        unsigned lastIdx = caseIdx;
        while (lastIdx + 1 < numBlocks && caseFallthrough[lastIdx])
            ++lastIdx;

        std::uint32_t baseSeq = key.sequenceId + 1;
        if (!entry->switchFrame) {
            SwitchFrameState<ValueType> frame;
            frame.switchOp = switchOp.getOperation();
            frame.baseSeq = baseSeq;
            frame.caseBlocks.assign(caseBlocks.begin(), caseBlocks.end());
            entry->switchFrame = std::move(frame);
        }
        auto &frame = *entry->switchFrame;
        baseSeq = frame.baseSeq;
        if (entry->expectedMask == 0)
            entry->expectedMask = parentExpected ? parentExpected : laneMask;
        entry->expectedMask |= laneMask;
        if (entry->pendingChildren.empty()) {
            for (unsigned idx = 0; idx < caseBlocks.size(); ++idx) {
                entry->pendingChildren.push_back(
                    DynamicBlockKey{caseBlocks[idx], baseSeq + idx});
                entry->childMasks.push_back(0);
            }
        }

        mlir::Block *targetBlock = caseBlocks[caseIdx];
        std::uint32_t seq = baseSeq + caseIdx;
        DynamicBlockKey childKey{targetBlock, seq};

        auto &childCtx = waveCtx.blocks[childKey];
        childCtx.block = childKey.block;
        childCtx.sequenceId = childKey.sequenceId;
        childCtx.parentKey = key;
        if (childCtx.expectedMask == 0)
            childCtx.expectedMask = parentExpected ? parentExpected : laneMask;
        childCtx.expectedMask |= laneMask;
        childCtx.activeMask |= laneBit;
        childCtx.completedMask &= ~laneBit;
        childCtx.kind = (caseIdx == static_cast<unsigned>(defaultIndex))
                            ? DynamicBlockKind::SwitchDefault
                            : DynamicBlockKind::SwitchCase;
        childCtx.switchOp = switchOp.getOperation();
        childCtx.loopOp = nullptr;
        childCtx.ifOp = nullptr;

        for (unsigned pathIdx = caseIdx + 1; pathIdx <= lastIdx; ++pathIdx) {
            DynamicBlockKey pathKey{caseBlocks[pathIdx], baseSeq + pathIdx};
            auto &pathCtx = waveCtx.blocks[pathKey];
            pathCtx.block = pathKey.block;
            pathCtx.sequenceId = pathKey.sequenceId;
            pathCtx.parentKey = key;
            pathCtx.switchOp = switchOp.getOperation();
            pathCtx.loopOp = nullptr;
            pathCtx.ifOp = nullptr;
            pathCtx.kind = (pathIdx == static_cast<unsigned>(defaultIndex))
                               ? DynamicBlockKind::SwitchDefault
                               : DynamicBlockKind::SwitchCase;
            if (pathCtx.expectedMask == 0)
                pathCtx.expectedMask =
                    parentExpected ? parentExpected : laneMask;
            pathCtx.expectedMask |= laneBit;
        }

        auto isDynamicDescendant = [&](const DynamicBlockKey &desc,
                                       const DynamicBlockKey &ancestor) {
            DynamicBlockKey cur = desc;
            while (true) {
                if (cur == ancestor)
                    return true;
                auto it = waveCtx.blocks.find(cur);
                if (it == waveCtx.blocks.end() || !it->second.parentKey)
                    return false;
                cur = *it->second.parentKey;
            }
        };

        for (unsigned otherIdx = 0; otherIdx < numBlocks; ++otherIdx) {
            if (otherIdx >= caseIdx && otherIdx <= lastIdx)
                continue;
            DynamicBlockKey otherKey{caseBlocks[otherIdx], baseSeq + otherIdx};
            auto &otherCtx = waveCtx.blocks[otherKey];
            otherCtx.block = otherKey.block;
            otherCtx.sequenceId = otherKey.sequenceId;
            otherCtx.parentKey = key;
            otherCtx.switchOp = switchOp.getOperation();
            otherCtx.loopOp = nullptr;
            otherCtx.ifOp = nullptr;
            otherCtx.kind = (otherIdx == static_cast<unsigned>(defaultIndex))
                                ? DynamicBlockKind::SwitchDefault
                                : DynamicBlockKind::SwitchCase;
            if (otherCtx.expectedMask == 0)
                otherCtx.expectedMask =
                    parentExpected ? parentExpected : laneMask;
            otherCtx.expectedMask &= ~laneMask;
            for (auto &kv : waveCtx.blocks) {
                const auto &descKey = kv.first;
                auto &desc = kv.second;
                if (isDynamicDescendant(descKey, otherKey))
                    desc.expectedMask &= ~laneMask;
            }
        }

        auto &env = childCtx.valueEnvs[lane];
        auto childArgs = targetBlock->getArguments();
        auto inits = switchOp.getInitialValues();
        llvm::SmallVector<ValueType, 8> initVals;
        initVals.reserve(inits.size());
        for (mlir::Value init : inits) {
            auto valOrErr =
                evaluateValue(waveCtx, key, init, lane, evalActive, evalExpected);
            if (!valOrErr)
                llvm::report_fatal_error("handleSwitchSplit: failed to evaluate init");
            initVals.push_back(*valOrErr);
        }
        frame.carried[lane] = initVals;
        for (auto indexed : llvm::enumerate(childArgs)) {
            if (indexed.index() < initVals.size())
                env[indexed.value()] = initVals[indexed.index()];
        }

        if (EnableCPSDebugLogs) {
            auto fmt = [&](std::uint64_t m) { return formatMaskBits(m, 32); };
            llvm::errs() << "[CPS] handleSwitchSplit lane=" << lane
                         << " parent=" << key.block << " seq=" << key.sequenceId
                         << " -> caseIdx=" << caseIdx
                         << " childSeq=" << seq
                         << " active=" << fmt(parentBlock.activeMask)
                         << " expected=" << fmt(parentBlock.expectedMask)
                         << "\n";
        }

        SemanticsContext laneCtx = context;
        laneCtx.overrideMode.reset();
        laneCtx.suppressStepTrace = false;
        laneCtx.activeMask = childCtx.activeMask;
        laneCtx.expectedMask =
            childCtx.expectedMask ? childCtx.expectedMask : childCtx.activeMask;
        laneCtx.laneId = lane;
        mlir::Block *childBlock = const_cast<mlir::Block *>(childKey.block);
        StepType childStep =
            makeNextOp(wave, childKey, childBlock, childBlock->begin(), laneCtx, lane);
        enqueue(wave, childKey, lane, std::move(childStep));
        parentBlock.activeMask &= ~laneBit;
        return StepType::halt();
    }

    MergeStackEntry<ValueType, StepType> *
    findLoopEntry(WaveContext<ValueType, StepType> &waveCtx,
                  const mlir::Operation *loopOp) {
        for (auto it = waveCtx.mergeStack.rbegin();
             it != waveCtx.mergeStack.rend(); ++it) {
            if (it->loopFrame && it->loopFrame->loopOp == loopOp)
                return &*it;
        }
        return nullptr;
    }

    std::optional<StepType> handleLoopPrepareTerminator(
        WaveId wave, const DynamicBlockKey &key, mlir::Block *block,
        mlir::Block::iterator it, SemanticsContext context, LaneId lane) {
        auto condOp = llvm::dyn_cast<simt::dialect::ConditionOp>(&*it);
        if (!condOp)
            return std::nullopt;

        (void)block;
        auto waveIt = state_.waves.find(wave);
        if (waveIt == state_.waves.end())
            llvm::report_fatal_error("handleLoopPrepareTerminator: missing wave context");
        auto &waveCtx = waveIt->second;
        auto *blockCtx = getBlock(waveCtx, key);
        if (!blockCtx || !blockCtx->isLoopPrepare || !blockCtx->loopOp)
            llvm::report_fatal_error("handleLoopPrepareTerminator: invalid block context");
        if ((blockCtx->activeMask & (1ull << lane)) == 0){
            llvm::report_fatal_error("handleLoopPrepareTerminator: invalid active mask");
        }

        auto *entry = findLoopEntry(waveCtx, blockCtx->loopOp);
        if (!entry || !entry->loopFrame) {
            llvm::errs() << "[CPS] handleLoopPrepareTerminator missing loop frame "
                         << "lane=" << lane << " key=" << key.block
                         << " seq=" << key.sequenceId << "\n";
            logMergeStackState<ValueType, StepType>(waveCtx);
            llvm::report_fatal_error("handleLoopPrepareTerminator: missing loop frame");
        }
        auto &loopFrame = *entry->loopFrame;

        std::uint64_t laneBit = 1ull << lane;
        auto condOrErr = evaluateBool(
            waveCtx, key, condOp.getCondition(), lane, blockCtx->activeMask,
            blockCtx->expectedMask ? blockCtx->expectedMask : blockCtx->activeMask);
        bool takeBody = false;
        if (condOrErr)
            takeBody = *condOrErr;
        else
            llvm::consumeError(condOrErr.takeError());

        llvm::SmallVector<ValueType, 4> forwarded;
        forwarded.reserve(condOp.getForwarded().size());
        for (mlir::Value v : condOp.getForwarded()) {
            auto valOrErr = evaluateValue(
                waveCtx, key, v, lane, blockCtx->activeMask,
                blockCtx->expectedMask ? blockCtx->expectedMask : blockCtx->activeMask);
            if (!valOrErr) {
                llvm::report_fatal_error("handleLoopPrepareTerminator: failed to evaluate forwarded value");
            }
            forwarded.push_back(*valOrErr);
        }
        loopFrame.carried[lane].assign(forwarded.begin(), forwarded.end());

        blockCtx->activeMask &= ~laneBit;
        blockCtx->completedMask |= laneBit;

        if (EnableCPSDebugLogs) {
            auto fmt = [&](std::uint64_t m) { return formatMaskBits(m, 32); };
            llvm::errs() << "[CPS] handleLoopPrepareTerminator lane=" << lane
                         << " block=" << key.block << " seq=" << key.sequenceId
                         << " cond=" << (takeBody ? "true" : "false")
                         << " takeBody=" << takeBody
                         << " active=" << fmt(blockCtx->activeMask)
                         << " expected=" << fmt(blockCtx->expectedMask)
                         << "\n";
        }

        LLVM_DEBUG(llvm::dbgs() << "[CPS] handleLoopPrepareTerminator lane=" << lane
                                << " block=" << key.block << " seq=" << key.sequenceId
                                << " takeBody=" << takeBody
                                << " active=0x" << llvm::format_hex(blockCtx->activeMask, 10)
                                << " expected=0x" << llvm::format_hex(blockCtx->expectedMask, 10)
                                << "\n");

        if (takeBody) {
            DynamicBlockKey bodyKey{loopFrame.bodyKey.block,
                                    static_cast<std::uint32_t>(key.sequenceId + 1)};
            bool isNew = !waveCtx.blocks.contains(bodyKey);
            auto &bodyCtx = waveCtx.blocks[bodyKey];
            bodyCtx.block = bodyKey.block;
            bodyCtx.sequenceId = bodyKey.sequenceId;
            if (bodyCtx.expectedMask == 0)
                bodyCtx.expectedMask =
                    blockCtx->expectedMask ? blockCtx->expectedMask
                                           : blockCtx->activeMask;
            bodyCtx.activeMask |= laneBit;
            bodyCtx.loopOp = blockCtx->loopOp;
            bodyCtx.isLoopBody = true;
            bodyCtx.isLoopPrepare = false;
            bodyCtx.kind = DynamicBlockKind::LoopBody;
            if (key.sequenceId >= loopFrame.prepareKey.sequenceId) {
                bodyCtx.loopIteration =
                    (key.sequenceId - loopFrame.prepareKey.sequenceId) / 2;
            } else {
                bodyCtx.loopIteration.reset();
            }

            auto &env = bodyCtx.valueEnvs[lane];
            auto bodyArgs =
                const_cast<mlir::Block *>(bodyKey.block)->getArguments();
            for (auto indexed : llvm::enumerate(bodyArgs)) {
                if (indexed.index() < forwarded.size())
                    env[indexed.value()] = forwarded[indexed.index()];
            }

            if (isNew && !llvm::is_contained(entry->pendingChildren, bodyKey)) {
                entry->pendingChildren.push_back(bodyKey);
                entry->childMasks.push_back(bodyCtx.activeMask);
            }

            SemanticsContext laneCtx = context;
            laneCtx.overrideMode.reset();
            laneCtx.activeMask = bodyCtx.activeMask;
            laneCtx.expectedMask =
                bodyCtx.expectedMask ? bodyCtx.expectedMask : bodyCtx.activeMask;
            laneCtx.laneId = lane;
            mlir::Block *childBlock = const_cast<mlir::Block *>(bodyKey.block);
            StepType childStep =
                makeNextOp(wave, bodyKey, childBlock, childBlock->begin(),
                           laneCtx, lane);
            enqueue(wave, bodyKey, lane, std::move(childStep));
            return StepType::halt();
        }

        auto parentIt = waveCtx.blocks.find(entry->parent);
        if (parentIt != waveCtx.blocks.end()) {
            auto &parentEnv = parentIt->second.valueEnvs[lane];
            unsigned idx = 0;
            auto *loopOperation = const_cast<mlir::Operation *>(loopFrame.loopOp);
            for (mlir::Value res : loopOperation->getResults()) {
                if (idx < forwarded.size())
                    parentEnv[res] = forwarded[idx];
                ++idx;
            }
            auto contIt = parentIt->second.continuations.find(lane);
            if (contIt != parentIt->second.continuations.end()) {
                parentIt->second.activeMask |= laneBit;
                state_.readyQueue.push(ReadyContinuation<ValueType, StepType>{
                    wave, entry->parent, lane, contIt->second});
                parentIt->second.continuations.erase(contIt);
                auto &laneCtx = waveCtx.lanes[lane];
                laneCtx.currentBlock = entry->parent;
            }
        }
        handleReconvergence(wave, waveCtx, key, lane);
        if (entry && entry->loopFrame) {
            bool loopDone =
                entry->expectedMask != 0
                    ? (entry->completedMask == entry->expectedMask)
                    : entry->pendingChildren.empty();
            if (loopDone)
                waveCtx.mergeStack.pop_back();
        }
        return StepType::halt();
    }

    std::optional<StepType> handleLoopYield(WaveId wave,
                                            const DynamicBlockKey &key,
                                            mlir::Block *block,
                                            mlir::Block::iterator it,
                                            SemanticsContext context,
                                            LaneId lane) {
        auto yieldOp = llvm::dyn_cast<simt::dialect::YieldOp>(&*it);
        if (!yieldOp)
            return std::nullopt;

        (void)block;
        auto waveIt = state_.waves.find(wave);
        if (waveIt == state_.waves.end())
            llvm::report_fatal_error("handleLoopYield: missing wave context");
        auto &waveCtx = waveIt->second;
        auto *blockCtx = getBlock(waveCtx, key);
        if (!blockCtx || !blockCtx->isLoopBody || !blockCtx->loopOp)
            return std::nullopt;
        if ((blockCtx->activeMask & (1ull << lane)) == 0) {
            llvm::report_fatal_error("handleLoopYield: invalid active mask");
        }
        assert(!(blockCtx->loopOp && blockCtx->switchOp) &&
               "dynamic block cannot have both loopOp and switchOp");

        if (EnableCPSDebugLogs) {
            auto fmt = [&](std::uint64_t m) { return formatMaskBits(m, 32); };
            llvm::errs() << "[CPS] handleLoopYield lane=" << lane
                         << " block=" << key.block << " seq=" << key.sequenceId
                         << " active=" << fmt(blockCtx->activeMask)
                         << " expected=" << fmt(blockCtx->expectedMask)
                         << "\n";
        }

        auto *entry = findLoopEntry(waveCtx, blockCtx->loopOp);
        if (!entry || !entry->loopFrame)
            llvm::report_fatal_error("handleLoopYield: missing loop frame");
        auto &loopFrame = *entry->loopFrame;
        std::uint64_t laneBit = 1ull << lane;

        llvm::SmallVector<ValueType, 4> nextCarried;
        nextCarried.reserve(yieldOp.getNumOperands());
        auto envIt = blockCtx->valueEnvs.find(lane);
        if (envIt == blockCtx->valueEnvs.end()) {
            llvm::errs() << "[CPS] handleLoopYield missing value env for lane=" << lane
                         << " seq=" << key.sequenceId << " block=" << key.block << "\n";
        }
        for (mlir::Value v : yieldOp.getOperands()) {
            auto valOrErr =
                evaluateValue(waveCtx, key, v, lane, blockCtx->activeMask,
                              blockCtx->expectedMask ? blockCtx->expectedMask
                                                     : blockCtx->activeMask);
            if (!valOrErr) {
                llvm::errs() << "[CPS] handleLoopYield eval failure lane=" << lane
                             << " seq=" << key.sequenceId << " block=" << key.block
                             << " operand=" << nextCarried.size() << "\n";
                if (envIt != blockCtx->valueEnvs.end()) {
                    llvm::errs() << "  env entries: " << envIt->second.size() << "\n";
                    for (auto &kv : envIt->second) {
                        llvm::errs() << "    - ";
                        kv.first.print(llvm::errs());
                        llvm::errs() << "\n";
                    }
                }
                v.print(llvm::errs());
                llvm::errs() << "\n";
                llvm::consumeError(valOrErr.takeError());
                llvm::report_fatal_error("handleLoopYield: failed to evaluate yield operand");
            }
            nextCarried.push_back(*valOrErr);
        }
        loopFrame.carried[lane].assign(nextCarried.begin(), nextCarried.end());

        blockCtx->activeMask &= ~laneBit;
        blockCtx->completedMask |= laneBit;

        if (EnableCPSDebugLogs) {
            auto fmt = [&](std::uint64_t m) { return formatMaskBits(m, 32); };
            llvm::errs() << "[CPS] handleLoopContinue lane=" << lane
                         << " block=" << key.block << " seq=" << key.sequenceId
                         << " active=" << fmt(blockCtx->activeMask)
                         << " expected=" << fmt(blockCtx->expectedMask)
                         << "\n";
        }

        std::uint32_t nextSeq =
            loopFrame.laneNextSeq.try_emplace(lane, key.sequenceId + 2).first->second;
        DynamicBlockKey nextPrep{loopFrame.prepareKey.block, nextSeq};
        DynamicBlockKey nextBody{loopFrame.bodyKey.block,
                                 static_cast<std::uint32_t>(nextSeq + 1)};
        std::uint32_t loopIteration = 0;
        if (nextPrep.sequenceId >= loopFrame.prepareKey.sequenceId)
            loopIteration =
                (nextPrep.sequenceId - loopFrame.prepareKey.sequenceId) / 2;
        auto &prepCtx = waveCtx.blocks[nextPrep];
        prepCtx.block = nextPrep.block;
        prepCtx.sequenceId = nextPrep.sequenceId;
        prepCtx.parentKey = key;
        prepCtx.expectedMask =
            blockCtx->expectedMask ? blockCtx->expectedMask : blockCtx->activeMask;
        prepCtx.activeMask |= laneBit;
        prepCtx.completedMask = 0;
        prepCtx.loopOp = blockCtx->loopOp;
        prepCtx.switchOp = nullptr;
        prepCtx.ifOp = nullptr;
        prepCtx.isLoopPrepare = true;
        prepCtx.isLoopBody = false;
        prepCtx.loopIteration = loopIteration;
        prepCtx.kind = DynamicBlockKind::LoopPrepare;

        auto &bodyCtx = waveCtx.blocks[nextBody];
        bodyCtx.block = nextBody.block;
        bodyCtx.sequenceId = nextBody.sequenceId;
        bodyCtx.parentKey = nextPrep;
        bodyCtx.expectedMask =
            blockCtx->expectedMask ? blockCtx->expectedMask : blockCtx->activeMask;
        bodyCtx.activeMask = 0;
        bodyCtx.completedMask = 0;
        bodyCtx.loopOp = blockCtx->loopOp;
        bodyCtx.switchOp = nullptr;
        bodyCtx.ifOp = nullptr;
        bodyCtx.isLoopPrepare = false;
        bodyCtx.isLoopBody = true;
        bodyCtx.loopIteration = loopIteration;
        bodyCtx.kind = DynamicBlockKind::LoopBody;
        assert(!(prepCtx.loopOp && prepCtx.switchOp) &&
               "dynamic block cannot have both loopOp and switchOp");
        assert(!(bodyCtx.loopOp && bodyCtx.switchOp) &&
               "dynamic block cannot have both loopOp and switchOp");

        bool nextExists = waveCtx.blocks.contains(nextPrep);
        if (!nextExists && !llvm::is_contained(entry->pendingChildren, nextPrep)) {
            entry->pendingChildren.push_back(nextPrep);
            entry->childMasks.push_back(prepCtx.activeMask);
        }
        if (!nextExists && !llvm::is_contained(entry->pendingChildren, nextBody)) {
            entry->pendingChildren.push_back(nextBody);
            entry->childMasks.push_back(bodyCtx.activeMask);
        }

        auto prepArgs =
            const_cast<mlir::Block *>(nextPrep.block)->getArguments();
        auto &env = prepCtx.valueEnvs[lane];
        for (auto indexed : llvm::enumerate(prepArgs)) {
            if (indexed.index() < nextCarried.size())
                env[indexed.value()] = nextCarried[indexed.index()];
        }

        SemanticsContext laneCtx = context;
        laneCtx.overrideMode.reset();
        laneCtx.activeMask = prepCtx.activeMask;
        laneCtx.expectedMask =
            prepCtx.expectedMask ? prepCtx.expectedMask : prepCtx.activeMask;
        laneCtx.laneId = lane;
        mlir::Block *prepBlock = const_cast<mlir::Block *>(nextPrep.block);
        StepType childStep =
            makeNextOp(wave, nextPrep, prepBlock, prepBlock->begin(), laneCtx, lane);
        enqueue(wave, nextPrep, lane, std::move(childStep));
        loopFrame.laneNextSeq[lane] = nextSeq + 2;
        return StepType::halt();
    }

    std::optional<StepType> handleLoopContinue(WaveId wave,
                                               const DynamicBlockKey &key,
                                               mlir::Block *block,
                                               mlir::Block::iterator it,
                                               SemanticsContext context,
                                               LaneId lane) {
        auto contOp = llvm::dyn_cast<simt::dialect::ContinueOp>(&*it);
        if (!contOp)
            return std::nullopt;

        (void)block;
        auto waveIt = state_.waves.find(wave);
        if (waveIt == state_.waves.end())
            llvm::report_fatal_error("handleLoopContinue: missing wave context");
        auto &waveCtx = waveIt->second;
        auto *blockCtx = getBlock(waveCtx, key);
        if (!blockCtx || !blockCtx->isLoopBody || !blockCtx->loopOp)
            llvm::report_fatal_error("handleLoopContinue: invalid block context");
        if ((blockCtx->activeMask & (1ull << lane)) == 0)
            llvm::report_fatal_error("handleLoopContinue: invalid active mask");

        if (EnableCPSDebugLogs) {
            auto fmt = [&](std::uint64_t m) { return formatMaskBits(m, 32); };
            llvm::errs() << "[CPS] handleLoopContinue lane=" << lane
                         << " block=" << key.block << " seq=" << key.sequenceId
                         << " active=" << fmt(blockCtx->activeMask)
                         << " expected=" << fmt(blockCtx->expectedMask)
                         << "\n";
        }

        auto *entry = findLoopEntry(waveCtx, blockCtx->loopOp);
        if (!entry || !entry->loopFrame)
            llvm::report_fatal_error("handleLoopContinue: missing loop frame");
        auto &loopFrame = *entry->loopFrame;
        std::uint64_t laneBit = 1ull << lane;

        llvm::SmallVector<ValueType, 4> nextCarried;
        nextCarried.reserve(contOp.getNumOperands());
        for (mlir::Value v : contOp.getOperands()) {
            auto valOrErr =
                evaluateValue(waveCtx, key, v, lane, blockCtx->activeMask,
                              blockCtx->expectedMask ? blockCtx->expectedMask
                                                     : blockCtx->activeMask);
            if (!valOrErr) {
                llvm::report_fatal_error("handleLoopContinue: failed to evaluate continue operand");
            }
            nextCarried.push_back(*valOrErr);
        }
        loopFrame.carried[lane].assign(nextCarried.begin(), nextCarried.end());

        blockCtx->activeMask &= ~laneBit;
        blockCtx->completedMask |= laneBit;

        std::uint32_t nextSeq =
            loopFrame.laneNextSeq.try_emplace(lane, key.sequenceId + 2).first->second;
        DynamicBlockKey nextPrep{loopFrame.prepareKey.block, nextSeq};
        DynamicBlockKey nextBody{loopFrame.bodyKey.block,
                                 static_cast<std::uint32_t>(nextSeq + 1)};
        std::uint32_t loopIteration = 0;
        if (nextPrep.sequenceId >= loopFrame.prepareKey.sequenceId)
            loopIteration =
                (nextPrep.sequenceId - loopFrame.prepareKey.sequenceId) / 2;
        bool nextExists = waveCtx.blocks.contains(nextPrep);

        auto &prepCtx = waveCtx.blocks[nextPrep];
        prepCtx.block = nextPrep.block;
        prepCtx.sequenceId = nextPrep.sequenceId;
        prepCtx.parentKey = key;
        prepCtx.expectedMask =
            blockCtx->expectedMask ? blockCtx->expectedMask : blockCtx->activeMask;
        prepCtx.activeMask |= laneBit;
        prepCtx.completedMask = 0;
        prepCtx.loopOp = blockCtx->loopOp;
        prepCtx.ifOp = nullptr;
        prepCtx.isLoopPrepare = true;
        prepCtx.isLoopBody = false;
        prepCtx.loopIteration = loopIteration;
        prepCtx.kind = DynamicBlockKind::LoopPrepare;

        auto &bodyCtx = waveCtx.blocks[nextBody];
        bodyCtx.block = nextBody.block;
        bodyCtx.sequenceId = nextBody.sequenceId;
        bodyCtx.parentKey = nextPrep;
        bodyCtx.expectedMask =
            blockCtx->expectedMask ? blockCtx->expectedMask : blockCtx->activeMask;
        bodyCtx.activeMask = 0;
        bodyCtx.completedMask = 0;
        bodyCtx.loopOp = blockCtx->loopOp;
        bodyCtx.ifOp = nullptr;
        bodyCtx.isLoopPrepare = false;
        bodyCtx.isLoopBody = true;
        bodyCtx.loopIteration = loopIteration;
        bodyCtx.kind = DynamicBlockKind::LoopBody;

        if (!nextExists && !llvm::is_contained(entry->pendingChildren, nextPrep)) {
            entry->pendingChildren.push_back(nextPrep);
            entry->childMasks.push_back(prepCtx.activeMask);
        }
        if (!nextExists && !llvm::is_contained(entry->pendingChildren, nextBody)) {
            entry->pendingChildren.push_back(nextBody);
            entry->childMasks.push_back(bodyCtx.activeMask);
        }

        auto prepArgs =
            const_cast<mlir::Block *>(nextPrep.block)->getArguments();
        auto &env = prepCtx.valueEnvs[lane];
        for (auto indexed : llvm::enumerate(prepArgs)) {
            if (indexed.index() < nextCarried.size())
                env[indexed.value()] = nextCarried[indexed.index()];
        }

        SemanticsContext laneCtx = context;
        laneCtx.overrideMode.reset();
        laneCtx.activeMask = prepCtx.activeMask;
        laneCtx.expectedMask =
            prepCtx.expectedMask ? prepCtx.expectedMask : prepCtx.activeMask;
        laneCtx.laneId = lane;
        mlir::Block *prepBlock = const_cast<mlir::Block *>(nextPrep.block);
        StepType childStep =
            makeNextOp(wave, nextPrep, prepBlock, prepBlock->begin(), laneCtx, lane);
        enqueue(wave, nextPrep, lane, std::move(childStep));
        loopFrame.laneNextSeq[lane] = nextSeq + 2;
        return StepType::halt();
    }

    std::optional<StepType> handleBreak(WaveId wave,
                                        const DynamicBlockKey &key,
                                        mlir::Block *block,
                                        mlir::Block::iterator it,
                                        SemanticsContext context,
                                        LaneId lane) {
        auto breakOp = llvm::dyn_cast<simt::dialect::BreakOp>(&*it);
        if (!breakOp)
            return std::nullopt;

        (void)block;
        auto waveIt = state_.waves.find(wave);
        if (waveIt == state_.waves.end())
            llvm::report_fatal_error("handleBreak: missing wave context");
        auto &waveCtx = waveIt->second;
        auto *blockCtx = getBlock(waveCtx, key);
        if (!blockCtx)
            llvm::report_fatal_error("handleBreak: missing block context");
        assert(!(blockCtx->loopOp && blockCtx->switchOp) &&
               "dynamic block cannot have both loopOp and switchOp");
        if ((blockCtx->activeMask & (1ull << lane)) == 0)
            llvm::report_fatal_error("handleBreak: invalid active mask");
        // Drop any parent continuation for the enclosing split so we don't resume
        // the rest of the block after breaking.
        if (blockCtx->parentKey) {
            auto parentIt = waveCtx.blocks.find(*blockCtx->parentKey);
            if (parentIt != waveCtx.blocks.end())
                parentIt->second.continuations.erase(lane);
        }

        // Find nearest enclosing loop (preferred) or switch merge entry that matches this block.
        MergeStackEntry<ValueType, StepType> *entry = nullptr;
        for (auto it = waveCtx.mergeStack.rbegin(); it != waveCtx.mergeStack.rend(); ++it) {
            if (blockCtx->loopOp && it->loopFrame &&
                it->loopFrame->loopOp == blockCtx->loopOp) {
                entry = &*it;
                break;
            }
            if (blockCtx->switchOp && !it->loopFrame) {
                auto parentIt = waveCtx.blocks.find(it->parent);
                if (parentIt != waveCtx.blocks.end() &&
                    parentIt->second.switchOp == blockCtx->switchOp) {
                    entry = &*it;
                    break;
                }
            }
        }
        if (!entry)
            llvm::report_fatal_error("handleBreak: no enclosing merge entry");

        if (entry->loopFrame)
            return handleLoopBreakInternal(wave, key, breakOp, lane, waveCtx, *entry);
        return handleSwitchBreakInternal(wave, key, breakOp, lane, waveCtx, *entry);
    }

    std::optional<StepType> handleSwitchYield(WaveId wave,
                                              const DynamicBlockKey &key,
                                              mlir::Block *block,
                                              mlir::Block::iterator it,
                                              SemanticsContext context,
                                              LaneId lane) {
        auto yieldOp = llvm::dyn_cast<simt::dialect::YieldOp>(&*it);
        if (!yieldOp)
            return std::nullopt;

        (void)block;
        auto waveIt = state_.waves.find(wave);
        if (waveIt == state_.waves.end())
            llvm::report_fatal_error("handleSwitchYield: missing wave context");
        auto &waveCtx = waveIt->second;
        auto *blockCtx = getBlock(waveCtx, key);
        if (!blockCtx || !blockCtx->switchOp)
            return std::nullopt;
        if (blockCtx->kind != DynamicBlockKind::SwitchCase &&
            blockCtx->kind != DynamicBlockKind::SwitchDefault)
            return std::nullopt;
        std::uint64_t laneBit = 1ull << lane;
        if ((blockCtx->activeMask & laneBit) == 0)
            llvm::report_fatal_error("handleSwitchYield: invalid active mask");

        auto *switchOperation = const_cast<mlir::Operation *>(blockCtx->switchOp);
        auto switchOp = llvm::dyn_cast<simt::dialect::SwitchOp>(switchOperation);
        if (!switchOp)
            return std::nullopt;
        auto defaultIndexAttr = switchOp.getDefaultIndexAttr();
        if (!defaultIndexAttr)
            llvm::report_fatal_error("handleSwitchYield: missing default_index attr");
        int64_t defaultIndex = defaultIndexAttr.getInt();
        if (defaultIndex < 0)
            llvm::report_fatal_error("handleSwitchYield: invalid default_index");

        llvm::SmallVector<ValueType, 8> values;
        values.reserve(yieldOp.getNumOperands());
        auto *envPtr =
            blockCtx && blockCtx->valueEnvs.count(lane)
                ? &blockCtx->valueEnvs.find(lane)->second
                : nullptr;
        for (mlir::Value v : yieldOp.getOperands()) {
            if (envPtr) {
                if (auto it = envPtr->find(v); it != envPtr->end()) {
                    values.push_back(it->second);
                    continue;
                }
            }
            auto valOrErr =
                evaluateValue(waveCtx, key, v, lane, blockCtx->activeMask,
                              blockCtx->expectedMask ? blockCtx->expectedMask
                                                     : blockCtx->activeMask);
            if (!valOrErr)
                llvm::report_fatal_error("handleSwitchYield: failed to evaluate yield operand");
            values.push_back(*valOrErr);
        }

        MergeStackEntry<ValueType, StepType> *entry = nullptr;
        for (auto it = waveCtx.mergeStack.rbegin(); it != waveCtx.mergeStack.rend(); ++it) {
            if (!it->loopFrame && it->switchFrame &&
                it->switchFrame->switchOp == blockCtx->switchOp) {
                entry = &*it;
                break;
            }
        }
        if (!entry && blockCtx->parentKey) {
            for (auto it = waveCtx.mergeStack.rbegin();
                 it != waveCtx.mergeStack.rend(); ++it) {
                if (!it->loopFrame && it->parent == *blockCtx->parentKey) {
                    entry = &*it;
                    break;
                }
            }
        }
        if (entry && !entry->switchFrame) {
            SwitchFrameState<ValueType> frame;
            frame.switchOp = switchOp.getOperation();
            frame.baseSeq = entry->parent.sequenceId + 1;
            for (mlir::Block &b : switchOp.getCaseBody())
                frame.caseBlocks.push_back(&b);
            entry->switchFrame = std::move(frame);
            if (entry->pendingChildren.empty()) {
                for (unsigned idx = 0; idx < entry->switchFrame->caseBlocks.size(); ++idx) {
                    entry->pendingChildren.push_back(DynamicBlockKey{
                        entry->switchFrame->caseBlocks[idx],
                        static_cast<std::uint32_t>(entry->switchFrame->baseSeq + idx)});
                    entry->childMasks.push_back(0);
                }
            }
            if (entry->expectedMask == 0)
                entry->expectedMask =
                    blockCtx->expectedMask ? blockCtx->expectedMask : blockCtx->activeMask;
        }
        if (!entry || !entry->switchFrame)
            llvm::report_fatal_error("handleSwitchYield: missing switch frame");
        auto &frame = *entry->switchFrame;
        unsigned numCases = static_cast<unsigned>(frame.caseBlocks.size());
        if (numCases == 0)
            llvm::report_fatal_error("handleSwitchYield: no switch cases");
        if (static_cast<std::size_t>(defaultIndex) >= numCases)
            llvm::report_fatal_error("handleSwitchYield: default_index out of range");
        if (key.sequenceId < frame.baseSeq)
            llvm::report_fatal_error("handleSwitchYield: invalid switch sequence");
        unsigned caseIdx = key.sequenceId - frame.baseSeq;
        if (caseIdx >= numCases)
            llvm::report_fatal_error("handleSwitchYield: case index out of range");
        auto fallthroughAttr = yieldOp->getAttrOfType<mlir::BoolAttr>("fallthrough");
        if (!fallthroughAttr)
            llvm::report_fatal_error("handleSwitchYield: missing fallthrough attr");
        bool fallthrough = fallthroughAttr.getValue();
        bool lastCase = (caseIdx + 1 >= numCases);
        if (EnableCPSDebugLogs) {
            llvm::errs() << "[CPS] handleSwitchYield lane=" << lane
                         << " caseIdx=" << caseIdx
                         << " fallthrough=" << fallthrough
                         << " lastCase=" << lastCase << "\n";
        }
        frame.carried[lane] = values;

        bool switchDoneNow = !fallthrough || lastCase;
        if (switchDoneNow) {
            if (auto pendingIt = frame.pendingCases.find(lane);
                pendingIt != frame.pendingCases.end()) {
                if (auto *pendingBlock = getBlock(waveCtx, pendingIt->second))
                    pendingBlock->continuations.erase(lane);
                frame.pendingCases.erase(pendingIt);
            }
            if (!blockCtx->parentKey)
                llvm::report_fatal_error("handleSwitchYield: missing parent key");
            auto parentIt = waveCtx.blocks.find(*blockCtx->parentKey);
            if (parentIt == waveCtx.blocks.end())
                llvm::report_fatal_error("handleSwitchYield: missing parent block");
            auto &parentEnv = parentIt->second.valueEnvs[lane];
            unsigned idx = 0;
            for (mlir::Value res : switchOp->getResults()) {
                if (idx < values.size())
                    parentEnv[res] = values[idx];
                ++idx;
            }
            blockCtx->activeMask &= ~laneBit;
            blockCtx->completedMask |= laneBit;
            handleReconvergence(wave, waveCtx, key, lane);
            return StepType::halt();
        }

        unsigned nextIdx = caseIdx + 1;
        DynamicBlockKey nextKey{frame.caseBlocks[nextIdx],
                                static_cast<std::uint32_t>(frame.baseSeq + nextIdx)};
        auto &nextCtx = waveCtx.blocks[nextKey];
        nextCtx.block = nextKey.block;
        nextCtx.sequenceId = nextKey.sequenceId;
        nextCtx.parentKey = entry->parent;
        std::uint64_t expected =
            entry->expectedMask ? entry->expectedMask
                                : (blockCtx->expectedMask ? blockCtx->expectedMask
                                                          : blockCtx->activeMask);
        if (nextCtx.expectedMask == 0)
            nextCtx.expectedMask = expected;
        nextCtx.expectedMask |= laneBit;
        nextCtx.activeMask |= laneBit;
        nextCtx.completedMask &= ~laneBit;
        nextCtx.kind = (nextIdx == static_cast<unsigned>(defaultIndex))
                           ? DynamicBlockKind::SwitchDefault
                           : DynamicBlockKind::SwitchCase;
        nextCtx.switchOp = blockCtx->switchOp;
        nextCtx.loopOp = nullptr;
        nextCtx.ifOp = nullptr;

        auto &env = nextCtx.valueEnvs[lane];
        auto nextArgs =
            const_cast<mlir::Block *>(nextKey.block)->getArguments();
        for (auto indexed : llvm::enumerate(nextArgs)) {
            if (indexed.index() < values.size())
                env[indexed.value()] = values[indexed.index()];
        }

        if (auto pendingIt = frame.pendingCases.find(lane);
            pendingIt != frame.pendingCases.end()) {
            if (auto *pendingBlock = getBlock(waveCtx, pendingIt->second))
                pendingBlock->continuations.erase(lane);
        }
        frame.pendingCases[lane] = key;
        blockCtx->continuations[lane] = StepType::halt();

        blockCtx->activeMask &= ~laneBit;
        blockCtx->completedMask |= laneBit;
        waveCtx.lanes[lane].currentBlock = nextKey;

        SemanticsContext laneCtx = context;
        laneCtx.overrideMode.reset();
        laneCtx.activeMask = nextCtx.activeMask;
        laneCtx.expectedMask =
            nextCtx.expectedMask ? nextCtx.expectedMask : nextCtx.activeMask;
        laneCtx.laneId = lane;
        mlir::Block *nextBlock = const_cast<mlir::Block *>(nextKey.block);
        StepType childStep = makeNextOp(wave, nextKey, nextBlock,
                                        nextBlock->begin(), laneCtx, lane);
        enqueue(wave, nextKey, lane, std::move(childStep));
        return StepType::halt();
    }

    std::optional<StepType> handleIfYield(WaveId wave,
                                          const DynamicBlockKey &key,
                                          mlir::Block *block,
                                          mlir::Block::iterator it,
                                          SemanticsContext context,
                                          LaneId lane) {
        auto yieldOp = llvm::dyn_cast<simt::dialect::YieldOp>(&*it);
        if (!yieldOp)
            return std::nullopt;

        (void)block;
        auto waveIt = state_.waves.find(wave);
        if (waveIt == state_.waves.end())
            llvm::report_fatal_error("handleIfYield: missing wave context");
        auto &waveCtx = waveIt->second;
        auto *blockCtx = getBlock(waveCtx, key);
        if (!blockCtx)
            llvm::report_fatal_error("handleIfYield: missing block context");
        assert(!(blockCtx->loopOp && blockCtx->switchOp) &&
               "dynamic block cannot have both loopOp and switchOp");
        if (!blockCtx->parentKey || !blockCtx->ifOp)
            return std::nullopt;
        std::uint64_t laneBit = 1ull << lane;
        if ((blockCtx->activeMask & laneBit) == 0)
            llvm::report_fatal_error("handleIfYield: invalid active mask");

        auto parentIt = waveCtx.blocks.find(*blockCtx->parentKey);
        if (parentIt == waveCtx.blocks.end())
            llvm::report_fatal_error("handleIfYield: missing parent block");
        auto &parentBlock = parentIt->second;
        auto &parentEnv = parentBlock.valueEnvs[lane];

        llvm::SmallVector<ValueType, 4> values;
        values.reserve(yieldOp.getNumOperands());
        auto *envPtr =
            blockCtx && blockCtx->valueEnvs.count(lane)
                ? &blockCtx->valueEnvs.find(lane)->second
                : nullptr;
        for (mlir::Value v : yieldOp.getOperands()) {
            // Try current block env first, then parent env, then full eval.
            if (envPtr) {
                if (auto it = envPtr->find(v); it != envPtr->end()) {
                    values.push_back(it->second);
                    continue;
                }
            }
            if (auto it = parentEnv.find(v); it != parentEnv.end()) {
                values.push_back(it->second);
                continue;
            }
            auto valOrErr =
                evaluateValue(waveCtx, key, v, lane, blockCtx->activeMask,
                              blockCtx->expectedMask ? blockCtx->expectedMask
                                                     : blockCtx->activeMask);
            if (!valOrErr)
                llvm::report_fatal_error("handleIfYield: failed to evaluate yield operand");
            values.push_back(*valOrErr);
        }

        unsigned idx = 0;
        auto *ifOp = const_cast<mlir::Operation *>(blockCtx->ifOp);
        for (mlir::Value res : ifOp->getResults()) {
            if (idx < values.size())
                parentEnv[res] = values[idx];
            ++idx;
        }

        blockCtx->activeMask &= ~laneBit;
        blockCtx->completedMask |= laneBit;

        if (EnableCPSDebugLogs) {
            auto fmt = [&](std::uint64_t m) { return formatMaskBits(m, 32); };
            llvm::errs() << "[CPS] handleIfYield lane=" << lane
                         << " block=" << key.block << " seq=" << key.sequenceId
                         << " parent=" << blockCtx->parentKey->block
                         << " ifOp=" << blockCtx->ifOp
                         << " active=" << fmt(blockCtx->activeMask)
                         << " expected=" << fmt(blockCtx->expectedMask)
                         << "\n";
        }

        handleReconvergence(wave, waveCtx, key, lane);
        return StepType::halt();
    }

    std::optional<StepType> handleLoopBreakInternal(
        WaveId wave, const DynamicBlockKey &key, simt::dialect::BreakOp breakOp,
        LaneId lane, WaveContext<ValueType, StepType> &waveCtx,
        MergeStackEntry<ValueType, StepType> &entry) {
        auto *blockCtx = getBlock(waveCtx, key);
        if (!blockCtx || !blockCtx->loopOp)
            llvm::report_fatal_error("handleLoopBreak: invalid loop block context");
        std::uint64_t laneBit = 1ull << lane;

        llvm::SmallVector<ValueType, 4> results;
        results.reserve(breakOp.getNumOperands());
        for (mlir::Value v : breakOp.getOperands()) {
            auto valOrErr = evaluateValue(
                waveCtx, key, v, lane, blockCtx->activeMask,
                blockCtx->expectedMask ? blockCtx->expectedMask
                                       : blockCtx->activeMask);
            if (!valOrErr) {
                llvm::consumeError(valOrErr.takeError());
                results.push_back(ValueType{});
            } else {
                results.push_back(*valOrErr);
            }
        }
        auto &loopFrame = *entry.loopFrame;
        loopFrame.carried[lane].assign(results.begin(), results.end());

        auto parentIt = waveCtx.blocks.find(entry.parent);
        if (parentIt != waveCtx.blocks.end()) {
            auto &parentEnv = parentIt->second.valueEnvs[lane];
            unsigned idx = 0;
            auto *loopOperation = const_cast<mlir::Operation *>(loopFrame.loopOp);
            for (mlir::Value res : loopOperation->getResults()) {
                if (idx < results.size())
                    parentEnv[res] = results[idx];
                ++idx;
            }
            auto contIt = parentIt->second.continuations.find(lane);
            if (contIt != parentIt->second.continuations.end()) {
                parentIt->second.activeMask |= laneBit;
                state_.readyQueue.push(ReadyContinuation<ValueType, StepType>{
                    wave, entry.parent, lane, contIt->second});
                parentIt->second.continuations.erase(contIt);
            }
        }

        blockCtx->activeMask &= ~laneBit;
        blockCtx->completedMask |= laneBit;
        if (EnableCPSDebugLogs) {
            auto fmt = [&](std::uint64_t m) { return formatMaskBits(m, 32); };
            llvm::errs() << "[CPS] handleLoopBreak lane=" << lane
                         << " block=" << key.block << " seq=" << key.sequenceId
                         << " active=" << fmt(blockCtx->activeMask)
                         << " expected=" << fmt(blockCtx->expectedMask)
                         << "\n";
        }
        shrinkExpectedForLane(wave, waveCtx, lane);
        handleReconvergence(wave, waveCtx, key, lane);
        return StepType::halt();
    }

    std::optional<StepType> handleSwitchBreakInternal(
        WaveId wave, const DynamicBlockKey &key, simt::dialect::BreakOp breakOp,
        LaneId lane, WaveContext<ValueType, StepType> &waveCtx,
        MergeStackEntry<ValueType, StepType> &entry) {
        auto *blockCtx = getBlock(waveCtx, key);
        if (!blockCtx || !blockCtx->switchOp)
            llvm::report_fatal_error("handleSwitchBreak: invalid switch block context");
        std::uint64_t laneBit = 1ull << lane;

        llvm::SmallVector<ValueType, 4> results;
        results.reserve(breakOp.getNumOperands());
        for (mlir::Value v : breakOp.getOperands()) {
            auto valOrErr = evaluateValue(
                waveCtx, key, v, lane, blockCtx->activeMask,
                blockCtx->expectedMask ? blockCtx->expectedMask
                                       : blockCtx->activeMask);
            if (!valOrErr) {
                llvm::consumeError(valOrErr.takeError());
                results.push_back(ValueType{});
            } else {
                results.push_back(*valOrErr);
            }
        }

        auto parentIt = waveCtx.blocks.find(entry.parent);
        if (parentIt != waveCtx.blocks.end()) {
            auto &parentEnv = parentIt->second.valueEnvs[lane];
            unsigned idx = 0;
            auto *switchOperation = const_cast<mlir::Operation *>(blockCtx->switchOp);
            for (mlir::Value res : switchOperation->getResults()) {
                if (idx < results.size())
                    parentEnv[res] = results[idx];
                ++idx;
            }
            auto contIt = parentIt->second.continuations.find(lane);
            if (contIt != parentIt->second.continuations.end()) {
                parentIt->second.activeMask |= laneBit;
                state_.readyQueue.push(ReadyContinuation<ValueType, StepType>{
                    wave, entry.parent, lane, contIt->second});
                parentIt->second.continuations.erase(contIt);
            }
        }

        blockCtx->activeMask &= ~laneBit;
        blockCtx->completedMask |= laneBit;
        if (EnableCPSDebugLogs) {
            auto fmt = [&](std::uint64_t m) { return formatMaskBits(m, 32); };
            llvm::errs() << "[CPS] handleSwitchBreak lane=" << lane
                         << " block=" << key.block << " seq=" << key.sequenceId
                         << " active=" << fmt(blockCtx->activeMask)
                         << " expected=" << fmt(blockCtx->expectedMask)
                         << "\n";
        }
        shrinkExpectedForLane(wave, waveCtx, lane);
        handleReconvergence(wave, waveCtx, key, lane);
        return StepType::halt();
    }

    std::optional<StepType> handleIfSplit(WaveId wave,
                                          const DynamicBlockKey &key,
                                          mlir::Block *block,
                                          mlir::Block::iterator it,
                                          SemanticsContext context,
                                          LaneId lane,
                                          const IfDecisionMap *decisions = nullptr) {
        auto ifOp = llvm::dyn_cast<simt::dialect::IfOp>(&*it);
        if (!ifOp)
            return std::nullopt;

        auto waveIt = state_.waves.find(wave);
        if (waveIt == state_.waves.end())
            llvm::report_fatal_error("handleIfSplit: missing wave context");
        auto &waveCtx = waveIt->second;
        auto parentBlockIt = waveCtx.blocks.find(key);
        if (parentBlockIt == waveCtx.blocks.end())
            llvm::report_fatal_error("handleIfSplit: missing parent block");
        auto &parentBlock = parentBlockIt->second;
        if ((parentBlock.activeMask & (1ull << lane)) == 0)
            return StepType::halt();

        if (auto gated = gateControlFlowOp(wave, key, block, it, context, lane))
            return gated;
        if (traceSink_ && !context.suppressStepTrace) {
            std::uint64_t expectedMask =
                context.expectedMask ? context.expectedMask : context.activeMask;
            if (expectedMask == 0)
                expectedMask = (1ull << lane);
            traceSink_->onStepBegin(
                wave, lane, it->getName().getStringRef().str(),
                context.activeMask, expectedMask, key.sequenceId, key.block,
                blockKindLabel(parentBlock.kind), parentBlock.loopIteration);
        }

        auto nextIt = std::next(it);
        SemanticsContext parentContext = context;
        parentContext.overrideMode.reset();
        parentContext.suppressStepTrace = false;
        StepType parentCont = StepType::continueWith(
            [this, wave, key, block, nextIt, parentContext, lane]() mutable
            -> StepType {
                return makeNextOp(wave, key, block, nextIt, parentContext, lane);
            });
        // Store for later reconvergence; do not enqueue until the lane returns.
        parentBlock.continuations[lane] = parentCont;

        std::uint64_t parentExpected =
            parentBlock.expectedMask;

        // Evaluate predicate only for this lane.
        std::uint64_t evalActive =
            context.activeMask ? context.activeMask : parentBlock.activeMask;
        std::uint64_t evalExpected =
            context.expectedMask
                ? context.expectedMask
                : (parentBlock.expectedMask ? parentBlock.expectedMask
                                            : parentBlock.activeMask);
        bool takeThen = false;
        bool takeElse = false;
        if (decisions) {
            auto decisionIt = decisions->find(lane);
            if (decisionIt == decisions->end())
                llvm::report_fatal_error(
                    "handleIfSplit: missing predicate decision");
            takeThen = decisionIt->second;
        } else {
            auto condOrErr =
                evaluateBool(waveCtx, key, ifOp.getCondition(), lane,
                             evalActive, evalExpected);
            if (condOrErr) {
                takeThen = *condOrErr;
            } else {
                llvm::consumeError(condOrErr.takeError());
            }
        }
        if (!takeThen && !ifOp.getElseRegion().empty())
            takeElse = true;

        auto isDynamicDescendant = [&](const DynamicBlockKey &desc,
                                       const DynamicBlockKey &ancestor) {
            DynamicBlockKey cur = desc;
            while (true) {
                if (cur == ancestor)
                    return true;
                auto it = waveCtx.blocks.find(cur);
                if (it == waveCtx.blocks.end() || !it->second.parentKey)
                    return false;
                cur = *it->second.parentKey;
            }
        };

        if (EnableCPSDebugLogs) {
            auto fmt = [&](std::uint64_t m) { return formatMaskBits(m, 32); };
            llvm::errs() << "[CPS] handleIfSplit lane=" << lane
                         << " parent=" << key.block << " seq=" << key.sequenceId
                         << " takeThen=" << takeThen << " takeElse=" << takeElse
                         << " active=0b" << fmt(parentBlock.activeMask)
                         << " expected=0b" << fmt(parentBlock.expectedMask)
                         << "\n";
        }

        // auto makeChildKey = [&](mlir::Block *b, std::uint32_t seq) {
        //     return DynamicBlockKey{b, seq};
        // };
        std::uint32_t baseSeq = key.sequenceId + 1;

        DynamicBlockKey thenKey{&ifOp.getThenRegion().front(), baseSeq};
        DynamicBlockKey elseKey{&ifOp.getElseRegion().front(), baseSeq + 1};
        
        auto findMergeEntry = [&](WaveContext<ValueType, StepType> &ctx)
            -> MergeStackEntry<ValueType, StepType> * {
            for (auto it = ctx.mergeStack.rbegin(); it != ctx.mergeStack.rend(); ++it) {
                if (!it->loopFrame && it->parent == key)
                    return &*it;
            }
            return nullptr;
        };

        MergeStackEntry<ValueType, StepType> *entry = findMergeEntry(waveCtx);
        if (!entry) {
            MergeStackEntry<ValueType, StepType> newEntry;
            newEntry.parent = key;
            waveCtx.mergeStack.push_back(std::move(newEntry));
            entry = &waveCtx.mergeStack.back();
            entry->expectedMask = parentExpected;
            if (EnableCPSDebugLogs) {
                llvm::errs() << "[CPS] push merge (if) parent=" << key.block
                             << " seq=" << key.sequenceId << "\n";
                logMergeStackState<ValueType, StepType>(waveCtx);
            }
        }
        // Start from a clean slate for this lane; add it back only to the taken path.
        // entry->expectedMask &= ~laneBit;

        if (takeThen) {
        auto &child = waveCtx.blocks[thenKey];
        child.block = thenKey.block;
        child.sequenceId = thenKey.sequenceId;
        child.parentKey = key;
        child.ifOp = ifOp.getOperation();
        child.loopOp = parentBlock.loopOp;
        child.switchOp = parentBlock.switchOp;
        std::uint64_t laneMask =
            parentExpected ? (parentExpected & (1ull << lane)) : (1ull << lane);
        if (child.expectedMask == 0)
            child.expectedMask = parentExpected ? parentExpected : (1ull << lane);
        child.expectedMask |= laneMask;
        child.activeMask |= (1ull << lane);
        // child.completedMask &= ~(1ull << lane);
        child.kind = DynamicBlockKind::IfThen;
        if (auto envIt = parentBlock.valueEnvs.find(lane);
            envIt != parentBlock.valueEnvs.end()) {
            child.valueEnvs[lane] = envIt->second;
        }
            // Ensure sibling exists so we can clear this lane from its expected set.
        auto &elseCtx = waveCtx.blocks[elseKey];
        elseCtx.block = elseKey.block;
        elseCtx.sequenceId = elseKey.sequenceId;
        elseCtx.parentKey = key;
        elseCtx.ifOp = ifOp.getOperation();
        elseCtx.loopOp = parentBlock.loopOp;
        elseCtx.switchOp = parentBlock.switchOp;
        elseCtx.kind = DynamicBlockKind::IfElse;
        if (elseCtx.expectedMask == 0)
            elseCtx.expectedMask = parentExpected ? parentExpected : laneMask;
        elseCtx.expectedMask &= ~laneMask;
            // Propagate the exclusion into any existing descendant dynamic blocks of the
            // else branch, including other sequenceIds (e.g., loop iterations).
            for (auto &kv : waveCtx.blocks) {
                const auto &descKey = kv.first;
                auto &desc = kv.second;
                if (isDynamicDescendant(descKey, elseKey))
                    desc.expectedMask &= ~laneMask;
            }

            if (!llvm::is_contained(entry->pendingChildren, thenKey)) {
                entry->pendingChildren.push_back(thenKey);
                entry->childMasks.push_back(child.activeMask);
            }
            // entry->expectedMask |= laneMask;

            if (EnableCPSDebugLogs) {
                llvm::errs() << "[CPS] handleIfSplit lane=" << lane
                             << " -> then block=" << thenKey.block
                             << " seq=" << thenKey.sequenceId
                             << " parent=" << key.block
                             << " parentSeq=" << key.sequenceId
                             << "\n";
            }

            SemanticsContext laneCtx = context;
            laneCtx.overrideMode.reset();
            laneCtx.suppressStepTrace = false;
            laneCtx.activeMask = child.activeMask;
            laneCtx.expectedMask =
                child.expectedMask ? child.expectedMask : child.activeMask;
            laneCtx.laneId = lane;
            mlir::Block *childBlock = const_cast<mlir::Block *>(thenKey.block);
            StepType childStep = makeNextOp(wave, thenKey, childBlock,
                                            childBlock->begin(), laneCtx, lane);
            enqueue(wave, thenKey, lane, std::move(childStep));
            parentBlock.activeMask &= ~(1ull << lane);
            logMergeStackState<ValueType, StepType>(waveCtx);
            return StepType::halt();
        }

        if (takeElse) {
        auto &child = waveCtx.blocks[elseKey];
        child.block = elseKey.block;
        child.sequenceId = elseKey.sequenceId;
        child.parentKey = key;
        child.ifOp = ifOp.getOperation();
        child.loopOp = parentBlock.loopOp;
        child.switchOp = parentBlock.switchOp;
        std::uint64_t laneMask =
            parentExpected ? (parentExpected & (1ull << lane)) : (1ull << lane);
        if (child.expectedMask == 0)
            child.expectedMask = parentExpected ? parentExpected : (1ull << lane);
        child.expectedMask |= laneMask;
        child.activeMask |= (1ull << lane);
        // child.completedMask &= ~(1ull << lane);
        child.kind = DynamicBlockKind::IfElse;
        if (auto envIt = parentBlock.valueEnvs.find(lane);
            envIt != parentBlock.valueEnvs.end()) {
            child.valueEnvs[lane] = envIt->second;
        }
        auto &thenCtx = waveCtx.blocks[thenKey];
        thenCtx.block = thenKey.block;
        thenCtx.sequenceId = thenKey.sequenceId;
        thenCtx.parentKey = key;
        thenCtx.ifOp = ifOp.getOperation();
        thenCtx.loopOp = parentBlock.loopOp;
        thenCtx.switchOp = parentBlock.switchOp;
        thenCtx.kind = DynamicBlockKind::IfThen;
        if (thenCtx.expectedMask == 0)
            thenCtx.expectedMask = parentExpected ? parentExpected : laneMask;
        thenCtx.expectedMask &= ~laneMask;
            for (auto &kv : waveCtx.blocks) {
                const auto &descKey = kv.first;
                auto &desc = kv.second;
                if (isDynamicDescendant(descKey, thenKey))
                    desc.expectedMask &= ~laneMask;
            }

            if (!llvm::is_contained(entry->pendingChildren, elseKey)) {
                entry->pendingChildren.push_back(elseKey);
                entry->childMasks.push_back(child.activeMask);
            }
            // entry->expectedMask |= laneMask;

            if (EnableCPSDebugLogs) {
                llvm::errs() << "[CPS] handleIfSplit lane=" << lane
                             << " -> else block=" << elseKey.block
                             << " seq=" << elseKey.sequenceId
                             << " parent=" << key.block
                             << " parentSeq=" << key.sequenceId
                             << "\n";
            }

            SemanticsContext laneCtx = context;
            laneCtx.overrideMode.reset();
            laneCtx.suppressStepTrace = false;
            laneCtx.activeMask = child.activeMask;
            laneCtx.expectedMask =
                child.expectedMask ? child.expectedMask : child.activeMask;
            laneCtx.laneId = lane;
            mlir::Block *childBlock = const_cast<mlir::Block *>(elseKey.block);
            StepType childStep = makeNextOp(wave, elseKey, childBlock,
                                            childBlock->begin(), laneCtx, lane);
            enqueue(wave, elseKey, lane, std::move(childStep));
            parentBlock.activeMask &= ~(1ull << lane);
                logMergeStackState<ValueType, StepType>(waveCtx);
            return StepType::halt();
        }

        // No else region and condition false: just resume parent continuation.
        auto contIt = parentBlock.continuations.find(lane);
        if (contIt != parentBlock.continuations.end()) {
            parentBlock.activeMask |= (1ull << lane);
            waveCtx.lanes[lane].currentBlock = key;
            state_.readyQueue.push(
                ReadyContinuation<ValueType, StepType>{wave, key, lane, contIt->second});
            parentBlock.continuations.erase(contIt);
        }
        return StepType::halt();
    }

    std::optional<StepType> handleCallOp(WaveId wave,
                                         const DynamicBlockKey &key,
                                         mlir::Block *block,
                                         mlir::Block::iterator it,
                                         SemanticsContext context,
                                         LaneId lane) {
        auto callOp = llvm::dyn_cast<mlir::func::CallOp>(&*it);
        if (!callOp)
            return std::nullopt;

        auto waveIt = state_.waves.find(wave);
        if (waveIt == state_.waves.end())
            llvm::report_fatal_error("call: missing wave context");
        auto &waveCtx = waveIt->second;
        auto &laneCtx = waveCtx.lanes[lane];
        auto *callerBlockCtx = getBlock(waveCtx, key);
        if (!callerBlockCtx)
            llvm::report_fatal_error("call: missing caller block");

        auto calleeAttr = callOp.getCalleeAttr();
        if (!calleeAttr)
            llvm::report_fatal_error("call: missing callee symbol");
        auto calleeOp =
            mlir::dyn_cast_or_null<mlir::func::FuncOp>(
                mlir::SymbolTable::lookupNearestSymbolFrom(callOp, calleeAttr));
        if (!calleeOp)
            llvm::report_fatal_error("call: unresolved callee");
        if (calleeOp.isExternal())
            llvm::report_fatal_error("call: external callee unsupported");

        auto currentFunc = callOp->getParentOfType<mlir::func::FuncOp>();
        if (!currentFunc)
            llvm::report_fatal_error("call: missing parent function");
        if (calleeOp == currentFunc)
            llvm::report_fatal_error("call: recursion unsupported");
        for (const auto &frame : laneCtx.callStack) {
            if (frame.calleeName == calleeOp.getName().str())
                llvm::report_fatal_error("call: recursion unsupported");
        }

        if (callOp.getNumResults() > 1)
            llvm::report_fatal_error("call: multiple results unsupported");
        if (callOp.getNumOperands() != calleeOp.getNumArguments())
            llvm::report_fatal_error("call: argument count mismatch");

        llvm::SmallVector<ValueType, 4> argValues;
        argValues.reserve(callOp.getNumOperands());
        for (auto arg : callOp.getOperands()) {
            auto valOrErr = evaluateValue(
                waveCtx, key, arg, lane, context.activeMask, context.expectedMask);
            if (!valOrErr)
                llvm::report_fatal_error("call: argument evaluation failed");
            argValues.push_back(*valOrErr);
        }

        DynamicBlockKey calleeKey;
        auto callChildIt = callerBlockCtx->callChildren.find(callOp.getOperation());
        if (callChildIt != callerBlockCtx->callChildren.end()) {
            calleeKey = callChildIt->second;
        } else {
            calleeKey = DynamicBlockKey{&calleeOp.getBody().front(),
                                        waveCtx.nextCallSeq++};
            callerBlockCtx->callChildren[callOp.getOperation()] = calleeKey;
        }

        auto &calleeBlockCtx = waveCtx.blocks[calleeKey];
        calleeBlockCtx.block = calleeKey.block;
        calleeBlockCtx.sequenceId = calleeKey.sequenceId;
        calleeBlockCtx.kind = DynamicBlockKind::Plain;
        std::uint64_t expected =
            context.expectedMask ? context.expectedMask : context.activeMask;
        if (calleeBlockCtx.expectedMask == 0)
            calleeBlockCtx.expectedMask = expected ? expected : (1ull << lane);
        calleeBlockCtx.activeMask |= (1ull << lane);
        calleeBlockCtx.completedMask &= ~(1ull << lane);

        auto &env = calleeBlockCtx.valueEnvs[lane];
        env.clear();
        auto &entryBlock = calleeOp.getBody().front();
        for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i)
            env[entryBlock.getArgument(i)] = argValues[i];

        callerBlockCtx->activeMask &= ~(1ull << lane);

        CallFrame<ValueType> frame;
        frame.callerKey = key;
        frame.callerBlock = block;
        frame.resumeIt = std::next(it);
        frame.results.assign(callOp.getResults().begin(), callOp.getResults().end());
        frame.calleeName = calleeOp.getName().str();
        laneCtx.callStack.push_back(std::move(frame));
        laneCtx.currentBlock = calleeKey;

        SemanticsContext calleeContext = context;
        calleeContext.overrideMode.reset();
        return StepType::continueWith(
            [this, wave, calleeKey, calleeBlock = const_cast<mlir::Block *>(
                                             calleeKey.block),
             calleeContext, lane]() mutable -> StepType {
                return makeNextOp(wave, calleeKey, calleeBlock,
                                  calleeBlock->begin(), calleeContext, lane);
            });
    }

    /// Evaluate an SSA value to a SemValue for a given lane in a block.
    llvm::Expected<ValueType> evaluateValue(WaveContext<ValueType, StepType> &waveCtx,
                                            const DynamicBlockKey &blockKey,
                                            mlir::Value value,
                                            LaneId lane,
                                            std::uint64_t activeMask,
                                            std::uint64_t expectedMask) {
        SemanticsContext ctx;
        ctx.laneId = lane;
        ctx.activeMask = activeMask;
        ctx.expectedMask = expectedMask;
        if (auto *blockCtx = getBlock(waveCtx, blockKey)) {
            auto envIt = blockCtx->valueEnvs.find(lane);
            if (envIt != blockCtx->valueEnvs.end())
                ctx.valueEnv = &envIt->second;
        }
        // If the value has a defining op, ask the semantics to evaluate it.
        if (auto *defOp = value.getDefiningOp()) {
            StepType step = adaptor_.eval(semantics_, defOp, ctx);
            if (!step.isProduce())
                return llvm::make_error<llvm::StringError>(
                    "value evaluation did not produce",
                    llvm::inconvertibleErrorCode());
            auto state = std::move(step).takeState();
            return std::get<typename StepType::Produce>(std::move(state)).value;
        }
        // Block arguments should be present in the value environment.
        if (ctx.valueEnv) {
            auto it = ctx.valueEnv->find(value);
            if (it != ctx.valueEnv->end())
                return it->second;
        }
        return llvm::make_error<llvm::StringError>(
            "unsupported SSA value in interpreter evaluateValue",
            llvm::inconvertibleErrorCode());
    }

    /// Evaluate a boolean SSA value for a given lane.
    llvm::Expected<bool> evaluateBool(WaveContext<ValueType, StepType> &waveCtx,
                                      const DynamicBlockKey &blockKey,
                                      mlir::Value value,
                                      LaneId lane,
                                      std::uint64_t activeMask,
                                      std::uint64_t expectedMask) {
        auto valOrErr =
            evaluateValue(waveCtx, blockKey, value, lane, activeMask, expectedMask);
        if (!valOrErr)
            return valOrErr.takeError();
        return valOrErr->asBool();
    }

    llvm::Error processReady(ReadyContinuation<ValueType, StepType> item) {
        ensureWaveBlock(item.wave, item.block, item.lane);
        auto &waveCtx = state_.waves[item.wave];
        auto &laneCtx = waveCtx.lanes[item.lane];
        laneCtx.currentBlock = item.block;
        if (auto *blockCtx = getBlock(waveCtx, item.block)) {
            waveCtx.currentMask = blockCtx->activeMask;
        }
        if (EnableCPSDebugLogs) {
            llvm::errs() << "[CPS] run lane=" << item.lane
                         << " block=" << item.block.block
                         << " seq=" << item.block.sequenceId << "\n";
        }
        StepType current = std::move(item.resume);
        for (;;) {
            typename StepType::State stateVariant = std::move(current).takeState();

            if (std::holds_alternative<typename StepType::Continue>(stateVariant)) {
                if (EnableCPSDebugLogs) {
                    llvm::errs() << "[CPS] state=Continue lane=" << item.lane
                                 << " block=" << item.block.block
                                 << " seq=" << item.block.sequenceId << "\n";
                }
                auto cont =
                    std::get<typename StepType::Continue>(std::move(stateVariant));
                if (!cont.next) {
                    return llvm::make_error<llvm::StringError>(
                        "continuation missing resume function",
                        llvm::inconvertibleErrorCode());
                }
                current = cont.next();
                continue;
            }

            if (std::holds_alternative<typename StepType::Produce>(stateVariant)) {
                auto prod =
                    std::get<typename StepType::Produce>(std::move(stateVariant));
                if (EnableCPSDebugLogs) {
                    llvm::errs() << "[CPS] state=Produce lane=" << item.lane
                                 << " block=" << item.block.block
                                 << " seq=" << item.block.sequenceId << "\n";
                }
                bool terminal = laneCtx.phase ==
                                LaneContext<ValueType, StepType>::Phase::Completed;
                laneCtx.hasReturned = laneCtx.hasReturned || terminal;
                laneCtx.returnValue = std::move(prod.value);
                if (auto *blockCtx = getBlock(waveCtx, item.block)) {
                    std::uint64_t laneBit = 1ull << item.lane;
                    blockCtx->activeMask &= ~laneBit;
                    blockCtx->completedMask |= laneBit;
                    if (terminal)
                        shrinkExpectedForLane(item.wave, waveCtx, item.lane);
                    // if (!terminal) {
                    //     // Resume parent execution for this lane.
                    //     handleReconvergence(item.wave, waveCtx, item.block, item.lane);
                    // }
                }
                return llvm::Error::success();
            }

            if (std::holds_alternative<typename StepType::Halt>(stateVariant)) {
                if (EnableCPSDebugLogs) {
                    llvm::errs() << "[CPS] state=Halt lane=" << item.lane
                                 << " block=" << item.block.block
                                 << " seq=" << item.block.sequenceId
                                 << " (continuation exhausted)\n";
                }
                if (auto *blockCtx = getBlock(waveCtx, item.block)) {
                    if (blockCtx->loopOp) {
                        // Let loop handlers drive reconvergence and parent resumption;
                        // don't treat this as end-of-function for the lane.
                        std::uint64_t laneBit = 1ull << item.lane;
                        blockCtx->activeMask &= ~laneBit;
                        blockCtx->completedMask |= laneBit;
                        return llvm::Error::success();
                    }
                    if (blockCtx->continuations.contains(item.lane)) {
                        // Control-split placeholder; the continuation will resume later.
                        return llvm::Error::success();
                    }
                    bool terminal = laneCtx.phase ==
                                    LaneContext<ValueType, StepType>::Phase::Completed;
                    laneCtx.hasReturned = laneCtx.hasReturned || terminal;
                    std::uint64_t laneBit = 1ull << item.lane;
                    blockCtx->activeMask &= ~laneBit;
                    blockCtx->completedMask |= laneBit;
                    if (terminal)
                        shrinkExpectedForLane(item.wave, waveCtx, item.lane);
                    // Account for completion and allow reconvergence unless this was
                    // a terminal return for the lane.
                    markMergeCompletion(item.wave, waveCtx, item.block, item.lane);
                    if (!terminal) {
                        handleReconvergence(item.wave, waveCtx, item.block, item.lane);
                    }
                }
                return llvm::Error::success();
            }

            if (std::holds_alternative<typename StepType::Suspend>(stateVariant)) {
                auto susp =
                    std::get<typename StepType::Suspend>(std::move(stateVariant));
                return handleSuspend(item.wave, item.block, item.lane,
                                     std::move(susp));
            }

            return llvm::make_error<llvm::StringError>(
                "unknown step state encountered in CPS interpreter",
                llvm::inconvertibleErrorCode());
        }
    }

    llvm::Error handleSuspend(WaveId wave, const DynamicBlockKey &block,
                              LaneId lane,
                              typename StepType::Suspend &&suspend) {
        if (suspend.effect.template isa<YieldEffect>()) {
            if (!suspend.resume) {
                return llvm::make_error<llvm::StringError>(
                    "yield effect missing resume continuation",
                    llvm::inconvertibleErrorCode());
            }
            StepType resumed = suspend.resume();
            enqueue(wave, block, lane, std::move(resumed));
            return llvm::Error::success();
        }

        if (auto *collective =
                suspend.effect.template get_if<CollectiveEffect>()) {
            auto &waveCtx = state_.waves[wave];
            auto *blockCtx = getBlock(waveCtx, block);
            if (!blockCtx) {
                return llvm::make_error<llvm::StringError>(
                    "collective effect missing dynamic block context",
                    llvm::inconvertibleErrorCode());
            }
            std::uint32_t key =
                collective->token.value_or(collective->operation);
            bool isControlFlow =
                waveCtx.controlTokenToOp.find(key) != waveCtx.controlTokenToOp.end();
            const mlir::Operation *waveOp = nullptr;
            auto waveIt = waveCtx.collectiveTokenToOp.find(key);
            if (waveIt != waveCtx.collectiveTokenToOp.end())
                waveOp = waveIt->second;
            bool isWaveCollective =
                waveOp && isWaveOp(const_cast<mlir::Operation *>(waveOp));
            bool isMemoryCollective =
                waveOp && isMemoryOp(const_cast<mlir::Operation *>(waveOp));
            auto &syncPoint = waveCtx.collectives[key];
            syncPoint.effect = *collective;
            syncPoint.block = block;
            if (syncPoint.expectedMask == 0) {
                std::uint64_t fallbackMask =
                    blockCtx->expectedMask ? blockCtx->expectedMask : blockCtx->activeMask;
                syncPoint.expectedMask = collective->activeMask
                                             ? collective->activeMask
                                             : fallbackMask;
            }
            syncPoint.arrivals.insert(lane);
            if (!isControlFlow) {
                syncPoint.continuations[lane] =
                    StepType::continueWith(
                        [resume = std::move(suspend.resume)]() mutable -> StepType {
                            return resume();
                        });
            }

            auto expectedCount =
                static_cast<unsigned>(std::popcount(syncPoint.expectedMask));
            if (syncPoint.arrivals.size() == expectedCount) {
                if (isControlFlow) {
                    auto controlIt = waveCtx.controlTokenToOp.find(key);
                    if (controlIt != waveCtx.controlTokenToOp.end()) {
                        mlir::Operation *controlOp =
                            const_cast<mlir::Operation *>(controlIt->second);
                        std::uint64_t expectedMask = syncPoint.expectedMask;
                        DynamicBlockKey controlBlock = syncPoint.block;
                        if (traceSink_) {
                            std::string opName =
                                controlOp->getName().getStringRef().str();
                            traceSink_->onCollectiveComplete(
                                wave, opName, expectedMask, expectedMask,
                                controlBlock.sequenceId, controlBlock.block,
                                blockKindLabel(blockCtx->kind),
                                blockCtx->loopIteration);
                        }
                        waveCtx.controlTokenToOp.erase(controlIt);
                        waveCtx.collectives.erase(key);
                        handleControlFlowCollective(wave, controlBlock, controlOp,
                                                    expectedMask);
                        return llvm::Error::success();
                    }
                }
                bool memoryProducesResults =
                    isMemoryCollective && !syncPoint.results.empty();
                if (isWaveCollective && syncPoint.results.empty())
                    computeWaveCollectiveResults(waveOp, syncPoint);
                if (isMemoryCollective && !memoryProducesResults)
                    memoryProducesResults =
                        computeMemoryCollectiveResults(waveOp, syncPoint);
                if (isWaveCollective && traceSink_) {
                    std::string opName;
                    if (waveOp)
                        opName = const_cast<mlir::Operation *>(waveOp)
                                     ->getName()
                                     .getStringRef()
                                     .str();
                    traceSink_->onCollectiveComplete(
                        wave,
                        opName,
                        syncPoint.expectedMask, syncPoint.expectedMask,
                        block.sequenceId, block.block,
                        blockKindLabel(blockCtx->kind),
                        blockCtx->loopIteration);
                }
                if (isMemoryCollective && traceSink_) {
                    std::string opName;
                    if (waveOp)
                        opName = const_cast<mlir::Operation *>(waveOp)
                                     ->getName()
                                     .getStringRef()
                                     .str();
                    traceSink_->onCollectiveComplete(
                        wave,
                        opName,
                        syncPoint.expectedMask, syncPoint.expectedMask,
                        block.sequenceId, block.block,
                        blockKindLabel(blockCtx->kind),
                        blockCtx->loopIteration);
                }
                std::uint64_t mask = syncPoint.expectedMask;
                while (mask) {
                    unsigned l = std::countr_zero(mask);
                    mask &= mask - 1;
                    auto contIt = syncPoint.continuations.find(l);
                    if (contIt != syncPoint.continuations.end()) {
                        blockCtx->activeMask |= (1ull << l);
                        state_.readyQueue.push(
                            ReadyContinuation<ValueType, StepType>{wave, block, l,
                                                                   contIt->second});
                    }
                }
                if (!isWaveCollective && !memoryProducesResults)
                    waveCtx.collectives.erase(key);
                if (isMemoryCollective && !memoryProducesResults)
                    waveCtx.collectiveTokenToOp.erase(key);
            }
            return llvm::Error::success();
        }

        if (auto *sync =
                suspend.effect.template get_if<SynchronizationEffect>()) {
            auto &waveCtx = state_.waves[wave];
            auto *blockCtx = getBlock(waveCtx, block);
            if (!blockCtx) {
                return llvm::make_error<llvm::StringError>(
                    "synchronization effect missing dynamic block context",
                    llvm::inconvertibleErrorCode());
            }
            std::uint32_t key = sync->token.value_or(sync->operation);
            auto &syncPoint = waveCtx.syncPoints[key];
            syncPoint.effect = *sync;
            syncPoint.block = block;
            if (syncPoint.expectedMask == 0) {
                std::uint64_t fallbackMask =
                    blockCtx->expectedMask ? blockCtx->expectedMask : blockCtx->activeMask;
                syncPoint.expectedMask =
                    sync->activeMask ? sync->activeMask : fallbackMask;
            }
            syncPoint.arrivals.insert(lane);
            syncPoint.continuations[lane] =
                StepType::continueWith(
                    [resume = std::move(suspend.resume)]() mutable -> StepType {
                        return resume();
                    });

            auto expectedCount =
                static_cast<unsigned>(std::popcount(syncPoint.expectedMask));
            if (syncPoint.arrivals.size() == expectedCount) {
                const mlir::Operation *controlOp = nullptr;
                auto tokenIt = waveCtx.syncTokenToOp.find(key);
                if (tokenIt != waveCtx.syncTokenToOp.end()) {
                    controlOp = tokenIt->second;
                    waveCtx.syncTokenToOp.erase(tokenIt);
                    if (blockCtx)
                        blockCtx->controlReadyMask[controlOp] |=
                            syncPoint.expectedMask;
                }
                std::uint64_t mask = syncPoint.expectedMask;
                while (mask) {
                    unsigned l = std::countr_zero(mask);
                    mask &= mask - 1;
                    auto contIt = syncPoint.continuations.find(l);
                    if (contIt != syncPoint.continuations.end()) {
                        blockCtx->activeMask |= (1ull << l);
                        if (traceSink_ && controlOp) {
                            traceSink_->onResume(
                                wave, l, syncPoint.expectedMask,
                                syncPoint.expectedMask, block.sequenceId,
                                block.block,
                                blockKindLabel(blockCtx->kind),
                                blockCtx->loopIteration);
                        }
                        state_.readyQueue.push(
                            ReadyContinuation<ValueType, StepType>{wave, block, l,
                                                                   contIt->second});
                    }
                }
                waveCtx.syncPoints.erase(key);
            }
            return llvm::Error::success();
        }

        return llvm::make_error<llvm::StringError>(
            "encountered suspend with unsupported effect",
            llvm::inconvertibleErrorCode());
    }

    void ensureWaveBlock(WaveId wave, const DynamicBlockKey &block, LaneId lane) {
        auto &waveCtx = state_.waves[wave];
        auto [blockIt, inserted] = waveCtx.blocks.try_emplace(block);
        if (inserted) {
            blockIt->second.activeMask = 0;
        }
        waveCtx.lanes.try_emplace(lane);
    }

    static DynamicBlock<ValueType, StepType> *
    getBlock(WaveContext<ValueType, StepType> &waveCtx,
             const DynamicBlockKey &key) {
        auto it = waveCtx.blocks.find(key);
        return it == waveCtx.blocks.end() ? nullptr : &it->second;
    }

    void shrinkExpectedForLane(WaveId waveId,
                               WaveContext<ValueType, StepType> &waveCtx,
                               LaneId lane) {
        // Clear from dynamic blocks.
        for (auto &entry : waveCtx.blocks) {
            entry.second.expectedMask &= ~(1ull << lane);
        }
        // Clear from collectives and release if now satisfied.
        for (auto it = waveCtx.collectives.begin();
             it != waveCtx.collectives.end();) {
            it->second.expectedMask &= ~(1ull << lane);
            it->second.arrivals.erase(lane);
            it->second.continuations.erase(lane);
            it->second.operands.erase(lane);
            it->second.results.erase(lane);
            it->second.memoryIndices.erase(lane);
            it->second.memoryValues.erase(lane);
            const mlir::Operation *waveOp = nullptr;
            auto waveIt = waveCtx.collectiveTokenToOp.find(it->first);
            if (waveIt != waveCtx.collectiveTokenToOp.end())
                waveOp = waveIt->second;
            bool isWaveCollective =
                waveOp && isWaveOp(const_cast<mlir::Operation *>(waveOp));
            bool isMemoryCollective =
                waveOp && isMemoryOp(const_cast<mlir::Operation *>(waveOp));
            bool ready = it->second.expectedMask &&
                         it->second.arrivals.size() ==
                             static_cast<unsigned>(
                                 std::popcount(it->second.expectedMask));
            if (ready) {
                auto *blockCtx = getBlock(waveCtx, it->second.block);
                bool scheduleNow = true;
                bool emitCollective = false;
                bool memoryHasResults = false;
                if (isWaveCollective) {
                    if (it->second.results.empty()) {
                        computeWaveCollectiveResults(waveOp, it->second);
                        emitCollective = true;
                    } else {
                        scheduleNow = false;
                    }
                } else if (isMemoryCollective) {
                    if (it->second.results.empty()) {
                        memoryHasResults =
                            computeMemoryCollectiveResults(waveOp, it->second);
                        emitCollective = true;
                    } else {
                        memoryHasResults = true;
                        scheduleNow = false;
                    }
                }
                if (emitCollective && traceSink_ && blockCtx) {
                    std::string opName;
                    if (waveOp)
                        opName = const_cast<mlir::Operation *>(waveOp)
                                     ->getName()
                                     .getStringRef()
                                     .str();
                    traceSink_->onCollectiveComplete(
                        waveId,
                        opName,
                        it->second.expectedMask, it->second.expectedMask,
                        it->second.block.sequenceId, it->second.block.block,
                        blockKindLabel(blockCtx->kind),
                        blockCtx->loopIteration);
                }
                if (blockCtx && scheduleNow) {
                    std::uint64_t mask = it->second.expectedMask;
                    while (mask) {
                        unsigned l = std::countr_zero(mask);
                        mask &= mask - 1;
                        auto contIt = it->second.continuations.find(l);
                        if (contIt != it->second.continuations.end()) {
                            blockCtx->activeMask |= (1ull << l);
                            state_.readyQueue.push(
                                ReadyContinuation<ValueType, StepType>{
                                    waveId, it->second.block, l, contIt->second});
                        }
                    }
                }
                bool keepCollective =
                    isWaveCollective || (isMemoryCollective && memoryHasResults);
                if (!keepCollective) {
                    if (isMemoryCollective)
                        waveCtx.collectiveTokenToOp.erase(it->first);
                    auto cur = it;
                    ++it;
                    waveCtx.collectives.erase(cur);
                    continue;
                }
            }
            ++it;
        }
        // Clear from sync points and release if now satisfied.
        for (auto it = waveCtx.syncPoints.begin();
             it != waveCtx.syncPoints.end();) {
            it->second.expectedMask &= ~(1ull << lane);
            it->second.arrivals.erase(lane);
            it->second.continuations.erase(lane);
            bool ready = it->second.expectedMask &&
                         it->second.arrivals.size() ==
                             static_cast<unsigned>(
                                 std::popcount(it->second.expectedMask));
            if (ready) {
                auto *blockCtx = getBlock(waveCtx, it->second.block);
                if (blockCtx) {
                    std::uint64_t mask = it->second.expectedMask;
                    while (mask) {
                        unsigned l = std::countr_zero(mask);
                        mask &= mask - 1;
                        auto contIt = it->second.continuations.find(l);
                        if (contIt != it->second.continuations.end()) {
                            blockCtx->activeMask |= (1ull << l);
                            state_.readyQueue.push(
                                ReadyContinuation<ValueType, StepType>{
                                    waveId, it->second.block, l, contIt->second});
                        }
                    }
                }
                auto cur = it;
                ++it;
                waveCtx.syncPoints.erase(cur);
                continue;
            }
            ++it;
        }
    }

    void markMergeCompletion(WaveId,
                             WaveContext<ValueType, StepType> &waveCtx,
                             const DynamicBlockKey &childKey,
                             LaneId lane) {
        if (waveCtx.mergeStack.empty())
            return;
        for (auto it = waveCtx.mergeStack.rbegin();
             it != waveCtx.mergeStack.rend(); ++it) {
            bool matchesChild = llvm::any_of(it->pendingChildren,
                                             [&](const DynamicBlockKey &k) {
                                                 return k == childKey;
                                             });
            if (!matchesChild)
                continue;
            it->completedMask |= (1ull << lane);
            if (it->completedMask == it->expectedMask) {
                auto base = it.base();
                waveCtx.mergeStack.erase(--base);
            }
            break;
        }
    }

    void handleReconvergence(WaveId waveId,
                             WaveContext<ValueType, StepType> &waveCtx,
                             const DynamicBlockKey &childKey,
                             LaneId lane) {
        if (waveCtx.mergeStack.empty())
            return;
        for (auto it = waveCtx.mergeStack.rbegin();
             it != waveCtx.mergeStack.rend(); ++it) {
            bool matchesChild = llvm::any_of(it->pendingChildren,
                                             [&](const DynamicBlockKey &k) {
                                                 return k == childKey;
                                             });
            if (!matchesChild)
                continue;

            if (EnableCPSDebugLogs) {
                auto fmt = [&](std::uint64_t m) { return formatMaskBits(m, 32); };
                llvm::errs() << "[CPS] handleReconvergence lane=" << lane
                             << " child=" << childKey.block
                             << " seq=" << childKey.sequenceId
                             << " parent=" << it->parent.block
                             << " parentSeq=" << it->parent.sequenceId
                             << " expected=0b" << fmt(it->expectedMask)
                             << " completed(before)=0b" << fmt(it->completedMask)
                             << "\n";
            }

            it->completedMask |= (1ull << lane);

            DynamicBlockKey parentKey = it->parent;
            auto parentBlockIt = waveCtx.blocks.find(parentKey);
            if (parentBlockIt != waveCtx.blocks.end()) {
                auto &parentBlock = parentBlockIt->second;
                parentBlock.activeMask |= (1ull << lane);
                // Resume parent continuation for this lane immediately.
                auto contIt = parentBlock.continuations.find(lane);
                if (contIt != parentBlock.continuations.end()) {
                    if (EnableCPSDebugLogs) {
                        llvm::errs() << "[CPS] enqueue from reconverge lane=" << lane
                                     << " parent=" << parentKey.block
                                     << " seq=" << parentKey.sequenceId << "\n";
                    }
                    state_.readyQueue.push(
                        ReadyContinuation<ValueType, StepType>{waveId, parentKey, lane,
                                                               contIt->second});
                    dumpReadyQueue();
                    parentBlock.continuations.erase(contIt);
                } else if (EnableCPSDebugLogs) {
                    llvm::errs() << "[CPS] no parent continuation for lane=" << lane
                                 << " parent=" << parentKey.block
                                 << " seq=" << parentKey.sequenceId << "\n";
                }
            }
            // Pop the merge entry only when all expected lanes are done.
            bool shouldPop = !it->loopFrame &&
                             (it->expectedMask != 0
                                  ? (it->completedMask == it->expectedMask)
                                  : it->pendingChildren.empty());
            if (shouldPop) {
                if (EnableCPSDebugLogs) {
                    auto fmt = [&](std::uint64_t m) { return formatMaskBits(m, 32); };
                    llvm::errs() << "[CPS] pop merge parent=" << parentKey.block
                                 << " seq=" << parentKey.sequenceId
                                 << " expected=0b" << fmt(it->expectedMask)
                                 << " completed=0b" << fmt(it->completedMask)
                                 << "\n";
                    llvm::errs() << "[CPS] resume parent continuations parent="
                                 << parentKey.block << " seq=" << parentKey.sequenceId
                                 << " mask=0b" << fmt(it->expectedMask) << " lanes:";
                    std::uint64_t dbgMask = it->expectedMask;
                    while (dbgMask) {
                        unsigned l = std::countr_zero(dbgMask);
                        dbgMask &= dbgMask - 1;
                        llvm::errs() << " " << l;
                    }
                    llvm::errs() << "\n";
                    // logMergeStackState<ValueType, StepType>(waveCtx);
                }
                // Enqueue any remaining parent continuations for lanes that have
                // not been resumed yet.
                // auto parentBlockIt2 = waveCtx.blocks.find(parentKey);
                // if (parentBlockIt2 != waveCtx.blocks.end()) {
                //     auto &parentBlock = parentBlockIt2->second;
                //     std::uint64_t mask = it->expectedMask;
                //     while (mask) {
                //         unsigned l = std::countr_zero(mask);
                //         mask &= mask - 1;
                //         auto contIt = parentBlock.continuations.find(l);
                //         if (contIt != parentBlock.continuations.end()) {
                //             parentBlock.activeMask |= (1ull << l);
                //             waveCtx.lanes[l].currentBlock = parentKey;
                //             state_.readyQueue.push(
                //                 ReadyContinuation<ValueType, StepType>{waveId, parentKey, l,
                //                                                        contIt->second});
                //             if (EnableCPSDebugLogs) {
                //                 llvm::errs() << "[CPS] enqueue parent cont lane=" << l
                //                              << " parent=" << parentKey.block
                //                              << " seq=" << parentKey.sequenceId << "\n";
                //                 dumpReadyQueue();
                //             }
                //             parentBlock.continuations.erase(contIt);
                //         }
                //     }
                // }
                auto base = it.base();
                waveCtx.mergeStack.erase(--base);
            }
            dumpReadyQueue();
            break;
        }
        // logMergeStackState<ValueType, StepType>(waveCtx);

    }

    SimtStepSemanticsAdaptor<SemanticsT> adaptor_;
    SemanticsT semantics_;
    TraceSink *traceSink_ = nullptr;
    StateType state_;
};

} // namespace simt::semantics
