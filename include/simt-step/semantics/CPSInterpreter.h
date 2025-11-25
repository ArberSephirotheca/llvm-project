#pragma once

#include "simt-step/semantics/Effects.h"
#include "simt-step/semantics/ExecutionState.h"
#include "simt-step/semantics/SemanticsContext.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <functional>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <queue>

#include <llvm/Support/Error.h>

namespace mlir {
class Operation;
} // namespace mlir

namespace simt::semantics {

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

    StateType &state() { return state_; }
    const StateType &state() const { return state_; }

    /// Enqueue an initial continuation for the given wave/block/lane triple.
    void enqueue(WaveId wave, const DynamicBlockKey &block, LaneId lane,
                 StepType step) {
        ensureWaveBlock(wave, block, lane);
        state_.readyQueue.push(
            ReadyContinuation<ValueType, StepType>{wave, block, lane, std::move(step)});
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

private:
    llvm::Error processReady(ReadyContinuation<ValueType, StepType> item) {
        ensureWaveBlock(item.wave, item.block, item.lane);
        auto &waveCtx = state_.waves[item.wave];
        auto &laneCtx = waveCtx.lanes[item.lane];
        laneCtx.currentBlock = item.block;
        if (auto *blockCtx = getBlock(waveCtx, item.block)) {
            waveCtx.currentMask = blockCtx->activeMask;
        }
        StepType current = std::move(item.resume);
        for (;;) {
            typename StepType::State stateVariant = std::move(current).takeState();

            if (std::holds_alternative<typename StepType::Continue>(stateVariant)) {
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
                laneCtx.hasReturned = true;
                laneCtx.returnValue = std::move(prod.value);
                if (auto *blockCtx = getBlock(waveCtx, item.block)) {
                    std::uint64_t laneBit = 1ull << item.lane;
                    blockCtx->activeMask &= ~laneBit;
                    blockCtx->completedMask |= laneBit;
                    shrinkExpectedForLane(item.wave, waveCtx, item.lane);
                    // Resume parent execution for this lane.
                    handleReconvergence(item.wave, waveCtx, item.block, item.lane);
                }
                return llvm::Error::success();
            }

            if (std::holds_alternative<typename StepType::Halt>(stateVariant)) {
                laneCtx.hasReturned = true;
                if (auto *blockCtx = getBlock(waveCtx, item.block)) {
                    std::uint64_t laneBit = 1ull << item.lane;
                    blockCtx->activeMask &= ~laneBit;
                    blockCtx->completedMask |= laneBit;
                    shrinkExpectedForLane(item.wave, waveCtx, item.lane);
                    // Do not re-add returned lanes to parent activeMask; just
                    // account for completion in merge tracking.
                    markMergeCompletion(item.wave, waveCtx, item.block, item.lane);
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
            syncPoint.continuations[lane] =
                StepType::continueWith(
                    [resume = std::move(suspend.resume)]() mutable -> StepType {
                        return resume();
                    });

            auto expectedCount =
                static_cast<unsigned>(std::popcount(syncPoint.expectedMask));
            if (syncPoint.arrivals.size() == expectedCount) {
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
                waveCtx.collectives.erase(key);
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
                waveCtx.collectives.erase(cur);
                continue;
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
                waveCtx.mergeStack.pop_back();
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

            it->completedMask |= (1ull << lane);

            DynamicBlockKey parentKey = it->parent;
            auto parentBlockIt = waveCtx.blocks.find(parentKey);
            if (parentBlockIt != waveCtx.blocks.end()) {
                auto &parentBlock = parentBlockIt->second;
                parentBlock.activeMask |= (1ull << lane);
                // Resume parent continuation for this lane only.
                auto contIt = parentBlock.continuations.find(lane);
                if (contIt != parentBlock.continuations.end()) {
                    state_.readyQueue.push(
                        ReadyContinuation<ValueType, StepType>{waveId, parentKey, lane,
                                                               contIt->second});
                    parentBlock.continuations.erase(contIt);
                }
            }
            // Pop the merge entry only when all expected lanes are done.
            if (it->completedMask == it->expectedMask) {
                waveCtx.mergeStack.pop_back();
            }
            break;
        }
    }

    SemanticsT semantics_;
    StateType state_;
};

} // namespace simt::semantics
