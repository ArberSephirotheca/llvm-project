#pragma once

#include "simt-step/semantics/SemanticsContext.h"

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

/// Describes a barrier rendezvous that suspends execution until the handler
/// releases the stored continuation.
struct BarrierEffect {
    /// Placeholder scope identifier; concrete semantics can reinterpret it.
    std::uint32_t scope = 0;
};

/// Models an explicit yield back to the scheduler.
struct YieldEffect {
    /// Optional payload for instrumentation hooks.
    std::optional<std::uint64_t> tag;
};

/// Coordinates collective operations (wave intrinsics, reductions, etc.).
struct CollectiveEffect {
    /// Identifier for the collective kind (e.g., opcode or enum value).
    std::uint32_t operation = 0;
    /// Bitmask of participating lanes; interpretation is semantics-specific.
    std::uint64_t activeMask = 0;
    /// Optional slot for implementation-defined payload indexing.
    std::optional<std::uint32_t> token;
};

/// Notifies the handler about an execution-wide synchronization point (e.g.,
/// subgroup barrier) using the same metadata layout as collectives for ease of
/// handling.
struct SynchronizationEffect {
    /// Identifier for the synchronization operation.
    std::uint32_t operation = 0;
    /// Bitmask of participating lanes; interpretation is semantics-specific.
    std::uint64_t activeMask = 0;
    /// Optional slot for implementation-defined payload indexing.
    std::optional<std::uint32_t> token;
};

/// Requests that the handler explore multiple continuations (e.g., for model
/// checking). The handler chooses which branch to resume.
struct NondeterministicChoiceEffect {
    std::uint32_t choiceCount = 0;
};

/// Aggregates all supported interpreter effects.
struct Effect {
    using Payload =
        std::variant<std::monostate, BarrierEffect, YieldEffect, CollectiveEffect,
                     SynchronizationEffect, NondeterministicChoiceEffect>;

    Effect() = default;

    template <typename EffectT>
    Effect(EffectT effect) : payload_(std::move(effect)) {}

    bool hasValue() const {
        return !std::holds_alternative<std::monostate>(payload_);
    }

    template <typename EffectT>
    bool isa() const {
        return std::holds_alternative<EffectT>(payload_);
    }

    template <typename EffectT>
    EffectT &get() {
        return std::get<EffectT>(payload_);
    }

    template <typename EffectT>
    const EffectT &get() const {
        return std::get<EffectT>(payload_);
    }

private:
    Payload payload_;
};

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

// Forward declarations for state structures defined in ExecutionState.h.
struct DynamicBlockKey;
template <typename ValueT, typename StepT>
struct ReadyContinuation;
template <typename ValueT, typename StepT>
struct InterpreterState;
using WaveId = std::uint32_t;
using LaneId = std::uint32_t;

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
                return llvm::Error::success();
            }

            if (std::holds_alternative<typename StepType::Halt>(stateVariant)) {
                laneCtx.hasReturned = true;
                laneCtx.returnValue.reset();
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

        return llvm::make_error<llvm::StringError>(
            "collective/synchronization effects are not implemented in CPS interpreter yet",
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

    SemanticsT semantics_;
    StateType state_;
};

} // namespace simt::semantics
