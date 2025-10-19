#pragma once

#include "simt-step/semantics/SemanticsContext.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <utility>
#include <variant>

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
        return impl.eval(op, context);
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

} // namespace simt::semantics
