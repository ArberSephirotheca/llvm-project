#pragma once

#include <cstdint>
#include <optional>
#include <variant>

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

/// Notifies the handler about an execution-wide synchronization point.
struct SynchronizationEffect {
    /// Identifier for the synchronization operation.
    std::uint32_t operation = 0;
    /// Bitmask of participating lanes; interpretation is semantics-specific.
    std::uint64_t activeMask = 0;
    /// Optional slot for implementation-defined payload indexing.
    std::optional<std::uint32_t> token;
};

/// Requests that the handler explore multiple continuations.
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
    EffectT *get_if() {
        if (auto *ptr = std::get_if<EffectT>(&payload_))
            return ptr;
        return nullptr;
    }

    template <typename EffectT>
    const EffectT *get_if() const {
        if (auto *ptr = std::get_if<EffectT>(&payload_))
            return ptr;
        return nullptr;
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

} // namespace simt::semantics
