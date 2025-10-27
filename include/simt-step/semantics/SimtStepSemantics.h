#pragma once

#include "simt-step/semantics/SemanticsContext.h"
#include "simt-step/semantics/SemValue.h"

namespace mlir {
class Operation;
}

namespace simt::semantics {

template <typename ValueT>
class Step;

/// Abstract interface implemented by concrete SIMT semantics providers.
class SimtStepSemantics {
public:
    using ValueType = SemValue;
    using StepType = Step<ValueType>;

    virtual ~SimtStepSemantics() = default;

    /// Evaluate a single operation for the active lane and return the next
    /// continuation.
    virtual StepType evalOperation(mlir::Operation *op,
                                   SemanticsContext &context) = 0;
};

} // namespace simt::semantics
