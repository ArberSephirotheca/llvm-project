#pragma once

#include "simt-step/semantics/SimtStepSemantics.h"

#include <mlir/Dialect/Arith/IR/Arith.h>

namespace simt::semantics {

/// Minimal semantics implementation that understands constants, basic
/// arithmetic, and lane queries. Intended for interpreter smoke tests.
class SimpleSemantics : public SimtStepSemantics {
public:
    StepType evalOperation(mlir::Operation *op,
                           SemanticsContext &context) override;

private:
    StepType handleConstant(mlir::arith::ConstantOp op);
    StepType handleLaneId(SemanticsContext &context);
    StepType handleAddIOp(mlir::arith::AddIOp op);
};

} // namespace simt::semantics
