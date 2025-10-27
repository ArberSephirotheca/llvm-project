#pragma once

#include "simt-step/semantics/SimtStepSemantics.h"

#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <llvm/Support/Error.h>

namespace simt::semantics {

/// Minimal semantics implementation that understands constants, basic
/// arithmetic, comparisons, selected state queries, and simple control flow.
class SimpleSemantics : public SimtStepSemantics {
public:
    StepType evalOperation(mlir::Operation *op,
                           SemanticsContext &context) override;

    llvm::Expected<ValueType> evaluateValue(mlir::Value value,
                                            SemanticsContext &context);

private:
    StepType handleConstant(mlir::arith::ConstantOp op);
    StepType handleLaneId(SemanticsContext &context);
    StepType handleAddIOp(mlir::arith::AddIOp op);
    StepType handleCmpIOp(mlir::arith::CmpIOp op, SemanticsContext &context);
    StepType handleDispatchThreadId(SemanticsContext &context);
    StepType handleYieldOp(simt::dialect::YieldOp op,
                           SemanticsContext &context);
    StepType handleReturnOp(mlir::func::ReturnOp op);

    StepType handleUnknown(mlir::Operation *op);
};

} // namespace simt::semantics
