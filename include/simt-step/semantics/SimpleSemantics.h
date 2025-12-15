#pragma once

#include "simt-step/semantics/SimtStepSemantics.h"

#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Value.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/Error.h>
#include <vector>

namespace simt::semantics {

/// Minimal semantics implementation that understands constants, basic
/// arithmetic, comparisons, selected state queries, and simple control flow.
class SimpleSemantics : public SimtStepSemantics {
public:
    StepType evalOperation(mlir::Operation *op,
                           SemanticsContext &context) override;

    llvm::Expected<ValueType> evaluateValue(mlir::Value value,
                                            SemanticsContext &context);

    static void clearMemory();
    static const llvm::DenseMap<mlir::Value, llvm::DenseMap<int64_t, ValueType>> &
    memory();

private:
    StepType handleConstant(mlir::arith::ConstantOp op);
    StepType handleLaneId(SemanticsContext &context);
    StepType handleAddIOp(mlir::arith::AddIOp op,
                          SemanticsContext &context);
    StepType handleRemSIOp(mlir::arith::RemSIOp op,
                           SemanticsContext &context);
    StepType handleAndIOp(mlir::arith::AndIOp op,
                          SemanticsContext &context);
    StepType handleCmpIOp(mlir::arith::CmpIOp op, SemanticsContext &context);
    StepType handleDispatchThreadId(SemanticsContext &context);
    StepType handleWaveCountBits(mlir::Operation *op, SemanticsContext &context);
    StepType handleBufferStore(mlir::Operation *op, SemanticsContext &context);
    StepType handleBufferLoad(mlir::Operation *op, SemanticsContext &context);
    StepType handleYieldOp(simt::dialect::YieldOp op,
                           SemanticsContext &context);
    StepType handleReturnOp(mlir::func::ReturnOp op);

    StepType handleUnknown(mlir::Operation *op);

    llvm::Expected<std::vector<SemValue>>
    evaluateLoopOp(simt::dialect::LoopOp loop, SemanticsContext &context);
};

} // namespace simt::semantics
