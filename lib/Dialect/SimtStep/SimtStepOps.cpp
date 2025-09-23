#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"

#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Builders.h>
#include <llvm/Support/raw_ostream.h>

#define GET_OP_CLASSES
#include "SimtStepOps.cpp.inc"

namespace simt::dialect {

mlir::LogicalResult BarrierOp::verify() {
    auto *op = getOperation();
    if (auto attr = op->getAttr("scope")) {
        if (!mlir::isa<mlir::IntegerAttr>(attr))
            return emitOpError("expected 'scope' attribute to be an integer enum");
    }
    if (auto attr = op->getAttr("memsem")) {
        if (!mlir::isa<mlir::IntegerAttr>(attr))
            return emitOpError("expected 'memsem' attribute to be an integer enum");
    }
    return mlir::success();
}

mlir::LogicalResult CustomOp::verify() {
    auto *op = getOperation();
    auto instr = op->getAttrOfType<mlir::StringAttr>("instr");
    if (!instr || instr.getValue().empty())
        return emitOpError("requires non-empty 'instr' string attribute");
    if (auto params = op->getAttr("params")) {
        if (!mlir::isa<mlir::DictionaryAttr>(params))
            return emitOpError("'params' must be a dictionary attribute when present");
    }
    return mlir::success();
}

void IfOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                 mlir::TypeRange resultTypes, mlir::Value condition,
                 bool withElseRegion) {
    state.addTypes(resultTypes);
    state.addOperands(condition);

    mlir::OpBuilder::InsertionGuard guard(builder);
    auto *thenRegion = state.addRegion();
    builder.createBlock(thenRegion);
    if (resultTypes.empty())
        IfOp::ensureTerminator(*thenRegion, builder, state.location);

    auto *elseRegion = state.addRegion();
    if (withElseRegion) {
        builder.createBlock(elseRegion);
        if (resultTypes.empty())
            IfOp::ensureTerminator(*elseRegion, builder, state.location);
    }
}

void IfOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                 mlir::Value condition, bool withElseRegion) {
    build(builder, state, mlir::TypeRange{}, condition, withElseRegion);
}

mlir::LogicalResult IfOp::verify() {
    if (getNumResults() != 0 && getElseRegion().empty())
        return emitOpError("must have an else block if defining values");
    return mlir::success();
}

} // namespace simt::dialect
