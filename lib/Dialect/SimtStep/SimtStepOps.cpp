#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"

#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Builders.h>

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

mlir::ParseResult IfOp::parse(mlir::OpAsmParser &parser,
                              mlir::OperationState &result) {
    result.regions.reserve(2);
    auto *thenRegion = result.addRegion();
    auto *elseRegion = result.addRegion();

    mlir::OpAsmParser::UnresolvedOperand condition;
    auto &builder = parser.getBuilder();
    mlir::Type i1Type = builder.getI1Type();
    if (parser.parseOperand(condition) ||
        parser.resolveOperand(condition, i1Type, result.operands))
        return mlir::failure();

    if (parser.parseOptionalArrowTypeList(result.types))
        return mlir::failure();

    if (parser.parseRegion(*thenRegion, {}, {}))
        return mlir::failure();
    IfOp::ensureTerminator(*thenRegion, parser.getBuilder(), result.location);

    if (!parser.parseOptionalKeyword("else")) {
        if (parser.parseRegion(*elseRegion, {}, {}))
            return mlir::failure();
        IfOp::ensureTerminator(*elseRegion, parser.getBuilder(),
                               result.location);
    }

    if (parser.parseOptionalAttrDict(result.attributes))
        return mlir::failure();

    if (!result.types.empty() && elseRegion->empty())
        return parser.emitError(parser.getNameLoc(),
                                "expected else block to yield results");
    return mlir::success();
}

void IfOp::print(mlir::OpAsmPrinter &printer) {
    bool printTerminators = !getResults().empty();
    printer << ' ' << getCondition();
    if (!getResults().empty())
        printer << " -> (" << getResultTypes() << ')';
    printer << ' ';
    printer.printRegion(getThenRegion(), /*printEntryBlockArgs=*/false,
                        /*printBlockTerminators=*/printTerminators);
    if (!getElseRegion().empty()) {
        printer << " else ";
        printer.printRegion(getElseRegion(), /*printEntryBlockArgs=*/false,
                            /*printBlockTerminators=*/printTerminators);
    }
    printer.printOptionalAttrDict((*this)->getAttrs());
}

} // namespace simt::dialect
