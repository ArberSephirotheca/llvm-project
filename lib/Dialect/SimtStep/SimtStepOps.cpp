#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/PatternMatch.h>

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
      return emitOpError(
          "'params' must be a dictionary attribute when present");
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

void BufferLoadOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                         mlir::Value resource, mlir::Value index) {
  state.addOperands({resource, index});
  auto resourceType = mlir::dyn_cast<ResourceType>(resource.getType());
  if (!resourceType)
    state.addTypes(builder.getNoneType());
  else
    state.addTypes(resourceType.getElementType());
}

mlir::LogicalResult BufferLoadOp::verify() {
  auto resourceType = mlir::dyn_cast<ResourceType>(getResource().getType());
  if (!resourceType)
    return emitOpError("requires operand 0 to be a simt.resource type");

  if (!mlir::isa<mlir::IntegerType>(getIndex().getType()))
    return emitOpError("index must have integer type");

  if (getResult().getType() != resourceType.getElementType())
    return emitOpError("result type must match resource element type");
  return mlir::success();
}

void BufferStoreOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          mlir::Value resource, mlir::Value index,
                          mlir::Value value) {
  state.addOperands({resource, index, value});
}

mlir::LogicalResult BufferStoreOp::verify() {
  auto resourceType = mlir::dyn_cast<ResourceType>(getResource().getType());
  if (!resourceType)
    return emitOpError("requires operand 0 to be a simt.resource type");

  if (!mlir::isa<mlir::IntegerType>(getIndex().getType()))
    return emitOpError("index must have integer type");

  if (getValue().getType() != resourceType.getElementType())
    return emitOpError("value type must match resource element type");
  return mlir::success();
}

} // namespace simt::dialect
