#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/PatternMatch.h>

#define GET_OP_CLASSES
#include "SimtStepOps.cpp.inc"

namespace simt::dialect {

static void addResultTypeFromResource(mlir::OpBuilder &builder,
                                      mlir::OperationState &state,
                                      mlir::Value resource) {
  if (auto resourceType = mlir::dyn_cast<ResourceType>(resource.getType())) {
    state.addTypes(resourceType.getElementType());
    return;
  }
  state.addTypes(builder.getNoneType());
}

static mlir::LogicalResult verifyAtomicOpCommon(
    mlir::Operation *op, mlir::Value resource, mlir::Value index,
    llvm::ArrayRef<mlir::Value> valueOperands, mlir::Type resultType,
    bool requireIntegerElement) {
  auto resourceType = mlir::dyn_cast<ResourceType>(resource.getType());
  if (!resourceType)
    return op->emitOpError("requires operand 0 to be a simt.resource type");

  if (!mlir::isa<mlir::IntegerType>(index.getType()))
    return op->emitOpError("index must have integer type");

  mlir::Type elementType = resourceType.getElementType();
  if (requireIntegerElement && !mlir::isa<mlir::IntegerType>(elementType))
    return op->emitOpError("requires integer element type");

  for (mlir::Value value : valueOperands) {
    if (!value)
      continue;
    if (value.getType() != elementType)
      return op->emitOpError("value type must match resource element type");
  }

  if (resultType && resultType != elementType)
    return op->emitOpError("result type must match resource element type");

  return mlir::success();
}

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

void BufferAtomicAddOp::build(mlir::OpBuilder &builder,
                              mlir::OperationState &state,
                              mlir::Value resource, mlir::Value index,
                              mlir::Value value) {
  state.addOperands({resource, index, value});
  addResultTypeFromResource(builder, state, resource);
}

mlir::LogicalResult BufferAtomicAddOp::verify() {
  llvm::SmallVector<mlir::Value, 1> values{getValue()};
  return verifyAtomicOpCommon(getOperation(), getResource(), getIndex(),
                              values, getOldValue().getType(),
                              /*requireIntegerElement=*/false);
}

void BufferAtomicExchangeOp::build(mlir::OpBuilder &builder,
                                   mlir::OperationState &state,
                                   mlir::Value resource, mlir::Value index,
                                   mlir::Value value) {
  state.addOperands({resource, index, value});
  addResultTypeFromResource(builder, state, resource);
}

mlir::LogicalResult BufferAtomicExchangeOp::verify() {
  llvm::SmallVector<mlir::Value, 1> values{getValue()};
  return verifyAtomicOpCommon(getOperation(), getResource(), getIndex(),
                              values, getOldValue().getType(),
                              /*requireIntegerElement=*/false);
}

void BufferAtomicCompareExchangeOp::build(mlir::OpBuilder &builder,
                                          mlir::OperationState &state,
                                          mlir::Value resource,
                                          mlir::Value index,
                                          mlir::Value compare,
                                          mlir::Value value) {
  state.addOperands({resource, index, compare, value});
  addResultTypeFromResource(builder, state, resource);
}

mlir::LogicalResult BufferAtomicCompareExchangeOp::verify() {
  llvm::SmallVector<mlir::Value, 2> values{getCompare(), getValue()};
  return verifyAtomicOpCommon(getOperation(), getResource(), getIndex(),
                              values, getOldValue().getType(),
                              /*requireIntegerElement=*/false);
}

void BufferAtomicMinOp::build(mlir::OpBuilder &builder,
                              mlir::OperationState &state,
                              mlir::Value resource, mlir::Value index,
                              mlir::Value value) {
  state.addOperands({resource, index, value});
  addResultTypeFromResource(builder, state, resource);
}

mlir::LogicalResult BufferAtomicMinOp::verify() {
  llvm::SmallVector<mlir::Value, 1> values{getValue()};
  return verifyAtomicOpCommon(getOperation(), getResource(), getIndex(),
                              values, getOldValue().getType(),
                              /*requireIntegerElement=*/true);
}

void BufferAtomicMaxOp::build(mlir::OpBuilder &builder,
                              mlir::OperationState &state,
                              mlir::Value resource, mlir::Value index,
                              mlir::Value value) {
  state.addOperands({resource, index, value});
  addResultTypeFromResource(builder, state, resource);
}

mlir::LogicalResult BufferAtomicMaxOp::verify() {
  llvm::SmallVector<mlir::Value, 1> values{getValue()};
  return verifyAtomicOpCommon(getOperation(), getResource(), getIndex(),
                              values, getOldValue().getType(),
                              /*requireIntegerElement=*/true);
}

void BufferAtomicAndOp::build(mlir::OpBuilder &builder,
                              mlir::OperationState &state,
                              mlir::Value resource, mlir::Value index,
                              mlir::Value value) {
  state.addOperands({resource, index, value});
  addResultTypeFromResource(builder, state, resource);
}

mlir::LogicalResult BufferAtomicAndOp::verify() {
  llvm::SmallVector<mlir::Value, 1> values{getValue()};
  return verifyAtomicOpCommon(getOperation(), getResource(), getIndex(),
                              values, getOldValue().getType(),
                              /*requireIntegerElement=*/true);
}

void BufferAtomicOrOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &state,
                             mlir::Value resource, mlir::Value index,
                             mlir::Value value) {
  state.addOperands({resource, index, value});
  addResultTypeFromResource(builder, state, resource);
}

mlir::LogicalResult BufferAtomicOrOp::verify() {
  llvm::SmallVector<mlir::Value, 1> values{getValue()};
  return verifyAtomicOpCommon(getOperation(), getResource(), getIndex(),
                              values, getOldValue().getType(),
                              /*requireIntegerElement=*/true);
}

void BufferAtomicXorOp::build(mlir::OpBuilder &builder,
                              mlir::OperationState &state,
                              mlir::Value resource, mlir::Value index,
                              mlir::Value value) {
  state.addOperands({resource, index, value});
  addResultTypeFromResource(builder, state, resource);
}

mlir::LogicalResult BufferAtomicXorOp::verify() {
  llvm::SmallVector<mlir::Value, 1> values{getValue()};
  return verifyAtomicOpCommon(getOperation(), getResource(), getIndex(),
                              values, getOldValue().getType(),
                              /*requireIntegerElement=*/true);
}

} // namespace simt::dialect
