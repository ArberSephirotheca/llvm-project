#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/PatternMatch.h>

#define GET_OP_CLASSES
#include "SimtStepOps.cpp.inc"

namespace simt::dialect {

mlir::ParseResult FenceOp::parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &state) {
  if (parser.parseOptionalAttrDict(state.attributes))
    return mlir::failure();
  return mlir::success();
}

void FenceOp::print(mlir::OpAsmPrinter &printer) {
  auto &os = printer.getStream();
  bool opened = false;
  auto emitSep = [&]() {
    if (!opened) {
      os << " {";
      opened = true;
    } else {
      os << ", ";
    }
  };

  if (auto scope = getScope()) {
    emitSep();
    os << "scope = #simt_step.scope<" << stringifyScope(*scope) << ">";
  }
  if (auto memSem = getMemsem()) {
    emitSep();
    os << "memsem = #simt_step.memsem<" << stringifyMemorySemantics(*memSem)
       << ">";
  }
  if (auto memSpace = getMemspace()) {
    emitSep();
    os << "memspace = #simt_step.memspace<" << stringifyMemorySpace(*memSpace)
       << ">";
  }

  if (opened)
    os << "}";
}

mlir::ParseResult BarrierOp::parse(mlir::OpAsmParser &parser,
                                   mlir::OperationState &state) {
  if (parser.parseOptionalAttrDict(state.attributes))
    return mlir::failure();
  return mlir::success();
}

void BarrierOp::print(mlir::OpAsmPrinter &printer) {
  auto &os = printer.getStream();
  bool opened = false;
  auto emitSep = [&]() {
    if (!opened) {
      os << " {";
      opened = true;
    } else {
      os << ", ";
    }
  };

  if (auto scope = getScope()) {
    emitSep();
    os << "scope = #simt_step.scope<" << stringifyScope(*scope) << ">";
  }
  if (auto memSem = getMemsem()) {
    emitSep();
    os << "memsem = #simt_step.memsem<" << stringifyMemorySemantics(*memSem)
       << ">";
  }

  if (opened)
    os << "}";
}

static mlir::ParseResult parseStateQueryOp(mlir::OpAsmParser &parser,
                                          mlir::OperationState &state) {
  if (parser.parseOptionalAttrDict(state.attributes))
    return mlir::failure();
  mlir::Type resultType;
  if (parser.parseColonType(resultType))
    return mlir::failure();
  state.addTypes(resultType);
  return mlir::success();
}

static void printStateQueryOp(mlir::Operation *op,
                              mlir::OpAsmPrinter &printer) {
  if (!op->getAttrs().empty()) {
    auto &os = printer.getStream();
    os << " {";
    bool first = true;
    for (auto attr : op->getAttrs()) {
      if (!first)
        os << ", ";
      first = false;
      os << attr.getName().str() << " = ";
      attr.getValue().print(os);
    }
    os << "}";
  }
  printer.getStream() << " : " << op->getResult(0).getType();
}

// mlir::ParseResult DispatchThreadIdOp::parse(mlir::OpAsmParser &parser,
//                                             mlir::OperationState &state) {
//   return parseStateQueryOp(parser, state);
// }

// void DispatchThreadIdOp::print(mlir::OpAsmPrinter &printer) {
//   printStateQueryOp(getOperation(), printer);
// }

mlir::ParseResult GroupThreadIdOp::parse(mlir::OpAsmParser &parser,
                                         mlir::OperationState &state) {
  return parseStateQueryOp(parser, state);
}

void GroupThreadIdOp::print(mlir::OpAsmPrinter &printer) {
  printStateQueryOp(getOperation(), printer);
}

mlir::ParseResult GroupIdOp::parse(mlir::OpAsmParser &parser,
                                   mlir::OperationState &state) {
  return parseStateQueryOp(parser, state);
}

void GroupIdOp::print(mlir::OpAsmPrinter &printer) {
  printStateQueryOp(getOperation(), printer);
}

mlir::ParseResult GroupIndexOp::parse(mlir::OpAsmParser &parser,
                                      mlir::OperationState &state) {
  return parseStateQueryOp(parser, state);
}

void GroupIndexOp::print(mlir::OpAsmPrinter &printer) {
  printStateQueryOp(getOperation(), printer);
}

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

void SwitchOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                     mlir::TypeRange resultTypes, mlir::Value selector,
                     mlir::ValueRange initialValues,
                     mlir::ArrayRef<int64_t> caseValues) {
  state.addTypes(resultTypes);
  state.addOperands(selector);
  state.addOperands(initialValues);
  state.addAttribute("case_values",
                     builder.getDenseI64ArrayAttr(caseValues));

  mlir::OpBuilder::InsertionGuard guard(builder);
  auto *body = state.addRegion();
  mlir::Block *entry = builder.createBlock(body);
  if (!resultTypes.empty()) {
    llvm::SmallVector<mlir::Location, 8> argLocs(resultTypes.size(),
                                                 state.location);
    entry->addArguments(resultTypes, argLocs);
  }
}

mlir::LogicalResult SwitchOp::verify() {
  if (getInitialValues().size() != getNumResults())
    return emitOpError(
        "requires the number of initial values to match result count");

  auto caseValues = getCaseValuesAttr();
  if (!caseValues)
    return emitOpError("requires 'case_values' attribute");

  mlir::Region &body = getCaseBody();
  if (body.empty())
    return emitOpError("requires a non-empty body region");

  for (mlir::Block &block : body) {
    if (block.getNumArguments() != getNumResults())
      return emitOpError(
          "each block in the body must have an argument for every result");
    if (block.empty() || !llvm::isa<YieldOp>(block.back()))
      return emitOpError(
          "each block in the body must terminate with simt_step.yield");
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
  auto *thenBlock = builder.createBlock(thenRegion);
  if (resultTypes.empty()) {
    builder.setInsertionPointToEnd(thenBlock);
    builder.create<simt::dialect::YieldOp>(state.location);
  }

  auto *elseRegion = state.addRegion();
  if (withElseRegion) {
    auto *elseBlock = builder.createBlock(elseRegion);
    if (resultTypes.empty()) {
      builder.setInsertionPointToEnd(elseBlock);
      builder.create<simt::dialect::YieldOp>(state.location);
    }
  }
}

void IfOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                 mlir::Value condition, bool withElseRegion) {
  build(builder, state, mlir::TypeRange{}, condition, withElseRegion);
}

mlir::LogicalResult IfOp::verify() {
  if (getNumResults() != 0 && getElseRegion().empty())
    return emitOpError("must have an else block if defining values");

  auto checkRegion = [&](mlir::Region &region, llvm::StringRef name)
      -> mlir::LogicalResult {
    if (region.empty())
      return emitOpError() << name << " region must not be empty";
    if (!region.hasOneBlock())
      return emitOpError() << name << " region must contain exactly one block";
    return mlir::success();
  };

  if (failed(checkRegion(getThenRegion(), "then")))
    return mlir::failure();
  if (!getElseRegion().empty() &&
      failed(checkRegion(getElseRegion(), "else")))
    return mlir::failure();

  bool withinLoop = false;
  for (mlir::Operation *parent = getOperation()->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (llvm::isa<simt::dialect::LoopOp>(parent)) {
      withinLoop = true;
      break;
    }
  }

  auto checkTerminator = [&](mlir::Block &block) -> mlir::LogicalResult {
    mlir::Operation *terminator = block.getTerminator();
    if (!terminator)
      return emitOpError("region must terminate with a structured terminator");
    const bool isYield = llvm::isa<simt::dialect::YieldOp>(terminator);
    const bool isContinue = llvm::isa<simt::dialect::ContinueOp>(terminator);
    const bool isBreak = llvm::isa<simt::dialect::BreakOp>(terminator);

    if (isYield)
      return mlir::success();
    if ((isContinue || isBreak) && withinLoop)
      return mlir::success();
    return emitOpError()
           << "region must terminate with simt_step.yield or, when nested in a "
              "simt_step.loop, with simt_step.continue/simt_step.break";
  };

  if (failed(checkTerminator(getThenRegion().front())))
    return mlir::failure();
  if (!getElseRegion().empty() &&
      failed(checkTerminator(getElseRegion().front())))
    return mlir::failure();

  auto checkYieldResults = [&](mlir::Block &block) -> mlir::LogicalResult {
    if (getNumResults() == 0)
      return mlir::success();
    mlir::Operation *terminator = block.getTerminator();
    auto yield = llvm::dyn_cast_or_null<simt::dialect::YieldOp>(terminator);
    if (!yield)
      return mlir::success();
    if (yield.getOperands().size() != getNumResults())
      return emitOpError("yield arity must match if results");
    for (auto [operand, type] : llvm::zip(yield.getOperands(), getResultTypes()))
      if (operand.getType() != type)
        return emitOpError("yield types must match if results");
    return mlir::success();
  };

  if (failed(checkYieldResults(getThenRegion().front())))
    return mlir::failure();
  if (!getElseRegion().empty() &&
      failed(checkYieldResults(getElseRegion().front())))
    return mlir::failure();

  return mlir::success();
}

static mlir::LogicalResult
verifyLoopAncestor(mlir::Operation *op, llvm::StringRef opName) {
  for (mlir::Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (llvm::isa<simt::dialect::LoopOp>(parent))
      return mlir::success();
    if (llvm::isa<mlir::ModuleOp>(parent))
      break;
  }
  return op->emitError() << opName << " must be nested inside 'simt_step.loop'";
}

static mlir::LogicalResult
verifyLoopOrSwitchAncestor(mlir::Operation *op, llvm::StringRef opName) {
  for (mlir::Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (llvm::isa<simt::dialect::LoopOp, simt::dialect::SwitchOp>(parent))
      return mlir::success();
    if (llvm::isa<mlir::ModuleOp>(parent))
      break;
  }
  return op->emitError() << opName
                         << " must be nested inside 'simt_step.loop' or "
                            "'simt_step.switch'";
}

mlir::LogicalResult ContinueOp::verify() {
  return verifyLoopAncestor(getOperation(), "simt_step.continue");
}

mlir::LogicalResult BreakOp::verify() {
  return verifyLoopOrSwitchAncestor(getOperation(), "simt_step.break");
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
