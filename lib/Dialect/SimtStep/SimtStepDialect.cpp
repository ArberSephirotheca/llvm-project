#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"

#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/DialectRegistry.h>

#include "SimtStepEnums.cpp.inc"
#include "SimtStepTypes.cpp.inc"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>

#define GET_TYPEDEF_CLASSES
#include "SimtStepTypes.cpp.inc"
#undef GET_TYPEDEF_CLASSES

using namespace mlir;

namespace simt::dialect {

void registerSimtStepDialect(DialectRegistry &registry) {
  registry.insert<::simt::dialect::SimtStepDialect>();
}

void SimtStepDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "SimtStepOps.cpp.inc"
#undef GET_OP_LIST
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "SimtStepTypes.cpp.inc"
#undef GET_TYPEDEF_LIST
      >();
}

} // namespace simt::dialect

#include "SimtStepDialect.cpp.inc"

namespace simt::dialect {

mlir::LogicalResult
MaskType::verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
                 uint64_t width) {
  if (width == 0)
    return emitError() << "mask width must be positive";
  return mlir::success();
}

mlir::LogicalResult ResourceType::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    simt::dialect::MemorySpace memorySpace, mlir::Type elementType) {
  if (!elementType)
    return emitError() << "resource element type must be non-null";
  switch (memorySpace) {
  case simt::dialect::MemorySpace::Generic:
  case simt::dialect::MemorySpace::Global:
  case simt::dialect::MemorySpace::Shared:
  case simt::dialect::MemorySpace::Private:
    return mlir::success();
  }
  return emitError() << "invalid memory space for resource";
}

} // namespace simt::dialect
