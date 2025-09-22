#include "simt-step/Dialect/SimtStructured/StructuredDialect.h"

#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/DialectRegistry.h>

#include "StructuredEnums.cpp.inc"

using namespace mlir;

namespace simt::structured {

void registerSimtStructuredDialect(DialectRegistry &registry) {
    registry.insert<::simt::structured::SimtStructDialect>();
}

void SimtStructDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "StructuredOps.cpp.inc"
#undef GET_OP_LIST
    >();
}

} // namespace simt::structured

#include "StructuredDialect.cpp.inc"
