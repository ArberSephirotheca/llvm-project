#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"

#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/DialectRegistry.h>

#include "SimtStepEnums.cpp.inc"

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
}

} // namespace simt::dialect

#include "SimtStepDialect.cpp.inc"
