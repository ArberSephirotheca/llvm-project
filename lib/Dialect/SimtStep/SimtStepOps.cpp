#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"

#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/PatternMatch.h>

#define GET_OP_CLASSES
#include "SimtStepOps.cpp.inc"
