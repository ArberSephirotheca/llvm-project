#pragma once

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>

namespace simt::dialect {

/// Register the SIMT-Step dialect with the given MLIR context.
void registerSimtStepDialect(::mlir::DialectRegistry &registry);

} // namespace simt::dialect

// Generated declarations.
#include "SimtStepDialect.h.inc"

#define GET_OP_CLASSES
#include "SimtStepOps.h.inc"
#undef GET_OP_CLASSES
