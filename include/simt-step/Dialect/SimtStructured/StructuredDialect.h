#pragma once

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/Builders.h>

namespace simt::structured {

/// Register the structured SIMT dialect with a dialect registry.
void registerSimtStructuredDialect(::mlir::DialectRegistry &registry);

} // namespace simt::structured

// Generated declarations.
#include "StructuredDialect.h.inc"

#include "StructuredEnums.h.inc"

#define GET_OP_CLASSES
#include "StructuredOps.h.inc"
#undef GET_OP_CLASSES
