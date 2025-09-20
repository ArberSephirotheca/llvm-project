#pragma once

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OperationSupport.h>

namespace simt::core {

enum class Scope {
    Thread,
    Subgroup,
    Workgroup,
};

enum class MemorySemantics {
    None,
    Acquire,
    Release,
    AcquireRelease,
};

mlir::StringAttr getScopeAttr(mlir::MLIRContext &context, Scope scope);
mlir::StringAttr getMemorySemanticsAttr(mlir::MLIRContext &context, MemorySemantics semantics);

mlir::Operation *buildWaveAllOp(mlir::OpBuilder &builder, mlir::Value predicate, mlir::Location loc);

} // namespace simt::core
