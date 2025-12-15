// Shared HLSL emitter for SIMT-Step MLIR kernels.
// Covers the subset used by the fuzz generator/raiser.

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>

namespace simt::raise {

// Emit the first func.func (preferring @main) as HLSL.
mlir::LogicalResult emitModuleAsHlsl(mlir::ModuleOp module,
                                     llvm::raw_ostream &os);

} // namespace simt::raise
