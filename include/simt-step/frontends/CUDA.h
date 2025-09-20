#pragma once

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>

#include <mlir/IR/BuiltinOps.h>

namespace mlir {
class MLIRContext;
}

namespace simt::frontends::cuda {

llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> importToMLIR(mlir::MLIRContext &context,
                                                               llvm::StringRef source);

} // namespace simt::frontends::cuda
