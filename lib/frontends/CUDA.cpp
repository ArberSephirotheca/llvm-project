#include "simt-step/frontends/CUDA.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>

namespace simt::frontends::cuda {

llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> importToMLIR(mlir::MLIRContext &context,
                                                               llvm::StringRef /*source*/) {
    auto location = mlir::UnknownLoc::get(&context);
    auto module = mlir::ModuleOp::create(location);
    // TODO: integrate CUDA frontend pipeline (clang -> MLIR) and map intrinsics.
    return mlir::OwningOpRef<mlir::ModuleOp>(module);
}

} // namespace simt::frontends::cuda
