#include "simt-step/frontends/HLSL.h"

#include "simt-step/core/Dialect.h"

#include <llvm/Support/JSON.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>

namespace simt::frontends::hlsl {

llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> importToMLIR(mlir::MLIRContext &context,
                                                               llvm::StringRef /*source*/) {
    auto location = mlir::UnknownLoc::get(&context);
    auto module = mlir::ModuleOp::create(location);
    // TODO: parse DXIL/AST and emit simt_step + simt.custom ops.
    return mlir::OwningOpRef<mlir::ModuleOp>(module);
}

} // namespace simt::frontends::hlsl
