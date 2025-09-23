#pragma once

#include <memory>
#include <string>

#include "llvm/ADT/StringRef.h"

#include "simt-hlsl-import/Result.h"

#include "mlir/IR/OwningOpRef.h"

namespace mlir {
class MLIRContext;
class ModuleOp;
} // namespace mlir

namespace simt_hlsl_import {

struct TranslationOptions {
  std::string shaderProfile = "cs";
};

Result<mlir::OwningOpRef<mlir::ModuleOp>>
translateComputeShader(mlir::MLIRContext &context, llvm::StringRef fileName,
                       llvm::StringRef source, const TranslationOptions &opts);

} // namespace simt_hlsl_import
