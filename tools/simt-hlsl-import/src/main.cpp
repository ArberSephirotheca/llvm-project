#include "simt-hlsl-import/Lowering.h"
#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"
#include "simt-step/Dialect/SimtStep/Transforms.h"

#include "clang/Driver/Driver.h"
#include "clang/Frontend/CompilerInvocation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/OperationSupport.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include <cstdlib>
#include <string_view>
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace simt_hlsl_import;

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);

  llvm::cl::OptionCategory toolCategory("simt-hlsl-import options");
  llvm::cl::opt<std::string> inputPath(
      llvm::cl::Positional, llvm::cl::desc("<input HLSL file>"),
      llvm::cl::Required, llvm::cl::cat(toolCategory));
  llvm::cl::opt<std::string> shaderProfile(
      "profile", llvm::cl::desc("Target shader profile (default: cs_6_7)"),
      llvm::cl::init("cs_6_7"), llvm::cl::cat(toolCategory));
  llvm::cl::opt<bool> normalizeLoopTerminators(
      "normalize-loop-terminators",
      llvm::cl::desc(
          "Preserve simt.normalized.loop_terminators annotations (normalization is always applied)"),
      llvm::cl::init(false), llvm::cl::cat(toolCategory));

  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "SIMT-Step HLSL importer (compute)\n");

  auto bufferOrErr = llvm::MemoryBuffer::getFile(inputPath);
  if (!bufferOrErr) {
    llvm::errs() << "simt-hlsl-import: failed to read input '" << inputPath
                 << "': " << bufferOrErr.getError().message() << "\n";
    return 1;
  }
  llvm::StringRef source = bufferOrErr.get()->getBuffer();

  mlir::DialectRegistry registry;
  registry.insert<mlir::BuiltinDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect, mlir::math::MathDialect,
                  mlir::vector::VectorDialect, simt::dialect::SimtStepDialect>();
  mlir::MLIRContext context(registry);
  context.loadDialect<mlir::BuiltinDialect, mlir::arith::ArithDialect,
                      mlir::func::FuncDialect, mlir::math::MathDialect,
                      mlir::vector::VectorDialect,
                      simt::dialect::SimtStepDialect>();

  TranslationOptions options;
  options.shaderProfile = shaderProfile;
  if (auto clangPath = llvm::sys::findProgramByName("clang"))
    options.resourceDir = clang::driver::Driver::GetResourcesPath(*clangPath);
  else
    options.resourceDir = clang::CompilerInvocation::GetResourcesPath(
        argv[0], reinterpret_cast<void *>(&main));

  auto hasHLSLBuiltins = [](llvm::StringRef root) {
    if (root.empty())
      return false;
    llvm::SmallString<256> candidate(root);
    llvm::sys::path::append(candidate, "include", "hlsl.h");
    if (llvm::sys::fs::exists(candidate))
      return true;
    candidate.assign(root);
    llvm::sys::path::append(candidate, "hlsl.h");
    return llvm::sys::fs::exists(candidate);
  };

#ifdef SIMT_CLANG_HEADERS_DIR
  if (!hasHLSLBuiltins(options.resourceDir)) {
    llvm::SmallString<256> fallback(SIMT_CLANG_HEADERS_DIR);
    llvm::sys::path::append(fallback, "hlsl.h");
    if (llvm::sys::fs::exists(fallback))
      options.extraIncludeDirs.emplace_back(SIMT_CLANG_HEADERS_DIR);
  }
#endif

#ifdef SIMT_HLSL_INTRINSICS_COMPAT
  options.forcedIncludeFiles.emplace_back(SIMT_HLSL_INTRINSICS_COMPAT);
#endif

  if (const char *env = std::getenv("SIMT_IMPORT_DEBUG_RESOURCE"))
    if (std::string_view(env) == "1")
      llvm::errs() << "[simt-hlsl-import] resource-dir="
                   << options.resourceDir << "\n";

  auto result = translateComputeShader(context, inputPath, source, options);
  if (!result) {
    llvm::errs() << "simt-hlsl-import: " << result.error() << "\n";
    return 1;
  }

  mlir::ModuleOp module = result.value().get();

  mlir::PassManager pm(&context);
  pm.enableVerifier(false);
  pm.addNestedPass<mlir::func::FuncOp>(
      simt::dialect::createNormalizeLoopTerminatorsPass());
  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "simt-hlsl-import: failed to normalize loop terminators\n";
    module.dump();
    return 1;
  }

  if (!normalizeLoopTerminators)
    module.walk([](mlir::Operation *op) {
      op->removeAttr("simt.normalized.loop_terminators");
    });

  if (failed(module.verify())) {
    llvm::errs() << "simt-hlsl-import: generated IR failed to verify\n";
    module.dump();
    return 1;
  }

  bool hasDanglingBlock = false;
  module.walk([&](mlir::Operation *op) {
    if (mlir::Block *block = op->getBlock())
      if (!block->getParentOp())
        hasDanglingBlock = true;
  });
  if (hasDanglingBlock) {
    llvm::errs() << "simt-hlsl-import: found operation in block without parent\n";
    module.dump();
    return 1;
  }

  module.print(llvm::outs());
  llvm::outs() << '\n';
  return 0;
}
