#include "simt-hlsl-import/Lowering.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
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

  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "SIMT-Step HLSL importer (compute)\n");

  auto bufferOrErr = llvm::MemoryBuffer::getFile(inputPath);
  if (!bufferOrErr) {
    llvm::errs() << "simt-hlsl-import: failed to read input '" << inputPath
                 << "': " << bufferOrErr.getError().message() << "\n";
    return 1;
  }
  llvm::StringRef source = bufferOrErr.get()->getBuffer();

  mlir::MLIRContext context;

  TranslationOptions options;
  options.shaderProfile = shaderProfile;

  auto result = translateComputeShader(context, inputPath, source, options);
  if (!result) {
    llvm::errs() << "simt-hlsl-import: " << result.error() << "\n";
    return 1;
  }

  result.value()->print(llvm::outs());
  llvm::outs() << '\n';
  return 0;
}
