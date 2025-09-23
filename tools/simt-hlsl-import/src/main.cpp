#include "simt-hlsl-import/Lowering.h"

#include "clang/Driver/Driver.h"
#include "clang/Frontend/CompilerInvocation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

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

  if (const char *env = std::getenv("SIMT_IMPORT_DEBUG_RESOURCE"))
    if (std::string_view(env) == "1")
      llvm::errs() << "[simt-hlsl-import] resource-dir="
                   << options.resourceDir << "\n";

  auto result = translateComputeShader(context, inputPath, source, options);
  if (!result) {
    llvm::errs() << "simt-hlsl-import: " << result.error() << "\n";
    return 1;
  }

  result.value()->print(llvm::outs());
  llvm::outs() << '\n';
  return 0;
}
