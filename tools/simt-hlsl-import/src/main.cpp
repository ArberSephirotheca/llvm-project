#include "clang/Frontend/CompilerInstance.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
    llvm::InitLLVM initLLVM(argc, argv);

    llvm::cl::OptionCategory toolCategory("simt-hlsl-import options");
    llvm::cl::opt<std::string> inputPath(
        llvm::cl::Positional, llvm::cl::desc("<input HLSL file>"),
        llvm::cl::Optional, llvm::cl::cat(toolCategory));
    llvm::cl::opt<std::string> shaderProfile(
        "profile", llvm::cl::desc("Target shader profile, e.g. ps_6_7"),
        llvm::cl::init("ps_6_7"), llvm::cl::cat(toolCategory));

    llvm::cl::ParseCommandLineOptions(argc, argv,
                                      "SIMT-Step HLSL importer (clang stub)\n");

    clang::CompilerInstance compiler;
    (void)compiler; // suppress unused warning for now

    if (inputPath.empty()) {
        llvm::outs() << "simt-hlsl-import: no input provided (stub)\n";
    } else {
        llvm::outs() << "simt-hlsl-import: would import '" << inputPath
                     << "' for profile '" << shaderProfile << "' (stub)\n";
    }

    return 0;
}
