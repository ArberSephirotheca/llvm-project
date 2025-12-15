#include "HlslEmitter.h"

#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Parser/Parser.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/raw_ostream.h>

using namespace mlir;

int main(int argc, char **argv) {
    llvm::InitLLVM y(argc, argv);
    llvm::cl::opt<std::string> inputFile(llvm::cl::Positional,
                                         llvm::cl::desc("<input mlir>"),
                                         llvm::cl::init("-"));
    llvm::cl::ParseCommandLineOptions(argc, argv, "simt-step raise to HLSL\n");

    DialectRegistry registry;
    simt::dialect::registerSimtStepDialect(registry);
    registry.insert<arith::ArithDialect, func::FuncDialect>();
    MLIRContext context(registry);
    context.loadAllAvailableDialects();

    auto module = parseSourceFile<ModuleOp>(inputFile, &context);
    if (!module) {
        llvm::errs() << "failed to parse module\n";
        return 1;
    }

    if (failed(simt::raise::emitModuleAsHlsl(*module, llvm::outs()))) {
        llvm::errs() << "failed to raise module\n";
        return 1;
    }

    return 0;
}
