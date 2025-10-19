#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);

  llvm::cl::OptionCategory category("simt-step-parse options");
  llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                           llvm::cl::desc("<input mlir file>"),
                                           llvm::cl::Required,
                                           llvm::cl::cat(category));
  llvm::cl::opt<bool> emitModule(
      "emit-module", llvm::cl::desc("Print the parsed module to stdout"),
      llvm::cl::init(false), llvm::cl::cat(category));

  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Simt-Step MLIR parsing utility\n");

  mlir::DialectRegistry registry;
  registry.insert<mlir::BuiltinDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect, simt::dialect::SimtStepDialect>();

  mlir::MLIRContext context(registry);
  context.loadDialect<mlir::BuiltinDialect, mlir::arith::ArithDialect,
                      mlir::func::FuncDialect, simt::dialect::SimtStepDialect>();

  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (!fileOrErr) {
    llvm::WithColor::error(llvm::errs())
        << "failed to open '" << inputFilename
        << "': " << fileOrErr.getError().message() << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  mlir::SourceMgrDiagnosticHandler diagHandler(sourceMgr, &context);

  auto module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::WithColor::error(llvm::errs())
        << "failed to parse '" << inputFilename << "' as MLIR\n";
    return 1;
  }

  if (emitModule)
    module->print(llvm::outs());

  llvm::outs() << "Parsed module: " << inputFilename << "\n";
  return 0;
}
