#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"
#include "simt-step/semantics/SimpleProgram.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/raw_ostream.h>

#include <vector>

int main(int argc, char **argv) {
    llvm::InitLLVM init(argc, argv);
    if (argc < 2) {
        llvm::errs() << "usage: run_operation_to_buffer <input.mlir>\n";
        return 1;
    }

    mlir::DialectRegistry registry;
    simt::dialect::registerSimtStepDialect(registry);
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect>();
    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();

    auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
    if (!module) {
        llvm::errs() << "failed to parse module\n";
        return 1;
    }

    simt::semantics::RunOperationOptions options;
    options.entry = "main";
    options.lanes = 8;
    options.subgroupWidth = 4;
    simt::semantics::BufferOptions buf0;
    buf0.argIndex = 0;
    buf0.size = 16;
    buf0.fill = 0;
    options.perBuffer.push_back(buf0);

    std::vector<simt::semantics::BufferInitEntry> initEntries = {
        {0, 0, 7},
        {0, 3, 42},
    };

    std::vector<simt::semantics::BufferResult> buffers;
    mlir::Operation &op = *module->getOperation();
    if (mlir::failed(simt::semantics::runOperationToBuffers(
            op, /*bufferArgIndices=*/{}, buffers, options, initEntries))) {
        return 1;
    }

    for (const auto &buffer : buffers) {
        for (size_t i = 0; i < buffer.values.size(); ++i)
            llvm::outs() << "buf" << buffer.argIndex << "[" << i
                         << "] = " << buffer.values[i] << "\n";
    }

    return 0;
}
