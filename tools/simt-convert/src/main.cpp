#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"
#include "simt-step/frontends/CUDA.h"
#include "simt-step/frontends/HLSL.h"

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/WithColor.h>

#include <memory>

#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>

using namespace simt;

namespace {

llvm::Expected<std::string> loadInput(llvm::StringRef path) {
    if (path == "-") {
        auto bufferOrErr = llvm::MemoryBuffer::getSTDIN();
        if (!bufferOrErr) {
            return llvm::errorCodeToError(bufferOrErr.getError());
        }
        return std::string(bufferOrErr.get()->getBuffer());
    }
    auto bufferOrErr = llvm::MemoryBuffer::getFile(path);
    if (!bufferOrErr) {
        return llvm::errorCodeToError(bufferOrErr.getError());
    }
    return std::string(bufferOrErr.get()->getBuffer());
}

} // namespace

int main(int argc, char **argv) {
    llvm::InitLLVM y(argc, argv);

    llvm::cl::opt<std::string> Frontend("frontend", llvm::cl::desc("Frontend to use"),
                                        llvm::cl::init("hlsl"));
    llvm::cl::opt<std::string> Input("input", llvm::cl::desc("Input file (- for stdin)"),
                                     llvm::cl::init("-"));
    llvm::cl::opt<std::string> Output("output", llvm::cl::desc("Output file (- for stdout)"),
                                      llvm::cl::init("-"));

    llvm::cl::ParseCommandLineOptions(argc, argv, "simt-convert");

    auto inputOrErr = loadInput(Input);
    if (!inputOrErr) {
        llvm::WithColor::error() << "failed to read input: " << llvm::toString(inputOrErr.takeError()) << "\n";
        return 1;
    }

    mlir::DialectRegistry dialectRegistry;
    simt::dialect::registerSimtStepDialect(dialectRegistry);

    mlir::MLIRContext context(dialectRegistry);
    context.loadDialect<simt::dialect::SimtStepDialect>();
    context.loadAllAvailableDialects();

    auto moduleOrErr = [&]() -> llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> {
        if (Frontend == "cuda") {
            return frontends::cuda::importToMLIR(context, *inputOrErr);
        }
        if (Frontend == "hlsl") {
            return frontends::hlsl::importToMLIR(context, *inputOrErr);
        }
        return llvm::make_error<llvm::StringError>(
            "unknown frontend", llvm::inconvertibleErrorCode());
    }();

    if (!moduleOrErr) {
        llvm::WithColor::error() << "failed to import module: " << llvm::toString(moduleOrErr.takeError()) << "\n";
        return 1;
    }

    std::unique_ptr<llvm::raw_fd_ostream> fileStream;
    llvm::raw_ostream *os = nullptr;
    if (Output == "-") {
        os = &llvm::outs();
    } else {
        std::error_code ec;
        fileStream = std::make_unique<llvm::raw_fd_ostream>(Output, ec, llvm::sys::fs::OF_Text);
        if (ec) {
            llvm::WithColor::error() << "failed to open output: " << ec.message() << "\n";
            return 1;
        }
        os = fileStream.get();
    }

    (*moduleOrErr)->print(*os);
    *os << '\n';

    if (fileStream) {
        fileStream->flush();
    }

    return 0;
}
