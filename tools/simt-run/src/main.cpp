#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"
#include "simt-step/frontends/CUDA.h"
#include "simt-step/frontends/HLSL.h"
#include "simt-step/plugins/Registry.h"
#include "simt-step/plugins/examples/ReduceAdd.h"
#include "simt-step/semantics/Interpreter.h"
#include "simt-step/semantics/SemanticsContext.h"

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/raw_ostream.h>

#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>

using namespace simt;

int main(int argc, char **argv) {
    llvm::InitLLVM y(argc, argv);

    llvm::cl::opt<std::string> Model("model", llvm::cl::desc("Interpreter model"), llvm::cl::init("baseline"));
    llvm::cl::opt<unsigned> Width("w", llvm::cl::desc("Subgroup width"), llvm::cl::init(32));

    llvm::cl::ParseCommandLineOptions(argc, argv, "simt-run demo\n");

    mlir::DialectRegistry dialectRegistry;
    simt::dialect::registerSimtStepDialect(dialectRegistry);

    mlir::MLIRContext context(dialectRegistry);
    context.loadDialect<simt::dialect::SimtStepDialect>();
    plugins::Registry registry;

    plugins::examples::registerReduceAdd(registry, Model);

    semantics::SemanticsContext semaContext;
    semaContext.subgroupWidth = Width;
    semaContext.activeMask = 0xFFFF'FFFFULL;

    semantics::Interpreter interpreter(registry, Model);
    llvm::errs() << "(demo) subgroup_width=" << semaContext.subgroupWidth
                 << " active_mask=0x" << llvm::formatv("{0:X}", semaContext.activeMask) << "\n";

    llvm::outs() << "handlers registered for '" << Model << "': reduce_add\n";

    auto moduleOrErr = frontends::hlsl::importToMLIR(context, "// wave stub");
    if (!moduleOrErr) {
        llvm::consumeError(moduleOrErr.takeError());
        llvm::errs() << "failed to build HLSL module stub\n";
        return 1;
    }

    auto cudaModuleOrErr = frontends::cuda::importToMLIR(context, "// cuda stub");
    if (!cudaModuleOrErr) {
        llvm::consumeError(cudaModuleOrErr.takeError());
        llvm::errs() << "failed to build CUDA module stub\n";
        return 1;
    }

    return 0;
}
