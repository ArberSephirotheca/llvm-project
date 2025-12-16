// Simple driver that generates a SIMT-Step kernel and optionally raises it to
// HLSL or runs it through the CPS interpreter.

#include "SimtProgramGenerator.h"

#include "HlslEmitter.h"
#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"
#include "simt-step/semantics/SimpleProgram.h"
#include "simt-step/semantics/SimpleSemantics.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Support/LogicalResult.h>

#include <algorithm>
#include <cstdlib>
#include <vector>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/raw_ostream.h>

using namespace mlir;

int main(int argc, char **argv) {

    llvm::cl::opt<unsigned> numLanes(
        "lanes", llvm::cl::desc("Number of lanes (mask bits)"),
        llvm::cl::init(4));
    llvm::cl::opt<bool> dumpIR("print-ir", llvm::cl::desc("Print generated MLIR"),
                               llvm::cl::init(false));
    llvm::cl::opt<bool> dumpHlsl("raise-hlsl",
                                 llvm::cl::desc("Print raised HLSL for generated module"),
                                 llvm::cl::init(false));
    llvm::cl::opt<bool> runInterp("run", llvm::cl::desc("Run generated module in interpreter"),
                                  llvm::cl::init(false));
    llvm::cl::opt<std::uint64_t> seedOpt("seed", llvm::cl::desc("Seed for RNG (0=deterministic)"),
                                         llvm::cl::init(0));
    llvm::cl::ParseCommandLineOptions(argc, argv, "simt-step fuzz driver\n");

    mlir::DialectRegistry registry;
    simt::dialect::registerSimtStepDialect(registry);
    registry.insert<arith::ArithDialect, func::FuncDialect>();
    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();
    (void)context.getOrLoadDialect<simt::dialect::SimtStepDialect>();
    (void)context.getOrLoadDialect<arith::ArithDialect>();
    (void)context.getOrLoadDialect<func::FuncDialect>();

    simt::fuzz::GeneratorConfig cfg;
    cfg.numThreads = {static_cast<std::int64_t>(numLanes), 1, 1};
    cfg.seed = seedOpt;
    llvm::errs() << "[fuzz] generating module...\n";
    llvm::errs().flush();
    auto module = simt::fuzz::createRicherRandomModule(context, cfg);
    if (!module) {
        llvm::errs() << "failed to build module\n";
        return 1;
    }

    llvm::errs() << "[fuzz] module built.\n";
    llvm::errs().flush();
    auto func = module->lookupSymbol<func::FuncOp>("main");
    if (!func) {
        llvm::errs() << "generated module missing @main\n";
        return 1;
    }
    if (dumpIR) {
        module->print(llvm::outs());
        llvm::outs() << "\n";
    }

    if (dumpHlsl) {
        if (failed(simt::raise::emitModuleAsHlsl(*module, llvm::outs())))
            return 1;
    }

    if (!runInterp)
        return 0;

    if (std::getenv("SIMT_CPS_DEBUG"))
        simt::semantics::EnableCPSDebugLogs = true;

    auto &entry = func.getBody().front();
    simt::semantics::SimpleProgramRunner runner;
    simt::semantics::SemanticsContext semaCtx;
    unsigned width = std::min<unsigned>(64, std::max<unsigned>(1, numLanes));
    semaCtx.activeMask =
        width >= 64 ? ~0ull : ((1ull << static_cast<std::uint64_t>(width)) - 1ull);
    simt::semantics::SimpleSemantics::clearMemory();

    if (llvm::Error err = runner.runBlock(&entry, semaCtx)) {
        llvm::errs() << "run failed: " << llvm::toString(std::move(err)) << "\n";
        return 1;
    }

    const auto &state = runner.state();
    for (const auto &waveIt : state.waves) {
        llvm::outs() << "Wave " << waveIt.first << "\n";
        const auto &waveCtx = waveIt.second;
        for (const auto &laneIt : waveCtx.lanes) {
            const auto &laneCtx = laneIt.second;
            llvm::outs() << "  Lane " << laneIt.first
                         << " returned=" << laneCtx.hasReturned;
            if (laneCtx.returnValue)
                llvm::outs() << " value=" << laneCtx.returnValue->asInt64();
            if (laneCtx.currentBlock)
                llvm::outs() << " block=" << laneCtx.currentBlock->block
                             << " seq=" << laneCtx.currentBlock->sequenceId;
            llvm::outs() << "\n";
        }
    }

    const auto &mem = simt::semantics::SimpleSemantics::memory();
    if (!mem.empty()) {
        llvm::outs() << "Memory:\n";
        for (const auto &resIt : mem) {
            std::string bufName = "res";
            if (auto barg = mlir::dyn_cast<BlockArgument>(resIt.first))
                bufName = "buf" + std::to_string(barg.getArgNumber());
            std::vector<std::pair<int64_t, simt::semantics::SemValue>> entries;
            entries.reserve(resIt.second.size());
            for (const auto &kv : resIt.second)
                entries.push_back(kv);
            std::sort(entries.begin(), entries.end(),
                      [](const auto &a, const auto &b) { return a.first < b.first; });
            for (const auto &kv : entries)
                llvm::outs() << "  " << bufName << "[" << kv.first << "] = "
                             << kv.second.asInt64() << "\n";
        }
    }

    return 0;
}
