// Simple driver that generates a deterministic SIMT-Step kernel and runs it
// through the CPS interpreter, printing per-lane completion.

#include "SimtProgramGenerator.h"

#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"
#include "simt-step/semantics/SimpleProgram.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/raw_ostream.h>

using namespace mlir;

namespace {
std::string formatMask(std::uint64_t mask, unsigned width) {
    std::string s;
    s.reserve(width + 2);
    s.append("0b");
    for (int i = static_cast<int>(width) - 1; i >= 0; --i)
        s.push_back((mask & (1ull << i)) ? '1' : '0');
    return s;
}
} // namespace

int main(int argc, char **argv) {
    llvm::InitLLVM y(argc, argv);

    llvm::cl::opt<unsigned> numLanes(
        "lanes", llvm::cl::desc("Number of lanes (mask bits)"),
        llvm::cl::init(4));
    llvm::cl::opt<bool> dumpIR("print-ir", llvm::cl::desc("Print generated MLIR"),
                               llvm::cl::init(false));
    llvm::cl::ParseCommandLineOptions(argc, argv, "simt-step fuzz driver\n");

    mlir::DialectRegistry registry;
    simt::dialect::registerSimtStepDialect(registry);
    registry.insert<arith::ArithDialect, func::FuncDialect>();
    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();

    simt::fuzz::GeneratorConfig cfg;
    cfg.numThreads = {static_cast<std::int64_t>(numLanes), 1, 1};
    auto module = simt::fuzz::createDeterministicIfLoopModule(context, cfg);
    if (!module) {
        llvm::errs() << "failed to build module\n";
        return 1;
    }

    auto func = module->lookupSymbol<func::FuncOp>("main");
    if (!func) {
        llvm::errs() << "generated module missing @main\n";
        return 1;
    }
    if (dumpIR) {
        module->print(llvm::outs());
        llvm::outs() << "\n";
    }

    auto &entry = func.getBody().front();
    simt::semantics::SimpleProgramRunner runner;
    simt::semantics::SemanticsContext semaCtx;
    unsigned width = std::min<unsigned>(64, std::max<unsigned>(1, numLanes));
    semaCtx.activeMask =
        width >= 64 ? ~0ull : ((1ull << static_cast<std::uint64_t>(width)) - 1ull);

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
        if (!waveCtx.blocks.empty()) {
            unsigned maskWidth = std::min<unsigned>(64, std::max<unsigned>(1, numLanes));
            for (const auto &blkIt : waveCtx.blocks) {
                const auto &blk = blkIt.second;
                llvm::outs() << "  Block " << blkIt.first.block
                             << " seq=" << blkIt.first.sequenceId
                             << " expected=" << formatMask(blk.expectedMask, maskWidth)
                             << " active=" << formatMask(blk.activeMask, maskWidth)
                             << " completed=" << formatMask(blk.completedMask, maskWidth)
                             << "\n";
            }
        }
    }

    return 0;
}
