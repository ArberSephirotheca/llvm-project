#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"
#include "simt-step/semantics/SimpleProgram.h"
#include "simt-step/semantics/TraceJsonWriter.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Value.h>
#include <mlir/Parser/Parser.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/raw_ostream.h>

#include <algorithm>
#include <vector>

using namespace mlir;

int main(int argc, char **argv) {
    llvm::InitLLVM y(argc, argv);
    llvm::cl::opt<std::string> inputFile(llvm::cl::Positional,
                                         llvm::cl::desc("<input mlir>"),
                                         llvm::cl::init("-"));
    llvm::cl::opt<std::string> funcName(
        "func", llvm::cl::desc("Entry function name"),
        llvm::cl::init("main"));
    llvm::cl::opt<unsigned> numLanes(
        "lanes", llvm::cl::desc("Number of lanes to execute"),
        llvm::cl::init(4));
    llvm::cl::opt<bool> dumpIR(
        "print-ir", llvm::cl::desc("Print parsed IR before running"),
        llvm::cl::init(false));
    llvm::cl::opt<bool> collectiveControlFlow(
        "collective-cf",
        llvm::cl::desc("Make control-flow ops collective before split"),
        llvm::cl::init(false));
    llvm::cl::opt<bool> syncControlFlow(
        "sync-cf",
        llvm::cl::desc("Make control-flow ops synchronous (barriered)"),
        llvm::cl::init(false));
    llvm::cl::opt<bool> syncMemory(
        "sync-mem",
        llvm::cl::desc("Make buffer load/store synchronous (barriered)"),
        llvm::cl::init(false));
    llvm::cl::opt<bool> collectiveMemory(
        "collective-mem",
        llvm::cl::desc("Make buffer load/store collective"),
        llvm::cl::init(false));
    llvm::cl::opt<std::string> traceFile(
        "trace-file", llvm::cl::desc("Write interpreter trace to JSONL file"),
        llvm::cl::init(""));
    llvm::cl::ParseCommandLineOptions(argc, argv, "simt-step runner\n");

    DialectRegistry registry;
    simt::dialect::registerSimtStepDialect(registry);
    registry.insert<arith::ArithDialect, func::FuncDialect>();
    MLIRContext context(registry);
    context.loadAllAvailableDialects();
    (void)context.getOrLoadDialect<simt::dialect::SimtStepDialect>();
    (void)context.getOrLoadDialect<arith::ArithDialect>();
    (void)context.getOrLoadDialect<func::FuncDialect>();

    auto module = parseSourceFile<ModuleOp>(inputFile, &context);
    if (!module) {
        llvm::errs() << "failed to parse module\n";
        return 1;
    }

    if (dumpIR) {
        module->print(llvm::outs());
        llvm::outs() << "\n";
    }

    auto func = module->lookupSymbol<func::FuncOp>(funcName);
    if (!func) {
        llvm::errs() << "module missing @" << funcName << "\n";
        return 1;
    }

    if (collectiveControlFlow && syncControlFlow) {
        llvm::errs() << "error: --collective-cf conflicts with --sync-cf\n";
        return 1;
    }
    if (collectiveMemory && syncMemory) {
        llvm::errs() << "error: --collective-mem conflicts with --sync-mem\n";
        return 1;
    }

    simt::semantics::ExecutionPolicy execPolicy;
    if (collectiveControlFlow)
        execPolicy.controlFlow = simt::semantics::ExecutionMode::Collective;
    else if (syncControlFlow)
        execPolicy.controlFlow = simt::semantics::ExecutionMode::Synchronous;
    if (collectiveMemory)
        execPolicy.memoryOps = simt::semantics::ExecutionMode::Collective;
    else if (syncMemory)
        execPolicy.memoryOps = simt::semantics::ExecutionMode::Synchronous;

    simt::semantics::SemanticsContext semaCtx;
    unsigned width = std::min<unsigned>(64, std::max<unsigned>(1, numLanes));
    semaCtx.activeMask =
        width >= 64 ? ~0ull : ((1ull << static_cast<std::uint64_t>(width)) - 1ull);
    semaCtx.policy = &execPolicy;

    simt::semantics::SimpleSemantics::clearMemory();
    simt::semantics::SimpleProgramRunner runner;
    std::unique_ptr<simt::semantics::TraceJsonWriter> traceWriter;
    if (!traceFile.empty()) {
        traceWriter = std::make_unique<simt::semantics::TraceJsonWriter>(traceFile);
        runner.setTraceSink(traceWriter.get());
    }

    auto &entry = func.getBody().front();
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
                llvm::outs() << "  " << bufName << "[" << kv.first
                             << "] = " << kv.second.asInt64() << "\n";
        }
    }

    return 0;
}
