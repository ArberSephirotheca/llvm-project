// Simple driver that generates a SIMT-Step kernel and optionally raises it to
// HLSL or runs it through the CPS interpreter.

#include "SimtProgramGenerator.h"

#include "HlslEmitter.h"
#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"
#include "simt-step/semantics/SimpleProgram.h"
#include "simt-step/semantics/SimpleSemantics.h"
#include "simt-step/semantics/TraceJsonWriter.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Support/LogicalResult.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <vector>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/raw_ostream.h>

using namespace mlir;

namespace {
struct LaneSnapshot {
    simt::semantics::WaveId wave = 0;
    simt::semantics::LaneId lane = 0;
    bool hasReturned = false;
    std::optional<int64_t> returnValue;
    const mlir::Block *block = nullptr;
    std::uint32_t sequenceId = 0;
};

struct BufferSnapshot {
    unsigned argIndex = 0;
    std::vector<std::pair<int64_t, int64_t>> entries;
};

struct RunSnapshot {
    std::vector<LaneSnapshot> lanes;
    std::vector<BufferSnapshot> buffers;
};

RunSnapshot captureSnapshot(const simt::semantics::SimpleProgramRunner &runner) {
    RunSnapshot snapshot;
    const auto &state = runner.state();
    snapshot.lanes.reserve(state.waves.size() * 8);
    for (const auto &waveIt : state.waves) {
        simt::semantics::WaveId waveId = waveIt.first;
        const auto &waveCtx = waveIt.second;
        for (const auto &laneIt : waveCtx.lanes) {
            LaneSnapshot laneSnap;
            laneSnap.wave = waveId;
            laneSnap.lane = laneIt.first;
            laneSnap.hasReturned = laneIt.second.hasReturned;
            if (laneIt.second.returnValue)
                laneSnap.returnValue = laneIt.second.returnValue->asInt64();
            if (laneIt.second.currentBlock) {
                laneSnap.block = laneIt.second.currentBlock->block;
                laneSnap.sequenceId = laneIt.second.currentBlock->sequenceId;
            }
            snapshot.lanes.push_back(laneSnap);
        }
    }
    std::sort(snapshot.lanes.begin(), snapshot.lanes.end(),
              [](const LaneSnapshot &a, const LaneSnapshot &b) {
                  if (a.wave != b.wave)
                      return a.wave < b.wave;
                  return a.lane < b.lane;
              });

    const auto &mem = simt::semantics::SimpleSemantics::memory();
    for (const auto &resIt : mem) {
        auto barg = mlir::dyn_cast<BlockArgument>(resIt.first);
        if (!barg)
            continue;
        BufferSnapshot buf;
        buf.argIndex = barg.getArgNumber();
        buf.entries.reserve(resIt.second.size());
        for (const auto &kv : resIt.second)
            buf.entries.emplace_back(kv.first, kv.second.asInt64());
        std::sort(buf.entries.begin(), buf.entries.end(),
                  [](const auto &a, const auto &b) { return a.first < b.first; });
        snapshot.buffers.push_back(std::move(buf));
    }
    std::sort(snapshot.buffers.begin(), snapshot.buffers.end(),
              [](const BufferSnapshot &a, const BufferSnapshot &b) {
                  return a.argIndex < b.argIndex;
              });

    return snapshot;
}

bool compareSnapshots(const RunSnapshot &a,
                      const RunSnapshot &b,
                      std::string &reason) {
    if (a.lanes.size() != b.lanes.size()) {
        reason = "lane snapshot size mismatch";
        return false;
    }
    for (std::size_t i = 0; i < a.lanes.size(); ++i) {
        const auto &lhs = a.lanes[i];
        const auto &rhs = b.lanes[i];
        if (lhs.wave != rhs.wave || lhs.lane != rhs.lane) {
            reason = "lane order mismatch";
            return false;
        }
        if (lhs.hasReturned != rhs.hasReturned ||
            lhs.returnValue != rhs.returnValue) {
            reason = "lane return mismatch";
            return false;
        }
    }
    if (a.buffers.size() != b.buffers.size()) {
        reason = "buffer count mismatch";
        return false;
    }
    for (std::size_t i = 0; i < a.buffers.size(); ++i) {
        const auto &lhs = a.buffers[i];
        const auto &rhs = b.buffers[i];
        if (lhs.argIndex != rhs.argIndex) {
            reason = "buffer arg mismatch";
            return false;
        }
        if (lhs.entries != rhs.entries) {
            reason = "buffer contents mismatch";
            return false;
        }
    }
    return true;
}
} // namespace

int main(int argc, char **argv) {

    llvm::cl::opt<unsigned> numLanes(
        "lanes", llvm::cl::desc("Number of lanes (mask bits)"),
        llvm::cl::init(4));
    llvm::cl::opt<unsigned> subgroupWidth(
        "subgroup-width", llvm::cl::desc("Subgroup width"),
        llvm::cl::init(8));
    llvm::cl::opt<bool> dumpIR("print-ir", llvm::cl::desc("Print generated MLIR"),
                               llvm::cl::init(false));
    llvm::cl::opt<bool> dumpHlsl("raise-hlsl",
                                 llvm::cl::desc("Print raised HLSL for generated module"),
                                 llvm::cl::init(false));
    llvm::cl::opt<bool> runInterp("run", llvm::cl::desc("Run generated module in interpreter"),
                                  llvm::cl::init(false));
    llvm::cl::opt<std::uint64_t> seedOpt("seed", llvm::cl::desc("Seed for RNG (0=deterministic)"),
                                         llvm::cl::init(0));
    llvm::cl::opt<unsigned> trials(
        "trials",
        llvm::cl::desc("Number of randomized scheduling trials"),
        llvm::cl::init(1));
    llvm::cl::opt<std::uint64_t> scheduleSeed(
        "schedule-seed",
        llvm::cl::desc("Seed for randomized scheduler (0 = deterministic order)"),
        llvm::cl::init(0));
    llvm::cl::opt<bool> randomSchedule(
        "random-schedule",
        llvm::cl::desc("Randomize scheduler order"),
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
    cfg.subgroupWidth = std::max<unsigned>(1, subgroupWidth);
    cfg.seed = seedOpt;
    llvm::errs() << "[fuzz] generating module...\n";
    llvm::errs().flush();
    mlir::OwningOpRef<mlir::ModuleOp> module =
        simt::fuzz::createRicherRandomModule(context, cfg);
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
    bool useRandomSchedule = randomSchedule || trials > 1;
    if (trials < 1) {
        llvm::errs() << "error: --trials must be >= 1\n";
        return 1;
    }
    if (trials > 1 && traceFile.empty() == false) {
        llvm::errs() << "error: --trace-file is only supported with --trials=1\n";
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
    semaCtx.subgroupWidth = std::max<unsigned>(1, subgroupWidth);
    semaCtx.policy = &execPolicy;
    auto runTrial = [&](std::uint64_t seed,
                        simt::semantics::TraceJsonWriter *traceWriter)
        -> std::optional<RunSnapshot> {
        simt::semantics::SimpleProgramRunner runner;
        if (traceWriter)
            runner.setTraceSink(traceWriter);
        if (useRandomSchedule) {
            runner.setScheduleMode(
                simt::semantics::SimpleProgramRunner::ScheduleMode::Randomized);
            runner.setScheduleSeed(seed);
        }
        simt::semantics::SimpleSemantics::clearMemory();
        if (llvm::Error err = runner.runBlock(&entry, semaCtx)) {
            llvm::errs() << "run failed: " << llvm::toString(std::move(err)) << "\n";
            return std::nullopt;
        }
        return captureSnapshot(runner);
    };

    std::unique_ptr<simt::semantics::TraceJsonWriter> traceWriter;
    if (!traceFile.empty()) {
        traceWriter = std::make_unique<simt::semantics::TraceJsonWriter>(traceFile);
    }

    std::uint64_t baseSeed = scheduleSeed;
    if (useRandomSchedule && baseSeed == 0) {
        llvm::errs() << "note: randomized scheduling enabled; using "
                        "--schedule-seed=1\n";
        baseSeed = 1;
    }
    auto baseline = runTrial(baseSeed, traceWriter.get());
    if (!baseline)
        return 1;
    for (unsigned trial = 1; trial < trials; ++trial) {
        auto snap = runTrial(baseSeed + trial, nullptr);
        if (!snap)
            return 1;
        std::string reason;
        if (!compareSnapshots(*baseline, *snap, reason)) {
            llvm::errs() << "determinism oracle failed: " << reason
                         << " (trial " << trial << ")\n";
            return 2;
        }
    }
    if (trials > 1)
        llvm::errs() << "[oracle] " << trials
                     << " trials consistent\n";

    simt::semantics::WaveId currentWave = std::numeric_limits<std::uint32_t>::max();
    for (const auto &lane : baseline->lanes) {
        if (lane.wave != currentWave) {
            currentWave = lane.wave;
            llvm::outs() << "Wave " << lane.wave << "\n";
        }
        llvm::outs() << "  Lane " << lane.lane
                     << " returned=" << lane.hasReturned;
        if (lane.returnValue)
            llvm::outs() << " value=" << *lane.returnValue;
        if (lane.block)
            llvm::outs() << " block=" << lane.block
                         << " seq=" << lane.sequenceId;
        llvm::outs() << "\n";
    }

    if (!baseline->buffers.empty()) {
        llvm::outs() << "Memory:\n";
        for (const auto &buf : baseline->buffers) {
            std::string bufName = "buf" + std::to_string(buf.argIndex);
            for (const auto &kv : buf.entries)
                llvm::outs() << "  " << bufName << "[" << kv.first << "] = "
                             << kv.second << "\n";
        }
    }

    return 0;
}
