#include "simt-step/semantics/SimpleProgram.h"
#include "simt-step/semantics/SemanticsContext.h"

#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"

#include <cstring>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <llvm/Support/Error.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

using namespace simt::semantics;

namespace {

static void dumpInterpreterState(const SimpleProgramRunner &runner);

llvm::Expected<int> run(llvm::StringRef path, SimpleProgramRunner &runner,
                        SemanticsContext &semaCtx, LaneId resultLane) {
    mlir::DialectRegistry registry;
    simt::dialect::registerSimtStepDialect(registry);
    registry.insert<mlir::BuiltinDialect, mlir::arith::ArithDialect,
                    mlir::func::FuncDialect, simt::dialect::SimtStepDialect>();

    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();
    (void)context.getOrLoadDialect<mlir::BuiltinDialect>();
    (void)context.getOrLoadDialect<mlir::arith::ArithDialect>();
    (void)context.getOrLoadDialect<mlir::func::FuncDialect>();
    (void)context.getOrLoadDialect<simt::dialect::SimtStepDialect>();

    mlir::OpBuilder builder(&context);
    mlir::OwningOpRef<mlir::ModuleOp> module;
    mlir::func::FuncOp func;

    if (path.empty() || path == "-") {
        module = mlir::ModuleOp::create(builder.getUnknownLoc());
        auto funcType = builder.getFunctionType({}, {});
        func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(),
                                                  "kernel", funcType);
        module->push_back(func);
        auto *entry = func.addEntryBlock();
        builder.setInsertionPointToStart(entry);
        auto loc = builder.getUnknownLoc();
        auto lhs = builder.create<mlir::arith::ConstantIntOp>(loc, 1, 32);
        auto rhs = builder.create<mlir::arith::ConstantIntOp>(loc, 2, 32);
        (void)builder.create<mlir::arith::AddIOp>(loc, lhs, rhs);
        auto cond = builder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
        auto ifOp = builder.create<simt::dialect::IfOp>(loc, cond, /*withElseRegion=*/true);

        {
            mlir::OpBuilder thenBuilder(ifOp.getThenRegion());
            thenBuilder.setInsertionPointToEnd(&ifOp.getThenRegion().front());
            thenBuilder.create<simt::dialect::LaneIdOp>(loc, thenBuilder.getIndexType());
            thenBuilder.create<simt::dialect::YieldOp>(loc);
        }

        {
            mlir::OpBuilder elseBuilder(ifOp.getElseRegion());
            elseBuilder.setInsertionPointToEnd(&ifOp.getElseRegion().front());
            elseBuilder.create<simt::dialect::YieldOp>(loc);
        }

        builder.setInsertionPointAfter(ifOp);
        builder.create<mlir::func::ReturnOp>(loc);
    } else {
        module = mlir::parseSourceFile<mlir::ModuleOp>(path, &context);
        if (!module)
            return llvm::make_error<llvm::StringError>(
                "failed to parse module", llvm::inconvertibleErrorCode());

        for (mlir::Operation &op : module->getBody()->getOperations())
            if (auto f = mlir::dyn_cast<mlir::func::FuncOp>(op)) {
                func = f;
                break;
            }

        if (!func)
            return llvm::make_error<llvm::StringError>(
                "module missing func.func", llvm::inconvertibleErrorCode());
    }

    mlir::Block &block = func.getBody().front();

    if (llvm::Error err = runner.runBlock(&block, semaCtx))
        return std::move(err);

    const auto &state = runner.state();
    auto waveIt = state.waves.find(0);
    if (waveIt == state.waves.end())
        return llvm::make_error<llvm::StringError>(
            "missing wave context", llvm::inconvertibleErrorCode());

    const auto &laneCtx = waveIt->second.lanes.lookup(resultLane);
    if (!laneCtx.returnValue) {
        dumpInterpreterState(runner);
        return llvm::make_error<llvm::StringError>(
            "lane produced no value", llvm::inconvertibleErrorCode());
    }

    int result = static_cast<int>(laneCtx.returnValue->asInt64());
    llvm::outs() << result << "\n";
    return result;
}

static void dumpInterpreterState(const SimpleProgramRunner &runner) {
    const auto &state = runner.state();
    llvm::outs() << "readyQueue=" << state.readyQueue.size() << "\n";
    for (const auto &waveIt : state.waves) {
        llvm::outs() << "Wave " << waveIt.first << "\n";
        const auto &waveCtx = waveIt.second;
        if (!waveCtx.mergeStack.empty()) {
            llvm::outs() << "  MergeStack:\n";
            for (const auto &entry : llvm::enumerate(waveCtx.mergeStack)) {
                llvm::outs() << "    [" << entry.index() << "] parent=" << entry.value().parent.block
                             << " exp=0x" << llvm::format_hex(entry.value().expectedMask, 10)
                             << " completed=0x" << llvm::format_hex(entry.value().completedMask, 10)
                             << " children=" << entry.value().pendingChildren.size() << "\n";
            }
        }
        for (const auto &blockIt : waveCtx.blocks) {
            const auto &key = blockIt.first;
            const auto &block = blockIt.second;
            std::string label;
            if (key.block) {
                mlir::Block *mutableBlock =
                    const_cast<mlir::Block *>(key.block);
                if (!mutableBlock->empty())
                    label =
                        mutableBlock->front().getName().getStringRef().str();
            }
            llvm::outs() << "  Block " << key.block << " seq "
                         << key.sequenceId << " exp=0x"
                         << llvm::format_hex(block.expectedMask, 10)
                         << " act=0x"
                         << llvm::format_hex(block.activeMask, 10)
                         << " completed=0x"
                         << llvm::format_hex(block.completedMask, 10)
                         << " pending=" << block.pendingOps.size()
                         << " envs=" << block.valueEnvs.size();
            if (!label.empty())
                llvm::outs() << " firstOp=" << label;
            llvm::outs() << "\n";
        }
        for (const auto &laneIt : waveCtx.lanes) {
            const auto &laneCtx = laneIt.second;
            llvm::outs() << "    Lane " << laneIt.first
                         << " hasReturned=" << laneCtx.hasReturned;
            if (laneCtx.returnValue)
                llvm::outs() << " value=" << laneCtx.returnValue->asInt64();
            if (laneCtx.currentBlock)
                llvm::outs() << " currentBlock=" << laneCtx.currentBlock->block
                             << " seq=" << laneCtx.currentBlock->sequenceId;
            llvm::outs() << "\n";
        }
    }
}

} // namespace

int main(int argc, char **argv) {
    bool listDialects = false;
    bool dumpState = false;
    uint64_t maskOverride = 0;
    LaneId resultLane = 0;
    llvm::StringRef path;
    for (int i = 1; i < argc; ++i) {
        llvm::StringRef arg(argv[i]);
        if (arg == "--list-dialects") {
            listDialects = true;
            continue;
        }
        if (arg == "--dump-blocks") {
            dumpState = true;
            continue;
        }
        if (arg.starts_with("--mask=")) {
            arg = arg.drop_front(strlen("--mask="));
            unsigned long long value = 0;
            if (!arg.getAsInteger(0, value))
                maskOverride = value;
            continue;
        }
        if (arg.starts_with("--lane=")) {
            arg = arg.drop_front(strlen("--lane="));
            unsigned long long value = 0;
            if (!arg.getAsInteger(0, value))
                resultLane = static_cast<LaneId>(value);
            continue;
        }
        if (path.empty())
            path = argv[i];
    }

    SemanticsContext semaCtx;
    semaCtx.subgroupWidth = 32;
    semaCtx.activeMask =
        maskOverride ? maskOverride : ((1ull << 4) - 1ull);
    semaCtx.laneId = resultLane;

    SimpleProgramRunner runner;

    if (listDialects) {
        mlir::DialectRegistry registry;
        simt::dialect::registerSimtStepDialect(registry);
        registry.insert<mlir::BuiltinDialect, mlir::arith::ArithDialect,
                        mlir::func::FuncDialect, simt::dialect::SimtStepDialect>();
        mlir::MLIRContext context(registry);
        llvm::outs() << "Registered dialects:\n"
                     << "  builtin\n  arith\n  func\n  simt_step\n";
        return 0;
    }

    auto resultOrErr = run(path, runner, semaCtx, resultLane);
    if (!resultOrErr) {
        llvm::errs() << llvm::toString(resultOrErr.takeError()) << "\n";
        return 1;
    }
    if (dumpState)
        dumpInterpreterState(runner);
    if (!path.empty() && path != "-")
        return 0;
    // Default in-memory program: expect lane id for the chosen lane.
    return *resultOrErr == static_cast<int>(resultLane) ? 0 : 1;
}
