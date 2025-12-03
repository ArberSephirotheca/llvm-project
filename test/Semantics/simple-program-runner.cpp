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
static std::string formatMaskBits(std::uint64_t mask, unsigned width);
static std::string describeBlockKind(
    const simt::semantics::DynamicBlock<simt::semantics::SemValue,
                                        simt::semantics::Step<simt::semantics::SemValue>> &blk);

llvm::Expected<int> run(llvm::StringRef path, SimpleProgramRunner &runner,
                        SemanticsContext &semaCtx) {
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
    bool expectResult = func.getFunctionType().getNumResults() > 0;

    if (llvm::Error err = runner.runBlock(&block, semaCtx))
        return std::move(err);

    if (expectResult) {
        // For now, just acknowledge execution completed; callers can inspect
        // dump output for per-lane returns if needed.
        return 0;
    }
    return 0;
}

static void dumpInterpreterState(const SimpleProgramRunner &runner) {
    const auto &state = runner.state();
    llvm::outs() << "readyQueue=" << state.readyQueue.size() << "\n";
    for (const auto &waveIt : state.waves) {
        llvm::outs() << "Wave " << waveIt.first << "\n";
        const auto &waveCtx = waveIt.second;
        unsigned maxLane = 0;
        if (!waveCtx.lanes.empty())
            maxLane = 1 + std::max_element(
                                 waveCtx.lanes.begin(), waveCtx.lanes.end(),
                                 [](const auto &a, const auto &b) {
                                     return a.first < b.first;
                                 })
                                 ->first;
        unsigned maskWidth = std::min<unsigned>(64, std::max<unsigned>(maxLane, 1));
        if (!waveCtx.mergeStack.empty()) {
            llvm::outs() << "  MergeStack:\n";
            for (const auto &entry : llvm::enumerate(waveCtx.mergeStack)) {
                llvm::outs() << "    [" << entry.index() << "] parent=" << entry.value().parent.block
                             << " expected=" << formatMaskBits(entry.value().expectedMask, maskWidth)
                             << " completed=" << formatMaskBits(entry.value().completedMask, maskWidth)
                             << " children=" << entry.value().pendingChildren.size() << "\n";
            }
        }
        for (const auto &blockIt : waveCtx.blocks) {
            const auto &key = blockIt.first;
            const auto &block = blockIt.second;
            std::string firstOp;
            if (key.block) {
                mlir::Block *mutableBlock =
                    const_cast<mlir::Block *>(key.block);
                if (!mutableBlock->empty())
                    firstOp = mutableBlock->front().getName().getStringRef().str();
            }
            llvm::outs() << "  Block " << key.block << " seq "
                         << key.sequenceId << " kind=" << describeBlockKind(block)
                         << " expected=" << formatMaskBits(block.expectedMask, maskWidth)
                         << " active=" << formatMaskBits(block.activeMask, maskWidth)
                         << " completed=" << formatMaskBits(block.completedMask, maskWidth)
                         << " pendingOps=" << block.pendingOps.size()
                         << " envs=" << block.valueEnvs.size();
            if (!firstOp.empty())
                llvm::outs() << " firstOp=" << firstOp;
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

static std::string formatMaskBits(std::uint64_t mask, unsigned width) {
    std::string s;
    s.reserve(width + 2);
    s.append("0b");
    for (int i = static_cast<int>(width) - 1; i >= 0; --i) {
        s.push_back((mask & (1ull << i)) ? '1' : '0');
    }
    return s;
}

static std::string describeBlockKind(
    const simt::semantics::DynamicBlock<simt::semantics::SemValue,
                                        simt::semantics::Step<simt::semantics::SemValue>> &blk) {
    if (blk.loopOp) {
        if (blk.isLoopPrepare)
            return "loop.prepare";
        if (blk.isLoopBody)
            return "loop.body";
        return "loop.unknown";
    }
    switch (blk.kind) {
    case simt::semantics::DynamicBlockKind::IfThen:
        return "if.then";
    case simt::semantics::DynamicBlockKind::IfElse:
        return "if.else";
    default:
        break;
    }
    return "plain";
}

} // namespace

int main(int argc, char **argv) {
    bool listDialects = false;
    bool dumpState = false;
    uint64_t maskOverride = 0;
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
        if (path.empty())
            path = argv[i];
    }

    SemanticsContext semaCtx;
    semaCtx.subgroupWidth = 32;
    semaCtx.activeMask =
        maskOverride ? maskOverride : ((1ull << 4) - 1ull);
    semaCtx.laneId = 0;

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

    auto resultOrErr = run(path, runner, semaCtx);
    if (!resultOrErr) {
        llvm::errs() << llvm::toString(resultOrErr.takeError()) << "\n";
        return 1;
    }
    if (dumpState)
        dumpInterpreterState(runner);
    return 0;
}
