#include "SimtProgramGenerator.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>

#include <cstdint>
#include <random>

#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"

using namespace mlir;

namespace simt::fuzz {

namespace {
struct RNG {
    std::mt19937_64 eng;
    explicit RNG(std::uint64_t seed) : eng(seed) {}
    int pick(int lo, int hi) {
        std::uniform_int_distribution<int> dist(lo, hi);
        return dist(eng);
    }
    bool coin() { return pick(0, 1) == 1; }
};
} // namespace

mlir::OwningOpRef<mlir::ModuleOp>
createDeterministicIfLoopModule(mlir::MLIRContext &context,
                                const GeneratorConfig &cfg) {
    ModuleOp module = ModuleOp::create(UnknownLoc::get(&context));
    llvm::errs() << "[fuzz-gen] module created\n";
    auto &modBlock = module.getBodyRegion().front();
    OpBuilder builder(&modBlock, modBlock.end());
    auto loc = module.getLoc();
    llvm::errs() << "[fuzz-gen] builder ready\n";

    auto resTy = simt::dialect::ResourceType::get(
        &context, simt::dialect::MemorySpace::Global, builder.getI32Type());
    llvm::errs() << "[fuzz-gen] resource type ready\n";

    auto funcType = builder.getFunctionType({resTy, resTy}, {});
    auto func = builder.create<func::FuncOp>(loc, "main", funcType);
    llvm::errs() << "[fuzz-gen] func created\n";
    func->setAttr("simt.num_threads",
                  builder.getI64ArrayAttr(
                      {cfg.numThreads[0], cfg.numThreads[1], cfg.numThreads[2]}));

    auto *entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    Value outMain = entry->getArgument(0);
    Value outWave = entry->getArgument(1);
    Value tid =
        builder.create<simt::dialect::DispatchThreadIdOp>(loc, builder.getI32Type());
    llvm::errs() << "[fuzz-gen] tid op created\n";
    Value c0 = builder.create<arith::ConstantIntOp>(loc, 0, 32);
    Value cond =
        builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tid, c0);
    llvm::errs() << "[fuzz-gen] cond created\n";

    auto ifOp = builder.create<simt::dialect::IfOp>(
        loc, TypeRange{builder.getI32Type()}, cond, /*withElseRegion=*/true);
    if (ifOp.getThenRegion().empty())
        ifOp.getThenRegion().push_back(new Block());
    if (ifOp.getElseRegion().empty())
        ifOp.getElseRegion().push_back(new Block());
    {
        // then block: small counted loop accumulating 0+1+2+3
        auto &thenBlock = ifOp.getThenRegion().front();
        OpBuilder thenB(&thenBlock, thenBlock.begin());
        Value zero = thenB.create<arith::ConstantIntOp>(loc, 0, 32);
        Value initI = thenB.create<arith::ConstantIntOp>(loc, 0, 32);
        auto loop = thenB.create<simt::dialect::LoopOp>(
            loc, TypeRange{builder.getI32Type(), builder.getI32Type()},
            ValueRange{zero, initI});
        if (loop.getPrepareRegion().empty()) {
            auto *prepBlock = new Block();
            prepBlock->addArguments({builder.getI32Type(), builder.getI32Type()},
                                    SmallVector<Location>{loc, loc});
            loop.getPrepareRegion().push_back(prepBlock);
        }
        if (loop.getBodyRegion().empty()) {
            auto *bodyBlock = new Block();
            bodyBlock->addArguments({builder.getI32Type(), builder.getI32Type()},
                                    SmallVector<Location>{loc, loc});
            loop.getBodyRegion().push_back(bodyBlock);
        }

        // prepare region
        {
            auto &prep = loop.getPrepareRegion().front();
            OpBuilder prepB(&prep, prep.begin());
            Value acc = prep.getArgument(0);
            Value i = prep.getArgument(1);
            Value c4 = prepB.create<arith::ConstantIntOp>(loc, 4, 32);
            Value lt =
                prepB.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, i, c4);
            prepB.create<simt::dialect::ConditionOp>(loc, lt, ValueRange{acc, i});
        }
        // body region
        {
            auto &body = loop.getBodyRegion().front();
            OpBuilder bodyB(&body, body.begin());
            Value acc = body.getArgument(0);
            Value i = body.getArgument(1);
            Value sum = bodyB.create<arith::AddIOp>(loc, acc, i);
            Value one = bodyB.create<arith::ConstantIntOp>(loc, 1, 32);
            Value next = bodyB.create<arith::AddIOp>(loc, i, one);
            bodyB.create<simt::dialect::YieldOp>(loc, ValueRange{sum, next});
        }
        Value loopVal = loop.getResult(0);
        thenB.create<simt::dialect::YieldOp>(loc, ValueRange{loopVal});
    }
    llvm::errs() << "[fuzz-gen] then/loop built\n";
    {
        // else block: just use tid
        auto &elseBlock = ifOp.getElseRegion().front();
        OpBuilder elseB(&elseBlock, elseBlock.begin());
        elseB.create<simt::dialect::YieldOp>(loc, ValueRange{tid});
    }
    llvm::errs() << "[fuzz-gen] else built\n";

    Value val = ifOp.getResult(0);
    Value out = outMain;
    builder.setInsertionPointToEnd(entry);
    builder.create<simt::dialect::BufferStoreOp>(loc, out, tid, val);

    // Wave-count branch: even lanes participate.
    Value two = builder.create<arith::ConstantIntOp>(loc, 2, 32);
    Value rem = builder.create<arith::RemSIOp>(loc, tid, two);
    Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
    Value even = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, rem, zero);
    // Store wave_count_bits result at idx = waveId * stride + tid.
    constexpr int waveId = 0;
    constexpr int stride = 64;
    Value waveIdC = builder.create<arith::ConstantIntOp>(loc, waveId * stride, 32);
    Value baseIdx = builder.create<arith::AddIOp>(loc, waveIdC, tid);
    Value count = builder.create<simt::dialect::WaveCountBitsOp>(loc, builder.getI32Type(), even);
    builder.create<simt::dialect::BufferStoreOp>(loc, outWave, baseIdx, count);

    builder.create<func::ReturnOp>(loc);
    llvm::errs() << "[fuzz-gen] return built\n";

    return OwningOpRef<ModuleOp>(module);
}

mlir::OwningOpRef<mlir::ModuleOp>
createRandomizedModule(mlir::MLIRContext &context,
                       const GeneratorConfig &cfg) {
    if (cfg.seed == 0)
        return createDeterministicIfLoopModule(context, cfg);

    RNG rng(cfg.seed);

    // Randomize a few knobs.
    int branchMode = rng.pick(0, 2); // 0: tid==0, 1: tid%2==0, 2: tid<k
    int tripCount = std::max(1, rng.pick(1, static_cast<int>(cfg.maxTripCount)));
    bool useWaveOp = rng.coin();
    int waveId = rng.pick(0, 3);
    int stride = 64;

    ModuleOp module = ModuleOp::create(UnknownLoc::get(&context));
    auto &modBlock = module.getBodyRegion().front();
    OpBuilder builder(&modBlock, modBlock.end());
    auto loc = module.getLoc();

    auto resTy = simt::dialect::ResourceType::get(
        &context, simt::dialect::MemorySpace::Global, builder.getI32Type());
    auto funcType = builder.getFunctionType({resTy, resTy}, {});
    auto func = builder.create<func::FuncOp>(loc, "main", funcType);
    func->setAttr("simt.num_threads",
                  builder.getI64ArrayAttr(
                      {cfg.numThreads[0], cfg.numThreads[1], cfg.numThreads[2]}));

    auto *entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    Value outMain = entry->getArgument(0);
    Value outWave = entry->getArgument(1);
    Value tid =
        builder.create<simt::dialect::DispatchThreadIdOp>(loc, builder.getI32Type());

    // Build branch predicate.
    Value cond;
    if (branchMode == 0) {
        Value c0 = builder.create<arith::ConstantIntOp>(loc, 0, 32);
        cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tid, c0);
    } else if (branchMode == 1) {
        Value two = builder.create<arith::ConstantIntOp>(loc, 2, 32);
        Value rem = builder.create<arith::RemSIOp>(loc, tid, two);
        Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
        cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, rem, zero);
    } else {
        int k = rng.pick(1, std::max<int>(1, cfg.numThreads[0]));
        Value ck = builder.create<arith::ConstantIntOp>(loc, k, 32);
        cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, tid, ck);
    }

    auto ifOp = builder.create<simt::dialect::IfOp>(
        loc, TypeRange{builder.getI32Type()}, cond, /*withElseRegion=*/true);
    if (ifOp.getThenRegion().empty())
        ifOp.getThenRegion().push_back(new Block());
    if (ifOp.getElseRegion().empty())
        ifOp.getElseRegion().push_back(new Block());

    // Then: counted loop with randomized trip count.
    {
        auto &thenBlock = ifOp.getThenRegion().front();
        OpBuilder thenB(&thenBlock, thenBlock.begin());
        Value zero = thenB.create<arith::ConstantIntOp>(loc, 0, 32);
        Value initI = thenB.create<arith::ConstantIntOp>(loc, 0, 32);
        auto loop = thenB.create<simt::dialect::LoopOp>(
            loc, TypeRange{builder.getI32Type(), builder.getI32Type()},
            ValueRange{zero, initI});
        if (loop.getPrepareRegion().empty()) {
            auto *prepBlock = new Block();
            prepBlock->addArguments({builder.getI32Type(), builder.getI32Type()},
                                    SmallVector<Location>{loc, loc});
            loop.getPrepareRegion().push_back(prepBlock);
        }
        if (loop.getBodyRegion().empty()) {
            auto *bodyBlock = new Block();
            bodyBlock->addArguments({builder.getI32Type(), builder.getI32Type()},
                                    SmallVector<Location>{loc, loc});
            loop.getBodyRegion().push_back(bodyBlock);
        }
        {
            auto &prep = loop.getPrepareRegion().front();
            OpBuilder prepB(&prep, prep.begin());
            Value acc = prep.getArgument(0);
            Value i = prep.getArgument(1);
            Value cTrip = prepB.create<arith::ConstantIntOp>(loc, tripCount, 32);
            Value lt =
                prepB.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, i, cTrip);
            prepB.create<simt::dialect::ConditionOp>(loc, lt, ValueRange{acc, i});
        }
        {
            auto &body = loop.getBodyRegion().front();
            OpBuilder bodyB(&body, body.begin());
            Value acc = body.getArgument(0);
            Value i = body.getArgument(1);
            Value sum = bodyB.create<arith::AddIOp>(loc, acc, i);
            Value one = bodyB.create<arith::ConstantIntOp>(loc, 1, 32);
            Value next = bodyB.create<arith::AddIOp>(loc, i, one);
            bodyB.create<simt::dialect::YieldOp>(loc, ValueRange{sum, next});
        }
        Value loopVal = loop.getResult(0);
        thenB.create<simt::dialect::YieldOp>(loc, ValueRange{loopVal});
    }
    {
        auto &elseBlock = ifOp.getElseRegion().front();
        OpBuilder elseB(&elseBlock, elseBlock.begin());
        elseB.create<simt::dialect::YieldOp>(loc, ValueRange{tid});
    }

    Value val = ifOp.getResult(0);
    builder.setInsertionPointToEnd(entry);
    builder.create<simt::dialect::BufferStoreOp>(loc, outMain, tid, val);

    if (useWaveOp) {
        Value two = builder.create<arith::ConstantIntOp>(loc, 2, 32);
        Value rem = builder.create<arith::RemSIOp>(loc, tid, two);
        Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
        Value even = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, rem, zero);
        Value waveBase = builder.create<arith::ConstantIntOp>(loc, waveId * stride, 32);
        Value baseIdx = builder.create<arith::AddIOp>(loc, waveBase, tid);
        Value count = builder.create<simt::dialect::WaveCountBitsOp>(loc, builder.getI32Type(), even);
        builder.create<simt::dialect::BufferStoreOp>(loc, outWave, baseIdx, count);
    } else {
        Value waveBase = builder.create<arith::ConstantIntOp>(loc, 0, 32);
        Value baseIdx = builder.create<arith::AddIOp>(loc, waveBase, tid);
        Value zeroVal = builder.create<arith::ConstantIntOp>(loc, 0, 32);
        builder.create<simt::dialect::BufferStoreOp>(loc, outWave, baseIdx, zeroVal);
    }

    builder.create<func::ReturnOp>(loc);
    return OwningOpRef<ModuleOp>(module);
}

mlir::OwningOpRef<mlir::ModuleOp>
createRicherRandomModule(mlir::MLIRContext &context,
                         const GeneratorConfig &cfg) {
    if (cfg.seed == 0)
        return createDeterministicIfLoopModule(context, cfg);

    RNG rng(cfg.seed);

    ModuleOp module = ModuleOp::create(UnknownLoc::get(&context));
    auto &modBlock = module.getBodyRegion().front();
    OpBuilder builder(&modBlock, modBlock.end());
    auto loc = module.getLoc();

    auto resTy = simt::dialect::ResourceType::get(
        &context, simt::dialect::MemorySpace::Global, builder.getI32Type());
    auto funcType = builder.getFunctionType({resTy, resTy}, {});
    auto func = builder.create<func::FuncOp>(loc, "main", funcType);
    func->setAttr("simt.num_threads",
                  builder.getI64ArrayAttr(
                      {cfg.numThreads[0], cfg.numThreads[1], cfg.numThreads[2]}));

    auto *entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    Value outMain = entry->getArgument(0);
    Value outWave = entry->getArgument(1);
    Value tid =
        builder.create<simt::dialect::DispatchThreadIdOp>(loc, builder.getI32Type());

    // Outer predicate: choose parity or range.
    Value outerCond;
    if (rng.coin()) {
        Value two = builder.create<arith::ConstantIntOp>(loc, 2, 32);
        Value rem = builder.create<arith::RemSIOp>(loc, tid, two);
        Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
        outerCond =
            builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, rem, zero);
    } else {
        int k = rng.pick(1, std::max<int>(1, static_cast<int>(cfg.numThreads[0])));
        Value ck = builder.create<arith::ConstantIntOp>(loc, k, 32);
        outerCond =
            builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, tid, ck);
    }

    auto ifOp = builder.create<simt::dialect::IfOp>(
        loc, TypeRange{builder.getI32Type()}, outerCond, /*withElseRegion=*/true);
    if (ifOp.getThenRegion().empty())
        ifOp.getThenRegion().push_back(new Block());
    if (ifOp.getElseRegion().empty())
        ifOp.getElseRegion().push_back(new Block());

    // Then branch: loop with random trip count and optional inner if.
    {
        auto &thenBlock = ifOp.getThenRegion().front();
        OpBuilder thenB(&thenBlock, thenBlock.begin());
        Value zero = thenB.create<arith::ConstantIntOp>(loc, 0, 32);
        Value initI = thenB.create<arith::ConstantIntOp>(loc, 0, 32);
        int trip = std::max(1, rng.pick(1, static_cast<int>(cfg.maxTripCount)));
        auto loop = thenB.create<simt::dialect::LoopOp>(
            loc, TypeRange{builder.getI32Type(), builder.getI32Type()},
            ValueRange{zero, initI});
        if (loop.getPrepareRegion().empty()) {
            auto *prepBlock = new Block();
            prepBlock->addArguments({builder.getI32Type(), builder.getI32Type()},
                                    SmallVector<Location>{loc, loc});
            loop.getPrepareRegion().push_back(prepBlock);
        }
        if (loop.getBodyRegion().empty()) {
            auto *bodyBlock = new Block();
            bodyBlock->addArguments({builder.getI32Type(), builder.getI32Type()},
                                    SmallVector<Location>{loc, loc});
            loop.getBodyRegion().push_back(bodyBlock);
        }
        {
            auto &prep = loop.getPrepareRegion().front();
            OpBuilder prepB(&prep, prep.begin());
            Value acc = prep.getArgument(0);
            Value i = prep.getArgument(1);
            Value cTrip = prepB.create<arith::ConstantIntOp>(loc, trip, 32);
            Value lt =
                prepB.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, i, cTrip);
            prepB.create<simt::dialect::ConditionOp>(loc, lt, ValueRange{acc, i});
        }
        {
            auto &body = loop.getBodyRegion().front();
            OpBuilder bodyB(&body, body.begin());
            Value acc = body.getArgument(0);
            Value i = body.getArgument(1);
            Value sum = bodyB.create<arith::AddIOp>(loc, acc, i);
            Value one = bodyB.create<arith::ConstantIntOp>(loc, 1, 32);
            Value next = bodyB.create<arith::AddIOp>(loc, i, one);

            if (rng.coin()) {
                int r = rng.pick(0, 2);
                Value three = bodyB.create<arith::ConstantIntOp>(loc, 3, 32);
                Value rem3 = bodyB.create<arith::RemSIOp>(loc, tid, three);
                Value cr = bodyB.create<arith::ConstantIntOp>(loc, r, 32);
                Value innerCond = bodyB.create<arith::CmpIOp>(
                    loc, arith::CmpIPredicate::eq, rem3, cr);
                auto innerIf = bodyB.create<simt::dialect::IfOp>(
                    loc, TypeRange{builder.getI32Type(), builder.getI32Type()},
                    innerCond, /*withElse=*/true);
                if (innerIf.getThenRegion().empty())
                    innerIf.getThenRegion().push_back(new Block());
                if (innerIf.getElseRegion().empty())
                    innerIf.getElseRegion().push_back(new Block());
                {
                    auto &innerThen = innerIf.getThenRegion().front();
                    OpBuilder ib(&innerThen, innerThen.begin());
                    ib.create<simt::dialect::YieldOp>(loc, ValueRange{sum, next});
                }
                {
                    auto &innerElse = innerIf.getElseRegion().front();
                    OpBuilder ib(&innerElse, innerElse.begin());
                    ib.create<simt::dialect::YieldOp>(loc, ValueRange{sum, next});
                }
                bodyB.create<simt::dialect::YieldOp>(loc, innerIf.getResults());
            } else {
                bodyB.create<simt::dialect::YieldOp>(loc, ValueRange{sum, next});
            }
        }
        Value loopVal = loop.getResult(0);
        thenB.create<simt::dialect::YieldOp>(loc, ValueRange{loopVal});
    }
    {
        auto &elseBlock = ifOp.getElseRegion().front();
        OpBuilder elseB(&elseBlock, elseBlock.begin());
        elseB.create<simt::dialect::YieldOp>(loc, ValueRange{tid});
    }

    Value val = ifOp.getResult(0);
    builder.setInsertionPointToEnd(entry);
    builder.create<simt::dialect::BufferStoreOp>(loc, outMain, tid, val);

    int stride = 64;
    auto emitWave = [&](int waveId, Value pred) {
        Value base = builder.create<arith::ConstantIntOp>(loc, waveId * stride, 32);
        Value idx = builder.create<arith::AddIOp>(loc, base, tid);
        Value count = builder.create<simt::dialect::WaveCountBitsOp>(loc, builder.getI32Type(), pred);
        builder.create<simt::dialect::BufferStoreOp>(loc, outWave, idx, count);
    };

    // Wave 0: even lanes.
    {
        Value two = builder.create<arith::ConstantIntOp>(loc, 2, 32);
        Value rem = builder.create<arith::RemSIOp>(loc, tid, two);
        Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
        Value even = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, rem, zero);
        emitWave(rng.pick(0, 3), even);
    }
    // Wave 1: tid < k'.
    {
        int k2 = rng.pick(1, std::max<int>(1, cfg.numThreads[0]));
        Value ck2 = builder.create<arith::ConstantIntOp>(loc, k2, 32);
        Value lt = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, tid, ck2);
        emitWave(rng.pick(4, 7), lt);
    }

    builder.create<func::ReturnOp>(loc);
    return OwningOpRef<ModuleOp>(module);
}

} // namespace simt::fuzz
