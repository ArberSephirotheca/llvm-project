#include "SimtProgramGenerator.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>

#include <algorithm>
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

struct BuildState {
    GeneratorConfig cfg;
    RNG &rng;
    int waveId = 0;
    Value tid;
    Value outMain;
    Value outWave;
};

static Value makeI32(OpBuilder &b, Location loc, int v) {
    return b.create<arith::ConstantIntOp>(loc, v, 32);
}

static Value makeBool(OpBuilder &b, Location loc, bool v) {
    return b.create<arith::ConstantIntOp>(loc, v ? 1 : 0, 1);
}

static void emitWaveCount(OpBuilder &b, Location loc, BuildState &st,
                          Value predicate, Value iteration = nullptr) {
    (void)predicate; // ignore caller-provided predicate; always count active lanes.
    int lanes = static_cast<int>(st.cfg.numThreads[0]);
    int stride = std::max<int>(1, st.cfg.maxTripCount * lanes);
    int waveBase = st.waveId * stride;
    Value idx = makeI32(b, loc, waveBase);
    if (iteration) {
        Value iterScaled = iteration; // avoid mul to stay within supported ops
        idx = b.create<arith::AddIOp>(loc, idx, iterScaled);
    }
    idx = b.create<arith::AddIOp>(loc, idx, st.tid);
    Value count = b.create<simt::dialect::WaveCountBitsOp>(
        loc, b.getI32Type(), makeBool(b, loc, true));
    b.create<simt::dialect::BufferStoreOp>(loc, st.outWave, idx, count);
    st.waveId++;
}

static Value buildValue(OpBuilder &b, Location loc, BuildState &st) {
    int choice = st.rng.pick(0, 3); // 0 const, 1 tid, 2 tid + const, 3 load
    if (choice == 0)
        return makeI32(b, loc, st.rng.pick(0, 4));
    if (choice == 1)
        return st.tid;
    if (choice == 2) {
        Value c = makeI32(b, loc, st.rng.pick(0, 4));
        return b.create<arith::AddIOp>(loc, st.tid, c);
    }
    // Load from outMain at tid (safe: default zero if unwritten).
    return b.create<simt::dialect::BufferLoadOp>(loc, st.outMain, st.tid);
}

static Value buildPattern(OpBuilder &b, Location loc, BuildState &st,
                          unsigned depth, unsigned maxDepth);

static Value buildIf(OpBuilder &b, Location loc, BuildState &st, unsigned depth,
                     unsigned maxDepth) {
    Value cond;
    if (st.rng.coin()) {
        Value two = makeI32(b, loc, 2);
        Value rem = b.create<arith::RemSIOp>(loc, st.tid, two);
        Value zero = makeI32(b, loc, 0);
        cond = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, rem, zero);
    } else {
        int k = std::max(1, st.rng.pick(1, static_cast<int>(st.cfg.numThreads[0])));
        Value ck = makeI32(b, loc, k);
        cond = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, st.tid, ck);
    }
    auto ifOp = b.create<simt::dialect::IfOp>(loc, TypeRange{b.getI32Type()},
                                              cond, /*withElseRegion=*/true);
    if (ifOp.getThenRegion().empty())
        ifOp.getThenRegion().push_back(new Block());
    if (ifOp.getElseRegion().empty())
        ifOp.getElseRegion().push_back(new Block());
    {
        auto &blk = ifOp.getThenRegion().front();
        OpBuilder tb(&blk, blk.begin());
        Value v = buildPattern(tb, loc, st, depth + 1, maxDepth);
        tb.create<simt::dialect::YieldOp>(loc, ValueRange{v});
    }
    {
        auto &blk = ifOp.getElseRegion().front();
        OpBuilder eb(&blk, blk.begin());
        Value v = buildPattern(eb, loc, st, depth + 1, maxDepth);
        eb.create<simt::dialect::YieldOp>(loc, ValueRange{v});
    }
    emitWaveCount(b, loc, st, cond);
    return ifOp.getResult(0);
}

static Value buildLoop(OpBuilder &b, Location loc, BuildState &st, unsigned depth,
                       unsigned maxDepth) {
    int trip = std::max(1, st.rng.pick(1, static_cast<int>(st.cfg.maxTripCount)));
    Value acc0 = makeI32(b, loc, 0);
    Value idx0 = makeI32(b, loc, 0);
    auto loop = b.create<simt::dialect::LoopOp>(
        loc, TypeRange{b.getI32Type(), b.getI32Type()}, ValueRange{acc0, idx0});
    if (loop.getPrepareRegion().empty()) {
        auto *prep = new Block();
        prep->addArguments({b.getI32Type(), b.getI32Type()},
                           SmallVector<Location>{loc, loc});
        loop.getPrepareRegion().push_back(prep);
    }
    if (loop.getBodyRegion().empty()) {
        auto *body = new Block();
        body->addArguments({b.getI32Type(), b.getI32Type()},
                           SmallVector<Location>{loc, loc});
        loop.getBodyRegion().push_back(body);
    }
    {
        auto &prep = loop.getPrepareRegion().front();
        OpBuilder pb(&prep, prep.begin());
        Value acc = prep.getArgument(0);
        Value idx = prep.getArgument(1);
        Value bound = makeI32(pb, loc, trip);
        Value lt = pb.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, idx, bound);
        pb.create<simt::dialect::ConditionOp>(loc, lt, ValueRange{acc, idx});
    }
    {
        auto &body = loop.getBodyRegion().front();
        OpBuilder bb(&body, body.begin());
        Value acc = body.getArgument(0);
        Value idx = body.getArgument(1);
        Value inner = (depth + 1 < maxDepth && st.rng.coin())
                          ? buildPattern(bb, loc, st, depth + 1, maxDepth)
                          : idx;
        Value sum = bb.create<arith::AddIOp>(loc, acc, inner);
        Value one = makeI32(bb, loc, 1);
        Value nextIdx = bb.create<arith::AddIOp>(loc, idx, one);
        emitWaveCount(bb, loc, st, makeBool(bb, loc, true), idx);
        bb.create<simt::dialect::YieldOp>(loc, ValueRange{sum, nextIdx});
    }
    return loop.getResult(0);
}

static Value buildPattern(OpBuilder &b, Location loc, BuildState &st,
                          unsigned depth, unsigned maxDepth) {
    if (depth >= maxDepth)
        return buildValue(b, loc, st);
    int choice = st.rng.pick(0, 2); // 0 leaf, 1 if, 2 loop
    if (choice == 0)
        return buildValue(b, loc, st);
    if (choice == 1)
        return buildIf(b, loc, st, depth, maxDepth);
    return buildLoop(b, loc, st, depth, maxDepth);
}
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

    // Wave-count branch: use constant true predicate.
    Value even = builder.create<arith::ConstantIntOp>(loc, 1, 1);
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
        Value trueVal = builder.create<arith::ConstantIntOp>(loc, 1, 1);
        Value waveBase = builder.create<arith::ConstantIntOp>(loc, waveId * stride, 32);
        Value baseIdx = builder.create<arith::AddIOp>(loc, waveBase, tid);
        Value count = builder.create<simt::dialect::WaveCountBitsOp>(loc, builder.getI32Type(), trueVal);
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
    RNG rng(cfg.seed == 0 ? 1 : cfg.seed);

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

    BuildState st{cfg, rng, /*waveId=*/0, tid, outMain, outWave};

    int roots = rng.pick(1, 3);
    for (int r = 0; r < roots; ++r) {
        Value val = buildPattern(builder, loc, st, /*depth=*/0, /*maxDepth=*/3);
        int lanes = static_cast<int>(cfg.numThreads[0]);
        Value offset = makeI32(builder, loc, r * lanes);
        Value idx = builder.create<arith::AddIOp>(loc, tid, offset);
        builder.create<simt::dialect::BufferStoreOp>(loc, outMain, idx, val);
    }

    builder.create<func::ReturnOp>(loc);
    return OwningOpRef<ModuleOp>(module);
}

} // namespace simt::fuzz
