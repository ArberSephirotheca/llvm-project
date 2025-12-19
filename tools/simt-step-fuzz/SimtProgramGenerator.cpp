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
    Value outWave;
};

static Value makeI32(OpBuilder &b, Location loc, int v) {
    return b.create<arith::ConstantIntOp>(loc, v, 32);
}

static Value makeBool(OpBuilder &b, Location loc, bool v) {
    return b.create<arith::ConstantIntOp>(loc, v ? 1 : 0, 1);
}

static func::FuncOp buildScalarHelper(OpBuilder &b, Location loc,
                                      llvm::StringRef name, RNG *rng) {
    auto i32 = b.getI32Type();
    auto funcType = b.getFunctionType({i32}, {i32});
    auto func = b.create<func::FuncOp>(loc, name, funcType);
    auto *entry = func.addEntryBlock();
    OpBuilder fb(entry, entry->begin());
    Value arg = entry->getArgument(0);
    int c1 = rng ? rng->pick(1, 4) : 1;
    int c2 = rng ? rng->pick(2, 5) : 3;
    int c3 = rng ? rng->pick(0, 4) : 2;
    Value v1 = fb.create<arith::AddIOp>(loc, arg, makeI32(fb, loc, c1));
    Value v2 = fb.create<arith::RemSIOp>(loc, v1, makeI32(fb, loc, c2));
    Value v3 = fb.create<arith::AddIOp>(loc, v2, makeI32(fb, loc, c3));
    fb.create<func::ReturnOp>(loc, ValueRange{v3});
    return func;
}

static Value makeNonUniformCond(OpBuilder &b, Location loc, RNG &rng,
                                const GeneratorConfig &cfg, Value tid) {
    int lanes = static_cast<int>(cfg.numThreads[0]);
    if (lanes < 2)
        return makeBool(b, loc, true);
    if (rng.coin()) {
        Value two = makeI32(b, loc, 2);
        Value rem = b.create<arith::RemSIOp>(loc, tid, two);
        Value zero = makeI32(b, loc, 0);
        return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, rem, zero);
    }
    int k = rng.pick(1, lanes - 1);
    Value ck = makeI32(b, loc, k);
    return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, tid, ck);
}

static Value makeNonUniformBound(OpBuilder &b, Location loc, RNG &rng,
                                 const GeneratorConfig &cfg, Value tid,
                                 int fallback) {
    int lanes = static_cast<int>(cfg.numThreads[0]);
    int maxTrip = static_cast<int>(cfg.maxTripCount);
    if (lanes < 2 || maxTrip < 2)
        return makeI32(b, loc, fallback);
    int k = rng.pick(2, maxTrip);
    Value ck = makeI32(b, loc, k);
    Value rem = b.create<arith::RemSIOp>(loc, tid, ck);
    Value one = makeI32(b, loc, 1);
    return b.create<arith::AddIOp>(loc, rem, one);
}

static void emitWaveCount(OpBuilder &b, Location loc, BuildState &st,
                          Value predicate, Value iteration = nullptr) {
    (void)predicate; // ignore caller-provided predicate; always count active lanes.
    int lanes = static_cast<int>(st.cfg.numThreads[0]);
    int stride = std::max<int>(1, st.cfg.maxTripCount * lanes);
    int waveBase = st.waveId * stride;
    Value idx = makeI32(b, loc, waveBase);
    if (iteration) {
        // iterScaled = iteration * lanes using adds (avoid mul).
        Value iterScaled = iteration;
        for (int i = 1; i < lanes; ++i)
            iterScaled = b.create<arith::AddIOp>(loc, iterScaled, iteration);
        idx = b.create<arith::AddIOp>(loc, idx, iterScaled);
    }
    idx = b.create<arith::AddIOp>(loc, idx, st.tid);
    Value count = b.create<simt::dialect::WaveCountBitsOp>(
        loc, b.getI32Type(), makeBool(b, loc, true));
    b.create<simt::dialect::BufferStoreOp>(loc, st.outWave, idx, count);
    st.waveId++;
}

static Value buildValue(OpBuilder &b, Location loc, BuildState &st) {
    int choice = st.rng.pick(0, 2); // 0 const, 1 tid, 2 tid + const
    if (choice == 0)
        return makeI32(b, loc, st.rng.pick(0, 4));
    if (choice == 1)
        return st.tid;
    if (choice == 2) {
        Value c = makeI32(b, loc, st.rng.pick(0, 4));
        return b.create<arith::AddIOp>(loc, st.tid, c);
    }
    // Fallback
    return st.tid;
}

static Value buildPattern(OpBuilder &b, Location loc, BuildState &st,
                          unsigned depth, unsigned maxDepth);

static Value buildSwitch(OpBuilder &b, Location loc, BuildState &st,
                         unsigned depth, unsigned maxDepth) {
    int numCases = st.rng.pick(2, 4);
    bool includeDefault = st.rng.coin();
    bool allowFallthrough = st.rng.coin();
    int defaultIndex = st.rng.pick(0, numCases - 1);
    llvm::SmallVector<bool, 4> emitWaveInCase;
    emitWaveInCase.reserve(numCases);
    bool anyWave = false;
    bool allWave = true;
    for (int i = 0; i < numCases; ++i) {
        bool pick = st.rng.coin();
        emitWaveInCase.push_back(pick);
        anyWave |= pick;
        allWave &= pick;
    }
    if (!anyWave)
        emitWaveInCase[st.rng.pick(0, numCases - 1)] = true;
    if (allWave)
        emitWaveInCase[st.rng.pick(0, numCases - 1)] = false;

    llvm::SmallVector<bool, 4> fallthroughCase;
    fallthroughCase.reserve(numCases);
    for (int i = 0; i < numCases; ++i) {
        bool fall = allowFallthrough && (i + 1) < numCases && st.rng.coin();
        fallthroughCase.push_back(fall);
    }

    int selectorMod = includeDefault ? numCases : (numCases - 1);
    Value selector = st.tid;
    if (selectorMod > 1) {
        Value mod = makeI32(b, loc, selectorMod);
        selector = b.create<arith::RemSIOp>(loc, selector, mod);
    }

    Value initVal = buildValue(b, loc, st);

    llvm::SmallVector<int64_t, 4> caseValues;
    caseValues.reserve(numCases - 1);
    int nextCaseValue = 0;
    for (int i = 0; i < numCases; ++i) {
        if (i == defaultIndex)
            continue;
        caseValues.push_back(nextCaseValue++);
    }

    auto switchOp = b.create<simt::dialect::SwitchOp>(
        loc, TypeRange{b.getI32Type()}, selector, ValueRange{initVal}, caseValues,
        defaultIndex);

    auto &region = switchOp.getCaseBody();
    while (static_cast<int>(region.getBlocks().size()) < numCases) {
        auto *blk = new Block();
        blk->addArguments({b.getI32Type()}, SmallVector<Location>{loc});
        region.push_back(blk);
    }

    int caseIdx = 0;
    for (auto &blk : region) {
        if (caseIdx >= numCases)
            break;
        if (blk.getNumArguments() == 0) {
            blk.addArguments({b.getI32Type()}, SmallVector<Location>{loc});
        }
        OpBuilder cb(&blk, blk.begin());
        Value bodyVal = buildPattern(cb, loc, st, depth + 1, maxDepth);
        if (emitWaveInCase[caseIdx])
            emitWaveCount(cb, loc, st, makeBool(cb, loc, true));
        auto yield = cb.create<simt::dialect::YieldOp>(loc, ValueRange{bodyVal});
        yield->setAttr("fallthrough", b.getBoolAttr(fallthroughCase[caseIdx]));
        ++caseIdx;
    }
    return switchOp.getResult(0);
}

static Value buildIf(OpBuilder &b, Location loc, BuildState &st, unsigned depth,
                     unsigned maxDepth) {
    Value cond = makeNonUniformCond(b, loc, st.rng, st.cfg, st.tid);
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
        // Non-uniform loop bounds derived from tid while keeping within maxTripCount.
        Value bound =
            makeNonUniformBound(pb, loc, st.rng, st.cfg, st.tid, trip);
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
        // Optionally emit a structured continue/break to exercise loop control.
        bool emitCtrl = st.rng.coin();
        bool doBreak = st.rng.coin();
        if (emitCtrl && doBreak) {
            // Break out with the current accum/next index.
            bb.create<simt::dialect::BreakOp>(loc, ValueRange{sum, nextIdx});
        } else if (emitCtrl) {
            // Continue with updated carried values to avoid stalling the loop.
            bb.create<simt::dialect::ContinueOp>(loc, ValueRange{sum, nextIdx});
        } else {
            bb.create<simt::dialect::YieldOp>(loc, ValueRange{sum, nextIdx});
        }
    }
    return loop.getResult(0);
}

static Value buildPattern(OpBuilder &b, Location loc, BuildState &st,
                          unsigned depth, unsigned maxDepth) {
    if (depth >= maxDepth)
        return buildValue(b, loc, st);
    int choice = st.rng.pick(0, 3); // 0 leaf, 1 if, 2 loop, 3 switch
    if (choice == 0)
        return buildValue(b, loc, st);
    if (choice == 1)
        return buildIf(b, loc, st, depth, maxDepth);
    if (choice == 2)
        return buildLoop(b, loc, st, depth, maxDepth);
    return buildSwitch(b, loc, st, depth, maxDepth);
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

    auto helper = buildScalarHelper(builder, loc, "helper0", nullptr);

    auto funcType = builder.getFunctionType({resTy}, {});
    auto func = builder.create<func::FuncOp>(loc, "main", funcType);
    llvm::errs() << "[fuzz-gen] func created\n";
    func->setAttr("simt.num_threads",
                  builder.getI64ArrayAttr(
                      {cfg.numThreads[0], cfg.numThreads[1], cfg.numThreads[2]}));

    auto *entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    Value outWave = entry->getArgument(0);
    Value tid =
        builder.create<simt::dialect::DispatchThreadIdOp>(loc, builder.getI32Type());
    llvm::errs() << "[fuzz-gen] tid op created\n";
    auto call = builder.create<func::CallOp>(loc, helper, ValueRange{tid});
    Value callRes = call.getResult(0);
    Value callBase = makeI32(builder, loc, 128);
    Value callIdx = builder.create<arith::AddIOp>(loc, callBase, tid);
    builder.create<simt::dialect::BufferStoreOp>(loc, outWave, callIdx, callRes);
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
            int k = static_cast<int>(cfg.maxTripCount);
            if (k < 2) {
                k = 1;
            }
            Value ck = prepB.create<arith::ConstantIntOp>(loc, k, 32);
            Value bound;
            if (k == 1) {
                bound = prepB.create<arith::ConstantIntOp>(loc, 1, 32);
            } else {
                Value rem = prepB.create<arith::RemSIOp>(loc, tid, ck);
                Value one = prepB.create<arith::ConstantIntOp>(loc, 1, 32);
                bound = prepB.create<arith::AddIOp>(loc, rem, one);
            }
            Value lt =
                prepB.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, i, bound);
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

    builder.setInsertionPointToEnd(entry);

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
    auto helper = buildScalarHelper(builder, loc, "helper0", &rng);
    auto funcType = builder.getFunctionType({resTy}, {});
    auto func = builder.create<func::FuncOp>(loc, "main", funcType);
    func->setAttr("simt.num_threads",
                  builder.getI64ArrayAttr(
                      {cfg.numThreads[0], cfg.numThreads[1], cfg.numThreads[2]}));

    auto *entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    Value outWave = entry->getArgument(0);
    Value tid =
        builder.create<simt::dialect::DispatchThreadIdOp>(loc, builder.getI32Type());
    auto call = builder.create<func::CallOp>(loc, helper, ValueRange{tid});
    Value callRes = call.getResult(0);
    Value callBase = makeI32(builder, loc, 128);
    Value callIdx = builder.create<arith::AddIOp>(loc, callBase, tid);
    builder.create<simt::dialect::BufferStoreOp>(loc, outWave, callIdx, callRes);

    // Build a non-uniform branch predicate.
    Value cond = makeNonUniformCond(builder, loc, rng, cfg, tid);

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
            Value bound =
                makeNonUniformBound(prepB, loc, rng, cfg, tid, tripCount);
            Value lt =
                prepB.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, i, bound);
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

    builder.setInsertionPointToEnd(entry);

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
    auto helper = buildScalarHelper(builder, loc, "helper0", &rng);
    auto funcType = builder.getFunctionType({resTy}, {});
    auto func = builder.create<func::FuncOp>(loc, "main", funcType);
    func->setAttr("simt.num_threads",
                  builder.getI64ArrayAttr(
                      {cfg.numThreads[0], cfg.numThreads[1], cfg.numThreads[2]}));

    auto *entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    Value outWave = entry->getArgument(0);
    Value tid =
        builder.create<simt::dialect::DispatchThreadIdOp>(loc, builder.getI32Type());
    auto call = builder.create<func::CallOp>(loc, helper, ValueRange{tid});
    Value callRes = call.getResult(0);
    Value callBase = makeI32(builder, loc, 128);
    Value callIdx = builder.create<arith::AddIOp>(loc, callBase, tid);
    builder.create<simt::dialect::BufferStoreOp>(loc, outWave, callIdx, callRes);

    BuildState st{cfg, rng, /*waveId=*/0, tid, outWave};

    int roots = rng.pick(1, 3);
    for (int r = 0; r < roots; ++r) {
        (void)buildPattern(builder, loc, st, /*depth=*/0, /*maxDepth=*/3);
    }

    builder.create<func::ReturnOp>(loc);
    return OwningOpRef<ModuleOp>(module);
}

} // namespace simt::fuzz
