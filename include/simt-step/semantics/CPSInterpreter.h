#pragma once

#include "simt-step/semantics/Effects.h"
#include "simt-step/semantics/ExecutionState.h"
#include "simt-step/semantics/SemanticsContext.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <functional>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <queue>

#include <llvm/Support/Error.h>

namespace mlir {
class Operation;
} // namespace mlir

namespace simt::semantics {

/// Continuation-Passing Style control primitive returned by interpreter steps.
template <typename ValueT>
class Step {
public:
    struct Halt {};

    struct Continue {
        std::function<Step()> next;
    };

    struct Produce {
        ValueT value;
    };

    struct Suspend {
        Effect effect;
        std::function<Step()> resume;
    };

    using State = std::variant<Halt, Continue, Produce, Suspend>;

    Step() : state_(Halt{}) {}

    explicit Step(Halt halt) : state_(std::move(halt)) {}
    explicit Step(Continue cont) : state_(std::move(cont)) {}
    explicit Step(Produce prod) : state_(std::move(prod)) {}
    explicit Step(Suspend susp) : state_(std::move(susp)) {}

    static Step halt() { return Step(Halt{}); }

    static Step continueWith(std::function<Step()> next) {
        return Step(Continue{std::move(next)});
    }

    static Step produce(ValueT value) {
        return Step(Produce{std::move(value)});
    }

    static Step suspend(Effect effect, std::function<Step()> resume) {
        return Step(Suspend{std::move(effect), std::move(resume)});
    }

    bool isHalt() const { return std::holds_alternative<Halt>(state_); }
    bool isContinue() const { return std::holds_alternative<Continue>(state_); }
    bool isProduce() const { return std::holds_alternative<Produce>(state_); }
    bool isSuspend() const { return std::holds_alternative<Suspend>(state_); }

    const State &state() const { return state_; }
    State takeState() && { return std::move(state_); }

private:
    State state_;
};

/// Minimal tagless interface wrapper. Semantic implementations are expected to
/// provide the aliases below.
template <typename Impl>
struct SimtStepSemanticsAdaptor {
    using ValueType = typename Impl::ValueType;
    using StepType = Step<ValueType>;

    StepType eval(Impl &impl, mlir::Operation *op, SemanticsContext &context) {
        if constexpr (requires { impl.evalOperation(op, context); }) {
            return impl.evalOperation(op, context);
        } else {
            return impl.eval(op, context);
        }
    }
};

/// High-level interpreter shell that delegates to a semantics implementation
/// and exposes a CPS stepping API.
template <typename SemanticsT>
class SimtStepExecutor {
public:
    using ValueType = typename SemanticsT::ValueType;
    using StepType = Step<ValueType>;

    explicit SimtStepExecutor(SemanticsT semantics)
        : semantics_(std::move(semantics)) {}

    StepType step(mlir::Operation *op, SemanticsContext &context) {
        return adaptor_.eval(semantics_, op, context);
    }

    SemanticsT &semantics() { return semantics_; }
    const SemanticsT &semantics() const { return semantics_; }

private:
    SemanticsT semantics_;
    SimtStepSemanticsAdaptor<SemanticsT> adaptor_;
};

/// Template interpreter harness that drives CPS-style semantics.
template <typename SemanticsT>
class CPSInterpreter {
public:
    using ValueType = typename SemanticsT::ValueType;
    using StepType = Step<ValueType>;
    using StateType = InterpreterState<ValueType, StepType>;

    explicit CPSInterpreter(SemanticsT semantics)
        : semantics_(std::move(semantics)) {}

    StateType &state() { return state_; }
    const StateType &state() const { return state_; }

    /// Enqueue an initial continuation for the given wave/block/lane triple.
    void enqueue(WaveId wave, const DynamicBlockKey &block, LaneId lane,
                 StepType step) {
        ensureWaveBlock(wave, block, lane);
        state_.readyQueue.push(
            ReadyContinuation<ValueType, StepType>{wave, block, lane, std::move(step)});
    }

    /// Execute a single ready continuation if available.
    llvm::Error runOne() {
        if (state_.readyQueue.empty())
            return llvm::Error::success();
        auto item = std::move(state_.readyQueue.front());
        state_.readyQueue.pop();
        return processReady(std::move(item));
    }

    /// Run until there are no ready continuations left.
    llvm::Error run() {
        while (!state_.readyQueue.empty()) {
            if (llvm::Error err = runOne())
                return err;
        }
        return llvm::Error::success();
    }

    /// Build a continuation that executes the operation at `it` for the given
    /// wave/block/lane and chains to the next iterator.
    StepType makeNextOp(WaveId wave,
                        const DynamicBlockKey &key,
                        mlir::Block *block,
                        mlir::Block::iterator it,
                        SemanticsContext context,
                        LaneId lane) {
        if (it == block->end())
            return StepType::halt();

        context.laneId = lane;
        if (auto waveIt = state_.waves.find(wave); waveIt != state_.waves.end()) {
            if (auto *blk = getBlock(waveIt->second, key)) {
                context.activeMask = blk->activeMask;
                auto envIt = blk->valueEnvs.find(lane);
                if (envIt != blk->valueEnvs.end())
                    context.valueEnv = &envIt->second;
            }
        }
        // Handle structured loop splitting here.
        if (auto loopOp = llvm::dyn_cast<simt::dialect::LoopOp>(&*it)) {
            auto waveIt = state_.waves.find(wave);
            if (waveIt == state_.waves.end())
                return StepType::halt();
            auto &waveCtx = waveIt->second;
            auto parentBlockIt = waveCtx.blocks.find(key);
            if (parentBlockIt == waveCtx.blocks.end())
                return StepType::halt();
            auto &parentBlock = parentBlockIt->second;
            std::uint64_t laneBit = 1ull << lane;
            if ((parentBlock.activeMask & laneBit) == 0)
                return StepType::halt();

            std::uint64_t activeMask = parentBlock.activeMask;
            if (activeMask == 0)
                return StepType::halt();
            std::uint64_t parentExpected =
                parentBlock.expectedMask ? parentBlock.expectedMask : activeMask;

            mlir::Block *prepareBlock = &loopOp.getPrepareRegion().front();
            mlir::Block *bodyBlock = &loopOp.getBodyRegion().front();

            std::uint32_t baseSeq = key.sequenceId + 1;
            DynamicBlockKey prepKey{prepareBlock, baseSeq};
            DynamicBlockKey bodyKey{bodyBlock, baseSeq + 1};

            auto &prepareCtx = waveCtx.blocks[prepKey];
            prepareCtx.block = prepareBlock;
            prepareCtx.sequenceId = prepKey.sequenceId;
            prepareCtx.expectedMask = parentExpected;
            prepareCtx.activeMask = activeMask;
            prepareCtx.completedMask = 0;
            prepareCtx.loopOp = loopOp.getOperation();
            prepareCtx.isLoopPrepare = true;
            prepareCtx.isLoopBody = false;

            auto &bodyCtx = waveCtx.blocks[bodyKey];
            bodyCtx.block = bodyBlock;
            bodyCtx.sequenceId = bodyKey.sequenceId;
            bodyCtx.expectedMask = parentExpected;
            bodyCtx.activeMask = 0;
            bodyCtx.completedMask = 0;
            bodyCtx.loopOp = loopOp.getOperation();
            bodyCtx.isLoopPrepare = false;
            bodyCtx.isLoopBody = true;

            auto nextIt = std::next(it);
            StepType parentCont = StepType::continueWith(
                [this, wave, key, block, nextIt, context, lane]() mutable -> StepType {
                    return makeNextOp(wave, key, block, nextIt, context, lane);
                });
            parentBlock.continuations[lane] = parentCont;

            MergeStackEntry<ValueType, StepType> entry;
            entry.parent = key;
            entry.pendingChildren.push_back(prepKey);
            entry.pendingChildren.push_back(bodyKey);
            entry.childMasks.push_back(activeMask);
            entry.expectedMask = parentExpected;
            entry.completedMask = 0;
            entry.loopFrame.emplace();
            auto &loopFrame = *entry.loopFrame;
            loopFrame.loopOp = loopOp.getOperation();
            loopFrame.prepareKey = prepKey;
            loopFrame.bodyKey = bodyKey;
            loopFrame.nextSequenceId = bodyKey.sequenceId + 1;

            auto inits = loopOp.getInits();
            llvm::ArrayRef<mlir::BlockArgument> prepArgs = prepareBlock->getArguments();
            std::uint64_t mask = activeMask;
            while (mask) {
                unsigned l = std::countr_zero(mask);
                mask &= mask - 1;
                auto &tuple = loopFrame.carried[l];
                tuple.clear();
                tuple.reserve(inits.size());
                for (mlir::Value init : inits) {
                    auto valueOrErr =
                        evaluateValue(waveCtx, key, init, l, parentBlock.activeMask);
                    if (!valueOrErr) {
                        llvm::consumeError(valueOrErr.takeError());
                        tuple.push_back(ValueType{});
                    } else {
                        tuple.push_back(*valueOrErr);
                    }
                }

                auto &env = prepareCtx.valueEnvs[l];
                env.clear();
                for (auto indexed : llvm::enumerate(prepArgs)) {
                    if (indexed.index() < tuple.size())
                        env[indexed.value()] = tuple[indexed.index()];
                }

                SemanticsContext childContext = context;
                childContext.activeMask = activeMask;
                childContext.laneId = l;
                StepType childStep = makeNextOp(wave, prepKey, prepareBlock,
                                                prepareBlock->begin(), childContext, l);
                enqueue(wave, prepKey, l, std::move(childStep));
                parentBlock.expectedMask &= ~(1ull << l);
            }

            parentBlock.activeMask &= ~activeMask;
            waveCtx.mergeStack.push_back(std::move(entry));
            return StepType::halt();
        }

        auto findLoopEntry = [&](WaveContext<ValueType, StepType> &waveCtx,
                                const mlir::Operation *loopOp)
            -> MergeStackEntry<ValueType, StepType> * {
            for (auto it = waveCtx.mergeStack.rbegin();
                 it != waveCtx.mergeStack.rend(); ++it) {
                if (it->loopFrame && it->loopFrame->loopOp == loopOp)
                    return &*it;
            }
            return nullptr;
        };

        // Handle loop prepare terminator.
        if (auto condOp = llvm::dyn_cast<simt::dialect::ConditionOp>(&*it)) {
            auto waveIt = state_.waves.find(wave);
            if (waveIt == state_.waves.end())
                return StepType::halt();
            auto &waveCtx = waveIt->second;
            auto *blockCtx = getBlock(waveCtx, key);
            if (!blockCtx || !blockCtx->isLoopPrepare || !blockCtx->loopOp)
                goto if_dispatch;

            auto *entry = findLoopEntry(waveCtx, blockCtx->loopOp);
            if (!entry || !entry->loopFrame)
                return StepType::halt();
            auto &loopFrame = *entry->loopFrame;

            std::uint64_t laneBit = 1ull << lane;
            auto condOrErr = evaluateBool(waveCtx, key, condOp.getCondition(), lane,
                                          blockCtx->activeMask);
            bool takeBody = false;
            if (condOrErr)
                takeBody = *condOrErr;
            else
                llvm::consumeError(condOrErr.takeError());

            llvm::SmallVector<ValueType, 4> forwarded;
            forwarded.reserve(condOp.getForwarded().size());
            for (mlir::Value v : condOp.getForwarded()) {
                auto valOrErr =
                    evaluateValue(waveCtx, key, v, lane, blockCtx->activeMask);
                if (!valOrErr) {
                    llvm::consumeError(valOrErr.takeError());
                    forwarded.push_back(ValueType{});
                } else {
                    forwarded.push_back(*valOrErr);
                }
            }
            loopFrame.carried[lane].assign(forwarded.begin(), forwarded.end());

            blockCtx->activeMask &= ~laneBit;
            blockCtx->completedMask |= laneBit;

            if (takeBody) {
                DynamicBlockKey bodyKey{loopFrame.bodyKey.block,
                                        static_cast<std::uint32_t>(key.sequenceId + 1)};
                auto &bodyCtx = waveCtx.blocks[bodyKey];
                bodyCtx.block = bodyKey.block;
                bodyCtx.sequenceId = bodyKey.sequenceId;
                if (bodyCtx.expectedMask == 0)
                    bodyCtx.expectedMask =
                        blockCtx->expectedMask ? blockCtx->expectedMask
                                               : blockCtx->activeMask;
                bodyCtx.activeMask |= laneBit;
                bodyCtx.loopOp = blockCtx->loopOp;
                bodyCtx.isLoopBody = true;
                bodyCtx.isLoopPrepare = false;

                auto &env = bodyCtx.valueEnvs[lane];
                env.clear();
                auto bodyArgs =
                    const_cast<mlir::Block *>(bodyKey.block)->getArguments();
                for (auto indexed : llvm::enumerate(bodyArgs)) {
                    if (indexed.index() < forwarded.size())
                        env[indexed.value()] = forwarded[indexed.index()];
                }

                if (!llvm::is_contained(entry->pendingChildren, bodyKey)) {
                    entry->pendingChildren.push_back(bodyKey);
                    entry->childMasks.push_back(bodyCtx.activeMask);
                }

                SemanticsContext laneCtx = context;
                laneCtx.activeMask = bodyCtx.activeMask;
                laneCtx.laneId = lane;
                mlir::Block *childBlock = const_cast<mlir::Block *>(bodyKey.block);
                StepType childStep =
                    makeNextOp(wave, bodyKey, childBlock, childBlock->begin(),
                               laneCtx, lane);
                enqueue(wave, bodyKey, lane, std::move(childStep));
                return StepType::halt();
            }

            // Exit the loop and reconverge to the parent.
            auto parentIt = waveCtx.blocks.find(entry->parent);
            if (parentIt != waveCtx.blocks.end()) {
                auto &parentEnv = parentIt->second.valueEnvs[lane];
                unsigned idx = 0;
                auto *loopOp = const_cast<mlir::Operation *>(loopFrame.loopOp);
                for (mlir::Value res : loopOp->getResults()) {
                    if (idx < forwarded.size())
                        parentEnv[res] = forwarded[idx];
                    ++idx;
                }
                auto contIt = parentIt->second.continuations.find(lane);
                if (contIt != parentIt->second.continuations.end()) {
                    parentIt->second.activeMask |= laneBit;
                    state_.readyQueue.push(ReadyContinuation<ValueType, StepType>{
                        wave, entry->parent, lane, contIt->second});
                    parentIt->second.continuations.erase(contIt);
                }
            }
            handleReconvergence(wave, waveCtx, key, lane);
            return StepType::halt();
        }

        // Handle loop body terminators (yield/continue/break).
        if (auto yieldOp = llvm::dyn_cast<simt::dialect::YieldOp>(&*it)) {
            auto waveIt = state_.waves.find(wave);
            if (waveIt == state_.waves.end())
                return StepType::halt();
            auto &waveCtx = waveIt->second;
            auto *blockCtx = getBlock(waveCtx, key);
            if (!blockCtx || !blockCtx->isLoopBody || !blockCtx->loopOp)
                goto if_dispatch;

            auto *entry = findLoopEntry(waveCtx, blockCtx->loopOp);
            if (!entry || !entry->loopFrame)
                return StepType::halt();
            auto &loopFrame = *entry->loopFrame;
            std::uint64_t laneBit = 1ull << lane;

            llvm::SmallVector<ValueType, 4> nextCarried;
            nextCarried.reserve(yieldOp.getNumOperands());
            for (mlir::Value v : yieldOp.getOperands()) {
                auto valOrErr =
                    evaluateValue(waveCtx, key, v, lane, blockCtx->activeMask);
                if (!valOrErr) {
                    llvm::consumeError(valOrErr.takeError());
                    nextCarried.push_back(ValueType{});
                } else {
                    nextCarried.push_back(*valOrErr);
                }
            }
            loopFrame.carried[lane].assign(nextCarried.begin(), nextCarried.end());

            blockCtx->activeMask &= ~laneBit;
            blockCtx->completedMask |= laneBit;

            // Spawn next iteration prepare/body pair.
            DynamicBlockKey nextPrep{loopFrame.prepareKey.block,
                                     loopFrame.nextSequenceId};
            DynamicBlockKey nextBody{loopFrame.bodyKey.block,
                                     static_cast<std::uint32_t>(loopFrame.nextSequenceId + 1)};
            loopFrame.nextSequenceId += 2;

            auto &prepCtx = waveCtx.blocks[nextPrep];
            prepCtx.block = nextPrep.block;
            prepCtx.sequenceId = nextPrep.sequenceId;
            prepCtx.expectedMask =
                blockCtx->expectedMask ? blockCtx->expectedMask : blockCtx->activeMask;
            prepCtx.activeMask |= laneBit;
            prepCtx.completedMask = 0;
            prepCtx.loopOp = blockCtx->loopOp;
            prepCtx.isLoopPrepare = true;
            prepCtx.isLoopBody = false;

            auto &bodyCtx = waveCtx.blocks[nextBody];
            bodyCtx.block = nextBody.block;
            bodyCtx.sequenceId = nextBody.sequenceId;
            bodyCtx.expectedMask =
                blockCtx->expectedMask ? blockCtx->expectedMask : blockCtx->activeMask;
            bodyCtx.activeMask = 0;
            bodyCtx.completedMask = 0;
            bodyCtx.loopOp = blockCtx->loopOp;
            bodyCtx.isLoopPrepare = false;
            bodyCtx.isLoopBody = true;

            if (!llvm::is_contained(entry->pendingChildren, nextPrep)) {
                entry->pendingChildren.push_back(nextPrep);
                entry->childMasks.push_back(prepCtx.activeMask);
            }
            if (!llvm::is_contained(entry->pendingChildren, nextBody)) {
                entry->pendingChildren.push_back(nextBody);
                entry->childMasks.push_back(bodyCtx.activeMask);
            }

            auto prepArgs =
                const_cast<mlir::Block *>(nextPrep.block)->getArguments();
            auto &env = prepCtx.valueEnvs[lane];
            env.clear();
            for (auto indexed : llvm::enumerate(prepArgs)) {
                if (indexed.index() < nextCarried.size())
                    env[indexed.value()] = nextCarried[indexed.index()];
            }

            SemanticsContext laneCtx = context;
            laneCtx.activeMask = prepCtx.activeMask;
            laneCtx.laneId = lane;
            mlir::Block *prepBlock = const_cast<mlir::Block *>(nextPrep.block);
            StepType childStep =
                makeNextOp(wave, nextPrep, prepBlock, prepBlock->begin(), laneCtx, lane);
            enqueue(wave, nextPrep, lane, std::move(childStep));
            return StepType::halt();
        }

        if (auto breakOp = llvm::dyn_cast<simt::dialect::BreakOp>(&*it)) {
            auto waveIt = state_.waves.find(wave);
            if (waveIt == state_.waves.end())
                return StepType::halt();
            auto &waveCtx = waveIt->second;
            auto *blockCtx = getBlock(waveCtx, key);
            if (!blockCtx || !blockCtx->isLoopBody || !blockCtx->loopOp)
                goto if_dispatch;

            auto *entry = findLoopEntry(waveCtx, blockCtx->loopOp);
            if (!entry || !entry->loopFrame)
                return StepType::halt();
            auto &loopFrame = *entry->loopFrame;
            std::uint64_t laneBit = 1ull << lane;

            llvm::SmallVector<ValueType, 4> results;
            results.reserve(breakOp.getNumOperands());
            for (mlir::Value v : breakOp.getOperands()) {
                auto valOrErr =
                    evaluateValue(waveCtx, key, v, lane, blockCtx->activeMask);
                if (!valOrErr) {
                    llvm::consumeError(valOrErr.takeError());
                    results.push_back(ValueType{});
                } else {
                    results.push_back(*valOrErr);
                }
            }
            loopFrame.carried[lane].assign(results.begin(), results.end());

            auto parentIt = waveCtx.blocks.find(entry->parent);
            if (parentIt != waveCtx.blocks.end()) {
                auto &parentEnv = parentIt->second.valueEnvs[lane];
                unsigned idx = 0;
                auto *loopOp = const_cast<mlir::Operation *>(loopFrame.loopOp);
                for (mlir::Value res : loopOp->getResults()) {
                    if (idx < results.size())
                        parentEnv[res] = results[idx];
                    ++idx;
                }
                auto contIt = parentIt->second.continuations.find(lane);
                if (contIt != parentIt->second.continuations.end()) {
                    parentIt->second.activeMask |= laneBit;
                    state_.readyQueue.push(ReadyContinuation<ValueType, StepType>{
                        wave, entry->parent, lane, contIt->second});
                    parentIt->second.continuations.erase(contIt);
                }
            }

            blockCtx->activeMask &= ~laneBit;
            blockCtx->completedMask |= laneBit;
            shrinkExpectedForLane(wave, waveCtx, lane);
            handleReconvergence(wave, waveCtx, key, lane);
            return StepType::halt();
        }

        if (auto contOp = llvm::dyn_cast<simt::dialect::ContinueOp>(&*it)) {
            auto waveIt = state_.waves.find(wave);
            if (waveIt == state_.waves.end())
                return StepType::halt();
            auto &waveCtx = waveIt->second;
            auto *blockCtx = getBlock(waveCtx, key);
            if (!blockCtx || !blockCtx->isLoopBody || !blockCtx->loopOp)
                goto if_dispatch;

            auto *entry = findLoopEntry(waveCtx, blockCtx->loopOp);
            if (!entry || !entry->loopFrame)
                return StepType::halt();
            auto &loopFrame = *entry->loopFrame;
            std::uint64_t laneBit = 1ull << lane;

            llvm::SmallVector<ValueType, 4> nextCarried;
            nextCarried.reserve(contOp.getNumOperands());
            for (mlir::Value v : contOp.getOperands()) {
                auto valOrErr =
                    evaluateValue(waveCtx, key, v, lane, blockCtx->activeMask);
                if (!valOrErr) {
                    llvm::consumeError(valOrErr.takeError());
                    nextCarried.push_back(ValueType{});
                } else {
                    nextCarried.push_back(*valOrErr);
                }
            }
            loopFrame.carried[lane].assign(nextCarried.begin(), nextCarried.end());

            blockCtx->activeMask &= ~laneBit;
            blockCtx->completedMask |= laneBit;

            DynamicBlockKey nextPrep{loopFrame.prepareKey.block,
                                     loopFrame.nextSequenceId};
            DynamicBlockKey nextBody{loopFrame.bodyKey.block,
                                     static_cast<std::uint32_t>(loopFrame.nextSequenceId + 1)};
            loopFrame.nextSequenceId += 2;

            auto &prepCtx = waveCtx.blocks[nextPrep];
            prepCtx.block = nextPrep.block;
            prepCtx.sequenceId = nextPrep.sequenceId;
            prepCtx.expectedMask =
                blockCtx->expectedMask ? blockCtx->expectedMask : blockCtx->activeMask;
            prepCtx.activeMask |= laneBit;
            prepCtx.completedMask = 0;
            prepCtx.loopOp = blockCtx->loopOp;
            prepCtx.isLoopPrepare = true;
            prepCtx.isLoopBody = false;

            auto &bodyCtx = waveCtx.blocks[nextBody];
            bodyCtx.block = nextBody.block;
            bodyCtx.sequenceId = nextBody.sequenceId;
            bodyCtx.expectedMask =
                blockCtx->expectedMask ? blockCtx->expectedMask : blockCtx->activeMask;
            bodyCtx.activeMask = 0;
            bodyCtx.completedMask = 0;
            bodyCtx.loopOp = blockCtx->loopOp;
            bodyCtx.isLoopPrepare = false;
            bodyCtx.isLoopBody = true;

            if (!llvm::is_contained(entry->pendingChildren, nextPrep)) {
                entry->pendingChildren.push_back(nextPrep);
                entry->childMasks.push_back(prepCtx.activeMask);
            }
            if (!llvm::is_contained(entry->pendingChildren, nextBody)) {
                entry->pendingChildren.push_back(nextBody);
                entry->childMasks.push_back(bodyCtx.activeMask);
            }

            auto prepArgs =
                const_cast<mlir::Block *>(nextPrep.block)->getArguments();
            auto &env = prepCtx.valueEnvs[lane];
            env.clear();
            for (auto indexed : llvm::enumerate(prepArgs)) {
                if (indexed.index() < nextCarried.size())
                    env[indexed.value()] = nextCarried[indexed.index()];
            }

            SemanticsContext laneCtx = context;
            laneCtx.activeMask = prepCtx.activeMask;
            laneCtx.laneId = lane;
            mlir::Block *prepBlock = const_cast<mlir::Block *>(nextPrep.block);
            StepType childStep =
                makeNextOp(wave, nextPrep, prepBlock, prepBlock->begin(), laneCtx, lane);
            enqueue(wave, nextPrep, lane, std::move(childStep));
            return StepType::halt();
        }

    if_dispatch:
        // Handle structured if here.
        if (auto ifOp = llvm::dyn_cast<simt::dialect::IfOp>(&*it)) {
            auto waveIt = state_.waves.find(wave);
            if (waveIt == state_.waves.end())
                return StepType::halt();
            auto &waveCtx = waveIt->second;
            auto parentBlockIt = waveCtx.blocks.find(key);
            if (parentBlockIt == waveCtx.blocks.end())
                return StepType::halt();
            auto &parentBlock = parentBlockIt->second;

            std::uint64_t trueMask = 0;
            std::uint64_t falseMask = 0;
            std::uint64_t activeMask = parentBlock.activeMask;
            while (activeMask) {
                unsigned laneId = std::countr_zero(activeMask);
                activeMask &= activeMask - 1;
                auto condOrErr = evaluateBool(waveCtx, key, ifOp.getCondition(),
                                              laneId, parentBlock.activeMask);
                if (!condOrErr) {
                    llvm::consumeError(condOrErr.takeError());
                    continue;
                }
                if (*condOrErr)
                    trueMask |= (1ull << laneId);
                else
                    falseMask |= (1ull << laneId);
            }

            if (ifOp.getElseRegion().empty())
                falseMask = 0;

            auto nextIt = std::next(it);
            StepType parentCont = StepType::continueWith(
                [this, wave, key, block, nextIt, context, lane]() mutable -> StepType {
                    return makeNextOp(wave, key, block, nextIt, context, lane);
                });
            parentBlock.continuations[lane] = parentCont;

            auto makeChildKey = [&](mlir::Block *b, std::uint32_t seq) {
                return DynamicBlockKey{b, seq};
            };
            std::uint32_t baseSeq = key.sequenceId + 1;

            std::optional<DynamicBlockKey> thenKey;
            std::optional<DynamicBlockKey> elseKey;

            std::uint64_t parentExpected =
                parentBlock.expectedMask ? parentBlock.expectedMask
                                         : parentBlock.activeMask;
            std::uint64_t childTrueExpected = parentExpected & trueMask;
            std::uint64_t childFalseExpected = parentExpected & falseMask;

            if (trueMask) {
                DynamicBlockKey k =
                    makeChildKey(&ifOp.getThenRegion().front(), baseSeq);
                thenKey = k;
                auto &childBlock = waveCtx.blocks[k];
                childBlock.block = k.block;
                childBlock.sequenceId = k.sequenceId;
                childBlock.expectedMask =
                    childTrueExpected ? childTrueExpected : trueMask;
                childBlock.activeMask = trueMask;
                childBlock.completedMask = 0;
            }

            if (falseMask && !ifOp.getElseRegion().empty()) {
                DynamicBlockKey k =
                    makeChildKey(&ifOp.getElseRegion().front(), baseSeq + 1);
                elseKey = k;
                auto &childBlock = waveCtx.blocks[k];
                childBlock.block = k.block;
                childBlock.sequenceId = k.sequenceId;
                childBlock.expectedMask =
                    childFalseExpected ? childFalseExpected : falseMask;
                childBlock.activeMask = falseMask;
                childBlock.completedMask = 0;
            }

            MergeStackEntry<ValueType, StepType> entry;
            entry.parent = key;
            if (thenKey) {
                entry.pendingChildren.push_back(*thenKey);
                entry.childMasks.push_back(trueMask);
            }
            if (elseKey) {
                entry.pendingChildren.push_back(*elseKey);
                entry.childMasks.push_back(falseMask);
            }
            entry.expectedMask = (childTrueExpected ? childTrueExpected : trueMask) |
                                 (childFalseExpected ? childFalseExpected : falseMask);
            entry.completedMask = 0;
            if (entry.expectedMask != 0)
                waveCtx.mergeStack.push_back(entry);

            parentBlock.activeMask &= ~(trueMask | falseMask);

            if (thenKey) {
                std::uint64_t mask = trueMask;
                while (mask) {
                    unsigned l = std::countr_zero(mask);
                    mask &= mask - 1;
                    SemanticsContext laneCtx = context;
                    laneCtx.activeMask = trueMask;
                    laneCtx.laneId = l;
                    auto *childBlock = const_cast<mlir::Block *>(thenKey->block);
                    StepType childStep = makeNextOp(wave, *thenKey, childBlock,
                                                    childBlock->begin(), laneCtx, l);
                    enqueue(wave, *thenKey, l, std::move(childStep));
                    if (elseKey) {
                        waveCtx.blocks[*elseKey].expectedMask &= ~(1ull << l);
                    }
                    parentBlock.expectedMask &= ~(1ull << l);
                }
            }
            if (elseKey) {
                std::uint64_t mask = falseMask;
                while (mask) {
                    unsigned l = std::countr_zero(mask);
                    mask &= mask - 1;
                    SemanticsContext laneCtx = context;
                    laneCtx.activeMask = falseMask;
                    laneCtx.laneId = l;
                    auto *childBlock = const_cast<mlir::Block *>(elseKey->block);
                    StepType childStep = makeNextOp(wave, *elseKey, childBlock,
                                                    childBlock->begin(), laneCtx, l);
                    enqueue(wave, *elseKey, l, std::move(childStep));
                    if (thenKey) {
                        waveCtx.blocks[*thenKey].expectedMask &= ~(1ull << l);
                    }
                    parentBlock.expectedMask &= ~(1ull << l);
                }
            }

            return StepType::halt();
        }

        StepType current = adaptor_.eval(semantics_, &*it, context);
        mlir::Block::iterator nextIt = std::next(it);
        bool isTerminator = it->hasTrait<mlir::OpTrait::IsTerminator>();
        const bool hasNext = nextIt != block->end();

        while (true) {
            typename StepType::State stateVariant = std::move(current).takeState();

            if (auto *cont =
                    std::get_if<typename StepType::Continue>(&stateVariant)) {
                if (!cont->next)
                    return StepType::halt();
                current = cont->next();
                continue;
            }

            if (auto *suspend =
                    std::get_if<typename StepType::Suspend>(&stateVariant)) {
                Effect effect = std::move(suspend->effect);
                auto resume = std::move(suspend->resume);
                auto chainedResume =
                    [this, wave, key, resume = std::move(resume), block, nextIt, context,
                     lane]() mutable -> StepType {
                    StepType resumed = resume();
                    return makeNextOp(wave, key, block, nextIt, context, lane);
                };
                return StepType::suspend(std::move(effect), std::move(chainedResume));
            }

            if (auto *prod =
                    std::get_if<typename StepType::Produce>(&stateVariant)) {
                if (!isTerminator && hasNext) {
                    return StepType::continueWith(
                        [this, wave, key, block, nextIt, context, lane]() mutable
                        -> StepType {
                            return makeNextOp(wave, key, block, nextIt, context, lane);
                        });
                }
                return StepType::produce(std::move(prod->value));
            }

            if (std::holds_alternative<typename StepType::Halt>(stateVariant)) {
                if (!isTerminator && hasNext) {
                    return StepType::continueWith(
                        [this, wave, key, block, nextIt, context, lane]() mutable
                        -> StepType {
                            return makeNextOp(wave, key, block, nextIt, context, lane);
                        });
                }
                return StepType::halt();
            }

            if (!isTerminator && hasNext) {
                return StepType::continueWith(
                    [this, wave, key, block, nextIt, context, lane]() mutable -> StepType {
                        return makeNextOp(wave, key, block, nextIt, context, lane);
                    });
            }

            return StepType::halt();
        }
    }

private:
    /// Evaluate an SSA value to a SemValue for a given lane in a block.
    llvm::Expected<ValueType> evaluateValue(WaveContext<ValueType, StepType> &waveCtx,
                                            const DynamicBlockKey &blockKey,
                                            mlir::Value value,
                                            LaneId lane,
                                            std::uint64_t activeMask) {
        SemanticsContext ctx;
        ctx.laneId = lane;
        ctx.activeMask = activeMask;
        if (auto *blockCtx = getBlock(waveCtx, blockKey)) {
            auto envIt = blockCtx->valueEnvs.find(lane);
            if (envIt != blockCtx->valueEnvs.end())
                ctx.valueEnv = &envIt->second;
        }
        // If the value has a defining op, ask the semantics to evaluate it.
        if (auto *defOp = value.getDefiningOp()) {
            StepType step = adaptor_.eval(semantics_, defOp, ctx);
            if (!step.isProduce())
                return llvm::make_error<llvm::StringError>(
                    "value evaluation did not produce",
                    llvm::inconvertibleErrorCode());
            auto state = std::move(step).takeState();
            return std::get<typename StepType::Produce>(std::move(state)).value;
        }
        // Block arguments should be present in the value environment.
        if (ctx.valueEnv) {
            auto it = ctx.valueEnv->find(value);
            if (it != ctx.valueEnv->end())
                return it->second;
        }
        return llvm::make_error<llvm::StringError>(
            "unsupported SSA value in interpreter evaluateValue",
            llvm::inconvertibleErrorCode());
    }

    /// Evaluate a boolean SSA value for a given lane.
    llvm::Expected<bool> evaluateBool(WaveContext<ValueType, StepType> &waveCtx,
                                      const DynamicBlockKey &blockKey,
                                      mlir::Value value,
                                      LaneId lane,
                                      std::uint64_t activeMask) {
        auto valOrErr = evaluateValue(waveCtx, blockKey, value, lane, activeMask);
        if (!valOrErr)
            return valOrErr.takeError();
        return valOrErr->asBool();
    }

    llvm::Error processReady(ReadyContinuation<ValueType, StepType> item) {
        ensureWaveBlock(item.wave, item.block, item.lane);
        auto &waveCtx = state_.waves[item.wave];
        auto &laneCtx = waveCtx.lanes[item.lane];
        laneCtx.currentBlock = item.block;
        if (auto *blockCtx = getBlock(waveCtx, item.block)) {
            waveCtx.currentMask = blockCtx->activeMask;
        }
        StepType current = std::move(item.resume);
        for (;;) {
            typename StepType::State stateVariant = std::move(current).takeState();

            if (std::holds_alternative<typename StepType::Continue>(stateVariant)) {
                auto cont =
                    std::get<typename StepType::Continue>(std::move(stateVariant));
                if (!cont.next) {
                    return llvm::make_error<llvm::StringError>(
                        "continuation missing resume function",
                        llvm::inconvertibleErrorCode());
                }
                current = cont.next();
                continue;
            }

            if (std::holds_alternative<typename StepType::Produce>(stateVariant)) {
                auto prod =
                    std::get<typename StepType::Produce>(std::move(stateVariant));
                laneCtx.hasReturned = true;
                laneCtx.returnValue = std::move(prod.value);
                if (auto *blockCtx = getBlock(waveCtx, item.block)) {
                    std::uint64_t laneBit = 1ull << item.lane;
                    blockCtx->activeMask &= ~laneBit;
                    blockCtx->completedMask |= laneBit;
                    shrinkExpectedForLane(item.wave, waveCtx, item.lane);
                    // Resume parent execution for this lane.
                    handleReconvergence(item.wave, waveCtx, item.block, item.lane);
                }
                return llvm::Error::success();
            }

            if (std::holds_alternative<typename StepType::Halt>(stateVariant)) {
                if (auto *blockCtx = getBlock(waveCtx, item.block)) {
                    if (blockCtx->loopOp) {
                        // Loop-internal scheduling halts shouldn't mark lane completion
                        // at the function level; the loop handlers drive reconvergence.
                        std::uint64_t laneBit = 1ull << item.lane;
                        blockCtx->activeMask &= ~laneBit;
                        blockCtx->completedMask |= laneBit;
                        return llvm::Error::success();
                    }
                    if (blockCtx->continuations.contains(item.lane)) {
                        // Control-split placeholder; the continuation will resume later.
                        return llvm::Error::success();
                    }
                    laneCtx.hasReturned = true;
                    std::uint64_t laneBit = 1ull << item.lane;
                    blockCtx->activeMask &= ~laneBit;
                    blockCtx->completedMask |= laneBit;
                    shrinkExpectedForLane(item.wave, waveCtx, item.lane);
                    // Do not re-add returned lanes to parent activeMask; just
                    // account for completion in merge tracking.
                    markMergeCompletion(item.wave, waveCtx, item.block, item.lane);
                }
                return llvm::Error::success();
            }

            if (std::holds_alternative<typename StepType::Suspend>(stateVariant)) {
                auto susp =
                    std::get<typename StepType::Suspend>(std::move(stateVariant));
                return handleSuspend(item.wave, item.block, item.lane,
                                     std::move(susp));
            }

            return llvm::make_error<llvm::StringError>(
                "unknown step state encountered in CPS interpreter",
                llvm::inconvertibleErrorCode());
        }
    }

    llvm::Error handleSuspend(WaveId wave, const DynamicBlockKey &block,
                              LaneId lane,
                              typename StepType::Suspend &&suspend) {
        if (suspend.effect.template isa<YieldEffect>()) {
            if (!suspend.resume) {
                return llvm::make_error<llvm::StringError>(
                    "yield effect missing resume continuation",
                    llvm::inconvertibleErrorCode());
            }
            StepType resumed = suspend.resume();
            enqueue(wave, block, lane, std::move(resumed));
            return llvm::Error::success();
        }

        if (auto *collective =
                suspend.effect.template get_if<CollectiveEffect>()) {
            auto &waveCtx = state_.waves[wave];
            auto *blockCtx = getBlock(waveCtx, block);
            if (!blockCtx) {
                return llvm::make_error<llvm::StringError>(
                    "collective effect missing dynamic block context",
                    llvm::inconvertibleErrorCode());
            }
            std::uint32_t key =
                collective->token.value_or(collective->operation);
            auto &syncPoint = waveCtx.collectives[key];
            syncPoint.effect = *collective;
            syncPoint.block = block;
            if (syncPoint.expectedMask == 0) {
                std::uint64_t fallbackMask =
                    blockCtx->expectedMask ? blockCtx->expectedMask : blockCtx->activeMask;
                syncPoint.expectedMask = collective->activeMask
                                             ? collective->activeMask
                                             : fallbackMask;
            }
            syncPoint.arrivals.insert(lane);
            syncPoint.continuations[lane] =
                StepType::continueWith(
                    [resume = std::move(suspend.resume)]() mutable -> StepType {
                        return resume();
                    });

            auto expectedCount =
                static_cast<unsigned>(std::popcount(syncPoint.expectedMask));
            if (syncPoint.arrivals.size() == expectedCount) {
                std::uint64_t mask = syncPoint.expectedMask;
                while (mask) {
                    unsigned l = std::countr_zero(mask);
                    mask &= mask - 1;
                    auto contIt = syncPoint.continuations.find(l);
                    if (contIt != syncPoint.continuations.end()) {
                        blockCtx->activeMask |= (1ull << l);
                        state_.readyQueue.push(
                            ReadyContinuation<ValueType, StepType>{wave, block, l,
                                                                   contIt->second});
                    }
                }
                waveCtx.collectives.erase(key);
            }
            return llvm::Error::success();
        }

        if (auto *sync =
                suspend.effect.template get_if<SynchronizationEffect>()) {
            auto &waveCtx = state_.waves[wave];
            auto *blockCtx = getBlock(waveCtx, block);
            if (!blockCtx) {
                return llvm::make_error<llvm::StringError>(
                    "synchronization effect missing dynamic block context",
                    llvm::inconvertibleErrorCode());
            }
            std::uint32_t key = sync->token.value_or(sync->operation);
            auto &syncPoint = waveCtx.syncPoints[key];
            syncPoint.effect = *sync;
            syncPoint.block = block;
            if (syncPoint.expectedMask == 0) {
                std::uint64_t fallbackMask =
                    blockCtx->expectedMask ? blockCtx->expectedMask : blockCtx->activeMask;
                syncPoint.expectedMask =
                    sync->activeMask ? sync->activeMask : fallbackMask;
            }
            syncPoint.arrivals.insert(lane);
            syncPoint.continuations[lane] =
                StepType::continueWith(
                    [resume = std::move(suspend.resume)]() mutable -> StepType {
                        return resume();
                    });

            auto expectedCount =
                static_cast<unsigned>(std::popcount(syncPoint.expectedMask));
            if (syncPoint.arrivals.size() == expectedCount) {
                std::uint64_t mask = syncPoint.expectedMask;
                while (mask) {
                    unsigned l = std::countr_zero(mask);
                    mask &= mask - 1;
                    auto contIt = syncPoint.continuations.find(l);
                    if (contIt != syncPoint.continuations.end()) {
                        blockCtx->activeMask |= (1ull << l);
                        state_.readyQueue.push(
                            ReadyContinuation<ValueType, StepType>{wave, block, l,
                                                                   contIt->second});
                    }
                }
                waveCtx.syncPoints.erase(key);
            }
            return llvm::Error::success();
        }

        return llvm::make_error<llvm::StringError>(
            "encountered suspend with unsupported effect",
            llvm::inconvertibleErrorCode());
    }

    void ensureWaveBlock(WaveId wave, const DynamicBlockKey &block, LaneId lane) {
        auto &waveCtx = state_.waves[wave];
        auto [blockIt, inserted] = waveCtx.blocks.try_emplace(block);
        if (inserted) {
            blockIt->second.activeMask = 0;
        }
        waveCtx.lanes.try_emplace(lane);
    }

    static DynamicBlock<ValueType, StepType> *
    getBlock(WaveContext<ValueType, StepType> &waveCtx,
             const DynamicBlockKey &key) {
        auto it = waveCtx.blocks.find(key);
        return it == waveCtx.blocks.end() ? nullptr : &it->second;
    }

    void shrinkExpectedForLane(WaveId waveId,
                               WaveContext<ValueType, StepType> &waveCtx,
                               LaneId lane) {
        // Clear from dynamic blocks.
        for (auto &entry : waveCtx.blocks) {
            entry.second.expectedMask &= ~(1ull << lane);
        }
        // Clear from collectives and release if now satisfied.
        for (auto it = waveCtx.collectives.begin();
             it != waveCtx.collectives.end();) {
            it->second.expectedMask &= ~(1ull << lane);
            it->second.arrivals.erase(lane);
            it->second.continuations.erase(lane);
            bool ready = it->second.expectedMask &&
                         it->second.arrivals.size() ==
                             static_cast<unsigned>(
                                 std::popcount(it->second.expectedMask));
            if (ready) {
                auto *blockCtx = getBlock(waveCtx, it->second.block);
                if (blockCtx) {
                    std::uint64_t mask = it->second.expectedMask;
                    while (mask) {
                        unsigned l = std::countr_zero(mask);
                        mask &= mask - 1;
                        auto contIt = it->second.continuations.find(l);
                        if (contIt != it->second.continuations.end()) {
                            blockCtx->activeMask |= (1ull << l);
                            state_.readyQueue.push(
                                ReadyContinuation<ValueType, StepType>{
                                    waveId, it->second.block, l, contIt->second});
                        }
                    }
                }
                auto cur = it;
                ++it;
                waveCtx.collectives.erase(cur);
                continue;
            }
            ++it;
        }
        // Clear from sync points and release if now satisfied.
        for (auto it = waveCtx.syncPoints.begin();
             it != waveCtx.syncPoints.end();) {
            it->second.expectedMask &= ~(1ull << lane);
            it->second.arrivals.erase(lane);
            it->second.continuations.erase(lane);
            bool ready = it->second.expectedMask &&
                         it->second.arrivals.size() ==
                             static_cast<unsigned>(
                                 std::popcount(it->second.expectedMask));
            if (ready) {
                auto *blockCtx = getBlock(waveCtx, it->second.block);
                if (blockCtx) {
                    std::uint64_t mask = it->second.expectedMask;
                    while (mask) {
                        unsigned l = std::countr_zero(mask);
                        mask &= mask - 1;
                        auto contIt = it->second.continuations.find(l);
                        if (contIt != it->second.continuations.end()) {
                            blockCtx->activeMask |= (1ull << l);
                            state_.readyQueue.push(
                                ReadyContinuation<ValueType, StepType>{
                                    waveId, it->second.block, l, contIt->second});
                        }
                    }
                }
                auto cur = it;
                ++it;
                waveCtx.syncPoints.erase(cur);
                continue;
            }
            ++it;
        }
    }

    void markMergeCompletion(WaveId,
                             WaveContext<ValueType, StepType> &waveCtx,
                             const DynamicBlockKey &childKey,
                             LaneId lane) {
        if (waveCtx.mergeStack.empty())
            return;
        for (auto it = waveCtx.mergeStack.rbegin();
             it != waveCtx.mergeStack.rend(); ++it) {
            bool matchesChild = llvm::any_of(it->pendingChildren,
                                             [&](const DynamicBlockKey &k) {
                                                 return k == childKey;
                                             });
            if (!matchesChild)
                continue;
            it->completedMask |= (1ull << lane);
            if (it->completedMask == it->expectedMask) {
                waveCtx.mergeStack.pop_back();
            }
            break;
        }
    }

    void handleReconvergence(WaveId waveId,
                             WaveContext<ValueType, StepType> &waveCtx,
                             const DynamicBlockKey &childKey,
                             LaneId lane) {
        if (waveCtx.mergeStack.empty())
            return;
        for (auto it = waveCtx.mergeStack.rbegin();
             it != waveCtx.mergeStack.rend(); ++it) {
            bool matchesChild = llvm::any_of(it->pendingChildren,
                                             [&](const DynamicBlockKey &k) {
                                                 return k == childKey;
                                             });
            if (!matchesChild)
                continue;

            it->completedMask |= (1ull << lane);

            DynamicBlockKey parentKey = it->parent;
            auto parentBlockIt = waveCtx.blocks.find(parentKey);
            if (parentBlockIt != waveCtx.blocks.end()) {
                auto &parentBlock = parentBlockIt->second;
                parentBlock.activeMask |= (1ull << lane);
                // Resume parent continuation for this lane only.
                auto contIt = parentBlock.continuations.find(lane);
                if (contIt != parentBlock.continuations.end()) {
                    state_.readyQueue.push(
                        ReadyContinuation<ValueType, StepType>{waveId, parentKey, lane,
                                                               contIt->second});
                    parentBlock.continuations.erase(contIt);
                }
            }
            // Pop the merge entry only when all expected lanes are done.
            if (it->completedMask == it->expectedMask) {
                waveCtx.mergeStack.pop_back();
            }
            break;
        }
    }

    SimtStepSemanticsAdaptor<SemanticsT> adaptor_;
    SemanticsT semantics_;
    StateType state_;
};

} // namespace simt::semantics
