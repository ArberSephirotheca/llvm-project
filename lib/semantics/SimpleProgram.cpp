#include "simt-step/semantics/SimpleProgram.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <bit>
#include <iterator>
#include <utility>

#include <mlir/IR/Block.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OpDefinition.h>

#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>

namespace simt::semantics {

namespace {
using StepType = SimpleProgramRunner::StepType;
using StateType = SimpleProgramRunner::StateType;
}

llvm::Error SimpleProgramRunner::runBlock(mlir::Block *block,
                                          SemanticsContext context) {
    auto &state = interpreter_.state();
    StateType newState;
    state = std::move(newState);

    if (block->empty())
        return llvm::Error::success();

    constexpr WaveId wave = 0;
    // Default to 32 active lanes if the caller does not supply a mask.
    // Honor requested mask; default to four lanes if unspecified for testing.
    std::uint64_t laneMask =
        context.activeMask ? context.activeMask : ((1ull << 4) - 1ull);

    auto &waveCtx = state.waves[wave];
    waveCtx.currentMask = 0;

    DynamicBlockKey entryKey{block, 0};
    auto &dynamicBlock = waveCtx.blocks[entryKey];
    dynamicBlock.block = block;
    dynamicBlock.iteration = 0;

    dynamicBlock.expectedMask = laneMask;
    dynamicBlock.activeMask = laneMask;
    dynamicBlock.completedMask = 0;

    waveCtx.currentMask = laneMask;

    std::uint64_t tmpMask = laneMask;
    while (tmpMask) {
        unsigned lane = std::countr_zero(tmpMask);
        tmpMask &= tmpMask - 1;
        auto &laneCtx = waveCtx.lanes[lane];
        laneCtx.values.clear();
        laneCtx.hasReturned = false;
        laneCtx.returnValue.reset();
        laneCtx.phase = decltype(laneCtx.phase)::Running;
        laneCtx.currentBlock = entryKey;

        SemanticsContext laneContext = context;
        laneContext.activeMask = laneMask;
        laneContext.laneId = lane;

        StepType initialStep =
            buildStepForIterator(block, block->begin(), laneContext, lane);
        interpreter_.enqueue(wave, entryKey, lane, std::move(initialStep));
    }

    if (llvm::Error err = interpreter_.run())
        return err;

    return llvm::Error::success();
}

SimpleProgramRunner::StepType
SimpleProgramRunner::buildStepForIterator(mlir::Block *block,
                                          mlir::Block::iterator it,
                                          SemanticsContext context,
                                          LaneId lane) {
    if (it == block->end())
        return StepType::halt();

    context.laneId = lane;
    // Refresh active mask from the interpreter's dynamic block state so ops
    // observe the per-block participation set.
    if (auto waveIt = interpreter_.state().waves.find(0);
        waveIt != interpreter_.state().waves.end()) {
        for (const auto &blockEntry : waveIt->second.blocks) {
            if (blockEntry.first.block == block) {
                context.activeMask = blockEntry.second.activeMask;
                break;
            }
        }
    }

    if (auto ifOp = llvm::dyn_cast<simt::dialect::IfOp>(&*it)) {
        auto &interpState = interpreter_.state();
        auto waveIt = interpState.waves.find(0);
        if (waveIt == interpState.waves.end())
            return StepType::halt();
        auto &waveCtx = waveIt->second;
        // Identify the current dynamic block key we are executing inside.
        DynamicBlockKey parentKey{block, 0};
        for (const auto &entry : waveCtx.blocks) {
            if (entry.first.block == block) {
                parentKey = entry.first;
                break;
            }
        }
        auto parentBlockIt = waveCtx.blocks.find(parentKey);
        if (parentBlockIt == waveCtx.blocks.end())
            return StepType::halt();
        auto &parentBlock = parentBlockIt->second;

        std::uint64_t trueMask = 0;
        std::uint64_t falseMask = 0;
        // Evaluate condition per active lane in this block.
        std::uint64_t activeMask = parentBlock.activeMask;
        while (activeMask) {
            unsigned laneId = std::countr_zero(activeMask);
            activeMask &= activeMask - 1;
            SemanticsContext condContext = context;
            condContext.laneId = laneId;
            condContext.activeMask = parentBlock.activeMask;
            auto condOrErr = evaluateBool(ifOp.getCondition(), condContext);
            if (!condOrErr) {
                llvm::consumeError(condOrErr.takeError());
                continue;
            }
            if (*condOrErr)
                trueMask |= (1ull << laneId);
            else
                falseMask |= (1ull << laneId);
        }

        // If there is no else region, lanes falling through rejoin the parent
        // immediately rather than spawning a child block.
        if (ifOp.getElseRegion().empty())
            falseMask = 0;

        auto nextIt = std::next(it);
        // Parent continuation to resume after reconvergence.
        StepType parentCont = StepType::continueWith(
            [this, block, nextIt, context, lane]() mutable -> StepType {
                return buildStepForIterator(block, nextIt, context, lane);
            });
        parentBlock.continuations[lane] = parentCont;

        // Create child dynamic blocks.
        auto makeChildKey = [&](mlir::Block *b, std::uint32_t iter) {
            return DynamicBlockKey{b, iter};
        };
        std::uint32_t baseIter = parentKey.iteration + 1;

        std::optional<DynamicBlockKey> thenKey;
        std::optional<DynamicBlockKey> elseKey;

        // Use parent.expectedMask to conservatively seed child expected masks.
        std::uint64_t parentExpected = parentBlock.expectedMask ? parentBlock.expectedMask
                                                                 : parentBlock.activeMask;
        std::uint64_t childTrueExpected = parentExpected & trueMask;
        std::uint64_t childFalseExpected = parentExpected & falseMask;

        if (trueMask) {
            DynamicBlockKey key =
                makeChildKey(&ifOp.getThenRegion().front(), baseIter);
            thenKey = key;
            auto &childBlock = waveCtx.blocks[key];
            childBlock.block = key.block;
            childBlock.iteration = key.iteration;
            childBlock.expectedMask = childTrueExpected ? childTrueExpected : trueMask;
            childBlock.activeMask = trueMask;
            childBlock.completedMask = 0;
        }

        if (falseMask && !ifOp.getElseRegion().empty()) {
            DynamicBlockKey key =
                makeChildKey(&ifOp.getElseRegion().front(), baseIter + 1);
            elseKey = key;
            auto &childBlock = waveCtx.blocks[key];
            childBlock.block = key.block;
            childBlock.iteration = key.iteration;
            childBlock.expectedMask =
                childFalseExpected ? childFalseExpected : falseMask;
            childBlock.activeMask = falseMask;
            childBlock.completedMask = 0;
        }

        // Push merge entry.
        MergeStackEntry<SemValue, StepType> entry;
        entry.parent = parentKey;
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
        waveCtx.mergeStack.push_back(entry);

        // Remove only dispatched lanes from the parent; others keep executing.
        parentBlock.activeMask &= ~(trueMask | falseMask);

        // Enqueue child continuations per lane.
        if (thenKey) {
            std::uint64_t mask = trueMask;
            while (mask) {
                unsigned l = std::countr_zero(mask);
                mask &= mask - 1;
                SemanticsContext laneCtx = context;
                laneCtx.activeMask = trueMask;
                laneCtx.laneId = l;
                auto *childBlock = const_cast<mlir::Block *>(thenKey->block);
                StepType childStep =
                    buildStepForIterator(childBlock, childBlock->begin(), laneCtx, l);
                interpreter_.enqueue(0, *thenKey, l, std::move(childStep));
                // This lane will not participate in other children; clear it from
                // their expected masks.
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
                StepType childStep =
                    buildStepForIterator(childBlock, childBlock->begin(), laneCtx, l);
                interpreter_.enqueue(0, *elseKey, l, std::move(childStep));
                // This lane will not participate in other children; clear it from
                // their expected masks.
                if (thenKey) {
                    waveCtx.blocks[*thenKey].expectedMask &= ~(1ull << l);
                }
                parentBlock.expectedMask &= ~(1ull << l);
            }
        }

        // Stop current lane; scheduler will pick up children.
        return StepType::halt();
    }

    mlir::Operation *op = &*it;
auto nextIt = std::next(it);
bool isTerminator = op->hasTrait<mlir::OpTrait::IsTerminator>();

    StepType step = semantics_.evalOperation(op, context);
    return evaluateAndChain(std::move(step), block, nextIt, context, lane,
                            isTerminator,
                            /*continueAfterResult=*/true);
}

SimpleProgramRunner::StepType
SimpleProgramRunner::evaluateAndChain(StepType step,
                                      mlir::Block *block,
                                      mlir::Block::iterator nextIt,
                                      SemanticsContext context,
                                      LaneId lane,
                                      bool isTerminator,
                                      bool continueAfterResult) {
    StepType current = std::move(step);

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
                [this, resume = std::move(resume), block, nextIt, context,
                 lane,
                 isTerminator,
                 continueAfterResult]() mutable -> StepType {
                StepType resumed = resume();
                return evaluateAndChain(std::move(resumed), block, nextIt,
                                        context, lane, isTerminator,
                                        continueAfterResult);
            };
            return StepType::suspend(std::move(effect), std::move(chainedResume));
        }

        const bool hasNext = nextIt != block->end();
        if (auto *prod =
                std::get_if<typename StepType::Produce>(&stateVariant)) {
            if (continueAfterResult && !isTerminator && hasNext) {
                return StepType::continueWith(
                    [this, block, nextIt, context, lane]() mutable -> StepType {
                        return buildStepForIterator(block, nextIt, context, lane);
                    });
            }
            return StepType::produce(std::move(prod->value));
        }

        if (std::holds_alternative<typename StepType::Halt>(stateVariant)) {
            if (continueAfterResult && !isTerminator && hasNext) {
                return StepType::continueWith(
                    [this, block, nextIt, context, lane]() mutable -> StepType {
                        return buildStepForIterator(block, nextIt, context, lane);
                    });
            }
            return StepType::halt();
        }

        if (continueAfterResult && !isTerminator && hasNext) {
            return StepType::continueWith(
                [this, block, nextIt, context, lane]() mutable -> StepType {
                    return buildStepForIterator(block, nextIt, context, lane);
                });
        }

        return StepType::halt();
    }
}

llvm::Expected<bool> SimpleProgramRunner::evaluateBool(mlir::Value value,
                                                       SemanticsContext &context) {
    auto semv = semantics_.evaluateValue(value, context);
    if (!semv)
        return semv.takeError();
    return semv->asBool();
}

llvm::Error SimpleProgramRunner::handleIfOp(simt::dialect::IfOp ifOp,
                                            SemanticsContext context) {
    auto condOrErr = evaluateBool(ifOp.getCondition(), context);
    if (!condOrErr)
        return condOrErr.takeError();

    bool cond = *condOrErr;
    if (cond) {
        return runBlock(&ifOp.getThenRegion().front(), context);
    }

    if (!ifOp.getElseRegion().empty())
        return runBlock(&ifOp.getElseRegion().front(), context);

    return llvm::Error::success();
}

} // namespace simt::semantics
