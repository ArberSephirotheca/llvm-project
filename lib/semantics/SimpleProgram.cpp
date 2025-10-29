#include "simt-step/semantics/SimpleProgram.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <iterator>
#include <utility>

#include <mlir/IR/Block.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OpDefinition.h>

#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>

namespace simt::semantics {

llvm::Error SimpleProgramRunner::runBlock(mlir::Block *block,
                                          SemanticsContext context) {
    auto &state = interpreter_.state();
    state = StateType{};

    if (block->empty())
        return llvm::Error::success();

    constexpr WaveId wave = 0;
    constexpr LaneId lane = 0;

    auto &waveCtx = state.waves[wave];
    waveCtx.currentMask = 0;

    DynamicBlockKey entryKey{block, 0};
    auto &dynamicBlock = waveCtx.blocks[entryKey];
    dynamicBlock.block = block;
    dynamicBlock.iteration = 0;

    std::uint64_t laneMask = 1ull << lane;
    dynamicBlock.expectedMask = laneMask;
    dynamicBlock.activeMask = laneMask;
    dynamicBlock.completedMask = 0;

    waveCtx.currentMask = laneMask;

    auto &laneCtx = waveCtx.lanes[lane];
    laneCtx.values.clear();
    laneCtx.hasReturned = false;
    laneCtx.returnValue.reset();
    laneCtx.phase = decltype(laneCtx)::Phase::Running;
    laneCtx.currentBlock = entryKey;

    StepType initialStep = buildStepForIterator(block, block->begin(), context);
    interpreter_.enqueue(wave, entryKey, lane, std::move(initialStep));

    if (llvm::Error err = interpreter_.run())
        return err;

    return llvm::Error::success();
}

StepType SimpleProgramRunner::buildStepForIterator(mlir::Block *block,
                                                   mlir::Block::iterator it,
                                                   SemanticsContext context) {
    if (it == block->end())
        return StepType::halt();

    if (auto ifOp = llvm::dyn_cast<simt::dialect::IfOp>(&*it)) {
        SemanticsContext condContext = context;
        auto condOrErr = evaluateBool(ifOp.getCondition(), condContext);
        if (!condOrErr) {
            llvm::consumeError(condOrErr.takeError());
            return StepType::halt();
        }

        mlir::Block *selectedBlock = *condOrErr
                                         ? &ifOp.getThenRegion().front()
                                         : (ifOp.getElseRegion().empty()
                                                ? nullptr
                                                : &ifOp.getElseRegion().front());

        auto nextIt = std::next(it);
        bool continueAfterBranch =
            !selectedBlock || selectedBlock->empty();
        StepType branchStep =
            selectedBlock && !selectedBlock->empty()
                ? buildStepForIterator(selectedBlock, selectedBlock->begin(),
                                       context)
                : StepType::halt();

        return evaluateAndChain(std::move(branchStep), block, nextIt, context,
                                /*isTerminator=*/false,
                                /*continueAfterResult=*/continueAfterBranch);
    }

mlir::Operation *op = &*it;
auto nextIt = std::next(it);
bool isTerminator = op->hasTrait<mlir::OpTrait::IsTerminator>();

StepType step = semantics_.evalOperation(op, context);
return evaluateAndChain(std::move(step), block, nextIt, context,
                        isTerminator,
                        /*continueAfterResult=*/true);
}

StepType SimpleProgramRunner::evaluateAndChain(StepType step,
                                               mlir::Block *block,
                                               mlir::Block::iterator nextIt,
                                               SemanticsContext context,
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
                 isTerminator,
                 continueAfterResult]() mutable -> StepType {
                StepType resumed = resume();
                return evaluateAndChain(std::move(resumed), block, nextIt,
                                        context, isTerminator,
                                        continueAfterResult);
            };
            return StepType::suspend(std::move(effect), std::move(chainedResume));
        }

        const bool hasNext = nextIt != block->end();
        if (auto *prod =
                std::get_if<typename StepType::Produce>(&stateVariant)) {
            if (continueAfterResult && !isTerminator && hasNext) {
                return StepType::continueWith(
                    [this, block, nextIt, context]() mutable -> StepType {
                        return buildStepForIterator(block, nextIt, context);
                    });
            }
            return StepType::produce(std::move(prod->value));
        }

        if (std::holds_alternative<typename StepType::Halt>(stateVariant)) {
            if (continueAfterResult && !isTerminator && hasNext) {
                return StepType::continueWith(
                    [this, block, nextIt, context]() mutable -> StepType {
                        return buildStepForIterator(block, nextIt, context);
                    });
            }
            return StepType::halt();
        }

        if (continueAfterResult && !isTerminator && hasNext) {
            return StepType::continueWith(
                [this, block, nextIt, context]() mutable -> StepType {
                    return buildStepForIterator(block, nextIt, context);
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
