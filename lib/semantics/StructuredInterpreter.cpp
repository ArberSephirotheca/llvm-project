#include "simt-step/semantics/StructuredInterpreter.h"

#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"
#include "simt-step/Dialect/SimtStructured/StructuredDialect.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include <optional>

#include <llvm/Support/Casting.h>

using namespace mlir;

namespace simt::semantics {

StructuredExecutionState::StructuredExecutionState(SemanticsContext context)
    : context_(context) {}

void StructuredExecutionState::pushMask(std::uint64_t mask,
                                        FlatSymbolRefAttr mergeTarget,
                                        FlatSymbolRefAttr continueTarget) {
    stack_.push_back(MaskFrame{mask, mergeTarget, continueTarget});
    context_.activeMask = mask;
}

llvm::Expected<std::uint64_t> StructuredExecutionState::popMask() {
    if (stack_.empty()) {
        return llvm::make_error<llvm::StringError>(
            "mask_pop encountered empty mask stack",
            llvm::inconvertibleErrorCode());
    }
    auto frame = stack_.pop_back_val();
    context_.activeMask = frame.activeMask;
    return frame.activeMask;
}

std::uint64_t StructuredExecutionState::mergeMask(std::uint64_t incomingMask) {
    context_.activeMask &= incomingMask;
    return context_.activeMask;
}

StructuredInterpreter::StructuredInterpreter(StructuredProgram &program,
                                             StructuredExecutionState state)
    : program_(&program), state_(state) {}

llvm::Error StructuredInterpreter::handleMaskPush(
    structured::MaskPushOp op) {
    auto maskOrErr = evaluateMaskValue(op.getMask());
    if (!maskOrErr)
        return maskOrErr.takeError();

    state_.pushMask(*maskOrErr, op.getMergeTargetAttr(), op.getContinueTargetAttr());
    if (program_)
        (void)program_->lookupBlock(op.getOperation()->getBlock());
    return llvm::Error::success();
}

llvm::Expected<std::uint64_t> StructuredInterpreter::handleMaskPop(
    structured::MaskPopOp op) {
    auto poppedOrErr = state_.popMask();
    if (!poppedOrErr)
        return poppedOrErr.takeError();
    bindMaskValue(op.getResult(), *poppedOrErr);
    return poppedOrErr;
}

llvm::Expected<std::uint64_t> StructuredInterpreter::handleMaskMerge(
    structured::MaskMergeOp op) {
    auto incomingOrErr = evaluateMaskValue(op.getIncoming());
    if (!incomingOrErr)
        return incomingOrErr.takeError();
    auto merged = state_.mergeMask(*incomingOrErr);
    bindMaskValue(op.getResult(), merged);
    bindMaskValue(op.getIncoming(), merged);
    return merged;
}

llvm::Expected<structured::BlockOp>
StructuredInterpreter::handleBranch(structured::BranchOp op) {
    auto maskOrErr = evaluateMaskValue(op.getMask());
    if (!maskOrErr)
        return maskOrErr.takeError();
    state_.setActiveMask(*maskOrErr);

    auto targetAttr = op.getTargetAttr();
    if (!targetAttr)
        return llvm::make_error<llvm::StringError>(
            "branch missing destination symbol",
            llvm::inconvertibleErrorCode());

    if (!program_)
        return llvm::make_error<llvm::StringError>(
            "structured program metadata absent during branch interpretation",
            llvm::inconvertibleErrorCode());

    if (auto *info = program_->lookupBlock(targetAttr.getValue())) {
        auto block = info->block;
        if (auto maskArg = block.getMaskArgument())
            bindMaskValue(maskArg, *maskOrErr);
        return block;
    }

    return llvm::make_error<llvm::StringError>(
        "branch references unknown structured block",
        llvm::inconvertibleErrorCode());
}

llvm::Expected<BranchDecision>
StructuredInterpreter::handleCondBranch(structured::CondBranchOp op) {
    auto trueMaskOrErr = evaluateMaskValue(op.getTrueMask());
    if (!trueMaskOrErr)
        return trueMaskOrErr.takeError();
    auto falseMaskOrErr = evaluateMaskValue(op.getFalseMask());
    if (!falseMaskOrErr)
        return falseMaskOrErr.takeError();

    BranchDecision decision;
    decision.trueMask = *trueMaskOrErr;
    decision.falseMask = *falseMaskOrErr;

    if (auto trueAttr = op.getTrueTargetAttr()) {
        if (program_)
            if (auto *info = program_->lookupBlock(trueAttr.getValue())) {
                decision.trueTarget = info->block;
                if (auto maskArg = decision.trueTarget.getMaskArgument())
                    bindMaskValue(maskArg, decision.trueMask);
            }
    }
    if (auto falseAttr = op.getFalseTargetAttr()) {
        if (program_)
            if (auto *info = program_->lookupBlock(falseAttr.getValue())) {
                decision.falseTarget = info->block;
                if (auto maskArg = decision.falseTarget.getMaskArgument())
                    bindMaskValue(maskArg, decision.falseMask);
            }
    }

    if (decision.trueMask)
        state_.setActiveMask(decision.trueMask);
    else if (decision.falseMask)
        state_.setActiveMask(decision.falseMask);

    return decision;
}

llvm::Expected<std::uint64_t>
StructuredInterpreter::lookupMaskValue(Value value) const {
    return evaluateMaskValue(value);
}

void StructuredInterpreter::bindMaskValue(Value value, std::uint64_t mask) {
    maskValues_[value] = mask;
}

llvm::Expected<std::uint64_t>
StructuredInterpreter::evaluateMaskValue(Value value) const {
    if (!value)
        return llvm::make_error<llvm::StringError>(
            "attempted to evaluate null value as mask",
            llvm::inconvertibleErrorCode());

    if (auto it = maskValues_.find(value); it != maskValues_.end())
        return it->second;

    if (auto constOp = value.getDefiningOp<arith::ConstantIntOp>())
        return static_cast<std::uint64_t>(constOp.value());

    if (auto active = value.getDefiningOp<simt::dialect::ActiveMaskOp>())
        return state_.getActiveMask();

    if (auto arg = mlir::dyn_cast<BlockArgument>(value)) {
        if (auto parent = llvm::dyn_cast<structured::BlockOp>(arg.getOwner()->getParentOp())) {
            // Carried block arguments are expected to be mapped ahead of time.
            if (auto itArg = maskValues_.find(arg); itArg != maskValues_.end())
                return itArg->second;
        }
    }

    return llvm::make_error<llvm::StringError>(
        "unable to resolve mask value during structured interpretation",
        llvm::inconvertibleErrorCode());
}

} // namespace simt::semantics
