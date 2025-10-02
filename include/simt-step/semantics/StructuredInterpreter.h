#pragma once

#include "simt-step/semantics/SemanticsContext.h"
#include "simt-step/semantics/StructuredExecutor.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>

#include <mlir/IR/BuiltinAttributes.h>

namespace mlir {
class Operation;
class Value;
} // namespace mlir

namespace simt::structured {
class BlockOp;
class BranchOp;
class CondBranchOp;
class MaskMergeOp;
class MaskPopOp;
class MaskPushOp;
} // namespace simt::structured

namespace simt::semantics {

/// Captures the active mask and optional reconvergence metadata pushed onto the
/// mask stack while interpreting structured SIMT control flow.
struct MaskFrame {
    std::uint64_t activeMask = 0;
    mlir::FlatSymbolRefAttr mergeTarget;
    mlir::FlatSymbolRefAttr continueTarget;
};

/// Execution state that accompanies interpretation of structured SIMT blocks.
class StructuredExecutionState {
public:
    explicit StructuredExecutionState(SemanticsContext context);

    std::uint64_t getActiveMask() const { return context_.activeMask; }
    void setActiveMask(std::uint64_t mask) { context_.activeMask = mask; }

    void pushMask(std::uint64_t mask,
                  mlir::FlatSymbolRefAttr mergeTarget,
                  mlir::FlatSymbolRefAttr continueTarget);
    llvm::Expected<std::uint64_t> popMask();

    /// Merge the incoming mask with the current active mask.
    std::uint64_t mergeMask(std::uint64_t incomingMask);

    const llvm::SmallVector<MaskFrame> &stack() const { return stack_; }

private:
    SemanticsContext context_;
    llvm::SmallVector<MaskFrame> stack_;
};

/// High level result when interpreting a conditional branch.
struct BranchDecision {
    structured::BlockOp trueTarget;
    structured::BlockOp falseTarget;
    std::uint64_t trueMask = 0;
    std::uint64_t falseMask = 0;
};

/// Helper that evaluates masks and dispatches structured control flow.
class StructuredInterpreter {
public:
    StructuredInterpreter(StructuredExecutor &executor,
                          StructuredExecutionState state);

    StructuredExecutionState &state() { return state_; }
    const StructuredExecutionState &state() const { return state_; }

    llvm::Expected<void> handleMaskPush(structured::MaskPushOp op);
    llvm::Expected<std::uint64_t> handleMaskPop(structured::MaskPopOp op);
    llvm::Expected<std::uint64_t> handleMaskMerge(structured::MaskMergeOp op);

    llvm::Expected<structured::BlockOp> handleBranch(structured::BranchOp op);
    llvm::Expected<BranchDecision> handleCondBranch(structured::CondBranchOp op);

    /// Lookup the cached mask value associated with the given SSA value.
    llvm::Expected<std::uint64_t> lookupMaskValue(mlir::Value value) const;
    void bindMaskValue(mlir::Value value, std::uint64_t mask);

private:
    llvm::Expected<std::uint64_t> evaluateMaskValue(mlir::Value value) const;

    StructuredExecutor *executor_;
    StructuredExecutionState state_;
    llvm::DenseMap<mlir::Value, std::uint64_t> maskValues_;
};

} // namespace simt::semantics
