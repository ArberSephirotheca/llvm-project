#include "simt-step/core/Dialect.h"

#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>

namespace simt::core {

mlir::StringAttr getScopeAttr(mlir::MLIRContext &context, Scope scope) {
    switch (scope) {
    case Scope::Thread:
        return mlir::StringAttr::get(&context, "Thread");
    case Scope::Subgroup:
        return mlir::StringAttr::get(&context, "Subgroup");
    case Scope::Workgroup:
        return mlir::StringAttr::get(&context, "Workgroup");
    }
    llvm_unreachable("unknown scope value");
}

mlir::StringAttr getMemorySemanticsAttr(mlir::MLIRContext &context, MemorySemantics semantics) {
    switch (semantics) {
    case MemorySemantics::None:
        return mlir::StringAttr::get(&context, "None");
    case MemorySemantics::Acquire:
        return mlir::StringAttr::get(&context, "Acquire");
    case MemorySemantics::Release:
        return mlir::StringAttr::get(&context, "Release");
    case MemorySemantics::AcquireRelease:
        return mlir::StringAttr::get(&context, "AcqRel");
    }
    llvm_unreachable("unknown memory semantics value");
}

mlir::Operation *buildWaveAllOp(mlir::OpBuilder &builder, mlir::Value predicate, mlir::Location loc) {
    auto i1 = builder.getI1Type();
    mlir::OperationState state(loc, "simt_step.wave_all");
    state.addOperands(predicate);
    state.addTypes(i1);
    return builder.create(state);
}

} // namespace simt::core
