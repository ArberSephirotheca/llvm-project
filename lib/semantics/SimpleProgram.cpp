#include "simt-step/semantics/SimpleProgram.h"

#include <mlir/IR/Block.h>
#include <mlir/IR/Operation.h>

namespace simt::semantics {

SimpleProgramRunner::SimpleProgramRunner()
    : semantics_(), interpreter_(semantics_) {}

auto SimpleProgramRunner::buildStep(mlir::Block *block,
                                    mlir::Block::iterator it,
                                    SemanticsContext context) -> StepType {
    if (it == block->end())
        return StepType::halt();

    mlir::Operation *op = &*it;
    StepType current = semantics_.evalOperation(op, context);

    return StepType::continueWith([=]() mutable {
        auto nextIt = std::next(it);
        return buildStep(block, nextIt, context);
    });
}

llvm::Error SimpleProgramRunner::runBlock(mlir::Block *block,
                                          SemanticsContext context) {
    auto initial = buildStep(block, block->begin(), context);
    interpreter_.enqueue(/*wave=*/0, DynamicBlockKey{block, 0},
                         /*lane=*/0, std::move(initial));
    return interpreter_.run();
}

} // namespace simt::semantics
