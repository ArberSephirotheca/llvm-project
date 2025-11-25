#pragma once

#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"
#include "simt-step/semantics/CPSInterpreter.h"
#include "simt-step/semantics/ExecutionState.h"
#include "simt-step/semantics/SimpleSemantics.h"

#include <mlir/IR/Block.h>

namespace simt::semantics {

/// Drives SimpleSemantics over a straight-line simt_step block.
class SimpleProgramRunner {
public:
    using ValueType = SemValue;
    using StepType = Step<ValueType>;
    using StateType = DefaultInterpreterState;

    SimpleProgramRunner() : semantics_(), interpreter_(semantics_) {}

    llvm::Error runBlock(mlir::Block *block,
                         SemanticsContext context = SemanticsContext{});

    const StateType &state() const { return interpreter_.state(); }

private:
    StepType buildStepForIterator(const DynamicBlockKey &key,
                                  mlir::Block *block,
                                  mlir::Block::iterator it,
                                  SemanticsContext context,
                                  LaneId lane);
    StepType evaluateAndChain(const DynamicBlockKey &key,
                              StepType step,
                              mlir::Block *block,
                              mlir::Block::iterator nextIt,
                              SemanticsContext context,
                              LaneId lane,
                              bool isTerminator,
                              bool continueAfterResult);

    llvm::Expected<bool> evaluateBool(mlir::Value value,
                                      SemanticsContext &context);
    llvm::Error handleIfOp(simt::dialect::IfOp ifOp,
                           SemanticsContext context);

    SimpleSemantics semantics_;
    CPSInterpreter<SimpleSemantics> interpreter_;
    bool enableLoopDispatch_ = false;
};

} // namespace simt::semantics
