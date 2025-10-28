#include "simt-step/semantics/SimpleProgram.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Operation.h>

#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>

namespace simt::semantics {

llvm::Error SimpleProgramRunner::runBlock(mlir::Block *block,
                                          SemanticsContext context) {
    auto &state = interpreter_.state();
    auto &waveCtx = state.waves[0];

    DynamicBlockKey entryKey{block, 0};
    auto &entryBlock = waveCtx.blocks[entryKey];
    entryBlock.block = block;
    entryBlock.iteration = 0;
    entryBlock.expectedMask = context.activeMask ? context.activeMask : 0x1;
    entryBlock.activeMask = entryBlock.expectedMask;
    entryBlock.completedMask = 0;

    waveCtx.lanes[0];

    StepType step = semantics_.evalOperation(&block->front(), context);
    interpreter_.enqueue(/*wave=*/0, entryKey, /*lane=*/0, std::move(step));
    return interpreter_.run();
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
