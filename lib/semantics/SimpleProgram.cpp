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
    auto &laneCtx = waveCtx.lanes[0];

    for (mlir::Operation &op : *block) {
        if (auto ifOp = llvm::dyn_cast<simt::dialect::IfOp>(&op)) {
            if (llvm::Error err = handleIfOp(ifOp, context))
                return err;
            continue;
        }

        StepType step = semantics_.evalOperation(&op, context);
        StepType current = std::move(step);

        while (true) {
            auto variant = std::move(current).takeState();

            if (std::holds_alternative<typename StepType::Continue>(variant)) {
                auto cont = std::get<typename StepType::Continue>(std::move(variant));
                if (!cont.next) {
                    return llvm::make_error<llvm::StringError>(
                        "continuation missing resume function",
                        llvm::inconvertibleErrorCode());
                }
                current = cont.next();
                continue;
            }

            if (std::holds_alternative<typename StepType::Produce>(variant)) {
                auto prod = std::get<typename StepType::Produce>(std::move(variant));
                laneCtx.hasReturned = true;
                laneCtx.returnValue = prod.value;
                break;
            }

            if (std::holds_alternative<typename StepType::Suspend>(variant)) {
                auto susp = std::get<typename StepType::Suspend>(std::move(variant));
                state.pendingStep = StepType::suspend(std::move(susp.effect), std::move(susp.resume));
                break;
            }

            if (std::holds_alternative<typename StepType::Halt>(variant)) {
                break;
            }
        }
    }

    return llvm::Error::success();
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
