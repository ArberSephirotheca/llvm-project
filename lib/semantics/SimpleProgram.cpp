#include "simt-step/semantics/SimpleProgram.h"

#include <mlir/IR/Block.h>
#include <mlir/IR/Operation.h>

namespace simt::semantics {

llvm::Error SimpleProgramRunner::runBlock(mlir::Block *block,
                                          SemanticsContext context) {
    auto &state = interpreter_.state();
    auto &waveCtx = state.waves[0];
    auto &laneCtx = waveCtx.lanes[0];

    for (mlir::Operation &op : *block) {
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
                return llvm::make_error<llvm::StringError>(
                    "suspend effects are not supported in SimpleProgramRunner",
                    llvm::inconvertibleErrorCode());
            }

            if (std::holds_alternative<typename StepType::Halt>(variant)) {
                // No value produced; leave returnValue unchanged.
                break;
            }
        }
    }

    return llvm::Error::success();
}

} // namespace simt::semantics
