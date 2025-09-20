#include "simt-step/plugins/examples/ReduceAdd.h"

#include "simt-step/semantics/SemanticsContext.h"

#include <llvm/Support/JSON.h>

namespace simt::plugins::examples {

namespace {

class ReduceAddHandler final : public InterpreterHandler {
public:
    bool interpret(InstructionId /*id*/,
                   const Params & /*params*/,
                   llvm::ArrayRef<llvm::json::Value> /*inputs*/,
                   std::vector<llvm::json::Value> &outputs,
                   semantics::SemanticsContext &context) const override {
        std::uint32_t sum = 0;
        for (std::uint32_t lane = 0; lane < context.subgroupWidth; ++lane) {
            if (context.activeMask & (1ULL << lane)) {
                sum += 1;
            }
        }
        outputs.emplace_back(static_cast<int64_t>(sum));
        return true;
    }
};

} // namespace

std::shared_ptr<const InterpreterHandler> makeReduceAddHandler() {
    return std::make_shared<ReduceAddHandler>();
}

void registerReduceAdd(Registry &registry, llvm::StringRef model) {
    InstructionSpec spec;
    spec.name = "reduce_add";
    spec.operands = {OperandDesc{"v", "i32"}};
    spec.results = {ResultDesc{"sum", "i32"}};
    spec.hasScope = true;
    spec.hasSync = true;
    spec.hasMemSem = false;
    spec.needsSubgroup = true;
    spec.needsSharedMemory = false;

    const InstructionId id = registry.registerInstruction(std::move(spec));
    registry.registerHandler(model, id, makeReduceAddHandler());
}

} // namespace simt::plugins::examples
