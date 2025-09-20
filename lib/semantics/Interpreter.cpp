#include "simt-step/semantics/Interpreter.h"

#include "simt-step/plugins/Registry.h"

#include <llvm/Support/Error.h>
#include <llvm/Support/JSON.h>

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>

namespace simt::semantics {

Interpreter::Interpreter(const plugins::Registry &registry, std::string model)
    : registry_(&registry), model_(std::move(model)) {}

llvm::Expected<llvm::json::Value> Interpreter::runCustom(mlir::Operation &op,
                                                         SemanticsContext &context) const {
    auto instrAttr = op.getAttrOfType<mlir::StringAttr>("instr");
    if (!instrAttr) {
        return llvm::make_error<llvm::StringError>("custom op missing 'instr' attribute",
                                                   llvm::inconvertibleErrorCode());
    }

    auto paramsAttr = op.getAttrOfType<mlir::StringAttr>("params");
    if (!paramsAttr) {
        return llvm::make_error<llvm::StringError>("custom op missing 'params' attribute",
                                                   llvm::inconvertibleErrorCode());
    }

    auto jsonOrErr = llvm::json::parse(paramsAttr.getValue());
    if (!jsonOrErr) {
        return jsonOrErr.takeError();
    }

    auto paramsOrErr = plugins::parseParams(*jsonOrErr);
    if (!paramsOrErr) {
        return paramsOrErr.takeError();
    }

    auto maybeId = registry_->lookup(instrAttr.getValue());
    if (!maybeId) {
        return llvm::make_error<llvm::StringError>("unknown instruction: " + instrAttr.getValue().str(),
                                                   llvm::inconvertibleErrorCode());
    }

    const auto *handler = registry_->getHandler(model_, *maybeId);
    if (!handler) {
        return llvm::make_error<llvm::StringError>(
            "no handler registered for model '" + model_ + "'",
            llvm::inconvertibleErrorCode());
    }

    std::vector<llvm::json::Value> outputs;
    llvm::ArrayRef<llvm::json::Value> inputs;
    const bool ok = handler->interpret(*maybeId, *paramsOrErr, inputs, outputs, context);
    if (!ok) {
        return llvm::make_error<llvm::StringError>("handler execution failed",
                                                   llvm::inconvertibleErrorCode());
    }

    if (!outputs.empty()) {
        return outputs.front();
    }
    return llvm::json::Value(nullptr);
}

} // namespace simt::semantics
