#pragma once

#include "simt-step/semantics/SemanticsContext.h"

#include <llvm/Support/Error.h>

#include <string>

#include <llvm/Support/JSON.h>

namespace llvm {
namespace json {
class Value;
} // namespace json
} // namespace llvm

namespace mlir {
class Operation;
}

namespace simt::plugins {
class Registry;
}

namespace simt::semantics {

class Interpreter {
public:
    Interpreter(const plugins::Registry &registry, std::string model);

    llvm::Expected<llvm::json::Value> runCustom(mlir::Operation &op, SemanticsContext &context) const;

private:
    const plugins::Registry *registry_;
    std::string model_;
};

} // namespace simt::semantics
