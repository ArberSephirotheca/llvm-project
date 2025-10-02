#include "simt-step/semantics/StructuredExecutor.h"

#include "simt-step/Dialect/SimtStructured/StructuredDialect.h"

namespace simt::semantics {

void StructuredExecutor::initialize(mlir::ModuleOp module) {
    blockTable_.clear();
    module.walk([&](simt::structured::BlockOp block) {
        blockTable_.try_emplace(block.getSymName(), block.getOperation());
    });
}

simt::structured::BlockOp StructuredExecutor::lookupBlock(llvm::StringRef name) const {
    auto it = blockTable_.find(name);
    if (it == blockTable_.end())
        return nullptr;
    return llvm::dyn_cast_or_null<simt::structured::BlockOp>(it->second);
}

} // namespace simt::semantics
