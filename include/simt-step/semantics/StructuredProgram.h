#pragma once

#include <llvm/ADT/DenseMap.h>
#include <mlir/IR/BuiltinOps.h>

namespace simt::structured {
class BlockOp;
}

namespace simt::semantics {

/// Light-weight helper that indexes structured SIMT blocks and prepares the
/// interpreter for execution. The full execution semantics will populate and
/// consume this map when the runtime wiring lands.
class StructuredExecutor {
public:
    StructuredExecutor() = default;

    /// Scan the given module for structured SIMT blocks and record them by
    /// symbol name. Existing entries are cleared.
    void initialize(mlir::ModuleOp module);

    /// Lookup a structured block by symbol name. Returns nullptr when the
    /// block has not been indexed yet.
    simt::structured::BlockOp lookupBlock(llvm::StringRef name) const;

private:
    llvm::DenseMap<llvm::StringRef, mlir::Operation *> blockTable_;
};

} // namespace simt::semantics
