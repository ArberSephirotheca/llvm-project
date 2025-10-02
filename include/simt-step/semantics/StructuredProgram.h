#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "simt-step/Dialect/SimtStructured/StructuredDialect.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Types.h>

namespace mlir {
class Block;
class BlockArgument;
} // namespace mlir

namespace simt::semantics {

struct BlockInfo {
    std::string symbol;
    simt::structured::BlockOp block;
    mlir::Block *body = nullptr;
    mlir::Type maskType;
    llvm::SmallVector<mlir::Type, 4> carriedTypes;
    mlir::FlatSymbolRefAttr mergeTarget;
    mlir::FlatSymbolRefAttr continueTarget;

    bool hasMergeTarget() const { return static_cast<bool>(mergeTarget); }
    bool hasContinueTarget() const { return static_cast<bool>(continueTarget); }
    bool hasBody() const { return body != nullptr; }
    unsigned getArgumentCount() const {
        return (maskType ? 1u : 0u) + static_cast<unsigned>(carriedTypes.size());
    }
    llvm::ArrayRef<mlir::Type> getCarriedTypes() const { return carriedTypes; }
    mlir::BlockArgument getMaskArgument() const;
};

/// Snapshot of structured SIMT blocks within a module. Used by the interpreter
/// to discover blocks, their carried arguments, and reconvergence metadata.
class StructuredProgram {
public:
    StructuredProgram() = default;

    /// Scan the module for `simt_struct.block` operations and cache their
    /// metadata. Any previously cached state is discarded.
    void initialize(mlir::ModuleOp module);

    mlir::ModuleOp getModule() const { return module_; }

    bool hasEntrySymbol() const { return entrySymbol_.has_value(); }
    llvm::StringRef getEntrySymbol() const {
        return entrySymbol_ ? llvm::StringRef(*entrySymbol_) : llvm::StringRef();
    }
    const BlockInfo *getEntryBlock() const;

    const BlockInfo *lookupBlock(llvm::StringRef symbol) const;
    BlockInfo *lookupBlock(llvm::StringRef symbol);

    const BlockInfo *lookupBlock(const mlir::Block *block) const;
    BlockInfo *lookupBlock(const mlir::Block *block);

    llvm::ArrayRef<BlockInfo> blocks() const { return blocks_; }

private:
    mlir::ModuleOp module_{nullptr};
    std::optional<std::string> entrySymbol_;
    std::vector<BlockInfo> blocks_;
    llvm::StringMap<std::size_t> indexBySymbol_;
    llvm::DenseMap<const mlir::Block *, std::size_t> indexByBodyBlock_;
};

/// Register utility passes associated with structured program analysis.
void registerDumpStructuredProgramPass();

} // namespace simt::semantics

