#include "simt-step/semantics/StructuredProgram.h"

#include "simt-step/Dialect/SimtStructured/StructuredDialect.h"

#include <mlir/IR/Block.h>

namespace simt::semantics {

namespace {

void populateBlockInfo(BlockInfo &info) {
    info.mergeTarget = info.block.getMergeTargetAttr();
    info.continueTarget = info.block.getContinueTargetAttr();

    auto &region = info.block.getBody();
    if (region.empty())
        return;
    auto &bodyBlock = region.front();
    info.argumentTypes.clear();
    info.argumentTypes.reserve(bodyBlock.getNumArguments());
    for (mlir::BlockArgument arg : bodyBlock.getArguments())
        info.argumentTypes.push_back(arg.getType());
}

} // namespace

void StructuredProgram::initialize(mlir::ModuleOp module) {
    module_ = module;
    entrySymbol_.reset();
    blocks_.clear();
    indexBySymbol_.clear();
    indexByBodyBlock_.clear();

    if (!module_)
        return;

    module_.walk([&](simt::structured::BlockOp block) {
        BlockInfo info;
        info.symbol = block.getSymName().str();
        info.block = block;
        populateBlockInfo(info);

        const std::size_t index = blocks_.size();
        blocks_.push_back(std::move(info));

        indexBySymbol_.try_emplace(blocks_.back().symbol, index);

        auto &region = block.getBody();
        if (!region.empty())
            indexByBodyBlock_[&region.front()] = index;

        if (!entrySymbol_) {
            entrySymbol_ = blocks_.back().symbol;
        } else if (*entrySymbol_ != "entry" && blocks_.back().symbol == "entry") {
            entrySymbol_ = blocks_.back().symbol;
        }
    });
}

const BlockInfo *StructuredProgram::getEntryBlock() const {
    if (!entrySymbol_)
        return nullptr;
    return lookupBlock(*entrySymbol_);
}

const BlockInfo *StructuredProgram::lookupBlock(llvm::StringRef symbol) const {
    auto it = indexBySymbol_.find(symbol);
    if (it == indexBySymbol_.end())
        return nullptr;
    return &blocks_[it->second];
}

BlockInfo *StructuredProgram::lookupBlock(llvm::StringRef symbol) {
    auto it = indexBySymbol_.find(symbol);
    if (it == indexBySymbol_.end())
        return nullptr;
    return &blocks_[it->second];
}

const BlockInfo *StructuredProgram::lookupBlock(const mlir::Block *block) const {
    auto it = indexByBodyBlock_.find(block);
    if (it == indexByBodyBlock_.end())
        return nullptr;
    return &blocks_[it->second];
}

BlockInfo *StructuredProgram::lookupBlock(const mlir::Block *block) {
    auto it = indexByBodyBlock_.find(block);
    if (it == indexByBodyBlock_.end())
        return nullptr;
    return &blocks_[it->second];
}

} // namespace simt::semantics

