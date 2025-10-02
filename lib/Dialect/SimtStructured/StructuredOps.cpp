#include "simt-step/Dialect/SimtStructured/StructuredDialect.h"
#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"

#include <mlir/IR/OpImplementation.h>
#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>

#include <optional>

#define GET_OP_CLASSES
#include "StructuredOps.cpp.inc"

namespace simt::structured {

namespace {

std::optional<uint64_t> getMaskBitWidth(mlir::Type type) {
    if (auto maskTy = mlir::dyn_cast<simt::dialect::MaskType>(type))
        return maskTy.getWidth();
    if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(type)) {
        if (!intTy.isSignless())
            return std::nullopt;
        return intTy.getWidth();
    }
    return std::nullopt;
}

bool areMaskTypesCompatible(mlir::Type lhs, mlir::Type rhs) {
    auto lhsWidth = getMaskBitWidth(lhs);
    auto rhsWidth = getMaskBitWidth(rhs);
    if (!lhsWidth || !rhsWidth)
        return false;
    return lhsWidth == rhsWidth;
}

} // namespace

bool isMaskLikeType(mlir::Type type) {
    if (mlir::isa<simt::dialect::MaskType>(type))
        return true;
    if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(type))
        return intTy.isSignless();
    return false;
}

void BlockOp::getEntrySuccessorRegions(::llvm::ArrayRef<::mlir::Attribute>,
                                       ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions) {
    if (!getBody().empty())
        regions.emplace_back(&getBody());
}

void BlockOp::getSuccessorRegions(::mlir::RegionBranchPoint point,
                                  ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions) {
    if (point.isParent()) {
        if (!getBody().empty())
            regions.emplace_back(&getBody());
        return;
    }

    if (auto *region = point.getRegionOrNull()) {
        if (region == &getBody()) {
            regions.emplace_back(mlir::RegionSuccessor(getOperation()->getResults()));
            return;
        }
        return;
    }
}

bool BlockOp::areTypesCompatible(::mlir::Type lhs, ::mlir::Type rhs) {
    if (lhs == rhs)
        return true;
    return areMaskTypesCompatible(lhs, rhs);
}

void BlockOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                    ::mlir::StringAttr symName, ::mlir::Value mask,
                    ::mlir::FlatSymbolRefAttr mergeTarget,
                    ::mlir::FlatSymbolRefAttr continueTarget,
                    ::simt::structured::ReconvergencePolicyAttr policy) {
    state.addAttribute(BlockOp::getSymNameAttrName(state.name), symName);
    if (mergeTarget)
        state.addAttribute(BlockOp::getMergeTargetAttrName(state.name), mergeTarget);
    if (continueTarget)
        state.addAttribute(BlockOp::getContinueTargetAttrName(state.name), continueTarget);
    if (policy)
        state.addAttribute(BlockOp::getPolicyAttrName(state.name), policy);

    auto *region = state.addRegion();
    mlir::Block &block = region->emplaceBlock();
    mlir::Location loc = mask ? mask.getLoc() : builder.getUnknownLoc();
    mlir::Type maskType = mask ? mask.getType() : builder.getI64Type();
    block.addArgument(maskType, loc);
}

void BlockOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                    ::mlir::StringAttr symName, ::mlir::Value mask) {
    build(builder, state, symName, mask, nullptr, nullptr, nullptr);
}

::mlir::BlockArgument BlockOp::getMaskArgument() {
    if (getBody().empty())
        return {};
    auto &bodyBlock = getBody().front();
    if (bodyBlock.getNumArguments() == 0)
        return {};
    return bodyBlock.getArgument(0);
}

mlir::LogicalResult BlockOp::verify() {
    if (getBody().empty())
        return emitOpError("requires a body block");
    if (!getBody().hasOneBlock())
        return emitOpError("body must contain exactly one block");

    auto &bodyBlock = getBody().front();
    if (bodyBlock.getNumArguments() == 0)
        return emitOpError("body block must start with a mask argument");

    mlir::BlockArgument maskArg = bodyBlock.getArgument(0);
    if (!isMaskLikeType(maskArg.getType()))
        return emitOpError("mask argument must be mask-like type");

    return mlir::success();
}

mlir::LogicalResult CondBranchOp::verify() {
    if (getTrueMask().getType() != getFalseMask().getType())
        return emitOpError("true/false masks must have matching types");

    return mlir::success();
}

mlir::LogicalResult MaskMergeOp::verify() {
    if (getIncoming().getType() != getMerged().getType())
        return emitOpError("incoming and merged masks must share the same type");
    return mlir::success();
}

} // namespace simt::structured
