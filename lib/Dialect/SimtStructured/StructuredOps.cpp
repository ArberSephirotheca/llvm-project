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

mlir::SuccessorOperands BranchOp::getSuccessorOperands(unsigned index) {
    assert(index == 0 && "branch has one successor");
    return mlir::SuccessorOperands(getDestOperandsMutable());
}

std::optional<mlir::BlockArgument> BranchOp::getSuccessorBlockArgument(unsigned operandIndex) {
    mlir::Block *destBlock = getDest();
    if (!destBlock || destBlock->getNumArguments() <= operandIndex)
        return std::nullopt;

    return destBlock->getArgument(operandIndex);
}

mlir::Block *BranchOp::getSuccessorForOperands(::llvm::ArrayRef<::mlir::Attribute>) {
    return getDest();
}

bool BranchOp::areTypesCompatible(::mlir::Type lhs, ::mlir::Type rhs) {
    return areMaskTypesCompatible(lhs, rhs);
}

mlir::SuccessorOperands CondBranchOp::getSuccessorOperands(unsigned index) {
    assert(index < 2 && "cond branch has two successors");
    if (index == 0)
        return mlir::SuccessorOperands(getTrueOperandsMutable());
    return mlir::SuccessorOperands(getFalseOperandsMutable());
}

std::optional<mlir::BlockArgument> CondBranchOp::getSuccessorBlockArgument(unsigned operandIndex) {
    if (operandIndex < getTrueOperands().size()) {
        mlir::Block *trueDest = getTrueDest();
        if (trueDest && trueDest->getNumArguments() > operandIndex)
            return trueDest->getArgument(operandIndex);
        return std::nullopt;
    }

    unsigned falseIndex = operandIndex - getTrueOperands().size();
    mlir::Block *falseDest = getFalseDest();
    if (falseDest && falseDest->getNumArguments() > falseIndex)
        return falseDest->getArgument(falseIndex);
    return std::nullopt;
}

mlir::Block *CondBranchOp::getSuccessorForOperands(::llvm::ArrayRef<::mlir::Attribute> operands) {
    if (operands.empty())
        return nullptr;
    mlir::Attribute condAttr = operands.front();
    if (auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(condAttr))
        return boolAttr.getValue() ? getTrueDest() : getFalseDest();
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(condAttr)) {
        auto value = intAttr.getValue();
        if (value == 1)
            return getTrueDest();
        if (value == 0)
            return getFalseDest();
    }
    return nullptr;
}

bool CondBranchOp::areTypesCompatible(::mlir::Type lhs, ::mlir::Type rhs) {
    return areMaskTypesCompatible(lhs, rhs);
}

mlir::LogicalResult CondBranchOp::verify() {
    if (getTrueMask().getType() != getFalseMask().getType())
        return emitOpError("true/false masks must have matching types");

    if (auto *trueDest = getTrueDest())
        if (trueDest->getNumArguments() != getTrueOperands().size())
            return emitOpError("number of true operands must match destination arguments");

    if (auto *falseDest = getFalseDest())
        if (falseDest->getNumArguments() != getFalseOperands().size())
            return emitOpError("number of false operands must match destination arguments");

    return mlir::success();
}

mlir::LogicalResult MaskMergeOp::verify() {
    if (getIncoming().getType() != getMerged().getType())
        return emitOpError("incoming and merged masks must share the same type");
    return mlir::success();
}

} // namespace simt::structured
