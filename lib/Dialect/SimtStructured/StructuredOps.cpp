#include "simt-step/Dialect/SimtStructured/StructuredDialect.h"

#include <mlir/IR/OpImplementation.h>
#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/Builders.h>

#define GET_OP_CLASSES
#include "StructuredOps.cpp.inc"

namespace simt::structured {

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
