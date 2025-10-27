#include "simt-step/semantics/SimpleSemantics.h"

#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"
#include "simt-step/semantics/CPSInterpreter.h"

#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>

namespace simt::semantics {

namespace {

SemValue makeValueFromAttribute(mlir::Attribute attr) {
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
        auto type = intAttr.getType();
        if (type.isSignlessInteger(1))
            return SemValue::fromBool(intAttr.getInt() != 0);
        if (type.isSignlessInteger() && type.getIntOrFloatBitWidth() <= 32)
            return SemValue::fromInt32(static_cast<int32_t>(intAttr.getInt()));
        return SemValue::fromInt64(intAttr.getInt());
    }
    if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(attr))
        return SemValue::fromFloat(static_cast<float>(floatAttr.getValueAsDouble()));

    llvm::errs() << "simple semantics: unsupported constant attribute\n";
    return SemValue();
}

} // namespace

auto SimpleSemantics::evalOperation(mlir::Operation *op,
                                    SemanticsContext &context) -> StepType {
    if (auto constOp = llvm::dyn_cast<mlir::arith::ConstantOp>(op))
        return handleConstant(constOp);

    if (auto addOp = llvm::dyn_cast<mlir::arith::AddIOp>(op))
        return handleAddIOp(addOp);

    if (llvm::isa<simt::dialect::LaneIdOp>(op))
        return handleLaneId(context);

    if (llvm::isa<simt::dialect::YieldOp>(op))
        return StepType::halt();

    if (llvm::isa<mlir::func::ReturnOp>(op))
        return StepType::halt();

    llvm::errs() << "simple semantics: unsupported op '"
                 << op->getName().getStringRef() << "'\n";
    return StepType::halt();
}

auto SimpleSemantics::handleConstant(mlir::arith::ConstantOp op) -> StepType {
    SemValue value = makeValueFromAttribute(op.getValue());
    return StepType::produce(std::move(value));
}

auto SimpleSemantics::handleLaneId(SemanticsContext &context) -> StepType {
    auto value = SemValue::fromInt32(static_cast<int32_t>(context.laneId));
    return StepType::produce(std::move(value));
}

auto SimpleSemantics::handleAddIOp(mlir::arith::AddIOp op) -> StepType {
    auto rhsConst = op.getRhs().getDefiningOp<mlir::arith::ConstantOp>();
    auto lhsConst = op.getLhs().getDefiningOp<mlir::arith::ConstantOp>();
    if (!lhsConst || !rhsConst) {
        llvm::errs() << "simple semantics: addi operands must be constants for now\n";
        return StepType::halt();
    }

    SemValue lhs = makeValueFromAttribute(lhsConst.getValue());
    SemValue rhs = makeValueFromAttribute(rhsConst.getValue());
    auto result = lhs.add(rhs);
    return StepType::produce(std::move(result));
}

} // namespace simt::semantics
