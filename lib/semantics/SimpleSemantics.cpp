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

    if (auto cmpOp = llvm::dyn_cast<mlir::arith::CmpIOp>(op))
        return handleCmpIOp(cmpOp, context);

    if (llvm::isa<simt::dialect::LaneIdOp>(op))
        return handleLaneId(context);

    if (llvm::isa<simt::dialect::DispatchThreadIdOp>(op))
        return handleDispatchThreadId(context);

    if (auto yieldOp = llvm::dyn_cast<simt::dialect::YieldOp>(op))
        return handleYieldOp(yieldOp, context);

    if (auto retOp = llvm::dyn_cast<mlir::func::ReturnOp>(op))
        return handleReturnOp(retOp);

    return handleUnknown(op);
}

auto SimpleSemantics::handleConstant(mlir::arith::ConstantOp) -> StepType {
    return StepType::halt();
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

llvm::Expected<SemValue>
SimpleSemantics::evaluateValue(mlir::Value value,
                               SemanticsContext &context) {
    if (auto constOp = value.getDefiningOp<mlir::arith::ConstantOp>())
        return makeValueFromAttribute(constOp.getValue());

    if (auto laneOp = value.getDefiningOp<simt::dialect::LaneIdOp>())
        return SemValue::fromInt32(static_cast<int32_t>(context.laneId));

    if (auto didOp = value.getDefiningOp<simt::dialect::DispatchThreadIdOp>()) {
        mlir::Type type = didOp.getType();
        if (mlir::isa<mlir::IndexType>(type) ||
            mlir::isa<mlir::IntegerType>(type)) {
            return SemValue::fromInt32(static_cast<int32_t>(context.laneId));
        }
        return llvm::make_error<llvm::StringError>(
            "dispatch_thread_id: unsupported result type",
            llvm::inconvertibleErrorCode());
    }

    if (auto cmpOp = value.getDefiningOp<mlir::arith::CmpIOp>()) {
        auto step = handleCmpIOp(cmpOp, context);
        if (!step.isProduce())
            return llvm::make_error<llvm::StringError>(
                "cmpi did not produce a value", llvm::inconvertibleErrorCode());
        auto state = std::move(step).takeState();
        return std::get<typename StepType::Produce>(std::move(state)).value;
    }

    return llvm::make_error<llvm::StringError>(
        "simple semantics: unsupported SSA value", llvm::inconvertibleErrorCode());
}

auto SimpleSemantics::handleCmpIOp(mlir::arith::CmpIOp op,
                                   SemanticsContext &context) -> StepType {
    auto lhsOrErr = evaluateValue(op.getLhs(), context);
    if (!lhsOrErr)
        return StepType::halt();
    auto rhsOrErr = evaluateValue(op.getRhs(), context);
    if (!rhsOrErr)
        return StepType::halt();

    bool result = false;
    switch (op.getPredicate()) {
    case mlir::arith::CmpIPredicate::eq:
        result = lhsOrErr->cmpEqual(*rhsOrErr).asBool();
        break;
    case mlir::arith::CmpIPredicate::ne:
        result = lhsOrErr->cmpNotEqual(*rhsOrErr).asBool();
        break;
    default:
        llvm::errs() << "simple semantics: unsupported cmp predicate\n";
        return StepType::halt();
    }

    return StepType::produce(SemValue::fromBool(result));
}

auto SimpleSemantics::handleDispatchThreadId(SemanticsContext &context)
    -> StepType {
    return StepType::produce(
        SemValue::fromInt32(static_cast<int32_t>(context.laneId)));
}

auto SimpleSemantics::handleYieldOp(simt::dialect::YieldOp op,
                                    SemanticsContext &context) -> StepType {
    if (op.getNumOperands() == 0)
        return StepType::halt();

    auto valueOrErr = evaluateValue(op.getOperand(0), context);
    if (!valueOrErr)
        return StepType::halt();

    return StepType::produce(std::move(*valueOrErr));
}

auto SimpleSemantics::handleReturnOp(mlir::func::ReturnOp) -> StepType {
    return StepType::halt();
}

auto SimpleSemantics::handleUnknown(mlir::Operation *op) -> StepType {
    llvm::errs() << "simple semantics: unsupported op '"
                 << op->getName().getStringRef() << "'\n";
    return StepType::halt();
}

} // namespace simt::semantics
