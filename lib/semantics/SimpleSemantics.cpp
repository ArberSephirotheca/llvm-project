#include "simt-step/semantics/SimpleSemantics.h"

#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"
#include "simt-step/semantics/CPSInterpreter.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/ErrorHandling.h>
#include <vector>
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
        return handleAddIOp(addOp, context);

    if (auto andOp = llvm::dyn_cast<mlir::arith::AndIOp>(op))
        return handleAndIOp(andOp, context);

    if (auto cmpOp = llvm::dyn_cast<mlir::arith::CmpIOp>(op))
        return handleCmpIOp(cmpOp, context);

    if (llvm::isa<simt::dialect::LaneIdOp>(op))
        return handleLaneId(context);

    if (llvm::isa<simt::dialect::DispatchThreadIdOp>(op))
        return handleDispatchThreadId(context);

    if (llvm::isa<simt::dialect::LoopOp>(op))
        return StepType::halt();

    if (auto yieldOp = llvm::dyn_cast<simt::dialect::YieldOp>(op))
        return handleYieldOp(yieldOp, context);

    if (auto retOp = llvm::dyn_cast<mlir::func::ReturnOp>(op))
        return handleReturnOp(retOp);

    return handleUnknown(op);
}

auto SimpleSemantics::handleConstant(mlir::arith::ConstantOp op) -> StepType {
    SemValue value = makeValueFromAttribute(op.getValue());
    return StepType::produce(std::move(value));
}

auto SimpleSemantics::handleLaneId(SemanticsContext &context) -> StepType {
    auto value = SemValue::fromInt32(static_cast<int32_t>(context.laneId));
    return StepType::produce(std::move(value));
}

auto SimpleSemantics::handleAddIOp(mlir::arith::AddIOp op,
                                   SemanticsContext &context) -> StepType {
    auto lhsOrErr = evaluateValue(op.getLhs(), context);
    if (!lhsOrErr) {
        llvm::consumeError(lhsOrErr.takeError());
        return StepType::halt();
    }
    auto rhsOrErr = evaluateValue(op.getRhs(), context);
    if (!rhsOrErr) {
        llvm::consumeError(rhsOrErr.takeError());
        return StepType::halt();
    }
    auto result = lhsOrErr->add(*rhsOrErr);
    return StepType::produce(std::move(result));
}

auto SimpleSemantics::handleAndIOp(mlir::arith::AndIOp op,
                                   SemanticsContext &context) -> StepType {
    auto lhsOrErr = evaluateValue(op.getLhs(), context);
    if (!lhsOrErr) {
        llvm::consumeError(lhsOrErr.takeError());
        return StepType::halt();
    }
    auto rhsOrErr = evaluateValue(op.getRhs(), context);
    if (!rhsOrErr) {
        llvm::consumeError(rhsOrErr.takeError());
        return StepType::halt();
    }
    auto result = lhsOrErr->bitAnd(*rhsOrErr);
    return StepType::produce(std::move(result));
}

llvm::Expected<SemValue>
SimpleSemantics::evaluateValue(mlir::Value value,
                               SemanticsContext &context) {
    if (EnableCPSDebugLogs) {
        llvm::errs() << "[Semantics] evaluateValue value=";
        if (auto *op = value.getDefiningOp()) {
            llvm::errs() << op->getName();
        } else {
            llvm::errs() << "<block-arg>";
        }
        llvm::errs() << " type=";
        value.getType().print(llvm::errs());
        llvm::errs() << " lane=" << context.laneId << "\n";
    }
    auto logVal = [&](const SemValue &v) {
        if (!EnableCPSDebugLogs)
            return;
        llvm::errs() << "  -> ";
        if (v.isBool())
            llvm::errs() << (v.asBool() ? "true" : "false");
        else if (v.isInteger())
            llvm::errs() << v.asInt64();
        else if (v.isFloat32())
            llvm::errs() << v.asFloat32();
        else
            llvm::errs() << "<none>";
        llvm::errs() << "\n";
    };
    if (context.valueEnv) {
        auto it = context.valueEnv->find(value);
        if (it != context.valueEnv->end()) {
            logVal(it->second);
            return it->second;
        }
    }

    if (auto constOp = value.getDefiningOp<mlir::arith::ConstantOp>()) {
        auto v = makeValueFromAttribute(constOp.getValue());
        logVal(v);
        return v;
    }

    if (auto laneOp = value.getDefiningOp<simt::dialect::LaneIdOp>()) {
        auto v = SemValue::fromInt32(static_cast<int32_t>(context.laneId));
        logVal(v);
        return v;
    }

    if (auto didOp = value.getDefiningOp<simt::dialect::DispatchThreadIdOp>()) {
        mlir::Type type = didOp.getType();
        if (mlir::isa<mlir::IndexType>(type) ||
            mlir::isa<mlir::IntegerType>(type)) {
            auto v = SemValue::fromInt32(static_cast<int32_t>(context.laneId));
            logVal(v);
            return v;
        }
        return llvm::make_error<llvm::StringError>(
            "dispatch_thread_id: unsupported result type",
            llvm::inconvertibleErrorCode());
    }

    if (auto loopOp = value.getDefiningOp<simt::dialect::LoopOp>()) {
        // If the interpreter already populated a valueEnv entry, prefer it.
        if (context.valueEnv) {
            auto it = context.valueEnv->find(value);
            if (it != context.valueEnv->end())
                return it->second;
        }
        auto allResultsOrErr = evaluateLoopOp(loopOp, context);
        if (!allResultsOrErr)
            return allResultsOrErr.takeError();
        unsigned idx = mlir::cast<mlir::OpResult>(value).getResultNumber();
        if (idx >= allResultsOrErr->size())
            return llvm::make_error<llvm::StringError>(
                "loop result index out of range",
                llvm::inconvertibleErrorCode());
        logVal((*allResultsOrErr)[idx]);
        return (*allResultsOrErr)[idx];
    }

    if (auto cmpOp = value.getDefiningOp<mlir::arith::CmpIOp>()) {
        auto step = handleCmpIOp(cmpOp, context);
        if (!step.isProduce())
            return llvm::make_error<llvm::StringError>(
                "cmpi did not produce a value", llvm::inconvertibleErrorCode());
        auto state = std::move(step).takeState();
        auto v = std::get<typename StepType::Produce>(std::move(state)).value;
        logVal(v);
        return v;
    }

    if (auto andOp = value.getDefiningOp<mlir::arith::AndIOp>()) {
        auto step = handleAndIOp(andOp, context);
        if (!step.isProduce())
            return llvm::make_error<llvm::StringError>(
                "andi did not produce a value", llvm::inconvertibleErrorCode());
        auto state = std::move(step).takeState();
        auto v = std::get<typename StepType::Produce>(std::move(state)).value;
        logVal(v);
        return v;
    }

    if (context.valueEnv) {
        auto it = context.valueEnv->find(value);
        if (it != context.valueEnv->end())
            return it->second;
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
    auto asUInt = [](const SemValue &v) -> std::uint64_t {
        return static_cast<std::uint64_t>(v.asInt64());
    };
    switch (op.getPredicate()) {
    case mlir::arith::CmpIPredicate::eq:
        result = lhsOrErr->cmpEqual(*rhsOrErr).asBool();
        break;
    case mlir::arith::CmpIPredicate::ne:
        result = lhsOrErr->cmpNotEqual(*rhsOrErr).asBool();
        break;
    case mlir::arith::CmpIPredicate::slt:
        result = lhsOrErr->cmpLess(*rhsOrErr).asBool();
        break;
    case mlir::arith::CmpIPredicate::sle:
        result = lhsOrErr->cmpLessEqual(*rhsOrErr).asBool();
        break;
    case mlir::arith::CmpIPredicate::sgt:
        result = lhsOrErr->cmpGreater(*rhsOrErr).asBool();
        break;
    case mlir::arith::CmpIPredicate::sge:
        result = lhsOrErr->cmpGreaterEqual(*rhsOrErr).asBool();
        break;
    case mlir::arith::CmpIPredicate::ult:
        result = asUInt(*lhsOrErr) < asUInt(*rhsOrErr);
        break;
    case mlir::arith::CmpIPredicate::ule:
        result = asUInt(*lhsOrErr) <= asUInt(*rhsOrErr);
        break;
    case mlir::arith::CmpIPredicate::ugt:
        result = asUInt(*lhsOrErr) > asUInt(*rhsOrErr);
        break;
    case mlir::arith::CmpIPredicate::uge:
        result = asUInt(*lhsOrErr) >= asUInt(*rhsOrErr);
        break;
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

namespace {

llvm::Error evalOpIntoEnv(SimpleSemantics &semantics, mlir::Operation &op,
                          SemanticsContext &context,
                          llvm::DenseMap<mlir::Value, SemValue> &env) {
    Step<SemValue> step = semantics.evalOperation(&op, context);
    auto state = std::move(step).takeState();
    if (!std::holds_alternative<Step<SemValue>::Produce>(state))
        return llvm::make_error<llvm::StringError>(
            "operation did not produce a value",
            llvm::inconvertibleErrorCode());

    auto &prod = std::get<Step<SemValue>::Produce>(state);
    if (op.getNumResults() != 1)
        return llvm::make_error<llvm::StringError>(
            "unsupported multi-result operation in loop",
            llvm::inconvertibleErrorCode());

    env[op.getResult(0)] = std::move(prod.value);
    return llvm::Error::success();
}

} // namespace

llvm::Expected<std::vector<SemValue>>
SimpleSemantics::evaluateLoopOp(simt::dialect::LoopOp loop,
                                SemanticsContext &context) {
    std::vector<SemValue> carried;
    auto inits = loop.getInits();
    carried.reserve(inits.size());
    for (mlir::Value init : inits) {
        auto valOrErr = evaluateValue(init, context);
        if (!valOrErr)
            return valOrErr.takeError();
        carried.push_back(std::move(*valOrErr));
    }

    llvm::SmallVector<SemValue, 4> forwarded;
    while (true) {
        // Evaluate prepare/condition region.
        auto &prepare = loop.getPrepareRegion().front();
        llvm::DenseMap<mlir::Value, SemValue> env;
        for (auto it : llvm::enumerate(prepare.getArguments())) {
            if (it.index() < carried.size())
                env[it.value()] = carried[it.index()];
        }
        SemanticsContext condCtx = context;
        condCtx.valueEnv = &env;

        bool shouldContinue = false;
        bool sawCondition = false;
        for (mlir::Operation &op : prepare) {
            if (auto condOp = llvm::dyn_cast<simt::dialect::ConditionOp>(&op)) {
                auto condValOrErr = evaluateValue(condOp.getCondition(), condCtx);
                if (!condValOrErr)
                    return condValOrErr.takeError();
                shouldContinue = condValOrErr->asBool();
                forwarded.clear();
                for (mlir::Value v : condOp.getForwarded()) {
                    auto valOrErr = evaluateValue(v, condCtx);
                    if (!valOrErr)
                        return valOrErr.takeError();
                    forwarded.push_back(std::move(*valOrErr));
                }
                sawCondition = true;
                break;
            }
            if (auto err = evalOpIntoEnv(*this, op, condCtx, env))
                return std::move(err);
        }
        if (!sawCondition)
            return llvm::make_error<llvm::StringError>(
                "simt.loop missing condition terminator",
                llvm::inconvertibleErrorCode());

        if (!shouldContinue)
            return std::vector<SemValue>(forwarded.begin(), forwarded.end());

        // Execute body region to produce new carried values.
        auto &body = loop.getBodyRegion().front();
        llvm::DenseMap<mlir::Value, SemValue> bodyEnv;
        for (auto it : llvm::enumerate(body.getArguments())) {
            if (it.index() < forwarded.size())
                bodyEnv[it.value()] = forwarded[it.index()];
        }
        SemanticsContext bodyCtx = context;
        bodyCtx.valueEnv = &bodyEnv;

        bool sawYield = false;
        llvm::SmallVector<SemValue, 4> nextCarried;
        for (mlir::Operation &op : body) {
            if (auto yieldOp = llvm::dyn_cast<simt::dialect::YieldOp>(&op)) {
                nextCarried.clear();
                for (mlir::Value v : yieldOp.getOperands()) {
                    auto valOrErr = evaluateValue(v, bodyCtx);
                    if (!valOrErr)
                        return valOrErr.takeError();
                    nextCarried.push_back(std::move(*valOrErr));
                }
                sawYield = true;
                break;
            }
            if (auto err = evalOpIntoEnv(*this, op, bodyCtx, bodyEnv))
                return std::move(err);
        }
        if (!sawYield)
            return llvm::make_error<llvm::StringError>(
                "simt.loop body missing yield", llvm::inconvertibleErrorCode());

        carried.assign(nextCarried.begin(), nextCarried.end());
    }
}

} // namespace simt::semantics
