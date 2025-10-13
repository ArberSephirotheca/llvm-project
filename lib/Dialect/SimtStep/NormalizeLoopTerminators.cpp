#include "simt-step/Dialect/SimtStep/Transforms.h"

#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <optional>

using namespace simt::dialect;

namespace {

constexpr llvm::StringLiteral kNormalizedAttrName(
    "simt.normalized.loop_terminators");

enum class ExitKind { None, Continue, Break };

struct BranchInfo {
  ExitKind kind = ExitKind::None;
  llvm::SmallVector<mlir::Value> ifValues;
  llvm::SmallVector<mlir::Value> extraValues;
};

static std::optional<ExitKind> classifyTerminator(mlir::Operation *op) {
  if (llvm::isa<simt::dialect::YieldOp>(op))
    return ExitKind::None;
  if (llvm::isa<simt::dialect::ContinueOp>(op))
    return ExitKind::Continue;
  if (llvm::isa<simt::dialect::BreakOp>(op))
    return ExitKind::Break;
  return std::nullopt;
}

static std::optional<BranchInfo> analyzeBranch(mlir::Region &region,
                                               unsigned resultCount) {
  if (region.empty())
    return std::nullopt;
  mlir::Block &block = region.front();
  mlir::Operation *terminator = block.getTerminator();
  std::optional<ExitKind> kind = classifyTerminator(terminator);
  if (!kind)
    return std::nullopt;
  BranchInfo info;
  info.kind = *kind;
  if (terminator->getNumOperands() < resultCount)
    return std::nullopt;
  info.ifValues.append(terminator->operand_begin(),
                       terminator->operand_begin() + resultCount);
  info.extraValues.append(terminator->operand_begin() + resultCount,
                          terminator->operand_end());
  return info;
}

static bool shouldRewrite(simt::dialect::IfOp ifOp) {
  if (!ifOp)
    return false;
  if (ifOp->hasAttr(kNormalizedAttrName))
    return false;
  if (ifOp.getNumResults() == 0)
    return false;

  auto loop = ifOp->getParentOfType<simt::dialect::LoopOp>();
  if (!loop)
    return false;
  if (&loop.getBodyRegion() != ifOp->getParentRegion())
    return false;
  if (ifOp.getElseRegion().empty())
    return false;

  mlir::Operation *thenTerm =
      ifOp.getThenRegion().front().getTerminator();
  mlir::Operation *elseTerm =
      ifOp.getElseRegion().front().getTerminator();
  auto thenKind = classifyTerminator(thenTerm);
  auto elseKind = classifyTerminator(elseTerm);
  if (!thenKind || !elseKind)
    return false;

  bool hasEarly = *thenKind != ExitKind::None || *elseKind != ExitKind::None;
  if (!hasEarly)
    return false;

  // Mismatched early exit kinds (continue vs break) are not supported yet.
  if (*thenKind != ExitKind::None && *elseKind != ExitKind::None &&
      *thenKind != *elseKind)
    return false;

  unsigned resultCount = ifOp.getNumResults();
  if (thenTerm->getNumOperands() < resultCount)
    return false;
  if (elseTerm->getNumOperands() < resultCount)
    return false;

  if (*thenKind != ExitKind::None && *elseKind != ExitKind::None &&
      thenTerm->getNumOperands() != elseTerm->getNumOperands())
    return false;

  return true;
}

class NormalizeLoopTerminatorsPass
    : public mlir::PassWrapper<NormalizeLoopTerminatorsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NormalizeLoopTerminatorsPass)

  void getDependentDialects(
      mlir::DialectRegistry &registry) const override {
    registry.insert<simt::dialect::SimtStepDialect,
                    mlir::arith::ArithDialect>();
  }

  mlir::StringRef getArgument() const final {
    return "simt-normalize-loop-terminators";
  }

  mlir::StringRef getDescription() const final {
    return "Normalize result-producing simt_step.if ops so loop continue/break "
           "terminators appear outside the value-producing region.";
  }

  void runOnOperation() final {
    mlir::func::FuncOp func = getOperation();
    llvm::SmallVector<simt::dialect::IfOp> worklist;
    func.walk([&](simt::dialect::IfOp ifOp) {
      if (shouldRewrite(ifOp))
        worklist.push_back(ifOp);
    });

    if (worklist.empty())
      return;

    mlir::IRRewriter rewriter(&getContext());

    for (simt::dialect::IfOp ifOp : worklist) {
      if (!ifOp || ifOp->hasAttr(kNormalizedAttrName))
        continue;
      if (!ifOp->getParentOp())
        continue;

      unsigned resultCount = ifOp.getNumResults();
      mlir::Operation *thenTerm =
          ifOp.getThenRegion().front().getTerminator();
      mlir::Operation *elseTerm =
          ifOp.getElseRegion().front().getTerminator();

      auto thenInfo = analyzeBranch(ifOp.getThenRegion(), resultCount);
      auto elseInfo = analyzeBranch(ifOp.getElseRegion(), resultCount);
      if (!thenInfo || !elseInfo)
        continue;

      ExitKind earlyKind = thenInfo->kind != ExitKind::None
                               ? thenInfo->kind
                               : elseInfo->kind;
      if (earlyKind == ExitKind::None)
        continue;
      // We've already verified consistency in shouldRewrite.

      mlir::Operation *earlyTerm =
          thenInfo->kind != ExitKind::None ? thenTerm : elseTerm;
      unsigned terminatorArity = earlyTerm->getNumOperands();
      unsigned extraSlots =
          terminatorArity > resultCount ? terminatorArity - resultCount : 0;

      llvm::SmallVector<mlir::Type> extraTypes;
      extraTypes.reserve(extraSlots);
      for (unsigned idx = 0; idx < extraSlots; ++idx) {
        mlir::Value operand = earlyTerm->getOperand(resultCount + idx);
        extraTypes.push_back(operand.getType());
      }

      auto supportsExtraType = [](mlir::Type type) {
        return mlir::isa<mlir::IntegerType, mlir::IndexType, mlir::FloatType>(
            type);
      };
      if (llvm::any_of(extraTypes,
                        [&](mlir::Type type) { return !supportsExtraType(type); }))
        continue;

      auto createDefaultValue = [&](mlir::Type type, mlir::Location loc)
          -> mlir::Value {
        if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(type))
          return rewriter.create<mlir::arith::ConstantIntOp>(loc, 0,
                                                             intTy.getWidth());
        if (mlir::isa<mlir::IndexType>(type))
          return rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
        if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(type))
          return rewriter.create<mlir::arith::ConstantOp>(
              loc, floatTy, rewriter.getFloatAttr(floatTy, 0.0));
        llvm_unreachable("unsupported extra type should have been filtered");
      };

      mlir::Location loc = ifOp.getLoc();

      rewriter.setInsertionPoint(ifOp);
      llvm::SmallVector<mlir::Type> stage1Types;
      stage1Types.push_back(rewriter.getI1Type());
      llvm::append_range(stage1Types, ifOp.getResultTypes());
      llvm::append_range(stage1Types, extraTypes);
      auto stage1If = rewriter.create<simt::dialect::IfOp>(
          loc, stage1Types, ifOp.getCondition(), /*withElseRegion=*/true);

      // Move the original bodies into the staging if.
      stage1If.getThenRegion().takeBody(ifOp.getThenRegion());
      stage1If.getElseRegion().takeBody(ifOp.getElseRegion());

      auto rewriteBranch = [&](mlir::Region &region, const BranchInfo &info) {
        mlir::Block &block = region.front();
        mlir::Operation *term = block.getTerminator();
        rewriter.setInsertionPoint(term);
        mlir::Value flag = rewriter.create<mlir::arith::ConstantIntOp>(
            term->getLoc(), info.kind == ExitKind::None ? 0 : 1, 1);
        llvm::SmallVector<mlir::Value> payload;
        payload.push_back(flag);
        payload.append(info.ifValues.begin(), info.ifValues.end());
        for (unsigned idx = 0; idx < extraSlots; ++idx) {
          if (idx < info.extraValues.size()) {
            payload.push_back(info.extraValues[idx]);
            continue;
          }
          if (!extraTypes.empty()) {
            payload.push_back(
                createDefaultValue(extraTypes[idx], term->getLoc()));
          }
        }
        rewriter.replaceOpWithNewOp<simt::dialect::YieldOp>(term, payload);
      };

      rewriteBranch(stage1If.getThenRegion(), *thenInfo);
      rewriteBranch(stage1If.getElseRegion(), *elseInfo);

      mlir::Value flag = stage1If.getResult(0);
      mlir::ValueRange payloadValues(stage1If.getResults().drop_front());
      mlir::ValueRange tupleValues(payloadValues.take_front(resultCount));
      mlir::ValueRange extraPayload(payloadValues.drop_front(resultCount));

      rewriter.setInsertionPointAfter(stage1If);
      auto stage2If =
          rewriter.create<simt::dialect::IfOp>(loc, mlir::TypeRange{}, flag,
                                               /*withElseRegion=*/true);

      auto &thenBlock = stage2If.getThenRegion().front();
      thenBlock.clear();
      rewriter.setInsertionPointToStart(&thenBlock);
      llvm::SmallVector<mlir::Value> terminatorOperands;
      terminatorOperands.append(tupleValues.begin(), tupleValues.end());
      terminatorOperands.append(extraPayload.begin(), extraPayload.end());
      if (earlyKind == ExitKind::Continue)
        rewriter.create<simt::dialect::ContinueOp>(loc, terminatorOperands);
      else
        rewriter.create<simt::dialect::BreakOp>(loc, terminatorOperands);

      auto &elseBlock = stage2If.getElseRegion().front();
      elseBlock.clear();
      rewriter.setInsertionPointToStart(&elseBlock);
      rewriter.create<simt::dialect::YieldOp>(loc);

      stage2If->setAttr(kNormalizedAttrName, rewriter.getUnitAttr());

      rewriter.replaceOp(ifOp, tupleValues);
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
simt::dialect::createNormalizeLoopTerminatorsPass() {
  return std::make_unique<NormalizeLoopTerminatorsPass>();
}

void simt::dialect::registerNormalizeLoopTerminatorsPass() {
  static bool init = []() {
    ::mlir::PassRegistration<NormalizeLoopTerminatorsPass>();
    return true;
  }();
  (void)init;
}
