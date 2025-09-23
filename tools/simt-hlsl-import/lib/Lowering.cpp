#include "simt-hlsl-import/Lowering.h"

#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Tooling.h"

#include <algorithm>
#include <optional>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Support/raw_ostream.h"

namespace simt_hlsl_import {

namespace {

struct LoweringContext {
  mlir::OpBuilder &builder;
  mlir::Location defaultLoc;
  mlir::Type returnType;
  llvm::DenseMap<const clang::ValueDecl *, mlir::Value> valueMap;
  llvm::SmallPtrSet<const clang::ValueDecl *, 8> mutatedVars;
  bool emittedTerminator = false;
  std::string &errorMessage;
  bool failed = false;

  LoweringContext(mlir::OpBuilder &builder, mlir::Location loc,
                  mlir::Type retType, std::string &error)
      : builder(builder), defaultLoc(loc), returnType(retType),
        errorMessage(error) {}

  bool fail(llvm::StringRef msg) {
    if (!failed)
      errorMessage = msg.str();
    failed = true;
    return false;
  }
};

static mlir::Type convertType(const clang::QualType &qt, mlir::OpBuilder &builder) {
  const clang::Type *type = qt.getCanonicalType().getTypePtrOrNull();
  if (!type)
    return {};

  if (const auto *builtin = llvm::dyn_cast<clang::BuiltinType>(type)) {
    switch (builtin->getKind()) {
    case clang::BuiltinType::Void:
      return builder.getNoneType();
    case clang::BuiltinType::Bool:
      return builder.getI1Type();
    case clang::BuiltinType::SChar:
    case clang::BuiltinType::Char_S:
    case clang::BuiltinType::Char_U:
    case clang::BuiltinType::UChar:
      return builder.getIntegerType(8);
    case clang::BuiltinType::Short:
    case clang::BuiltinType::UShort:
      return builder.getIntegerType(16);
    case clang::BuiltinType::Int:
    case clang::BuiltinType::UInt:
    case clang::BuiltinType::Long:
    case clang::BuiltinType::ULong:
      return builder.getIntegerType(32);
    case clang::BuiltinType::LongLong:
    case clang::BuiltinType::ULongLong:
      return builder.getIntegerType(64);
    case clang::BuiltinType::Half:
      return builder.getF16Type();
    case clang::BuiltinType::Float:
      return builder.getF32Type();
    case clang::BuiltinType::Double:
      return builder.getF64Type();
    default:
      break;
    }
  }

  return {};
}

static std::optional<std::string>
buildDxilTripleForProfile(llvm::StringRef profile) {
  llvm::SmallVector<llvm::StringRef, 4> parts;
  profile.split(parts, '_', /*MaxSplit=*/3, /*KeepEmpty=*/false);
  if (parts.empty())
    return std::nullopt;

  std::string stageLower = parts[0].lower();
  llvm::StringRef environment;
  if (stageLower == "cs")
    environment = "compute";
  else if (stageLower == "ps")
    environment = "pixel";
  else if (stageLower == "vs")
    environment = "vertex";
  else if (stageLower == "gs")
    environment = "geometry";
  else if (stageLower == "ds")
    environment = "domain";
  else if (stageLower == "hs")
    environment = "hull";
  else if (stageLower == "ms")
    environment = "mesh";
  else if (stageLower == "as")
    environment = "amplification";
  else if (stageLower == "lib" || stageLower == "library")
    environment = "library";
  else
    return std::nullopt;

  llvm::StringRef major = parts.size() > 1 ? parts[1] : "6";
  llvm::StringRef minor = parts.size() > 2 ? parts[2] : "0";
  std::string version = (llvm::Twine(major) + "." + minor).str();

  return (llvm::Twine("dxil-pc-shadermodel") + version + "-" + environment)
      .str();
}

static mlir::Location getLocation(const clang::Stmt *stmt,
                                  LoweringContext &ctx) {
  (void)stmt;
  return ctx.defaultLoc;
}

static mlir::Value lowerExpr(const clang::Expr *expr, LoweringContext &ctx) {
  if (!expr)
    return {};

  mlir::Type type = convertType(expr->getType(), ctx.builder);
  if (!type)
    return ctx.fail("unsupported expression type"), mlir::Value();

  mlir::Location loc = getLocation(expr, ctx);

  if (const auto *intLit = llvm::dyn_cast<clang::IntegerLiteral>(expr)) {
    if (!mlir::isa<mlir::IntegerType>(type))
      return ctx.fail("integer literal expects integer type"), mlir::Value();
    auto attr = ctx.builder.getIntegerAttr(mlir::cast<mlir::IntegerType>(type),
                                           intLit->getValue());
    return ctx.builder.create<mlir::arith::ConstantOp>(loc, attr);
  }

  if (const auto *paren = llvm::dyn_cast<clang::ParenExpr>(expr))
    return lowerExpr(paren->getSubExpr(), ctx);

  if (const auto *implicitCast =
          llvm::dyn_cast<clang::ImplicitCastExpr>(expr))
    return lowerExpr(implicitCast->getSubExpr(), ctx);

  if (const auto *floatLit = llvm::dyn_cast<clang::FloatingLiteral>(expr)) {
    if (!mlir::isa<mlir::FloatType>(type))
      return ctx.fail("floating literal expects floating type"), mlir::Value();
    auto attr = ctx.builder.getFloatAttr(mlir::cast<mlir::FloatType>(type),
                                         floatLit->getValue());
    return ctx.builder.create<mlir::arith::ConstantOp>(loc, attr);
  }

  if (const auto *declRef = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
    const clang::ValueDecl *vd = declRef->getDecl();
    auto it = ctx.valueMap.find(vd);
    if (it != ctx.valueMap.end())
      return it->second;
    return ctx.fail("reference to unknown value"), mlir::Value();
  }

  if (const auto *binOp = llvm::dyn_cast<clang::BinaryOperator>(expr)) {
    mlir::Value lhs = lowerExpr(binOp->getLHS(), ctx);
    mlir::Value rhs = lowerExpr(binOp->getRHS(), ctx);
    if (ctx.failed)
      return {};
    if (!lhs || !rhs)
      return {};

    switch (binOp->getOpcode()) {
    case clang::BinaryOperatorKind::BO_EQ:
    case clang::BinaryOperatorKind::BO_NE:
    case clang::BinaryOperatorKind::BO_LT:
    case clang::BinaryOperatorKind::BO_LE:
    case clang::BinaryOperatorKind::BO_GT:
    case clang::BinaryOperatorKind::BO_GE: {
      if (mlir::isa<mlir::IntegerType>(lhs.getType()) ||
          mlir::isa<mlir::IndexType>(lhs.getType())) {
        mlir::arith::CmpIPredicate predicate;
        switch (binOp->getOpcode()) {
        case clang::BinaryOperatorKind::BO_EQ:
          predicate = mlir::arith::CmpIPredicate::eq;
          break;
        case clang::BinaryOperatorKind::BO_NE:
          predicate = mlir::arith::CmpIPredicate::ne;
          break;
        case clang::BinaryOperatorKind::BO_LT:
          predicate = mlir::arith::CmpIPredicate::slt;
          break;
        case clang::BinaryOperatorKind::BO_LE:
          predicate = mlir::arith::CmpIPredicate::sle;
          break;
        case clang::BinaryOperatorKind::BO_GT:
          predicate = mlir::arith::CmpIPredicate::sgt;
          break;
        case clang::BinaryOperatorKind::BO_GE:
          predicate = mlir::arith::CmpIPredicate::sge;
          break;
        default:
          llvm_unreachable("unsupported integer comparison");
        }
        return ctx.builder.create<mlir::arith::CmpIOp>(loc, predicate, lhs, rhs);
      }
      if (mlir::isa<mlir::FloatType>(lhs.getType())) {
        mlir::arith::CmpFPredicate predicate;
        switch (binOp->getOpcode()) {
        case clang::BinaryOperatorKind::BO_EQ:
          predicate = mlir::arith::CmpFPredicate::OEQ;
          break;
        case clang::BinaryOperatorKind::BO_NE:
          predicate = mlir::arith::CmpFPredicate::UNE;
          break;
        case clang::BinaryOperatorKind::BO_LT:
          predicate = mlir::arith::CmpFPredicate::OLT;
          break;
        case clang::BinaryOperatorKind::BO_LE:
          predicate = mlir::arith::CmpFPredicate::OLE;
          break;
        case clang::BinaryOperatorKind::BO_GT:
          predicate = mlir::arith::CmpFPredicate::OGT;
          break;
        case clang::BinaryOperatorKind::BO_GE:
          predicate = mlir::arith::CmpFPredicate::OGE;
          break;
        default:
          llvm_unreachable("unsupported float comparison");
        }
        return ctx.builder.create<mlir::arith::CmpFOp>(loc, predicate, lhs, rhs);
      }
      return ctx.fail("unsupported comparison operands"), mlir::Value();
    }
    case clang::BinaryOperatorKind::BO_LAnd:
      if (lhs.getType() == ctx.builder.getI1Type())
        return ctx.builder.create<mlir::arith::AndIOp>(loc, lhs, rhs);
      return ctx.fail("logical and requires boolean operands"), mlir::Value();
    case clang::BinaryOperatorKind::BO_LOr:
      if (lhs.getType() == ctx.builder.getI1Type())
        return ctx.builder.create<mlir::arith::OrIOp>(loc, lhs, rhs);
      return ctx.fail("logical or requires boolean operands"), mlir::Value();
    case clang::BinaryOperatorKind::BO_Add:
      if (mlir::isa<mlir::IntegerType>(lhs.getType()))
        return ctx.builder.create<mlir::arith::AddIOp>(loc, lhs, rhs);
      if (mlir::isa<mlir::FloatType>(lhs.getType()))
        return ctx.builder.create<mlir::arith::AddFOp>(loc, lhs, rhs);
      break;
    case clang::BinaryOperatorKind::BO_Sub:
      if (mlir::isa<mlir::IntegerType>(lhs.getType()))
        return ctx.builder.create<mlir::arith::SubIOp>(loc, lhs, rhs);
      if (mlir::isa<mlir::FloatType>(lhs.getType()))
        return ctx.builder.create<mlir::arith::SubFOp>(loc, lhs, rhs);
      break;
    case clang::BinaryOperatorKind::BO_Mul:
      if (mlir::isa<mlir::IntegerType>(lhs.getType()))
        return ctx.builder.create<mlir::arith::MulIOp>(loc, lhs, rhs);
      if (mlir::isa<mlir::FloatType>(lhs.getType()))
        return ctx.builder.create<mlir::arith::MulFOp>(loc, lhs, rhs);
      break;
    case clang::BinaryOperatorKind::BO_Div:
      if (mlir::isa<mlir::IntegerType>(lhs.getType()))
        return ctx.builder.create<mlir::arith::DivSIOp>(loc, lhs, rhs);
      if (mlir::isa<mlir::FloatType>(lhs.getType()))
        return ctx.builder.create<mlir::arith::DivFOp>(loc, lhs, rhs);
      break;
    case clang::BinaryOperatorKind::BO_Rem:
      if (mlir::isa<mlir::IntegerType>(lhs.getType()))
        return ctx.builder.create<mlir::arith::RemSIOp>(loc, lhs, rhs);
      break;
    case clang::BinaryOperatorKind::BO_Assign:
      if (auto *lhsDeclRef = llvm::dyn_cast<clang::DeclRefExpr>(binOp->getLHS())) {
        auto it = ctx.valueMap.find(lhsDeclRef->getDecl());
        if (it != ctx.valueMap.end()) {
          it->second = rhs;
          ctx.mutatedVars.insert(lhsDeclRef->getDecl());
          return rhs;
        }
      }
      return ctx.fail("unsupported assignment target"), mlir::Value();
    default:
      break;
    }

    return ctx.fail("unsupported binary operator"), mlir::Value();
  }

  return ctx.fail("unsupported expression lowering"), mlir::Value();
}

static mlir::Value buildZeroValue(LoweringContext &ctx, mlir::Type type) {
  mlir::Location loc = ctx.defaultLoc;
  if (mlir::isa<mlir::IntegerType>(type)) {
    auto attr = ctx.builder.getIntegerAttr(mlir::cast<mlir::IntegerType>(type), 0);
    return ctx.builder.create<mlir::arith::ConstantOp>(loc, attr);
  }
  if (mlir::isa<mlir::FloatType>(type)) {
    auto attr = ctx.builder.getFloatAttr(mlir::cast<mlir::FloatType>(type), 0.0);
    return ctx.builder.create<mlir::arith::ConstantOp>(loc, attr);
  }
  ctx.fail("unable to build default value for return type");
  return {};
}

static bool lowerStatement(const clang::Stmt *stmt, LoweringContext &ctx);
static mlir::Value lowerExpr(const clang::Expr *expr, LoweringContext &ctx);
static bool lowerForStmt(const clang::ForStmt *stmt, LoweringContext &ctx);

static void lowerCompoundStmt(const clang::CompoundStmt *compound,
                              LoweringContext &ctx) {
  for (const clang::Stmt *child : compound->body()) {
    if (ctx.emittedTerminator || ctx.failed)
      break;
    lowerStatement(child, ctx);
  }
}

static mlir::Value getLoopCarriedValue(const LoweringContext &ctx,
                                       const clang::ValueDecl *vd) {
  auto it = ctx.valueMap.find(vd);
  if (it != ctx.valueMap.end())
    return it->second;
  return {};
}

static bool lowerStatement(const clang::Stmt *stmt, LoweringContext &ctx) {
  if (ctx.failed)
    return false;

  if (const auto *compound = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
    lowerCompoundStmt(compound, ctx);
    return true;
  }

  if (const auto *forStmt = llvm::dyn_cast<clang::ForStmt>(stmt))
    return lowerForStmt(forStmt, ctx);

  if (const auto *ifStmt = llvm::dyn_cast<clang::IfStmt>(stmt)) {
    mlir::Value cond = lowerExpr(ifStmt->getCond(), ctx);
    if (!cond)
      return false;

    mlir::Location loc = ctx.defaultLoc;
    bool hasElse = ifStmt->getElse() != nullptr;

    auto lowerIntoRegion = [&](const clang::Stmt *body, mlir::Region &region,
                               std::optional<mlir::OpBuilder> &builderStorage,
                               std::optional<LoweringContext> &ctxStorage,
                               LoweringContext *&outCtx) -> bool {
      region.emplaceBlock();
      builderStorage.emplace(ctx.builder.getContext());
      builderStorage->setInsertionPointToEnd(&region.front());
      ctxStorage.emplace(*builderStorage, loc, ctx.returnType, ctx.errorMessage);
      outCtx = &*ctxStorage;
      outCtx->valueMap = ctx.valueMap;
      outCtx->mutatedVars.clear();
      if (body && !lowerStatement(body, *outCtx))
        return false;
      return !outCtx->failed;
    };

    mlir::Region tmpThen;
    std::optional<mlir::OpBuilder> tmpThenBuilderStorage;
    std::optional<LoweringContext> tmpThenCtxStorage;
    LoweringContext *tmpThenCtx = nullptr;
    if (!lowerIntoRegion(ifStmt->getThen(), tmpThen, tmpThenBuilderStorage,
                         tmpThenCtxStorage, tmpThenCtx))
      return false;

    std::optional<mlir::Region> tmpElse;
    std::optional<mlir::OpBuilder> tmpElseBuilderStorage;
    std::optional<LoweringContext> tmpElseCtxStorage;
    LoweringContext *tmpElseCtx = nullptr;
    if (hasElse) {
      tmpElse.emplace();
      if (!lowerIntoRegion(ifStmt->getElse(), *tmpElse, tmpElseBuilderStorage,
                           tmpElseCtxStorage, tmpElseCtx))
        return false;
    }

    llvm::SmallPtrSet<const clang::ValueDecl *, 8> mutatedSet;
    llvm::SmallVector<const clang::ValueDecl *, 8> mutatedVars;
    auto addMutations = [&](const llvm::SmallPtrSet<const clang::ValueDecl *, 8> &source) {
      for (const clang::ValueDecl *vd : source)
        if (mutatedSet.insert(vd).second)
          mutatedVars.push_back(vd);
    };
    addMutations(tmpThenCtx->mutatedVars);
    if (tmpElseCtx)
      addMutations(tmpElseCtx->mutatedVars);

    llvm::SmallVector<mlir::Type, 8> resultTypes;
    resultTypes.reserve(mutatedVars.size());
    for (const clang::ValueDecl *vd : mutatedVars)
      resultTypes.push_back(ctx.valueMap.lookup(vd).getType());

    bool needElseRegion = hasElse || !mutatedVars.empty();
    auto ifOp = ctx.builder.create<simt::dialect::IfOp>(loc, resultTypes, cond,
                                                        needElseRegion);

    auto replaceRegionBody = [](mlir::Region &dest, mlir::Region &src) {
      if (!dest.empty())
        dest.front().erase();
      dest.takeBody(src);
      if (dest.empty())
        dest.emplaceBlock();
    };

    auto &finalThen = ifOp.getThenRegion();
    replaceRegionBody(finalThen, tmpThen);

    mlir::Region *finalElse = nullptr;
    if (needElseRegion) {
      finalElse = &ifOp.getElseRegion();
      if (tmpElse)
        replaceRegionBody(*finalElse, *tmpElse);
      else {
        if (!finalElse->empty())
          finalElse->front().erase();
        finalElse->emplaceBlock();
      }
    }

    if (!mutatedVars.empty()) {
      auto materializeYield = [&](mlir::Region &region,
                                  const LoweringContext *branchCtx) {
        llvm::SmallVector<mlir::Value, 8> operands;
        operands.reserve(mutatedVars.size());
        for (const clang::ValueDecl *vd : mutatedVars) {
          if (branchCtx) {
            auto it = branchCtx->valueMap.find(vd);
            if (it != branchCtx->valueMap.end()) {
              operands.push_back(it->second);
              continue;
            }
          }
          operands.push_back(ctx.valueMap.lookup(vd));
        }
        auto &block = region.front();
        mlir::Operation *maybeTerm = block.empty() ? nullptr : &block.back();
        if (!maybeTerm || !maybeTerm->hasTrait<mlir::OpTrait::IsTerminator>())
          maybeTerm = mlir::OpBuilder::atBlockEnd(&block)
                          .create<simt::dialect::YieldOp>(loc)
                          .getOperation();
        if (auto yield = llvm::dyn_cast<simt::dialect::YieldOp>(maybeTerm))
          yield.getOperation()->setOperands(operands);
      };

      materializeYield(finalThen, tmpThenCtx);
      if (finalElse)
        materializeYield(*finalElse, tmpElseCtx);
    }

    for (auto [index, vd] : llvm::enumerate(mutatedVars)) {
      ctx.valueMap[vd] = ifOp.getResult(index);
      ctx.mutatedVars.insert(vd);
    }

    if (tmpThenCtx->emittedTerminator &&
        (!needElseRegion || (tmpElseCtx && tmpElseCtx->emittedTerminator)))
      ctx.emittedTerminator = true;

    return true;
  }



  if (const auto *ret = llvm::dyn_cast<clang::ReturnStmt>(stmt)) {
    bool expectsValue = static_cast<bool>(ctx.returnType) &&
                        !mlir::isa<mlir::NoneType>(ctx.returnType);
    if (ret->getRetValue() != nullptr) {
      mlir::Value value = lowerExpr(ret->getRetValue(), ctx);
      if (!value)
        return false;
      if (!expectsValue)
        return ctx.fail("unexpected return value in void function");
      if (value.getType() != ctx.returnType)
        return ctx.fail("return type mismatch");
      ctx.builder.create<mlir::func::ReturnOp>(getLocation(ret, ctx), value);
    } else {
      if (expectsValue)
        return ctx.fail("missing return value");
      ctx.builder.create<mlir::func::ReturnOp>(getLocation(ret, ctx));
    }
    ctx.emittedTerminator = true;
    return true;
  }

  if (const auto *declStmt = llvm::dyn_cast<clang::DeclStmt>(stmt)) {
    for (const clang::Decl *decl : declStmt->decls()) {
      const auto *var = llvm::dyn_cast<clang::VarDecl>(decl);
      if (!var)
        continue;
      if (!convertType(var->getType(), ctx.builder))
        return ctx.fail("unsupported variable type");
      mlir::Value initValue;
      if (const clang::Expr *init = var->getInit()) {
        initValue = lowerExpr(init, ctx);
        if (!initValue)
          return false;
      }
      if (initValue)
        ctx.valueMap[var] = initValue;
    }
    return true;
  }

  if (const auto *exprStmt = llvm::dyn_cast<clang::Expr>(stmt)) {
    (void)lowerExpr(exprStmt, ctx);
    return !ctx.failed;
  }

  return ctx.fail("unsupported statement");
}

static bool lowerForStmt(const clang::ForStmt *forStmt, LoweringContext &ctx) {
  mlir::Location loc = ctx.defaultLoc;

  if (const clang::Stmt *init = forStmt->getInit()) {
    if (llvm::isa<clang::DeclStmt>(init)) {
      if (!lowerStatement(init, ctx) || ctx.failed)
        return false;
    } else if (const auto *initExpr = llvm::dyn_cast<clang::Expr>(init)) {
      (void)lowerExpr(initExpr, ctx);
      if (ctx.failed)
        return false;
    } else {
      return ctx.fail("unsupported for-loop initializer");
    }
  }

  llvm::SmallPtrSet<const clang::ValueDecl *, 8> mutatedSet;
  {
    mlir::Region analysisRegion;
    analysisRegion.emplaceBlock();
    mlir::OpBuilder analysisBuilder(ctx.builder.getContext());
    analysisBuilder.setInsertionPointToStart(&analysisRegion.front());
    LoweringContext analysisCtx(analysisBuilder, loc, ctx.returnType,
                                ctx.errorMessage);
    analysisCtx.valueMap = ctx.valueMap;
    analysisCtx.mutatedVars.clear();

    if (const clang::Stmt *body = forStmt->getBody()) {
      if (!lowerStatement(body, analysisCtx) || analysisCtx.failed)
        return false;
    }

    if (!analysisCtx.emittedTerminator) {
      if (const auto *incExpr =
              llvm::dyn_cast_or_null<clang::Expr>(forStmt->getInc())) {
        (void)lowerExpr(incExpr, analysisCtx);
        if (analysisCtx.failed)
          return false;
      } else if (const clang::Stmt *incStmt = forStmt->getInc()) {
        if (!lowerStatement(incStmt, analysisCtx) || analysisCtx.failed)
          return false;
      }
    }

    mutatedSet.insert(analysisCtx.mutatedVars.begin(),
                      analysisCtx.mutatedVars.end());
  }

  llvm::SmallVector<const clang::ValueDecl *, 8> mutatedVars(mutatedSet.begin(),
                                                             mutatedSet.end());
  llvm::sort(mutatedVars, [](const clang::ValueDecl *lhs,
                              const clang::ValueDecl *rhs) { return lhs < rhs; });

  llvm::SmallVector<mlir::Type, 8> resultTypes;
  llvm::SmallVector<mlir::Value, 8> initValues;
  resultTypes.reserve(mutatedVars.size());
  initValues.reserve(mutatedVars.size());
  for (const clang::ValueDecl *vd : mutatedVars) {
    mlir::Value initial = getLoopCarriedValue(ctx, vd);
    if (!initial)
      return ctx.fail("reference to unknown loop variable");
    resultTypes.push_back(initial.getType());
    initValues.push_back(initial);
  }

  auto loopOp = ctx.builder.create<simt::dialect::LoopOp>(loc, resultTypes,
                                                          initValues);

  auto &prepareRegion = loopOp.getPrepareRegion();
  if (prepareRegion.empty())
    prepareRegion.emplaceBlock();
  auto &prepareBlock = prepareRegion.front();
  if (!resultTypes.empty()) {
    llvm::SmallVector<mlir::Location, 8> argLocs(resultTypes.size(), loc);
    prepareBlock.addArguments(resultTypes, argLocs);
  }

  auto &bodyRegion = loopOp.getBodyRegion();
  if (bodyRegion.empty())
    bodyRegion.emplaceBlock();
  auto &bodyBlock = bodyRegion.front();
  if (!resultTypes.empty()) {
    llvm::SmallVector<mlir::Location, 8> argLocs(resultTypes.size(), loc);
    bodyBlock.addArguments(resultTypes, argLocs);
  }

  // Prepare region: evaluate loop condition.
  {
    mlir::OpBuilder prepBuilder(ctx.builder.getContext());
    prepBuilder.setInsertionPointToStart(&prepareBlock);
    LoweringContext prepCtx(prepBuilder, loc, ctx.returnType, ctx.errorMessage);
    prepCtx.valueMap = ctx.valueMap;
    for (auto [index, vd] : llvm::enumerate(mutatedVars))
      prepCtx.valueMap[vd] = prepareBlock.getArgument(index);

    mlir::Value condValue;
    if (const clang::Expr *condExpr = forStmt->getCond()) {
      condValue = lowerExpr(condExpr, prepCtx);
      if (!condValue || prepCtx.failed)
        return false;
    } else {
      condValue = prepBuilder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
    }

    llvm::SmallVector<mlir::Value, 8> forwarded;
    forwarded.reserve(mutatedVars.size());
    for (const clang::ValueDecl *vd : mutatedVars)
      forwarded.push_back(prepCtx.valueMap.lookup(vd));

    prepBuilder.create<simt::dialect::ConditionOp>(loc, condValue, forwarded);
  }

  // Body region: execute loop body and increment.
  {
    mlir::OpBuilder bodyBuilder(ctx.builder.getContext());
    bodyBuilder.setInsertionPointToStart(&bodyBlock);
    LoweringContext bodyCtx(bodyBuilder, loc, ctx.returnType, ctx.errorMessage);
    bodyCtx.valueMap = ctx.valueMap;
    for (auto [index, vd] : llvm::enumerate(mutatedVars))
      bodyCtx.valueMap[vd] = bodyBlock.getArgument(index);

    if (const clang::Stmt *body = forStmt->getBody()) {
      if (!lowerStatement(body, bodyCtx) || bodyCtx.failed)
        return false;
    }

    if (!bodyCtx.emittedTerminator) {
      if (const auto *incExpr =
              llvm::dyn_cast_or_null<clang::Expr>(forStmt->getInc())) {
        (void)lowerExpr(incExpr, bodyCtx);
        if (bodyCtx.failed)
          return false;
      } else if (const clang::Stmt *incStmt = forStmt->getInc()) {
        if (!lowerStatement(incStmt, bodyCtx) || bodyCtx.failed)
          return false;
      }
    }

    if (!bodyCtx.emittedTerminator) {
      llvm::SmallVector<mlir::Value, 8> yieldOperands;
      yieldOperands.reserve(mutatedVars.size());
      for (const clang::ValueDecl *vd : mutatedVars) {
        mlir::Value value = bodyCtx.valueMap.lookup(vd);
        if (!value)
          value = ctx.valueMap.lookup(vd);
        yieldOperands.push_back(value);
      }
      bodyBuilder.create<simt::dialect::YieldOp>(loc, yieldOperands);
    }
  }

  for (auto [index, vd] : llvm::enumerate(mutatedVars)) {
    ctx.valueMap[vd] = loopOp.getResult(index);
    ctx.mutatedVars.insert(vd);
  }

  return true;
}

class FunctionLoweringVisitor
    : public clang::RecursiveASTVisitor<FunctionLoweringVisitor> {
public:
  FunctionLoweringVisitor(mlir::OwningOpRef<mlir::ModuleOp> &module,
                          mlir::OpBuilder &builder)
      : module(module), moduleBuilder(builder) {}

  bool VisitFunctionDecl(const clang::FunctionDecl *decl) {
    const auto *shaderAttr = decl->getAttr<clang::HLSLShaderAttr>();
    if (!shaderAttr || shaderAttr->getType() != llvm::Triple::Compute)
      return true;

    foundComputeShader = true;

    auto name = decl->getNameAsString();
    mlir::Location loc = moduleBuilder.getUnknownLoc();

    mlir::OpBuilder::InsertionGuard guard(moduleBuilder);
    moduleBuilder.setInsertionPointToEnd(module->getBody());

    llvm::SmallVector<mlir::Type> argTypes;
    argTypes.reserve(decl->getNumParams());
    for (const clang::ParmVarDecl *param : decl->parameters()) {
      mlir::Type type = convertType(param->getType(), moduleBuilder);
      if (!type || mlir::isa<mlir::NoneType>(type)) {
        recordError("unsupported parameter type in function '" + name + "'");
        return false;
      }
      argTypes.push_back(type);
    }

    mlir::Type resultType = convertType(decl->getReturnType(), moduleBuilder);
    if (!resultType)
      resultType = moduleBuilder.getNoneType();

    llvm::SmallVector<mlir::Type> resultTypes;
    if (!mlir::isa<mlir::NoneType>(resultType))
      resultTypes.push_back(resultType);

    auto funcType = moduleBuilder.getFunctionType(argTypes, resultTypes);
    auto func = moduleBuilder.create<mlir::func::FuncOp>(loc, name, funcType);

    if (const auto *numThreads = decl->getAttr<clang::HLSLNumThreadsAttr>()) {
      llvm::SmallVector<int64_t, 3> dims = {
          static_cast<int64_t>(numThreads->getX()),
          static_cast<int64_t>(numThreads->getY()),
          static_cast<int64_t>(numThreads->getZ())};
      auto attr = mlir::DenseI64ArrayAttr::get(func.getContext(), dims);
      func->setAttr("simt.num_threads", attr);
    }

    mlir::Block *entry = func.addEntryBlock();
    mlir::OpBuilder funcBuilder(entry, entry->begin());
    funcBuilder.create<simt::dialect::ActiveMaskOp>(loc,
                                                   funcBuilder.getI64Type());

    LoweringContext ctx(funcBuilder, loc,
                        resultTypes.empty() ? mlir::Type() : resultTypes.front(),
                        errorMessage);
    for (auto [param, arg] : llvm::zip(decl->parameters(), entry->getArguments()))
      ctx.valueMap[param] = arg;

    const clang::Stmt *body = decl->getBody();
    if (const auto *compound = llvm::dyn_cast<clang::CompoundStmt>(body))
      lowerCompoundStmt(compound, ctx);
    else if (body)
      lowerStatement(body, ctx);

    if (ctx.failed)
      return false;

    if (!ctx.emittedTerminator) {
      bool expectsValue = static_cast<bool>(ctx.returnType) &&
                          !mlir::isa<mlir::NoneType>(ctx.returnType);
      if (expectsValue) {
        mlir::Value zero = buildZeroValue(ctx, ctx.returnType);
        if (!zero)
          return false;
        ctx.builder.create<mlir::func::ReturnOp>(ctx.builder.getUnknownLoc(), zero);
      } else {
        ctx.builder.create<mlir::func::ReturnOp>(ctx.builder.getUnknownLoc());
      }
    }

    return true;
  }

  bool encounteredError() const { return !errorMessage.empty(); }
  const std::string &error() const { return errorMessage; }
  bool hasComputeShader() const { return foundComputeShader; }

private:
  void recordError(const std::string &msg) {
    if (errorMessage.empty())
      errorMessage = msg;
  }

  mlir::OwningOpRef<mlir::ModuleOp> &module;
  mlir::OpBuilder &moduleBuilder;
  std::string errorMessage;
  bool foundComputeShader = false;
};

class TranslationASTConsumer : public clang::ASTConsumer {
public:
  explicit TranslationASTConsumer(FunctionLoweringVisitor &visitor)
      : visitor(visitor) {}

  void HandleTranslationUnit(clang::ASTContext &context) override {
    visitor.TraverseDecl(context.getTranslationUnitDecl());
  }

private:
  FunctionLoweringVisitor &visitor;
};

class TranslationFrontendAction : public clang::ASTFrontendAction {
public:
  explicit TranslationFrontendAction(FunctionLoweringVisitor &visitor)
      : visitor(visitor) {}

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &, llvm::StringRef) override {
    return std::make_unique<TranslationASTConsumer>(visitor);
  }

private:
  FunctionLoweringVisitor &visitor;
};

} // namespace

Result<mlir::OwningOpRef<mlir::ModuleOp>>
translateComputeShader(mlir::MLIRContext &context, llvm::StringRef fileName,
                       llvm::StringRef source, const TranslationOptions &options) {
  context.loadDialect<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                      simt::dialect::SimtStepDialect>();

  mlir::OpBuilder builder(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module->getBody());

  FunctionLoweringVisitor visitor(module, builder);
  std::vector<std::string> clangArgs = {
      "-x", "hlsl", "-std=hlsl2021", "-D__HLSL__"};

  if (auto triple = buildDxilTripleForProfile(options.shaderProfile)) {
    clangArgs.emplace_back("-target");
    clangArgs.emplace_back(std::move(*triple));
  } else {
    return Result<mlir::OwningOpRef<mlir::ModuleOp>>::err(
        "unsupported shader profile '" + options.shaderProfile + "'");
  }

  clangArgs.emplace_back("-Xclang");
  clangArgs.emplace_back("-finclude-default-header");
  clangArgs.emplace_back("-Wno-hlsl-dxc-compatability");

  for (const std::string &dir : options.extraIncludeDirs) {
    clangArgs.emplace_back("-isystem");
    clangArgs.emplace_back(dir);
  }

  if (!options.resourceDir.empty()) {
    clangArgs.emplace_back("-resource-dir");
    clangArgs.emplace_back(options.resourceDir);
  }

  auto action = std::make_unique<TranslationFrontendAction>(visitor);
  if (!clang::tooling::runToolOnCodeWithArgs(std::move(action), source.str(),
                                             clangArgs, fileName.str()))
    return Result<mlir::OwningOpRef<mlir::ModuleOp>>::err(
        "failed to translate HLSL input");

  if (visitor.encounteredError())
    return Result<mlir::OwningOpRef<mlir::ModuleOp>>::err(visitor.error());

  if (!visitor.hasComputeShader())
    return Result<mlir::OwningOpRef<mlir::ModuleOp>>::err(
        "no compute shader entry point found");

  return Result<mlir::OwningOpRef<mlir::ModuleOp>>::ok(std::move(module));
}

} // namespace simt_hlsl_import
