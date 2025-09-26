#include "simt-hlsl-import/Lowering.h"

#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/Builders.h"
#include "simt-hlsl-import/LoopScopeSupport.h"
#include "simt-hlsl-import/LoweringAlgebra.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/Basic/AddressSpaces.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Tooling.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <memory>
#include <optional>
#include <tuple>
#include <type_traits>
#include <string>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

namespace simt_hlsl_import {

namespace {

static mlir::Location resolveLoc(const simt_hlsl_import::SourceLoc &src,
                                 LoweringContext &ctx) {
  // if (src.mlirLoc != mlir::Location())
  //   return src.mlirLoc;
  // return ctx.defaultLoc;
  return src.mlirLoc;
}

static std::string buildIntegerTag(mlir::IntegerType type) {
  return (llvm::Twine("i") + llvm::Twine(type.getWidth())).str();
}

static std::string buildFloatTag(mlir::FloatType type) {
  if (type.isF16())
    return "f16";
  if (type.isBF16())
    return "bf16";
  if (type.isF32())
    return "f32";
  if (type.isF64())
    return "f64";
  if (type.isF80())
    return "f80";
  if (type.isF128())
    return "f128";
  return "";
}

static void ensureYield(LoweringContext &branchCtx, mlir::Region &region,
                        llvm::ArrayRef<mlir::Value> operands,
                        mlir::Location loc) {
  if (operands.empty() || branchCtx.emittedTerminator)
    return;

  mlir::Block &block = region.front();
  if (!block.empty()) {
    mlir::Operation &terminator = block.back();
    if (auto yield = llvm::dyn_cast<simt::dialect::YieldOp>(&terminator)) {
      yield.getOperation()->setOperands(operands);
      return;
    }
    if (terminator.hasTrait<mlir::OpTrait::IsTerminator>())
      return;
  }

  mlir::OpBuilder::atBlockEnd(&block)
      .create<simt::dialect::YieldOp>(loc, operands);
}

static mlir::Type parseTypeTag(llvm::StringRef tag, LoweringContext &ctx) {
  if (tag.empty())
    return {};
  auto &builder = ctx.builder;
  if (tag.consume_front("i")) {
    unsigned width = 0;
    if (tag.getAsInteger(10, width))
      return {};
    return builder.getIntegerType(width);
  }
  if (tag.consume_front("f")) {
    unsigned width = 0;
    if (tag.getAsInteger(10, width))
      return {};
    switch (width) {
    case 16:
      return builder.getF16Type();
    case 32:
      return builder.getF32Type();
    case 64:
      return builder.getF64Type();
    case 80:
      return builder.getF80Type();
    case 128:
      return builder.getF128Type();
    default:
      return {};
    }
  }
  if (tag == "bf16")
    return builder.getBF16Type();
  if (tag == "index")
    return builder.getIndexType();
  return {};
}

static SymValue makeSymValueForType(mlir::Type type) {
  SymValue sym;
  if (!type)
    return sym;

  if (auto intType = llvm::dyn_cast<mlir::IntegerType>(type)) {
    sym.kind = SymKind::ScalarInt;
    sym.bitWidth = intType.getWidth();
    return sym;
  }

  if (auto floatType = llvm::dyn_cast<mlir::FloatType>(type)) {
    sym.kind = SymKind::ScalarFloat;
    sym.bitWidth = floatType.getWidth();
    return sym;
  }

  if (auto vectorType = llvm::dyn_cast<mlir::VectorType>(type)) {
    sym.kind = SymKind::Vector;
    sym.elementCount = vectorType.getNumElements();
    if (auto elemInt =
            llvm::dyn_cast<mlir::IntegerType>(vectorType.getElementType()))
      sym.bitWidth = elemInt.getWidth();
    else if (auto elemFloat =
                 llvm::dyn_cast<mlir::FloatType>(vectorType.getElementType()))
      sym.bitWidth = elemFloat.getWidth();
    return sym;
  }

  if (mlir::isa<mlir::IndexType>(type)) {
    sym.kind = SymKind::ScalarInt;
    return sym;
  }

  return sym;
}

template <typename Interp>
static typename Interp::Value wrapMlirValue(Interp &, mlir::Value value) {
  if constexpr (std::is_same_v<typename Interp::Value, mlir::Value>) {
    return value;
  } else {
    AnalysisValue result = AnalysisValue::fromValue(value);
    result.setTypeHint(value.getType());
    result.setSym(makeSymValueForType(value.getType()));
    return result;
  }
}

template <typename ValueT>
static mlir::Value unwrapValue(const ValueT &value) {
  if constexpr (std::is_same_v<ValueT, mlir::Value>) {
    return value;
  } else {
    return value.getValueOrNull();
  }
}

template <typename ValueT>
static mlir::Type getValueType(const ValueT &value) {
  if constexpr (std::is_same_v<ValueT, mlir::Value>) {
    return value ? value.getType() : mlir::Type();
  } else {
    return value.getType();
  }
}


struct EmitInterpreter
    : simt_hlsl_import::LoweringAlgebra<EmitInterpreter, mlir::Value> {

  explicit EmitInterpreter(LoweringContext &ctx) : ctx(ctx) {
    assert(isEmitContext(ctx) && "EmitInterpreter requires anchored builder");
  }

  using Value = mlir::Value;

  EmitInterpreter fork(LoweringContext &childCtx) {
    return EmitInterpreter(childCtx);
  }

  struct IfScope {
    IfScope(LoweringContext &parentCtx, simt::dialect::IfOp op,
            llvm::ArrayRef<const clang::ValueDecl *> carried, bool hasElse,
            bool needsElseRegion, mlir::Location loc)
        : parent(parentCtx), ifOp(op),
          mutated(carried.begin(), carried.end()), hasElseBranch(hasElse),
          needsElseRegion(needsElseRegion), loc(loc) {}

    LoweringContext &thenContext() {
      if (!thenCtxStorage) {
        mlir::Region &region = ifOp.getThenRegion();
        thenBuilder.emplace(parent.builder.getContext());
        thenBuilder->setInsertionPointToEnd(&region.front());
        thenCtxStorage.emplace(*thenBuilder, loc, parent.returnType,
                               parent.errorMessage, parent.sourceManager);
        cloneContextState(parent, *thenCtxStorage);
      }
      return *thenCtxStorage;
    }

    bool userHasElse() const { return hasElseBranch; }

    LoweringContext &elseContext() {
      if (!hasElseBranch)
        llvm::report_fatal_error("elseContext requested but no else branch");
      if (!elseCtxStorage) {
        mlir::Region &region = ifOp.getElseRegion();
        elseBuilder.emplace(parent.builder.getContext());
        elseBuilder->setInsertionPointToEnd(&region.front());
        elseCtxStorage.emplace(*elseBuilder, loc, parent.returnType,
                               parent.errorMessage, parent.sourceManager);
        cloneContextState(parent, *elseCtxStorage);
      }
      return *elseCtxStorage;
    }

    bool finalizeThen() {
      if (thenClosed)
        return !parent.failed;
      thenClosed = true;
      if (!thenCtxStorage || mutated.empty())
        return !parent.failed;

      llvm::SmallVector<mlir::Value, 8> operands;
      operands.reserve(mutated.size());
      for (const clang::ValueDecl *vd : mutated) {
        mlir::Value value;
        if (auto it = thenCtxStorage->valueMap.find(vd);
            it != thenCtxStorage->valueMap.end())
          value = it->second;
        else if (auto pit = parent.valueMap.find(vd);
                 pit != parent.valueMap.end())
          value = pit->second;
        if (!value) {
          parent.fail("conditional branch missing carried value");
          return false;
        }
        operands.push_back(value);
      }
      ensureYield(*thenCtxStorage, ifOp.getThenRegion(), operands, loc);
      return !parent.failed;
    }

    bool finalizeElse() {
      if (elseClosed)
        return !parent.failed;
      elseClosed = true;
      if (!elseCtxStorage || mutated.empty())
        return !parent.failed;

      llvm::SmallVector<mlir::Value, 8> operands;
      operands.reserve(mutated.size());
      for (const clang::ValueDecl *vd : mutated) {
        mlir::Value value;
        if (auto it = elseCtxStorage->valueMap.find(vd);
            it != elseCtxStorage->valueMap.end())
          value = it->second;
        else if (auto pit = parent.valueMap.find(vd);
                 pit != parent.valueMap.end())
          value = pit->second;
        if (!value) {
          parent.fail("conditional branch missing carried value");
          return false;
        }
        operands.push_back(value);
      }
      ensureYield(*elseCtxStorage, ifOp.getElseRegion(), operands, loc);
      return !parent.failed;
    }

    bool finalizeImplicitElse() {
      if (!needsElseRegion || hasElseBranch || mutated.empty())
        return true;
      llvm::SmallVector<mlir::Value, 8> operands;
      operands.reserve(mutated.size());
      for (const clang::ValueDecl *vd : mutated) {
        auto it = parent.valueMap.find(vd);
        if (it == parent.valueMap.end()) {
          parent.fail("conditional missing carried value for implicit else");
          return false;
        }
        operands.push_back(it->second);
      }
      ensureYield(parent, ifOp.getElseRegion(), operands, loc);
      return !parent.failed;
    }

    bool done() {
      if (!finalizeThen())
        return false;
      if (hasElseBranch) {
        if (!finalizeElse())
          return false;
      } else if (!finalizeImplicitElse()) {
        return false;
      }

      unsigned resultIndex = 0;
      for (const clang::ValueDecl *vd : mutated) {
        parent.valueMap[vd] = ifOp.getResult(resultIndex++);
        parent.mutatedVars.insert(vd);
        if (thenCtxStorage) {
          if (auto it = thenCtxStorage->symValueMap.find(vd);
              it != thenCtxStorage->symValueMap.end())
            parent.symValueMap[vd] = it->second;
        }
        if (elseCtxStorage) {
          if (auto it = elseCtxStorage->symValueMap.find(vd);
              it != elseCtxStorage->symValueMap.end())
            parent.symValueMap[vd] = it->second;
        }
      }

      bool thenTerminated =
          thenCtxStorage && thenCtxStorage->emittedTerminator;
      bool elseTerminated = false;
      if (hasElseBranch)
        elseTerminated = elseCtxStorage && elseCtxStorage->emittedTerminator;
      else if (!needsElseRegion)
        elseTerminated = true;

      if (thenTerminated && (!needsElseRegion || elseTerminated))
        parent.emittedTerminator = true;

      return !parent.failed;
    }

    LoweringContext &parent;
    simt::dialect::IfOp ifOp;
    llvm::SmallVector<const clang::ValueDecl *, 8> mutated;
    bool hasElseBranch = false;
    bool needsElseRegion = false;
    mlir::Location loc;

    std::optional<mlir::OpBuilder> thenBuilder;
    std::optional<LoweringContext> thenCtxStorage;
    bool thenClosed = false;

    std::optional<mlir::OpBuilder> elseBuilder;
    std::optional<LoweringContext> elseCtxStorage;
    bool elseClosed = false;
  };

  using LoopScope = LoopScopeState;

  LoopScope beginLoop(llvm::ArrayRef<const clang::ValueDecl *> carriedVars,
                      bool hasFirstIterFlag, mlir::Value firstIterInit,
                      mlir::Location loc);

  IfScope beginIf(mlir::Value cond,
                  llvm::ArrayRef<const clang::ValueDecl *> carriedVars,
                  bool hasElseBranch, bool needsElseRegion, mlir::Location loc) {
    llvm::SmallVector<mlir::Type, 8> resultTypes;
    resultTypes.reserve(carriedVars.size());
    for (const clang::ValueDecl *vd : carriedVars) {
      mlir::Value incoming = ctx.valueMap.lookup(vd);
      assert(incoming &&
             "conditional carried variable must already have a bound value");
      resultTypes.push_back(incoming.getType());
    }

    auto ifOp = ctx.builder.create<simt::dialect::IfOp>(
        loc, resultTypes, cond, needsElseRegion);
    return IfScope(ctx, ifOp, carriedVars, hasElseBranch, needsElseRegion, loc);
  }

  Value emitConstantInt(int64_t value, const char *tag,
                        simt_hlsl_import::SourceLoc loc) {
    llvm::StringRef tagRef(tag ? tag : "");
    mlir::Type type = parseTypeTag(tagRef, ctx);
    if (!type) {
      ctx.fail("unsupported integer constant tag");
      return {};
    }
    auto intType = llvm::dyn_cast<mlir::IntegerType>(type);
    if (!intType) {
      ctx.fail("expected integer type tag for integer constant");
      return {};
    }
    auto attr = ctx.builder.getIntegerAttr(intType, value);
    return ctx.builder.create<mlir::arith::ConstantOp>(resolveLoc(loc, ctx),
                                                       attr);
  }
  Value emitConstantFloat(double value, const char *tag,
                          simt_hlsl_import::SourceLoc loc) {
    llvm::StringRef tagRef(tag ? tag : "");
    mlir::Type type = parseTypeTag(tagRef, ctx);
    if (!type) {
      ctx.fail("unsupported float constant tag");
      return {};
    }
    auto floatType = llvm::dyn_cast<mlir::FloatType>(type);
    if (!floatType) {
      ctx.fail("expected float type tag for float constant");
      return {};
    }
    auto attr = ctx.builder.getFloatAttr(floatType, value);
    return ctx.builder.create<mlir::arith::ConstantOp>(resolveLoc(loc, ctx),
                                                       attr);
  }
  Value emitArithmetic(simt_hlsl_import::ArithOp op, Value lhs, Value rhs,
                       simt_hlsl_import::SourceLoc loc) {
    mlir::Type type = lhs.getType();
    mlir::Location mlirLoc = resolveLoc(loc, ctx);
    if (mlir::isa<mlir::IntegerType>(type)) {
      auto intType = llvm::cast<mlir::IntegerType>(type);
      switch (op) {
      case simt_hlsl_import::ArithOp::Add:
        return ctx.builder.create<mlir::arith::AddIOp>(mlirLoc, lhs, rhs)
            .getResult();
      case simt_hlsl_import::ArithOp::Sub:
        return ctx.builder.create<mlir::arith::SubIOp>(mlirLoc, lhs, rhs)
            .getResult();
      case simt_hlsl_import::ArithOp::Mul:
        return ctx.builder.create<mlir::arith::MulIOp>(mlirLoc, lhs, rhs)
            .getResult();
      case simt_hlsl_import::ArithOp::Div:
        return ctx.builder.create<mlir::arith::DivSIOp>(mlirLoc, lhs, rhs)
            .getResult();
      case simt_hlsl_import::ArithOp::Rem:
        return ctx.builder.create<mlir::arith::RemSIOp>(mlirLoc, lhs, rhs)
            .getResult();
      case simt_hlsl_import::ArithOp::Neg: {
        auto zero = ctx.builder.create<mlir::arith::ConstantIntOp>(
            mlirLoc, 0, intType.getWidth());
        return ctx.builder.create<mlir::arith::SubIOp>(mlirLoc, zero, lhs)
            .getResult();
      }
      case simt_hlsl_import::ArithOp::BitAnd:
        return ctx.builder.create<mlir::arith::AndIOp>(mlirLoc, lhs, rhs)
            .getResult();
      case simt_hlsl_import::ArithOp::BitOr:
        return ctx.builder.create<mlir::arith::OrIOp>(mlirLoc, lhs, rhs)
            .getResult();
      case simt_hlsl_import::ArithOp::BitXor:
        return ctx.builder.create<mlir::arith::XOrIOp>(mlirLoc, lhs, rhs)
            .getResult();
      }
    }

    if (mlir::isa<mlir::FloatType>(type)) {
      switch (op) {
      case simt_hlsl_import::ArithOp::Add:
        return ctx.builder.create<mlir::arith::AddFOp>(mlirLoc, lhs, rhs)
            .getResult();
      case simt_hlsl_import::ArithOp::Sub:
        return ctx.builder.create<mlir::arith::SubFOp>(mlirLoc, lhs, rhs)
            .getResult();
      case simt_hlsl_import::ArithOp::Mul:
        return ctx.builder.create<mlir::arith::MulFOp>(mlirLoc, lhs, rhs)
            .getResult();
      case simt_hlsl_import::ArithOp::Div:
        return ctx.builder.create<mlir::arith::DivFOp>(mlirLoc, lhs, rhs)
            .getResult();
      case simt_hlsl_import::ArithOp::Rem:
        return ctx.builder.create<mlir::arith::RemFOp>(mlirLoc, lhs, rhs)
            .getResult();
      case simt_hlsl_import::ArithOp::Neg:
        return ctx.builder.create<mlir::arith::NegFOp>(mlirLoc, lhs)
            .getResult();
      case simt_hlsl_import::ArithOp::BitAnd:
      case simt_hlsl_import::ArithOp::BitOr:
      case simt_hlsl_import::ArithOp::BitXor:
        break;
      }
    }

    ctx.fail("unsupported arithmetic operation for type");
    return {};
  }
  Value emitCompare(simt_hlsl_import::CmpOp op, Value lhs, Value rhs,
                    simt_hlsl_import::SourceLoc loc) {
    mlir::Location mlirLoc = resolveLoc(loc, ctx);
    mlir::Type type = lhs.getType();
    if (mlir::isa<mlir::IntegerType>(type) || mlir::isa<mlir::IndexType>(type)) {
      mlir::arith::CmpIPredicate pred;
      switch (op) {
      case simt_hlsl_import::CmpOp::EQ:
        pred = mlir::arith::CmpIPredicate::eq;
        break;
      case simt_hlsl_import::CmpOp::NE:
        pred = mlir::arith::CmpIPredicate::ne;
        break;
      case simt_hlsl_import::CmpOp::LT:
        pred = mlir::arith::CmpIPredicate::slt;
        break;
      case simt_hlsl_import::CmpOp::LE:
        pred = mlir::arith::CmpIPredicate::sle;
        break;
      case simt_hlsl_import::CmpOp::GT:
        pred = mlir::arith::CmpIPredicate::sgt;
        break;
      case simt_hlsl_import::CmpOp::GE:
        pred = mlir::arith::CmpIPredicate::sge;
        break;
      }
      return ctx.builder.create<mlir::arith::CmpIOp>(mlirLoc, pred, lhs, rhs)
          .getResult();
    }

    if (mlir::isa<mlir::FloatType>(type)) {
      mlir::arith::CmpFPredicate pred;
      switch (op) {
      case simt_hlsl_import::CmpOp::EQ:
        pred = mlir::arith::CmpFPredicate::OEQ;
        break;
      case simt_hlsl_import::CmpOp::NE:
        pred = mlir::arith::CmpFPredicate::UNE;
        break;
      case simt_hlsl_import::CmpOp::LT:
        pred = mlir::arith::CmpFPredicate::OLT;
        break;
      case simt_hlsl_import::CmpOp::LE:
        pred = mlir::arith::CmpFPredicate::OLE;
        break;
      case simt_hlsl_import::CmpOp::GT:
        pred = mlir::arith::CmpFPredicate::OGT;
        break;
      case simt_hlsl_import::CmpOp::GE:
        pred = mlir::arith::CmpFPredicate::OGE;
        break;
      }
      return ctx.builder.create<mlir::arith::CmpFOp>(mlirLoc, pred, lhs, rhs)
          .getResult();
    }

    ctx.fail("unsupported compare operands");
    return {};
  }
  Value emitSelect(Value cond, Value trueV, Value falseV,
                   simt_hlsl_import::SourceLoc loc) {
    return ctx.builder
        .create<mlir::arith::SelectOp>(resolveLoc(loc, ctx), cond, trueV, falseV)
        .getResult();
  }

  template <typename ThenMake, typename ElseMake>
  Value emitConditional(Value cond, ThenMake &&thenBuilder,
                        ElseMake &&elseBuilder,
                        simt_hlsl_import::SourceLoc loc) {
    mlir::Location mlirLoc = resolveLoc(loc, ctx);

    mlir::Region thenRegion;
    thenRegion.emplaceBlock();
    mlir::OpBuilder thenBuilderImpl(ctx.builder.getContext());
    thenBuilderImpl.setInsertionPointToEnd(&thenRegion.front());
    LoweringContext thenCtx(thenBuilderImpl, mlirLoc, ctx.returnType,
                            ctx.errorMessage, ctx.sourceManager);
    cloneContextState(ctx, thenCtx);

    mlir::Value thenValue = thenBuilder(thenCtx);
    if (!thenValue)
      return {};
    if (thenCtx.failed) {
      ctx.failed = true;
      return {};
    }

    mlir::Region elseRegion;
    elseRegion.emplaceBlock();
    mlir::OpBuilder elseBuilderImpl(ctx.builder.getContext());
    elseBuilderImpl.setInsertionPointToEnd(&elseRegion.front());
    LoweringContext elseCtx(elseBuilderImpl, mlirLoc, ctx.returnType,
                            ctx.errorMessage, ctx.sourceManager);
    cloneContextState(ctx, elseCtx);

    mlir::Value elseValue = elseBuilder(elseCtx);
    if (!elseValue)
      return {};
    if (elseCtx.failed) {
      ctx.failed = true;
      return {};
    }

    if (thenValue.getType() != elseValue.getType()) {
      ctx.fail("conditional operator branch type mismatch");
      return {};
    }
    mlir::Type resultType = thenValue.getType();

    llvm::SmallPtrSet<const clang::ValueDecl *, 8> mutatedSet;
    mutatedSet.insert(thenCtx.mutatedVars.begin(), thenCtx.mutatedVars.end());
    mutatedSet.insert(elseCtx.mutatedVars.begin(), elseCtx.mutatedVars.end());

    llvm::SmallVector<const clang::ValueDecl *, 8> mutatedVars(
        mutatedSet.begin(), mutatedSet.end());
    llvm::sort(mutatedVars,
               [](const clang::ValueDecl *lhs, const clang::ValueDecl *rhs) {
                 return lhs < rhs;
               });

    llvm::SmallVector<mlir::Type, 8> resultTypes;
    resultTypes.push_back(resultType);
    for (const clang::ValueDecl *vd : mutatedVars) {
      auto lookupType =
          [&](LoweringContext &context) -> std::optional<mlir::Type> {
        auto it = context.valueMap.find(vd);
        if (it != context.valueMap.end())
          return it->second.getType();
        return std::nullopt;
      };
      std::optional<mlir::Type> branchType = lookupType(thenCtx);
      if (!branchType)
        branchType = lookupType(ctx);
      if (!branchType)
        branchType = lookupType(elseCtx);
      if (!branchType) {
        ctx.fail("conditional operator missing carried value");
        return {};
      }
      resultTypes.push_back(*branchType);
    }

    auto ifOp = ctx.builder.create<simt::dialect::IfOp>(
        mlirLoc, resultTypes, cond, /*withElseRegion=*/true);

    auto replaceRegionBody = [](mlir::Region &dest, mlir::Region &src) {
      if (!dest.empty())
        dest.front().erase();
      dest.takeBody(src);
      if (dest.empty())
        dest.emplaceBlock();
    };

    auto appendOperands =
        [&](LoweringContext &branchCtx,
            llvm::SmallVectorImpl<mlir::Value> &operands) -> bool {
      for (const clang::ValueDecl *vd : mutatedVars) {
        mlir::Value value;
        if (auto it = branchCtx.valueMap.find(vd);
            it != branchCtx.valueMap.end())
          value = it->second;
        else if (auto it = ctx.valueMap.find(vd); it != ctx.valueMap.end())
          value = it->second;
        else
          return false;
        operands.push_back(value);
      }
      return true;
    };

    auto buildYield = [&](mlir::Region &region, LoweringContext &branchCtx,
                          mlir::Value branchValue) -> bool {
      llvm::SmallVector<mlir::Value, 8> operands;
      operands.push_back(branchValue);
      if (!appendOperands(branchCtx, operands)) {
        ctx.fail("conditional operator missing carried value");
        return false;
      }
      mlir::OpBuilder yieldBuilder(ctx.builder.getContext());
      yieldBuilder.setInsertionPointToEnd(&region.front());
      yieldBuilder.create<simt::dialect::YieldOp>(mlirLoc, operands);
      return true;
    };

    if (!buildYield(thenRegion, thenCtx, thenValue) ||
        !buildYield(elseRegion, elseCtx, elseValue))
      return {};

    replaceRegionBody(ifOp.getThenRegion(), thenRegion);
    replaceRegionBody(ifOp.getElseRegion(), elseRegion);

    unsigned resultIndex = 1;
    for (const clang::ValueDecl *vd : mutatedVars) {
      ctx.valueMap[vd] = ifOp.getResult(resultIndex++);
      ctx.mutatedVars.insert(vd);
    }

    return ifOp.getResult(0);
  }
  template <typename RHSMake>
  Value emitShortCircuit(simt_hlsl_import::LogicalOp op, Value lhs,
                         RHSMake &&rhsBuilder,
                         simt_hlsl_import::SourceLoc loc) {
    auto *binOp = llvm::dyn_cast_or_null<clang::BinaryOperator>(loc.clangNode);
    if (!binOp) {
      ctx.fail("short-circuit requires clang::BinaryOperator source");
      return {};
    }

    mlir::Location mlirLoc = resolveLoc(loc, ctx);
    auto boolType = ctx.builder.getI1Type();
    if (lhs.getType() != boolType)
      return ctx.fail("logical operator requires boolean operands"),
             mlir::Value();

    auto propagateFailure = [&](LoweringContext &branchCtx) {
      if (branchCtx.failed) {
        ctx.failed = true;
        return true;
      }
      return false;
    };

    if (op == simt_hlsl_import::LogicalOp::And) {
      mlir::Region thenRegion;
      thenRegion.emplaceBlock();
    mlir::OpBuilder thenBuilder(ctx.builder.getContext());
    thenBuilder.setInsertionPointToEnd(&thenRegion.front());
    LoweringContext thenCtx(thenBuilder, mlirLoc, ctx.returnType,
                            ctx.errorMessage, ctx.sourceManager);
    cloneContextState(ctx, thenCtx);

      mlir::Value rhsVal = rhsBuilder(thenCtx);
      if (!rhsVal)
        return {};
      if (propagateFailure(thenCtx))
        return {};
      if (rhsVal.getType() != boolType)
        return ctx.fail("logical and requires boolean operands"),
               mlir::Value();

      llvm::SmallVector<const clang::ValueDecl *, 8> mutatedVars(
          thenCtx.mutatedVars.begin(), thenCtx.mutatedVars.end());
      llvm::sort(mutatedVars, [](const clang::ValueDecl *lhsDecl,
                                 const clang::ValueDecl *rhsDecl) {
        return lhsDecl < rhsDecl;
      });

      llvm::SmallVector<mlir::Type, 8> resultTypes;
      resultTypes.push_back(boolType);
      for (const clang::ValueDecl *vd : mutatedVars) {
        auto it = thenCtx.valueMap.find(vd);
        if (it == thenCtx.valueMap.end())
          it = ctx.valueMap.find(vd);
        if (it == ctx.valueMap.end())
          return ctx.fail("logical and missing carried value"), mlir::Value();
        resultTypes.push_back(it->second.getType());
      }

      auto ifOp = ctx.builder.create<simt::dialect::IfOp>(
          mlirLoc, resultTypes, lhs, /*withElseRegion=*/true);

      auto replaceRegionBody = [](mlir::Region &dest, mlir::Region &src) {
        if (!dest.empty())
          dest.front().erase();
        dest.takeBody(src);
        if (dest.empty())
          dest.emplaceBlock();
      };
      replaceRegionBody(ifOp.getThenRegion(), thenRegion);

      auto lookupValue = [&](LoweringContext &valueCtx,
                             const clang::ValueDecl *vd) -> mlir::Value {
        auto it = valueCtx.valueMap.find(vd);
        if (it != valueCtx.valueMap.end())
          return it->second;
        return {};
      };

      auto ensureYield = [&](mlir::Region &region,
                             llvm::ArrayRef<mlir::Value> operands) -> bool {
        auto &block = region.front();
        if (!block.empty() &&
            block.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
          if (auto yield =
                  llvm::dyn_cast<simt::dialect::YieldOp>(&block.back())) {
            yield.getOperation()->setOperands(operands);
            return true;
          }
          ctx.fail("unexpected terminator while lowering logical and");
          return false;
        }
        mlir::OpBuilder::atBlockEnd(&block).create<simt::dialect::YieldOp>(
            mlirLoc, operands);
        return true;
      };

      llvm::SmallVector<mlir::Value, 8> thenOperands;
      thenOperands.push_back(rhsVal);
      for (const clang::ValueDecl *vd : mutatedVars) {
        mlir::Value value = lookupValue(thenCtx, vd);
        if (!value)
          value = lookupValue(ctx, vd);
        if (!value)
          return ctx.fail("logical and missing carried value"), mlir::Value();
        thenOperands.push_back(value);
      }
      if (!ensureYield(ifOp.getThenRegion(), thenOperands))
        return {};

      auto &elseRegion = ifOp.getElseRegion();
      if (!elseRegion.empty())
        elseRegion.front().erase();
      elseRegion.emplaceBlock();
      auto elseBuilder = mlir::OpBuilder::atBlockEnd(&elseRegion.front());
      auto falseConst =
          elseBuilder.create<mlir::arith::ConstantIntOp>(mlirLoc, 0, 1);
      llvm::SmallVector<mlir::Value, 8> elseOperands;
      elseOperands.push_back(falseConst);
      for (const clang::ValueDecl *vd : mutatedVars) {
        mlir::Value value = lookupValue(ctx, vd);
        if (!value)
          return ctx.fail("logical and missing carried value"), mlir::Value();
        elseOperands.push_back(value);
      }
      if (!ensureYield(elseRegion, elseOperands))
        return {};

      unsigned resultIndex = 1;
      for (const clang::ValueDecl *vd : mutatedVars) {
        ctx.valueMap[vd] = ifOp.getResult(resultIndex++);
        if (auto it = thenCtx.symValueMap.find(vd); it != thenCtx.symValueMap.end())
          ctx.symValueMap[vd] = it->second;
        ctx.mutatedVars.insert(vd);
      }

      return ifOp.getResult(0);
    }

    // Logical OR
    mlir::Region elseRegion;
    elseRegion.emplaceBlock();
    mlir::OpBuilder elseBuilder(ctx.builder.getContext());
    elseBuilder.setInsertionPointToEnd(&elseRegion.front());
    LoweringContext elseCtx(elseBuilder, mlirLoc, ctx.returnType,
                            ctx.errorMessage, ctx.sourceManager);
    cloneContextState(ctx, elseCtx);

    mlir::Value rhsVal = rhsBuilder(elseCtx);
    if (!rhsVal)
      return {};
    if (propagateFailure(elseCtx))
      return {};
    if (rhsVal.getType() != boolType)
      return ctx.fail("logical or requires boolean operands"), mlir::Value();

    llvm::SmallVector<const clang::ValueDecl *, 8> mutatedVars(
        elseCtx.mutatedVars.begin(), elseCtx.mutatedVars.end());
    llvm::sort(mutatedVars, [](const clang::ValueDecl *lhsDecl,
                               const clang::ValueDecl *rhsDecl) {
      return lhsDecl < rhsDecl;
    });

    llvm::SmallVector<mlir::Type, 8> resultTypes;
    resultTypes.push_back(boolType);
    for (const clang::ValueDecl *vd : mutatedVars) {
      auto it = elseCtx.valueMap.find(vd);
      if (it == elseCtx.valueMap.end())
        it = ctx.valueMap.find(vd);
      if (it == ctx.valueMap.end())
        return ctx.fail("logical or missing carried value"), mlir::Value();
      resultTypes.push_back(it->second.getType());
    }

    auto ifOp = ctx.builder.create<simt::dialect::IfOp>(
        mlirLoc, resultTypes, lhs, /*withElseRegion=*/true);

    auto replaceRegionBody = [](mlir::Region &dest, mlir::Region &src) {
      if (!dest.empty())
        dest.front().erase();
      dest.takeBody(src);
      if (dest.empty())
        dest.emplaceBlock();
    };

    auto lookupValue = [&](LoweringContext &valueCtx,
                           const clang::ValueDecl *vd) -> mlir::Value {
      auto it = valueCtx.valueMap.find(vd);
      if (it != valueCtx.valueMap.end())
        return it->second;
      return {};
    };

    auto ensureYield = [&](mlir::Region &region,
                           llvm::ArrayRef<mlir::Value> operands) -> bool {
      auto &block = region.front();
      if (!block.empty() &&
          block.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
        if (auto yield =
                llvm::dyn_cast<simt::dialect::YieldOp>(&block.back())) {
          yield.getOperation()->setOperands(operands);
          return true;
        }
        ctx.fail("unexpected terminator while lowering logical or");
        return false;
      }
      mlir::OpBuilder::atBlockEnd(&block).create<simt::dialect::YieldOp>(
          mlirLoc, operands);
      return true;
    };

    auto &thenRegion = ifOp.getThenRegion();
    if (!thenRegion.empty())
      thenRegion.front().erase();
    thenRegion.emplaceBlock();
    auto thenBuilder = mlir::OpBuilder::atBlockEnd(&thenRegion.front());
    auto trueConst = thenBuilder.create<mlir::arith::ConstantIntOp>(mlirLoc, 1, 1);
    llvm::SmallVector<mlir::Value, 8> thenOperands;
    thenOperands.push_back(trueConst);
    for (const clang::ValueDecl *vd : mutatedVars) {
      mlir::Value value = lookupValue(ctx, vd);
      if (!value)
        return ctx.fail("logical or missing carried value"), mlir::Value();
      thenOperands.push_back(value);
    }
    if (!ensureYield(thenRegion, thenOperands))
      return {};

    replaceRegionBody(ifOp.getElseRegion(), elseRegion);
    auto &finalElseRegion = ifOp.getElseRegion();
    llvm::SmallVector<mlir::Value, 8> elseOperands;
    elseOperands.push_back(rhsVal);
    for (const clang::ValueDecl *vd : mutatedVars) {
      mlir::Value value = lookupValue(elseCtx, vd);
      if (!value)
        value = lookupValue(ctx, vd);
      if (!value)
        return ctx.fail("logical or missing carried value"), mlir::Value();
      elseOperands.push_back(value);
    }
    if (!ensureYield(finalElseRegion, elseOperands))
      return {};

    unsigned resultIndex = 1;
    for (const clang::ValueDecl *vd : mutatedVars) {
      ctx.valueMap[vd] = ifOp.getResult(resultIndex++);
      if (auto it = elseCtx.symValueMap.find(vd); it != elseCtx.symValueMap.end())
        ctx.symValueMap[vd] = it->second;
      ctx.mutatedVars.insert(vd);
    }

    return ifOp.getResult(0);
  }

  Value emitBufferLoad(Value resourceHandle, Value index,
                       const clang::ValueDecl *resourceDecl,
                       simt_hlsl_import::SourceLoc loc) {
    mlir::Location mlirLoc = resolveLoc(loc, ctx);
    auto resourceType =
        mlir::dyn_cast<simt::dialect::ResourceType>(resourceHandle.getType());
    if (!resourceType)
      return ctx.fail("buffer load requires resource handle"), mlir::Value();

    return ctx.builder
        .create<simt::dialect::BufferLoadOp>(mlirLoc, resourceHandle, index)
        .getResult();
  }

  void emitBufferStore(Value resourceHandle, Value index, Value storedValue,
                       const clang::ValueDecl *resourceDecl,
                       simt_hlsl_import::SourceLoc loc) {
    mlir::Location mlirLoc = resolveLoc(loc, ctx);
    auto resourceType =
        mlir::dyn_cast<simt::dialect::ResourceType>(resourceHandle.getType());
    if (!resourceType) {
      ctx.fail("buffer store requires resource handle");
      return;
    }
    if (storedValue.getType() != resourceType.getElementType()) {
      ctx.fail("buffer store value must match element type");
      return;
    }

    ctx.builder.create<simt::dialect::BufferStoreOp>(mlirLoc, resourceHandle,
                                                     index, storedValue);
    if (resourceDecl) {
      ctx.mutatedVars.insert(resourceDecl);
      ctx.symValueMap[resourceDecl] = makeSymValue(resourceDecl);
    }
  }

  Value emitAtomic(simt_hlsl_import::BufferAtomicOp op, Value resourceHandle,
                   Value index, Value value, Value compare,
                   const clang::ValueDecl *resourceDecl,
                   simt_hlsl_import::SourceLoc loc) {
    auto resourceType =
        mlir::dyn_cast<simt::dialect::ResourceType>(resourceHandle.getType());
    if (!resourceType)
      return ctx.fail("buffer atomic requires resource handle"), mlir::Value();

    auto elementType = resourceType.getElementType();
    auto checkElementType = [&](mlir::Value operand,
                               llvm::StringRef message) -> bool {
      if (!operand)
        return true;
      if (operand.getType() == elementType)
        return true;
      ctx.fail(message);
      return false;
    };

    auto requireValue = [&](mlir::Value operand, llvm::StringRef message)
        -> bool {
      if (operand)
        return true;
      ctx.fail(message);
      return false;
    };

    mlir::Location mlirLoc = resolveLoc(loc, ctx);
    mlir::Value oldValue;
    switch (op) {
    case simt_hlsl_import::BufferAtomicOp::Add:
      if (!requireValue(value, "InterlockedAdd requires a value operand") ||
          !checkElementType(value,
                            "InterlockedAdd value must match element type"))
        return {};
      oldValue = ctx.builder
                      .create<simt::dialect::BufferAtomicAddOp>(
                          mlirLoc, resourceHandle, index, value)
                      .getOldValue();
      break;
    case simt_hlsl_import::BufferAtomicOp::Exchange:
      if (!requireValue(value, "InterlockedExchange requires a value operand") ||
          !checkElementType(value,
                            "InterlockedExchange value must match element type"))
        return {};
      oldValue = ctx.builder
                      .create<simt::dialect::BufferAtomicExchangeOp>(
                          mlirLoc, resourceHandle, index, value)
                      .getOldValue();
      break;
    case simt_hlsl_import::BufferAtomicOp::CompareExchange:
      if (!requireValue(compare,
                        "InterlockedCompareExchange requires a compare operand") ||
          !checkElementType(compare, "InterlockedCompareExchange compare must "
                                         "match element type") ||
          !requireValue(value, "InterlockedCompareExchange requires a value "
                                "operand") ||
          !checkElementType(value, "InterlockedCompareExchange value must match "
                                  "element type"))
        return {};
      oldValue = ctx.builder
                      .create<simt::dialect::BufferAtomicCompareExchangeOp>(
                          mlirLoc, resourceHandle, index, compare, value)
                      .getOldValue();
      break;
    case simt_hlsl_import::BufferAtomicOp::Min:
      if (!requireValue(value, "InterlockedMin requires a value operand") ||
          !checkElementType(value,
                            "InterlockedMin value must match element type"))
        return {};
      oldValue = ctx.builder
                      .create<simt::dialect::BufferAtomicMinOp>(
                          mlirLoc, resourceHandle, index, value)
                      .getOldValue();
      break;
    case simt_hlsl_import::BufferAtomicOp::Max:
      if (!requireValue(value, "InterlockedMax requires a value operand") ||
          !checkElementType(value,
                            "InterlockedMax value must match element type"))
        return {};
      oldValue = ctx.builder
                      .create<simt::dialect::BufferAtomicMaxOp>(
                          mlirLoc, resourceHandle, index, value)
                      .getOldValue();
      break;
    case simt_hlsl_import::BufferAtomicOp::And:
      if (!requireValue(value, "InterlockedAnd requires a value operand") ||
          !checkElementType(value,
                            "InterlockedAnd value must match element type"))
        return {};
      oldValue = ctx.builder
                      .create<simt::dialect::BufferAtomicAndOp>(
                          mlirLoc, resourceHandle, index, value)
                      .getOldValue();
      break;
    case simt_hlsl_import::BufferAtomicOp::Or:
      if (!requireValue(value, "InterlockedOr requires a value operand") ||
          !checkElementType(value,
                            "InterlockedOr value must match element type"))
        return {};
      oldValue = ctx.builder
                      .create<simt::dialect::BufferAtomicOrOp>(
                          mlirLoc, resourceHandle, index, value)
                      .getOldValue();
      break;
    case simt_hlsl_import::BufferAtomicOp::Xor:
      if (!requireValue(value, "InterlockedXor requires a value operand") ||
          !checkElementType(value,
                            "InterlockedXor value must match element type"))
        return {};
      oldValue = ctx.builder
                      .create<simt::dialect::BufferAtomicXorOp>(
                          mlirLoc, resourceHandle, index, value)
                      .getOldValue();
      break;
    }

    if (resourceDecl)
      ctx.symValueMap[resourceDecl] = makeSymValue(resourceDecl);

    return oldValue;
  }

  Value emitWaveIntrinsic(simt_hlsl_import::WaveIntrinsic intrinsic,
                          llvm::ArrayRef<Value> operands,
                          mlir::Type resultType,
                          simt_hlsl_import::SourceLoc loc) {
    mlir::Location mlirLoc = resolveLoc(loc, ctx);
    auto fail = [&](llvm::StringRef message) {
      ctx.fail(message);
      return mlir::Value();
    };

    switch (intrinsic) {
    case simt_hlsl_import::WaveIntrinsic::ActiveAllTrue: {
      if (operands.size() != 1)
        return fail("WaveActiveAllTrue expects one operand");
      if (!resultType)
        return fail("WaveActiveAllTrue requires result type");
      return ctx.builder
          .create<simt::dialect::WaveAllOp>(mlirLoc, resultType, operands[0])
          .getResult();
    }
    case simt_hlsl_import::WaveIntrinsic::ActiveAnyTrue: {
      if (operands.size() != 1)
        return fail("WaveActiveAnyTrue expects one operand");
      if (!resultType)
        return fail("WaveActiveAnyTrue requires result type");
      return ctx.builder
          .create<simt::dialect::WaveAnyOp>(mlirLoc, resultType, operands[0])
          .getResult();
    }
    case simt_hlsl_import::WaveIntrinsic::ActiveCountBits: {
      if (operands.size() != 1)
        return fail("WaveActiveCountBits expects one operand");
      if (!resultType)
        return fail("WaveActiveCountBits requires result type");

      mlir::Value mask =
          ctx.builder
              .create<simt::dialect::WaveBallotOp>(mlirLoc,
                                                   ctx.builder.getI64Type(),
                                                   operands[0])
              .getMask();
      mlir::Value pop =
          ctx.builder.create<mlir::math::CtPopOp>(mlirLoc, mask).getResult();

      if (pop.getType() == resultType)
        return pop;

      if (auto intType = llvm::dyn_cast<mlir::IntegerType>(resultType)) {
        auto popInt = llvm::cast<mlir::IntegerType>(pop.getType());
        unsigned targetWidth = intType.getWidth();
        unsigned sourceWidth = popInt.getWidth();
        if (targetWidth == sourceWidth)
          return pop;
        if (targetWidth < sourceWidth)
          return ctx.builder
              .create<mlir::arith::TruncIOp>(mlirLoc, resultType, pop)
              .getResult();
        return ctx.builder
            .create<mlir::arith::ExtUIOp>(mlirLoc, resultType, pop)
            .getResult();
      }

      if (mlir::isa<mlir::IndexType>(resultType))
        return ctx.builder
            .create<mlir::arith::IndexCastOp>(mlirLoc, resultType, pop)
            .getResult();

      return fail("WaveActiveCountBits unsupported result type");
    }
    case simt_hlsl_import::WaveIntrinsic::GetLaneIndex: {
      if (!operands.empty())
        return fail("WaveGetLaneIndex expects no operands");
      if (!resultType)
        return fail("WaveGetLaneIndex requires result type");

      mlir::Value lane =
          ctx.builder
              .create<simt::dialect::LaneIdOp>(mlirLoc,
                                               ctx.builder.getIndexType())
              .getLane();

      if (mlir::isa<mlir::IndexType>(resultType))
        return lane;
      if (mlir::isa<mlir::IntegerType>(resultType))
        return ctx.builder
            .create<mlir::arith::IndexCastOp>(mlirLoc, resultType, lane)
            .getResult();
      return fail("WaveGetLaneIndex unsupported result type");
    }
    }

    llvm_unreachable("unknown wave intrinsic kind");
  }

  Value lookupVariable(const clang::ValueDecl *decl) {
    auto it = ctx.valueMap.find(decl);
    return it != ctx.valueMap.end() ? it->second : mlir::Value();
  }

  void bindVariable(const clang::ValueDecl *decl, Value value) {
    ctx.valueMap[decl] = value;
    ctx.symValueMap[decl] = makeSymValue(decl);
  }

  void noteMutation(const clang::ValueDecl *decl) {
    ctx.mutatedVars.insert(decl);
  }

  void emitReturn(std::optional<Value> value, simt_hlsl_import::SourceLoc loc);

  void emitBarrier(simt_hlsl_import::BarrierKind kind,
                   simt_hlsl_import::SourceLoc loc);
  void emitFence(simt_hlsl_import::BarrierKind kind, const char *memSpace,
                 simt_hlsl_import::SourceLoc loc);

  void trace(const char *message, simt_hlsl_import::SourceLoc) {
    (void)message;
  }

  LoweringContext &ctx;
};

struct AnalysisInterpreter
    : simt_hlsl_import::LoweringAlgebra<AnalysisInterpreter,
                                        simt_hlsl_import::AnalysisValue> {

  explicit AnalysisInterpreter(LoweringContext &ctx) : ctx(ctx) {}

  AnalysisInterpreter fork(LoweringContext &childCtx) {
    return AnalysisInterpreter(childCtx);
  }

  struct IfScope {
    IfScope(LoweringContext &parentCtx, simt::dialect::IfOp op,
            llvm::ArrayRef<const clang::ValueDecl *> carried, bool hasElse,
            bool needsElseRegion, mlir::Location loc)
        : parent(parentCtx), ifOp(op),
          mutated(carried.begin(), carried.end()), hasElseBranch(hasElse),
          needsElseRegion(needsElseRegion), loc(loc) {}

    LoweringContext &thenContext() {
      if (!thenCtxStorage) {
        mlir::Region &region = ifOp.getThenRegion();
        thenBuilder.emplace(parent.builder.getContext());
        thenBuilder->setInsertionPointToEnd(&region.front());
        thenCtxStorage.emplace(*thenBuilder, loc, parent.returnType,
                               parent.errorMessage, parent.sourceManager);
        cloneContextState(parent, *thenCtxStorage);
      }
      return *thenCtxStorage;
    }

    bool userHasElse() const { return hasElseBranch; }

    LoweringContext &elseContext() {
      if (!hasElseBranch)
        llvm::report_fatal_error("elseContext requested but no else branch");
      if (!elseCtxStorage) {
        mlir::Region &region = ifOp.getElseRegion();
        elseBuilder.emplace(parent.builder.getContext());
        elseBuilder->setInsertionPointToEnd(&region.front());
        elseCtxStorage.emplace(*elseBuilder, loc, parent.returnType,
                               parent.errorMessage, parent.sourceManager);
        cloneContextState(parent, *elseCtxStorage);
      }
      return *elseCtxStorage;
    }

    bool finalizeThen() {
      if (thenClosed)
        return !parent.failed;
      thenClosed = true;
      if (!thenCtxStorage || mutated.empty())
        return !parent.failed;

      llvm::SmallVector<mlir::Value, 8> operands;
      operands.reserve(mutated.size());
      for (const clang::ValueDecl *vd : mutated) {
        mlir::Value value;
        if (auto it = thenCtxStorage->valueMap.find(vd);
            it != thenCtxStorage->valueMap.end())
          value = it->second;
        else if (auto pit = parent.valueMap.find(vd);
                 pit != parent.valueMap.end())
          value = pit->second;
        if (!value) {
          parent.fail("conditional branch missing carried value");
          return false;
        }
        operands.push_back(value);
      }
      ensureYield(*thenCtxStorage, ifOp.getThenRegion(), operands, loc);
      return !parent.failed;
    }

    bool finalizeElse() {
      if (elseClosed)
        return !parent.failed;
      elseClosed = true;
      if (!elseCtxStorage || mutated.empty())
        return !parent.failed;

      llvm::SmallVector<mlir::Value, 8> operands;
      operands.reserve(mutated.size());
      for (const clang::ValueDecl *vd : mutated) {
        mlir::Value value;
        if (auto it = elseCtxStorage->valueMap.find(vd);
            it != elseCtxStorage->valueMap.end())
          value = it->second;
        else if (auto pit = parent.valueMap.find(vd);
                 pit != parent.valueMap.end())
          value = pit->second;
        if (!value) {
          parent.fail("conditional branch missing carried value");
          return false;
        }
        operands.push_back(value);
      }
      ensureYield(*elseCtxStorage, ifOp.getElseRegion(), operands, loc);
      return !parent.failed;
    }

    bool finalizeImplicitElse() {
      if (!needsElseRegion || hasElseBranch || mutated.empty())
        return true;
      llvm::SmallVector<mlir::Value, 8> operands;
      operands.reserve(mutated.size());
      for (const clang::ValueDecl *vd : mutated) {
        auto it = parent.valueMap.find(vd);
        if (it == parent.valueMap.end()) {
          parent.fail("conditional missing carried value for implicit else");
          return false;
        }
        operands.push_back(it->second);
      }
      ensureYield(parent, ifOp.getElseRegion(), operands, loc);
      return !parent.failed;
    }

    bool done() {
      if (!finalizeThen())
        return false;
      if (hasElseBranch || needsElseRegion) {
        if (hasElseBranch) {
          if (!finalizeElse())
            return false;
        } else if (!finalizeImplicitElse()) {
          return false;
        }
      }

      unsigned resultIndex = 0;
      for (const clang::ValueDecl *vd : mutated) {
        parent.valueMap[vd] = ifOp.getResult(resultIndex++);
        parent.mutatedVars.insert(vd);
        if (thenCtxStorage) {
          if (auto it = thenCtxStorage->symValueMap.find(vd);
              it != thenCtxStorage->symValueMap.end())
            parent.symValueMap[vd] = it->second;
        }
        if (elseCtxStorage) {
          if (auto it = elseCtxStorage->symValueMap.find(vd);
              it != elseCtxStorage->symValueMap.end())
            parent.symValueMap[vd] = it->second;
        }
      }

      bool thenTerminated =
          thenCtxStorage && thenCtxStorage->emittedTerminator;
      bool elseTerminated = false;
      if (hasElseBranch)
        elseTerminated = elseCtxStorage && elseCtxStorage->emittedTerminator;
      else if (!needsElseRegion)
        elseTerminated = true;

      if (thenTerminated && (!needsElseRegion || elseTerminated))
        parent.emittedTerminator = true;

      return !parent.failed;
    }

    LoweringContext &parent;
    simt::dialect::IfOp ifOp;
    llvm::SmallVector<const clang::ValueDecl *, 8> mutated;
    bool hasElseBranch = false;
    bool needsElseRegion = false;
    mlir::Location loc;

    std::optional<mlir::OpBuilder> thenBuilder;
    std::optional<LoweringContext> thenCtxStorage;
    bool thenClosed = false;

    std::optional<mlir::OpBuilder> elseBuilder;
    std::optional<LoweringContext> elseCtxStorage;
    bool elseClosed = false;
  };

  using LoopScope = LoopScopeState;

  LoopScope beginLoop(llvm::ArrayRef<const clang::ValueDecl *> carriedVars,
                      bool hasFirstIterFlag, mlir::Value firstIterInit,
                      mlir::Location loc);

  IfScope beginIf(mlir::Value cond,
                  llvm::ArrayRef<const clang::ValueDecl *> carriedVars,
                  bool hasElseBranch, bool needsElseRegion, mlir::Location loc) {
    llvm::SmallVector<mlir::Type, 8> resultTypes;
    resultTypes.reserve(carriedVars.size());
    for (const clang::ValueDecl *vd : carriedVars) {
      mlir::Value incoming = ctx.valueMap.lookup(vd);
      assert(incoming &&
             "conditional carried variable must already have a bound value");
      resultTypes.push_back(incoming.getType());
    }

    auto ifOp = ctx.builder.create<simt::dialect::IfOp>(
        loc, resultTypes, cond, needsElseRegion);
    return IfScope(ctx, ifOp, carriedVars, hasElseBranch, needsElseRegion, loc);
  }

  Value emitConstantInt(int64_t value, const char *tag,
                        simt_hlsl_import::SourceLoc) {
    llvm::StringRef tagRef(tag ? tag : "");
    mlir::Type type = parseTypeTag(tagRef, ctx);
    if (!type)
      type = ctx.builder.getI32Type();

    Value result;
    result.setTypeHint(type);

    SymValue sym = makeSymValueForType(type);
    sym.isConst = true;
    result.setSym(sym);
    result.setConstantInt(value);
    return result;
  }
  Value emitConstantFloat(double value, const char *tag,
                          simt_hlsl_import::SourceLoc) {
    llvm::StringRef tagRef(tag ? tag : "");
    mlir::Type type = parseTypeTag(tagRef, ctx);
    if (!type)
      type = ctx.builder.getF32Type();

    Value result;
    result.setTypeHint(type);

    SymValue sym = makeSymValueForType(type);
    sym.isConst = true;
    result.setSym(sym);
    result.setConstantFloat(value);
    return result;
  }
  Value emitArithmetic(simt_hlsl_import::ArithOp op, Value lhs, Value rhs,
                       simt_hlsl_import::SourceLoc loc) {
    mlir::Location mlirLoc = resolveLoc(loc, ctx);

    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(lhs.getType())) {
      switch (op) {
      case simt_hlsl_import::ArithOp::Add:
        return ctx.builder.create<mlir::arith::AddIOp>(mlirLoc, lhs, rhs)
            .getResult();
      case simt_hlsl_import::ArithOp::Sub:
        return ctx.builder.create<mlir::arith::SubIOp>(mlirLoc, lhs, rhs)
            .getResult();
      case simt_hlsl_import::ArithOp::Mul:
        return ctx.builder.create<mlir::arith::MulIOp>(mlirLoc, lhs, rhs)
            .getResult();
      case simt_hlsl_import::ArithOp::Div:
        return ctx.builder.create<mlir::arith::DivSIOp>(mlirLoc, lhs, rhs)
            .getResult();
      case simt_hlsl_import::ArithOp::Rem:
        return ctx.builder.create<mlir::arith::RemSIOp>(mlirLoc, lhs, rhs)
            .getResult();
      case simt_hlsl_import::ArithOp::Neg: {
        auto zero = ctx.builder.create<mlir::arith::ConstantIntOp>(
            mlirLoc, 0, intType.getWidth());
        return ctx.builder
            .create<mlir::arith::SubIOp>(mlirLoc, zero.getResult(), lhs)
            .getResult();
      }
      case simt_hlsl_import::ArithOp::BitAnd:
        return ctx.builder.create<mlir::arith::AndIOp>(mlirLoc, lhs, rhs)
            .getResult();
      case simt_hlsl_import::ArithOp::BitOr:
        return ctx.builder.create<mlir::arith::OrIOp>(mlirLoc, lhs, rhs)
            .getResult();
      case simt_hlsl_import::ArithOp::BitXor:
        return ctx.builder.create<mlir::arith::XOrIOp>(mlirLoc, lhs, rhs)
            .getResult();
      }
    }

    if (mlir::isa<mlir::FloatType>(lhs.getType())) {
      switch (op) {
      case simt_hlsl_import::ArithOp::Add:
        return ctx.builder.create<mlir::arith::AddFOp>(mlirLoc, lhs, rhs)
            .getResult();
      case simt_hlsl_import::ArithOp::Sub:
        return ctx.builder.create<mlir::arith::SubFOp>(mlirLoc, lhs, rhs)
            .getResult();
      case simt_hlsl_import::ArithOp::Mul:
        return ctx.builder.create<mlir::arith::MulFOp>(mlirLoc, lhs, rhs)
            .getResult();
      case simt_hlsl_import::ArithOp::Div:
        return ctx.builder.create<mlir::arith::DivFOp>(mlirLoc, lhs, rhs)
            .getResult();
      case simt_hlsl_import::ArithOp::Rem:
        return ctx.builder.create<mlir::arith::RemFOp>(mlirLoc, lhs, rhs)
            .getResult();
      case simt_hlsl_import::ArithOp::Neg:
        return ctx.builder.create<mlir::arith::NegFOp>(mlirLoc, lhs)
            .getResult();
      case simt_hlsl_import::ArithOp::BitAnd:
      case simt_hlsl_import::ArithOp::BitOr:
      case simt_hlsl_import::ArithOp::BitXor:
        break;
      }
    }

    ctx.fail("unsupported arithmetic operation for type");
    return {};
  }
  Value emitCompare(simt_hlsl_import::CmpOp op, Value lhs, Value rhs,
                    simt_hlsl_import::SourceLoc loc) {
    mlir::Location mlirLoc = resolveLoc(loc, ctx);
    mlir::Type type = lhs.getType();

    if (mlir::isa<mlir::IntegerType>(type) || mlir::isa<mlir::IndexType>(type)) {
      mlir::arith::CmpIPredicate pred;
      switch (op) {
      case simt_hlsl_import::CmpOp::EQ:
        pred = mlir::arith::CmpIPredicate::eq;
        break;
      case simt_hlsl_import::CmpOp::NE:
        pred = mlir::arith::CmpIPredicate::ne;
        break;
      case simt_hlsl_import::CmpOp::LT:
        pred = mlir::arith::CmpIPredicate::slt;
        break;
      case simt_hlsl_import::CmpOp::LE:
        pred = mlir::arith::CmpIPredicate::sle;
        break;
      case simt_hlsl_import::CmpOp::GT:
        pred = mlir::arith::CmpIPredicate::sgt;
        break;
      case simt_hlsl_import::CmpOp::GE:
        pred = mlir::arith::CmpIPredicate::sge;
        break;
      }
      return ctx.builder.create<mlir::arith::CmpIOp>(mlirLoc, pred, lhs, rhs)
          .getResult();
    }

    if (mlir::isa<mlir::FloatType>(type)) {
      mlir::arith::CmpFPredicate pred;
      switch (op) {
      case simt_hlsl_import::CmpOp::EQ:
        pred = mlir::arith::CmpFPredicate::OEQ;
        break;
      case simt_hlsl_import::CmpOp::NE:
        pred = mlir::arith::CmpFPredicate::UNE;
        break;
      case simt_hlsl_import::CmpOp::LT:
        pred = mlir::arith::CmpFPredicate::OLT;
        break;
      case simt_hlsl_import::CmpOp::LE:
        pred = mlir::arith::CmpFPredicate::OLE;
        break;
      case simt_hlsl_import::CmpOp::GT:
        pred = mlir::arith::CmpFPredicate::OGT;
        break;
      case simt_hlsl_import::CmpOp::GE:
        pred = mlir::arith::CmpFPredicate::OGE;
        break;
      }
      return ctx.builder.create<mlir::arith::CmpFOp>(mlirLoc, pred, lhs, rhs)
          .getResult();
    }

    ctx.fail("unsupported compare operands");
    return {};
  }
  Value emitSelect(Value cond, Value trueV, Value falseV,
                   simt_hlsl_import::SourceLoc loc) {
    return ctx.builder
        .create<mlir::arith::SelectOp>(resolveLoc(loc, ctx), cond, trueV, falseV)
        .getResult();
  }

  template <typename RHSMake>
  Value emitShortCircuit(simt_hlsl_import::LogicalOp op, Value lhs,
                         RHSMake &&rhsBuilder,
                         simt_hlsl_import::SourceLoc loc) {
    mlir::Location mlirLoc = resolveLoc(loc, ctx);
    mlir::Type boolType = ctx.builder.getI1Type();

    if (lhs.getType() != boolType)
      return ctx.fail("logical operator requires boolean operands"),
             mlir::Value();

    auto propagateFailure = [&](LoweringContext &branchCtx) {
      if (branchCtx.failed) {
        ctx.failed = true;
        return true;
      }
      return false;
    };

    if (op == simt_hlsl_import::LogicalOp::And) {
      mlir::Region thenRegion;
      thenRegion.emplaceBlock();
    mlir::OpBuilder thenBuilder(ctx.builder.getContext());
    thenBuilder.setInsertionPointToEnd(&thenRegion.front());
    LoweringContext thenCtx(thenBuilder, mlirLoc, ctx.returnType,
                            ctx.errorMessage, ctx.sourceManager);
    cloneContextState(ctx, thenCtx);

      mlir::Value rhsVal = rhsBuilder(thenCtx);
      if (!rhsVal)
        return {};
      if (propagateFailure(thenCtx))
        return {};
      if (rhsVal.getType() != boolType)
        return ctx.fail("logical and requires boolean operands"),
               mlir::Value();

      llvm::SmallVector<const clang::ValueDecl *, 8> mutatedVars(
          thenCtx.mutatedVars.begin(), thenCtx.mutatedVars.end());
      llvm::sort(mutatedVars, [](const clang::ValueDecl *lhsDecl,
                                 const clang::ValueDecl *rhsDecl) {
        return lhsDecl < rhsDecl;
      });

      llvm::SmallVector<mlir::Type, 8> resultTypes;
      resultTypes.push_back(boolType);
      for (const clang::ValueDecl *vd : mutatedVars) {
        auto it = thenCtx.valueMap.find(vd);
        if (it == thenCtx.valueMap.end())
          it = ctx.valueMap.find(vd);
        if (it == ctx.valueMap.end())
          return ctx.fail("logical and missing carried value"), mlir::Value();
        resultTypes.push_back(it->second.getType());
      }

      auto ifOp = ctx.builder.create<simt::dialect::IfOp>(
          mlirLoc, resultTypes, lhs, /*withElseRegion=*/true);

      auto replaceRegionBody = [](mlir::Region &dest, mlir::Region &src) {
        if (!dest.empty())
          dest.front().erase();
        dest.takeBody(src);
        if (dest.empty())
          dest.emplaceBlock();
      };
      replaceRegionBody(ifOp.getThenRegion(), thenRegion);

      auto lookupValue = [&](LoweringContext &valueCtx,
                             const clang::ValueDecl *vd) -> mlir::Value {
        auto it = valueCtx.valueMap.find(vd);
        if (it != valueCtx.valueMap.end())
          return it->second;
        return {};
      };

      auto ensureYield = [&](mlir::Region &region,
                             llvm::ArrayRef<mlir::Value> operands) -> bool {
        auto &block = region.front();
        if (!block.empty() &&
            block.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
          if (auto yield =
                  llvm::dyn_cast<simt::dialect::YieldOp>(&block.back())) {
            yield.getOperation()->setOperands(operands);
            return true;
          }
          ctx.fail("unexpected terminator while lowering logical and");
          return false;
        }
        mlir::OpBuilder::atBlockEnd(&block).create<simt::dialect::YieldOp>(
            mlirLoc, operands);
        return true;
      };

      llvm::SmallVector<mlir::Value, 8> thenOperands;
      thenOperands.push_back(rhsVal);
      for (const clang::ValueDecl *vd : mutatedVars) {
        mlir::Value value = lookupValue(thenCtx, vd);
        if (!value)
          value = lookupValue(ctx, vd);
        if (!value)
          return ctx.fail("logical and missing carried value"), mlir::Value();
        thenOperands.push_back(value);
      }
      if (!ensureYield(ifOp.getThenRegion(), thenOperands))
        return {};

      auto &elseRegion = ifOp.getElseRegion();
      if (!elseRegion.empty())
        elseRegion.front().erase();
      elseRegion.emplaceBlock();
      auto elseBuilder = mlir::OpBuilder::atBlockEnd(&elseRegion.front());
      auto falseConst =
          elseBuilder.create<mlir::arith::ConstantIntOp>(mlirLoc, 0, 1);
      llvm::SmallVector<mlir::Value, 8> elseOperands;
      elseOperands.push_back(falseConst);
      for (const clang::ValueDecl *vd : mutatedVars) {
        mlir::Value value = lookupValue(ctx, vd);
        if (!value)
          return ctx.fail("logical and missing carried value"), mlir::Value();
        elseOperands.push_back(value);
      }
      if (!ensureYield(elseRegion, elseOperands))
        return {};

      unsigned resultIndex = 1;
      for (const clang::ValueDecl *vd : mutatedVars) {
        ctx.valueMap[vd] = ifOp.getResult(resultIndex++);
        if (auto it = thenCtx.symValueMap.find(vd); it != thenCtx.symValueMap.end())
          ctx.symValueMap[vd] = it->second;
        ctx.mutatedVars.insert(vd);
      }

      return ifOp.getResult(0);
    }

    // Logical OR
    mlir::Region elseRegion;
    elseRegion.emplaceBlock();
    mlir::OpBuilder elseBuilder(ctx.builder.getContext());
    elseBuilder.setInsertionPointToEnd(&elseRegion.front());
    LoweringContext elseCtx(elseBuilder, mlirLoc, ctx.returnType,
                            ctx.errorMessage, ctx.sourceManager);
    cloneContextState(ctx, elseCtx);

    mlir::Value rhsVal = rhsBuilder(elseCtx);
    if (!rhsVal)
      return {};
    if (propagateFailure(elseCtx))
      return {};
    if (rhsVal.getType() != boolType)
      return ctx.fail("logical or requires boolean operands"), mlir::Value();

    llvm::SmallVector<const clang::ValueDecl *, 8> mutatedVars(
        elseCtx.mutatedVars.begin(), elseCtx.mutatedVars.end());
    llvm::sort(mutatedVars, [](const clang::ValueDecl *lhsDecl,
                               const clang::ValueDecl *rhsDecl) {
      return lhsDecl < rhsDecl;
    });

    llvm::SmallVector<mlir::Type, 8> resultTypes;
    resultTypes.push_back(boolType);
    for (const clang::ValueDecl *vd : mutatedVars) {
      auto it = elseCtx.valueMap.find(vd);
      if (it == elseCtx.valueMap.end())
        it = ctx.valueMap.find(vd);
      if (it == ctx.valueMap.end())
        return ctx.fail("logical or missing carried value"), mlir::Value();
      resultTypes.push_back(it->second.getType());
    }

    auto ifOp = ctx.builder.create<simt::dialect::IfOp>(
        mlirLoc, resultTypes, lhs, /*withElseRegion=*/true);

    auto replaceRegionBody = [](mlir::Region &dest, mlir::Region &src) {
      if (!dest.empty())
        dest.front().erase();
      dest.takeBody(src);
      if (dest.empty())
        dest.emplaceBlock();
    };

    auto lookupValue = [&](LoweringContext &valueCtx,
                           const clang::ValueDecl *vd) -> mlir::Value {
      auto it = valueCtx.valueMap.find(vd);
      if (it != valueCtx.valueMap.end())
        return it->second;
      return {};
    };

    auto ensureYield = [&](mlir::Region &region,
                           llvm::ArrayRef<mlir::Value> operands) -> bool {
      auto &block = region.front();
      if (!block.empty() &&
          block.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
        if (auto yield =
                llvm::dyn_cast<simt::dialect::YieldOp>(&block.back())) {
          yield.getOperation()->setOperands(operands);
          return true;
        }
        ctx.fail("unexpected terminator while lowering logical or");
        return false;
      }
      mlir::OpBuilder::atBlockEnd(&block).create<simt::dialect::YieldOp>(
          mlirLoc, operands);
      return true;
    };

    auto &thenRegion = ifOp.getThenRegion();
    if (!thenRegion.empty())
      thenRegion.front().erase();
    thenRegion.emplaceBlock();
    auto thenBuilder = mlir::OpBuilder::atBlockEnd(&thenRegion.front());
    auto trueConst = thenBuilder.create<mlir::arith::ConstantIntOp>(mlirLoc, 1, 1);
    llvm::SmallVector<mlir::Value, 8> thenOperands;
    thenOperands.push_back(trueConst);
    for (const clang::ValueDecl *vd : mutatedVars) {
      mlir::Value value = lookupValue(ctx, vd);
      if (!value)
        return ctx.fail("logical or missing carried value"), mlir::Value();
      thenOperands.push_back(value);
    }
    if (!ensureYield(thenRegion, thenOperands))
      return {};

    replaceRegionBody(ifOp.getElseRegion(), elseRegion);
    auto &finalElseRegion = ifOp.getElseRegion();
    llvm::SmallVector<mlir::Value, 8> elseOperands;
    elseOperands.push_back(rhsVal);
    for (const clang::ValueDecl *vd : mutatedVars) {
      mlir::Value value = lookupValue(elseCtx, vd);
      if (!value)
        value = lookupValue(ctx, vd);
      if (!value)
        return ctx.fail("logical or missing carried value"), mlir::Value();
      elseOperands.push_back(value);
    }
    if (!ensureYield(finalElseRegion, elseOperands))
      return {};

    unsigned resultIndex = 1;
    for (const clang::ValueDecl *vd : mutatedVars) {
      ctx.valueMap[vd] = ifOp.getResult(resultIndex++);
      if (auto it = elseCtx.symValueMap.find(vd); it != elseCtx.symValueMap.end())
        ctx.symValueMap[vd] = it->second;
      ctx.mutatedVars.insert(vd);
    }

    return ifOp.getResult(0);
  }

  Value emitBufferLoad(Value resourceHandle, Value index,
                       const clang::ValueDecl *decl,
                       simt_hlsl_import::SourceLoc loc) {
    mlir::Location mlirLoc = resolveLoc(loc, ctx);
    mlir::Value load =
        ctx.builder
            .create<simt::dialect::BufferLoadOp>(mlirLoc, resourceHandle,
                                                 index)
            .getResult();
    simt_hlsl_import::AnalysisValue result =
        simt_hlsl_import::AnalysisValue::fromValue(load);
    if (decl) {
      if (auto symIt = ctx.symValueMap.find(decl);
          symIt != ctx.symValueMap.end())
        result.setSym(symIt->second);
    }
    return result;
  }

  void emitBufferStore(Value, Value, Value, const clang::ValueDecl *,
                       simt_hlsl_import::SourceLoc) {}

  Value emitAtomic(simt_hlsl_import::BufferAtomicOp, Value, Value, Value, Value,
                   const clang::ValueDecl *resourceDecl,
                   simt_hlsl_import::SourceLoc) {
    if (resourceDecl) {
      ctx.mutatedVars.insert(resourceDecl);
      ctx.symValueMap[resourceDecl] = makeSymValue(resourceDecl);
    }
    return {};
  }

  Value emitWaveIntrinsic(simt_hlsl_import::WaveIntrinsic,
                          llvm::ArrayRef<Value>, mlir::Type resultType,
                          simt_hlsl_import::SourceLoc) {
    Value result;
    result.setTypeHint(resultType);
    result.setSym(makeSymValueForType(resultType));
    return result;
  }

  Value lookupVariable(const clang::ValueDecl *decl) {
    auto it = ctx.valueMap.find(decl);
    simt_hlsl_import::AnalysisValue result;
    if (it != ctx.valueMap.end())
      result.setValue(it->second);
    if (auto symIt = ctx.symValueMap.find(decl);
        symIt != ctx.symValueMap.end())
      result.setSym(symIt->second);
    return result;
  }

  void bindVariable(const clang::ValueDecl *decl, Value value) {
    if (mlir::Value mlirVal = value.getValueOrNull())
      ctx.valueMap[decl] = mlirVal;
    if (const SymValue *sym = value.getSym())
      ctx.symValueMap[decl] = *sym;
    else
      ctx.symValueMap[decl] = makeSymValue(decl);
  }

  void noteMutation(const clang::ValueDecl *decl) {
    ctx.mutatedVars.insert(decl);
  }

  void emitReturn(std::optional<Value>, simt_hlsl_import::SourceLoc) {
    ctx.emittedTerminator = true;
  }

  void emitBarrier(simt_hlsl_import::BarrierKind,
                   simt_hlsl_import::SourceLoc) {}
  void emitFence(simt_hlsl_import::BarrierKind, const char *,
                 simt_hlsl_import::SourceLoc) {}

  void trace(const char *, simt_hlsl_import::SourceLoc) {}

  LoweringContext &ctx;
};

} // namespace

void EmitInterpreter::emitReturn(std::optional<mlir::Value> value,
                                 simt_hlsl_import::SourceLoc loc) {
  bool expectsValue = static_cast<bool>(ctx.returnType) &&
                      !mlir::isa<mlir::NoneType>(ctx.returnType);
  mlir::Location mlirLoc = resolveLoc(loc, ctx);

  if (value) {
    if (!expectsValue) {
      ctx.fail("unexpected return value in void function");
      return;
    }
    if (value->getType() != ctx.returnType) {
      ctx.fail("return type mismatch");
      return;
    }
    ctx.builder.create<mlir::func::ReturnOp>(mlirLoc, *value);
  } else {
    if (expectsValue) {
      ctx.fail("missing return value");
      return;
    }
    ctx.builder.create<mlir::func::ReturnOp>(mlirLoc);
  }
  ctx.emittedTerminator = true;
}

void EmitInterpreter::emitBarrier(simt_hlsl_import::BarrierKind kind,
                                  simt_hlsl_import::SourceLoc loc) {
  auto *context = ctx.builder.getContext();
  auto scopeAttr =
      simt::dialect::ScopeAttr::get(context, simt::dialect::Scope::Workgroup);
  auto memSemAttr = simt::dialect::MemorySemanticsAttr::get(
      context, simt::dialect::MemorySemantics::AcqRel);
  (void)kind;
  ctx.builder.create<simt::dialect::BarrierOp>(resolveLoc(loc, ctx), scopeAttr,
                                               memSemAttr);
}

static std::optional<simt::dialect::MemorySpace>
decodeMemorySpace(llvm::StringRef memSpace) {
  if (memSpace == "Shared")
    return simt::dialect::MemorySpace::Shared;
  if (memSpace == "Global")
    return simt::dialect::MemorySpace::Global;
  if (memSpace == "Generic")
    return simt::dialect::MemorySpace::Generic;
  return std::nullopt;
}

void EmitInterpreter::emitFence(simt_hlsl_import::BarrierKind kind,
                                const char *memSpace,
                                simt_hlsl_import::SourceLoc loc) {
  auto *context = ctx.builder.getContext();
  auto scopeAttr =
      simt::dialect::ScopeAttr::get(context, simt::dialect::Scope::Workgroup);
  auto memSemAttr = simt::dialect::MemorySemanticsAttr::get(
      context, simt::dialect::MemorySemantics::AcqRel);
  (void)kind;
  auto decoded = decodeMemorySpace(memSpace);
  if (!decoded) {
    ctx.fail("unrecognised memory space for barrier");
    return;
  }
  ctx.builder.create<simt::dialect::FenceOp>(
      resolveLoc(loc, ctx), scopeAttr, memSemAttr,
      simt::dialect::MemorySpaceAttr::get(context, *decoded));
}


static mlir::Type convertType(const clang::QualType &qt,
                              mlir::OpBuilder &builder) {
  clang::LangAS addressSpace = qt.getAddressSpace();
  const clang::Type *type = qt.getCanonicalType().getTypePtrOrNull();
  if (!type)
    return {};

  auto desugar = [](const clang::Type *ty) -> const clang::Type * {
    while (auto *elab = llvm::dyn_cast<clang::ElaboratedType>(ty))
      ty = elab->getNamedType().getTypePtr();
    return ty->getUnqualifiedDesugaredType();
  };
  type = desugar(type);

  auto *tmplSpec = [&]() -> const clang::ClassTemplateSpecializationDecl * {
    if (const auto *recordType = llvm::dyn_cast<clang::RecordType>(type)) {
      const auto *cxxRecord =
          llvm::dyn_cast<clang::CXXRecordDecl>(recordType->getDecl());
      if (const auto *spec =
              llvm::dyn_cast_or_null<clang::ClassTemplateSpecializationDecl>(
                  cxxRecord))
        return spec;
    }
    if (const auto *specType =
            llvm::dyn_cast<clang::TemplateSpecializationType>(type)) {
      if (auto *templDecl = specType->getTemplateName().getAsTemplateDecl())
        if (const auto *record = llvm::dyn_cast<clang::CXXRecordDecl>(
                templDecl->getTemplatedDecl())) {
          if (const auto *spec =
                  llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(
                      record))
            return spec;
        }
    }
    return nullptr;
  }();

  if (!tmplSpec) {
    if (const auto *recordType = llvm::dyn_cast<clang::RecordType>(type)) {
      const auto *cxxRecord =
          llvm::dyn_cast<clang::CXXRecordDecl>(recordType->getDecl());
      if (cxxRecord) {
        llvm::SmallPtrSet<const clang::CXXRecordDecl *, 8> visited;
        std::function<const clang::ClassTemplateSpecializationDecl *(
            const clang::CXXRecordDecl *)>
            findResourceBase = [&](const clang::CXXRecordDecl *record)
                -> const clang::ClassTemplateSpecializationDecl * {
          if (!record || !visited.insert(record).second)
            return nullptr;
          if (const auto *spec =
                  llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(
                      record))
            return spec;
          for (const clang::CXXBaseSpecifier &base : record->bases()) {
            const clang::Type *baseType =
                base.getType().getCanonicalType().getTypePtrOrNull();
            if (!baseType)
              continue;
            baseType = baseType->getUnqualifiedDesugaredType();
            if (const auto *baseRecord =
                    baseType->getAsCXXRecordDecl())
              if (const auto *spec = findResourceBase(baseRecord))
                return spec;
          }
          return nullptr;
        };

        tmplSpec = findResourceBase(cxxRecord);
      }
    }
  }

  if (tmplSpec) {
    auto name = tmplSpec->getName();
    if ((name == "Buffer" || name == "RWBuffer") &&
        tmplSpec->getTemplateArgs().size() >= 1) {
      const auto &arg = tmplSpec->getTemplateArgs()[0];
      if (arg.getKind() == clang::TemplateArgument::Type) {
        mlir::Type elementType = convertType(arg.getAsType(), builder);
        if (!elementType)
          return {};
        auto memorySpace = simt::dialect::MemorySpace::Global;
        return simt::dialect::ResourceType::get(builder.getContext(),
                                                memorySpace, elementType);
      }
    }
  }

  if (const auto *recordType = llvm::dyn_cast<clang::RecordType>(type)) {
    const auto *recordDecl = recordType->getDecl();
    llvm::StringRef recordName = recordDecl->getName();
    if (recordName == "ByteAddressBuffer" || recordName == "RWByteAddressBuffer") {
      mlir::Type elementType = builder.getIntegerType(32);
      auto memorySpace = simt::dialect::MemorySpace::Global;
      return simt::dialect::ResourceType::get(builder.getContext(), memorySpace,
                                              elementType);
    }

    if (const auto *cxxRecord =
            llvm::dyn_cast<clang::CXXRecordDecl>(recordDecl)) {
      for (const auto *annot :
           cxxRecord->specific_attrs<clang::AnnotateAttr>()) {
        llvm::StringRef text = annot->getAnnotation();
        if (!text.consume_front("simt.resource:"))
          continue;

        llvm::StringRef memSpaceStr;
        llvm::StringRef elementStr;
        std::tie(memSpaceStr, elementStr) = text.split(':');
        if (elementStr.empty())
          continue;

        auto memorySpace = simt::dialect::MemorySpace::Global;
        if (memSpaceStr == "Shared")
          memorySpace = simt::dialect::MemorySpace::Shared;
        else if (memSpaceStr == "Private")
          memorySpace = simt::dialect::MemorySpace::Private;
        else if (memSpaceStr == "Generic")
          memorySpace = simt::dialect::MemorySpace::Generic;
        else if (!memSpaceStr.empty() && memSpaceStr != "Global")
          continue;

        mlir::Type elementType;
        if (elementStr == "i8")
          elementType = builder.getIntegerType(8);
        else if (elementStr == "i16")
          elementType = builder.getIntegerType(16);
        else if (elementStr == "i32")
          elementType = builder.getIntegerType(32);
        else if (elementStr == "i64")
          elementType = builder.getIntegerType(64);
        else if (elementStr == "f16")
          elementType = builder.getF16Type();
        else if (elementStr == "f32")
          elementType = builder.getF32Type();
        else if (elementStr == "f64")
          elementType = builder.getF64Type();
        else
          continue;

        return simt::dialect::ResourceType::get(builder.getContext(), memorySpace,
                                                elementType);
      }
    }
  }

  if (const auto *vectorType = llvm::dyn_cast<clang::VectorType>(type)) {
    mlir::Type elementType = convertType(vectorType->getElementType(), builder);
    if (!elementType)
      return {};

    auto numElements = static_cast<int64_t>(vectorType->getNumElements());
    return mlir::VectorType::get({numElements}, elementType);
  }

  if (const auto *arrayType = llvm::dyn_cast<clang::ArrayType>(type)) {
    if (addressSpace == clang::LangAS::hlsl_groupshared) {
      mlir::Type elementType =
          convertType(arrayType->getElementType(), builder);
      if (!elementType)
        return {};
      return simt::dialect::ResourceType::get(builder.getContext(),
                                              simt::dialect::MemorySpace::Shared,
                                              elementType);
    }
  }

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

static mlir::Value buildZeroValue(LoweringContext &ctx, mlir::Type type);

struct BufferAccessInfo {
  mlir::Value resource;
  mlir::Value index;
  simt::dialect::ResourceType resourceType;
  const clang::ValueDecl *decl = nullptr;
};

static std::optional<BufferAccessInfo>
getBufferAccessInfo(const clang::Expr *baseExpr, const clang::Expr *indexExpr,
                    LoweringContext &ctx);

static std::optional<BufferAccessInfo>
lowerBufferAccessOperands(const clang::CXXOperatorCallExpr *opCall,
                          LoweringContext &ctx);

template <typename Interp>
static std::optional<std::tuple<typename Interp::Value, typename Interp::Value,
                                simt::dialect::ResourceType,
                                const clang::ValueDecl *>>
lowerBufferAccessInterp(const clang::Expr *baseExpr,
                        const clang::Expr *indexExpr, LoweringContext &ctx,
                        Interp &interp) {
  auto resource = lowerExprInterp(baseExpr->IgnoreParenImpCasts(), ctx, interp);
  if (!resource)
    return std::nullopt;

  auto index = lowerExprInterp(indexExpr->IgnoreParenImpCasts(), ctx, interp);
  if (!index)
    return std::nullopt;

  mlir::Type indexType = getValueType(index);
  if (!mlir::isa<mlir::IntegerType>(indexType)) {
    ctx.fail("buffer subscript index must be integer");
    return std::nullopt;
  }

  auto resourceType =
      mlir::dyn_cast<simt::dialect::ResourceType>(getValueType(resource));
  if (!resourceType) {
    ctx.fail("subscript base must be a buffer resource");
    return std::nullopt;
  }

  const clang::ValueDecl *decl = nullptr;
  if (const auto *declRef =
          llvm::dyn_cast<clang::DeclRefExpr>(
              baseExpr->IgnoreParenImpCasts()))
    decl = declRef->getDecl();

  return std::make_optional(std::make_tuple(resource, index, resourceType, decl));
}

static mlir::Value lowerAssignment(const clang::BinaryOperator *binOp,
                                   LoweringContext &ctx);

static std::optional<mlir::Value>
lowerAtomicMemberCall(const clang::CXXMemberCallExpr *call,
                      LoweringContext &ctx);

static mlir::Location getLocation(const clang::Stmt *stmt,
                                  LoweringContext &ctx);
static mlir::Value lowerExprLegacy(const clang::Expr *expr, LoweringContext &ctx);

template <typename Interp>
static typename Interp::Value lowerExprInterp(const clang::Expr *expr,
                                              LoweringContext &ctx,
                                              Interp &interp) {
  using ValueT = typename Interp::Value;

  if (!expr)
    return ValueT();

  mlir::Type type = convertType(expr->getType(), ctx.builder);
  if (!type) {
    ctx.fail("unsupported expression type");
    return ValueT();
  }

  mlir::Location loc = getLocation(expr, ctx);

  if (const auto *intLit = llvm::dyn_cast<clang::IntegerLiteral>(expr)) {
    if (!mlir::isa<mlir::IntegerType>(type)) {
      ctx.fail("integer literal expects integer type");
      return ValueT();
    }
    auto intType = mlir::cast<mlir::IntegerType>(type);
    std::string tag = buildIntegerTag(intType);
    simt_hlsl_import::SourceLoc src{expr, loc};
    auto result = interp.emitConstantInt(intLit->getValue().getSExtValue(),
                                         tag.c_str(), src);
    if constexpr (!std::is_same_v<typename Interp::Value, mlir::Value>) {
      auto attr = ctx.builder.getIntegerAttr(intType, intLit->getValue());
      mlir::Value mlirConst =
          ctx.builder.create<mlir::arith::ConstantOp>(loc, attr);
      result.setValue(mlirConst);
      if (!result.hasSymValue())
        result.setSym(makeSymValueForType(type));
    }
    return result;
  }

  if (const auto *floatLit = llvm::dyn_cast<clang::FloatingLiteral>(expr)) {
    if (!mlir::isa<mlir::FloatType>(type)) {
      ctx.fail("floating literal expects floating type");
      return ValueT();
    }
    auto floatType = mlir::cast<mlir::FloatType>(type);
    std::string tag = buildFloatTag(floatType);
    simt_hlsl_import::SourceLoc src{expr, loc};
    llvm::APFloat apValue = floatLit->getValue();
    double value = apValue.convertToDouble();
    auto result = interp.emitConstantFloat(value, tag.c_str(), src);
    if constexpr (!std::is_same_v<typename Interp::Value, mlir::Value>) {
      auto attr = ctx.builder.getFloatAttr(floatType, floatLit->getValue());
      mlir::Value mlirConst =
          ctx.builder.create<mlir::arith::ConstantOp>(loc, attr);
      result.setValue(mlirConst);
      if (!result.hasSymValue())
        result.setSym(makeSymValueForType(type));
    }
    return result;
  }

  if (const auto *paren = llvm::dyn_cast<clang::ParenExpr>(expr))
    return lowerExprInterp(paren->getSubExpr(), ctx, interp);

  if (const auto *implicitCast = llvm::dyn_cast<clang::ImplicitCastExpr>(expr))
    return lowerExprInterp(implicitCast->getSubExpr(), ctx, interp);

  if (const auto *constExpr = llvm::dyn_cast<clang::ConstantExpr>(expr))
    return lowerExprInterp(constExpr->getSubExpr(), ctx, interp);

  if (const auto *declRef = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
    const clang::ValueDecl *vd = declRef->getDecl();
    auto it = ctx.valueMap.find(vd);
    if (it == ctx.valueMap.end()) {
      ctx.fail("reference to unknown value");
      return ValueT();
    }
    auto result = wrapMlirValue(interp, it->second);
    if constexpr (!std::is_same_v<typename Interp::Value, mlir::Value>) {
      auto symIt = ctx.symValueMap.find(vd);
      if (symIt != ctx.symValueMap.end())
        result.setSym(symIt->second);
      if (!result.hasTypeHint())
        result.setTypeHint(it->second.getType());
    }
    return result;
  }

  if (const auto *opCall = llvm::dyn_cast<clang::CXXOperatorCallExpr>(expr)) {
    if (opCall->getOperator() == clang::OO_Subscript &&
        opCall->getNumArgs() >= 2) {
      auto resource = lowerExprInterp(opCall->getArg(0)->IgnoreParenImpCasts(),
                                      ctx, interp);
      if (!resource)
        return ValueT();

      auto index = lowerExprInterp(opCall->getArg(1)->IgnoreParenImpCasts(), ctx,
                                   interp);
      if (!index)
        return ValueT();

      mlir::Type indexType = getValueType(index);
      if (!mlir::isa<mlir::IntegerType>(indexType)) {
        ctx.fail("buffer subscript index must be integer");
        return ValueT();
      }

      auto resourceType =
          mlir::dyn_cast<simt::dialect::ResourceType>(getValueType(resource));
      if (!resourceType) {
        ctx.fail("subscript base must be a buffer resource");
        return ValueT();
      }

      const clang::ValueDecl *decl = nullptr;
      if (const auto *declRef =
              llvm::dyn_cast<clang::DeclRefExpr>(
                  opCall->getArg(0)->IgnoreParenImpCasts()))
        decl = declRef->getDecl();

      return interp.emitBufferLoad(resource, index, decl, {expr, loc});
    }
  }

  if (const auto *unOp = llvm::dyn_cast<clang::UnaryOperator>(expr)) {
    auto operand = lowerExprInterp(unOp->getSubExpr(), ctx, interp);
    if (!operand)
      return typename Interp::Value();

    simt_hlsl_import::SourceLoc src{unOp, loc};

    switch (unOp->getOpcode()) {
    case clang::UnaryOperatorKind::UO_Plus:
      return operand;
    case clang::UnaryOperatorKind::UO_Minus: {
      if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
        std::string tag = buildFloatTag(floatType);
        auto zero = interp.emitConstantFloat(0.0, tag.c_str(), src);
        return interp.emitArithmetic(simt_hlsl_import::ArithOp::Sub, zero,
                                     operand, src);
      }
      if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
        std::string tag = buildIntegerTag(intType);
        auto zero = interp.emitConstantInt(0, tag.c_str(), src);
        return interp.emitArithmetic(simt_hlsl_import::ArithOp::Sub, zero,
                                     operand, src);
      }
      break;
    }
    default:
      break;
    }
  }

  if (const auto *binOp = llvm::dyn_cast<clang::BinaryOperator>(expr)) {
    auto getOperand = [&](const clang::Expr *subExpr) -> typename Interp::Value {
      return lowerExprInterp(subExpr, ctx, interp);
    };

    simt_hlsl_import::SourceLoc src{binOp, loc};

    auto requireOperands = [&](typename Interp::Value &lhs,
                               typename Interp::Value &rhs) -> bool {
      if (!lhs || ctx.failed)
        return false;
      rhs = getOperand(binOp->getRHS());
      if (!rhs || ctx.failed)
        return false;
      return true;
    };

    switch (binOp->getOpcode()) {
    case clang::BinaryOperatorKind::BO_Add:
    case clang::BinaryOperatorKind::BO_Sub:
    case clang::BinaryOperatorKind::BO_Mul:
    case clang::BinaryOperatorKind::BO_Div:
    case clang::BinaryOperatorKind::BO_Rem: {
      auto lhs = getOperand(binOp->getLHS());
      typename Interp::Value rhs;
      if (!requireOperands(lhs, rhs))
        return typename Interp::Value();

      simt_hlsl_import::ArithOp arithOp;
      switch (binOp->getOpcode()) {
      case clang::BinaryOperatorKind::BO_Add:
        arithOp = simt_hlsl_import::ArithOp::Add;
        break;
      case clang::BinaryOperatorKind::BO_Sub:
        arithOp = simt_hlsl_import::ArithOp::Sub;
        break;
      case clang::BinaryOperatorKind::BO_Mul:
        arithOp = simt_hlsl_import::ArithOp::Mul;
        break;
      case clang::BinaryOperatorKind::BO_Div:
        arithOp = simt_hlsl_import::ArithOp::Div;
        break;
      case clang::BinaryOperatorKind::BO_Rem:
        arithOp = simt_hlsl_import::ArithOp::Rem;
        break;
      default:
        llvm_unreachable("unexpected opcode");
      }

      return interp.emitArithmetic(arithOp, lhs, rhs, src);
    }
    case clang::BinaryOperatorKind::BO_EQ:
    case clang::BinaryOperatorKind::BO_NE:
    case clang::BinaryOperatorKind::BO_LT:
    case clang::BinaryOperatorKind::BO_LE:
    case clang::BinaryOperatorKind::BO_GT:
    case clang::BinaryOperatorKind::BO_GE: {
      auto lhs = getOperand(binOp->getLHS());
      auto rhs = getOperand(binOp->getRHS());
      if (!lhs || !rhs)
        return typename Interp::Value();

      simt_hlsl_import::CmpOp cmpOp;
      switch (binOp->getOpcode()) {
      case clang::BinaryOperatorKind::BO_EQ:
        cmpOp = simt_hlsl_import::CmpOp::EQ;
        break;
      case clang::BinaryOperatorKind::BO_NE:
        cmpOp = simt_hlsl_import::CmpOp::NE;
        break;
      case clang::BinaryOperatorKind::BO_LT:
        cmpOp = simt_hlsl_import::CmpOp::LT;
        break;
      case clang::BinaryOperatorKind::BO_LE:
        cmpOp = simt_hlsl_import::CmpOp::LE;
        break;
      case clang::BinaryOperatorKind::BO_GT:
        cmpOp = simt_hlsl_import::CmpOp::GT;
        break;
      case clang::BinaryOperatorKind::BO_GE:
        cmpOp = simt_hlsl_import::CmpOp::GE;
        break;
      default:
        llvm_unreachable("unexpected comparison opcode");
      }

      return interp.emitCompare(cmpOp, lhs, rhs, src);
    }
    case clang::BinaryOperatorKind::BO_LAnd:
    case clang::BinaryOperatorKind::BO_LOr: {
      auto lhs = getOperand(binOp->getLHS());
      if (!lhs)
        return typename Interp::Value();

      mlir::Type boolType = ctx.builder.getI1Type();
      if (getValueType(lhs) != boolType) {
        ctx.fail("logical operator requires boolean operands");
        return typename Interp::Value();
      }

      auto rhsBuilder = [&](LoweringContext &branchCtx) -> typename Interp::Value {
        auto childInterp = interp.fork(branchCtx);
        return lowerExprInterp(binOp->getRHS(), branchCtx, childInterp);
      };

      auto logicalOp = binOp->getOpcode() == clang::BinaryOperatorKind::BO_LAnd
                           ? simt_hlsl_import::LogicalOp::And
                           : simt_hlsl_import::LogicalOp::Or;
      return interp.emitShortCircuit(logicalOp, lhs, rhsBuilder, src);
    }
    default:
      break;
    }
  }

  if (const auto *arraySub = llvm::dyn_cast<clang::ArraySubscriptExpr>(expr)) {
    auto resource = lowerExprInterp(arraySub->getBase()->IgnoreParenImpCasts(),
                                    ctx, interp);
    if (!resource)
      return typename Interp::Value();

    auto index = lowerExprInterp(arraySub->getIdx()->IgnoreParenImpCasts(), ctx,
                                 interp);
    if (!index)
      return typename Interp::Value();

    mlir::Type indexType = getValueType(index);
    if (!mlir::isa<mlir::IntegerType>(indexType)) {
      ctx.fail("buffer subscript index must be integer");
      return typename Interp::Value();
    }

    auto resourceType =
        mlir::dyn_cast<simt::dialect::ResourceType>(getValueType(resource));
    if (!resourceType) {
      ctx.fail("subscript base must be a buffer resource");
      return typename Interp::Value();
    }

    const clang::ValueDecl *decl = nullptr;
    if (const auto *declRef =
            llvm::dyn_cast<clang::DeclRefExpr>(
                arraySub->getBase()->IgnoreParenImpCasts()))
      decl = declRef->getDecl();

    return interp.emitBufferLoad(resource, index, decl, {expr, loc});
  }

  if (const auto *condOp = llvm::dyn_cast<clang::ConditionalOperator>(expr)) {
    auto condValue = lowerExprInterp(condOp->getCond(), ctx, interp);
    if (!condValue)
      return typename Interp::Value();

    if (getValueType(condValue) != ctx.builder.getI1Type()) {
      ctx.fail("conditional operator requires boolean condition");
      return typename Interp::Value();
    }

    auto makeBranchBuilder = [&](const clang::Expr *branchExpr) {
      return [&, branchExpr](LoweringContext &branchCtx)
                 -> typename Interp::Value {
        auto branchInterp = interp.fork(branchCtx);
        auto branchValue = lowerExprInterp(branchExpr, branchCtx, branchInterp);
        if (!branchValue)
          return typename Interp::Value();
        if (getValueType(branchValue) != type) {
          branchCtx.fail("conditional operator branch type mismatch");
          return typename Interp::Value();
        }
        return branchValue;
      };
    };

    auto thenBuilder = makeBranchBuilder(condOp->getTrueExpr());
    auto elseBuilder = makeBranchBuilder(condOp->getFalseExpr());

    simt_hlsl_import::SourceLoc src{condOp, loc};
    return interp.emitConditional(condValue, thenBuilder, elseBuilder, src);
  }

  mlir::Value value = lowerExprLegacy(expr, ctx);
  if (!value)
    return typename Interp::Value();
  return wrapMlirValue(interp, value);
}

static mlir::Value lowerExpr(const clang::Expr *expr, LoweringContext &ctx) {
  if (isEmitContext(ctx))
    return lowerExprLegacy(expr, ctx);
  AnalysisInterpreter interp(ctx);
  auto result = lowerExprInterp(expr, ctx, interp);
  return result.getValueOrNull();
}

static std::optional<mlir::Value>
lowerAtomicCall(const clang::CallExpr *call, LoweringContext &ctx);

static std::optional<mlir::Value>
lowerWaveIntrinsicCall(const clang::CallExpr *call, LoweringContext &ctx);

template <typename Interp>
static std::optional<typename Interp::Value>
lowerWaveIntrinsicCallInterp(const clang::CallExpr *call, LoweringContext &ctx,
                             Interp &interp) {
  using ValueT = typename Interp::Value;

  if (!call)
    return std::nullopt;

  const auto *callee = call->getDirectCallee();
  if (!callee)
    return std::nullopt;

  llvm::StringRef name = callee->getName();
  mlir::Location loc = getLocation(call, ctx);

  mlir::Type resultType = convertType(call->getType(), ctx.builder);
  if (!resultType)
    return std::optional<ValueT>(ValueT());

  auto buildOperands = [&](llvm::ArrayRef<unsigned> indices,
                           llvm::SmallVectorImpl<ValueT> &operands)
      -> bool {
    operands.clear();
    operands.reserve(indices.size());
    for (unsigned idx : indices) {
      if (idx >= call->getNumArgs()) {
        ctx.fail("wave intrinsic argument index out of range");
        return false;
      }
      auto operand = lowerExprInterp(call->getArg(idx), ctx, interp);
      if (!operand)
        return false;
      operands.push_back(std::move(operand));
    }
    return true;
  };

  auto dispatch = [&](simt_hlsl_import::WaveIntrinsic intrinsic,
                      llvm::ArrayRef<unsigned> operandIndices)
      -> std::optional<ValueT> {
    llvm::SmallVector<ValueT, 2> operands;
    if (!buildOperands(operandIndices, operands))
      return std::optional<ValueT>(ValueT());
    ValueT value =
        interp.emitWaveIntrinsic(intrinsic, operands, resultType, {call, loc});
    if (!value && ctx.failed)
      return std::optional<ValueT>(ValueT());
    return value;
  };

  if (name == "WaveActiveAllTrue") {
    if (call->getNumArgs() != 1) {
      ctx.fail("WaveActiveAllTrue expects one argument");
      return std::optional<ValueT>(ValueT());
    }
    return dispatch(simt_hlsl_import::WaveIntrinsic::ActiveAllTrue, {0});
  }
  if (name == "WaveActiveAnyTrue") {
    if (call->getNumArgs() != 1) {
      ctx.fail("WaveActiveAnyTrue expects one argument");
      return std::optional<ValueT>(ValueT());
    }
    return dispatch(simt_hlsl_import::WaveIntrinsic::ActiveAnyTrue, {0});
  }
  if (name == "WaveActiveCountBits") {
    if (call->getNumArgs() != 1) {
      ctx.fail("WaveActiveCountBits expects one argument");
      return std::optional<ValueT>(ValueT());
    }
    return dispatch(simt_hlsl_import::WaveIntrinsic::ActiveCountBits, {0});
  }
  if (name == "WaveGetLaneIndex") {
    if (call->getNumArgs() != 0) {
      ctx.fail("WaveGetLaneIndex expects no arguments");
      return std::optional<ValueT>(ValueT());
    }
    return dispatch(simt_hlsl_import::WaveIntrinsic::GetLaneIndex, {});
  }

  return std::nullopt;
}

static std::optional<BufferAccessInfo>
getBufferAccessInfoFromLValue(const clang::Expr *expr,
                              LoweringContext &ctx) {
  if (!expr)
    return std::nullopt;

  expr = expr->IgnoreParenImpCasts();

  if (const auto *subscript =
          llvm::dyn_cast<clang::CXXOperatorCallExpr>(expr)) {
    if (subscript->getOperator() == clang::OO_Subscript)
      return lowerBufferAccessOperands(subscript, ctx);
  }

  if (const auto *arraySub = llvm::dyn_cast<clang::ArraySubscriptExpr>(expr))
    return getBufferAccessInfo(arraySub->getBase(), arraySub->getIdx(), ctx);

  return std::nullopt;
}

static std::optional<mlir::Value>
lowerAtomicCall(const clang::CallExpr *call, LoweringContext &ctx) {
  if (!call)
    return std::nullopt;

  const auto *callee = call->getDirectCallee();
  if (!callee)
    return std::nullopt;

  auto kind =
      llvm::StringSwitch<std::optional<simt_hlsl_import::BufferAtomicOp>>(
          callee->getName())
          .Case("InterlockedAdd", simt_hlsl_import::BufferAtomicOp::Add)
          .Case("InterlockedExchange",
                 simt_hlsl_import::BufferAtomicOp::Exchange)
          .Case("InterlockedCompareExchange",
                 simt_hlsl_import::BufferAtomicOp::CompareExchange)
          .Case("InterlockedMin", simt_hlsl_import::BufferAtomicOp::Min)
          .Case("InterlockedMax", simt_hlsl_import::BufferAtomicOp::Max)
          .Case("InterlockedAnd", simt_hlsl_import::BufferAtomicOp::And)
          .Case("InterlockedOr", simt_hlsl_import::BufferAtomicOp::Or)
          .Case("InterlockedXor", simt_hlsl_import::BufferAtomicOp::Xor)
          .Default(std::nullopt);
  if (!kind)
    return std::nullopt;

  unsigned numArgs = call->getNumArgs();
  unsigned valueOperandCount =
      *kind == simt_hlsl_import::BufferAtomicOp::CompareExchange ? 2U : 1U;
  unsigned baseArgCount = 1 + valueOperandCount;
  if (numArgs != baseArgCount && numArgs != baseArgCount + 1)
    return ctx.fail("unexpected argument count for atomic call"),
           std::optional<mlir::Value>(mlir::Value());

  const clang::Expr *destArg = call->getArg(0);
  if (!destArg)
    return ctx.fail("atomic call missing destination argument"),
           std::optional<mlir::Value>(mlir::Value());

  destArg = destArg->IgnoreParenImpCasts();
  if (const auto *outExpr =
          llvm::dyn_cast<clang::HLSLOutArgExpr>(destArg))
    destArg = outExpr->getArgLValue()->IgnoreParenImpCasts();

  auto infoOpt = getBufferAccessInfoFromLValue(destArg, ctx);
  if (!infoOpt)
    return ctx.fail("atomic destination must be a buffer element"),
           std::optional<mlir::Value>(mlir::Value());

  BufferAccessInfo info = *infoOpt;
  mlir::Location loc = getLocation(call, ctx);

  unsigned argIndex = 1;

  mlir::Value compareValue;
  mlir::Value valueValue;
  if (*kind == simt_hlsl_import::BufferAtomicOp::CompareExchange) {
    compareValue = lowerExpr(call->getArg(argIndex++), ctx);
    if (!compareValue)
      return mlir::Value();
    valueValue = lowerExpr(call->getArg(argIndex++), ctx);
    if (!valueValue)
      return mlir::Value();
  } else {
    valueValue = lowerExpr(call->getArg(argIndex++), ctx);
    if (!valueValue)
      return mlir::Value();
  }

  const clang::Expr *outArg = nullptr;
  if (numArgs == baseArgCount + 1)
    outArg = call->getArg(argIndex++);

  if (isEmitContext(ctx)) {
    EmitInterpreter interp(ctx);
    mlir::Value oldValue = interp.emitAtomic(
        *kind, info.resource, info.index, valueValue, compareValue, info.decl,
        {call, loc});
    if (!oldValue && ctx.failed)
      return mlir::Value();

    if (info.decl)
      ctx.mutatedVars.insert(info.decl);

    if (outArg) {
      const clang::Expr *stripped = outArg->IgnoreParenImpCasts();
      if (const auto *outExpr =
              llvm::dyn_cast<clang::HLSLOutArgExpr>(stripped))
        stripped = outExpr->getArgLValue()->IgnoreParenImpCasts();
      const clang::ValueDecl *outDecl = nullptr;
      if (const auto *declRef =
              llvm::dyn_cast<clang::DeclRefExpr>(stripped))
        outDecl = declRef->getDecl();
      if (!outDecl)
        return ctx.fail("atomic original value argument must reference a "
                        "variable"),
               std::optional<mlir::Value>(mlir::Value());
      ctx.valueMap[outDecl] = oldValue;
      ctx.symValueMap[outDecl] = makeSymValue(outDecl);
      ctx.mutatedVars.insert(outDecl);
    }

    return mlir::Value();
  }

  AnalysisInterpreter interp(ctx);
  interp.emitAtomic(*kind, info.resource, info.index, valueValue, compareValue,
                    info.decl, {call, loc});

  if (info.decl)
    ctx.mutatedVars.insert(info.decl);

  if (outArg) {
    const clang::Expr *stripped = outArg->IgnoreParenImpCasts();
    if (const auto *outExpr =
            llvm::dyn_cast<clang::HLSLOutArgExpr>(stripped))
      stripped = outExpr->getArgLValue()->IgnoreParenImpCasts();
    const clang::ValueDecl *outDecl = nullptr;
    if (const auto *declRef =
            llvm::dyn_cast<clang::DeclRefExpr>(stripped))
      outDecl = declRef->getDecl();
    if (!outDecl)
      return ctx.fail("atomic original value argument must reference a "
                      "variable"),
             std::optional<mlir::Value>(mlir::Value());
    ctx.symValueMap[outDecl] = makeSymValue(outDecl);
    ctx.mutatedVars.insert(outDecl);
  }

  return mlir::Value();
}

static std::optional<mlir::Value>
lowerWaveIntrinsicCall(const clang::CallExpr *call, LoweringContext &ctx) {
  if (!call)
    return std::nullopt;

  if (isEmitContext(ctx)) {
    EmitInterpreter interp(ctx);
    auto result = lowerWaveIntrinsicCallInterp(call, ctx, interp);
    if (!result)
      return std::nullopt;
    return *result;
  }

  AnalysisInterpreter interp(ctx);
  auto result = lowerWaveIntrinsicCallInterp(call, ctx, interp);
  if (!result)
    return std::nullopt;
  return result->getValueOrNull();
}

template <typename Interpreter>
static bool lowerBarrierWithInterpreter(const clang::CallExpr *call,
                                        LoweringContext &ctx,
                                        Interpreter &interp) {
  const auto *callee = call->getDirectCallee();
  if (!callee)
    return false;

  llvm::StringRef name = callee->getName();
  bool emitGroupSync = false;
  const char *memSpace = nullptr;

  if (name == "GroupMemoryBarrier")
    memSpace = "Shared";
  else if (name == "GroupMemoryBarrierWithGroupSync") {
    memSpace = "Shared";
    emitGroupSync = true;
  } else if (name == "DeviceMemoryBarrier")
    memSpace = "Global";
  else if (name == "DeviceMemoryBarrierWithGroupSync") {
    memSpace = "Global";
    emitGroupSync = true;
  } else if (name == "AllMemoryBarrier")
    memSpace = "Generic";
  else if (name == "AllMemoryBarrierWithGroupSync") {
    memSpace = "Generic";
    emitGroupSync = true;
  } else {
    return false;
  }

  if (call->getNumArgs() != 0)
    return ctx.fail("memory barrier utilities do not take arguments"), false;

  simt_hlsl_import::SourceLoc src{call, getLocation(call, ctx)};

  interp.emitFence(simt_hlsl_import::BarrierKind::Workgroup, memSpace, src);
  if (emitGroupSync)
    interp.emitBarrier(simt_hlsl_import::BarrierKind::Workgroup, src);
  return true;
}

template <typename Interpreter>
static bool lowerBarrierUtilityCall(const clang::CallExpr *call,
                                    LoweringContext &ctx,
                                    Interpreter &interp) {
  return lowerBarrierWithInterpreter(call, ctx, interp);
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
  if (!stmt || !ctx.sourceManager)
    return ctx.defaultLoc;

  const clang::SourceManager &sm = *ctx.sourceManager;
  clang::SourceLocation loc = stmt->getBeginLoc();
  if (loc.isInvalid())
    loc = stmt->getEndLoc();
  if (loc.isInvalid())
    return ctx.defaultLoc;

  loc = sm.getExpansionLoc(loc);
  clang::PresumedLoc presumed = sm.getPresumedLoc(loc);
  if (!presumed.isValid())
    return ctx.defaultLoc;

  mlir::MLIRContext *mlirCtx = ctx.builder.getContext();
  mlir::StringAttr fileAttr =
      mlir::StringAttr::get(mlirCtx, presumed.getFilename());
  return mlir::FileLineColLoc::get(fileAttr, presumed.getLine(),
                                   presumed.getColumn());
}

static mlir::Value lowerExprLegacy(const clang::Expr *expr,
                                   LoweringContext &ctx) {

  if (!expr)
    return {};

  mlir::Type type = convertType(expr->getType(), ctx.builder);
  if (!type)
    return ctx.fail("unsupported expression type"), mlir::Value();

  mlir::Location loc = getLocation(expr, ctx);

  if (const auto *intLit = llvm::dyn_cast<clang::IntegerLiteral>(expr)) {
    if (!mlir::isa<mlir::IntegerType>(type))
      return ctx.fail("integer literal expects integer type"), mlir::Value();
    if (isEmitContext(ctx)) {
      EmitInterpreter interp(ctx);
      auto intType = mlir::cast<mlir::IntegerType>(type);
      std::string tag = buildIntegerTag(intType);
      simt_hlsl_import::SourceLoc src{expr, loc};
      return interp.emitConstantInt(intLit->getValue().getSExtValue(),
                                    tag.c_str(), src);
    }
    auto attr = ctx.builder.getIntegerAttr(mlir::cast<mlir::IntegerType>(type),
                                           intLit->getValue());
    return ctx.builder.create<mlir::arith::ConstantOp>(loc, attr);
  }

  if (const auto *paren = llvm::dyn_cast<clang::ParenExpr>(expr))
    return lowerExprLegacy(paren->getSubExpr(), ctx);

  if (const auto *implicitCast = llvm::dyn_cast<clang::ImplicitCastExpr>(expr))
    return lowerExprLegacy(implicitCast->getSubExpr(), ctx);

  if (const auto *constExpr = llvm::dyn_cast<clang::ConstantExpr>(expr))
    return lowerExprLegacy(constExpr->getSubExpr(), ctx);

  if (const auto *floatLit = llvm::dyn_cast<clang::FloatingLiteral>(expr)) {
    if (!mlir::isa<mlir::FloatType>(type))
      return ctx.fail("floating literal expects floating type"), mlir::Value();
    if (isEmitContext(ctx)) {
      EmitInterpreter interp(ctx);
      auto floatType = mlir::cast<mlir::FloatType>(type);
      std::string tag = buildFloatTag(floatType);
      simt_hlsl_import::SourceLoc src{expr, loc};
      llvm::APFloat apValue = floatLit->getValue();
      double value = apValue.convertToDouble();
      return interp.emitConstantFloat(value, tag.c_str(), src);
    }
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

  if (const auto *opCall = llvm::dyn_cast<clang::CXXOperatorCallExpr>(expr)) {
    if (opCall->getOperator() == clang::OO_Subscript) {
      auto infoOpt = lowerBufferAccessOperands(opCall, ctx);
      if (!infoOpt)
        return {};
      const auto &info = *infoOpt;
      if (isEmitContext(ctx)) {
        EmitInterpreter interp(ctx);
        return interp.emitBufferLoad(info.resource, info.index, info.decl,
                                     {expr, loc});
      }
      AnalysisInterpreter interp(ctx);
      return interp.emitBufferLoad(info.resource, info.index, info.decl,
                                   {expr, loc});
    }
  }

  if (const auto *memberCall = llvm::dyn_cast<clang::CXXMemberCallExpr>(expr)) {
    if (auto lowered = lowerAtomicMemberCall(memberCall, ctx))
      return *lowered;
  }

  if (const auto *callExpr = llvm::dyn_cast<clang::CallExpr>(expr)) {
    if (auto lowered = lowerAtomicCall(callExpr, ctx))
      return *lowered;
    if (auto lowered = lowerWaveIntrinsicCall(callExpr, ctx))
      return *lowered;
  }

  if (const auto *arraySub = llvm::dyn_cast<clang::ArraySubscriptExpr>(expr)) {
    auto infoOpt = getBufferAccessInfo(arraySub->getBase(), arraySub->getIdx(),
                                       ctx);
    if (!infoOpt)
      return {};
    const auto &info = *infoOpt;
    if (isEmitContext(ctx)) {
      EmitInterpreter interp(ctx);
      return interp.emitBufferLoad(info.resource, info.index, info.decl,
                                   {expr, loc});
    }
    AnalysisInterpreter interp(ctx);
    return interp.emitBufferLoad(info.resource, info.index, info.decl,
                                 {expr, loc});
  }

  if (const auto *condOp = llvm::dyn_cast<clang::ConditionalOperator>(expr)) {
    mlir::Value condValue = lowerExprLegacy(condOp->getCond(), ctx);
    if (!condValue)
      return {};
    if (condValue.getType() != ctx.builder.getI1Type())
      return ctx.fail("conditional operator requires boolean condition"),
             mlir::Value();

    auto thenBuilder = [&](LoweringContext &branchCtx) -> mlir::Value {
      mlir::Value value = lowerExprLegacy(condOp->getTrueExpr(), branchCtx);
      if (!value)
        return mlir::Value();
      if (value.getType() != type) {
        branchCtx.fail("conditional operator branch type mismatch");
        return mlir::Value();
      }
      return value;
    };

    auto elseBuilder = [&](LoweringContext &branchCtx) -> mlir::Value {
      mlir::Value value = lowerExprLegacy(condOp->getFalseExpr(), branchCtx);
      if (!value)
        return mlir::Value();
      if (value.getType() != type) {
        branchCtx.fail("conditional operator branch type mismatch");
        return mlir::Value();
      }
      return value;
    };

    simt_hlsl_import::SourceLoc src{condOp, loc};
    if (isEmitContext(ctx)) {
      EmitInterpreter interp(ctx);
      return interp.emitConditional(condValue, thenBuilder, elseBuilder, src);
    }
    AnalysisInterpreter interp(ctx);
    return interp.emitConditional(condValue, thenBuilder, elseBuilder, src);
  }

  if (const auto *vecElem = llvm::dyn_cast<clang::ExtVectorElementExpr>(expr)) {
    if (vecElem->isArrow())
      return ctx.fail("pointer-based vector swizzles are unsupported"),
             mlir::Value();

    mlir::Value base = lowerExprLegacy(vecElem->getBase(), ctx);
    if (!base)
      return {};

    auto baseVecType = mlir::dyn_cast<mlir::VectorType>(base.getType());
    if (!baseVecType || baseVecType.getRank() != 1)
      return ctx.fail("vector element access requires 1-D vector operand"),
             mlir::Value();

    llvm::SmallVector<uint32_t, 4> elementIndices32;
    vecElem->getEncodedElementAccess(elementIndices32);
    llvm::SmallVector<int64_t, 4> elements;
    elements.reserve(elementIndices32.size());
    for (uint32_t idx : elementIndices32) {
      if (idx >= baseVecType.getShape()[0])
        return ctx.fail("vector element index out of range"), mlir::Value();
      elements.push_back(static_cast<int64_t>(idx));
    }
    if (elements.empty())
      return ctx.fail("vector element access with no components"),
             mlir::Value();

    if (elements.size() == 1) {
      llvm::SmallVector<int64_t, 1> position = {elements[0]};
      return ctx.builder.create<mlir::vector::ExtractOp>(loc, base, position)
          .getResult();
    }

    mlir::Type elementType = baseVecType.getElementType();
    auto resultType = mlir::VectorType::get(
        {static_cast<int64_t>(elements.size())}, elementType);
    mlir::Value result = buildZeroValue(ctx, resultType);
    if (!result)
      return {};

    for (auto [outIdx, elementIndex] : llvm::enumerate(elements)) {
      llvm::SmallVector<int64_t, 1> extractPos = {elementIndex};
      mlir::Value component =
          ctx.builder.create<mlir::vector::ExtractOp>(loc, base, extractPos)
              .getResult();
      llvm::SmallVector<int64_t, 1> insertPos = {static_cast<int64_t>(outIdx)};
      result =
          ctx.builder
              .create<mlir::vector::InsertOp>(loc, component, result, insertPos)
              .getResult();
    }

    return result;
  }

  if (const auto *unOp = llvm::dyn_cast<clang::UnaryOperator>(expr)) {
    mlir::Value operand = lowerExprLegacy(unOp->getSubExpr(), ctx);
    if (!operand)
      return {};

    auto makeIntegerConstant = [&](int64_t value,
                                   mlir::IntegerType type) -> mlir::Value {
      return ctx.builder
          .create<mlir::arith::ConstantIntOp>(loc, value, type.getWidth())
          .getResult();
    };
    auto makeFloatConstant = [&](double value,
                                 mlir::FloatType type) -> mlir::Value {
      auto attr = ctx.builder.getFloatAttr(type, value);
      return ctx.builder.create<mlir::arith::ConstantOp>(loc, attr).getResult();
    };

    auto getMutableDeclRef =
        [&](const clang::Expr *expr) -> const clang::ValueDecl * {
      const clang::Expr *stripped = expr->IgnoreParenImpCasts();
      if (const auto *ref = llvm::dyn_cast<clang::DeclRefExpr>(stripped))
        return ref->getDecl();
      return nullptr;
    };

    switch (unOp->getOpcode()) {
    case clang::UnaryOperatorKind::UO_Plus:
      return operand;
    case clang::UnaryOperatorKind::UO_Minus: {
      if (isEmitContext(ctx)) {
        EmitInterpreter interp(ctx);
        simt_hlsl_import::SourceLoc src{unOp, loc};
        if (auto floatType = mlir::dyn_cast<mlir::FloatType>(operand.getType())) {
          std::string tag = buildFloatTag(floatType);
          mlir::Value zero = interp.emitConstantFloat(0.0, tag.c_str(), src);
          return interp.emitArithmetic(simt_hlsl_import::ArithOp::Sub, zero,
                                       operand, src);
        }
        if (auto intType = mlir::dyn_cast<mlir::IntegerType>(operand.getType())) {
          std::string tag = buildIntegerTag(intType);
          mlir::Value zero =
              interp.emitConstantInt(0, tag.c_str(), src);
          return interp.emitArithmetic(simt_hlsl_import::ArithOp::Sub, zero,
                                       operand, src);
        }
      }
      if (auto floatType = mlir::dyn_cast<mlir::FloatType>(operand.getType()))
        return ctx.builder.create<mlir::arith::NegFOp>(loc, operand)
            .getResult();
      if (auto intType = mlir::dyn_cast<mlir::IntegerType>(operand.getType())) {
        mlir::Value zero = makeIntegerConstant(0, intType);
        return ctx.builder.create<mlir::arith::SubIOp>(loc, zero, operand)
            .getResult();
      }
      return ctx.fail("unary minus requires numeric operand"), mlir::Value();
    }
    case clang::UnaryOperatorKind::UO_LNot: {
      if (isEmitContext(ctx)) {
        EmitInterpreter interp(ctx);
        simt_hlsl_import::SourceLoc src{unOp, loc};
        if (auto intType = mlir::dyn_cast<mlir::IntegerType>(operand.getType())) {
          std::string tag = buildIntegerTag(intType);
          mlir::Value zero = interp.emitConstantInt(0, tag.c_str(), src);
          return interp.emitCompare(simt_hlsl_import::CmpOp::EQ, operand, zero,
                                    src);
        }
        if (auto floatType = mlir::dyn_cast<mlir::FloatType>(operand.getType())) {
          std::string tag = buildFloatTag(floatType);
          mlir::Value zero =
              interp.emitConstantFloat(0.0, tag.c_str(), src);
          return interp.emitCompare(simt_hlsl_import::CmpOp::EQ, operand, zero,
                                    src);
        }
      }
      if (auto intType = mlir::dyn_cast<mlir::IntegerType>(operand.getType())) {
        mlir::Value zero = makeIntegerConstant(0, intType);
        return ctx.builder
            .create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq,
                                         operand, zero)
            .getResult();
      }
      if (auto floatType = mlir::dyn_cast<mlir::FloatType>(operand.getType())) {
        mlir::Value zero = makeFloatConstant(0.0, floatType);
        return ctx.builder
            .create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OEQ,
                                         operand, zero)
            .getResult();
      }
      return ctx.fail("logical not requires scalar operand"), mlir::Value();
    }
    case clang::UnaryOperatorKind::UO_Not: {
      if (isEmitContext(ctx)) {
        EmitInterpreter interp(ctx);
        if (auto intType = mlir::dyn_cast<mlir::IntegerType>(operand.getType())) {
          simt_hlsl_import::SourceLoc src{unOp, loc};
          std::string tag = buildIntegerTag(intType);
          mlir::Value allOnes =
              interp.emitConstantInt(-1, tag.c_str(), src);
          return interp.emitArithmetic(simt_hlsl_import::ArithOp::BitXor,
                                       operand, allOnes, src);
        }
      }
      if (auto intType = mlir::dyn_cast<mlir::IntegerType>(operand.getType())) {
        mlir::Value allOnes = makeIntegerConstant(-1, intType);
        return ctx.builder.create<mlir::arith::XOrIOp>(loc, operand, allOnes)
            .getResult();
      }
      return ctx.fail("bitwise not requires integer operand"), mlir::Value();
    }
    case clang::UnaryOperatorKind::UO_PreInc:
    case clang::UnaryOperatorKind::UO_PreDec:
    case clang::UnaryOperatorKind::UO_PostInc:
    case clang::UnaryOperatorKind::UO_PostDec: {
      const clang::ValueDecl *target = getMutableDeclRef(unOp->getSubExpr());
      if (!target)
        return ctx.fail("increment/decrement requires simple variable"),
               mlir::Value();

      auto it = ctx.valueMap.find(target);
      if (it == ctx.valueMap.end())
        return ctx.fail("reference to unknown value"), mlir::Value();

      mlir::Value original = operand;
      mlir::Value updated;
      bool isIncrement =
          unOp->getOpcode() == clang::UnaryOperatorKind::UO_PreInc ||
          unOp->getOpcode() == clang::UnaryOperatorKind::UO_PostInc;

      if (isEmitContext(ctx)) {
        EmitInterpreter interp(ctx);
        simt_hlsl_import::SourceLoc src{unOp, loc};
        if (auto intType = mlir::dyn_cast<mlir::IntegerType>(operand.getType())) {
          std::string tag = buildIntegerTag(intType);
          mlir::Value one = interp.emitConstantInt(1, tag.c_str(), src);
          updated = interp.emitArithmetic(isIncrement
                                              ? simt_hlsl_import::ArithOp::Add
                                              : simt_hlsl_import::ArithOp::Sub,
                                          operand, one, src);
        } else if (auto floatType =
                       mlir::dyn_cast<mlir::FloatType>(operand.getType())) {
          std::string tag = buildFloatTag(floatType);
          mlir::Value one = interp.emitConstantFloat(1.0, tag.c_str(), src);
          updated = interp.emitArithmetic(isIncrement
                                              ? simt_hlsl_import::ArithOp::Add
                                              : simt_hlsl_import::ArithOp::Sub,
                                          operand, one, src);
        } else {
          return ctx.fail("increment/decrement requires numeric operand"),
                 mlir::Value();
        }
        ctx.valueMap[target] = updated;
        ctx.mutatedVars.insert(target);
        interp.noteMutation(target);
        bool isPost = unOp->getOpcode() == clang::UnaryOperatorKind::UO_PostInc ||
                      unOp->getOpcode() == clang::UnaryOperatorKind::UO_PostDec;
        return isPost ? original : updated;
      }

      if (auto intType = mlir::dyn_cast<mlir::IntegerType>(operand.getType())) {
        mlir::Value one = makeIntegerConstant(1, intType);
        updated =
            isIncrement
                ? ctx.builder.create<mlir::arith::AddIOp>(loc, operand, one)
                      .getResult()
                : ctx.builder.create<mlir::arith::SubIOp>(loc, operand, one)
                      .getResult();
      } else if (auto floatType =
                     mlir::dyn_cast<mlir::FloatType>(operand.getType())) {
        mlir::Value one = makeFloatConstant(1.0, floatType);
        updated =
            isIncrement
                ? ctx.builder.create<mlir::arith::AddFOp>(loc, operand, one)
                      .getResult()
                : ctx.builder.create<mlir::arith::SubFOp>(loc, operand, one)
                      .getResult();
      } else {
        return ctx.fail("increment/decrement requires numeric operand"),
               mlir::Value();
      }

      ctx.valueMap[target] = updated;
      ctx.mutatedVars.insert(target);

      bool isPost = unOp->getOpcode() == clang::UnaryOperatorKind::UO_PostInc ||
                    unOp->getOpcode() == clang::UnaryOperatorKind::UO_PostDec;
      return isPost ? original : updated;
    }
    default:
      return ctx.fail("unsupported unary operator"), mlir::Value();
    }
  }


  if (const auto *binOp = llvm::dyn_cast<clang::BinaryOperator>(expr)) {
    if (binOp->getOpcode() == clang::BinaryOperatorKind::BO_Assign)
      return lowerAssignment(binOp, ctx);

    mlir::Value lhs = lowerExprLegacy(binOp->getLHS(), ctx);
    if (ctx.failed || !lhs)
      return {};

    mlir::Value rhsStorage;
    bool rhsEvaluated = false;
    auto getRHS = [&]() -> mlir::Value {
      if (!rhsEvaluated) {
        rhsStorage = lowerExprLegacy(binOp->getRHS(), ctx);
        rhsEvaluated = true;
      }
      return rhsStorage;
    };

    switch (binOp->getOpcode()) {
    case clang::BinaryOperatorKind::BO_EQ:
    case clang::BinaryOperatorKind::BO_NE:
    case clang::BinaryOperatorKind::BO_LT:
    case clang::BinaryOperatorKind::BO_LE:
    case clang::BinaryOperatorKind::BO_GT:
    case clang::BinaryOperatorKind::BO_GE: {
      mlir::Value rhs = getRHS();
      if (ctx.failed || !rhs)
        return {};
      simt_hlsl_import::SourceLoc src{binOp, loc};
      if (isEmitContext(ctx)) {
        simt_hlsl_import::CmpOp cmpOp;
        switch (binOp->getOpcode()) {
        case clang::BinaryOperatorKind::BO_EQ:
          cmpOp = simt_hlsl_import::CmpOp::EQ;
          break;
        case clang::BinaryOperatorKind::BO_NE:
          cmpOp = simt_hlsl_import::CmpOp::NE;
          break;
        case clang::BinaryOperatorKind::BO_LT:
          cmpOp = simt_hlsl_import::CmpOp::LT;
          break;
        case clang::BinaryOperatorKind::BO_LE:
          cmpOp = simt_hlsl_import::CmpOp::LE;
          break;
        case clang::BinaryOperatorKind::BO_GT:
          cmpOp = simt_hlsl_import::CmpOp::GT;
          break;
        case clang::BinaryOperatorKind::BO_GE:
          cmpOp = simt_hlsl_import::CmpOp::GE;
          break;
        default:
          llvm_unreachable("unhandled cmp opcode");
        }
        EmitInterpreter interp(ctx);
        return interp.emitCompare(cmpOp, lhs, rhs, src);
      }
      AnalysisInterpreter interp(ctx);
      simt_hlsl_import::CmpOp cmpOp;
      switch (binOp->getOpcode()) {
      case clang::BinaryOperatorKind::BO_EQ:
        cmpOp = simt_hlsl_import::CmpOp::EQ;
        break;
      case clang::BinaryOperatorKind::BO_NE:
        cmpOp = simt_hlsl_import::CmpOp::NE;
        break;
      case clang::BinaryOperatorKind::BO_LT:
        cmpOp = simt_hlsl_import::CmpOp::LT;
        break;
      case clang::BinaryOperatorKind::BO_LE:
        cmpOp = simt_hlsl_import::CmpOp::LE;
        break;
      case clang::BinaryOperatorKind::BO_GT:
        cmpOp = simt_hlsl_import::CmpOp::GT;
        break;
      case clang::BinaryOperatorKind::BO_GE:
        cmpOp = simt_hlsl_import::CmpOp::GE;
        break;
      default:
        llvm_unreachable("unhandled cmp opcode");
      }
      return interp.emitCompare(cmpOp, lhs, rhs, src);
    }
    case clang::BinaryOperatorKind::BO_LAnd: {
      if (lhs.getType() != ctx.builder.getI1Type())
        return ctx.fail("logical and requires boolean operands"), mlir::Value();

      auto rhsBuilder = [&](LoweringContext &branchCtx) -> mlir::Value {
        return lowerExprLegacy(binOp->getRHS(), branchCtx);
      };
      simt_hlsl_import::SourceLoc src{binOp, loc};
      if (isEmitContext(ctx)) {
        EmitInterpreter interp(ctx);
        return interp.emitShortCircuit(simt_hlsl_import::LogicalOp::And, lhs,
                                       rhsBuilder, src);
      }
      AnalysisInterpreter interp(ctx);
      return interp.emitShortCircuit(simt_hlsl_import::LogicalOp::And, lhs,
                                     rhsBuilder, src);
    }
    case clang::BinaryOperatorKind::BO_LOr: {
      if (lhs.getType() != ctx.builder.getI1Type())
        return ctx.fail("logical or requires boolean operands"), mlir::Value();

      auto rhsBuilder = [&](LoweringContext &branchCtx) -> mlir::Value {
        return lowerExprLegacy(binOp->getRHS(), branchCtx);
      };
      simt_hlsl_import::SourceLoc src{binOp, loc};
      if (isEmitContext(ctx)) {
        EmitInterpreter interp(ctx);
        return interp.emitShortCircuit(simt_hlsl_import::LogicalOp::Or, lhs,
                                       rhsBuilder, src);
      }
      AnalysisInterpreter interp(ctx);
      return interp.emitShortCircuit(simt_hlsl_import::LogicalOp::Or, lhs,
                                     rhsBuilder, src);
    }
    case clang::BinaryOperatorKind::BO_Add: {
      mlir::Value rhs = getRHS();
      if (ctx.failed || !rhs)
        return {};
      simt_hlsl_import::SourceLoc src{binOp, loc};
      if (isEmitContext(ctx)) {
        EmitInterpreter interp(ctx);
        return interp.emitArithmetic(simt_hlsl_import::ArithOp::Add, lhs, rhs,
                                     src);
      }
      AnalysisInterpreter interp(ctx);
      return interp.emitArithmetic(simt_hlsl_import::ArithOp::Add, lhs, rhs,
                                   src);
    }
    case clang::BinaryOperatorKind::BO_Sub: {
      mlir::Value rhs = getRHS();
      if (ctx.failed || !rhs)
        return {};
      simt_hlsl_import::SourceLoc src{binOp, loc};
      if (isEmitContext(ctx)) {
        EmitInterpreter interp(ctx);
        return interp.emitArithmetic(simt_hlsl_import::ArithOp::Sub, lhs, rhs,
                                     src);
      }
      AnalysisInterpreter interp(ctx);
      return interp.emitArithmetic(simt_hlsl_import::ArithOp::Sub, lhs, rhs,
                                   src);
    }
    case clang::BinaryOperatorKind::BO_Mul: {
      mlir::Value rhs = getRHS();
      if (ctx.failed || !rhs)
        return {};
      simt_hlsl_import::SourceLoc src{binOp, loc};
      if (isEmitContext(ctx)) {
        EmitInterpreter interp(ctx);
        return interp.emitArithmetic(simt_hlsl_import::ArithOp::Mul, lhs, rhs,
                                     src);
      }
      AnalysisInterpreter interp(ctx);
      return interp.emitArithmetic(simt_hlsl_import::ArithOp::Mul, lhs, rhs,
                                   src);
    }
    case clang::BinaryOperatorKind::BO_Div: {
      mlir::Value rhs = getRHS();
      if (ctx.failed || !rhs)
        return {};
      simt_hlsl_import::SourceLoc src{binOp, loc};
      if (isEmitContext(ctx)) {
        EmitInterpreter interp(ctx);
        return interp.emitArithmetic(simt_hlsl_import::ArithOp::Div, lhs, rhs,
                                     src);
      }
      AnalysisInterpreter interp(ctx);
      return interp.emitArithmetic(simt_hlsl_import::ArithOp::Div, lhs, rhs,
                                   src);
    }
    case clang::BinaryOperatorKind::BO_Rem: {
      mlir::Value rhs = getRHS();
      if (ctx.failed || !rhs)
        return {};
      simt_hlsl_import::SourceLoc src{binOp, loc};
      if (isEmitContext(ctx)) {
        EmitInterpreter interp(ctx);
        return interp.emitArithmetic(simt_hlsl_import::ArithOp::Rem, lhs, rhs,
                                     src);
      }
      AnalysisInterpreter interp(ctx);
      return interp.emitArithmetic(simt_hlsl_import::ArithOp::Rem, lhs, rhs,
                                   src);
    }
    default:
      break;
    }

    return ctx.fail("unsupported binary operator"), mlir::Value();
  }

  return ctx.fail("unsupported expression lowering"), mlir::Value();
}

static std::optional<BufferAccessInfo>
getBufferAccessInfo(const clang::Expr *baseExpr, const clang::Expr *indexExpr,
                    LoweringContext &ctx) {
  if (!baseExpr || !indexExpr)
    return std::nullopt;

  baseExpr = baseExpr->IgnoreParenImpCasts();
  indexExpr = indexExpr->IgnoreParenImpCasts();

  mlir::Value resource = lowerExprLegacy(baseExpr, ctx);
  if (!resource)
    return std::nullopt;

  auto resourceType =
      mlir::dyn_cast<simt::dialect::ResourceType>(resource.getType());
  if (!resourceType)
    return ctx.fail("subscript base must be a buffer resource"), std::nullopt;

  mlir::Value index = lowerExprLegacy(indexExpr, ctx);
  if (!index)
    return std::nullopt;
  if (!mlir::isa<mlir::IntegerType>(index.getType()))
    return ctx.fail("buffer subscript index must be integer"), std::nullopt;

  const clang::ValueDecl *decl = nullptr;
  if (const auto *declRef = llvm::dyn_cast<clang::DeclRefExpr>(baseExpr))
    decl = declRef->getDecl();

  return BufferAccessInfo{resource, index, resourceType, decl};
}

static std::optional<BufferAccessInfo>
lowerBufferAccessOperands(const clang::CXXOperatorCallExpr *opCall,
                          LoweringContext &ctx) {
  if (!opCall || opCall->getNumArgs() < 2)
    return std::nullopt;
  return getBufferAccessInfo(opCall->getArg(0), opCall->getArg(1), ctx);
}

static std::optional<mlir::Value>
lowerAtomicMemberCall(const clang::CXXMemberCallExpr *call,
                      LoweringContext &ctx) {
  if (!call)
    return std::nullopt;

  const auto *methodDecl = call->getMethodDecl();
  if (!methodDecl)
    return std::nullopt;

  auto kind =
      llvm::StringSwitch<std::optional<simt_hlsl_import::BufferAtomicOp>>(
          methodDecl->getName())
          .Case("InterlockedAdd", simt_hlsl_import::BufferAtomicOp::Add)
          .Case("InterlockedExchange",
                 simt_hlsl_import::BufferAtomicOp::Exchange)
          .Case("InterlockedCompareExchange",
                 simt_hlsl_import::BufferAtomicOp::CompareExchange)
          .Case("InterlockedMin", simt_hlsl_import::BufferAtomicOp::Min)
          .Case("InterlockedMax", simt_hlsl_import::BufferAtomicOp::Max)
          .Case("InterlockedAnd", simt_hlsl_import::BufferAtomicOp::And)
          .Case("InterlockedOr", simt_hlsl_import::BufferAtomicOp::Or)
          .Case("InterlockedXor", simt_hlsl_import::BufferAtomicOp::Xor)
          .Default(std::nullopt);
  if (!kind)
    return std::nullopt;

  const clang::Expr *objectExpr = call->getImplicitObjectArgument();
  if (!objectExpr)
    return ctx.fail("atomic call requires an object expression"),
           std::optional<mlir::Value>(mlir::Value());

  mlir::Value resource = lowerExpr(objectExpr, ctx);
  if (!resource)
    return mlir::Value();

  auto resourceType =
      mlir::dyn_cast<simt::dialect::ResourceType>(resource.getType());
  if (!resourceType)
    return ctx.fail("atomic call requires a buffer resource"),
           std::optional<mlir::Value>(mlir::Value());

  mlir::Location loc = getLocation(call, ctx);

  unsigned numArgs = call->getNumArgs();
  unsigned valueOperandCount =
      *kind == simt_hlsl_import::BufferAtomicOp::CompareExchange ? 2U : 1U;
  unsigned baseArgCount = 1 + valueOperandCount;
  if (numArgs != baseArgCount && numArgs != baseArgCount + 1)
    return ctx.fail("unexpected argument count for atomic call"),
           std::optional<mlir::Value>(mlir::Value());

  unsigned argIndex = 0;
  mlir::Value indexValue = lowerExpr(call->getArg(argIndex++), ctx);
  if (!indexValue)
    return mlir::Value();

  BufferAccessInfo info{resource, indexValue, resourceType, nullptr};

  mlir::Value compareValue;
  mlir::Value valueValue;
  if (*kind == simt_hlsl_import::BufferAtomicOp::CompareExchange) {
    compareValue = lowerExpr(call->getArg(argIndex++), ctx);
    if (!compareValue)
      return mlir::Value();
    valueValue = lowerExpr(call->getArg(argIndex++), ctx);
    if (!valueValue)
      return mlir::Value();
  } else {
    valueValue = lowerExpr(call->getArg(argIndex++), ctx);
    if (!valueValue)
      return mlir::Value();
  }

  const clang::Expr *outArg = nullptr;
  if (numArgs == baseArgCount + 1)
    outArg = call->getArg(argIndex++);

  const clang::ValueDecl *objectDecl = nullptr;
  if (const auto *declRef = llvm::dyn_cast<clang::DeclRefExpr>(
          objectExpr->IgnoreParenImpCasts()))
    objectDecl = declRef->getDecl();

  if (isEmitContext(ctx)) {
    EmitInterpreter interp(ctx);
    mlir::Value oldValue = interp.emitAtomic(
        *kind, info.resource, info.index, valueValue, compareValue,
        /*resourceDecl=*/nullptr, {call, loc});
    if (!oldValue && ctx.failed)
      return mlir::Value();

    if (objectDecl) {
      ctx.mutatedVars.insert(objectDecl);
      ctx.symValueMap[objectDecl] = makeSymValue(objectDecl);
    }

    if (outArg) {
      const clang::Expr *stripped = outArg->IgnoreParenImpCasts();
      if (const auto *outExpr =
              llvm::dyn_cast<clang::HLSLOutArgExpr>(stripped))
        stripped = outExpr->getArgLValue()->IgnoreParenImpCasts();
      const clang::ValueDecl *outDecl = nullptr;
      if (const auto *declRef =
              llvm::dyn_cast<clang::DeclRefExpr>(stripped))
        outDecl = declRef->getDecl();
      if (!outDecl)
        return ctx.fail("atomic original value argument must reference a "
                        "variable"),
               std::optional<mlir::Value>(mlir::Value());
      ctx.valueMap[outDecl] = oldValue;
      ctx.symValueMap[outDecl] = makeSymValue(outDecl);
      ctx.mutatedVars.insert(outDecl);
    }

    return mlir::Value();
  }

  AnalysisInterpreter interp(ctx);
  interp.emitAtomic(*kind, info.resource, info.index, valueValue, compareValue,
                    /*resourceDecl=*/nullptr, {call, loc});

  if (objectDecl) {
    ctx.mutatedVars.insert(objectDecl);
    ctx.symValueMap[objectDecl] = makeSymValue(objectDecl);
  }

  if (outArg) {
    const clang::Expr *stripped = outArg->IgnoreParenImpCasts();
    if (const auto *outExpr =
            llvm::dyn_cast<clang::HLSLOutArgExpr>(stripped))
      stripped = outExpr->getArgLValue()->IgnoreParenImpCasts();
    const clang::ValueDecl *outDecl = nullptr;
    if (const auto *declRef =
            llvm::dyn_cast<clang::DeclRefExpr>(stripped))
      outDecl = declRef->getDecl();
    if (!outDecl)
      return ctx.fail("atomic original value argument must reference a "
                      "variable"),
             std::optional<mlir::Value>(mlir::Value());
    ctx.symValueMap[outDecl] = makeSymValue(outDecl);
    ctx.mutatedVars.insert(outDecl);
  }

  return mlir::Value();
}

template <typename Interp>
static typename Interp::Value
lowerAssignmentInterp(const clang::BinaryOperator *binOp, LoweringContext &ctx,
                      Interp &interp) {
  auto rhs = lowerExprInterp(binOp->getRHS(), ctx, interp);
  if (!rhs)
    return typename Interp::Value();

  mlir::Value rhsValue = unwrapValue(rhs);
  mlir::Type rhsType = getValueType(rhs);
  mlir::Location loc = getLocation(binOp, ctx);

  const clang::Expr *lhsExpr = binOp->getLHS()->IgnoreParenImpCasts();
  if (const auto *lhsDeclRef = llvm::dyn_cast<clang::DeclRefExpr>(lhsExpr)) {
    const clang::ValueDecl *vd = lhsDeclRef->getDecl();
    auto it = ctx.valueMap.find(vd);
    if (it == ctx.valueMap.end()) {
      ctx.fail("reference to unknown value");
      return typename Interp::Value();
    }

    interp.bindVariable(vd, rhs);
    interp.noteMutation(vd);

    if (rhsValue)
      it->second = rhsValue;
    ctx.mutatedVars.insert(vd);
    ctx.symValueMap[vd] = makeSymValue(vd);
    return rhs;
  }

  auto emitBufferStore = [&](typename Interp::Value resource,
                             typename Interp::Value index,
                             simt::dialect::ResourceType resourceType,
                             const clang::ValueDecl *decl) -> typename Interp::Value {
    if (rhsType != resourceType.getElementType()) {
      ctx.fail("assignment value must match buffer element type");
      return typename Interp::Value();
    }

    interp.emitBufferStore(resource, index, rhs, decl, {binOp, loc});
    if (decl)
      ctx.mutatedVars.insert(decl);
    return rhs;
  };

  if (const auto *subscript =
          llvm::dyn_cast<clang::CXXOperatorCallExpr>(lhsExpr)) {
    if (subscript->getOperator() == clang::OO_Subscript) {
      auto info = lowerBufferAccessInterp(subscript->getArg(0),
                                          subscript->getArg(1), ctx, interp);
      if (!info)
        return typename Interp::Value();
      auto [resource, index, resourceType, decl] = *info;
      return emitBufferStore(resource, index, resourceType, decl);
    }
  }

  if (const auto *arraySub = llvm::dyn_cast<clang::ArraySubscriptExpr>(lhsExpr)) {
    auto info = lowerBufferAccessInterp(arraySub->getBase(), arraySub->getIdx(),
                                        ctx, interp);
    if (!info)
      return typename Interp::Value();
    auto [resource, index, resourceType, decl] = *info;
    return emitBufferStore(resource, index, resourceType, decl);
  }

  ctx.fail("unsupported assignment target");
  return typename Interp::Value();
}

static mlir::Value lowerAssignment(const clang::BinaryOperator *binOp,
                                   LoweringContext &ctx) {
  if (isEmitContext(ctx)) {
    EmitInterpreter interp(ctx);
    return lowerAssignmentInterp(binOp, ctx, interp);
  }
  AnalysisInterpreter interp(ctx);
  auto result = lowerAssignmentInterp(binOp, ctx, interp);
  return result.getValueOrNull();
}

static mlir::Value buildZeroValue(LoweringContext &ctx, mlir::Type type) {
  mlir::Location loc = ctx.defaultLoc;
  if (mlir::isa<mlir::IntegerType>(type)) {
    auto attr =
        ctx.builder.getIntegerAttr(mlir::cast<mlir::IntegerType>(type), 0);
    return ctx.builder.create<mlir::arith::ConstantOp>(loc, attr);
  }
  if (mlir::isa<mlir::FloatType>(type)) {
    auto attr =
        ctx.builder.getFloatAttr(mlir::cast<mlir::FloatType>(type), 0.0);
    return ctx.builder.create<mlir::arith::ConstantOp>(loc, attr);
  }
  if (auto vectorType = mlir::dyn_cast<mlir::VectorType>(type)) {
    mlir::Type elementType = vectorType.getElementType();
    mlir::Attribute elementAttr;
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elementType))
      elementAttr = ctx.builder.getIntegerAttr(intType, 0);
    else if (auto floatType = mlir::dyn_cast<mlir::FloatType>(elementType))
      elementAttr = ctx.builder.getFloatAttr(floatType, 0.0);
    else
      return ctx.fail("unable to build default value for return type"),
             mlir::Value();

    auto zeroAttr = mlir::DenseElementsAttr::get(vectorType, elementAttr);
    return ctx.builder.create<mlir::arith::ConstantOp>(loc, zeroAttr);
  }
  ctx.fail("unable to build default value for return type");
  return {};
}

template <typename Interp>
static bool lowerStatement(const clang::Stmt *stmt, LoweringContext &ctx,
                           Interp &interp);
static bool lowerStatement(const clang::Stmt *stmt, LoweringContext &ctx);
static mlir::Value lowerExprLegacy(const clang::Expr *expr,
                                   LoweringContext &ctx);

template <typename Interp>
static bool lowerForStmt(const clang::ForStmt *stmt, LoweringContext &ctx,
                         Interp &interp);
template <typename Interp>
static bool lowerWhileStmt(const clang::WhileStmt *stmt, LoweringContext &ctx,
                           Interp &interp);
template <typename Interp>
static bool lowerDoStmt(const clang::DoStmt *stmt, LoweringContext &ctx,
                        Interp &interp);
template <typename Interp>
static bool lowerSwitchStmt(const clang::SwitchStmt *stmt,
                            LoweringContext &ctx, Interp &interp);
template <typename Interp>
static void lowerCompoundStmt(const clang::CompoundStmt *compound,
                              LoweringContext &ctx, Interp &interp);

template <typename Interp>
static void lowerCompoundStmt(const clang::CompoundStmt *compound,
                              LoweringContext &ctx, Interp &interp) {
  for (const clang::Stmt *child : compound->body()) {
    if (ctx.emittedTerminator || ctx.failed)
      break;
    lowerStatement(child, ctx, interp);
  }
}

static void lowerCompoundStmt(const clang::CompoundStmt *compound,
                              LoweringContext &ctx) {
  if (isEmitContext(ctx)) {
    EmitInterpreter interp(ctx);
    lowerCompoundStmt(compound, ctx, interp);
  } else {
    AnalysisInterpreter interp(ctx);
    lowerCompoundStmt(compound, ctx, interp);
  }
}


static bool collectLoopMutations(
    LoweringContext &ctx, const clang::Stmt *body,
    llvm::function_ref<bool(LoweringContext &)> extraWork,
    llvm::SmallVector<const clang::ValueDecl *, 8> &mutatedVars) {
  mutatedVars.clear();

  mlir::Region analysisRegion;
  analysisRegion.emplaceBlock();
  mlir::OpBuilder analysisBuilder(ctx.builder.getContext());
  analysisBuilder.setInsertionPointToStart(&analysisRegion.front());

  LoweringContext analysisCtx(analysisBuilder, ctx.defaultLoc, ctx.returnType,
                              ctx.errorMessage, ctx.sourceManager);
  cloneContextState(ctx, analysisCtx);

  if (body) {
    if (!lowerStatement(body, analysisCtx) || analysisCtx.failed)
      return false;
  }

  if (!analysisCtx.emittedTerminator) {
    if (!extraWork(analysisCtx) || analysisCtx.failed)
      return false;
  }

  llvm::SmallPtrSet<const clang::ValueDecl *, 8> mutatedSet;
  mutatedSet.insert(analysisCtx.mutatedVars.begin(),
                    analysisCtx.mutatedVars.end());

  mutatedVars.assign(mutatedSet.begin(), mutatedSet.end());
  llvm::sort(mutatedVars,
             [](const clang::ValueDecl *lhs, const clang::ValueDecl *rhs) {
               return lhs < rhs;
             });

  return true;
}

static bool collectIfMutations(
    LoweringContext &ctx, const clang::Stmt *thenStmt,
    const clang::Stmt *elseStmt,
    llvm::SmallVector<const clang::ValueDecl *, 8> &mutatedVars) {
  mutatedVars.clear();

  llvm::SmallPtrSet<const clang::ValueDecl *, 8> mutatedSet;

  auto analyzeBranch = [&](const clang::Stmt *branchStmt) -> bool {
    if (!branchStmt)
      return true;

    mlir::Region analysisRegion;
    analysisRegion.emplaceBlock();
    mlir::OpBuilder analysisBuilder(ctx.builder.getContext());
    analysisBuilder.setInsertionPointToStart(&analysisRegion.front());

    LoweringContext analysisCtx(analysisBuilder, ctx.defaultLoc, ctx.returnType,
                                ctx.errorMessage, ctx.sourceManager);
    cloneContextState(ctx, analysisCtx);

    AnalysisInterpreter analysisInterp(analysisCtx);
    if (!lowerStatement(branchStmt, analysisCtx, analysisInterp) ||
        analysisCtx.failed)
      return false;

    mutatedSet.insert(analysisCtx.mutatedVars.begin(),
                      analysisCtx.mutatedVars.end());
    return true;
  };

  if (!analyzeBranch(thenStmt))
    return false;
  if (!analyzeBranch(elseStmt))
    return false;

  mutatedVars.assign(mutatedSet.begin(), mutatedSet.end());
  llvm::sort(mutatedVars,
             [](const clang::ValueDecl *lhs, const clang::ValueDecl *rhs) {
               return lhs < rhs;
             });

  return true;
}

using LoopScope = LoopScopeState;

EmitInterpreter::LoopScope EmitInterpreter::beginLoop(llvm::ArrayRef<const clang::ValueDecl *> carriedVars,
                           bool hasFirstIterFlag, mlir::Value firstIterInit,
                           mlir::Location loc) {
  return LoopScopeState(ctx, carriedVars, hasFirstIterFlag, firstIterInit, loc);
}

AnalysisInterpreter::AnalysisInterpreter::LoopScope AnalysisInterpreter::beginLoop(
    llvm::ArrayRef<const clang::ValueDecl *> carriedVars, bool hasFirstIterFlag,
    mlir::Value firstIterInit, mlir::Location loc) {
  return LoopScopeState(ctx, carriedVars, hasFirstIterFlag, firstIterInit, loc);
}

template <typename Interp>
static bool lowerStatement(const clang::Stmt *stmt, LoweringContext &ctx,
                           Interp &interp) {
  if (ctx.failed)
    return false;

  if (const auto *compound = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
    lowerCompoundStmt(compound, ctx, interp);
    return true;
  }

  if (const auto *forStmt = llvm::dyn_cast<clang::ForStmt>(stmt))
    return lowerForStmt(forStmt, ctx, interp);

  if (const auto *whileStmt = llvm::dyn_cast<clang::WhileStmt>(stmt))
    return lowerWhileStmt(whileStmt, ctx, interp);

  if (const auto *doStmt = llvm::dyn_cast<clang::DoStmt>(stmt))
    return lowerDoStmt(doStmt, ctx, interp);

  if (const auto *switchStmt = llvm::dyn_cast<clang::SwitchStmt>(stmt))
    return lowerSwitchStmt(switchStmt, ctx, interp);

  if (llvm::isa<clang::BreakStmt>(stmt)) {
    BreakTarget target = getInnermostBreakTarget(ctx);
    if (!target)
      return true;

    if (target.kind == ControlEntryKind::Loop && target.loop) {
      if (target.loop->analysisOnly) {
        ctx.mutatedVars.insert(target.loop->carriedVars.begin(),
                               target.loop->carriedVars.end());
        for (const clang::ValueDecl *vd : target.loop->carriedVars)
          interp.noteMutation(vd);
        ctx.emittedTerminator = true;
        return true;
      }
      llvm::SmallVector<mlir::Value, 8> operands;
      collectLoopBreakOperands(ctx, *target.loop, operands);
      ctx.builder.create<simt::dialect::BreakOp>(ctx.defaultLoc, operands);
      ctx.mutatedVars.insert(target.loop->carriedVars.begin(),
                             target.loop->carriedVars.end());
      for (const clang::ValueDecl *vd : target.loop->carriedVars)
        interp.noteMutation(vd);
      ctx.emittedTerminator = true;
      return true;
    }

    if (target.kind == ControlEntryKind::Switch && target.switchFrame) {
      auto *switchFrame = target.switchFrame;
      if (switchFrame->analysisOnly) {
        ctx.emittedTerminator = true;
        return true;
      }

      llvm::SmallVector<mlir::Value, 8> yieldOperands;
      yieldOperands.reserve(switchFrame->carriedVars.size() + 3);
      for (auto [index, vd] : llvm::enumerate(switchFrame->carriedVars)) {
        mlir::Value value = ctx.valueMap.lookup(vd);
        if (!value && index < switchFrame->initialValues.size())
          value = switchFrame->initialValues[index];
        if (!value) {
          ctx.fail("switch break missing value for case variable");
          return false;
        }
        yieldOperands.push_back(value);
        ctx.mutatedVars.insert(vd);
        interp.noteMutation(vd);
      }

      auto ensureBool = [&](mlir::Value v, int constant) -> mlir::Value {
        if (v)
          return v;
        return ctx.builder.create<mlir::arith::ConstantIntOp>(ctx.defaultLoc,
                                                              constant, 1);
      };

      yieldOperands.push_back(ensureBool(switchFrame->breakHasMatchedValue, 1));
      yieldOperands.push_back(ensureBool(switchFrame->breakExecutingValue, 0));
      yieldOperands.push_back(ensureBool(switchFrame->breakCompletedValue, 1));

      ctx.builder.create<simt::dialect::YieldOp>(ctx.defaultLoc, yieldOperands);
      ctx.emittedTerminator = true;
    }
    return true;
  }

  if (llvm::isa<clang::ContinueStmt>(stmt)) {
    if (LoopFrame *loopFrame = getInnermostLoop(ctx)) {
      if (loopFrame->analysisOnly) {
        ctx.mutatedVars.insert(loopFrame->carriedVars.begin(),
                               loopFrame->carriedVars.end());
        for (const clang::ValueDecl *vd : loopFrame->carriedVars)
          interp.noteMutation(vd);
        ctx.emittedTerminator = true;
        return true;
      }
      llvm::SmallVector<mlir::Value, 8> operands;
      collectLoopContinueOperands(ctx, *loopFrame, operands);
      ctx.builder.create<simt::dialect::ContinueOp>(ctx.defaultLoc, operands);
      ctx.mutatedVars.insert(loopFrame->carriedVars.begin(),
                             loopFrame->carriedVars.end());
      for (const clang::ValueDecl *vd : loopFrame->carriedVars)
        interp.noteMutation(vd);
      ctx.emittedTerminator = true;
    }
    return true;
  }

  if (const auto *ifStmt = llvm::dyn_cast<clang::IfStmt>(stmt)) {
    mlir::Value cond = lowerExpr(ifStmt->getCond(), ctx);
    if (!cond)
      return false;

    mlir::Location loc = ctx.defaultLoc;
    bool hasElse = ifStmt->getElse() != nullptr;

    llvm::SmallVector<const clang::ValueDecl *, 8> mutatedVars;
    if (!collectIfMutations(ctx, ifStmt->getThen(), ifStmt->getElse(),
                            mutatedVars))
      return false;

    llvm::SmallVector<const clang::ValueDecl *, 8> carriedVars;
    carriedVars.reserve(mutatedVars.size());
    for (const clang::ValueDecl *vd : mutatedVars)
      if (ctx.valueMap.count(vd))
        carriedVars.push_back(vd);

    bool needsElseRegion = hasElse || !carriedVars.empty();
    auto scope =
        interp.beginIf(cond, carriedVars, hasElse, needsElseRegion, loc);

    LoweringContext &thenCtx = scope.thenContext();
    auto thenInterp = interp.fork(thenCtx);
    if (const clang::Stmt *thenBody = ifStmt->getThen())
      if (!lowerStatement(thenBody, thenCtx, thenInterp))
        return false;

    if (scope.userHasElse()) {
      LoweringContext &elseCtx = scope.elseContext();
      auto elseInterp = interp.fork(elseCtx);
      if (const clang::Stmt *elseBody = ifStmt->getElse())
        if (!lowerStatement(elseBody, elseCtx, elseInterp))
          return false;
    }

    if (!scope.done())
      return false;

    return true;
  }

  if (const auto *ret = llvm::dyn_cast<clang::ReturnStmt>(stmt)) {
    std::optional<mlir::Value> retValue;
    if (const clang::Expr *retExpr = ret->getRetValue()) {
      mlir::Value value = lowerExpr(retExpr, ctx);
      if (!value)
        return false;
      retValue = value;
    }
    simt_hlsl_import::SourceLoc src{ret, getLocation(ret, ctx)};
    interp.emitReturn(retValue, src);
    return !ctx.failed;
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
        interp.bindVariable(var, initValue);
      if (isEmitContext(ctx))
        ctx.symValueMap[var] = makeSymValue(var);
      else
        ctx.symValueMap[var] = makeSymValue(var);
    }
    return true;
  }

  if (const auto *exprStmt = llvm::dyn_cast<clang::Expr>(stmt)) {
    const clang::Expr *stripped = exprStmt->IgnoreParenImpCasts();
    if (const auto *call = llvm::dyn_cast<clang::CallExpr>(stripped)) {
      if (lowerBarrierUtilityCall(call, ctx, interp)) {
        if (!call->getType()->isVoidType())
          return ctx.fail("barrier call must return void");
        return true;
      }
    }

    (void)lowerExpr(exprStmt, ctx);
    return !ctx.failed;
  }

  return ctx.fail("unsupported statement");
}

static bool lowerStatement(const clang::Stmt *stmt, LoweringContext &ctx) {
  if (isEmitContext(ctx)) {
    EmitInterpreter interp(ctx);
    return lowerStatement(stmt, ctx, interp);
  }
  AnalysisInterpreter interp(ctx);
  return lowerStatement(stmt, ctx, interp);
}

template <typename Interp>
static bool lowerForStmt(const clang::ForStmt *forStmt, LoweringContext &ctx,
                         Interp &interp) {
  mlir::Location loc = ctx.defaultLoc;

  if (const clang::Stmt *init = forStmt->getInit()) {
    if (llvm::isa<clang::DeclStmt>(init)) {
      if (!lowerStatement(init, ctx, interp) || ctx.failed)
        return false;
    } else if (const auto *initExpr = llvm::dyn_cast<clang::Expr>(init)) {
      (void)lowerExpr(initExpr, ctx);
      if (ctx.failed)
        return false;
    } else {
      return ctx.fail("unsupported for-loop initializer");
    }
  }

  llvm::SmallVector<const clang::ValueDecl *, 8> mutatedVars;
  const clang::Stmt *incStmt = forStmt->getInc();
  auto analyzeIncrement = [&](LoweringContext &analysisCtx) -> bool {
    if (!incStmt)
      return true;
    if (const auto *incExpr = llvm::dyn_cast<clang::Expr>(incStmt)) {
      (void)lowerExpr(incExpr, analysisCtx);
      return !analysisCtx.failed;
    }
    if (!lowerStatement(incStmt, analysisCtx) || analysisCtx.failed)
      return false;
    return true;
  };
  if (!collectLoopMutations(ctx, forStmt->getBody(), analyzeIncrement,
                            mutatedVars))
    return false;

  auto loopScope = interp.beginLoop(mutatedVars, /*hasFirstIterFlag=*/false,
                                    mlir::Value(), loc);
  if (!loopScope.isValid())
    return false;

  LoweringContext &prepareCtx = loopScope.prepareContext();
  mlir::OpBuilder &prepBuilder = prepareCtx.builder;

  mlir::Value condValue;
  if (const clang::Expr *condExpr = forStmt->getCond()) {
    condValue = lowerExpr(condExpr, prepareCtx);
    if (!condValue || prepareCtx.failed)
      return false;
  } else {
    condValue = prepBuilder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
  }

  llvm::SmallVector<mlir::Value, 8> forwarded;
  forwarded.reserve(mutatedVars.size());
  for (const clang::ValueDecl *vd : mutatedVars) {
    mlir::Value value = prepareCtx.valueMap.lookup(vd);
    if (!value)
      value = ctx.valueMap.lookup(vd);
    forwarded.push_back(value);
  }

  prepBuilder.create<simt::dialect::ConditionOp>(loc, condValue, forwarded);

  LoweringContext &bodyCtx = loopScope.bodyContext();
  auto bodyInterp = interp.fork(bodyCtx);

  if (const clang::Stmt *body = forStmt->getBody()) {
    if (!lowerStatement(body, bodyCtx, bodyInterp) || bodyCtx.failed)
      return false;
  }

  if (!bodyCtx.emittedTerminator) {
    if (const auto *incExpr =
            llvm::dyn_cast_or_null<clang::Expr>(forStmt->getInc())) {
      (void)lowerExpr(incExpr, bodyCtx);
      if (bodyCtx.failed)
        return false;
    } else if (const clang::Stmt *inc = forStmt->getInc()) {
      if (!lowerStatement(inc, bodyCtx, bodyInterp) || bodyCtx.failed)
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
    bodyCtx.builder.create<simt::dialect::YieldOp>(loc, yieldOperands);
  }

  return loopScope.close();
}

template <typename Interp>
static bool lowerWhileStmt(const clang::WhileStmt *whileStmt,
                           LoweringContext &ctx, Interp &interp) {
  mlir::Location loc = ctx.defaultLoc;

  const clang::Expr *condExpr = whileStmt->getCond();
  llvm::SmallVector<const clang::ValueDecl *, 8> mutatedVars;
  auto analyzeCond = [&](LoweringContext &analysisCtx) -> bool {
    if (!condExpr)
      return true;
    (void)lowerExpr(condExpr, analysisCtx);
    return !analysisCtx.failed;
  };
  if (!collectLoopMutations(ctx, whileStmt->getBody(), analyzeCond,
                            mutatedVars))
    return false;

  auto loopScope = interp.beginLoop(mutatedVars, /*hasFirstIterFlag=*/false,
                                    mlir::Value(), loc);
  if (!loopScope.isValid())
    return false;

  LoweringContext &prepareCtx = loopScope.prepareContext();
  mlir::OpBuilder &prepBuilder = prepareCtx.builder;

  mlir::Value condValue;
  if (condExpr) {
    condValue = lowerExpr(condExpr, prepareCtx);
    if (!condValue || prepareCtx.failed)
      return false;
  } else {
    condValue = prepBuilder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
  }

  llvm::SmallVector<mlir::Value, 8> forwarded;
  forwarded.reserve(mutatedVars.size());
  for (const clang::ValueDecl *vd : mutatedVars) {
    mlir::Value value = prepareCtx.valueMap.lookup(vd);
    if (!value)
      value = ctx.valueMap.lookup(vd);
    forwarded.push_back(value);
  }

  prepBuilder.create<simt::dialect::ConditionOp>(loc, condValue, forwarded);

  LoweringContext &bodyCtx = loopScope.bodyContext();
  auto bodyInterp = interp.fork(bodyCtx);

  if (const clang::Stmt *body = whileStmt->getBody()) {
    if (!lowerStatement(body, bodyCtx, bodyInterp) || bodyCtx.failed)
      return false;
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
    bodyCtx.builder.create<simt::dialect::YieldOp>(loc, yieldOperands);
  }

  return loopScope.close();
}

template <typename Interp>
static bool lowerDoStmt(const clang::DoStmt *doStmt, LoweringContext &ctx,
                        Interp &interp) {
  mlir::Location loc = ctx.defaultLoc;

  const clang::Expr *condExpr = doStmt->getCond();
  llvm::SmallVector<const clang::ValueDecl *, 8> mutatedVars;
  auto analyzeCond = [&](LoweringContext &analysisCtx) -> bool {
    if (!condExpr)
      return true;
    (void)lowerExpr(condExpr, analysisCtx);
    return !analysisCtx.failed;
  };
  if (!collectLoopMutations(ctx, doStmt->getBody(), analyzeCond, mutatedVars))
    return false;

  mlir::Value firstIterInit =
      ctx.builder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);

  auto loopScope = interp.beginLoop(mutatedVars, /*hasFirstIterFlag=*/true,
                                    firstIterInit, loc);
  if (!loopScope.isValid())
    return false;

  LoweringContext &prepareCtx = loopScope.prepareContext();
  mlir::OpBuilder &prepBuilder = prepareCtx.builder;

  mlir::Value condValue;
  if (loopScope.hasFirstIterFlag()) {
    mlir::Value firstIterFlag = loopScope.getPrepareFirstIterArg();

    llvm::SmallVector<mlir::Type, 8> condResultTypes;
    condResultTypes.push_back(prepBuilder.getI1Type());
    for (const clang::ValueDecl *vd : mutatedVars)
      condResultTypes.push_back(prepareCtx.valueMap.lookup(vd).getType());

    auto condIf = prepBuilder.create<simt::dialect::IfOp>(
        loc, condResultTypes, firstIterFlag, /*withElseRegion=*/true);

    // Then region: first iteration, bypass condition.
    {
      auto &thenBlock = condIf.getThenRegion().front();
      thenBlock.clear();
      mlir::OpBuilder thenBuilder(prepBuilder.getContext());
      thenBuilder.setInsertionPointToStart(&thenBlock);
      auto trueConst =
          thenBuilder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
      llvm::SmallVector<mlir::Value, 8> thenOperands;
      thenOperands.reserve(condResultTypes.size());
      thenOperands.push_back(trueConst);
      for (const clang::ValueDecl *vd : mutatedVars)
        thenOperands.push_back(prepareCtx.valueMap.lookup(vd));
      thenBuilder.create<simt::dialect::YieldOp>(loc, thenOperands);
    }

    // Else region: evaluate condition on subsequent iterations.
    {
      auto &elseBlock = condIf.getElseRegion().front();
      elseBlock.clear();
      mlir::OpBuilder elseBuilder(prepBuilder.getContext());
      elseBuilder.setInsertionPointToStart(&elseBlock);
      LoweringContext condCtx(elseBuilder, loc, ctx.returnType, ctx.errorMessage,
                              ctx.sourceManager);
      condCtx.valueMap = prepareCtx.valueMap;
      condCtx.symValueMap = prepareCtx.symValueMap;
      condCtx.loopStack = prepareCtx.loopStack;
      condCtx.switchStack = prepareCtx.switchStack;
      condCtx.controlStack = prepareCtx.controlStack;

      mlir::Value evaluated;
      if (condExpr) {
        evaluated = lowerExpr(condExpr, condCtx);
        if (!evaluated || condCtx.failed)
          return false;
      } else {
        evaluated = elseBuilder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
      }

      llvm::SmallVector<mlir::Value, 8> elseOperands;
      elseOperands.reserve(condResultTypes.size());
      elseOperands.push_back(evaluated);
      for (const clang::ValueDecl *vd : mutatedVars) {
        mlir::Value value = condCtx.valueMap.lookup(vd);
        if (!value)
          value = prepareCtx.valueMap.lookup(vd);
        elseOperands.push_back(value);
      }
      elseBuilder.create<simt::dialect::YieldOp>(loc, elseOperands);
    }

    condValue = condIf.getResult(0);
    for (auto [index, vd] : llvm::enumerate(mutatedVars))
      prepareCtx.valueMap[vd] = condIf.getResult(index + 1);
  } else {
    if (condExpr) {
      condValue = lowerExpr(condExpr, prepareCtx);
      if (!condValue || prepareCtx.failed)
        return false;
    } else {
      condValue = prepBuilder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
    }
  }

  llvm::SmallVector<mlir::Value, 8> forwarded;
  forwarded.reserve(mutatedVars.size() + (loopScope.hasFirstIterFlag() ? 1 : 0));
  for (const clang::ValueDecl *vd : mutatedVars) {
    mlir::Value value = prepareCtx.valueMap.lookup(vd);
    if (!value)
      value = ctx.valueMap.lookup(vd);
    forwarded.push_back(value);
  }

  mlir::Value continueFlagInit;
  if (loopScope.hasFirstIterFlag()) {
    continueFlagInit =
        prepBuilder.create<mlir::arith::ConstantIntOp>(loc, 0, 1);
    forwarded.push_back(continueFlagInit);
    loopScope.setCurrentFirstIterValue(continueFlagInit);
  }

  prepBuilder.create<simt::dialect::ConditionOp>(loc, condValue, forwarded);

  // Body region: execute loop body and forward flag state.
  LoweringContext &bodyCtx = loopScope.bodyContext();
  auto bodyInterp = interp.fork(bodyCtx);

  mlir::Value continueFlag;
  if (loopScope.hasFirstIterFlag()) {
    continueFlag =
        bodyCtx.builder.create<mlir::arith::ConstantIntOp>(loc, 0, 1);
    loopScope.setCurrentFirstIterValue(continueFlag);
  }

  if (const clang::Stmt *body = doStmt->getBody()) {
    if (!lowerStatement(body, bodyCtx, bodyInterp) || bodyCtx.failed)
      return false;
  }

  if (!bodyCtx.emittedTerminator) {
    llvm::SmallVector<mlir::Value, 8> yieldOperands;
    yieldOperands.reserve(mutatedVars.size() +
                          (loopScope.hasFirstIterFlag() ? 1 : 0));
    for (const clang::ValueDecl *vd : mutatedVars) {
      mlir::Value value = bodyCtx.valueMap.lookup(vd);
      if (!value)
        value = ctx.valueMap.lookup(vd);
      yieldOperands.push_back(value);
    }
    if (loopScope.hasFirstIterFlag())
      yieldOperands.push_back(continueFlag);
    bodyCtx.builder.create<simt::dialect::YieldOp>(loc, yieldOperands);
  }

  return loopScope.close();
}

template <typename Interp>
static bool lowerSwitchStmt(const clang::SwitchStmt *switchStmt,
                            LoweringContext &ctx, Interp &interp) {
  mlir::Location loc = ctx.defaultLoc;

  if (const clang::Stmt *init = switchStmt->getInit()) {
    if (!lowerStatement(init, ctx, interp) || ctx.failed)
      return false;
  }

  if (switchStmt->getConditionVariable())
    return ctx.fail("switch condition variables are not supported");

  struct CaseInfo {
    const clang::SwitchCase *label = nullptr;
    llvm::SmallVector<const clang::Stmt *, 8> statements;
  };

  llvm::SmallVector<CaseInfo, 8> cases;

  std::function<bool(CaseInfo *, const clang::Stmt *)> addStatement;
  addStatement = [&](CaseInfo *current, const clang::Stmt *stmt) -> bool {
    if (!stmt)
      return true;
    if (llvm::isa<clang::NullStmt>(stmt))
      return true;
    if (const auto *attr = llvm::dyn_cast<clang::AttributedStmt>(stmt))
      return addStatement(current, attr->getSubStmt());
    if (!current)
      return ctx.fail("statement outside of switch cases is not supported"),
             false;
    current->statements.push_back(stmt);
    return true;
  };

  auto pushCase = [&](const clang::SwitchCase *sc) -> CaseInfo * {
    cases.push_back({sc, {}});
    return &cases.back();
  };

  const clang::Stmt *body = switchStmt->getBody();
  llvm::SmallVector<const clang::Stmt *, 8> topLevel;
  if (const auto *compound = llvm::dyn_cast<clang::CompoundStmt>(body)) {
    topLevel.append(compound->body_begin(), compound->body_end());
  } else if (body) {
    topLevel.push_back(body);
  }

  CaseInfo *current = nullptr;
  for (const clang::Stmt *child : topLevel) {
    if (const auto *sc = llvm::dyn_cast<clang::SwitchCase>(child)) {
      const clang::SwitchCase *active = sc;
      const clang::Stmt *sub = nullptr;
      do {
        current = pushCase(active);
        sub = active->getSubStmt();
        active = llvm::dyn_cast<clang::SwitchCase>(sub);
      } while (active);
      if (sub && !addStatement(current, sub))
        return false;
    } else {
      if (!addStatement(current, child))
        return false;
    }
  }

  mlir::Value selector = lowerExpr(switchStmt->getCond(), ctx);
  if (!selector)
    return false;

  llvm::SmallPtrSet<const clang::ValueDecl *, 8> mutatedSet;
  for (const CaseInfo &info : cases) {
    mlir::Region analysisRegion;
    analysisRegion.emplaceBlock();
    mlir::OpBuilder analysisBuilder(ctx.builder.getContext());
    analysisBuilder.setInsertionPointToStart(&analysisRegion.front());
    LoweringContext analysisCtx(analysisBuilder, loc, ctx.returnType,
                                ctx.errorMessage, ctx.sourceManager);
    cloneContextState(ctx, analysisCtx);
    SwitchScopeGuard analysisGuard(analysisCtx, SwitchFrame{});
    analysisGuard.frame().analysisOnly = true;

    for (const clang::Stmt *caseStmt : info.statements) {
      if (!lowerStatement(caseStmt, analysisCtx) || analysisCtx.failed)
        return false;
      if (analysisCtx.emittedTerminator)
        break;
    }
    mutatedSet.insert(analysisCtx.mutatedVars.begin(),
                      analysisCtx.mutatedVars.end());
  }

  llvm::SmallVector<const clang::ValueDecl *, 8> mutatedVars(mutatedSet.begin(),
                                                             mutatedSet.end());
  llvm::sort(mutatedVars,
             [](const clang::ValueDecl *lhs, const clang::ValueDecl *rhs) {
               return lhs < rhs;
             });

  llvm::SmallVector<mlir::Value, 8> currentValues;
  currentValues.reserve(mutatedVars.size());
  for (const clang::ValueDecl *vd : mutatedVars) {
    mlir::Value initial = ctx.valueMap.lookup(vd);
    if (!initial)
      return ctx.fail("reference to unknown switch variable");
    currentValues.push_back(initial);
  }

  mlir::Value boolZero =
      ctx.builder.create<mlir::arith::ConstantIntOp>(loc, 0, 1);
  mlir::Value currentHasMatched = boolZero;
  mlir::Value currentExecuting = boolZero;
  mlir::Value currentCompleted = boolZero;

  for (const CaseInfo &info : cases) {
    mlir::Value notCompleted = ctx.builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, currentCompleted, boolZero);
    mlir::Value hasNotMatched = ctx.builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, currentHasMatched, boolZero);

    mlir::Value caseMatch;
    if (const auto *caseStmt = llvm::dyn_cast<clang::CaseStmt>(info.label)) {
      mlir::Value caseValue = lowerExpr(caseStmt->getLHS(), ctx);
      if (!caseValue)
        return false;
      mlir::Value valueEquals = ctx.builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::eq, selector, caseValue);
      mlir::Value available = ctx.builder.create<mlir::arith::AndIOp>(
          loc, hasNotMatched, notCompleted);
      caseMatch =
          ctx.builder.create<mlir::arith::AndIOp>(loc, available, valueEquals);
    } else {
      caseMatch = ctx.builder.create<mlir::arith::AndIOp>(loc, hasNotMatched,
                                                          notCompleted);
    }

    mlir::Value executeCondition = ctx.builder.create<mlir::arith::OrIOp>(
        loc, currentExecuting, caseMatch);
    mlir::Value enterCase = ctx.builder.create<mlir::arith::AndIOp>(
        loc, executeCondition, notCompleted);

    llvm::SmallVector<mlir::Type, 8> resultTypes;
    resultTypes.reserve(mutatedVars.size() + 3);
    for (mlir::Value value : currentValues)
      resultTypes.push_back(value.getType());
    resultTypes.push_back(ctx.builder.getI1Type());
    resultTypes.push_back(ctx.builder.getI1Type());
    resultTypes.push_back(ctx.builder.getI1Type());

    auto ifOp =
        ctx.builder.create<simt::dialect::IfOp>(loc, resultTypes, enterCase,
                                                /*withElseRegion=*/true);

    auto &thenBlock = ifOp.getThenRegion().front();
    thenBlock.clear();
    mlir::OpBuilder thenBuilder(ctx.builder.getContext());
    thenBuilder.setInsertionPointToStart(&thenBlock);

    LoweringContext caseCtx(thenBuilder, loc, ctx.returnType, ctx.errorMessage,
                            ctx.sourceManager);
    cloneContextState(ctx, caseCtx);
    for (auto [vd, value] : llvm::zip(mutatedVars, currentValues))
      caseCtx.valueMap[vd] = value;

    auto caseInterp = interp.fork(caseCtx);

    SwitchScopeGuard caseGuard(
        caseCtx, makeSwitchFrame(caseCtx, mutatedVars, currentValues, loc));

    for (const clang::Stmt *caseStmt : info.statements) {
      if (!lowerStatement(caseStmt, caseCtx, caseInterp) || caseCtx.failed)
        return false;
      if (caseCtx.emittedTerminator)
        break;
    }

    if (!caseCtx.emittedTerminator) {
      llvm::SmallVector<mlir::Value, 8> yieldValues;
      yieldValues.reserve(mutatedVars.size() + 3);
      for (auto [index, vd] : llvm::enumerate(mutatedVars)) {
        mlir::Value value = caseCtx.valueMap.lookup(vd);
        if (!value && index < currentValues.size())
          value = currentValues[index];
        if (!value) {
          caseCtx.fail("switch case missing value for variable");
          return false;
        }
        yieldValues.push_back(value);
        caseCtx.mutatedVars.insert(vd);
      }
      mlir::Value updatedHasMatched = thenBuilder.create<mlir::arith::OrIOp>(
          loc, currentHasMatched, caseMatch);
      mlir::Value updatedExecuting = thenBuilder.create<mlir::arith::OrIOp>(
          loc, currentExecuting, caseMatch);
      yieldValues.push_back(updatedHasMatched);
      yieldValues.push_back(updatedExecuting);
      yieldValues.push_back(currentCompleted);
      thenBuilder.create<simt::dialect::YieldOp>(loc, yieldValues);
    }
    ctx.mutatedVars.insert(caseCtx.mutatedVars.begin(),
                           caseCtx.mutatedVars.end());

    auto &elseBlock = ifOp.getElseRegion().front();
    elseBlock.clear();
    mlir::OpBuilder elseBuilder(ctx.builder.getContext());
    elseBuilder.setInsertionPointToStart(&elseBlock);

    llvm::SmallVector<mlir::Value, 8> elseValues = currentValues;
    elseValues.push_back(currentHasMatched);
    elseValues.push_back(currentExecuting);
    elseValues.push_back(currentCompleted);
    elseBuilder.create<simt::dialect::YieldOp>(loc, elseValues);

    currentValues.clear();
    currentValues.reserve(mutatedVars.size());
    for (size_t index = 0; index < mutatedVars.size(); ++index)
      currentValues.push_back(ifOp.getResult(index));
    currentHasMatched = ifOp.getResult(mutatedVars.size());
    currentExecuting = ifOp.getResult(mutatedVars.size() + 1);
    currentCompleted = ifOp.getResult(mutatedVars.size() + 2);
  }

  for (auto [vd, value] : llvm::zip(mutatedVars, currentValues))
    ctx.valueMap[vd] = value;

  ctx.mutatedVars.insert(mutatedVars.begin(), mutatedVars.end());

  return true;
}

class FunctionLoweringVisitor
    : public clang::RecursiveASTVisitor<FunctionLoweringVisitor> {
public:
  FunctionLoweringVisitor(mlir::OwningOpRef<mlir::ModuleOp> &module,
                          mlir::OpBuilder &builder)
      : module(module), moduleBuilder(builder) {}

  bool VisitVarDecl(const clang::VarDecl *decl) {
    if (!decl->hasGlobalStorage() || decl->isStaticLocal())
      return true;
    if (decl->isImplicit())
      return true;

    mlir::Type type = convertType(decl->getType(), moduleBuilder);
    if (!type) {
      clang::LangAS addressSpace = decl->getType().getAddressSpace();
      if (addressSpace == clang::LangAS::hlsl_groupshared) {
        clang::QualType varType = decl->getType();
        const clang::Type *elementTy =
            varType.getCanonicalType().getTypePtrOrNull();
        if (!elementTy)
          return true;
        while (auto *arrayTy = llvm::dyn_cast<clang::ArrayType>(elementTy))
          elementTy =
              arrayTy->getElementType().getCanonicalType().getTypePtr();
        mlir::Type elementType = convertType(clang::QualType(elementTy, 0),
                                             moduleBuilder);
        if (!elementType)
          return true;
        type = simt::dialect::ResourceType::get(
            moduleBuilder.getContext(),
            simt::dialect::MemorySpace::Shared, elementType);
      }
    }
    if (!type)
      return true;

    if (!mlir::isa<simt::dialect::ResourceType>(type))
      return true;

    if (resourceSet.insert(decl).second)
      resourceDecls.push_back(decl);
    return true;
  }

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
    argTypes.reserve(decl->getNumParams() + resourceDecls.size());
    for (const clang::ParmVarDecl *param : decl->parameters()) {
      mlir::Type type = convertType(param->getType(), moduleBuilder);
      if (!type || mlir::isa<mlir::NoneType>(type)) {
        recordError("unsupported parameter type in function '" + name + "'");
        return false;
      }
      argTypes.push_back(type);
    }

    for (const clang::VarDecl *resourceDecl : resourceDecls) {
      mlir::Type type = convertType(resourceDecl->getType(), moduleBuilder);
      if (!type || !mlir::isa<simt::dialect::ResourceType>(type)) {
        recordError("unsupported resource type in function '" + name + "'");
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

    const clang::SourceManager &sourceManager =
        decl->getASTContext().getSourceManager();

    LoweringContext ctx(funcBuilder, loc,
                        resultTypes.empty() ? mlir::Type()
                                            : resultTypes.front(),
                        errorMessage, &sourceManager);
    auto entryArgs = entry->getArguments();
    size_t paramCount = decl->getNumParams();
    for (size_t index = 0; index < paramCount; ++index)
      ctx.valueMap[decl->getParamDecl(index)] = entryArgs[index];
    for (auto [resourceDecl, arg] :
         llvm::zip(resourceDecls, entryArgs.drop_front(paramCount)))
      ctx.valueMap[resourceDecl] = arg;

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
        ctx.builder.create<mlir::func::ReturnOp>(ctx.builder.getUnknownLoc(),
                                                 zero);
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
  llvm::SmallVector<const clang::VarDecl *, 8> resourceDecls;
  llvm::SmallPtrSet<const clang::VarDecl *, 8> resourceSet;
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

Result<mlir::OwningOpRef<mlir::ModuleOp>>
translateComputeShader(mlir::MLIRContext &context, llvm::StringRef fileName,
                       llvm::StringRef source,
                       const TranslationOptions &options) {
  context.loadDialect<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                      mlir::math::MathDialect, mlir::vector::VectorDialect,
                      simt::dialect::SimtStepDialect>();

  mlir::OpBuilder builder(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module->getBody());

  FunctionLoweringVisitor visitor(module, builder);
  std::vector<std::string> clangArgs = {"-x", "hlsl", "-std=hlsl2021",
                                        "-D__HLSL__"};

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

  for (const std::string &include : options.forcedIncludeFiles) {
    clangArgs.emplace_back("-include");
    clangArgs.emplace_back(include);
  }

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
