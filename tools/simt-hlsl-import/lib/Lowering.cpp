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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
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

static void lowerCompoundStmt(const clang::CompoundStmt *compound,
                              LoweringContext &ctx) {
  for (const clang::Stmt *child : compound->body()) {
    if (ctx.emittedTerminator || ctx.failed)
      break;
    lowerStatement(child, ctx);
  }
}

static bool lowerStatement(const clang::Stmt *stmt, LoweringContext &ctx) {
  if (ctx.failed)
    return false;

  if (const auto *compound = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
    lowerCompoundStmt(compound, ctx);
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
                       llvm::StringRef source, const TranslationOptions &) {
  context.loadDialect<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                      simt::dialect::SimtStepDialect>();

  mlir::OpBuilder builder(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module->getBody());

  FunctionLoweringVisitor visitor(module, builder);
  std::vector<std::string> clangArgs = {
      "-x", "hlsl", "-std=hlsl2021", "-D__HLSL__"};

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
