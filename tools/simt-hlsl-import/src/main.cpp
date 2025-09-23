#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Tooling.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

using namespace llvm;

namespace {

struct LoweringContext {
  mlir::OpBuilder &builder;
  mlir::Location defaultLoc;
  mlir::Type returnType;
  llvm::DenseMap<const clang::ValueDecl *, mlir::Value> valueMap;
  bool emittedTerminator = false;

  LoweringContext(mlir::OpBuilder &builder, mlir::Location loc,
                  mlir::Type retType)
      : builder(builder), defaultLoc(loc), returnType(retType) {}
};

static mlir::Location getLocation(const clang::Stmt *stmt, mlir::OpBuilder &builder) {
  (void)stmt;
  return builder.getUnknownLoc();
}

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
    case clang::BuiltinType::Short:
    case clang::BuiltinType::UShort:
    case clang::BuiltinType::Int:
    case clang::BuiltinType::UInt:
    case clang::BuiltinType::Long:
    case clang::BuiltinType::ULong: {
      unsigned bits = builtin->getKind() == clang::BuiltinType::Short ||
                              builtin->getKind() == clang::BuiltinType::UShort
                          ? 16
                          : builtin->getKind() == clang::BuiltinType::SChar ||
                                    builtin->getKind() == clang::BuiltinType::UChar
                                ? 8
                                : 32;
      return builder.getIntegerType(bits);
    }
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

static mlir::Value lowerExpr(const clang::Expr *expr, LoweringContext &ctx) {
  if (!expr)
    return {};

  mlir::Type type = convertType(expr->getType(), ctx.builder);
  if (!type) {
    llvm::errs() << "simt-hlsl-import: unsupported expression type\n";
    return {};
  }

  mlir::Location loc = getLocation(expr, ctx.builder);

  if (const auto *intLit = llvm::dyn_cast<clang::IntegerLiteral>(expr)) {
    if (!mlir::isa<mlir::IntegerType>(type)) {
      llvm::errs() << "simt-hlsl-import: integer literal with non-integer type\n";
      return {};
    }
    auto intType = mlir::cast<mlir::IntegerType>(type);
    auto attr = ctx.builder.getIntegerAttr(intType, intLit->getValue());
    return ctx.builder.create<mlir::arith::ConstantOp>(loc, attr);
  }

  if (const auto *floatLit = llvm::dyn_cast<clang::FloatingLiteral>(expr)) {
    if (!mlir::isa<mlir::FloatType>(type)) {
      llvm::errs() << "simt-hlsl-import: unexpected floating literal type\n";
      return {};
    }
    auto floatType = mlir::cast<mlir::FloatType>(type);
    auto attr = ctx.builder.getFloatAttr(floatType, floatLit->getValue());
    return ctx.builder.create<mlir::arith::ConstantOp>(loc, attr);
  }

  if (const auto *declRef = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
    const clang::ValueDecl *vd = declRef->getDecl();
    auto it = ctx.valueMap.find(vd);
    if (it != ctx.valueMap.end())
      return it->second;
    llvm::errs() << "simt-hlsl-import: reference to unknown value\n";
    return {};
  }

  llvm::errs() << "simt-hlsl-import: unsupported expression lowering\n";
  return {};
}

static mlir::Value buildZeroValue(LoweringContext &ctx, mlir::Type type) {
  mlir::Location loc = ctx.defaultLoc;
  if (mlir::isa<mlir::IntegerType>(type)) {
    auto intType = mlir::cast<mlir::IntegerType>(type);
    auto attr = ctx.builder.getIntegerAttr(intType, 0);
    return ctx.builder.create<mlir::arith::ConstantOp>(loc, attr);
  }
  if (mlir::isa<mlir::FloatType>(type)) {
    auto floatType = mlir::cast<mlir::FloatType>(type);
    auto attr = ctx.builder.getFloatAttr(floatType, 0.0);
    return ctx.builder.create<mlir::arith::ConstantOp>(loc, attr);
  }
  return {};
}

static bool lowerStatement(const clang::Stmt *stmt, LoweringContext &ctx) {
  if (const auto *ret = llvm::dyn_cast<clang::ReturnStmt>(stmt)) {
    bool expectsValue = static_cast<bool>(ctx.returnType) &&
                        !mlir::isa<mlir::NoneType>(ctx.returnType);
    if (ret->getRetValue() != nullptr) {
      mlir::Value value = lowerExpr(ret->getRetValue(), ctx);
      if (!value) {
        llvm::errs() << "simt-hlsl-import: unable to lower return value\n";
        return false;
      }
      if (!expectsValue) {
        llvm::errs() << "simt-hlsl-import: unexpected return value in void function\n";
        return false;
      }
      if (value.getType() != ctx.returnType) {
        llvm::errs() << "simt-hlsl-import: return type mismatch\n";
        return false;
      }
      ctx.builder.create<mlir::func::ReturnOp>(getLocation(ret, ctx.builder), value);
    } else {
      if (expectsValue) {
        llvm::errs() << "simt-hlsl-import: missing return value\n";
        return false;
      }
      ctx.builder.create<mlir::func::ReturnOp>(getLocation(ret, ctx.builder));
    }
    ctx.emittedTerminator = true;
    return true;
  }

  if (const auto *declStmt = llvm::dyn_cast<clang::DeclStmt>(stmt)) {
    for (const clang::Decl *decl : declStmt->decls()) {
      const auto *var = llvm::dyn_cast<clang::VarDecl>(decl);
      if (!var)
        continue;
      if (!convertType(var->getType(), ctx.builder)) {
        llvm::errs() << "simt-hlsl-import: unsupported variable type\n";
        return false;
      }
      mlir::Value initValue;
      if (const clang::Expr *init = var->getInit())
        initValue = lowerExpr(init, ctx);
      if (!initValue && var->getInit()) {
        llvm::errs() << "simt-hlsl-import: unable to lower variable initializer\n";
        return false;
      }
      if (initValue)
        ctx.valueMap[var] = initValue;
    }
    return true;
  }

  llvm::errs() << "simt-hlsl-import: ignoring unsupported statement\n";
  return false;
}

static void lowerCompoundStmt(const clang::CompoundStmt *compound,
                              LoweringContext &ctx) {
  for (const clang::Stmt *child : compound->body()) {
    if (ctx.emittedTerminator)
      break;
    lowerStatement(child, ctx);
  }
}

class FunctionLoweringVisitor
    : public clang::RecursiveASTVisitor<FunctionLoweringVisitor> {
public:
  FunctionLoweringVisitor(mlir::ModuleOp module, mlir::OpBuilder &builder)
      : module(module), moduleBuilder(builder) {}

  bool VisitFunctionDecl(const clang::FunctionDecl *decl) {
    if (!decl->doesThisDeclarationHaveABody() || !decl->isThisDeclarationADefinition())
      return true;

    auto name = decl->getNameAsString();
    mlir::Location loc = moduleBuilder.getUnknownLoc();

    mlir::OpBuilder::InsertionGuard guard(moduleBuilder);
    moduleBuilder.setInsertionPointToEnd(module.getBody());

    SmallVector<mlir::Type> argTypes;
    argTypes.reserve(decl->getNumParams());
    for (const clang::ParmVarDecl *param : decl->parameters()) {
      mlir::Type type = convertType(param->getType(), moduleBuilder);
      if (!type || mlir::isa<mlir::NoneType>(type)) {
        llvm::errs() << "simt-hlsl-import: unsupported parameter type in function '"
                     << name << "'\n";
        return true;
      }
      argTypes.push_back(type);
    }

    mlir::Type resultType = convertType(decl->getReturnType(), moduleBuilder);
    if (!resultType) {
      llvm::errs() << "simt-hlsl-import: unsupported return type in function '"
                   << name << "', treating as void\n";
      resultType = moduleBuilder.getNoneType();
    }
    SmallVector<mlir::Type> resultTypes;
    if (!mlir::isa<mlir::NoneType>(resultType))
      resultTypes.push_back(resultType);

    auto funcType = moduleBuilder.getFunctionType(argTypes, resultTypes);
    auto func = moduleBuilder.create<mlir::func::FuncOp>(loc, name, funcType);

    mlir::Block *entry = func.addEntryBlock();
    mlir::OpBuilder funcBuilder(entry, entry->begin());
    auto mask = funcBuilder.create<simt::dialect::ActiveMaskOp>(loc,
                                                                funcBuilder.getI64Type());
    (void)mask;

    LoweringContext ctx(funcBuilder, loc,
                        resultTypes.empty() ? mlir::Type() : resultTypes.front());
    for (auto [param, arg] : llvm::zip(decl->parameters(), entry->getArguments()))
      ctx.valueMap[param] = arg;
    if (const auto *body = llvm::dyn_cast<clang::CompoundStmt>(decl->getBody()))
      lowerCompoundStmt(body, ctx);
    else if (const clang::Stmt *bodyStmt = decl->getBody())
      lowerStatement(bodyStmt, ctx);

    if (!ctx.emittedTerminator) {
      if (ctx.builder.getInsertionBlock())
        ctx.builder.setInsertionPointToEnd(ctx.builder.getInsertionBlock());
      bool expectsValue = static_cast<bool>(ctx.returnType) &&
                          !mlir::isa<mlir::NoneType>(ctx.returnType);
      if (expectsValue) {
        if (mlir::Value zero = buildZeroValue(ctx, ctx.returnType))
          ctx.builder.create<mlir::func::ReturnOp>(loc, zero);
        else
          ctx.builder.create<mlir::func::ReturnOp>(loc);
      } else {
        ctx.builder.create<mlir::func::ReturnOp>(loc);
      }
    }

    return true;
  }

private:
  mlir::ModuleOp module;
  mlir::OpBuilder &moduleBuilder;
};

class SimtAstConsumer : public clang::ASTConsumer {
public:
  SimtAstConsumer(FunctionLoweringVisitor &visitor) : visitor(visitor) {}

  void HandleTranslationUnit(clang::ASTContext &context) override {
    visitor.TraverseDecl(context.getTranslationUnitDecl());
  }

private:
  FunctionLoweringVisitor &visitor;
};

class SimtFrontendAction : public clang::ASTFrontendAction {
public:
  SimtFrontendAction(FunctionLoweringVisitor &visitor) : visitor(visitor) {}

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &, llvm::StringRef) override {
    return std::make_unique<SimtAstConsumer>(visitor);
  }

private:
  FunctionLoweringVisitor &visitor;
};

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);

  llvm::cl::OptionCategory toolCategory("simt-hlsl-import options");
  llvm::cl::opt<std::string> inputPath(
      llvm::cl::Positional, llvm::cl::desc("<input HLSL file>"),
      llvm::cl::Required, llvm::cl::cat(toolCategory));
  llvm::cl::opt<std::string> shaderProfile(
      "profile", llvm::cl::desc("Target shader profile, e.g. ps_6_7"),
      llvm::cl::init("ps_6_7"), llvm::cl::cat(toolCategory));

  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "SIMT-Step HLSL importer (clang)\n");
  (void)shaderProfile;

  auto bufferOrErr = llvm::MemoryBuffer::getFile(inputPath);
  if (!bufferOrErr) {
    llvm::errs() << "simt-hlsl-import: failed to read input '" << inputPath
                 << "': " << bufferOrErr.getError().message() << "\n";
    return 1;
  }
  std::string source = bufferOrErr.get()->getBuffer().str();

  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect>();
  simt::dialect::registerSimtStepDialect(registry);

  mlir::MLIRContext context(registry);
  context.loadDialect<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                      simt::dialect::SimtStepDialect>();

  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module.getBody());

  FunctionLoweringVisitor visitor(module, builder);
  std::vector<std::string> clangArgs = {
      "-x", "hlsl", "-std=hlsl2021", "-D__HLSL__"};

  auto action = std::make_unique<SimtFrontendAction>(visitor);
  if (!clang::tooling::runToolOnCodeWithArgs(std::move(action), source,
                                             clangArgs, inputPath)) {
    llvm::errs() << "simt-hlsl-import: failed to translate input\n";
    return 1;
  }

  module.print(llvm::outs());
  llvm::outs() << '\n';
  return 0;
}
