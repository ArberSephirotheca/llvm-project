#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Tooling.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

using namespace llvm;

namespace {

class FunctionLoweringVisitor
    : public clang::RecursiveASTVisitor<FunctionLoweringVisitor> {
public:
  FunctionLoweringVisitor(mlir::ModuleOp module, mlir::OpBuilder &builder)
      : module(module), moduleBuilder(builder) {}

  bool VisitFunctionDecl(const clang::FunctionDecl *decl) {
    if (!decl->hasBody() || !decl->isDefinedOutsideFunctionOrMethod())
      return true;

    auto name = decl->getNameAsString();
    mlir::Location loc = moduleBuilder.getUnknownLoc();

    mlir::OpBuilder::InsertionGuard guard(moduleBuilder);
    moduleBuilder.setInsertionPointToEnd(module.getBody());

    auto funcType = moduleBuilder.getFunctionType({}, {});
    auto func = moduleBuilder.create<mlir::func::FuncOp>(loc, name, funcType);

    mlir::Block *entry = func.addEntryBlock();
    mlir::OpBuilder funcBuilder(entry, entry->begin());

    auto mask = funcBuilder.create<simt::dialect::ActiveMaskOp>(loc,
                                                                funcBuilder.getI64Type());
    (void)mask;
    funcBuilder.create<mlir::func::ReturnOp>(loc);
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
  registry.insert<mlir::func::FuncDialect>();
  simt::dialect::registerSimtStepDialect(registry);

  mlir::MLIRContext context(registry);
  context.loadDialect<mlir::func::FuncDialect, simt::dialect::SimtStepDialect>();

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
