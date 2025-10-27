#include "simt-step/semantics/SimpleProgram.h"
#include "simt-step/semantics/SemanticsContext.h"

#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>

using namespace simt::semantics;

namespace {

constexpr llvm::StringLiteral kProgramIR = R"mlir(
module {
  func.func @kernel() {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %sum = arith.addi %c1, %c2 : i32
    %lane = simt_step.lane_id : index
    simt_step.yield
  }
}
)mlir";

llvm::Expected<int> run() {
    mlir::DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    simt::dialect::SimtStepDialect>();

    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();

    auto module = mlir::parseSourceString<mlir::ModuleOp>(kProgramIR, &context);
    if (!module)
        return llvm::make_error<llvm::StringError>(
            "failed to parse simple program", llvm::inconvertibleErrorCode());

    auto func = mlir::dyn_cast<mlir::func::FuncOp>(*module->getBody()->begin());
    if (!func)
        return llvm::make_error<llvm::StringError>(
            "expected func.func entry", llvm::inconvertibleErrorCode());

    mlir::Block &block = func.getBody().front();

    SemanticsContext semaCtx;
    semaCtx.subgroupWidth = 32;
    semaCtx.activeMask = ~0ull;
    semaCtx.laneId = 4;

    SimpleProgramRunner runner;
    if (llvm::Error err = runner.runBlock(&block, semaCtx))
        return std::move(err);

    const auto &state = runner.state();
    auto waveIt = state.waves.find(0);
    if (waveIt == state.waves.end())
        return llvm::make_error<llvm::StringError>(
            "missing wave context", llvm::inconvertibleErrorCode());

    const auto &laneCtx = waveIt->second.lanes.lookup(0);
    if (!laneCtx.returnValue)
        return llvm::make_error<llvm::StringError>(
            "lane produced no value", llvm::inconvertibleErrorCode());

    int result = static_cast<int>(laneCtx.returnValue->asInt64());
    llvm::outs() << result << "\n";
    return result;
}

} // namespace

int main() {
    auto resultOrErr = run();
    if (!resultOrErr) {
        llvm::errs() << llvm::toString(resultOrErr.takeError()) << "\n";
        return 1;
    }
    // Expect (1 + 2) = 3 as the last computed value.
    return *resultOrErr == 3 ? 0 : 1;
}
