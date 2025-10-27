#include "simt-step/semantics/SimpleProgram.h"
#include "simt-step/semantics/SemanticsContext.h"

#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>

#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>

using namespace simt::semantics;

namespace {

llvm::Expected<int> run() {
    mlir::DialectRegistry registry;
    simt::dialect::registerSimtStepDialect(registry);
    registry.insert<mlir::BuiltinDialect, mlir::arith::ArithDialect,
                    mlir::func::FuncDialect, simt::dialect::SimtStepDialect>();

    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();
    (void)context.getOrLoadDialect<mlir::BuiltinDialect>();
    (void)context.getOrLoadDialect<mlir::arith::ArithDialect>();
    (void)context.getOrLoadDialect<mlir::func::FuncDialect>();
    (void)context.getOrLoadDialect<simt::dialect::SimtStepDialect>();

    mlir::OpBuilder builder(&context);
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(),
                                                   "kernel", funcType);
    module.push_back(func);
    auto *entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);
    auto loc = builder.getUnknownLoc();
    auto c1 = builder.create<mlir::arith::ConstantIntOp>(loc, 1, 32);
    auto c2 = builder.create<mlir::arith::ConstantIntOp>(loc, 2, 32);
    builder.create<mlir::arith::AddIOp>(loc, c1, c2);
    builder.create<simt::dialect::LaneIdOp>(loc, builder.getIndexType());
    builder.create<mlir::func::ReturnOp>(loc);

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
    // Expect lane id (4) to be the final produced value.
    return *resultOrErr == 4 ? 0 : 1;
}
