#include "simt-step/Conversion/SimtStepToStructured.h"
#include "simt-step/Conversion/StructuredCFGBuilder.h"
#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"
#include "simt-step/Dialect/SimtStructured/StructuredDialect.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace simt::conversion {

namespace {

struct SimtStepToStructuredPass
    : public PassWrapper<SimtStepToStructuredPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimtStepToStructuredPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<simt::structured::SimtStructDialect, simt::dialect::SimtStepDialect, cf::ControlFlowDialect>();
  }

  StringRef getArgument() const final { return "simt-step-to-structured"; }
  StringRef getDescription() const final {
    return "Lower simt_step operations into the structured SIMT dialect";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    llvm::errs() << "[SimtStepToStructuredPass] running on function '"
                 << func.getName() << "'\n";
    llvm::errs().flush();

    StructuredCFGBuilder builder(func);
    if (failed(builder.build())) {
      getOperation().emitError() << "simt-step-to-structured: builder failed";
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createSimtStepToStructuredPass() {
  return std::make_unique<SimtStepToStructuredPass>();
}

void registerSimtStepToStructuredPass() {
  PassRegistration<SimtStepToStructuredPass>();
}

} // namespace simt::conversion
