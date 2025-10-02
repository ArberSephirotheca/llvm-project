#include "simt-step/Conversion/SimtStepToStructured.h"

#include "simt-step/Dialect/SimtStructured/StructuredDialect.h"
#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>

#include <llvm/ADT/STLExtras.h>

using namespace mlir;

namespace simt::conversion {

namespace {

struct SimtStepToStructuredPass
    : public PassWrapper<SimtStepToStructuredPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimtStepToStructuredPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<simt::structured::SimtStructDialect>();
  }

  StringRef getArgument() const final { return "simt-step-to-structured"; }
  StringRef getDescription() const final {
    return "Lower simt_step operations into the structured SIMT dialect";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    if (func.getBody().empty())
      return;

    if (!llvm::hasSingleElement(func.getBlocks())) {
      func.emitError()
          << "SimtStepToStructured currently supports single-block functions";
      signalPassFailure();
      return;
    }

    Block &entryBlock = func.front();

    if (!func.getFunctionType().getResults().empty()) {
      func.emitError()
          << "SimtStepToStructured currently supports only void functions";
      signalPassFailure();
      return;
    }

    SmallVector<Operation *> originalOps;
    originalOps.reserve(entryBlock.getOperations().size());
    for (Operation &op : entryBlock)
      originalOps.push_back(&op);

    OpBuilder topBuilder(&entryBlock, entryBlock.begin());
    auto loc = func.getLoc();
    auto blockOp = topBuilder.create<simt::structured::BlockOp>(
        loc, topBuilder.getStringAttr("entry"), FlatSymbolRefAttr(),
        FlatSymbolRefAttr(),
        simt::structured::ReconvergencePolicyAttr());

    mlir::Region &structuredRegion = blockOp.getBody();
    if (structuredRegion.empty())
        structuredRegion.emplaceBlock();
    Block &structuredEntry = structuredRegion.front();
    IRMapping mapper;

    for (BlockArgument arg : entryBlock.getArguments()) {
      auto newArg = structuredEntry.addArgument(arg.getType(), arg.getLoc());
      mapper.map(arg, newArg);
    }

    OpBuilder bodyBuilder(&structuredEntry, structuredEntry.begin());
    bool insertedStructuredReturn = false;

    for (Operation *op : originalOps) {
      if (isa<simt::structured::BlockOp>(op))
        continue;

      if (auto ret = dyn_cast<func::ReturnOp>(op)) {
        if (!ret.getOperands().empty()) {
          ret.emitError() << "expected void return";
          signalPassFailure();
          return;
        }
        bodyBuilder.create<simt::structured::ReturnOp>(ret.getLoc(), mlir::ValueRange{});
        insertedStructuredReturn = true;
        continue;
      }

      Operation *cloned = bodyBuilder.clone(*op, mapper);
      for (auto [index, result] : llvm::enumerate(op->getResults()))
        mapper.map(result, cloned->getResult(index));
    }

    if (!insertedStructuredReturn)
      bodyBuilder.create<simt::structured::ReturnOp>(loc, mlir::ValueRange{});

    // Remove the original operations now that the structured form exists.
    for (Operation *op : originalOps)
      op->erase();

    // Recreate a func.return to keep the function well-formed.
    Block *parentBlock = blockOp.getOperation()->getBlock();
    OpBuilder retBuilder(parentBlock,
                         std::next(Block::iterator(blockOp.getOperation())));
    retBuilder.create<func::ReturnOp>(loc);
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
