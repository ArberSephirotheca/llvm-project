#include "simt-step/Conversion/SimtStepToStructured.h"

#include "simt-step/Dialect/SimtStructured/StructuredDialect.h"
#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
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

    Block &entryBlock = func.front();

    SmallVector<Block *> originalBlocks;
    DenseMap<Block *, SmallVector<Operation *>> blockOriginalOps;
    for (Block &block : func) {
      originalBlocks.push_back(&block);
      for (Operation &op : block)
        blockOriginalOps[&block].push_back(&op);
    }

    OpBuilder topBuilder(&entryBlock, entryBlock.begin());
    auto loc = func.getLoc();

    IRMapping mapper;
    DenseMap<Block *, simt::structured::BlockOp> blockMapping;
    DenseMap<Block *, mlir::Value> blockActiveMask;

    for (auto [index, block] : llvm::enumerate(originalBlocks)) {
      std::string name = (index == 0) ? std::string("entry")
                                      : ("block" + std::to_string(index));
      auto symName = topBuilder.getStringAttr(name);
      auto blockOp = topBuilder.create<simt::structured::BlockOp>(
          loc, symName, FlatSymbolRefAttr(), FlatSymbolRefAttr(),
          simt::structured::ReconvergencePolicyAttr());
      topBuilder.setInsertionPointAfter(blockOp);

      mlir::Region &region = blockOp.getBody();
      if (region.empty())
        region.emplaceBlock();
      Block &structuredBlock = region.front();
      for (BlockArgument arg : block->getArguments()) {
        auto newArg = structuredBlock.addArgument(arg.getType(), arg.getLoc());
        mapper.map(arg, newArg);
      }

      blockMapping[block] = blockOp;
    }

    auto getOrCreateMask = [&](Block *origBlock, Block &structuredBlock,
                               mlir::Location loc) -> mlir::Value {
      auto it = blockActiveMask.find(origBlock);
      if (it != blockActiveMask.end())
        return it->second;

      OpBuilder maskBuilder = OpBuilder::atBlockBegin(&structuredBlock);
      auto mask = maskBuilder
                      .create<simt::dialect::ActiveMaskOp>(loc,
                                                          maskBuilder.getI64Type())
                      .getResult();
      blockActiveMask[origBlock] = mask;
      return mask;
    };

    SmallVector<Value> functionReturnValues;
    bool hasFunctionReturn = false;

    for (Block *origBlock : originalBlocks) {
      Block &structuredBlock = blockMapping[origBlock].getBody().front();
      OpBuilder bodyBuilder(&structuredBlock, structuredBlock.begin());

      for (Operation *op : blockOriginalOps[origBlock]) {
        if (isa<cf::BranchOp, cf::CondBranchOp, func::ReturnOp>(op))
          continue;

        Operation *cloned = bodyBuilder.clone(*op, mapper);
        for (auto [index, result] : llvm::enumerate(op->getResults()))
          mapper.map(result, cloned->getResult(index));

        if (auto maskOp = dyn_cast<simt::dialect::ActiveMaskOp>(op))
          blockActiveMask[origBlock] = cloned->getResult(0);
      }

      Operation *terminator = origBlock->getTerminator();
      Location termLoc = terminator->getLoc();

      auto mapValues = [&](ValueRange operands,
                           SmallVectorImpl<Value> &mapped) -> bool {
        for (Value operand : operands) {
          Value mappedValue = mapper.lookupOrNull(operand);
          if (!mappedValue) {
            terminator->emitError()
                << "unmapped operand during SimtStepToStructured lowering";
            return false;
          }
          mapped.push_back(mappedValue);
        }
        return true;
      };

      if (auto branch = dyn_cast<cf::BranchOp>(terminator)) {
        SmallVector<Value> destOperands;
        if (!mapValues(branch.getDestOperands(), destOperands)) {
          signalPassFailure();
          return;
        }

        Value mask =
            getOrCreateMask(origBlock, structuredBlock, terminator->getLoc());

        simt::structured::BlockOp destBlockOp = blockMapping[branch.getDest()];
        auto targetAttr = FlatSymbolRefAttr::get(destBlockOp.getContext(), destBlockOp.getSymName());

        bodyBuilder.create<simt::structured::BranchOp>(termLoc, mask, targetAttr,
                                                       destOperands);
      } else if (auto cond = dyn_cast<cf::CondBranchOp>(terminator)) {
        SmallVector<Value> trueOperands;
        SmallVector<Value> falseOperands;
        if (!mapValues(cond.getTrueDestOperands(), trueOperands) ||
            !mapValues(cond.getFalseDestOperands(), falseOperands)) {
          signalPassFailure();
          return;
        }

        Value condition = mapper.lookup(cond.getCondition());
        Value mask =
            getOrCreateMask(origBlock, structuredBlock, terminator->getLoc());

        simt::structured::BlockOp trueBlockOp = blockMapping[cond.getTrueDest()];
        simt::structured::BlockOp falseBlockOp = blockMapping[cond.getFalseDest()];

        auto trueTarget = FlatSymbolRefAttr::get(trueBlockOp.getContext(), trueBlockOp.getSymName());
        auto falseTarget = FlatSymbolRefAttr::get(falseBlockOp.getContext(), falseBlockOp.getSymName());

        bodyBuilder.create<simt::structured::CondBranchOp>(
            termLoc, condition, mask, mask, trueTarget, falseTarget, trueOperands,
            falseOperands, FlatSymbolRefAttr(),
            simt::structured::ReconvergencePolicyAttr());
      } else if (auto ret = dyn_cast<func::ReturnOp>(terminator)) {
        SmallVector<Value> returnValues;
        if (!mapValues(ret.getOperands(), returnValues)) {
          signalPassFailure();
          return;
        }
        bodyBuilder.create<simt::structured::ReturnOp>(termLoc, returnValues);

        if (!hasFunctionReturn) {
          functionReturnValues = returnValues;
          hasFunctionReturn = true;
        } else {
          if (functionReturnValues.size() != returnValues.size()) {
            ret.emitError()
                << "mismatched return arity across SimtStepToStructured lowering";
            signalPassFailure();
            return;
          }
          for (auto [expected, current] : llvm::zip(functionReturnValues, returnValues)) {
            if (expected.getType() != current.getType()) {
              ret.emitError()
                  << "mismatched return type across SimtStepToStructured lowering";
              signalPassFailure();
              return;
            }
          }
        }
      } else {
        terminator->emitError()
            << "unsupported terminator in SimtStepToStructured lowering";
        signalPassFailure();
        return;
      }
    }

    for (Block *origBlock : originalBlocks) {
      for (Operation *op : blockOriginalOps[origBlock])
        op->erase();
      if (origBlock != &entryBlock)
        origBlock->erase();
    }

    OpBuilder retBuilder(&entryBlock, entryBlock.end());
    if (hasFunctionReturn)
      retBuilder.create<func::ReturnOp>(loc, functionReturnValues);
    else
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
