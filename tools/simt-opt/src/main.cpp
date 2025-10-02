#include "simt-step/Conversion/SimtStepToStructured.h"
#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"
#include "simt-step/Dialect/SimtStructured/StructuredDialect.h"
#include "simt-step/semantics/StructuredProgram.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<mlir::BuiltinDialect, mlir::arith::ArithDialect,
                  mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  mlir::vector::VectorDialect>();
  simt::dialect::registerSimtStepDialect(registry);
  simt::structured::registerSimtStructuredDialect(registry);

  simt::conversion::registerSimtStepToStructuredPass();
  simt::semantics::registerDumpStructuredProgramPass();

  return failed(mlir::MlirOptMain(argc, argv, "SIMT-Step optimizer", registry));
}
