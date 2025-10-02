#include "simt-step/Conversion/SimtStepToStructured.h"
#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"
#include "simt-step/Dialect/SimtStructured/StructuredDialect.h"

#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerAllDialects(registry);
  simt::dialect::registerSimtStepDialect(registry);
  simt::structured::registerSimtStructuredDialect(registry);

  registerAllPasses();
  simt::conversion::registerSimtStepToStructuredPass();

  return failed(mlir::MlirOptMain(argc, argv, "SIMT-Step optimizer", registry));
}
