#include "simt-step/Conversion/StructuredCFGBuilder.h"

#include "mlir/IR/IRMapping.h"

using namespace mlir;

namespace simt::conversion {

StructuredCFGBuilder::StructuredCFGBuilder(FunctionOpInterface func)
    : func(func) {}

LogicalResult StructuredCFGBuilder::build() {
  // Stub implementation.  The full structured CFG rewrite will materialise in
  // follow-up patches.
  return failure();
}

} // namespace simt::conversion
