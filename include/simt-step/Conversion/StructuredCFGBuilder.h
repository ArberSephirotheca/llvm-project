#ifndef SIMT_STEP_CONVERSION_STRUCTURED_CFGBUILDER_H
#define SIMT_STEP_CONVERSION_STRUCTURED_CFGBUILDER_H

#include "mlir/Support/LogicalResult.h"

namespace simt {
namespace conversion {

namespace mlir {
class FunctionOpInterface;
}

/// Forward declaration of the structured CFG builder.
///
/// The builder will eventually replace the ad-hoc lowering helpers in
/// `SimtStepToStructured.cpp`.  It takes a `func.func` that contains the
/// high-level `simt_step` control flow and materialises the structured SIMT
/// form in one pass.
class StructuredCFGBuilder {
public:
  explicit StructuredCFGBuilder(mlir::FunctionOpInterface func);

  /// Execute the structured lowering.  The initial implementation is a stub so
  /// the boilerplate can land independently from the functional rewrite.
  mlir::LogicalResult build();

private:
  mlir::FunctionOpInterface func;
};

} // namespace conversion
} // namespace simt

#endif // SIMT_STEP_CONVERSION_STRUCTURED_CFGBUILDER_H
