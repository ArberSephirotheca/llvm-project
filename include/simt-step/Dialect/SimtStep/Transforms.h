#ifndef SIMT_STEP_DIALECT_SIMTSTEP_TRANSFORMS_H
#define SIMT_STEP_DIALECT_SIMTSTEP_TRANSFORMS_H

#include <memory>

#include "mlir/Pass/Pass.h"

namespace simt::dialect {

/// Create a pass that normalizes loop body control so result-producing
/// `simt_step.if` ops no longer end with `simt_step.continue` /
/// `simt_step.break`. The rewrite lifts the early terminator out of the
/// value-producing `if`, making the IR SSA-clean for downstream passes.
std::unique_ptr<mlir::Pass> createNormalizeLoopTerminatorsPass();

/// Register the normalize pass with the MLIR pass registry so it can be invoked
/// via `mlir-opt` style pipelines.
void registerNormalizeLoopTerminatorsPass();

} // namespace simt::dialect

#endif // SIMT_STEP_DIALECT_SIMTSTEP_TRANSFORMS_H
