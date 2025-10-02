#pragma once

#include <memory>

namespace mlir {
class Pass;
class PassRegistryEntry;
} // namespace mlir

namespace simt::conversion {

/// Create a pass that lowers the high-level simt_step dialect into the
/// structured SIMT dialect.
std::unique_ptr<mlir::Pass> createSimtStepToStructuredPass();

/// Register the SimtStep to SimtStructured conversion pass with the pass
/// registry.
void registerSimtStepToStructuredPass();

} // namespace simt::conversion
