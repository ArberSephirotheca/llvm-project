#pragma once

#include <mlir/IR/OpDefinition.h>

namespace simt::dialect {

/// Trait indicating an op executes independently per lane (no cross-lane state).
template <typename ConcreteType>
class SimtIndependent
    : public mlir::OpTrait::TraitBase<ConcreteType, SimtIndependent> {};

/// Trait for ops that require lane synchronization at a specified scope.
template <typename ConcreteType>
class SimtSynchronized
    : public mlir::OpTrait::TraitBase<ConcreteType, SimtSynchronized> {};

/// Trait for ops that perform collective behaviour across the active mask.
template <typename ConcreteType>
class SimtCollective
    : public mlir::OpTrait::TraitBase<ConcreteType, SimtCollective> {};

} // namespace simt::dialect
