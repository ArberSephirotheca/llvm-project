#pragma once

#include "simt-step/semantics/SemValue.h"

#include <cstdint>

#include <llvm/ADT/DenseMap.h>
#include <mlir/IR/Value.h>

namespace simt::semantics {

struct SemanticsContext {
    std::uint32_t subgroupWidth = 0;
    std::uint64_t activeMask = 0;
    std::uint64_t expectedMask = 0;
    std::uint32_t laneId = 0;
    const llvm::DenseMap<mlir::Value, SemValue> *valueEnv = nullptr;
};

} // namespace simt::semantics
