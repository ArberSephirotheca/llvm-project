#pragma once

#include "simt-step/semantics/SemValue.h"

#include <cstdint>
#include <optional>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringMap.h>
#include <mlir/IR/Value.h>

namespace simt::semantics {

enum class ExecutionMode {
    Independent,
    Synchronous,
    Collective,
};

struct ExecutionPolicy {
    ExecutionMode controlFlow = ExecutionMode::Independent;
    ExecutionMode waveOps = ExecutionMode::Collective;
    llvm::StringMap<ExecutionMode> overrides;
};

struct SemanticsContext {
    std::uint32_t subgroupWidth = 0;
    std::uint64_t activeMask = 0;
    std::uint64_t expectedMask = 0;
    std::uint32_t laneId = 0;
    const llvm::DenseMap<mlir::Value, SemValue> *valueEnv = nullptr;
    const ExecutionPolicy *policy = nullptr;
    std::optional<ExecutionMode> overrideMode;
};

} // namespace simt::semantics
