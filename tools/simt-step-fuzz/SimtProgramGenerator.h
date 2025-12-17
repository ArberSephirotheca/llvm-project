// Copyright (c) 2024.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <array>
#include <mlir/IR/BuiltinOps.h>

namespace simt::fuzz {

struct GeneratorConfig {
    std::array<std::int64_t, 3> numThreads{1, 1, 1};
    std::uint64_t seed = 0; // 0 = deterministic default
    std::uint32_t maxTripCount = 4;
};

/// Build a simple, deterministic SIMT-Step module:
/// - func @main(%out_wave: !simt_step.resource<Global, i32>)
/// - if (tid == 0) run a small counted loop, else use tid
/// - in a branch, emit wave_count_bits with a deterministic predicate and
///   store its result to %out_wave at an index derived from waveId/iter/tid
mlir::OwningOpRef<mlir::ModuleOp>
createDeterministicIfLoopModule(mlir::MLIRContext &context,
                                const GeneratorConfig &cfg = {});

/// Generate a small, deterministic-but-seeded SIMT-Step module with simple
/// structured control and wave ops. When seed=0, falls back to the default
/// template. When seed != 0, randomizes predicates/bounds within safe limits.
mlir::OwningOpRef<mlir::ModuleOp>
createRandomizedModule(mlir::MLIRContext &context,
                       const GeneratorConfig &cfg = {});

/// Generate a composite program with nested control and multiple wave ops,
/// using a seeded RNG for variability.
mlir::OwningOpRef<mlir::ModuleOp>
createRicherRandomModule(mlir::MLIRContext &context,
                         const GeneratorConfig &cfg = {});

} // namespace simt::fuzz
