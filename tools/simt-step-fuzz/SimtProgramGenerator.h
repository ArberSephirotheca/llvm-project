// Copyright (c) 2024.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <array>
#include <mlir/IR/BuiltinOps.h>

namespace simt::fuzz {

struct GeneratorConfig {
    std::array<std::int64_t, 3> numThreads{1, 1, 1};
};

/// Build a simple, deterministic SIMT-Step module:
/// - func @main(%out_main: !simt_step.resource<Global, i32>,
///              %out_wave: !simt_step.resource<Global, i32>)
/// - if (tid == 0) run a small counted loop, else use tid
/// - store per-lane value to %out_main[tid]
/// - in a branch, emit wave_count_bits with a deterministic predicate and
///   store its result to %out_wave at an index derived from waveId/iter/tid
mlir::OwningOpRef<mlir::ModuleOp>
createDeterministicIfLoopModule(mlir::MLIRContext &context,
                                const GeneratorConfig &cfg = {});

} // namespace simt::fuzz
