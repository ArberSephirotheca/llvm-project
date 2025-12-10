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
/// - func @main()
/// - if (tid == 0) run a small counted loop, else use tid
/// - yields are local; meant as a runnable scaffold for interpreter tests
mlir::OwningOpRef<mlir::ModuleOp>
createDeterministicIfLoopModule(mlir::MLIRContext &context,
                                const GeneratorConfig &cfg = {});

} // namespace simt::fuzz
