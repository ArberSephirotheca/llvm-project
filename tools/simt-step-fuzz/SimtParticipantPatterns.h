// Copyright (c) 2024.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>

namespace simt::fuzz {

/// Common lane-participation predicates built from dispatch_thread_id.
///
/// These helpers mirror the HLSL participant patterns in the DXC fuzzer but
/// emit MLIR values directly so they can be composed when constructing
/// SimtStep programs.
namespace patterns {

/// Returns `tid == laneIdx`.
mlir::Value laneEquals(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Value tid, std::int64_t laneIdx);

/// Returns `(tid & 1) == (even ? 0 : 1)`.
mlir::Value parity(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value tid, bool even);

/// Returns `tid < bound`.
mlir::Value lessThan(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Value tid, std::int64_t bound);

/// Returns `(tid % mod) == rem`.
mlir::Value modEquals(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Value tid, std::int64_t mod, std::int64_t rem);

} // namespace patterns

} // namespace simt::fuzz
