#pragma once

#include <cstdint>

namespace simt::semantics {

bool simtWaveAll(std::uint64_t activeMask, std::uint64_t predicateMask);
std::uint64_t simtBallot(std::uint64_t activeMask, std::uint64_t predicateMask);
void simtBarrier(std::uint32_t scope, std::uint32_t semantics);
void simtFence(std::uint32_t scope, std::uint32_t semantics);
std::uint32_t simtShuffle(std::uint32_t value,
                          std::uint32_t sourceLane,
                          std::uint32_t width,
                          const std::uint32_t *lanes,
                          std::size_t laneCount);

} // namespace simt::semantics
