#include "simt-step/semantics/Runtime.h"

#include <algorithm>

namespace simt::semantics {

bool simtWaveAll(std::uint64_t activeMask, std::uint64_t predicateMask) {
    return (activeMask & ~predicateMask) == 0;
}

std::uint64_t simtBallot(std::uint64_t activeMask, std::uint64_t predicateMask) {
    return activeMask & predicateMask;
}

void simtBarrier(std::uint32_t /*scope*/, std::uint32_t /*semantics*/) {
    // Placeholder: real implementation will synchronize threads at the given scope.
}

void simtFence(std::uint32_t /*scope*/, std::uint32_t /*semantics*/) {
    // Placeholder: real implementation will emit appropriate memory fencing.
}

std::uint32_t simtShuffle(std::uint32_t /*value*/,
                          std::uint32_t sourceLane,
                          std::uint32_t /*width*/,
                          const std::uint32_t *lanes,
                          std::size_t laneCount) {
    if (!lanes || laneCount == 0) {
        return 0;
    }
    const auto clamped = std::min<std::size_t>(sourceLane, laneCount - 1);
    return lanes[clamped];
}

} // namespace simt::semantics
