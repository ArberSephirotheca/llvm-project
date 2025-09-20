#pragma once

#include <cstdint>

namespace simt::core::passes {

struct SpecializeConfig {
    std::uint32_t subgroupWidth = 32;
    std::uint64_t activeMask = ~0ULL;
};

} // namespace simt::core::passes
