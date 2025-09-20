#pragma once

#include <cstdint>

namespace simt::semantics {

struct SemanticsContext {
    std::uint32_t subgroupWidth = 0;
    std::uint64_t activeMask = 0;
};

} // namespace simt::semantics
