#pragma once

#include "simt-step/semantics/Effects.h"

#include <cstdint>
#include <string>

namespace simt::semantics {

using WaveId = std::uint32_t;
using LaneId = std::uint32_t;

/// Sink interface for interpreter trace events.
class TraceSink {
public:
    virtual ~TraceSink() = default;

    virtual void onStepBegin(WaveId wave, LaneId lane, const std::string &opName,
                             std::uint64_t activeMask, std::uint64_t expectedMask,
                             std::uint32_t blockSeq, const void *blockPtr,
                             const char *blockKind) = 0;

    virtual void onSuspend(WaveId wave, LaneId lane, const Effect &effect,
                           std::uint64_t activeMask, std::uint64_t expectedMask,
                           std::uint32_t blockSeq, const void *blockPtr,
                           const char *blockKind) = 0;

    virtual void onResume(WaveId wave, LaneId lane, std::uint64_t activeMask,
                          std::uint64_t expectedMask, std::uint32_t blockSeq,
                          const void *blockPtr, const char *blockKind) = 0;

    virtual void onReturn(WaveId wave, LaneId lane, bool hasValue,
                          std::uint64_t activeMask, std::uint64_t expectedMask,
                          std::uint32_t blockSeq, const void *blockPtr,
                          const char *blockKind) = 0;
};

} // namespace simt::semantics
