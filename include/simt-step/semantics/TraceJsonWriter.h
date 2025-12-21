#pragma once

#include "simt-step/semantics/Trace.h"

#include <llvm/Support/raw_ostream.h>

#include <memory>
#include <string>

namespace simt::semantics {

/// Simple JSONL trace writer for interpreter events.
class TraceJsonWriter : public TraceSink {
public:
    explicit TraceJsonWriter(const std::string &filePath);
    ~TraceJsonWriter() override = default;

    bool isOpen() const { return static_cast<bool>(os_); }

    void onStepBegin(WaveId wave, LaneId lane, const std::string &opName,
                     std::uint64_t activeMask, std::uint64_t expectedMask,
                     std::uint32_t blockSeq, const void *blockPtr,
                     const char *blockKind) override;

    void onSuspend(WaveId wave, LaneId lane, const Effect &effect,
                   std::uint64_t activeMask, std::uint64_t expectedMask,
                   std::uint32_t blockSeq, const void *blockPtr,
                   const char *blockKind) override;

    void onResume(WaveId wave, LaneId lane, std::uint64_t activeMask,
                  std::uint64_t expectedMask, std::uint32_t blockSeq,
                  const void *blockPtr, const char *blockKind) override;

    void onCollectiveComplete(WaveId wave, const std::string &opName,
                              std::uint64_t activeMask,
                              std::uint64_t expectedMask,
                              std::uint32_t blockSeq, const void *blockPtr,
                              const char *blockKind) override;

    void onReturn(WaveId wave, LaneId lane, bool hasValue,
                  std::uint64_t activeMask, std::uint64_t expectedMask,
                  std::uint32_t blockSeq, const void *blockPtr,
                  const char *blockKind) override;

private:
    void writeMask(std::uint64_t mask);
    void writeBlock(std::uint32_t blockSeq, const void *blockPtr,
                    const char *blockKind);
    void writeEventPrefix(const char *kind, WaveId wave, int lane,
                          std::uint64_t activeMask, std::uint64_t expectedMask,
                          std::uint32_t blockSeq, const void *blockPtr,
                          const char *blockKind);

    std::unique_ptr<llvm::raw_fd_ostream> os_;
    std::uint64_t counter_ = 0;
};

} // namespace simt::semantics
