#include "simt-step/semantics/TraceJsonWriter.h"

#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/JSON.h>

using namespace simt::semantics;

namespace {

static std::string effectName(const Effect &effect) {
    if (effect.isa<BarrierEffect>())
        return "barrier";
    if (effect.isa<YieldEffect>())
        return "yield";
    if (effect.isa<CollectiveEffect>())
        return "collective";
    if (effect.isa<SynchronizationEffect>())
        return "sync";
    if (effect.isa<NondeterministicChoiceEffect>())
        return "nondet";
    return "unknown";
}

} // namespace

TraceJsonWriter::TraceJsonWriter(const std::string &filePath) {
    std::error_code ec;
    os_ = std::make_unique<llvm::raw_fd_ostream>(filePath, ec);
    if (ec)
        llvm::report_fatal_error(
            llvm::Twine("TraceJsonWriter: failed to open ") + filePath + ": " + ec.message());
}

void TraceJsonWriter::writeMask(std::uint64_t mask) {
    (*os_) << "\"0b";
    for (int i = 63; i >= 0; --i)
        (*os_) << ((mask >> i) & 1ull);
    (*os_) << "\"";
}

void TraceJsonWriter::writeBlock(std::uint32_t blockSeq, const void *blockPtr,
                                 const char *blockKind,
                                 std::optional<std::uint32_t> loopIteration) {
    (*os_) << ",\"blockSeq\":" << blockSeq;
    if (blockKind && blockKind[0])
        (*os_) << ",\"blockKind\":\"" << blockKind << "\"";
    if (loopIteration)
        (*os_) << ",\"loopIter\":" << *loopIteration;
    if (blockPtr) {
        auto addr = reinterpret_cast<std::uintptr_t>(blockPtr);
        (*os_) << ",\"blockAddr\":\"0x" << llvm::format_hex_no_prefix(addr, 0)
               << "\"";
    }
}

void TraceJsonWriter::writeEventPrefix(const char *kind, WaveId wave, int lane,
                                       std::uint64_t activeMask,
                                       std::uint64_t expectedMask,
                                       std::uint32_t blockSeq, const void *blockPtr,
                                       const char *blockKind,
                                       std::optional<std::uint32_t> loopIteration) {
    (*os_) << "{\"t\":" << counter_++ << ",\"event\":\"" << kind
           << "\",\"wave\":" << wave << ",\"lane\":" << lane << ",";
    (*os_) << "\"active\":";
    writeMask(activeMask);
    (*os_) << ",\"expected\":";
    writeMask(expectedMask);
    writeBlock(blockSeq, blockPtr, blockKind, loopIteration);
}

void TraceJsonWriter::onStepBegin(WaveId wave, LaneId lane,
                                  const std::string &opName,
                                  std::uint64_t activeMask,
                                  std::uint64_t expectedMask,
                                  std::uint32_t blockSeq, const void *blockPtr,
                                  const char *blockKind,
                                  std::optional<std::uint32_t> loopIteration) {
    if (!os_)
        return;
    writeEventPrefix("step", wave, static_cast<int>(lane), activeMask,
                     expectedMask, blockSeq, blockPtr, blockKind, loopIteration);
    (*os_) << ",\"op\":\"" << opName << "\"}\n";
}

void TraceJsonWriter::onSuspend(WaveId wave, LaneId lane, const Effect &effect,
                                std::uint64_t activeMask,
                                std::uint64_t expectedMask,
                                std::uint32_t blockSeq, const void *blockPtr,
                                const char *blockKind,
                                std::optional<std::uint32_t> loopIteration) {
    if (!os_)
        return;
    writeEventPrefix("suspend", wave, static_cast<int>(lane), activeMask,
                     expectedMask, blockSeq, blockPtr, blockKind, loopIteration);
    (*os_) << ",\"effect\":\"" << effectName(effect) << "\"}\n";
}

void TraceJsonWriter::onResume(WaveId wave, LaneId lane, std::uint64_t activeMask,
                               std::uint64_t expectedMask,
                               std::uint32_t blockSeq, const void *blockPtr,
                               const char *blockKind,
                               std::optional<std::uint32_t> loopIteration) {
    if (!os_)
        return;
    writeEventPrefix("resume", wave, static_cast<int>(lane), activeMask,
                     expectedMask, blockSeq, blockPtr, blockKind, loopIteration);
    (*os_) << "}\n";
}

void TraceJsonWriter::onCollectiveComplete(WaveId wave, const std::string &opName,
                                           std::uint64_t activeMask,
                                           std::uint64_t expectedMask,
                                           std::uint32_t blockSeq,
                                           const void *blockPtr,
                                           const char *blockKind,
                                           std::optional<std::uint32_t> loopIteration) {
    if (!os_)
        return;
    writeEventPrefix("collective", wave, -1, activeMask, expectedMask, blockSeq,
                     blockPtr, blockKind, loopIteration);
    if (!opName.empty())
        (*os_) << ",\"op\":\"" << opName << "\"";
    (*os_) << "}\n";
}

void TraceJsonWriter::onReturn(WaveId wave, LaneId lane, bool hasValue,
                               std::uint64_t activeMask,
                               std::uint64_t expectedMask,
                               std::uint32_t blockSeq, const void *blockPtr,
                               const char *blockKind,
                               std::optional<std::uint32_t> loopIteration) {
    if (!os_)
        return;
    writeEventPrefix("return", wave, static_cast<int>(lane), activeMask,
                     expectedMask, blockSeq, blockPtr, blockKind, loopIteration);
    (*os_) << ",\"hasValue\":" << (hasValue ? "true" : "false") << "}\n";
}
