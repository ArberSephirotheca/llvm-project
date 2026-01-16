#pragma once

#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"
#include "simt-step/semantics/CPSInterpreter.h"
#include "simt-step/semantics/ExecutionState.h"
#include "simt-step/semantics/SimpleSemantics.h"
#include "simt-step/semantics/Trace.h"

#include <mlir/IR/Block.h>
#include <mlir/Support/LogicalResult.h>

#include <cstdint>
#include <optional>
#include <vector>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>

namespace mlir {
class Operation;
} // namespace mlir

namespace simt::semantics {

/// Drives SimpleSemantics over a straight-line simt_step block.
class SimpleProgramRunner {
public:
    using ValueType = SemValue;
    using StepType = Step<ValueType>;
    using StateType = DefaultInterpreterState;
    using ScheduleMode = CPSInterpreter<SimpleSemantics>::ScheduleMode;

    SimpleProgramRunner() : semantics_(), interpreter_(semantics_) {}

    void setTraceSink(TraceSink *sink) { interpreter_.setTraceSink(sink); }
    void setScheduleMode(ScheduleMode mode) { interpreter_.setScheduleMode(mode); }
    void setScheduleSeed(std::uint64_t seed) { interpreter_.setScheduleSeed(seed); }

    llvm::Error runBlock(mlir::Block *block,
                         SemanticsContext context = SemanticsContext{});

    const StateType &state() const { return interpreter_.state(); }

private:
    StepType buildStepForIterator(WaveId wave,
                                  const DynamicBlockKey &key,
                                  mlir::Block *block,
                                  mlir::Block::iterator it,
                                  SemanticsContext context,
                                  LaneId lane);

    SimpleSemantics semantics_;
    CPSInterpreter<SimpleSemantics> interpreter_;
    [[maybe_unused]] bool enableLoopDispatch_ = false;
};

struct BufferInitEntry {
    unsigned argIndex = 0;
    int64_t index = 0;
    int64_t value = 0;
};

struct BufferResult {
    unsigned argIndex = 0;
    std::vector<int64_t> values;
};

struct BufferOptions {
    unsigned argIndex = 0;
    std::optional<int64_t> size;
    std::optional<int64_t> fill;
};

struct RunOperationOptions {
    llvm::StringRef entry = "main";
    unsigned lanes = 4;
    unsigned subgroupWidth = 8;
    int64_t bufferSize = 0;  // 0 = infer from written entries
    int64_t fillValue = 0;   // default fill when size is specified
    std::vector<BufferOptions> perBuffer;
    const ExecutionPolicy *policy = nullptr;
    TraceSink *trace = nullptr;
};

/// Run a module or function and extract a resource buffer by argument index.
mlir::LogicalResult runOperationToBuffer(
    mlir::Operation &op,
    unsigned bufferArgIndex,
    std::vector<int64_t> &buffer,
    const RunOperationOptions &options = {},
    llvm::ArrayRef<BufferInitEntry> initEntries = {});

/// Run a module or function and extract resource buffers by argument index.
/// When bufferArgIndices is empty, all resource arguments are captured.
mlir::LogicalResult runOperationToBuffers(
    mlir::Operation &op,
    llvm::ArrayRef<unsigned> bufferArgIndices,
    std::vector<BufferResult> &buffers,
    const RunOperationOptions &options = {},
    llvm::ArrayRef<BufferInitEntry> initEntries = {});

} // namespace simt::semantics
