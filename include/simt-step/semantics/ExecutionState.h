#pragma once

#include "simt-step/semantics/Effects.h"
#include "simt-step/semantics/SemValue.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <queue>
#include <utility>

#include <mlir/IR/Block.h>
#include <mlir/IR/Value.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>

namespace simt::semantics {

using LaneId = std::uint32_t;
using WaveId = std::uint32_t;

template <typename ValueT>
class Step;

struct DynamicBlockKey {
    const mlir::Block *block = nullptr;
    std::uint32_t sequenceId = 0;

    friend bool operator==(const DynamicBlockKey &lhs, const DynamicBlockKey &rhs) {
        return lhs.block == rhs.block && lhs.sequenceId == rhs.sequenceId;
    }
};

enum class DynamicBlockKind {
    Plain,
    IfThen,
    IfElse,
    SwitchCase,
    SwitchDefault,
};

template <typename ValueT, typename StepT>
struct DynamicBlock {
    const mlir::Block *block = nullptr;
    std::uint32_t sequenceId = 0;

    std::uint64_t expectedMask = 0;
    std::uint64_t activeMask = 0;
    std::uint64_t completedMask = 0;

    std::optional<DynamicBlockKey> parentKey;
    const mlir::Operation *loopOp = nullptr;
    const mlir::Operation *switchOp = nullptr;
    const mlir::Operation *ifOp = nullptr;
    bool isLoopPrepare = false;
    bool isLoopBody = false;

    DynamicBlockKind kind = DynamicBlockKind::Plain;

    llvm::DenseMap<LaneId, StepT> continuations;
    llvm::DenseMap<LaneId, StepT> pendingOps;
    llvm::DenseMap<LaneId, llvm::DenseMap<mlir::Value, ValueT>> valueEnvs;
};

template <typename ValueT>
struct LoopFrameState {
    const mlir::Operation *loopOp = nullptr;
    DynamicBlockKey prepareKey;
    DynamicBlockKey bodyKey;
    // Per-lane next sequence id for the next iteration prep/body.
    llvm::DenseMap<LaneId, std::uint32_t> laneNextSeq;
    llvm::DenseMap<LaneId, llvm::SmallVector<ValueT, 4>> carried;
};

template <typename ValueT>
struct SwitchFrameState {
    const mlir::Operation *switchOp = nullptr;
    std::uint32_t baseSeq = 0;
    llvm::SmallVector<const mlir::Block *, 4> caseBlocks;
    llvm::DenseMap<LaneId, llvm::SmallVector<ValueT, 8>> carried;
    llvm::DenseMap<LaneId, DynamicBlockKey> pendingCases;
};

template <typename ValueT, typename StepT>
struct CollectiveSyncPoint {
    CollectiveEffect effect;
    DynamicBlockKey block;
    std::uint64_t expectedMask = 0;
    llvm::DenseSet<LaneId> arrivals;
    llvm::DenseMap<LaneId, ValueT> operands;
    llvm::DenseMap<LaneId, StepT> continuations;
};

template <typename ValueT, typename StepT>
struct SynchronizationSyncPoint {
    SynchronizationEffect effect;
    DynamicBlockKey block;
    std::uint64_t expectedMask = 0;
    llvm::DenseSet<LaneId> arrivals;
    llvm::DenseMap<LaneId, StepT> continuations;
};

template <typename ValueT, typename StepT>
struct LaneContext {
    llvm::DenseMap<mlir::Value, ValueT> values;
    bool hasReturned = false;
    std::optional<ValueT> returnValue;
    std::optional<DynamicBlockKey> currentBlock;
    enum class Phase { Running, Waiting, Completed } phase = Phase::Running;
};

template <typename ValueT, typename StepT>
struct MergeStackEntry {
    DynamicBlockKey parent;
    llvm::SmallVector<DynamicBlockKey, 4> pendingChildren;
    llvm::SmallVector<std::uint64_t, 4> childMasks;
    std::uint64_t expectedMask = 0;
    std::uint64_t completedMask = 0;
    std::optional<LoopFrameState<ValueT>> loopFrame;
    std::optional<SwitchFrameState<ValueT>> switchFrame;
};

template <typename ValueT, typename StepT>
struct WaveContext {
    std::uint64_t currentMask = 0;
    llvm::DenseMap<DynamicBlockKey, DynamicBlock<ValueT, StepT>> blocks;
    llvm::SmallVector<MergeStackEntry<ValueT, StepT>, 8> mergeStack;
    llvm::DenseMap<std::uint32_t, CollectiveSyncPoint<ValueT, StepT>> collectives;
    llvm::DenseMap<std::uint32_t, SynchronizationSyncPoint<ValueT, StepT>> syncPoints;
    llvm::DenseMap<LaneId, LaneContext<ValueT, StepT>> lanes;
};

template <typename ValueT, typename StepT>
struct ReadyContinuation {
    WaveId wave = 0;
    DynamicBlockKey block;
    LaneId lane = 0;
    StepT resume;
};

template <typename ValueT, typename StepT>
struct InterpreterState {
    llvm::DenseMap<WaveId, WaveContext<ValueT, StepT>> waves;
    std::queue<ReadyContinuation<ValueT, StepT>> readyQueue;
    StepT pendingStep;
};

using DefaultValue = SemValue;
using DefaultStep = Step<SemValue>;
using DefaultInterpreterState = InterpreterState<DefaultValue, DefaultStep>;

} // namespace simt::semantics

namespace llvm {

template <>
struct DenseMapInfo<simt::semantics::DynamicBlockKey> {
    using Key = simt::semantics::DynamicBlockKey;

    static Key getEmptyKey() {
        static auto *empty = reinterpret_cast<const mlir::Block *>(-1);
        return {empty, std::numeric_limits<std::uint32_t>::max()};
    }

    static Key getTombstoneKey() {
        static auto *tombstone = reinterpret_cast<const mlir::Block *>(-2);
        return {tombstone, std::numeric_limits<std::uint32_t>::max()};
    }

    static unsigned getHashValue(const Key &key) {
        return static_cast<unsigned>(reinterpret_cast<std::uintptr_t>(key.block)) ^
               (key.sequenceId * 37u + 0x9e3779b9u);
    }

    static bool isEqual(const Key &lhs, const Key &rhs) {
        return lhs == rhs;
    }
};

} // namespace llvm
