# Dynamic Block Scheduler Plan

This document captures the next steps for giving the CPS interpreter a real
dynamic-block execution model. The goals are:

* Track which lanes are executing each structured region.
* Suspend/resume continuations when control flow splits and reconverges.
* Drive execution through the ready queue instead of the simple for-loop.

Below is a detailed plan, framed around `simt_step.if`; once this works we can
apply the same mechanics to loops and switches.

---

## 1. Extend Interpreter State

### DynamicBlock

Maintain the block-level bookkeeping:

```c++
struct DynamicBlock {
  const mlir::Block *block = nullptr;
  std::uint32_t iteration = 0;       // for loop bodies

  std::uint64_t expectedMask = 0;    // lanes that entered this block
  std::uint64_t activeMask = 0;      // lanes currently executing
  std::uint64_t completedMask = 0;   // lanes that have finished

  llvm::DenseMap<LaneId, StepType> suspendedLanes; // continuations parked on effects
  llvm::DenseMap<LaneId, StepType> nextOp;         // resume points inside the block

  llvm::DenseMap<mlir::Value, SemValue> carriedValues; // payload forwarded by yields
};
```

### LaneContext

Add per-lane fields for control flow:

```c++
struct LaneContext {
  llvm::DenseMap<mlir::Value, SemValue> values;
  bool hasReturned = false;
  std::optional<SemValue> returnValue;
  std::optional<DynamicBlockKey> currentBlock; // which block this lane is executing
  enum class Phase { Running, Waiting, Completed } phase = Phase::Running;
};
```

### MergeStackEntry

Engineer a map from parent block to the set of child block keys it spawned,
plus the mask of lanes expected in each child. Example:

```c++
struct MergeStackEntry {
  DynamicBlockKey parent;
  llvm::SmallVector<DynamicBlockKey, 4> children;
  llvm::SmallVector<std::uint64_t, 4> childMasks;
  std::uint64_t expectedMask = 0;
  std::uint64_t completedMask = 0;
};
```

These entries sit in `WaveContext::mergeStack`, acting like a structured
call stack—when a child block finishes, we pop the corresponding entry and merge
its mask/value data into the parent entry.

---

## 2. Handling `simt_step.if`

### 2.1 Entry

1. **Evaluate the condition per lane.** Use `SimpleSemantics::evaluateValue` to
   fetch a boolean for each active lane.
2. **Compute masks.**
   ```c++
   trueMask = laneMask & lanesWhere(cond == true);
   falseMask = laneMask & lanesWhere(cond == false);
   ```
3. **Create dynamic blocks.** For each non-empty mask:
   ```c++
   DynamicBlockKey key{block*, iteration++};
   auto &child = waveCtx.blocks[key];
   child.block = regionBlock;
   child.expectedMask = mask;
   child.activeMask = mask;
   child.completedMask = 0;
   child.suspendedLanes.clear();
   child.carriedValues.clear();
   ```
4. **Capture the parent continuation.** Before dispatching children, create a
   `Step::continueWith` that resumes the *next* instruction in the parent block.
   Store it in the parent `DynamicBlock` (e.g., `parent.nextOp[lane]`).
5. **Push a merge entry.**
   ```c++
   MergeStackEntry entry;
   entry.parent = parentKey;
   entry.children = {trueKey, falseKey};
   entry.childMasks = {trueMask, falseMask};
   entry.expectedMask = trueMask | falseMask;
   entry.completedMask = 0;
   mergeStack.push_back(entry);
   ```
6. **Enqueue child work.** For each lane bit set in the child mask, create a
   continuation that starts at the child block’s first operation and push it to
   `readyQueue`.

### 2.2 Execution

- Child blocks execute just like the entry block: evaluate op → produce/continue
  step. When a lane reaches `Step::produce` (e.g., from `simt_step.yield`), stash
  the produced value in `LaneContext::returnValue` and in the child block’s
  `carriedValues`.
- For ops that return `Step::suspend` (barriers, collectives), insert the
  continuation into `child.suspendedLanes` and park it in the appropriate
  sync-point table.
- `Step::halt` indicates the lane finished without producing a payload.

### 2.3 Completion & Reconvergence

Whenever a lane finishes (halt or produce) inside a child block:

1. Clear its bit in `child.activeMask` and OR it into `child.completedMask`.
2. Pull the top `MergeStackEntry` whose `parent == currentParent`. Update
   `entry.completedMask |= (1ull << laneId)`.
3. If `child.activeMask == 0`, the entire child block is done:
   - Merge `child.carriedValues` into the parent’s payload map.
   - Pop this child from the merge stack entry (`entry.children`).
4. After updating all children, check whether
   `entry.completedMask == entry.expectedMask`. If so, reconvergence is reached:
   - Pop the merge-stack entry.
   - Re-activate the parent continuation: retrieve the `Step::continueWith`
     stored earlier and push it to `readyQueue` for every lane in
     `entry.expectedMask`.
   - Parent `DynamicBlock`’s `activeMask` becomes `entry.expectedMask` again
     (ready to execute the next instruction). If some lanes never entered the
     branch (bits outside `expectedMask`), they continue from the parent block
     immediately—they were never removed from `activeMask` in the first place.

This ensures the parent only resumes once all participating lanes report back.

---

## 3. Scheduler Integration & Queue

### 3.1 Seed the entry block

```c++
DynamicBlockKey entryKey{&funcBlock, /*iteration=*/0};
auto &entryBlock = waveCtx.blocks[entryKey];
entryBlock.block = &funcBlock;
entryBlock.expectedMask = initialLaneMask;
entryBlock.activeMask = initialLaneMask;

interpreter_.enqueue(/*wave=*/0, entryKey, laneId,
                     buildStepForBlock(entryKey, block.begin()));
```

`buildStepForBlock` returns a `Step` that executes the first op and captures a
continuation to the next operation (using `Step::continueWith`).

### 3.2 Consume the queue

`CPSInterpreter::run()` repeatedly dequeues `ReadyContinuation{wave, blockKey,
lane, step}`. For each step:

1. Mark `laneCtx.currentBlock = blockKey`;
2. Run the `Step`: handle `Continue`, `Produce`, `Halt`, `Suspend` exactly as
   outlined earlier.
3. When the step returns, update `DynamicBlock::activeMask` and the merge
   stack accordingly.

This replaces the old “for each op in block” loop; from now on, the queue is the
sole driver of execution order.

---

## 4. Synchronization Effects (stage 2)

Once the basic reconvergence works, implement effect handlers:

- **CollectiveSyncPoint**: store the effect (`CollectiveEffect`), expected mask,
  arrivals, operands, and continuations. Resume all lanes when `arrivals == expectedMask`.
- **SynchronizationSyncPoint**: same idea for barriers/fences.

When a `Step::suspend(effect, resume)` appears, place the lane into the sync
point; reconvene the parent block only when all lanes in `expectedMask` have
reached the sync point. This matches the “temporal” reconvergence requirement
for independent thread execution.

---

## 5. Test Plan

1. **Branch smoke test** – file `test/simple-interpreter/if_else.mlir` already
   does `dispatch_thread_id == 0 ? yield 1 : yield 0`. After implementing the
   above, ensure lane 0 produces `1`, lane 1 produces `0`. Hook it into CTest
   (`simple_interpreter_if`).
2. **Multiple lanes** – extend the sample to cover more lanes (e.g., mask width 4)
   and check every lane’s `returnValue` matches expectations.
3. **Barrier scenario (optional)** – make a variant where each branch hits
   `simt_step.barrier`. Confirm the parent block only resumes once both branches
   report arrivals.

---

## 6. Next Steps After `if`

Once `simt_step.if` works end-to-end:

1. Apply the same mechanics to `simt_step.loop` (`simt_step.condition`,
   `simt_step.break`, `simt_step.continue`). Each iteration is just another
   dynamic block with its own iteration index.
2. Implement `simt_step.switch` by spawning a block per case, managing fallthrough
   flags similarly to the merge stack.
3. Fill out effect handlers (`CollectiveSyncPoint`, `SynchronizationSyncPoint`)
   so yields, barriers, and collectives cooperate with the dynamic-block
   scheduler.

With these pieces in place, the interpreter can truly run multiple lanes
independently, only reconverging when the semantics dictate (e.g., at a barrier
or at the end of a block where all active lanes have completed).

---

## 3. Scheduler Integration

Replace the current "iterate ops" loop with the CPS queue:

```c++
SimpleProgramRunner::runBlock(block, context) {
  interpreter_.enqueue(/*wave=*/0, blockKey(block), /*lane=*/0,
                       buildContinuationForBlock(block));
  return interpreter_.run();
}
```

Every time `evalOperation` creates a `Step::continueWith`, send it back through
the queue. `Step::suspend` should push the lane into the block’s continuation
map; triggers like barriers will resume it later.

---

## 4. Test Milestones

1. **Branch test** – use the provided module:
   ```mlir
   %tid = dispatch_thread_id
   %cond = cmpi eq %tid, 0
   %result = simt_step.if (%cond) {
     yield 1
   } else {
     yield 0
   }
   ```
   Expected outcome: lane 0 produces 1, others 0.

2. **Ensure CTest still works** – run
   `ctest --test-dir build -R simple_interpreter_if --output-on-failure`.

Once `simt_step.if` works, extend the same approach to `simt_step.loop`
(`simt_step.condition`, `break`, `continue`) and eventually `simt_step.switch`.
Finally wire up effect cases (`Step::suspend` for collectives/barriers).

---

This plan gets us from a linear, single-lane runner to a proper CPS interpreter
with mask-aware control flow, ready for the more advanced semantics work.
