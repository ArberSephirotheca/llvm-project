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

### 3.3 Implementation Checklist (Step 2)

The transition from the sequential loop to the CPS queue involves several
mechanical steps. Work through them one by one:

1. **Per-op continuations** – add a helper that, given a block iterator and
   lane, returns a `Step::continueWith` wrapping `evalOperation`. The lambda
   should call `evalOperation`, then decide whether to enqueue the next iterator
   (for `Continue`) or return `halt()` when the block ends.

2. **Seed entry block** – in `runBlock`, construct the entry `DynamicBlock`
   (`expectedMask`, `activeMask` seeded with the initial lane mask), build the
   first continuation, enqueue it, and call `CPSInterpreter::run()`.

3. **Unify queue handling** – eliminate any direct use of the sequential
   for-loop; all operation execution should flow through the ready queue. Remove
   the temporary `pendingStep` field in `InterpreterState` once everything is
   queued.

4. **`simt_step.if` split** – replace the current recursive call in
   `handleIfOp` with the mask-splitting logic described in Section 2. Populate
   child blocks, push merge-stack entries, enqueue their continuations, and store
   the parent continuation to be resumed later.

5. **Reconvergence signal** – when a child block finishes a lane, update
   `activeMask`, `completedMask`, and the merge stack. Once all participating
   lanes complete (`completedMask == expectedMask`), pop the merge-stack entry
   and enqueue the stored parent continuation.

6. **Testing** – run the existing branch test and extend it to cover multiple
   lanes once mask splitting is live. Ensure the interpreter still prints the
   expected values via CLI and CTest.

The checklist mirrors the narrative sections above, but makes the step-by-step
implementation explicit so you can follow it without keeping the entire document
in your head.

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

## 7. Loop Scheduling Plan

The final stage is to drive `simt_step.loop` through the CPS scheduler. Loops
must allow different lanes to sit in different iterations independently while
still honoring collectives and reconvergence.

### 7.1 Loop Frame State

Each loop merge entry carries a `LoopFrameState`:

```c++
struct IterationState {
  DynamicBlockKey prepareKey;
  DynamicBlockKey bodyKey;
  std::uint64_t expectedMask;  // conservative participants for this iteration
  std::uint64_t activeMask;    // lanes currently executing it
};

struct LoopFrameState {
  const mlir::Operation *loopOp = nullptr;
  std::uint32_t nextSequenceId = 0;

  llvm::DenseMap<unsigned, IterationState> liveIterations;
  llvm::DenseMap<LaneId, unsigned> laneIteration;
  llvm::DenseMap<LaneId, llvm::SmallVector<SemValue, 4>> carriedTuples;
};
```

`liveIterations[i]` records the prepare/body dynamic block keys for iteration
`i` plus its masks. `laneIteration[lane]` tells us which iteration a lane is in,
and `carriedTuples` stores the loop-carried SSA tuple per lane.

### 7.2 Loop Entry

1. Compute the conservative participant mask
   (`parent.expectedMask ? parent.expectedMask : parent.activeMask`).
2. Create iteration 0: allocate prepare/body keys with fresh sequence IDs, seed
   `expectedMask` with the conservative mask, and set `activeMask` to the lanes
   currently entering.
3. Evaluate the loop inits per lane, store them in `carriedTuples`, and populate
   the prepare block’s per-lane value environment.
4. Push a merge entry containing the `LoopFrameState`, capture the parent
   continuation, enqueue the prepare block for each active lane, and remove those
   lanes from the parent block’s `activeMask`.

Late arrivals reuse iteration 0 if it is still live; otherwise we allocate a new
iteration entry for that cohort.

### 7.3 `simt_step.condition`

When a lane hits the condition terminator:

* If the predicate is false:
  - Remove its bit from the iteration’s `expectedMask`/`activeMask`.
  - Call `shrinkExpectedForLane` so collectives stop waiting for it.
  - Record the forwarded operands as the loop’s SSA results (store them in the
    parent block’s per-lane value environment).
  - Mark completion in the loop merge entry and call `handleReconvergence`.
  - Erase the lane from `laneIteration`/`carriedTuples`.

* If true:
  - Store the forwarded operands in `carriedTuples[lane]`.
  - Ensure the body block for this iteration exists, enqueue the lane at the
    body entry using that iteration’s `bodyKey`, and keep the iteration’s
    `activeMask` up to date.

### 7.4 `simt_step.yield` / `continue`

* Update `carriedTuples[lane]` with the yielded tuple.
* Increment `laneIteration[lane]` to the next iteration index.
* Look up (or create) the next `IterationState`. If we create a new iteration we
  allocate prepare/body keys with `nextSequenceId`.
* Seed the new prepare block’s value environment and enqueue the lane there.

### 7.5 `simt_step.break`

Break acts like “condition false” immediately: shrink expectations, mark
completion, and reconverge the parent continuation for that lane.

### 7.6 Cleanup & Collectives

* When an iteration’s `activeMask` drops to zero, erase it from `liveIterations`.
* When the loop merge entry sees `completedMask == expectedMask`, pop it and
  resume the parent continuation.
* Each iteration has its own `DynamicBlockKey`, so collectives automatically
  scope themselves per iteration. `IterationState.expectedMask` carries the
  conservative set; `shrinkExpectedForLane` removes bits when lanes exit for
  good.

### 7.7 Tests

1. Single-lane loop – verify CPS output matches the old evaluator.
2. Divergent loop – late arrivals should get their own iteration entries and
   collectives should wait appropriately.
3. Loop with a collective – ensure collectives block until every expected lane
   arrives or is shrunk away.
4. Nested loops – push multiple loop frames to verify nesting works.

---

This plan gets us from a linear, single-lane runner to a proper CPS interpreter
with mask-aware control flow, ready for the more advanced semantics work.
