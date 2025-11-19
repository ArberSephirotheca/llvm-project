# Mask Flow in CPS Interpreter (Current Prototype)

This note documents how lane masks are propagated and updated through the CPS interpreter with non-collective branching and conservative expected masks. The goal is to let lanes progress independently while avoiding early completion of collectives.

## State Summary
- **DynamicBlock**
  - `expectedMask`: conservative set of lanes that could participate in this block (seeded from the parent’s `expectedMask` ∩ branch mask).
  - `activeMask`: lanes currently executing ops in this block.
  - `completedMask`: lanes that have finished this block.
- **MergeStackEntry** (per wave)
  - `parent`: parent block key.
  - `pendingChildren`: child block keys spawned by the split.
  - `childMasks`: masks used to spawn children.
  - `expectedMask`: union of child conservative expected masks for this split.
  - `completedMask`: lanes that have reported back to the parent for this split.
- **Sync points (collectives/barriers)**
  - `expectedMask`: lanes this effect is waiting on (conservative; typically block’s `activeMask` at the effect or the effect’s mask).
  - `arrivals`: lanes that have arrived at the effect.
  - Parked `continuations` to resume when ready.
- **LaneContext**
  - Per-lane SSA/env state, `currentBlock`, `hasReturned`, etc.

## Entry
- Caller provides `SemanticsContext.activeMask` (default: 32 bits set). Entry block `expectedMask` and `activeMask` are set to this mask.
- For each set bit, initialize `LaneContext` and enqueue the first continuation.

## Per-op Execution
- Before evaluating an op, `SemanticsContext.laneId` is set to the lane, and `SemanticsContext.activeMask` is taken from the current block’s `activeMask` (wave’s `currentMask` mirrors it). Builtins see the block’s live participants.

## Branch Split (`simt_step.if`, non-collective)
1. Evaluate the condition for lanes currently in the parent block’s `activeMask` that have reached the branch.
2. Compute `trueMask` / `falseMask` from those evaluations.
3. Create child `DynamicBlock`s:
   - `expectedMask = (parent.expectedMask ? parent.expectedMask : parent.activeMask) ∩ branchMask` (conservative potential participants).
   - `activeMask =` the lanes actually dispatched now (true/false masks respectively).
   - `completedMask = 0`.
4. Push a `MergeStackEntry` with `parent`, `pendingChildren`, and `expectedMask = childTrueExpected | childFalseExpected`.
5. Parent masks: clear only dispatched lanes from `parent.activeMask` (`&= ~(trueMask | falseMask)`). Lanes not dispatched stay active in the parent and keep executing parent ops; they may hit the branch later.
6. Enqueue child continuations per lane in `trueMask`/`falseMask`. Store the parent continuation to resume after child completion.

## Reconvergence
- When a child lane produces/halts:
  - Clear its bit from the child block’s `activeMask`, set it in `completedMask`.
  - Update the merge entry `completedMask` with that bit.
  - Re-add this lane’s bit to the parent block’s `activeMask` and enqueue the stored parent continuation for this lane immediately (per-lane resume; no collective wait).
  - If `completedMask == expectedMask` for the merge entry (after any expected-mask shrink), pop the merge entry.

## Expected Mask Shrinking
- `expectedMask` is conservative. When a lane definitively takes another path or finishes without entering a child, clear its bit from any `expectedMask` that was waiting on it (child blocks, merge entries, sync points). This keeps collectives/reconvergence from waiting on lanes that will never arrive.

## Collectives/Barriers
- On `Step::suspend(CollectiveEffect/SynchronizationEffect)`:
  - Create/update a sync point keyed by block+token with `expectedMask = effect.activeMask` or a conservative mask (e.g., block’s `activeMask` at the effect if unspecified).
  - Record arrival and park the wrapped continuation.
  - When `arrivals` cover `expectedMask` (after any expected-mask shrink), restore those lane bits to the block’s `activeMask` and re-enqueue the parked continuations; remove the sync point.

## Example: Nested If (lanes 0–3 active, non-collective split)
- Outer split: lanes 0–1 take THEN, 2–3 take ELSE. Parent clears dispatched lanes; if other lanes existed they’d stay active in the parent. Child `expectedMask` is seeded from parent’s `expectedMask`. Push outer merge entry (`expectedMask = union of child expected masks`).
- Inner split inside THEN: lanes 0→inner THEN, 1→inner ELSE. Clear only dispatched lanes from the outer-THEN `activeMask`; push inner merge entry. Parent THEN resumes per lane as child paths finish.
- Reconvergence: each lane rejoins its parent block as soon as its child path completes; merge entries pop when their `completedMask` matches their (shrunk) `expectedMask`.

This flow keeps branches independent: only dispatched lanes leave the parent, parent work continues for other lanes, and child collectives/reconvergence wait on conservative expected masks that are shrunk as lanes resolve elsewhere.
