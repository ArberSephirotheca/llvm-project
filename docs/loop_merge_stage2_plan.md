# Stage 2 Design: Loop Merge Block as a PHI Join

## Prerequisites
After Stage 1:
- Every `simt_step.loop` has a dedicated synthetic merge block (`LoopInfo.mergeBlock`).
- The loop body’s `simt_step.break` / `continue` ops are recorded.
- Mask-entry logic is already emitted at loop headers. Stage 2 only adds the missing join semantics.

---

## Goals
1. Replace all uses of the loop’s SSA results with the merge block’s block arguments (PHI semantics).
2. Ensure every predecessor of the merge block (header false path and all `break` edges) forwards the correct payload tuple.
3. Balance the mask stack: every exit to the merge block pops the loop mask before branching, and the merge block merges the mask at entry.

---

## Step-by-Step Plan

### 1. Add Merge Block Arguments & Map Loop Results
- Add block arguments to the merge block—one per loop result (and optionally mask if threaded in SSA).
- Build a mapping `loopResult -> mergeBlockArgument`.
- Replace all uses of `loop.getResult(i)` with `mergeBlock->getArgument(i)`.

### 2. Seed Merge Payloads
- Initialize the merge block’s `payloadSeed` with these new block arguments so structured emission can reuse them.
- Store the list of merge arguments in `LoopInfo.mergeArgs` for later emission.

### 3. Header False Edge
- Ensure the structured header’s false edge is a `simt_struct.cond_branch` that forwards the loop’s carried tuple to the merge block (use `ensurePayloadShape`).
- Pop the mask on that edge (`mask_pop`), then branch to `merge` with the normalized payload.

### 4. Rewrite `simt_step.break`
For each recorded `break`:
1. Build the payload tuple to match the merge block arguments.
2. Pop the loop mask if required.
3. Replace the `simt_step.break` terminator with a `simt_struct.branch` to the merge block, passing the payload tuple.

### 5. Continue Edges
- Continue edges keep branching to the loop’s `continueBlock` / header; no change is needed in Stage 2.

### 6. Mask Discipline
- Header false and each `break` must call `materialiseMaskExit()` before branching to `merge`.
- At the top of the merge block, call `materialiseMaskEntry()` to merge the popped masks back into the loop’s mask argument.

### 7. Erase the Loop Op (later)
- Once downstream code uses only the merge block arguments, the `simt_step.loop` op becomes redundant and can be erased during structured emission cleanup.

---

## Tests Required
1. `loop_break`: loop that immediately breaks.
2. `loop_continue_break`: loop with both `continue` and `break` paths.
3. `loop_with_payload`: loop carrying an accumulator tuple.

Each test should assert:
- The merge block receives the correct number of operands.
- Mask pops/merge appear on break paths and merge entry.
- No `simt_step.loop` remains after lowering.

---

## Notes
- Mirrors glslang’s lowering to SPIR-V (`OpLoopMerge`): loop results become PHI nodes at the merge block, each exit forwards the tuple, and mask state is rebalanced.
- Stage 2 is independent of Stage 3 (structured emission cleanup) and sets the groundwork for removing the original CFG blocks.

