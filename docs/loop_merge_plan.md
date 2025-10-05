# Loop Merge Block Refactor Plan

## Goal
Ensure every `simt_step.loop` lowered by `StructuredCFGBuilder` has a concrete merge block with the correct carried payloads and structured branches, even when the loop is the final op in a block or function.

---

## Stage 1 – Block Splitting Utility
### Objective
Guarantee each loop op has a successor block that will become the structured merge block.

### Tasks
1. **Helper** `splitLoopParentBlock(loopOp, info)`
   - Split `loopOp`’s parent block immediately after the op (`parentBlock->splitBlock(loopOp->getIterator())`).
   - Move trailing ops (e.g., `return`) into the new block. The original block now ends with the loop op only.
2. **BlockInfo creation**
   - Call `getOrCreateBlockInfo` on the new block so it has a `BlockInfo` entry.
   - Insert the new block in `blockOrder` immediately after the original block (preserving emission order).
3. **LoopInfo update**
   - Set `LoopInfo.mergeBlock` to this new block in `analyseLoopOp`.
   - Record the synthetic block in a `LoopInfo::synthesisedMerge` flag for debugging/regression checks.

### Tests
- Loop at end of function that immediately breaks.
- Loop followed by additional code to ensure we do not double-split.

---

## Stage 2 – Merge Block Argument Wiring
### Objective
Align SSA payloads / loop results with the synthetic merge block.

### Tasks
1. **Block args**
   - Ensure the merge block has block arguments that mirror the loop’s result types (copy the carried tuple from the loop header).
2. **SSA remap**
   - Replace all uses of the original `simt_step.loop` results with the new merge block block arguments (update `IRMapping` in analysis).
3. **Payload seeds**
   - Update `propagatePayload` so break edges write into `LoopInfo.mergeBlock->payloadSeed` using the mapped block arguments.
4. **Edge targets**
   - In `enumerateEdges`, route all break-related `EdgeInfo` to `loopInfo.mergeBlock` (synthetic or otherwise).

### Tests
- Break loop with accumulator to see payload forwarded to merge args.
- Continue loop to ensure header payload is unaffected.

---

## Stage 3 – Structured Emission Updates
### Objective
Emit structured blocks/branches against the synthetic merge block.

### Tasks
1. **Structured merge block**
   - `emitStructuredBlock` should create a `simt_struct.block` for the merge block (with block args). No additional logic inside besides straight-line ops + terminator.
2. **Break emission**
   - Loop body `simt_step.break` should emit a `simt_struct.branch` to the merge block, passing the block arguments and using the popped mask.
3. **Cleanup**
   - `cleanupOriginalCFG()` must remove the synthetic block after structured blocks are inserted so the final IR has only structured form.

### Tests
- `loop_control.mlir`: ensures mask pop/merge and branch target are as expected.
- Loop with both break and continue to ensure merged block receives correct payloads.

---

## Stage 4 – Regression Suite
- Loop at end of function (break only).
- Loop at end of function (continue only).
- Loop with both break and continue.
- Nested loop with break to ensure splitting is re-entrant.

Each test should assert mask pushes/pops, structured branch targets, and block arguments via `FileCheck`.

---

## Stage 5 – Follow-up Opportunities
- Apply similar block-splitting + PHI wiring to nested `simt_step.if`/`simt_step.switch` using the plan in `nested_lowering_plan_review.md`.
- Once all nested constructs are rebuilt structurally, delete the remaining control-op cloning paths.

---

## Deliverables
1. Helper utilities and builder changes (Stages 1–3).
2. Regression tests in `test/Conversion/` covering scenarios above.
3. Documentation/log entry describing the new behaviour.

