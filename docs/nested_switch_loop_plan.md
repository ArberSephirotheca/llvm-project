# Nested Loop + Switch Operand Normalisation Plan

## Goal
Ensure `lowerSwitchToCFG` stitches operand tuples correctly when a `simt_step.switch`
appears inside loop-carried regions, so every reconstructed `simt_struct.block`
sees the values it expects and deeply nested control flow remains well-defined.

## Approach

1. **Record Required Payloads**
   - Introduce `DenseMap<Block *, SmallVector<Value>> blockPayload` while cloning
     switch cases.
   - For each relevant block (case, fall-through, exit, after-switch) store the
     exact tuple it needs to forward: loop-carried values followed by switch results.

2. **Normalise Branch Operands**
   - For every block whose argument list grows (header, exit, fall-through stubs),
     walk all predecessors.
   - Copy existing operands (loop-carried slice), append the payload slice from
     `blockPayload`, and update the terminator with
     `MutableOperandRange::assign`, letting segment attrs adjust automatically.
   - Assert that the branch either already provided the full tuple or exactly the
     carried values.

3. **Rewrite `switch` Result Uses**
   - Replace the `blockValueForResult` lookups with `blockPayload` queries:
     block-local uses take their recorded value, others fall back to the
     after-block arguments.
   - Keeps one source of truth for both branch operands and SSA rewrites.

4. **Keep Loop Lowering Simple**
   - Leave `lowerLoopToCFG` emitting back-edges with only the loop-carried
     operands; the post-switch normalisation upgrades them later, handling
     arbitrary nesting uniformly.

5. **Add Guards & Debugging**
   - Sprinkle asserts/logging when appending payloads to catch mismatches early.
   - Consider a verifier ensuring every branch into a block supplies the exact
     argument count expected.

6. **Update Tests**
   - Remove `XFAIL` from `loop_switch.mlir` / `switch_loop.mlir`.
   - Extend checks to show the header block receiving the combined operand tuple.
   - Add deeper nesting scenarios to `check-simt-opt` to guard the fix.

## Outcome
With the payload map driving both branch operands and SSA rewrites, the lowering
pipeline stops depending on ad-hoc operand counts and scales across nested
topologies (loop → switch → loop, switch inside loop inside switch, etc.).
