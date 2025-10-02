# Structured SIMT Dialect Reference

The `simt_struct` dialect captures SIMT control flow after lowering from the
high-level `simt_step` dialect. It mirrors the dynamic-block semantics described
in the MiniHLSL interpreter and supports configurable reconvergence policies.

## Attributes
- `simt_struct.reconvergence` – enum (`PostDominator`, `Explicit`, `None`) guiding
  how lanes reconverge when execution leaves a divergent region.
- `merge_target` / `continue_target` – optional `FlatSymbolRefAttr` attributes
  pointing at blocks used for reconvergence or loop continuation. When omitted,
  the default policy applies (post-dominator-style reconvergence).

## Operations

### `simt_struct.block`
```
simt_struct.block %label {
  ...
}
```
Represents a structured SIMT block. Attributes:
- `merge_target` (optional) – explicit reconvergence block.
- `continue_target` (optional) – loop continue block.
- `reconvergence` (optional) – overrides block-level policy.
The block owns a region whose terminator must be one of the structured
terminators below.

### Structured terminators
- `simt_struct.branch ^dest (%mask : i64)` – unconditional transfer with mask.
- `simt_struct.cond_branch %cond, ^t (%tmask : i64), ^f (%fmask : i64)` –
  conditional branch with explicit masks and optional `merge_target`/
  `reconvergence` attributes.
- `simt_struct.return` – exits the structured region.
These terminators implement `BranchOpInterface` so generic MLIR control-flow
analyses can operate on them.

### Mask stack helpers
- `simt_struct.mask_push %mask {merge_target = @L}` – pushes a mask (and optional
  merge/continue targets) onto the merge stack.
- `simt_struct.mask_pop -> %mask` – pops the next mask from the stack.
- `simt_struct.mask_merge %incoming -> %merged` – merges the incoming mask with
  the active mask; both operands must share the same type.

These operations correspond to the merge-stack management performed by the HLSL
interpreter.

### Example lowering (work-in-progress)

```
// Input (simt_step excerpt)
%mask = simt_step.active_mask : i64
%tid = simt_step.dispatch_thread_id : i32
simt_step.return

// After running the `simt-step-to-structured` pass (current prototype)
simt_struct.block @entry {
  %mask = simt_step.active_mask : i64
  %tid = simt_step.dispatch_thread_id : i32
  simt_struct.return
}
```

The pass currently handles straight-line, void functions and preserves the
original `simt_step` operations inside a single structured block. Divergent
control flow and value-returning functions remain to be lowered.

---

This dialect is the staging ground for the interpreter and the entry point for
backend lowering pipelines.
