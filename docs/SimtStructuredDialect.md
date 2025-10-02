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
The block owns a region whose entry block always begins with a **mask block
argument** followed by any carried SSA values. The mask argument is the
canonical representation of the active lanes executing the block; lowering and
the interpreter treat it as authoritative and no longer materialise implicit
`simt_step.active_mask` ops. The region’s terminator must be one of the
structured terminators below.

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
interpreter. `mask_pop`/`mask_merge` now appear at reconvergence points to feed
the block’s mask argument before any user operations execute.

### Example lowering (work-in-progress)

```
// Input (simt_step excerpt)
%mask = simt_step.active_mask : i64
%tid = simt_step.dispatch_thread_id : i32
simt_step.return

// After running the `simt-step-to-structured` pass (current prototype)
"simt_struct.block"() ({
^bb0(%mask: i64, %tid: i32):
  "simt_struct.return"() : () -> ()
}) {sym_name = "entry"}

Structured control flow introduces additional blocks. Each block’s first block
argument threads the active mask while subsequent arguments carry SSA values
propagated across the CFG.
```

The lowering now materialises mask threading and block arguments across the CFG
so divergent control flow reconverges through explicit `mask_pop`/`mask_merge`
sequences. Additional work remains to broaden operator coverage, but the block
structure illustrated above reflects the semantics relied upon by the
interpreter.

---

This dialect is the staging ground for the interpreter and the entry point for
backend lowering pipelines.

## Switch Lowering Blueprint

When lowering `simt_step.switch`, create one `simt_struct.block` per case (plus
optional default) and a shared exit block. The header block computes per-case
masks and branches to the first matching case while pushing the mask stack with
`merge_target = @exit`. Each case block then:

- begins with `mask_pop` followed immediately by `mask_merge` using its incoming
  mask argument, recovering the active lanes for that case;
- executes the case body; any `break` emits a `simt_struct.branch` to the exit
  block carrying the current mask;
- if fallthrough is required, pushes the mask again (preserving merge/continue
  metadata) and branches to the next case block using the case mask argument so
  only surviving lanes proceed.

The exit block mirrors loop reconvergence: `mask_pop`, `mask_merge`, and then
`simt_struct.return` or the next enclosing block. Because each branch forwards
its mask operand, the interpreter binds the destination block’s leading mask
argument to the correct participant set (including lanes that fell through from
previous cases).
