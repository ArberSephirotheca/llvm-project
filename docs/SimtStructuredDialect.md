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
Represents a named SIMT block:
```
simt_struct.block @label { ... }
```
Attributes:
- `merge_target` (optional) – explicit reconvergence block.
- `continue_target` (optional) – loop continue block.
- `reconvergence` (optional) – overrides block-level policy.
The block owns a single region whose terminator must be one of the structured
terminators below.

### Terminators
- `simt_struct.branch %mask to @dest` – unconditional transfer with mask.
- `simt_struct.cond_branch %cond ? %tmask -> @t : %fmask -> @f` – conditional
  branch with explicit masks and optional `merge_target`/`reconvergence` attrs.
- `simt_struct.return` – exits the structured region.

### Mask stack helpers
- `simt_struct.mask_push %mask` – pushes a mask (and optional merge/continue
  targets) onto the merge stack.
- `simt_struct.mask_pop -> %mask` – pops the next mask from the stack.
- `simt_struct.mask_merge %incoming -> %merged` – merges the incoming mask with
  the active mask according to interpreter policy.

These operations correspond to the merge-stack management performed by the HLSL
interpreter.

---

This dialect is the staging ground for the interpreter and the entry point for
backend lowering pipelines.
