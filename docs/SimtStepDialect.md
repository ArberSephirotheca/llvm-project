# SIMT-Step Dialect Reference

## Types

### `simt_step.mask`
Bitmask type encoding active lanes. Syntax:

```
simt_step.mask           // defaults to <64>
simt_step.mask<32>       // explicit bit width
```

Width must be non-zero. The type will be extended with mask-specific utilities as the interpreter matures.

## Attributes

- `simt_step.scope`: enum (`Thread`, `Subgroup`, `Workgroup`)
- `simt_step.memsem`: enum (`None`, `Acquire`, `Release`, `AcqRel`)
- `simt_step.memspace`: enum (`Generic`, `Global`, `Shared`, `Private`)

## Operations

### Wave collectives
- `simt_step.wave_all` – returns an `i1` indicating whether all active lanes satisfy the predicate.
- `simt_step.wave_any` – returns an `i1` indicating whether any active lane satisfies the predicate.
- `simt_step.wave_ballot` – produces an integer mask (`i64`) identifying lanes with `true` predicate values. Future revisions will return `simt_step.mask`.

### Mask management (lowering/internal form)
- `simt_step.mask_push` – pushes an explicit mask (`i64`) onto the execution stack.
- `simt_step.mask_pop` – pops the current mask, restoring the previous value.
- `simt_step.mask_merge` – merges an incoming mask with the current active mask and returns the merged value.

### Synchronization & state
- `simt_step.barrier` – synchronization barrier with optional `scope` / `memsem` enums.
- `simt_step.fence` – memory-ordering fence (attributes optional for now).
- `simt_step.dispatch_thread_id` – yields the dispatch-space thread id; the result
  type matches the builtin usage (scalar or vector).
- `simt_step.group_thread_id` – yields the workgroup-local thread id in the form
  requested by the builtin (scalar or vector).
- `simt_step.group_id` – returns the workgroup identifier within the dispatch
  grid (typically a three-component vector).
- `simt_step.group_index` – returns the flattened workgroup index (scalar).
- `simt_step.lane_id` – yields the current lane identifier as `index`.
- `simt_step.active_mask` – exposes the current active mask as `i64` (will become `simt_step.mask`).

### Memory (placeholders)
- `simt_step.mem_load` – load from the supplied address operand, yielding a value of arbitrary type.
- `simt_step.mem_store` – store the given value to the supplied address.
- `simt_step.buffer.load` / `simt_step.buffer.store` – typed buffer resource access with explicit element indices.
- `simt_step.buffer.atomic_add` – atomic add returning the previous element value.
- `simt_step.buffer.atomic_exchange` – atomic exchange returning the previous element value.
- `simt_step.buffer.atomic_compare_exchange` – compare-and-exchange on a buffer element, returning the previous value.
- `simt_step.buffer.atomic_min` / `.max` – atomic min/max updates returning the previous element value.
- `simt_step.buffer.atomic_and` / `.or` / `.xor` – atomic bitwise operations returning the previous element value.

### Extension hook
- `simt_step.custom` – plugin-defined instruction. Requires an `instr` string attribute and optional `params` dictionary; operands/results are variadic and unconstrained.

### Structured control flow
- `simt_step.switch` – structured multi-way branch. Operands are the selector followed by the initial values for each carried result. The op yields the carried values plus three bookkeeping flags `(matchSeen, fallthroughActive, switchDone)` that track fallthrough and termination state. The `case_values` dense array attribute records the integral selector values associated with each case; the final entry corresponds to the default arm. The region now contains one block per case (plus optional default) and each block terminates with `simt_step.yield`, making the control-flow structure explicit.

---

This reference is a living document; update it when new ops/attributes/types land.
