# SIMT-Step Switches vs. SPIR-V Fallthrough

## Motivation
SIMT-Step’s structured CFG is designed to drive a semantics-preserving
interpreter. While traditional IRs like SPIR-V bake switch fallthrough decisions
into the structure of the control-flow graph, SIMT-Step leaves those decisions
as SSA values computed by each case. This note explains why the dynamic approach
is better for semantic analysis and how it relates to more static encodings.

## Running Example

```c
switch (laneValue) {
case 0:
  if (laneValue & 1)
    break;          // some lanes exit early
  [[fallthrough]];
case 1:
  accumulate();
  break;
default:
  break;
}
```

### SIMT-Step Lowering (Conceptual)

- Each case is a region ending in `simt_step.yield(payload…, matchSeen,
  fallthroughActive, switchDone)`.
- `matchSeen` indicates whether this case handled the current lane.
- `fallthroughActive` is computed dynamically—e.g. lanes that took the `break`
  will set it to false.
- `switchDone` tracks whether a terminating statement (`break`, `return`, etc.)
  finished the switch for that lane.
- The structured CFG builder turns these into:
  - An entry edge from the switch header to each case block.
  - A conditional edge from a case to the next case when
    `fallthroughActive && !switchDone` is true.
  - An exit edge from the case back to the parent block (merge) carrying the
    yielded payload.
- The decision to fall through vs. exit is per-lane and left as an SSA
  predicate for the terminator.

### SPIR-V-Style Lowering (Simplified)

- Switch lowering declares blocks for each case and fallthrough is encoded by a
  plain forward `OpBranch` from one case to the next.
- The choice of successor is fixed at codegen time; if a case conditionally
  `break`s, the compiler needs extra phis or to restructure the CFG to mimic
  the dynamic behaviour.
- Lanes that exit early require additional bookkeeping because the fallthrough
  chain itself cannot encode “some lanes exit here”.

## Why SIMT-Step Keeps Fallthrough Dynamic

1. **Per-Lane Semantics.** GPU lanes can diverge inside a case. Tracking
   `fallthroughActive` as an SSA value preserves exactly which lanes keep
   running and which exit.
2. **Interpreter Fidelity.** The StructuredInterpreter binds directly to the
   yielded tuple; no need to reverse-engineer static fallthrough assumptions.
3. **Differential Testing.** When fuzzing or comparing semantics engines, the
   tuple explicitly records which cases matched, which fell through, and why a
   lane exits—aiding diagnosis.
4. **Source Reconstruction.** Rehydrating the original HLSL/CUDA structure is
   easier when the builder maintains explicit merge/continue edges and leaves
   fallthrough under SSA control.

## Optimisation Story

- Although the builder emits `simt_struct.cond_branch` for fallthrough edges,
  the conditions are ordinary SSA values. Once optimisation runs prove that
  `fallthroughActive` is constant (e.g. `true` for an unconditionally falling
  case), a simplifier can fold the conditional into a plain branch.
- This mirrors SPIR-V: when structure is indeed static, the optimiser will
  rewrite the CFG to the simpler shape. Until then, the structured IR remains
  accurate for semantics checks.

## Summary

- SPIR-V’s static fallthrough is compact but hides per-lane decisions.
- SIMT-Step carries fallthrough, match, and done flags in SSA so semantics and
  interpreters remain accurate for divergent control flow.
- Dynamic switches enable better semantic validation today, while still
  allowing later passes to canonically simplify the CFG when proofs permit.

