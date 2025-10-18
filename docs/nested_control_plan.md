# Nested Structured Control Plan

## Goals
- Finish recursive lowering so nested `simt_step.if` / `switch` / `loop` constructs emit `simt_struct.*` ops without cloning.
- Ensure payload tuples and mask state converge correctly across nested merges.
- Keep importer and conversion tests in lockstep so real HLSL inputs exercise the new behaviour.

## Step 1 – Stabilise Analysis & Payload Seeds
1. Audit `computePayloads` / `enumerateEdges` / `stabilisePayloadSeeds` to confirm each nested region records:
   - Merge / continue block info
   - Initial payload seeds (block args, loop carried values, switch bundles)
   - Edge lists with payloads per branch
2. Add targeted tests that expose gaps:
   - `if` inside `loop` with distinct values per branch
   - `switch` inside `if` (including fallthrough)
   - Nested loops
   - Mirror each scenario with HLSL importer tests.

## Step 2 – Recursive Emission Helpers
Implement dedicated helpers:
- `emitStructuredIf(BlockInfo &header, IfInfo &info)`
- `emitStructuredSwitch(BlockInfo &parent, SwitchInfo &info)`
- `emitStructuredLoop(BlockInfo &header, LoopInfo &info)`

Each helper should:
- Reuse pre-created `BlockInfo` entries (no cloning fallback).
- Emit nested `simt_struct.block` ops for subregions, setting merge/continue attrs via `BlockInfo`.
- Use `EdgeInfo` payloads to build `branch` / `cond_branch` terminators.
- Call back into `emitStructuredBlock` recursively for contained blocks.

Wire these helpers into `emitStructuredBlock` so structured emission becomes purely recursive.

## Step 3 – Payload Convergence Across Nesting
- Update `stabilisePayloadSeeds()` to iterate until nested merge blocks stabilise their payload tuples (e.g., loop body branches feeding distinct values into the header and switch merge).
- Ensure loop/switch/if payload seeds are updated when nested edges introduce new operands.
- Validate mask stack handling for nested exits (`materialiseMaskEntry/Exit`).

## Step 4 – Regression Cover & Verification
- Extend `test/Conversion/` with nested combinations, each mirrored in `tools/simt-hlsl-import/test/`.
- For every scenario, run:
  - `build/tools/simt-opt/simt-opt … | FileCheck …`
  - `cmake --build build --target check-simt-hlsl-import`
- Update documentation / TODO lists to reflect the new coverage.

