# Test Workflow For SIMT-Step Improvements

When landing new `simt-opt` regression tests that exercise hand-crafted
`simt_step` programs, always pair them with a frontend-generated test so the
input IR stays realistic.

- Add / update the `simt-opt` FileCheck case under `test/Conversion/`.
- Mirror the scenario in `tools/simt-hlsl-import/test/` with an HLSL snippet and
  its corresponding `*.mlir` FileCheck file.
- Run both `build/tools/simt-opt/simt-opt … | FileCheck …` and the full
  `cmake --build build --target check-simt-hlsl-import` target before shipping.

This keeps the structured lowering in sync with real importer output and avoids
regressions when the frontend evolves.

## Design Reference: glslang Nested Control Lowering

- glslang walks a structured AST (nodes like `TIntermIf`, `TIntermFor`, `TIntermSwitch`).
- During SPIR-V emission the traverser pushes merge/continue targets, emits the
  structured op (`OpSelectionMerge`, `OpLoopMerge`, `OpSwitch`), and recursively
  lowers nested regions before returning to the parent scope.
- Payload joining is handled at merge blocks with `OpPhi` creation, so nested
  branches/loops naturally reconcile their carried values.
- Auxiliary blocks (merge, continue) are created on the fly to satisfy SPIR-V’s
  single-entry/single-exit rules; no cloning fallback is used for nested
  control.

We’re mirroring that approach in `StructuredCFGBuilder`: analysis preserves the
structured regions, `ensureLoopMergeBlock`/`ensurePayloadShape` pre-create merge
blocks and PHI tuples, and `emitStructuredBlock` recurses into nested control so
structured emission stays 1:1 with the source hierarchy.
