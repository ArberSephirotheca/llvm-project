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
