# Dialect & Interpreter Workstream

## Immediate Tasks
1. **Create TableGen skeleton**
   - Add `include/simt-step/Dialect/SimtStep.td` defining the dialect (name, namespace, summary).
   - Declare placeholder ops for `wave.ballot`, `barrier`, `simt.custom` with minimal operands/results.
   - List trait enums for `Independent`, `Synchronized`, `Collective`.
2. **CMake integration**
   - Install MLIR tablegen helpers (`AddMLIR`) to generate dialect headers/sources during build.
   - Ensure generated files compile into `libsimt-step` and are available to tools.
3. **Dialect registration C++**
   - Add `lib/core/Dialect.cpp` (or rename existing) to register the MLIR dialect and expose a `registerSimtStepDialect` helper.
   - Include the generated dialect header and link initialization into library startup.
4. **Interpreter bridge scaffolding**
   - Introduce C++ stubs that query op traits (independent/sync/collective) and fall back to plugin metadata.
   - Unit test via `simt-run` with mock ops once dialect is generated.

## Near-Term Follow-Ups
- Flesh out TableGen definitions for remaining built-ins (collectives, state queries, memory ops).
- Implement trait classes and attach them in TableGen.
- Write verifier skeletons in generated C++ for masks and scope attrs.
- Extend lowering pipeline plan to convert `scf.if`/`scf.while` into continuation form for interpreter.

Update this document as milestones land.
