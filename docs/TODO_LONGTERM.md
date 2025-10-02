# SIMT-Step Extended Roadmap

## Current Status (July 2025)
- [x] Structured control-flow scaffolding for `simt_step` landed (if/loop/yield, break/continue) with TableGen + C++ plumbing.
- [x] HLSL importer now targets Clang’s frontend directly, propagates DXIL shader model triples, and seeds SSA-aware value merging for conditionals and `for` loops.
- [x] Build tooling wires in a fallback search path for the upstream Clang HLSL headers so developer builds remain hermetic.
- [ ] Remaining loop forms (`while`/`do-while`) plus `break` / `continue` lowering still pending; block argument merging exists only for `for` today.
- [ ] Regression coverage extends to `if` and `for`; add lit tests for other loop shapes, break/continue, and mask semantics as they land.

## Dialect Foundations
- [x] Flesh out TableGen definitions for core SIMT ops (mask modifiers, synchronizations, collectives, memory ops, state queries, `simt.custom`).
- [x] Implement custom attributes/types (scope, memory semantics, memory space enums, mask type) in C++.
- [x] Add trait classes tying MLIR traits to registry metadata, expose helpers and verification hooks.
- [x] Write per-op/type verifiers (barrier/custom ops, mask type) and diagnostic messages.
- [x] Create dialect documentation (TableGen + `docs/SimtStepDialect.md`).

## Structured SIMT Dialect
- [x] Design a `simt_struct` dialect modeling block-based execution (labels, merges, mask operands, terminators).
- [x] Implement TableGen + C++ scaffolding for block ops, branch ops, merge metadata, loop constructs.
- [x] Ensure compatibility with MLIR’s ControlFlow / BranchOpInterface for analyses.
- [x] Document how this dialect corresponds to SPIR-V style structured CFG and references the MiniHLSL interpreter.
- [x] Expose reconvergence policy (post-dominator, explicit merge, none, etc.) as attributes and pass options.
- [x] Provide mask/mask-stack ops to align with the HLSL interpreter model.

## Lowering Passes
- [ ] Implement `SimtToStructuredPass` converting `simt_step` regions into `simt_struct` blocks (basic straight-line lowering exists; extend to divergent control flow, carried values, and loop/switch metadata).
- [ ] Implement `StructuredToDynamicPass` that materializes dynamic block frames/mask stack used by interpreter.
- [ ] Provide canonicalization and optimization passes on both dialects (fold redundant mask ops, dead blocks, constant propagation with masks).
- [ ] Create pass pipelines (e.g., `simt-interpreter-init`, `simt-llvm-lowering`).

## Interpreter & Runtime
- [ ] Refactor interpreter to consume `simt_struct` or dynamic block IR (iterate blocks, manage mask stack).
- [ ] Integrate trait queries for built-ins and `simt.custom` handlers (independent/sync/collective behavior).
- [ ] Extend runtime intrinsics to cover all collectives and barriers, ensure parity with dialect semantics.
- [ ] Add unit/integration tests verifying interpreter output vs. expected results (reduce, ballot, barrier scenarios).

## Frontends & Tooling
- [ ] Port CUDA/HLSL importers to produce realistic `simt_step` ops (parse actual intrinsics, generate attributes).
- [ ] Expand `simt-convert` to emit MLIR text in different stages (`--emit simt-step`, `--emit structured`, `--emit dynamic`).
- [ ] Enhance `simt-run` to load MLIR files, run pass pipelines, and present execution traces (mask evolution, block order).
- [ ] Create sample kernels covering divergence, synchronization, collectives; host them under `examples/` with scripts.

## Plugin Ecosystem
- [ ] Define plugin metadata schema (traits, resources, verification rules) and enforce it in registry/dialect.
- [ ] Provide plugin author toolkit: headers, registration macros, documentation.
- [ ] Implement additional example plugins (e.g., prefix sum, custom barrier), with tests and docs.
- [ ] Support dynamic loading of plugin libraries and runtime registration.

## LLVM/Backend Lowering
- [ ] Develop lowering from `simt_step` and/or `simt_struct` to LLVM dialect (map collectives to intrinsics, synchronize to `llvm.nvvm.barrier0`, etc.).
- [ ] Integrate optional ORC JIT path to execute structured IR via LLVM.
- [ ] Validate lowering with lit tests comparing generated LLVM IR / assembly to expectations.

## Infrastructure & CI
- [ ] Set up unit tests (GoogleTest/LLVM lit) for dialect verifiers, passes, interpreter.
- [ ] Configure CI pipeline targeting `/opt/llvm-20` (build, tests, formatting).
- [ ] Add clang-format/clang-tidy configs; integrate with CMake (`ninja format`).
- [ ] Write developer onboarding docs (build, test, coding conventions, plugin guide).

## Research & Validation
- [ ] Map semantics against reference paper(s) (e.g., GPU concurrency semantics) to ensure fidelity.
- [ ] Prototype translation to SPIR-V to cross-check structured dialect design.
- [ ] Benchmark interpreter on representative kernels; profile hotspots and optimize data structures.
