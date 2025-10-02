# TODO – HLSL Importer Parity

This list breaks the gap with `MiniHLSLInterpreter` into scoped tasks so we can
plan and prioritize the work needed to bring the MLIR importer up to parity.

## 1. Type System
- [ ] Add support for HLSL vector types (e.g. `float2`, `float4`) by mapping to
      `vector<N x T>` or dedicated Simt types.
- [ ] Extend to matrices and arrays used in shaders.
- [ ] Handle user-defined structs and resource types (buffers, textures) with
      appropriate MLIR representations or opaque wrappers.

## 2. Expression Lowering
- [ ] Unary operators: increment/decrement, logical/bitwise negation.
- [ ] Conditional operator (`?:`).
- [ ] Member access (struct fields, swizzles) and `HLSLVectorElementExpr`.
- [ ] Array/buffer subscripts (`CXXOperatorCallExpr` for `[]`).
- [ ] Call expressions, including distinguishing between regular calls and
      intrinsics.
- [ ] Compound assignments and full set of binary arithmetic/bitwise ops.

## 3. Intrinsics & Built-ins
- [x] Model thread and dispatch IDs (`SV_DispatchThreadID`, `SV_GroupThreadID`,
      `SV_GroupID`, `SV_GroupIndex`).
- [ ] Lower wave intrinsics (`WaveActiveSum`, `WaveActiveMin`, …) to SimtStep
      operations or placeholders.
- [ ] Represent resource accesses (SRV/UAV) and memory operations.
- [ ] Add barrier and synchronization constructs where required.

## 4. Control Flow Robustness
- [ ] Ensure `if`, loops, and `switch` gracefully handle richer expressions in
      their components rather than failing when unsupported constructs appear.
- [ ] Add structured lowering patterns for short-circuit constructs beyond
      boolean scalars (e.g., vectors).
- [ ] Rework `switch` lowering to build one block per case instead of the nested
      `simt_step.if` cascade. The body should contain: (1) an entry block that
      unpacks carried variables and dispatches to the first case/default, (2) a
      block per explicit case label that lowers the associated statements and
      ends with a `simt_step.yield` carrying the updated variables plus the
      `(matchSeen, fallthroughActive, switchDone)` flags, and (3) a single exit block.
      Fallthrough uses the recorded metadata to branch to the next case or exit,
      and the default block is only created when a `default` label exists.

## 5. Testing & Tooling
- [ ] Build a comprehensive regression suite mirroring interpreter samples,
      including vector arithmetic, resource usage, and wave ops.
- [ ] Integrate CI targets (e.g., `ninja check-simt-hlsl-import`) that cover the
      expanded feature set.
- [ ] Document limitations and newly added capabilities.

## 6. Design & Dialect Work
- [ ] Decide whether to extend existing Simt dialect ops or introduce new ones
      for HLSL-specific features (wave ops, resource loads).
- [ ] Update dialect documentation with new operations and types.
- [ ] Explore refactoring lowering around a tagless-final style "lowering
      algebra" so we can swap interpreters: analysis-only, IR emission, and
      optional debug tracing.
- [x] Guard barrier/fence emission so we only build ops when the builder is
      attached to a concrete block—analysis walks no longer leave dangling
      operations behind.

Each bullet can be decomposed further, but this structure should help us stage
work, assign owners, and track progress toward parity.
