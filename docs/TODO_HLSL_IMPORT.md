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
- [ ] Model thread and dispatch IDs (`SV_DispatchThreadID`, etc.).
- [ ] Lower wave intrinsics (`WaveActiveSum`, `WaveActiveMin`, …) to SimtStep
      operations or placeholders.
- [ ] Represent resource accesses (SRV/UAV) and memory operations.
- [ ] Add barrier and synchronization constructs where required.

## 4. Control Flow Robustness
- [ ] Ensure `if`, loops, and `switch` gracefully handle richer expressions in
      their components rather than failing when unsupported constructs appear.
- [ ] Add structured lowering patterns for short-circuit constructs beyond
      boolean scalars (e.g., vectors).

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

Each bullet can be decomposed further, but this structure should help us stage
work, assign owners, and track progress toward parity.
