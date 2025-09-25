# HLSL Resource Model Plan

This note captures the gap in the current MLIR importer for Shader resources
(buffers, textures) and the proposed path to close it so that expressions such
as `buffer[idx]` can be lowered faithfully.

## Problem Statement
- The importer only understands scalar values, vectors, and basic control flow.
  Resource handles (`RWBuffer<T>`, `Buffer<T>`, `ByteAddressBuffer`, textures)
  are not given any dedicated type.
- Subscript expressions (`operator[]`) on resources therefore surface as
  `CXXOperatorCallExpr` nodes without any lowering path. The importer currently
  fails with "unsupported expression lowering" when encountering them.
- Without a resource model we cannot emit MLIR operations that represent loads
  or stores, nor can we reason about side effects or aliasing for those
  operations, leaving a substantial portion of practical HLSL unhandled.

## Goals
1. Represent buffer resource handles (global and shared) and their element types in the SimtStep dialect.
2. Lower reads/writes (and eventually atomics) to explicit MLIR operations with
   well-defined memory effects.
3. Preserve SSA mutation tracking so loops and conditionals continue to work
   when resource values are updated.
4. Lay the groundwork for extending to textures/samplers later.

## Proposed Solution
### 1. Dialect Extensions
- Add a `simt.resource<memspace, element-type>` handle type capturing global
  and shared buffer resources.
- Introduce operations such as:
  - `simt_step.buffer.load %handle, %index : (!simt.resource<Global, !type>, i32) -> !type`
  - `simt_step.buffer.store %handle, %index, %value : ...`
- Attach memory-effect traits (`Read`, `Write`) and attributes for address space
  if needed.

### 2. Type Conversion
- Update `convertType` to detect Clang's HLSL resource types (e.g.
  `clang::HLSLResourceType`, `clang::AttributedType` wrapping a resource).
- Extract the element type via Clang's APIs (`GetOriginalMatrixOrVectorElementType`
  or direct template arguments) and recursively map it to an MLIR type.
- Produce the new `simt.resource` type, cached in the lowering context to keep
  parameter lowering consistent.

### 3. Expression Lowering
- Extend `lowerExpr` with a case for `clang::CXXOperatorCallExpr` whose callee
  name is `operator[]` and whose base type is our resource type.
- Lower the base expression to the resource SSA value, lower the index
  expression to an integer, and emit a `simt_step.buffer.load` op.
- For assignment targets (`buffer[idx] = value`) hook into the binary operator
  handling to emit `buffer.store`, updating the mutation set with the resource
  handle.

### 4. Supporting Infrastructure
- Record resource handles in the value map so they can be threaded through
  loops and conditionals (similar to other mutated values).
- Update zero-initialisation helpers (if functions can return resource handles)
  or reject returning resources explicitly.
- Document the new types/ops in `SimtStepDialect.md`.

### 5. Testing
- Add regression tests that declare buffers, read from them, and write to them.
- Include control-flow cases (e.g., inside loops) to ensure mutation forwarding
  works.
- As more resource kinds are added, add targeted tests for each.

## Next Steps
1. Prototype `simt.resource` type and `buffer.load/store` ops in the dialect.
2. Teach `convertType` to recognize buffer types and produce the new type.
3. Implement load lowering for `operator[]` reads, then extend to stores.
4. Iterate with real shader examples to validate the design before expanding to
   textures/samplers.
