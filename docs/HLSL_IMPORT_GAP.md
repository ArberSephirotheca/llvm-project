# HLSL Importer vs. MiniHLSL Interpreter

This note records the current feature gap between the existing MLIR importer in
`tools/simt-hlsl-import` and the legacy interpreter-based HLSL frontend located
at `DirectXShaderCompiler/tools/clang/tools/dxc-fuzzer`.

## Summary

The MLIR importer is currently limited to a small scalar fragment of HLSL and
does not yet cover many language constructs supported by the interpreter.
Closing this gap requires expanding type conversion, expression lowering, and
built-in handling.

## Details

- **Type coverage**
  - Importer (`convertType`) accepts only scalar `BuiltinType`s (bool and
    integer/float widths). Vector, matrix, array, and user-defined types return
    failure. (`tools/simt-hlsl-import/lib/Lowering.cpp`)
  - Interpreter preserves the full Clang type string and works with vector
    accessors, buffer types, and custom structs via `HLSLTypeInfo::fromString`
    and related helpers. (`MiniHLSLInterpreter.cpp` around expression
    conversion)

- **Expression coverage**
  - Importer lowers literals, `DeclRefExpr`, a subset of binary operations, and
    the short-circuit `&&`/`||` forms recently added. All other expression kinds
    (calls, unary ops, array subscripts, member access, ternary `?:`, compound
    assignments, vector swizzles, etc.) are currently unsupported.
  - Interpreter has dedicated conversion paths for these constructs, including
    `convertCallExpressionToExpression`, `convertUnaryExpression`,
    `convertOperatorCall` (covering `[]`), and `convertConditionalOperator`.

- **Intrinsics and built-ins**
  - Importer now materialises thread identifier intrinsics: parameters tagged
    with `SV_DispatchThreadID`, `SV_GroupThreadID`, `SV_GroupID`, or
    `SV_GroupIndex` lower to the corresponding `simt_step.dispatch_thread_id`,
    `.group_thread_id`, `.group_id`, and `.group_index` ops. Other system values
    (e.g., subgroup size) remain unsupported.
  - Compatibility wrappers cover the initial wave suite and count helper:
    `WaveActiveAllTrue`, `WaveActiveAnyTrue`, `WaveActiveCountBits`, and
    `WaveGetLaneIndex` now lower to the corresponding `simt_step.wave_*`,
    maths `ctpop`, and `simt_step.lane_id` operations.
    `GroupMemoryBarrierWithGroupSync` is accepted for parsing (still lowered to
    a placeholder until the dialect semantics land).
  - Interpreter simulates threadgroups and waves, tracks built-in state, and
    recognises the associated intrinsic call patterns.

- **Control flow and statements**
  - Importer currently handles `if`, loops (`for`, `while`, `do`), `switch`,
    and basic assignments, but the coverage is fragile: unsupported expressions
    inside those constructs make lowering fail.
  - Interpreter provides conversions for the same statements and is tolerant of
    richer expression forms inside them, so long as the interpreter runtime can
    execute them.

## Next Steps

1. Extend `convertType` with vector/matrix handling and fall-backs for pointer
   and aggregate types used by HLSL shaders.
2. Incrementally add expression lowering cases mirroring the interpreter paths
   (calls, unary ops, ternaries, subscripts, member access, compound
   assignments).
3. Introduce an MLIR representation for common HLSL built-ins (thread and wave
   IDs, UAV/SRV accesses, wave intrinsics) so importer users do not hit
   unsupported errors.
4. Back new functionality with regression tests similar to
   `logical_short_circuit` to guard the semantics.
