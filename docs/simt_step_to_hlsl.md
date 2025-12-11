# Raising SIMT-Step to HLSL: Design Notes

## Goal
Provide a source-to-source "raiser" from a *subset* of the SimtStep dialect back to HLSL text, so we can:
- Compare interpreter results against a real GPU by running the raised HLSL.
- Use the fuzzer to generate SimtStep, raise to HLSL, and cross-check outputs.

## Raisable Subset (initial)
- **Types**: `i1`, `i32` (optionally `i64`/`f32` if needed), `!simt_step.resource<Global, T>`.
- **Builtins**: `simt_step.dispatch_thread_id`, `group_thread_id`, `group_id`.
- **Structured control**: `simt_step.if` (with yields), `simt_step.loop` (prepare/condition + body/yield), `simt_step.switch` (when available), `break/continue`.
- **Ops**: `arith` (`add`, `cmp`, `rem`, etc.), `simt_step.buffer.load/store`, `simt_step.wave_count_bits` (mapped to HLSL wave intrinsics), simple constants.
- **Exclusions** (for v1): collectives/sync ops without clear HLSL analog, custom masks/push/pop, unsupported types.

## Mapping to HLSL
- **Function signature**:
  - SimtStep `func @main(%buf: !simt_step.resource<Global, i32>, …)` → HLSL `RWStructuredBuffer<int> buf : register(u0)` (resource binding policy TBD).
  - Builtin thread IDs map to HLSL system values (`uint3 SV_DispatchThreadID`, `SV_GroupThreadID`, `SV_GroupID`).
  - `simt.num_threads` attr → `[numthreads(x,y,z)]`.
- **Control flow**:
  - `simt_step.if` → HLSL `if` / `else` with explicit evaluation of the condition SSA value.
  - `simt_step.loop` with prepare/body:
    - Lower to `for`/`while` using the prepare region’s `condition` and carried values as loop vars.
    - Carried values become local variables updated per iteration; yields in body update loop vars.
    - `break`/`continue` map directly.
  - `simt_step.switch` → HLSL `switch`/`case` (once supported).
- **Values/ops**:
  - `buffer.load/store` → `buf[idx]` reads/writes (ensure idx is `uint` or cast).
  - `wave_count_bits(pred)` → `WaveActiveCountBits(pred)` (or equivalent intrinsic); ensure the expected mask semantics match HLSL’s active lanes.
  - Arithmetic/compare → straightforward HLSL expressions; signed vs unsigned must match SimtStep semantics.
- **Masks/active lanes**:
  - For the raisable subset, require predicates to be pure functions of thread IDs/constants so HLSL active masks match interpreter expectations. Avoid relying on interpreter-specific merge/mask behavior.

## Implementation Sketch
- **MLIR pass**: Walk the SimtStep module and emit HLSL text.
  - Validate that only raisable ops/types appear; otherwise, emit an error.
  - Maintain an environment mapping SSA values to emitted HLSL expressions (strings) and track loop-carried vars.
  - Emit resources first, then `main` with `[numthreads]` and parameters for resources/builtins.
  - Structured regions: recursively emit blocks with proper indentation; convert `yield`/`condition` to loop var updates / condition checks.
- **Verification**:
  - Lower the emitted HLSL back through your existing HLSL→SimtStep pipeline and diff IR, or
  - Run both the CPS interpreter and the GPU on the raised HLSL and compare buffer outputs.
- **Limitations/assumptions**:
  - Single `func @main`, single workgroup size attribute.
  - No side-effecting ops beyond buffer stores.
  - Wave ops limited to `wave_count_bits` and assumed to match HLSL active lanes semantics.

## Next Steps
1) Implement a small “raiser” pass that targets the fuzzer’s generated kernels: builtins, buffer load/store, if/loop, wave_count_bits.
2) Add a CLI tool (e.g., `simt-step-raise`) that reads SimtStep MLIR and writes HLSL text or a DXIL blob (via existing compiler).
3) Wire a cross-check harness: run CPS interpreter and raised HLSL (on GPU or a DXIL simulator), compare buffer contents.
4) Expand the raisable subset incrementally (types, ops, switch) as needed.***
