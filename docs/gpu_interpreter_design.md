# Designing a Flexible GPU Interpreter: Discussion Notes

## Motivation
The user wants to design an interpreter for GPU programs with **swappable semantics**:
- Focus is on **control flow** and **subgroup synchronization**.
- Need to model different execution models (e.g., strict SIMT, independent thread scheduling).
- Flexibility to change meaning of instructions (barriers, votes, shuffles) without rewriting the interpreter.

---

## Key Concepts

### Continuation-Passing Style (CPS)
- Instead of returning values, functions receive an extra argument (continuation) representing *what to do next*.
- Enables:
  - Explicit suspension/resumption (`yield`, `await`, `barrier`).
  - Non-standard control flow (exceptions, backtracking, nondet).
  - Avoiding deep recursion via trampolines.

**GPU relevance**: barriers and reconvergence points can be modeled as CPS *effects*.

---

### Tagless-Final Style
- Programs are written against an **algebra (trait/concept)** that defines semantic actions.
- Different interpretations are provided by implementing the algebra differently.
- Example: `Semantics` interface with methods for `arith`, `cmp`, `shuffle`, `vote`, `barrier`.

**GPU relevance**:  
Swapping semantics = plugging in different models:
- Strict SIMT lockstep.
- Independent thread scheduling (Volta+).
- Instrumented semantics (logging, tracing).
- Model-checking semantics (nondet exploration).

---

### Effect Handlers
- Higher-level abstraction over CPS: effects like `Barrier`, `Yield`, `Nondet` suspend execution.
- A **handler** decides what to do when effect occurs:
  - Wait for all lanes (barrier).
  - Choose one continuation or explore all (nondet).
  - Resume with fairness/priority (scheduler).

**GPU relevance**:  
Barrier semantics, reconvergence, and subgroup scheduling policies are all naturally expressed as effect handlers.

---

## Architecture (C++20)

1. **Structured IR**
   - Basic blocks and operations (`Arith`, `Cmp`, `If`, `Loop`, `Barrier`, `Shuffle`, `Vote`).
   - Each `simt_struct.block` region begins with a mask block argument that carries the active lanes; subsequent block arguments represent carried SSA values. Lowering injects `mask_pop`/`mask_merge` at reconvergence so every block sees the authoritative mask without re-emitting `simt_step.active_mask`.

2. **Semantics Concept**
   ```cpp
   template<class S>
   concept Semantics = requires(S s, ArithOp aop, int64_t x, int64_t y, Mask m) {
     { s.arith(aop, x, y) } -> std::same_as<int64_t>;
     { s.shuffle_xor(m, 1, x) } -> std::same_as<int64_t>;
     { s.vote_all(m, true) } -> std::same_as<bool>;
     { s.on_barrier(Scope::Subgroup, m) };
   };
   ```

3. **CPS Trampoline**
   - Each step returns `Step`:
     - `Continue(f)` — execute next instruction.
     - `Produce(v)` — return final value.
     - `Suspend(eff)` — pause at an effect.
     - `Halt` — stop execution.
   - Handlers interpret `Effect` values (`BarrierHit`, `Yield`, `Nondet`).

4. **Barrier Handler Example**
   - Collects arriving lanes until full wave/subgroup is present.
   - Releases all parked continuations once ready.
   - Reconvergence and fairness policy are encoded here.

---

## Why CPS + Tagless-Final is Good for GPU Interpreter
- **CPS**: Models suspension/resumption, reconvergence, barriers, and nondet scheduling directly.
- **Tagless-Final**: Makes semantics swappable and modular (SIMT, independent threads, checker).
- **Handlers**: Centralize policies (barrier release conditions, scheduling, fairness).
- **Extensible**: Easy to add instrumentation (trace events, logging, testing).
- **Testable**: Differential testing between semantics is straightforward.

---

## Practical Advice (C++20)
- Use **templates/CRTP** for `Semantics` to inline hot paths.
- Use **bitmasks** for lane activity; keep per-lane state in SoA layout for vectorization.
- Represent continuations as `std::function<Step()>` or coroutines.
- Write handlers as composable functors (`SubgroupBarrierHandler`, `ModelCheckerHandler`).
- Test with random programs, compare traces under different semantics.

## Mask Threading Contract
- Structured lowering threads masks explicitly through block arguments. Branches
  forward the current mask operand, and reconvergence points insert
  `mask_pop`/`mask_merge` before any user-level operations execute.
- The interpreter binds each block’s leading mask argument when a branch target
  is chosen, so `lookupMaskValue` on that argument returns the active lane set
  without re-reading hardware state.
- Debug tooling (`simt-dump-structured-program`) reports both mask and carried
  argument counts, reflecting the shape the interpreter consumes.

---

## Compile-Time Evaluation (C++20 vs Mojo)

### Mojo-style capabilities
Mojo allows “compile-time evaluation” via **alias parameters** and **constant evaluation**, letting you precompute values, generate types, and specialize semantics before runtime.

### C++20 equivalents
- **Templates** → specialize engine by wave size, mask width, semantics.
- **`constexpr` functions** → evaluate at compile time when inputs are constant.
- **`consteval` functions** → must run at compile time (ideal for LUTs/opcode tables).
- **`if consteval`** → choose compile-time vs runtime code path.
- **Concepts** → constrain valid semantics policies.

### Examples

**1. Wave size specialization**
```cpp
template<std::size_t WaveSize, class Sem>
struct Engine { /* fully inlined semantics */ };
```

**2. Precomputed LUTs**
```cpp
template <std::size_t W>
consteval auto build_shuffle_xor_lut() {
  std::array<uint32_t, W> lut{};
  for (uint32_t lane = 0; lane < W; ++lane) lut[lane] = lane ^ 1;
  return lut;
}

constexpr auto SHUFFLE_XOR_LUT32 = build_shuffle_xor_lut<32>();
```

**3. Dual-mode APIs**
```cpp
template <class F>
constexpr auto make_decode_table(F&& f) {
  if consteval {
    // compile-time init
  } else {
    // runtime path
  }
}
```

### What you gain
- Precompute **shuffle/vote LUTs**, verification tables, constant folds at compile time.
- Zero-overhead specialization for wave size and semantics policies.
- Extensible to consteval transforms for embedded DSL programs.

### Limitations
- Can’t run full CPS interpreter at compile time for large dynamic programs.
- C++20’s `constexpr` requires data structures to be constexpr-friendly (no dynamic allocators).
- Coroutines are runtime only, though handlers/policies can still be constexpr.

---

## Summary
Yes — combining **CPS + tagless-final** in C++20 is a strong design choice:
- CPS provides *precise control over suspension and resumption* (barriers, reconvergence).
- Tagless-final provides *pluggable semantics* for different GPU models.
- Effect handlers naturally capture GPU synchronization, nondet, and reconvergence rules.
- C++20’s **constexpr/consteval** features let you adopt Mojo-style compile-time evaluation for policies, tables, and static specialization.

This design gives flexibility, clarity, and extensibility — ideal for research and testing GPU forward progress and subgroup semantics.

---

## Implementation Roadmap

### Phase 0 – Foundations
- Publish this design as the canonical reference (link from TODO docs) and file tracking issues for CPS engine, semantics algebra, handlers.
- Audit existing `simt_struct` ops + traits to confirm required runtime hooks (mask stack, branch metadata, loop markers, collectives).
- Capture interpreter requirements from lowering (carried values, merge/continue targets, block symbols).

### Phase 1 – Structured Program View
- Extend the structured metadata cache into a `StructuredProgram` façade: `(ModuleOp module, DenseMap<SymbolRef, BlockOp>, entry symbol)`.
- Add `BlockInfo` records (mask + carried argument types, merge/continue targets)
  materialised at creation time.
- Unit test by parsing a tiny MLIR module and verifying block lookup + attribute capture.

### Phase 2 – Value Representation
- Introduce `ValueStore` (bool/int/float plus future vector support) with SoA layout for per-lane data.
- Provide `ValueTraits<WaveWidth>` templates; use `consteval` helpers to precompute shuffle/vote look-up tables.
- Add type mapping utilities from MLIR types → runtime slots (including mask
  operands for block arguments); cover with focused tests.

### Phase 3 – Lane/Wave State
- Define `LaneState` (current block symbol, op index, SSA environment id, active-bit).
- Define `WaveState<WaveWidth>` aggregating lanes, active mask bitset, merge stack, pending effects; integrate existing `MaskFrame` logic.
- Implement `WaveConfig<WaveWidth>` compile-time specialisations holding LUTs/state defaults.
- Tests: create a 2-lane wave, push/pop masks, ensure state updates match expectations.

### Phase 4 – Semantics Algebra
- Declare a `Semantics` concept exposing hooks: `{ arith, cmp, mask_ops, collective, sync, memory, custom }`.
- Implement `StrictSimtSemantics<WaveWidth>` using compile-time LUTs for wave ops.
- Provide optional wrappers: logging semantics, independent-thread scheduling stub.
- Tests: call each hook with synthetic inputs, confirm semantics outputs + side-effects.

### Phase 5 – CPS Core
- Introduce `Step` variant (`Continue`, `Suspend`, `Produce`, `Halt`).
- Add `Continuation` wrappers (`using Continuation = std::function<Step()>;` initially, evolve to coroutines later).
- Implement `InterpreterEngine<WaveWidth, Semantics>`: fetch next op, dispatch to handler, return `Step`.
- Define effect payloads (`BarrierHit`, `MaskOp`, `Reconverge`, `Yield`, `Nondet`).
- Tests: run engine on straight-line block, ensure `Produce` returns final values.

### Phase 6 – Effect Handlers
- `BarrierHandler`: collect lane continuations until quorum, release per reconvergence policy; parameterise by scope.
- `ReconvergenceHandler`: respect `PostDominator`, `Explicit`, `None` attributes; integrate mask stack frames.
- `SchedulerHandler`: provide lockstep policy vs independent-thread scheduling (Volta-style); expose fairness knobs.
- Compose handlers (e.g., `SchedulerHandler<BarrierHandler<ReconvergenceHandler<Engine>>>>`).
- Tests: simulate barrier arrival ordering, ensure lanes resume correctly; verify scheduler fairness on small traces.

### Phase 7 – Operation Dispatch
- Straight-line ops: evaluate per lane (arith, cmp, load/store placeholders), update `ValueStore`.
- Mask ops: hook into mask stack + effect system using helpers from Phase 3.
- Branches: compute per-lane conditions, produce true/false masks, queue continuations for destination blocks, update merge stack via handler.
- Loops: treat `continue_target` with iteration markers (mirroring MiniHLSL semantics) to reconnect correctly.
- Returns: mark lanes complete; once all lanes produce values, collapse to final `Produce`.
- Collectives: use semantics hook + compile-time LUTs; ensure handler enforces required participation.
- Tests: start with 2-lane branch sample, loop with continue, and wave ballot scenario.

### Phase 8 – Runtime Integration
- Extend `simt-run` to drive the interpreter: parse MLIR, run `SimtStepToStructured`, build `StructuredProgram`, instantiate engine + handlers.
- CLI flags: semantics model (`--semantics=strict|independent|trace`), wave width (`--wave-width=32`), reconvergence policy overrides.
- Add output modes: pretty trace, JSON event stream, final value dump.
- Tests: lit-style invocation verifying CLI outputs on known kernels.

### Phase 9 – Testing Strategy
- **Unit tests** (gtest): value conversions, mask stack invariants, semantics hooks.
- **Interpreter tests** (lit): run interpreter on `.mlir` fixtures (straight-line, divergent branch, loop, barrier) with `CHECK`ed traces/results.
- **Differential tests**: run same kernel under two semantics; confirm either identical state or expected divergence logs.
- **Static asserts**: ensure compile-time LUTs instantiate correctly for supported wave widths.

### Phase 10 – Optimisation & Extensions
- Swap `Continuation` to C++20 coroutines for lower overhead.
- Add `consteval` simplification pass for constant subgraphs before runtime.
- Support multi-wave/threadgroup execution and memory hierarchy (shared/global).
- Integrate `simt.custom` plugin callbacks via semantics hook.
- Profiling hooks to measure interpreter performance on benchmarks.

### Phase 11 – Documentation & Examples
- Update dialect docs with interpreter expectations (mask semantics, effect triggers).
- Document semantics API and handler configuration (developer guide).
- Provide sample MLIR kernels + scripts showing interpreter usage.
- Record tutorial on swapping semantics models and comparing results.

### Phase 12 – Rollout
- Land phases incrementally with feature flags.
- Keep CI green; ensure new tests gate regressions.
- Coordinate with lowering pipeline for any metadata requirements (e.g., block result carrying).
- Announce availability once straight-line + branching semantics validated; iterate toward full feature parity.
