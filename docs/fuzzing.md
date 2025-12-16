# SIMT-Step Fuzzing Plan (Determinism Focus)

## Goals
- Generate random SIMT-Step MLIR programs (structured control only) and check they run deterministically across different scheduling choices.
- Stress CPS interpreter corner cases: divergence/reconvergence, loop iterations, breaks/continues, nested if/switch, collectives/sync.
- Produce reproducible repros (seeded), with shrinking guidance for triage.

## High-Level Approach
1) **Program generator** emits `func.func @main` with structured ops:
   - `simt_step.if`, `simt_step.loop`, `simt_step.switch` (when available), simple arith/compare, constants, optional `simt_step.dispatch_thread_id`.
   - Proper terminators: `simt_step.yield`, `simt_step.condition`, `func.return`.
2) **Scheduler perturbation**: interpreter gains a seedable randomized ready-queue policy (shuffle order). Each program is run `K` trials with different seeds.
3) **Oracle**: determinism = identical per-lane observable state across all trials:
   - Lane `hasReturned`/return value (or void), per-lane side effects if modeled later.
   - Optionally the final dynamic block masks/merge stacks for stricter checking.
4) **Failure handling**: On divergence, write MLIR + seed(s) to a corpus dir; optionally run a shrinker.
5) **Driver tool** (`simt-step-fuzz`): accepts `--programs N`, `--trials K`, bounds (depth, ops, loops), `--seed`, and writes failures.

## Generator Shape
- **Types/values**: stick to `i1`, `i32`, and vectors if already supported; avoid UB (no div by zero).
- **Structured ops**:
  - `simt_step.if`: two regions; yields optional value(s).
  - `simt_step.loop`: prepare/body regions; loop-carried values; bounded loop conditions favored.
  - `simt_step.switch` (once wired): multiple cases + default.
  - `simt_step.break`/`continue` only where valid (inside loop/switch).
  - Collectives/sync ops included sparingly with well-defined expected masks.
- **Control bounds**: parameters for max depth, max ops, max cases, max loop trip count. Prefer small bounds (e.g., trip count <= 4) for fast runs.
- **Well-formedness**: every region ends with correct terminator; SSA uses block args for carried values; no unused ops if that simplifies shrinking.
- **Biasing**: knobs to favor tricky patterns (nested if-in-loop, break+continue, empty else, single-lane predicates, mixed predicates per lane).

## Determinism Oracle
- Run each program `K` times with different scheduler seeds.
- Collect per-lane results: `(hasReturned, returnValue?, currentBlock?)` plus optional final block masks.
- Compare trial 0 vs trial i; any mismatch = failure.
- Optional stronger oracle: record the sequence of (block, op, lane) executed and require identical traces (slower).

## Scheduler Perturbation
- Interpreter option to randomize ready-queue order using a PRNG seeded from CLI/env.
- Keep semantics identical; only perturb execution order to expose race-like nondeterminism.

## Tooling / Interfaces
- New tool `tools/simt-step-fuzz`:
  - Flags: `--programs N`, `--trials K`, `--seed`, `--max-depth`, `--max-ops`, `--max-trip`, `--outdir`, `--trace` (optional), `--stop-after-fail`.
  - Emits each generated module to a temp path; on failure, preserves it with seeds and oracle dump.
- Library utilities:
  - `generateProgram(cfg, rng)` -> MLIR ModuleOp (or string).
  - `runDeterminismCheck(module, trials, seeds[], oracleMode)` -> pass/fail + evidence.

## Shrinking (nice-to-have)
- Greedy reducer: remove ops/blocks/regions while keeping the failure.
- Seed-based repro: failing module + seeds is sufficient even without shrinking.

## Safety / Performance
- Keep programs small (e.g., <= 30 ops, depth <= 4, trip count <= 4).
- Timeouts per run; bail on verifier errors.
- Determinism runs can be parallelized per program.

## Next Steps
1) Add interpreter flag to randomize ready-queue with a seed.
2) Implement minimal generator for `if` + `loop` + arith, with bounds.
3) Build `simt-step-fuzz` driver wiring generator + oracle.
4) Add CI smoke: small N,K (e.g., N=5, K=2) to catch regressions; leave long runs for manual.***

## Initial Deterministic Suite (non-uniform but reproducible)
- Make predicates pure functions of `dispatch_thread_id` and literals so each lane’s path is fixed:
  - Examples: `tid % 2 == 0`, `tid < k`, `tid == c`, loop trip `(tid % 3) + 1`.
- Exercise non-uniform control without shared mutable state:
  - `if` with per-lane predicates; both regions end in `simt_step.yield` of lane-local values.
  - `loop` with per-lane trip counts; body `acc += i`; optional `break`/`continue` guarded by `tid`/`i` (e.g., `if (tid == 0 && i == 1) break;`).
  - Nested if inside a loop (outer `tid < 2`, inner `tid == 0` vs else).
- Types/ops subset: `i1`, `i32`, `simt_step.dispatch_thread_id`, `arith.constant`, `arith.addi`, `arith.cmpi`, `simt_step.if`, `simt_step.loop`, `simt_step.yield`, `simt_step.condition`, `func.return`.
- Bounds: depth ≤ 3, loop trip count ≤ 3, small blocks.
- Oracle: run each generated module twice (or K times) with different scheduler seeds; compare per-lane `(hasReturned, returnValue)`; any mismatch = bug.***

## Metamorphic Testing Ideas
- **Control rewrites**: generate a baseline program, then produce semantics-preserving variants:
  - Wrap in `if (true) { … }` or `if (false) { yield carried }`.
  - Negate condition and swap branches: `if (c) A else B` ↔ `if (!c) B else A`.
  - Distribute conjunctions: `if (c1 && c2) { … }` ↔ `if (c1) { if (c2) { … } }`.
  - Small loop transforms: unroll 1–2 iterations vs. original bounded loop.
- **Value rewrites**: apply algebraic identities (`x + 0`, commutative swaps, double-negation on cmp), reorder independent ops, inject dead code.
- **Mask-safe transforms**: use predicates that are pure functions of `dispatch_thread_id`/literals so lane paths stay deterministic; hoist common yields to both branches to keep results identical.
- **Oracle**: run original and transformed modules under different scheduler seeds; require per-lane `(hasReturned, returnValue)` to match. Any mismatch flags a bug.

## SIMT-Step Trace-Guided Fuzzer (mini design)
Borrowing the MiniHLSL fuzzer pattern, add trace capture to drive context-aware mutations and better triage.

### Trace data to log from CPS interpreter
- **Dynamic blocks**: `block*`, `sequenceId`, `kind`, `parentKey`, `loopOp/switchOp/ifOp`, masks (expected/active/completed) per lane, value env snapshot (optional for deltas).
- **Control decisions**: per-lane branch outcomes (`if` condition result), loop condition booleans, break/continue events with block keys.
- **Collectives/sync** (if included): op kind, block key, participant mask, operands/results per lane.
- **Execution trace** (optional, for deep diffs): sequence of `(block key, op name, lane)` executed.
- **Final state**: per-lane `(hasReturned, returnValue, currentBlock)` and final block masks.

### Fuzz flow
1) **Generate** a deterministic SIMT-Step program (see Initial Deterministic Suite).
2) **Run & capture**: execute with CPS interpreter, record trace.
3) **Mutate**: apply semantics-preserving rewrites (Metamorphic Testing Ideas). Use trace to target specific dynamic blocks/iterations for mutations (e.g., flip a predicate only in one loop iteration).
4) **Validate**: rerun original and mutated under different scheduler seeds; compare per-lane outputs and (optionally) traces. Mismatch = bug.

### Determinism oracle with traces
- Baseline: per-lane `(hasReturned, returnValue)` equality across seeds.
- Stronger: also require control decision log and/or execution trace to match to catch “same output, different path” issues.

### Integration notes
- Add an interpreter flag to emit traces (maybe JSON) keyed by `block`+`seq`.
- Keep traces small: omit value envs unless needed; record masks and decisions instead.
- Use traces to guide shrinking: preserve divergent block/decision while dropping unrelated ops.

## Program Generator Sketch (SIMT-Step)
Mirrors the MiniHLSL generator but builds MLIR/SimtStep directly.

- **Participant patterns**: utilities that build lane predicates from `dispatch_thread_id`:
  - Single lane (`tid == k`), ranges (`tid < N`, `tid >= N`), parity (`(tid & 1) == 0`), mod classes (`tid % m == r`), OR/AND combos for sparse sets.
- **Structured emitters**: helpers to assemble well-formed control:
  - `emitIf(cond, thenFn, elseFn)` producing yields and threading SSA.
  - `emitLoop(init, condFn, bodyFn)` for `simt_step.loop`, with loop-carried values; optional `break`/`continue` guarded by `tid`/loop index.
  - `emitSwitch` later, when the op is wired.
- **Deterministic templates to fuzz** (non-uniform but reproducible):
  1) If/else on `tid` parity; then yields `tid+1`, else `tid`.
  2) Loop with trip `(tid % 3) + 1`; body `acc += i`; optional `break` if `tid == 0 && i == 1`.
  3) Nested if inside loop: outer `tid < 2`, inner `tid == 0` vs else.
  All paths end in `simt_step.yield` and `func.return`.
- **Observable side-effect for oracle**:
  - Signature: `func @main(%out_main: !simt_step.resource<Global, i32>, %out_wave: !simt_step.resource<Global, i32>)`.
  - `out_main` holds the main results; each generated “root” writes to `out_main[root * lanes + tid]`, so per-root writes never overlap across lanes.
  - `out_wave` logs each `wave_count_bits` site. For wave id `w`, `waveBase = w * (maxTripCount * lanes)`, and each dynamic loop iteration `i` uses `waveBase + i * lanes + tid`. This stride keeps iterations, lanes, and call sites disjoint, so looped wave-counts do not collide.
- **Randomization knobs**: max depth, max ops, max trip count, pattern choice, whether to insert nested ifs or break/continue; keep counts small for speed.
- **Trace use**: optional JSON trace from the CPS interpreter (block key+seq, masks, decisions) to target mutations and aid shrinking; per-lane buffer contents are the primary oracle.***
