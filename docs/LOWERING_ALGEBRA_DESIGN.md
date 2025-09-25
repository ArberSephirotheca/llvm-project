# Tagless-Final Lowering Algebra for `simt-hlsl-import`

## Motivation

- Our lowering pipeline is reused for two distinct purposes:
  1. **Analysis walks** (e.g., `collectLoopMutations`) that simply inspect
     statements and track mutations.
  2. **Emit walks** that build real MLIR operations.
- Today both paths share the same helpers and a shared `LoweringContext`. The
  analysis walk uses a scratch `OpBuilder` that is *not* anchored in the MLIR
  module. When helpers such as `lowerBarrierUtilityCall` run during analysis
  they accidentally create `simt_step.fence`/`barrier` ops in a block with no
  parent, leading to fatal printer crashes (`INVALIDBLOCK`).
- We want one unified lowering implementation, but we must guarantee that
  analysis mode cannot produce IR or expose `mlir::Value` instances.

## Design Overview

Adopt a *tagless-final* style “lowering algebra”. Each lowering helper is
templated on an interpreter that provides the operations we need (build
constants, emit arithmetic, enter/exit scopes, etc.). The algebra exposes an
API that is expressive enough for the current emit logic, but also specific
enough that analysis interpreters can implement it without side effects.

### Key Goals

- **Interpreter-defined value type** – emit returns `mlir::Value`, analysis
  returns a symbolic `SymValue` that only tracks shape/constness.
- **No MLIR types in analysis** – emit sees `mlir::Type`, analysis uses a light
  weight tag (e.g., `I32`, `F32`, `Vector<…>` descriptors).
- **Anchored emit** – constructing the emit interpreter requires an
  `OpBuilder` whose insertion block has a parent operation, making it
  impossible to accidentally emit ops while “detached”.
- **Statement-level barriers** – barriers/fences lower through the algebra in a
  void context, so they can never leak `mlir::Value()` placeholders.
- **Unified diagnostics** – both interpreters call into a shared
  `DiagSink`/`emitError` helper so messages stay consistent.
- **Scoped control flow** – use RAII objects for `if`/loop scopes instead of
  raw push/pop to avoid mismatched stack usage.

## Algebra Sketch

```cpp
struct SourceLoc {
  const clang::Stmt *clangNode = nullptr;
  mlir::Location mlirLoc; // only populated by emit interpreter
};

enum class ValueClass { ScalarInt, ScalarFloat, Vector, Pointer, Unknown };

enum class ArithOp { Add, Sub, Mul, Div, Rem, Neg, BitAnd, BitOr, BitXor };
enum class CmpOp { EQ, NE, LT, LE, GT, GE };
enum class BarrierKind { Workgroup, Device, All };

template <typename Self>
struct LoweringAlgebra {
  using Value = typename Self::ValueType;

  // Expressions
  Value emitConstantInt(int64_t v, const char *tag, SourceLoc);
  Value emitConstantFloat(double v, const char *tag, SourceLoc);
  Value emitArithmetic(ArithOp, Value lhs, Value rhs, SourceLoc);
  Value emitCompare(CmpOp, Value lhs, Value rhs, SourceLoc);
  Value emitSelect(Value cond, Value trueV, Value falseV, SourceLoc);

  // Variables / SSA
  Value lookupVariable(const clang::ValueDecl *);
  void bindVariable(const clang::ValueDecl *, Value);
  void noteMutation(const clang::ValueDecl *);

  // Statements / control flow
  struct LoopScope {
    void close();
  };
  LoopScope beginLoop(SourceLoc);

  struct IfScope {
    void close();
  };
  IfScope beginIf(SourceLoc, Value cond);
  void emitElse(IfScope &scope);

  void emitYield(ArrayRef<Value> values, SourceLoc);
  void emitReturn(std::optional<Value>, SourceLoc);

  void emitBarrier(BarrierKind, SourceLoc);
  void emitFence(BarrierKind, const char *memSpace, SourceLoc);

  // Diagnostics / tracing
  void trace(const char *message, SourceLoc);
};
```

Notes:

- The algebra is intentionally minimal. We add hooks as we refactor existing
  lowering helpers. Value tags (`const char *tag`) are symbolic tokens such
  as `"i32"`, `"f64"`, `"vector<4xf32>"`.
- `Value` is interpreter-specific; the emit interpreter aliases it to
  `mlir::Value` while analysis stores a small `SymValue` (e.g., `{ValueClass,
  std::optional<unsigned> elementCount, std::optional<bool> isConst`).
- `SourceLoc` carries the clang node for analysis errors and the MLIR location
  for emit.

## Interpreters

### EmitInterpreter

```cpp
struct AnchoredBuilder {
  mlir::OpBuilder &builder;
  AnchoredBuilder(mlir::OpBuilder &b) : builder(b) {
    auto *blk = builder.getInsertionBlock();
    assert(blk && blk->getParentOp() &&
           "EmitInterpreter requires anchored builder");
  }
};

struct EmitInterpreter : LoweringAlgebra<EmitInterpreter> {
  using ValueType = mlir::Value;

  AnchoredBuilder ab;
  DiagSink &diag;

  Value emitConstantInt(int64_t v, const char *tag, SourceLoc loc);
  // ... build real MLIR ops as today ...
};
```

### AnalysisInterpreter

```cpp
struct SymValue {
  ValueClass kind;
  std::optional<unsigned> elementCount;
  std::optional<bool> isConst;
  // Additional metadata as needed (e.g., pointer space).
};

struct AnalysisInterpreter : LoweringAlgebra<AnalysisInterpreter> {
  using ValueType = SymValue;

  llvm::DenseMap<const clang::ValueDecl *, SymValue> symbolTable;
  llvm::SmallPtrSet<const clang::ValueDecl *, 8> mutated;
  DiagSink &diag;

  Value emitConstantInt(int64_t v, const char *tag, SourceLoc); // record tag
  // ... other methods just update metadata, never emit MLIR
};
```

### TraceInterpreter

```cpp
template <typename Inner>
struct TraceInterpreter : LoweringAlgebra<TraceInterpreter<Inner>> {
  using ValueType = typename Inner::ValueType;
  Inner inner;
  raw_ostream &os;

  ValueType emitConstantInt(int64_t v, const char *tag, SourceLoc loc) {
    os << "const " << tag << " = " << v << "\n";
    return inner.emitConstantInt(v, tag, loc);
  }
  // forward + log other methods
};
```

## Refactoring Plan

1. **Introduce the algebra scaffolding** (templates + interpreters) without
   changing behaviour. Emit interpreter forwards to existing `OpBuilder`
   helper functions; analysis interpreter mirrors the logic but only records
   facts. **Status:** Landed via `LoweringAlgebra.h`; `Lowering.cpp` now
   instantiates `EmitInterpreter`/`AnalysisInterpreter` with fork support for
   nested regions.
2. **Refactor barrier + buffer intrinsics first** – replace direct builder
   calls with algebra calls. Run both interpreters: analysis should produce no
   MLIR; emit should pass existing tests. **Status:** Barrier/fence calls use
   the new interpreters; buffer ops still rely on the legacy helpers.
3. **Gradually migrate remaining helpers** (expressions, control flow, returns,
   etc.). After each stage run `mlir::verify` + unit tests. **Status:** In
   progress. Statement lowering now routes `DeclStmt` variable binding,
   barrier utilities, and `ReturnStmt` emission through the algebra; expression
   helpers still hard-code `LoweringContext`.
4. **Introduce trace interpreter** for debugging and optional unit tests.
   **Status:** Not started.
5. **Once fully migrated**, delete direct builder access from lowering helpers
   and remove the scratch builder path used by analysis. **Status:** Blocked on
   the tasks below.

## Current Status Snapshot (July 2025)

- Header `tools/simt-hlsl-import/include/simt-hlsl-import/LoweringAlgebra.h`
  exposes the minimal algebra surface used by the barrier interpreters
  (constants, arithmetic, variable binding, barriers/fences, diagnostics).
- `Lowering.cpp` still owns the full `LoweringContext` stacks. `lowerStatement`
  instantiates `EmitInterpreter`/`AnalysisInterpreter` (with `fork` for loops
  and switch cases) so barrier utilities, local declarations, returns, and
  scalar arithmetic/comparisons now flow through the algebra. Analysis paths
  track symbolic `SymValue` metadata for bound variables.
- Analysis mode continues to run with a detached `OpBuilder`. The algebraized
  paths suppress barrier/fence emission, and emit-mode literals / basic
  arithmetic/comparisons now go through the algebra. Expression helpers still
  materialise IR during analysis.
- No dedicated diagnostics sink exists yet; all errors continue to flow through
  `LoweringContext::fail`.

## Detailed TODO Breakdown

1. **Unify Interpreter Entry Points**
   - Templatise `lowerExpr`, `lowerStatement`, and the helper family
     (`lowerAssignment`, `lowerWaveIntrinsicCall`, etc.) on an interpreter type
     `Interp`. Each helper should consume an `Interp &` instead of the raw
     `LoweringContext`/`OpBuilder` pairs.
   - Replace direct `OpBuilder` usage with algebra calls (`emitArithmetic`,
     `emitCompare`, `emitSelect`, `emitReturn`, `emitBarrier`, `emitFence`).
   - Ensure shared utilities (e.g., `lowerBufferAccessOperands`) return
     interpreter values or algebra-friendly structs so both emit and analysis
     paths can reuse them.

2. **Expand Algebra Surface While Migrating**
   - Add structured control-flow hooks promised in the sketch (`LoopScope`,
     `IfScope`, `emitYield`) once the first helper requires them. Keep the API
     conservative—only introduce a new hook when a migrated helper cannot be
     expressed with the existing ones.
   - Extend `BarrierKind` / mem-space tags as the codebase demands. The C++
     enum currently uses `{Workgroup, Device, All}`; keep the documentation in
     sync with the implementation.
   - Introduce optional type-tag helpers (e.g., `ValueTag`) if needed for
     vector/matrix lowering so the analysis interpreter can stay metadata-only.

3. **Interpreters Beyond Barriers**
   - Flesh out `EmitInterpreter` with real implementations that wrap
     `OpBuilder`. Constructor already asserts an anchored builder; arithmetic
     and constant hooks still need real implementations.
   - Implement `AnalysisInterpreter` to produce symbolic `SymValue`s. The value
     should record class, width, and const-ness so existing analyses can reason
     about mutations without grabbing `mlir::Value` instances.
   - Barrier-specific interpreters have been removed; ensure the generic
     interpreters cover the remaining helper surfaces before deleting direct
     `OpBuilder` usage.

4. **Diagnostics and Tracing**
   - Thread a `DiagSink` through the interpreters. Emit mode turns `SourceLoc`
     into `mlir::Location` and issues `emitError`; analysis mode records
     human-readable diagnostics without relying on `LoweringContext::fail`.
   - Add the optional `TraceInterpreter` wrapper for debugging (forwarding to an
     inner interpreter and logging calls). This becomes useful once more helpers
     are algebra-based.

5. **Context Simplification**
   - Slim `LoweringContext` down to shared state (loop stacks, result types,
     diagnostics). The interpreter becomes the single façade for IR emission and
     analysis bookkeeping.
   - Replace manual push/pop bookkeeping with RAII guards once the interpreter
     exposes `LoopScope` / `IfScope`. Update the loop/switch helpers to use
     these guards and delete the ad-hoc `controlStack` management.

6. **Testing & Verification**
   - Add regression coverage for both interpreters. At a minimum, ensure the
     analysis interpreter leaves the module untouched (e.g., `CHECK-NOT` on
     emitted IR) and that emit mode continues to pass existing importer tests.
   - Consider unit-testing individual helpers with the `TraceInterpreter` or a
     mock interpreter to confirm we do not regress analysis-only behaviour.

Work through the items in order—the algebra surface may need to grow as each
helper migrates, but the list above keeps the scope explicit for code review.

## Implementation Notes

- `LoweringContext` becomes a lightweight owner of common state
  (diagnostics, return type, loop/switch stacks). Each interpreter holds a
  reference to the context and exposes typed accessors.
- The algebra may need additional hooks (e.g., for branch masks) as we refactor
  more helpers. Add them incrementally with clear semantics.
- Replace push/pop stacks with RAII guard objects (`LoopScope`, `IfScope`).
  Emit uses them to manage MLIR regions; analysis tracks scope nesting for
  correctness checks.
- Diagnostics should flow through a common sink so both interpreters emit the
  same messages. The emit interpreter converts `SourceLoc` into
  `mlir::Location` when creating IR, the analysis interpreter only keeps clang
  info.
- Add regression tests that run analysis mode and assert it produces zero IR
  (e.g., CHECK-NOT: `INVALIDBLOCK`).

With this design documented we can tackle the refactor methodically without
re-visiting the rationale.
