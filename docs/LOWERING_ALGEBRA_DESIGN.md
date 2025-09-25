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
enum class BarrierKind { WorkgroupSync, DeviceSync, AllSync };

template <typename Self>
struct LoweringAlgebra {
  using Value = typename Self::ValueType;

  // Expressions
  Value emitConstantInt(int64_t v, llvm::StringRef tag, SourceLoc);
  Value emitConstantFloat(double v, llvm::StringRef tag, SourceLoc);
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
  void emitReturn(Optional<Value>, SourceLoc);

  void emitBarrier(BarrierKind, SourceLoc);
  void emitFence(BarrierKind, llvm::StringRef memSpace, SourceLoc);

  // Diagnostics / tracing
  void trace(llvm::StringRef message, SourceLoc);
  void reportError(SourceLoc, llvm::StringRef message);

  static Self &get(LoweringContext &ctx);
};
```

Notes:

- The algebra is intentionally minimal. We add hooks as we refactor existing
  lowering helpers. Value tags (`llvm::StringRef tag`) are symbolic tokens such
  as `"i32"`, `"f64"`, `"vector<4xf32>"`.
- `Value` is interpreter-specific; the emit interpreter aliases it to
  `mlir::Value` while analysis stores a small `SymValue` (e.g., `{ValueClass,
  Optional<unsigned> elementCount, Optional<bool> isConst}`).
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

  Value emitConstantInt(int64_t v, llvm::StringRef tag, SourceLoc loc);
  // ... build real MLIR ops as today ...
};
```

### AnalysisInterpreter

```cpp
struct SymValue {
  ValueClass kind;
  Optional<unsigned> elementCount;
  Optional<bool> isConst;
  // Additional metadata as needed (e.g., pointer space).
};

struct AnalysisInterpreter : LoweringAlgebra<AnalysisInterpreter> {
  using ValueType = SymValue;

  llvm::DenseMap<const clang::ValueDecl *, SymValue> symbolTable;
  llvm::SmallPtrSet<const clang::ValueDecl *, 8> mutated;
  DiagSink &diag;

  Value emitConstantInt(int64_t v, llvm::StringRef tag, SourceLoc); // record tag
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

  ValueType emitConstantInt(int64_t v, llvm::StringRef tag, SourceLoc loc) {
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
   facts.
2. **Refactor barrier + buffer intrinsics first** – replace direct builder
   calls with algebra calls. Run both interpreters: analysis should produce no
   MLIR; emit should pass existing tests.
3. **Gradually migrate remaining helpers** (expressions, control flow, returns,
   etc.). After each stage run `mlir::verify` + unit tests.
4. **Introduce trace interpreter** for debugging and optional unit tests.
5. **Once fully migrated**, delete direct builder access from lowering helpers.
   The only way to emit IR will be through the algebra, guaranteeing analysis
   walks are side-effect free.

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
