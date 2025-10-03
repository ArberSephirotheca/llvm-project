# Structured CFG Builder Redesign

## Goals
- Replace the incremental loop/switch lowering with a single-pass CFG rewriter.
- Track payloads (loop-carried values + switch bundles) explicitly so every edge forwards the correct tuple.
- Emit structured metadata (merge/continue targets) once, keeping the IR easy to analyse or lift back to HLSL/CUDA.
- Provide a focused helper (`StructuredCFGBuilder`) so `SimtStepToStructured` stays small and debuggable.

## High-Level Architecture

```
SimtStepToStructuredPass::runOnOperation():
  StructuredCFGBuilder builder(func);
  if (failed(builder.build()))
    signalPassFailure();
```

### BlockInfo
- Original block pointer.
- Carried types (original block arguments).
- Payload values recorded for the block (loop-carried + switch results).
- Merge target and continue target (if any).
- Additional metadata (switch case constants, etc.).

### EdgeInfo
- Pointer to source and destination `BlockInfo`.
- Explicit payload tuple required on the edge.
- Kind (plain branch, fallthrough, loop back-edge).

## Build Steps
1. **Analyse Blocks**
   - Walk the original CFG and populate `BlockInfo`.
   - Record loop/switch metadata from the high-level ops.

2. **Compute Payloads**
   - Seed loop headers/switch headers with their initial payload (loop-carried args + switch init values).
   - Propagate switch `yield` payloads to case blocks and fallthrough blocks.
   - Loop bodies inherit the payload of their predecessors and record any carried updates.

3. **Enumerate Edges**
   - For each terminator, create an `EdgeInfo` with the concatenated payload tuple. No patch-up later.

4. **Emit Structured Blocks**
   - Create one `simt_struct.block` per `BlockInfo` with the carried+payload types.
   - Attach merge/continue attributes from the recorded metadata.
   - Clone the original block body into the new block via `IRMapping`.

5. **Emit Terminators**
   - Use the precomputed `EdgeInfo` list to generate `simt_struct.branch`/`cond_branch` terminators. Every operand list is already correct.

6. **Cleanup**
   - Delete the old CFG blocks and leave the structured function in place.

## API Sketch

```c++
class StructuredCFGBuilder {
public:
  explicit StructuredCFGBuilder(FunctionOpInterface func);
  LogicalResult build();
private:
  LogicalResult analyseBlocks();
  LogicalResult computePayloads();
  LogicalResult enumerateEdges();
  LogicalResult emitStructuredBlocks();
  LogicalResult cleanupOriginalCFG();

  LogicalResult emitStructuredBlock(BlockInfo &info);
  LogicalResult emitStructuredTerminator(BlockInfo &source,
                                         const EdgeInfo &edge);

  LogicalResult ensurePayloadShape(EdgeInfo &edge);
  LogicalResult propagatePayload(BlockInfo &source, BlockInfo &dest,
                                 llvm::ArrayRef<Value> values);
  LogicalResult materialiseMaskEntry(BlockInfo &info);
  LogicalResult materialiseMaskExit(BlockInfo &info);

  FunctionOpInterface func;
  llvm::SmallVector<Block *> blockOrder;
  llvm::DenseMap<Block *, BlockInfo> blockInfos;
  llvm::SmallVector<EdgeInfo> edges;
  std::unique_ptr<IRMapping> mapper;
  std::unique_ptr<DominanceInfo> domInfo;
};
```

## Current Skeleton (July 2025)
- The analysis helpers populate `BlockInfo` for every block (including nested
  `simt.if`/`loop`/`switch` regions) and stash coarse metadata in per-op maps so
  later stages can recover case/loop structure without mutating the CFG.
- The header now forward-declares `BlockInfo`, `EdgeInfo`, and operation-specific
  info records so helpers remain private implementation details.
- `build()` wires the staged pipeline (analyse → payload → edges → emit →
  cleanup) but each stage currently returns a `signalUnimplemented` failure to
  keep behaviour identical to the legacy lowering until functionality lands.
- `BlockInfo` records the original block pointer, carried argument types,
  payload seeds, requested mask operations, and the eventual structured block
  handle.
- `EdgeInfo` captures source/destination `BlockInfo` pointers plus the payload
  tuple that must be forwarded along that terminator; an enum distinguishes
  plain branches, conditional arms, and loop back-edges.
- Helper stubs (`analyseIfOp`, `analyseLoopOp`, `materialiseMaskEntry`, etc.)
  are in place with TODO comments that map directly onto the build steps above
  so incremental implementations can focus on one concern at a time.

## Migration Plan
1. Introduce the builder alongside the existing lowering.
2. Port simple cases first (straight-line + `if`).
3. Gradually move loop/switch handling into the builder.
4. Delete legacy helpers once all tests pass.
5. Document the new pipeline and add regression tests for loop/switch interactions.
