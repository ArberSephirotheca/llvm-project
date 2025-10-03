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
  explicit StructuredCFGBuilder(func::FuncOp func);
  LogicalResult build();
private:
  LogicalResult analyseBlocks();
  LogicalResult computePayloads();
  LogicalResult emitStructuredBlocks();
  void emitTerminator(const EdgeInfo &edge,
                      simt::structured::BlockOp destBlock);

  func::FuncOp func;
  DenseMap<Block *, BlockInfo> blockInfos;
  SmallVector<EdgeInfo> edges;
  IRMapping mapper;
};
```

## Migration Plan
1. Introduce the builder alongside the existing lowering.
2. Port simple cases first (straight-line + `if`).
3. Gradually move loop/switch handling into the builder.
4. Delete legacy helpers once all tests pass.
5. Document the new pipeline and add regression tests for loop/switch interactions.

