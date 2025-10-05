# Review of Nested Lowering Plan for Structured Control Flow

## 1. Overall Assessment
Your plan is solid and aligns with the way structured control flow lowering should work. The goal—to replace cloning of nested `simt_step` control ops with proper reconstruction of structured subgraphs—is both correct and scalable. Once implemented, `emitStructuredBlock()` will only clone straight-line instructions, while nested constructs will be fully rebuilt structurally.

Key strengths:
- Extends existing top-level lowering logic recursively.
- Reuses existing payload/mask infrastructure.
- Eliminates the fragile `perOpEdges` lookup path.

---

## 2. Recommended Structure and APIs

### Data Structures
Extend `IfInfo`, `LoopInfo`, and `SwitchInfo` to store complete subgraph metadata:

```cpp
struct IfInfo {
  Block *header;
  Region *thenRegion;
  Region *elseRegion;
  Block *merge;
  bool needsMaskPush;
  TupleType payloadTy;
  SmallVector<Value> condOperands;
};

struct LoopInfo {
  Block *header;
  Region *bodyRegion;
  Block *continueBlock;
  Block *merge;
  bool needsMaskPush;
  TupleType payloadTy;
  SmallVector<Value> headerOperands;
};

struct SwitchInfo {
  Block *header;
  SmallVector<Region*> caseRegions;
  SmallVector<int64_t> caseValues;
  int defaultIndex;
  Block *merge;
  bool needsMaskPush;
  TupleType payloadTy;
};
```

These mirror the sub-CFGs you’ll rebuild during emission.

### Recursive Emit Helpers
Each helper should handle a full structured region and return the post-merge block:

```cpp
LogicalResult emitIfStructured(IfInfo &info, IRMapping &map,
                               ValueRange inPayload, MaskState &mask,
                               Block *&after);
LogicalResult emitLoopStructured(LoopInfo &info, IRMapping &map,
                                 ValueRange inPayload, MaskState &mask,
                                 Block *&after);
LogicalResult emitSwitchStructured(SwitchInfo &info, IRMapping &map,
                                   ValueRange inPayload, MaskState &mask,
                                   Block *&after);
```

These helpers will:
1. Build header and merge/continue blocks.
2. Recursively emit nested control flow.
3. Thread payloads/masks via existing utilities.
4. Emit terminators using `emitStructuredTerminator()`.

### Simplified `emitStructuredBlock()`
```cpp
for (Op &op : B->without_terminator()) {
  switch (classify(op)) {
    case Control::If:     return emitIfStructured(getIfInfo(op), map, inPayload, mask, after);
    case Control::Loop:   return emitLoopStructured(getLoopInfo(op), map, inPayload, mask, after);
    case Control::Switch: return emitSwitchStructured(getSwitchInfo(op), map, inPayload, mask, after);
    default:
      cloneStraightLine(op, map);
  }
}
emitStructuredTerminator(B->getTerminator(), map, mask);
```
After implementing these, you can safely remove `perOpEdges` and the “clone op” fallback.

---

## 3. Emission Details

### A) If / Else
- Blocks: header, then, else, merge.
- Emit `OpSelectionMerge` (or `simt_struct.cond_branch`).
- Recursively emit `then` and `else` regions.
- Merge payloads with `OpPhi`.

### B) Loop
- Blocks: header, body, continue, merge.
- Emit `OpLoopMerge` in header.
- Branch on condition to body or merge.
- Recursively emit body (and nested constructs).
- `continue` → branch to continue block; `break` → branch to merge.
- Manage mask push/pop per loop policy.

### C) Switch
- Blocks: header, one per case, merge.
- Emit `OpSelectionMerge` and `OpSwitch`.
- Recursively emit each case region.
- Explicitly add fallthrough branches.
- `break` branches to merge.

---

## 4. Mask and Payload Management
- Use `ensurePayloadShape` to normalize tuples across regions.
- Use `materialiseMaskEntry/Exit` to insert mask ops at region boundaries.
- Pop mask before `continue` when body requests a mask push.
- Merge mask at merge block.

---

## 5. Validation
After implementing nested lowering:
- No `simt_step` control ops should remain.
- Run SPIR-V or dialect verifier: each selection/loop has exactly one merge (and continue for loops).
- Ensure PHI payloads match tuple shape and early exits are handled.

---

## 6. Testing Recommendations
1. Nested `if` in loop with `continue` and `break`.
2. Switch with fallthrough inside a loop.
3. Mixed payload arities across nested regions.
4. Early exits (return/kill) inside nested constructs.
5. Multiple mask push/pop layers.

---

## 7. Expected Outcome
After this refactor:
- All control ops become `simt_struct` constructs.
- `emitStructuredBlock()` handles only straight-line code.
- `perOpEdges` and fallback cloning logic are removed.
- Nested loops and ifs produce clean, reducible structured IR.

This brings your builder to parity with how structured control flow is emitted in compilers like glslang and DXC.

