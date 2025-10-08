# Structured CFG Mask Model – SSA First

This note sketches how to migrate the Simt-Step structured lowering away from
the current `mask_push` / `mask_pop` scaffolding ("Option A") toward a pure
SSA mask representation ("Option B"), while preserving the ability to recover
the stack semantics for interpreters or back-ends that expect it.

## Motivation

- The push/pop form mirrors SPIR-V’s reconvergence stack but makes mid-level
  reasoning awkward (unused pops, missing merge pops, brittle TODOs).
- Analysis of “dynamic blocks” really depends on the active mask per block,
  not on the existence of push/pop instructions.
- SSA masks make divergence explicit and simplify payload / terminator logic;
  we can still re-synthesise push/pop later if a target requires it.

## High-Level Plan

1. Treat the active mask entering each structured block as an SSA value `M_B`.
2. Propagate masks through the structured CFG using boolean algebra.
3. Classify blocks as dynamic vs uniform by inspecting `M_B`.
4. Emit structured IR using these SSA masks (no implicit stack while analysing).
5. Optionally reconstruct push/pop at the end for interpreters/back-ends.

## 1. SSA Mask Propagation

For every block `B`, introduce an SSA expression `M_B` describing the lanes that
execute `B`. Initial conditions:

- Entry block `M_entry = WarpMask`.
- For a fall-through edge `P → S`: `M_S := simplify(M_P)`.
- For `cond_branch` out of header `H` with condition `cond_H`:
  - True edge: `M_T := simplify(M_H ∧ cond_H)`
  - False edge: `M_F := simplify(M_H ∧ ¬cond_H)`
- For `switch` cases, use the case masks: `M_case := simplify(M_pred ∧ C_i)`.
- For merges: `M_merge := simplify(∨ incoming masks)`.
- For loop headers: `M_header := φ(M_pre, M_back)` and iterate to fixed point.

`simplify` performs constant folding: `FULL ∧ X = X`, `X ∧ X = X`,
`X ∧ ¬X = EMPTY`, `X ∨ X = X`, `X ∨ ¬X = FULL`, etc.

## 2. Block Classification

Once all `M_B` are computed:

- `B` is **uniform** if `M_B` simplifies to the full warp mask.
- `B` is **unreachable** if `M_B = EMPTY`.
- Otherwise `B` is **dynamic** (may execute under a proper subset of lanes).

This directly mirrors the “dynamic block” notion from SIMT-Step: a block
creates distinct dynamic instances at run time whenever its mask corresponds to
different subsets of lanes (e.g., loop iterations, divergent arms).

## 3. Changes to `StructuredCFGBuilder`

1. **Analysis (`analyseBlocks`)** stays – we still record structured CFG
   metadata (headers, merges, loop info) and branch guards.
2. **New mask propagation stage** computes `M_B` for each `BlockInfo` using the
   rules above (worklist / fixed point).
3. **Structured emission** now threads mask SSA values instead of inserting
   `mask_push`/`mask_pop`. Each block’s mask argument represents `M_B`; the
   true/false edges of a conditional branch pass the corresponding SSA masks.
4. **Drop push/pop ops** from the mid-level IR (or keep them only for the final
   codegen stage). Any remaining safeguard logic uses SSA masks directly.
5. **Optional**: store `merge_target` / `continue_target` metadata so a later
   pass can re-insert push/pop if required.

## 4. Reconstructing the Stack (Optional)

If a downstream consumer wants the stack semantics, run a late pass that:

- At headers, inserts `mask_push(M_header)` using stored merge/continue attrs.
- At successor entries, re-inserts `mask_pop`/`mask_merge` patterns derived
  from the SSA expressions.
- At merge blocks, pops and merges the recorded mask to recover the lanes.

This pass can also remove the explicit SSA mask arguments if the back-end no
longer needs them.

## 5. Example (Nested `if`)

For the nested `if` that checks `(lane & 1)` and `(lane & 2)`:

```
M_entry       = FULL
M_block3      = FULL ∧ cond0
M_block2      = FULL ∧ ¬cond0
M_block7      = FULL ∧ cond0 ∧ cond1
M_block4      = FULL ∧ cond0 ∧ ¬cond1
M_block3.merge = simplify(M_block7 ∨ M_block4) = FULL ∧ cond0
M_block1      = simplify(M_block2 ∨ M_block3.merge) = FULL
```

Thus `block3/4/7` are dynamic; `block1` is uniform. Runtime still creates new
dynamic block instances per iteration (loops) or branch path, but the compiler
can reason about lane subsets directly from `M_B`.

## 6. Implementation Checklist

- Define a `MaskExpr` AST (Full, Empty, Var, And, Or, Not) with `simplify`.
- Extend `EdgeInfo` to carry guard expressions for mask propagation.
- Add `MaskExpr currentMask` (SSA) and `bool isDynamic` to `BlockInfo`.
- Update structured emission so `simt_struct.cond_branch` takes mask operands
  directly instead of relying on push/pop.
- Remove the temporary push/pop, or guard them behind a final reconstruction
  pass for interpreters.
- Re-run tests, add new ones for mask propagation (loops, nested branches).

## 7. Loop Handling Notes

- Use φ-style joins for loop headers (`M_header := φ(M_pre, M_back)`) and
  iterate until masks stabilise.
- Continue targets remain as metadata; optional push/pop reconstruction can use
  them.
- Each loop iteration still constitutes a dynamic block when `M_body` simplifies
  to something other than `FULL`.

## 8. Benefits

- Cleaner structured emission: no more unused `mask_pop` values or missing
  merge pops.
- Easier analysis: dynamic blocks are identified by SSA reasoning (`M_B`).
- Push/pop semantics can still be recovered for codegen/interpreters.
- Leaves room to reuse SSA masks for further optimisations (dead subblocks,
  uniform control flow elimination, etc.).

