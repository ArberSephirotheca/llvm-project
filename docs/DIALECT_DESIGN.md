# SIMT-Step Dialect Design

## Execution Model
- Warp/subgroup is the primary execution context.
- Every operation executes under an explicit lane mask and subgroup width preserved in the semantics context.
- Scope hierarchy (thread → subgroup → workgroup) is first-class so ordering, visibility, and resource constraints are unambiguous.

## Types & Attributes
- **Mask / lane-aware types**: custom types encode whether values are per-lane or collective and expose lane counts when available.
- **ScopeAttr**: enumerates `thread`, `subgroup`, `workgroup`.
- **MemSemAttr**: `none`, `acquire`, `release`, `acq_rel` for memory ordering.
- **MemorySpaceAttr**: identifies the space (`global`, `shared`, `local`, extensible string).
- Optional attributes for reduction kind, custom collective metadata, and constant masks used during specialization.

## Traits & Interfaces
- **SimtIndependentOp**: pure per-lane behavior, no cross-lane coordination.
- **SimtSynchronizedOp**: rendezvous semantics (barrier, fence) with accessors for scope and ordering.
- **SimtCollectiveOp**: wave-wide behavior (ballot, shuffle, reductions) with helpers for required scope, uniform mask expectations, and needed resources.
- **SimtMaskModifier**: ops that push/pop or adjust the active mask.
- **SimtPluginOpInterface**: allows `simt.custom` to expose the same APIs (traits, resource requirements, side effects) via registry metadata.

## Built-in Operations
- Structured control flow regions (`scf.if`, `scf.while`, dialect-specific ops) are the user-facing form. Mask/continuation ops are introduced only by the lowering pass that prepares IR for execution/analysis.
- Synchronization (`barrier`, `fence`) with scope and memory semantics attributes.
- Collectives (`wave.ballot`, `wave.all`, `wave.any`, `wave.shuffle`, `wave.reduce`).
- Memory operations with explicit space and ordering metadata.
- State queries (`lane.id`, `subgroup.width`, `active.mask`).
- `simt.custom` for plugin-defined instructions carrying traits and parameter blobs.

## Verification & Analysis
- Ensure mask stack pushes/pops balance across structured control flow.
- Validate synchronized ops are reachable by all active lanes given the CFG.
- Require collectives to have valid subgroup width or be marked width-agnostic.
- Cross-check `simt.custom` metadata (operands, traits, resources) against the registry.
- Memory ops participate in MLIR’s memory effect interface and enforce legal scope/ordering combinations.

## Lowering Strategy
- User IR stays in structured control flow; a dedicated lowering pass (or interpreter layer) materializes dynamic block frames and explicit mask transitions similar to the MiniHLSL interpreter.
- Specialization pass folds constant masks/values and simplifies collectives when possible.
- Canonicalization cleans redundant mask operations and hoists invariants.
- LLVM lowering translates traits to concrete intrinsics/runtime calls (`llvm.nvvm.barrier0`, shuffles, etc.).

## Plugin Integration
- Registry metadata declares instruction traits (independent/synchronized/collective), operand/result schema, side effects, and resource requirements.
- `simt.custom` implements `SimtPluginOpInterface` to expose the metadata uniformly.
- Plugin verifiers ensure custom ops obey declared contracts; the interpreter dispatches via the registry.

## Documentation & Debugging
- Ship pseudo-code descriptions for built-in ops so interpreter and lowerings stay aligned.
- Provide debug metadata hooks for tracing back to source constructs.
- Offer plugin author guidelines on trait selection and verification practices.

---

# Dialect Implementation TODO

1. **TableGen Scaffolding**
   - Define `SimtStepDialect` in TableGen with core traits/interfaces.
   - Generate op classes for built-ins (mask ops, collectives, sync ops).
2. **Trait Infrastructure**
   - Implement C++ trait interfaces (`SimtIndependentOpTrait`, `SimtSynchronizedOpTrait`, `SimtCollectiveOpTrait`, `SimtMaskModifierTrait`).
   - Connect `simt.custom` to the registry via `SimtPluginOpInterface`.
3. **Verifier & Analysis Passes**
   - Mask balance verifier (push/pop parity, structured divergence checks).
  - Synchronization reachability verification.
  - Divergence/uniformity analysis utilities.
4. **Specialization & Canonicalization**
   - Port specialization pass to fold constant masks and subgroup widths.
   - Add canonicalization patterns for redundant mask ops and collectives.
5. **Lowering Pipeline**
   - Implement lowering from structured control flow to explicit mask/dynamic block form for interpreter.
   - Provide LLVM dialect lowering that maps traits to intrinsics/runtime calls.
6. **Interpreter Integration**
   - Update interpreter dispatch to query traits/interfaces instead of hard-coded names.
   - Handle dynamic block construction per MiniHLSL strategy.
7. **Plugin Tooling**
   - Extend registry to emit/consume dialect metadata (traits, resources) automatically.
   - Provide sample plugin definitions demonstrating new interfaces.
8. **Testing & Examples**
   - Author MLIR lit tests covering each op category and verification rule.
   - Add end-to-end tests using `simt-run` with plugins to validate interpreter behavior.
9. **Documentation**
   - Integrate dialect reference into docs and link from `Design.md`.
   - Provide plugin authoring guide aligned with the new trait system.
