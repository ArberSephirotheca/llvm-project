# SIMT-Step Semantics Framework (C++20 + MLIR)

## Purpose
A **unified, language-agnostic framework** to represent and execute GPU SIMT semantics in modern C++ using MLIR + LLVM libraries. See `docs/DIALECT_DESIGN.md` for the design sketch, `docs/SimtStepDialect.md` for the `simt_step` surface, and `docs/SimtStructuredDialect.md` for the structured control-flow layer.

- **Inputs (frontends):** HLSL and CUDA, extensible via plugins.  
- **Core:** MLIR `simt_step` dialect + generic `simt.custom` op.  
- **Semantics engines:** swappable effect-handlers (baseline, vendor-like, etc.).  
- **Goal:** one place to define and evolve GPU semantics, independent of language or backend.

---

## Scope (MVP)
- **Frontends:** HLSL and CUDA, with plugin hooks.  
- **Core dialect:** `simt_step` ops + `simt.custom`.  
- **Engines:** Compiled oracle + interpreter.  
- **Plugin system:** register new instructions, provide handlers, add shims.  
- **Custom PLs:** possible via plugins.  

---

## Workflow

```text
        HLSL source             CUDA source             Custom PL (plugin)
   + optional plugin calls   + optional plugin calls    + plugin syntax
            │                         │                        │
            ├─ HLSL frontend ─────────┼─ CUDA frontend ────────┤
            ▼                         ▼                        ▼
                Normalize → MLIR: simt_step + simt.custom
                           (single source of truth)
                     ┌──────────────┬───────────────┐
                     │              │               │
             Compile-time      Compiled oracle    Interpreter
             specialization    (LLVM + runtime)   (effect handlers)
