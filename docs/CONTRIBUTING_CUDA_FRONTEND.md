# Contributing to the SIMT-Step CUDA Frontend

Welcome aboard! This guide is aimed at new contributors—especially students—who
want to help build the CUDA importer for SIMT-Step. It pulls together the key
context, tooling steps, and bite-sized tasks so you can be productive quickly.

---

## 1. Project Overview

SIMT-Step provides an MLIR dialect (`simt_step`) that captures SIMT execution
explicitly, plus a lowering pipeline that turns structured control flow into the
mask-aware form our interpreter understands. We already have an HLSL importer
(`tools/simt-hlsl-import`) producing this dialect; the CUDA frontend will follow
the same architecture:

1. Parse CUDA via Clang.
2. Translate the AST into our tagless-final lowering algebra (shared between all
   frontends).
3. Emit `simt_step` IR for kernels.

The shared frontend infrastructure lives under:

- `include/simt-step/Frontends/Common/` – headers for lowering algebra, loop/switch
  support, result helpers.
- `lib/frontends/common/` – implementations (e.g., loop scope support).

The CUDA-specific pieces will live under `lib/frontends/` and
`tools/simt-cuda-import/` (prototype name—you can iterate on it).

---

## 2. Environment & Build Checklist

1. **Clone the repo and LLVM toolchain dependency**
   ```bash
   git clone https://github.com/<your-account>/SIMT-Step.git
   ```

2. **Toolchain dependency** – we expect an LLVM/Clang/MLIR toolchain with DXIL
   enabled at `/opt/llvm-hlsl` (see `docs/BUILD.md` for full cmake invocations).
   Install it or tweak `LLVM_DIR`/`MLIR_DIR` in your build if the path differs.

3. **Configure and build**
   ```bash
   cmake -S . -B build -DLLVM_DIR=/opt/llvm-hlsl/lib/cmake/llvm \
         -DMLIR_DIR=/opt/llvm-hlsl/lib/cmake/mlir -DCMAKE_BUILD_TYPE=RelWithDebInfo
   cmake --build build
   ```

4. **Run tests (serially if sandboxed)**
   ```bash
   cmake --build build --target simt-hlsl-import
   build/tools/simt-hlsl-import/simt-hlsl-import tools/simt-hlsl-import/test/simple.hlsl
   ```
   Once the CUDA frontend emerges, we will mirror these tests with CUDA kernels.

---

## 3. Codebase Tour for the CUDA Frontend

| Path | Purpose |
| ---- | ------- |
| `lib/frontends/CUDA.cpp` | Current placeholder—returns an empty module. |
| `include/simt-step/frontends/Common/*` | Tagless-final lowering algebra & loop/switch helpers shared across frontends. |
| `tools/simt-hlsl-import/` | Useful reference implementation (HLSL importer) showing how to wire interpretations, diagnostics, and tests. |
| `docs/LOWERING_ALGEBRA_DESIGN.md` | Design rationale behind the algebra you will reuse. |
| `docs/HLSL_IMPORT_GAP.md` | Example of keeping a “gap list” between frontends and existing interpreters—do the same for CUDA. |

---

## 4. Suggested First Tasks

1. **Bootstrap the tool skeleton**
   - Add a new executable target (`tools/simt-cuda-import/`) mirroring the layout
     of the HLSL importer.
   - Link against `simt-frontends-common`, `simt-step`, Clang libraries.
   - Keep a simple `translateCudaToMLIR` function returning an empty module until
     lowering code lands.

2. **Hook Clang to parse CUDA**
   - Look at `FunctionLoweringVisitor` in the HLSL importer for reference.
   - Drive Clang in CUDA mode (pass `-x cuda` and relevant `--cuda-gpu-arch`
     flags).
   - Dump the AST or use Clang’s diagnostics to ensure kernels are discovered.

3. **Lower straight-line kernels**
   - Start from a trivial kernel (`__global__ void add(int* out, int v)`).
   - Use the shared lowering algebra to emit `simt_step.func` with loads,
     stores, and arithmetic. Focus on `DeclRefExpr`, literals, and assignments.

4. **Add regression tests**
   - Mirror the HLSL workflow (see `docs/test_workflow.md`): pair each CUDA
     kernel (`*.cu`) with a `*.mlir` FileCheck file.
   - Add a lit config (similar to `tools/simt-hlsl-import/test/CMakeLists.txt`).

5. **Track missing features**
   - Draft `docs/TODO_CUDA_IMPORT.md` to document gaps (control flow, intrinsics,
     memory spaces, etc.). This helps future contributors pick a task.

---

## 5. Development Tips

- **Stay tagless-final** – don’t call `OpBuilder` directly from shared helpers.
  Always go through the algebra so analysis/emit modes stay aligned.
- **Reuse HLSL patterns** – the HLSL importer already solved loop carried values,
  switch lowering, wave/op intrinsics. Port functionality gradually, keeping
  tests in sync.
- **Lean on design docs** – `docs/structured_cfg_mask_model.md`,
  `docs/loop_merge_plan.md`, `docs/gpu_interpreter_design.md` explain the lowering
  expectations for dynamic blocks.
- **Prefer small, reviewable patches** – Introduce new functionality with tests,
  update the CUDA TODO doc, and keep commits focused.
- **Ask questions** – File issues or comment on PRs when a design choice is
  unclear. We value communication over guesswork.

---

## 6. Checklist Before Sending a PR

1. `cmake --build build --target simt-cuda-import` (or your CUDA tool target).
2. Run relevant tests (for now, HLSL regression tests and any CUDA ones you add).
3. Update documentation/TODO lists if behaviour changed or work remains.
4. Ensure `git clang-format` (or `ninja format` once configured) covers touched
   files.
5. Write a concise commit message and PR description summarising:
   - What changed
   - Tests added or run
   - Remaining follow-up tasks (if any)

---

## 7. Staying in Touch

- **Issues/Discussions** – use the project issue tracker for bugs and design
  proposals. Tag them with “CUDA frontend” for discoverability.
- **PR Reviews** – expect feedback on code style, test coverage, and alignment
  with the lowering design. We’re happy to mentor—just push small, iterative
  PRs so we can respond quickly.
- **Learning Resources** – MLIR’s “Toy” tutorial, Clang’s LibTooling docs, and
  NVIDIA’s CUDA documentation pair well with this work.

Welcome to the project! We’re excited to have you contribute to the CUDA
frontend and grow the SIMT-Step tooling ecosystem.

