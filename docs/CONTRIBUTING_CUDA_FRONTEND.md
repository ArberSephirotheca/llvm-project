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
the same architecture. At the moment `simt-hlsl-import` is the only fully
functional importer—treat it as the reference when you want to see how HLSL
lowering translates into `simt_step` MLIR:

```bash
./build/tools/simt-hlsl-import/simt-hlsl-import tools/simt-hlsl-import/test/simple.hlsl
```

The CUDA frontend will reuse the same lowering algebra:

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

2. **Toolchain dependency** – install an LLVM/Clang/MLIR toolchain with DXIL
   enabled. Export the install location via `LLVM_PREFIX` (matching `docs/BUILD.md`):
   ```bash
   export LLVM_PREFIX=/path/to/llvm-install
   ```
   Adjust `LLVM_DIR` / `MLIR_DIR` if you choose a different prefix.

3. **Configure and build**
   ```bash
   cmake -S . -B build -DLLVM_DIR=$LLVM_PREFIX/lib/cmake/llvm \
         -DMLIR_DIR=$LLVM_PREFIX/lib/cmake/mlir -DCMAKE_BUILD_TYPE=RelWithDebInfo
   cmake --build build
   ```

4. **Run tests**
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

4. **Add regression tests**
   - Mirror the HLSL workflow: pair each CUDA
     kernel (`*.cu`) with a `*.mlir` FileCheck file.
   - Add a lit config (similar to `tools/simt-hlsl-import/test/CMakeLists.txt`).

5. **Track missing features**
   - Draft `docs/TODO_CUDA_IMPORT.md` to document gaps (control flow, intrinsics,
     memory spaces, etc.). This helps future contributors pick a task.

---

## 5. Development Tips

- **Stay tagless-final** – don’t call `OpBuilder` directly from shared helpers.
  Always go through the algebra so analysis/emit modes stay aligned.
