# Building SIMT-Step (C++)

The C++ port now expects the in-tree LLVM/MLIR toolchain you installed at `/opt/llvm-hlsl`. Adjust the paths if your environment differs (older binaries under `/opt/llvm-20` continue to work for legacy setups, but the Clang-based HLSL frontend requires the new build).

## Configure

```bash
cmake -S . -B build \
  -DLLVM_DIR=/opt/llvm-hlsl/lib/cmake/llvm \
  -DMLIR_DIR=/opt/llvm-hlsl/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Release
```

The root `CMakeLists.txt` already seeds defaults for these variables when it detects `/opt/llvm-hlsl`, so running `cmake -S . -B build` may be enough if the directory is present.

## Build

```bash
cmake --build build
```

This produces the shared library `libsimt-step` and the command-line tools under `build/tools/`.

- `simt-run` exercises the registry + interpreter infrastructure, loads the built-in `reduce_add` plugin example, and instantiates stub MLIR modules via the CUDA/HLSL frontend helpers.
- `simt-convert` reads source text, runs the selected frontend (`--frontend=hlsl|cuda`), and prints the resulting MLIR module.
- `check-simt-hlsl-import` runs lit-based regression tests for the HLSL
  importer (requires the LLVM build tree’s `llvm-lit`).
- `check-simt-opt` runs the structured conversion/interpreter smoke tests, including the
  `--simt-dump-structured-program` utility pass added to `simt-opt`.

After building you can exercise the structured inspection pass manually:

```bash
build/tools/simt-opt/simt-opt \
  --simt-step-to-structured \
  --simt-dump-structured-program \
  path/to/input.mlir
```

## Environment

Before configuring, ensure your shell uses the Clang toolchain that ships with the same LLVM install:

```bash
export PATH=/opt/llvm-hlsl/bin:$PATH
export CC=/opt/llvm-hlsl/bin/clang
export CXX=/opt/llvm-hlsl/bin/clang++
export LD_LIBRARY_PATH=/opt/llvm-hlsl/lib:$LD_LIBRARY_PATH
```

These settings mirror `.cargo/config.toml` so both the legacy Rust pieces and the new CMake build can share one dependency layout.

### HLSL builtin headers

`simt-hlsl-import` relies on Clang’s bundled HLSL headers (`hlsl.h`, `hlsl_intrinsics.h`, …). When the tool cannot find an installed resource directory (for example on minimal developer machines), it automatically falls back to the checked-in headers under `llvm-project/clang/lib/Headers`. No manual configuration is required unless you want to override the search path—set `SIMT_CLANG_HEADERS_DIR` at build time or export `SIMT_IMPORT_DEBUG_RESOURCE=1` to print the resolved directory.
