# Building SIMT-Step (C++)

The C++ port expects the LLVM/MLIR 20.0 toolchain installed at `/opt/llvm-20`, matching the prior Rust configuration. Adjust the paths if your environment differs.

## Configure

```bash
cmake -S . -B build \
  -DLLVM_DIR=/opt/llvm-20/lib/cmake/llvm \
  -DMLIR_DIR=/opt/llvm-20/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Release
```

The root `CMakeLists.txt` already seeds defaults for these variables when it detects `/opt/llvm-20`, so running `cmake -S . -B build` may be enough if the directory is present.

## Build

```bash
cmake --build build
```

This produces the shared library `libsimt-step` and the command-line tools under `build/tools/`.

- `simt-run` exercises the registry + interpreter infrastructure, loads the built-in `reduce_add` plugin example, and instantiates stub MLIR modules via the CUDA/HLSL frontend helpers.
- `simt-convert` reads source text, runs the selected frontend (`--frontend=hlsl|cuda`), and prints the resulting MLIR module.

## Environment

Before configuring, ensure your shell uses the Clang toolchain that ships with the same LLVM install:

```bash
export PATH=/opt/llvm-20/bin:$PATH
export CC=/opt/llvm-20/bin/clang
export CXX=/opt/llvm-20/bin/clang++
export LD_LIBRARY_PATH=/opt/llvm-20/lib:$LD_LIBRARY_PATH
```

These settings mirror `.cargo/config.toml` so both the legacy Rust pieces and the new CMake build can share one dependency layout.
