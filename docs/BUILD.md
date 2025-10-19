# Building SIMT-Step (C++)

The C++ tree assumes you have a recent out-of-tree LLVM/MLIR build (21.x or newer). Point CMake at the install prefix you produced when building LLVM. Throughout this guide the prefix is referenced via the shell variable `LLVM_PREFIX`; set it once to avoid hard-coded `/opt/llvm` paths:

```bash
export LLVM_PREFIX=/path/to/your/llvm-install   # e.g. $(pwd)/llvm-install or /opt/llvm
```

If you followed the quick-start scripts that install into `/opt/llvm`, setting `LLVM_PREFIX=/opt/llvm-hlsl` keeps the old behaviour.

## Configure

```bash
cmake -S . -B build \
  -DLLVM_DIR=$LLVM_PREFIX/lib/cmake/llvm \
  -DMLIR_DIR=$LLVM_PREFIX/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Release
```

The root `CMakeLists.txt` still auto-detects `/opt/llvm`; if you export `LLVM_PREFIX` to another location the commands above keep everything in sync.

## Build

```bash
cmake --build build
```

This produces the shared library `libsimt-step` and several command-line tools under `build/tools/`. At the moment **`simt-hlsl-import` is the only fully functional driver** (use it to translate HLSL sources to `simt_step` MLIR). The other executables (`simt-convert`, `simt-run`, `simt-opt`, `simt-step-parse`) are still work in progress or utility stubs, and the `check-*` targets simply exercise their current smoke tests.
```

## Environment

Before configuring, ensure your shell uses the Clang toolchain that ships with the same LLVM install:

```bash
export PATH=$LLVM_PREFIX/bin:$PATH
export CC=$LLVM_PREFIX/bin/clang
export CXX=$LLVM_PREFIX/bin/clang++
export LD_LIBRARY_PATH=$LLVM_PREFIX/lib:$LD_LIBRARY_PATH
```


### HLSL builtin headers

`simt-hlsl-import` relies on Clang’s bundled HLSL headers (`hlsl.h`, `hlsl_intrinsics.h`, …). When the tool cannot find an installed resource directory (for example on minimal developer machines), it automatically falls back to the checked-in headers under `llvm-project/clang/lib/Headers`. No manual configuration is required unless you want to override the search path—set `SIMT_CLANG_HEADERS_DIR` at build time or export `SIMT_IMPORT_DEBUG_RESOURCE=1` to print the resolved directory.
