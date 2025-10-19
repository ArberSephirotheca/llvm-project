# HLSL Toolchain Notes

The upstream Clang tree only emits DXIL/SPIR-V when the corresponding LLVM
backends are built. If you see Clang failing with `HLSL code generation is
unsupported for target 'dxil-…'`, rebuild the toolchain with the DXIL targets
enabled. Throughout this guide we assume you exported `LLVM_PREFIX` to the
desired install location (see `docs/BUILD.md`):

```bash
export LLVM_PREFIX=/path/to/llvm-install
```

Then configure and build Clang/LLVM with HLSL enabled:

```bash
cmake \
  -DLLVM_ENABLE_PROJECTS="clang;mlir" \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="DirectX" \
  -DCLANG_ENABLE_HLSL=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$LLVM_PREFIX \
  -S llvm-project/llvm \
  -B llvm-project/build
cmake --build llvm-project/build --target install --parallel
```

Once installed, point our build at `$LLVM_PREFIX` and the importer will see a
DXIL-capable Clang.
