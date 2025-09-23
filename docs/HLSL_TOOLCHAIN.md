# HLSL Toolchain Notes

The upstream Clang tree only emits DXIL/SPIR-V when the corresponding LLVM
backends are built. If you see Clang failing with `HLSL code generation is
unsupported for target 'dxil-…'`, rebuild the toolchain with the DXIL targets
enabled:

```bash
cmake \
  -DLLVM_ENABLE_PROJECTS="clang;mlir" \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="DirectX" \
  -DCLANG_ENABLE_HLSL=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/llvm-hlsl \
  -S llvm-project/llvm \
  -B llvm-project/build
cmake --build llvm-project/build --target install --parallel
```

Once installed, point our build at `/opt/llvm-hlsl` and the importer will see a
DXIL-capable Clang.
