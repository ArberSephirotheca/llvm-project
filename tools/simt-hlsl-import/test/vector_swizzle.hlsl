// RUN: %simt-hlsl-import %s | %mlir-file-check --check-prefix=MLIR %S/vector_swizzle.mlir

[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint x = tid.x;
  uint2 yz = tid.yz;
}
