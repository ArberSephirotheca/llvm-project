// RUN: %simt-hlsl-import %s | %mlir-file-check --check-prefix=MLIR %S/simple.mlir

[numthreads(8, 4, 1)]
void main(uint tid : SV_DispatchThreadID) {
  uint a = 42;
  uint b = 13;
  uint c = a + b * 2;
  c = c - tid;
}
