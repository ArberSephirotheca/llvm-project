// RUN: %simt-hlsl-import %s | %mlir-file-check %S/for_loop.mlir

[numthreads(1, 1, 1)]
void main(uint tid : SV_DispatchThreadID) {
  uint acc = 0;
  for (uint i = 0; i < 3; i = i + 1) {
    acc = acc + i;
  }
}
