// RUN: %simt-hlsl-import %s | %mlir-file-check --check-prefix=MLIR %S/while_loop.mlir

[numthreads(1, 1, 1)]
void main(uint tid : SV_DispatchThreadID) {
  uint i = 0;
  uint acc = 0;
  while (i < 4) {
    acc = acc + i;
    i = i + 1;
  }
}
