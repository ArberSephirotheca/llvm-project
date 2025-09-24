// RUN: %simt-hlsl-import %s | %mlir-file-check --check-prefix=MLIR %S/switch_loop_break.mlir

[numthreads(1, 1, 1)]
void main(uint tid : SV_DispatchThreadID) {
  uint sum = 0;
  switch (tid) {
  case 0:
    for (uint j = 0; j < 3; j = j + 1) {
      sum = sum + j;
      break;
    }
    break;
  default:
    sum = sum + 5;
    break;
  }
}
