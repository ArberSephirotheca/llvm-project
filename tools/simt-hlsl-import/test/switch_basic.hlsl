// RUN: %simt-hlsl-import %s | %mlir-file-check --check-prefix=MLIR %S/switch_basic.mlir

[numthreads(1, 1, 1)]
void main(uint tid : SV_DispatchThreadID) {
  uint value = tid;
  uint result = 0;
  switch (value) {
  case 0:
    result = 1;
    break;
  case 1:
    result = 2;
    break;
  default:
    result = 3;
    break;
  }
}
