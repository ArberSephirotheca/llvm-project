// RUN: %simt-hlsl-import %s | %mlir-file-check --check-prefix=MLIR %S/switch_fallthrough.mlir

[numthreads(1, 1, 1)]
void main(uint tid : SV_DispatchThreadID) {
  uint value = tid;
  uint result = 10;
  switch (value) {
  case 0:
    result = result + 1;
  case 1:
    result = result + 2;
    break;
  case 2:
    result = result + 4;
  default:
    result = result + 8;
    break;
  }
}
