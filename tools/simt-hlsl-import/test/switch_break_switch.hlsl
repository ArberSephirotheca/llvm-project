// RUN: %simt-hlsl-import %s | %mlir-file-check --check-prefix=MLIR %S/switch_break_switch.mlir

[numthreads(1, 1, 1)]
void main(uint tid : SV_DispatchThreadID) {
  uint sum = 0;
  for (uint i = 0; i < 4; i = i + 1) {
    switch (i) {
    case 0:
      sum = sum + 1;
      break;
    case 1:
      sum = sum + 2;
      break;
    default:
      sum = sum + 4;
      break;
    }
  }
}
