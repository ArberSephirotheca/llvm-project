// RUN: %simt-hlsl-import %s | %mlir-file-check --check-prefix=MLIR %S/for_break_continue.mlir

[numthreads(1, 1, 1)]
void main(uint tid : SV_DispatchThreadID) {
  uint acc = 0;
  for (uint i = 0; i < 4; i = i + 1) {
    if (i == 1)
      continue;
    if (i == 3)
      break;
    acc = acc + i;
  }
}
