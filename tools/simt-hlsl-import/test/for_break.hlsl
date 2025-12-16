// RUN: %simt-hlsl-import %s | %mlir-file-check --check-prefix=MLIR %S/for_break.mlir

[numthreads(4, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint acc = 0;
  for (uint i = 0; i < 4; i = i + 1) {
    if (tid.x == 3)
      break;
    acc = acc + i;
  }
}



