// RUN: %simt-hlsl-import %s | %mlir-file-check --check-prefix=MLIR %S/if_for.mlir

[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint acc = 0;
  if (acc == 0) {
    for (uint i = 0; i < 4; i = i + 1) {
      acc = acc + i;
    }
  }
}
