// RUN: %simt-hlsl-import %s | %mlir-file-check --check-prefix=MLIR %S/if_statement.mlir

[numthreads(1, 1, 1)]
void main(uint tid : SV_DispatchThreadID) {
  uint value = 0;
  if (tid == 0) {
    value = 1;
  }
}
