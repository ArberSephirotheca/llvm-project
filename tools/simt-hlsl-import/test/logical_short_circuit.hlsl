// RUN: %simt-hlsl-import %s | %mlir-file-check --check-prefix=MLIR %S/logical_short_circuit.mlir

[numthreads(1, 1, 1)]
void main(uint tid : SV_DispatchThreadID) {
  uint tmp = 0;
  bool useAnd = (tid == 0) && ((tmp = 1) == 1);
  bool useOr = (tid == 0) || ((tmp = 2) == 2);
}
