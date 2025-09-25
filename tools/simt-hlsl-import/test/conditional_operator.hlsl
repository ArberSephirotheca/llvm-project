// RUN: %simt-hlsl-import %s | %mlir-file-check --check-prefix=MLIR %S/conditional_operator.mlir

[numthreads(1, 1, 1)]
void main(uint tid : SV_DispatchThreadID) {
  int value = 0;
  int result = (tid == 0) ? ++value : (value - 1);
  int after = value;
}
