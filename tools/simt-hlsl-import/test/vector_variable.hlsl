// RUN: %simt-hlsl-import %s | %mlir-file-check --check-prefix=MLIR %S/vector_variable.mlir

[numthreads(1, 1, 1)]
void main(uint tid : SV_DispatchThreadID) {
  float4 value;
}
