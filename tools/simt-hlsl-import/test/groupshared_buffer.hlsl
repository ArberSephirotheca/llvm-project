// RUN: %simt-hlsl-import %s | %mlir-file-check --check-prefix=MLIR %S/groupshared_buffer.mlir

groupshared int sharedData[32];

[numthreads(1, 1, 1)]
void main(uint tid : SV_DispatchThreadID) {
  sharedData[tid] = tid;
  int value = sharedData[tid];
}
