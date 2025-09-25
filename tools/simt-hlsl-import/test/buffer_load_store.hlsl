// RUN: %simt-hlsl-import %s | %mlir-file-check --check-prefix=MLIR %S/buffer_load_store.mlir

RWBuffer<int> gData : register(u0);

[numthreads(1, 1, 1)]
void main(uint tid : SV_DispatchThreadID) {
  int value = gData[tid];
  gData[tid] = value + 1;
}
