// RUN: %simt-hlsl-import %s | %mlir-file-check --check-prefix=MLIR %S/buffer_atomic_add.mlir

RWBuffer<int> gData : register(u0);

[numthreads(1, 1, 1)]
void main(uint tid : SV_DispatchThreadID) {
  uint original;
  InterlockedAdd(gData[tid], 1, original);
}
