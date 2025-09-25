// RUN: %simt-hlsl-import %s | %mlir-file-check --check-prefix=MLIR %S/group_memory_barrier_with_group_sync.mlir

[numthreads(1, 1, 1)]
void main(uint tid : SV_DispatchThreadID) {
  GroupMemoryBarrierWithGroupSync();
}
