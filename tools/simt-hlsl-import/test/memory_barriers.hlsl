// RUN: %simt-hlsl-import %s | %mlir-file-check --check-prefix=MLIR %S/memory_barriers.mlir

[numthreads(1, 1, 1)]
void main(uint tid : SV_DispatchThreadID) {
  GroupMemoryBarrier();
  GroupMemoryBarrierWithGroupSync();
  DeviceMemoryBarrier();
  DeviceMemoryBarrierWithGroupSync();
  AllMemoryBarrier();
  AllMemoryBarrierWithGroupSync();
}
