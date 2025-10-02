// RUN: %simt-hlsl-import %s | %mlir-file-check --check-prefix=MLIR %S/thread_ids.mlir

[numthreads(2, 3, 4)]
void main(uint3 localId : SV_GroupThreadID,
          uint3 groupId : SV_GroupID,
          uint groupIndex : SV_GroupIndex,
          uint3 dispatchId : SV_DispatchThreadID) {
  uint guard = localId.x + groupId.y + groupIndex + dispatchId.z;
  if (guard == 0)
    return;
}
