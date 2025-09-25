// RUN: %simt-hlsl-import %s | %mlir-file-check --check-prefix=MLIR %S/wave_intrinsics.mlir

[numthreads(1, 1, 1)]
void main(uint tid : SV_DispatchThreadID) {
  bool all_true = WaveActiveAllTrue(tid == 0);
  bool any_true = WaveActiveAnyTrue(tid != 0);
  uint lane = WaveGetLaneIndex();

  if (all_true && any_true && lane == 0)
    return;
}
