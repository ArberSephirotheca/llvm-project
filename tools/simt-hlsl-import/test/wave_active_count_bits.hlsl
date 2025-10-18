// RUN: %simt-hlsl-import %s | %mlir-file-check --check-prefix=MLIR %S/wave_active_count_bits.mlir

[numthreads(1, 1, 1)]
void main(uint tid : SV_DispatchThreadID) {
  uint count = WaveActiveCountBits(tid == 0);
  if (count == 0)
    return;
}
