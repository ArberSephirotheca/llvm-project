// RUN: %simt-hlsl-import %s | %mlir-file-check --check-prefix=MLIR %S/wave_active_count_bits.mlir

[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint count = WaveActiveCountBits(tid.x == 0);
  if (count == 0)
    return;
}
