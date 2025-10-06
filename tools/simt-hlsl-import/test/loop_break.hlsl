// RUN: %simt-hlsl-import %s | %mlir-file-check --check-prefix=MLIR %S/loop_break.mlir

[numthreads(1, 1, 1)]
void main(uint input : SV_DispatchThreadID) {
  uint value = input;
  for (uint iter = 0; iter < 1; iter = iter + 1) {
    value = value;
    break;
  }
}
