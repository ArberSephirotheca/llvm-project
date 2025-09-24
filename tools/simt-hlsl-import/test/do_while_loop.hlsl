// RUN: %simt-hlsl-import %s | %mlir-file-check --check-prefix=MLIR %S/do_while_loop.mlir

[numthreads(1, 1, 1)]
void main(uint tid : SV_DispatchThreadID) {
  uint i = 0;
  uint acc = 0;
  do {
    if (i == 1) {
      i = i + 1;
      continue;
    }
    if (i == 3) {
      break;
    }
    acc = acc + i;
    i = i + 1;
  } while (i < 5);
}
