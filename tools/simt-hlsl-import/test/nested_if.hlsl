// RUN: %simt-hlsl-import %s | %mlir-file-check --check-prefix=MLIR %S/nested_if.mlir

[numthreads(1, 1, 1)]
void main(uint tid : SV_DispatchThreadID) {
  int result = 0;
  // lane 1-3
  bool cond0 = tid != 0u;
  // lane 0
  bool cond1 = tid == 0u;
  if (cond0) {
    if (cond1)
      result = 1;
    else
      result = 2;
  } else {
    result = 0;
  }
}
