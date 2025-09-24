// RUN: %simt-hlsl-import %s | %mlir-file-check --check-prefix=MLIR %S/unary_ops.mlir

[numthreads(1, 1, 1)]
void main(uint tid : SV_DispatchThreadID) {
  int counter = 0;
  int pre = ++counter;
  int post = counter--;
  int neg = -counter;
  int bit = ~counter;
  bool flag = (counter == 0);
  bool notFlag = !flag;
}
