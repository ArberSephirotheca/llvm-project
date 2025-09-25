// RUN: %simt-hlsl-import %s | %mlir-file-check --check-prefix=MLIR %S/buffer_atomic_methods.mlir

RWBuffer<int> gData : register(u0);

[numthreads(1, 1, 1)]
void main(uint tid : SV_DispatchThreadID) {
  int oldExchange;
  InterlockedExchange(gData[tid], 2, oldExchange);

  int oldCompare;
  InterlockedCompareExchange(gData[tid], tid, tid + 3, oldCompare);

  int oldMin;
  InterlockedMin(gData[tid], tid, oldMin);

  int oldMax;
  InterlockedMax(gData[tid], tid, oldMax);

  int oldAnd;
  InterlockedAnd(gData[tid], tid, oldAnd);

  int oldOr;
  InterlockedOr(gData[tid], tid, oldOr);

  int oldXor;
  InterlockedXor(gData[tid], tid, oldXor);
}
