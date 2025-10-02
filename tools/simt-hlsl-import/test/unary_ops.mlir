// MLIR-LABEL: module {
// MLIR: func.func @main(
// MLIR: %[[DISPATCH:.*]] = simt_step.dispatch_thread_id : i32
// MLIR: %[[ADD:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
// MLIR: %[[SUB:.*]] = arith.subi %[[ADD]], %{{.*}} : i32
// MLIR: %[[NEG:.*]] = arith.subi %{{.*}}, %[[SUB]] : i32
// MLIR: %[[NOT:.*]] = arith.xori %[[SUB]], %{{.*}} : i32
// MLIR: %[[CMP:.*]] = arith.cmpi eq, %[[SUB]], %{{.*}} : i32
// MLIR: %[[LNOT:.*]] = arith.cmpi eq, %[[CMP]], %false : i1
// MLIR: return
