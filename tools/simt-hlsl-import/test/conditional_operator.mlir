// MLIR-LABEL: func.func @main(
// MLIR: %[[COND:.*]] = arith.cmpi eq, %{{.*}}, %{{.*}} : i32
// MLIR: %[[RESULT:.*]]:2 = "simt_step.if"(%[[COND]]) ({
// MLIR:   %[[INC:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
// MLIR:   "simt_step.yield"(%[[INC]], %[[INC]]) : (i32, i32) -> ()
// MLIR: }, {
// MLIR:   %[[DEC:.*]] = arith.subi %{{.*}}, %{{.*}} : i32
// MLIR:   "simt_step.yield"(%[[DEC]], %{{.*}}) : (i32, i32) -> ()
// MLIR: }) : (i1) -> (i32, i32)
// MLIR: return
