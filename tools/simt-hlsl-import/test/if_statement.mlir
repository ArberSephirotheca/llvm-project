// MLIR-LABEL: func.func @main(
// MLIR-SAME: %[[TID:arg0]]: i32)
// MLIR: %[[MASK:.*]] = "simt_step.active_mask"() : () -> i64
// MLIR: %[[ZERO0:.*]] = arith.constant 0 : i32
// MLIR: %[[ZERO1:.*]] = arith.constant 0 : i32
// MLIR: %[[COND:.*]] = arith.cmpi eq, %[[TID]], %[[ZERO1]] : i32
// MLIR: %[[VALUE:.*]] = "simt_step.if"(%[[COND]]) ({
// MLIR:   %[[ONE:.*]] = arith.constant 1 : i32
// MLIR:   "simt_step.yield"(%[[ONE]]) : (i32) -> ()
// MLIR: }, {
// MLIR:   "simt_step.yield"(%[[ZERO0]]) : (i32) -> ()
// MLIR: }) : (i1) -> i32
// MLIR: return
