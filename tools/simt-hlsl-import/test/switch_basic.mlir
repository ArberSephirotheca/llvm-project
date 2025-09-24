// MLIR-LABEL: func.func @main(
// MLIR-SAME: %[[TID:arg0]]: i32)
// MLIR: %[[MASK:.*]] = "simt_step.active_mask"() : () -> i64
// MLIR: %[[ZERO0:.*]] = arith.constant 0 : i32
// MLIR: %[[FALSE:.*]] = arith.constant false
// MLIR: %[[NOT_MATCH0:.*]] = arith.cmpi eq, %[[FALSE]], %[[FALSE]] : i1
// MLIR: %[[ZERO1:.*]] = arith.constant 0 : i32
// MLIR: %[[IS_ZERO:.*]] = arith.cmpi eq, %[[TID]], %[[ZERO1]] : i32
// MLIR: %[[COND0:.*]] = arith.andi %[[NOT_MATCH0]], %[[IS_ZERO]] : i1
// MLIR: %[[CASE0:.*]]:2 = "simt_step.if"(%[[COND0]]) ({
// MLIR:   %[[TRUE0:.*]] = arith.constant true
// MLIR:   %[[ONE:.*]] = arith.constant 1 : i32
// MLIR:   "simt_step.yield"(%[[ONE]], %[[TRUE0]]) : (i32, i1) -> ()
// MLIR: }, {
// MLIR:   "simt_step.yield"(%[[ZERO0]], %[[FALSE]]) : (i32, i1) -> ()
// MLIR: }) : (i1) -> (i32, i1)
// MLIR: %[[NOT_MATCH1:.*]] = arith.cmpi eq, %[[CASE0]]#1, %[[FALSE]] : i1
// MLIR: %[[ONE_CASE:.*]] = arith.constant 1 : i32
// MLIR: %[[IS_ONE:.*]] = arith.cmpi eq, %[[TID]], %[[ONE_CASE]] : i32
// MLIR: %[[COND1:.*]] = arith.andi %[[NOT_MATCH1]], %[[IS_ONE]] : i1
// MLIR: %[[CASE1:.*]]:2 = "simt_step.if"(%[[COND1]]) ({
// MLIR:   %[[TRUE1:.*]] = arith.constant true
// MLIR:   %[[TWO:.*]] = arith.constant 2 : i32
// MLIR:   "simt_step.yield"(%[[TWO]], %[[TRUE1]]) : (i32, i1) -> ()
// MLIR: }, {
// MLIR:   "simt_step.yield"(%[[CASE0]]#0, %[[CASE0]]#1) : (i32, i1) -> ()
// MLIR: }) : (i1) -> (i32, i1)
// MLIR: %[[NOT_MATCH2:.*]] = arith.cmpi eq, %[[CASE1]]#1, %[[FALSE]] : i1
// MLIR: %[[CASE2:.*]]:2 = "simt_step.if"(%[[NOT_MATCH2]]) ({
// MLIR:   %[[TRUE2:.*]] = arith.constant true
// MLIR:   %[[THREE:.*]] = arith.constant 3 : i32
// MLIR:   "simt_step.yield"(%[[THREE]], %[[TRUE2]]) : (i32, i1) -> ()
// MLIR: }, {
// MLIR:   "simt_step.yield"(%[[CASE1]]#0, %[[CASE1]]#1) : (i32, i1) -> ()
// MLIR: }) : (i1) -> (i32, i1)
// MLIR: return
