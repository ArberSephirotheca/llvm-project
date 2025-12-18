// MLIR-LABEL: module {
// MLIR:   func.func @main(%arg0: i32) attributes {simt.num_threads = array<i64: 1, 1, 1>} {
// MLIR:     %[[TID:.*]] = "simt_step.dispatch_thread_id"() : () -> i32
// MLIR:     %[[C10:.*]] = arith.constant 10 : i32
// MLIR:     %[[SWITCH:.*]] = "simt_step.switch"(%[[TID]], %[[C10]]) ({
// MLIR:       ^bb0(%[[CASE0_VAL:.*]]: i32):
// MLIR:         %[[C1:.*]] = arith.constant 1 : i32
// MLIR:         %[[ADD1:.*]] = arith.addi %[[CASE0_VAL]], %[[C1]] : i32
// MLIR:         "simt_step.yield"(%[[ADD1]]) {fallthrough = true} : (i32) -> ()
// MLIR:       ^bb1(%[[CASE1_VAL:.*]]: i32):
// MLIR:         %[[C2:.*]] = arith.constant 2 : i32
// MLIR:         %[[ADD2:.*]] = arith.addi %[[CASE1_VAL]], %[[C2]] : i32
// MLIR:         "simt_step.yield"(%[[ADD2]]) {fallthrough = false} : (i32) -> ()
// MLIR:       ^bb2(%[[CASE2_VAL:.*]]: i32):
// MLIR:         %[[C4:.*]] = arith.constant 4 : i32
// MLIR:         %[[ADD4:.*]] = arith.addi %[[CASE2_VAL]], %[[C4]] : i32
// MLIR:         "simt_step.yield"(%[[ADD4]]) {fallthrough = true} : (i32) -> ()
// MLIR:       ^bb3(%[[DEF_VAL:.*]]: i32):
// MLIR:         %[[C8:.*]] = arith.constant 8 : i32
// MLIR:         %[[ADD8:.*]] = arith.addi %[[DEF_VAL]], %[[C8]] : i32
// MLIR:         "simt_step.yield"(%[[ADD8]]) {fallthrough = false} : (i32) -> ()
// MLIR:     }) {case_values = array<i64: 0, 1, 2>, default_index = 3 : i64} : (i32, i32) -> i32
// MLIR:     return
// MLIR:   }
// MLIR: }
