// MLIR-LABEL: module {
// MLIR:   func.func @main(%arg0: i32) attributes {simt.num_threads = array<i64: 1, 1, 1>} {
// MLIR:     %[[DISPATCH:.*]] = "simt_step.dispatch_thread_id"() : () -> i32
// MLIR:     %[[C0:.*]] = arith.constant 0 : i32
// MLIR:     %[[SWITCH:.*]] = "simt_step.switch"(%[[DISPATCH]], %[[C0]]) ({
// MLIR:       ^bb0(%[[CASE0_VAL:.*]]: i32):
// MLIR:         "simt_step.yield"(%[[CASE0_VAL]]) {fallthrough = true} : (i32) -> ()
// MLIR:       ^bb1(%[[CASE1_VAL:.*]]: i32):
// MLIR:         %[[C2:.*]] = arith.constant 2 : i32
// MLIR:         "simt_step.yield"(%[[C2]]) {fallthrough = false} : (i32) -> ()
// MLIR:       ^bb2(%[[CASE2_VAL:.*]]: i32):
// MLIR:         %[[C5:.*]] = arith.constant 5 : i32
// MLIR:         "simt_step.yield"(%[[C5]]) {fallthrough = true} : (i32) -> ()
// MLIR:       ^bb3(%[[DEF_VAL:.*]]: i32):
// MLIR:         %[[C7:.*]] = arith.constant 7 : i32
// MLIR:         "simt_step.yield"(%[[C7]]) {fallthrough = false} : (i32) -> ()
// MLIR:     }) {case_values = array<i64: 0, 1, 2>, default_index = 3 : i64} : (i32, i32) -> i32
// MLIR:     return
// MLIR:   }
// MLIR: }
