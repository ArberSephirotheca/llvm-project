// MLIR-LABEL: module {
// MLIR:   func.func @main(%arg0: i32) attributes {simt.num_threads = array<i64: 1, 1, 1>} {
// MLIR:     %[[DISPATCH:.*]] = "simt_step.dispatch_thread_id"() : () -> i32
// MLIR:     %[[C0:.*]] = arith.constant 0 : i32
// MLIR:     %[[SWITCH:.*]] = "simt_step.switch"(%[[DISPATCH]], %[[C0]]) ({
// MLIR:       ^bb0(%[[CASE0_VAL:.*]]: i32):
// MLIR:         %[[C0B:.*]] = arith.constant 0 : i32
// MLIR:         %[[LOOP:.*]] = "simt_step.loop"(%[[CASE0_VAL]]) ({
// MLIR:         ^bb0(%[[ACC:.*]]: i32):
// MLIR:           %[[C3:.*]] = arith.constant 3 : i32
// MLIR:           %[[CMP:.*]] = arith.cmpi slt, %[[C0B]], %[[C3]] : i32
// MLIR:           "simt_step.condition"(%[[CMP]], %[[ACC]]) : (i1, i32) -> ()
// MLIR:         }, {
// MLIR:         ^bb0(%[[ACC_B:.*]]: i32):
// MLIR:           %[[ADD:.*]] = arith.addi %[[ACC_B]], %[[C0B]] : i32
// MLIR:           "simt_step.break"(%[[ADD]]) : (i32) -> ()
// MLIR:         }) : (i32) -> i32
// MLIR:         "simt_step.yield"(%[[LOOP]]) {fallthrough = false} : (i32) -> ()
// MLIR:       ^bb1(%[[DEF_VAL:.*]]: i32):
// MLIR:         %[[C5:.*]] = arith.constant 5 : i32
// MLIR:         %[[ADD5:.*]] = arith.addi %[[DEF_VAL]], %[[C5]] : i32
// MLIR:         "simt_step.yield"(%[[ADD5]]) {fallthrough = false} : (i32) -> ()
// MLIR:     }) {case_values = array<i64: 0>, default_index = 1 : i64} : (i32, i32) -> i32
// MLIR:     return
// MLIR:   }
// MLIR: }
