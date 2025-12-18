// MLIR-LABEL: module {
// MLIR:   func.func @main(%arg0: i32) attributes {simt.num_threads = array<i64: 1, 1, 1>} {
// MLIR:     %[[DISPATCH:.*]] = "simt_step.dispatch_thread_id"() : () -> i32
// MLIR:     %[[C0:.*]] = arith.constant 0 : i32
// MLIR:     %[[C0B:.*]] = arith.constant 0 : i32
// MLIR:     %[[LOOP:.*]]:2 = "simt_step.loop"(%[[C0]], %[[C0B]]) ({
// MLIR:     ^bb0(%[[ACC:.*]]: i32, %[[IDX:.*]]: i32):
// MLIR:       %[[C4:.*]] = arith.constant 4 : i32
// MLIR:       %[[CMP:.*]] = arith.cmpi slt, %[[IDX]], %[[C4]] : i32
// MLIR:       "simt_step.condition"(%[[CMP]], %[[ACC]], %[[IDX]]) : (i1, i32, i32) -> ()
// MLIR:     }, {
// MLIR:     ^bb0(%[[ACC_B:.*]]: i32, %[[IDX_B:.*]]: i32):
// MLIR:       %[[SWITCH:.*]] = "simt_step.switch"(%[[IDX_B]], %[[ACC_B]]) ({
// MLIR:       ^bb0(%[[CASE0_VAL:.*]]: i32):
// MLIR:         %[[C1:.*]] = arith.constant 1 : i32
// MLIR:         %[[ADD1:.*]] = arith.addi %[[CASE0_VAL]], %[[C1]] : i32
// MLIR:         "simt_step.yield"(%[[ADD1]]) {fallthrough = false} : (i32) -> ()
// MLIR:       ^bb1(%[[CASE1_VAL:.*]]: i32):
// MLIR:         %[[C2:.*]] = arith.constant 2 : i32
// MLIR:         %[[ADD2:.*]] = arith.addi %[[CASE1_VAL]], %[[C2]] : i32
// MLIR:         "simt_step.yield"(%[[ADD2]]) {fallthrough = false} : (i32) -> ()
// MLIR:       ^bb2(%[[DEF_VAL:.*]]: i32):
// MLIR:         %[[C4B:.*]] = arith.constant 4 : i32
// MLIR:         %[[ADD4:.*]] = arith.addi %[[DEF_VAL]], %[[C4B]] : i32
// MLIR:         "simt_step.yield"(%[[ADD4]]) {fallthrough = false} : (i32) -> ()
// MLIR:       }) {case_values = array<i64: 0, 1>, default_index = 2 : i64} : (i32, i32) -> i32
// MLIR:       %[[C1B:.*]] = arith.constant 1 : i32
// MLIR:       %[[IDX_NEXT:.*]] = arith.addi %[[IDX_B]], %[[C1B]] : i32
// MLIR:       "simt_step.yield"(%[[SWITCH]], %[[IDX_NEXT]]) : (i32, i32) -> ()
// MLIR:     }) : (i32, i32) -> (i32, i32)
// MLIR:     return
// MLIR:   }
// MLIR: }
