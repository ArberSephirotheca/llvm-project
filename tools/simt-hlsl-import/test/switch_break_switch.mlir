// MLIR-LABEL: "builtin.module"()
// MLIR: "func.func"() <{function_type = (i32) -> (), sym_name = "main"}> ({
// MLIR:   %{{.*}} = "simt_step.dispatch_thread_id"() : () -> i32
// MLIR:   %[[SWITCH:.*]]:4 = "simt_step.switch"(%[[SELECT:.*]], %[[VAL0:.*]], %[[HAS0:.*]], %[[EXEC0:.*]], %[[DONE0:.*]]) ({
// MLIR-COUNT: 3 "simt_step.if"
// MLIR:     ^bb0(%[[CASE0_VAL:.*]]: i32, %[[CASE0_HAS:.*]]: i1, %[[CASE0_EXEC:.*]]: i1, %[[CASE0_DONE:.*]]: i1):
// MLIR:       %[[CASE0_IF:.*]]:4 = "simt_step.if"(%{{.*}}) ({
// MLIR:         %[[CASE0_TRUE1:.*]] = "arith.constant"() <{value = true}>
// MLIR:         %[[CASE0_FALSE:.*]] = "arith.constant"() <{value = false}>
// MLIR:         %[[CASE0_TRUE2:.*]] = "arith.constant"() <{value = true}>
// MLIR:         %[[CASE0_ONE:.*]] = "arith.constant"() <{value = 1 : i32}>
// MLIR:         %[[CASE0_ADD:.*]] = "arith.addi"(%[[CASE0_VAL]], %[[CASE0_ONE]])
// MLIR:         "simt_step.yield"(%[[CASE0_ADD]], %[[CASE0_TRUE1]], %[[CASE0_FALSE]], %[[CASE0_TRUE2]])
// MLIR:       }, {
// MLIR:         "simt_step.yield"(%[[CASE0_VAL]], %[[CASE0_HAS]], %[[CASE0_EXEC]], %[[CASE0_DONE]])
// MLIR:       }) : (i1) -> (i32, i1, i1, i1)
// MLIR:       "simt_step.yield"(%[[CASE0_IF]]#0, %[[CASE0_IF]]#1, %[[CASE0_IF]]#2, %[[CASE0_IF]]#3)
// MLIR:     ^bb1(%[[CASE1_VAL:.*]]: i32, %[[CASE1_HAS:.*]]: i1, %[[CASE1_EXEC:.*]]: i1, %[[CASE1_DONE:.*]]: i1):
// MLIR:       %[[CASE1_IF:.*]]:4 = "simt_step.if"(%{{.*}}) ({
// MLIR:         %[[CASE1_TRUE1:.*]] = "arith.constant"() <{value = true}>
// MLIR:         %[[CASE1_FALSE:.*]] = "arith.constant"() <{value = false}>
// MLIR:         %[[CASE1_TRUE2:.*]] = "arith.constant"() <{value = true}>
// MLIR:         %[[CASE1_TWO:.*]] = "arith.constant"() <{value = 2 : i32}>
// MLIR:         %[[CASE1_ADD:.*]] = "arith.addi"(%[[CASE1_VAL]], %[[CASE1_TWO]])
// MLIR:         "simt_step.yield"(%[[CASE1_ADD]], %[[CASE1_TRUE1]], %[[CASE1_FALSE]], %[[CASE1_TRUE2]])
// MLIR:       }, {
// MLIR:         "simt_step.yield"(%[[CASE1_VAL]], %[[CASE1_HAS]], %[[CASE1_EXEC]], %[[CASE1_DONE]])
// MLIR:       }) : (i1) -> (i32, i1, i1, i1)
// MLIR:       "simt_step.yield"(%[[CASE1_IF]]#0, %[[CASE1_IF]]#1, %[[CASE1_IF]]#2, %[[CASE1_IF]]#3)
// MLIR:     ^bb2(%[[DEF_VAL:.*]]: i32, %[[DEF_HAS:.*]]: i1, %[[DEF_EXEC:.*]]: i1, %[[DEF_DONE:.*]]: i1):
// MLIR:       %[[DEF_IF:.*]]:4 = "simt_step.if"(%{{.*}}) ({
// MLIR:         %[[DEF_TRUE1:.*]] = "arith.constant"() <{value = true}>
// MLIR:         %[[DEF_FALSE:.*]] = "arith.constant"() <{value = false}>
// MLIR:         %[[DEF_TRUE2:.*]] = "arith.constant"() <{value = true}>
// MLIR:         %[[DEF_FOUR:.*]] = "arith.constant"() <{value = 4 : i32}>
// MLIR:         %[[DEF_ADD:.*]] = "arith.addi"(%[[DEF_VAL]], %[[DEF_FOUR]])
// MLIR:         "simt_step.yield"(%[[DEF_ADD]], %[[DEF_TRUE1]], %[[DEF_FALSE]], %[[DEF_TRUE2]])
// MLIR:       }, {
// MLIR:         "simt_step.yield"(%[[DEF_VAL]], %[[DEF_HAS]], %[[DEF_EXEC]], %[[DEF_DONE]])
// MLIR:       }) : (i1) -> (i32, i1, i1, i1)
// MLIR:   }) {case_values = array<i64: 0, 1, 0>} : (i32, i32, i1, i1, i1) -> (i32, i1, i1, i1)
// MLIR:   "func.return"()
// MLIR: })
