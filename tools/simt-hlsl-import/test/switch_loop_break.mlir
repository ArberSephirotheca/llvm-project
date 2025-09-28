// MLIR-LABEL: "builtin.module"()
// MLIR: "func.func"() <{function_type = (i32) -> (), sym_name = "main"}> ({
// MLIR:   %[[SWITCH:.*]]:4 = "simt_step.switch"(%[[SELECT:.*]], %[[SUM0:.*]], %[[HAS0:.*]], %[[EXEC0:.*]], %[[DONE0:.*]]) ({
// MLIR-COUNT: 2 "simt_step.if"
// MLIR:     ^bb0(%[[CASE0_VAL:.*]]: i32, %[[CASE0_HAS:.*]]: i1, %[[CASE0_EXEC:.*]]: i1, %[[CASE0_DONE:.*]]: i1):
// MLIR:       %[[CASE0_IF:.*]]:4 = "simt_step.if"(%{{.*}}) ({
// MLIR:         %[[LOOP:.*]] = "simt_step.loop"(%[[CASE0_VAL]]) ({
// MLIR:         ^bb0(%[[LOOP_ARG:.*]]: i32):
// MLIR:           "simt_step.condition"
// MLIR:         }, {
// MLIR:         ^bb0(%[[LOOP_BODY:.*]]: i32):
// MLIR:           %[[INCREMENT:.*]] = "arith.addi"(%[[LOOP_BODY]], %{{.*}})
// MLIR:           "simt_step.break"(%[[INCREMENT]])
// MLIR:         }) : (i32) -> i32
// MLIR:         "simt_step.yield"(%[[LOOP]], {{.*}}, {{.*}}, {{.*}})
// MLIR:       }, {
// MLIR:         "simt_step.yield"(%[[CASE0_VAL]], %[[CASE0_HAS]], %[[CASE0_EXEC]], %[[CASE0_DONE]])
// MLIR:       }) : (i1) -> (i32, i1, i1, i1)
// MLIR:       "simt_step.yield"(%[[CASE0_IF]]#0, %[[CASE0_IF]]#1, %[[CASE0_IF]]#2, %[[CASE0_IF]]#3)
// MLIR:     ^bb1(%[[DEF_VAL:.*]]: i32, %[[DEF_HAS:.*]]: i1, %[[DEF_EXEC:.*]]: i1, %[[DEF_DONE:.*]]: i1):
// MLIR:       %[[DEF_IF:.*]]:4 = "simt_step.if"(%{{.*}}) ({
// MLIR:         %[[FIVE:.*]] = "arith.constant"() <{value = 5 : i32}>
// MLIR:         %[[ADD5:.*]] = "arith.addi"(%[[DEF_VAL]], %[[FIVE]])
// MLIR:         "simt_step.yield"(%[[ADD5]], {{.*}}, {{.*}}, {{.*}})
// MLIR:       }, {
// MLIR:         "simt_step.yield"(%[[DEF_VAL]], %[[DEF_HAS]], %[[DEF_EXEC]], %[[DEF_DONE]])
// MLIR:       }) : (i1) -> (i32, i1, i1, i1)
// MLIR:   }) {case_values = array<i64: 0, 0>} : (i32, i32, i1, i1, i1) -> (i32, i1, i1, i1)
// MLIR:   "func.return"()
// MLIR: })
