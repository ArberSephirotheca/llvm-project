// MLIR-LABEL: "builtin.module"()
// MLIR: "func.func"() <{function_type = (i32) -> (), sym_name = "main"}> ({
// MLIR:   %[[SWITCH:.*]]:4 = "simt_step.switch"(%[[SELECT:.*]], %[[ZERO:.*]], %[[HAS0:.*]], %[[EXEC0:.*]], %[[DONE0:.*]]) ({
// MLIR-COUNT: 4 "simt_step.if"
// MLIR:     ^bb0(%[[CASE0_VAL:.*]]: i32, %[[CASE0_HAS:.*]]: i1, %[[CASE0_EXEC:.*]]: i1, %[[CASE0_DONE:.*]]: i1):
// MLIR:       %[[CASE0_IF:.*]]:4 = "simt_step.if"(%{{.*}}) ({
// MLIR:         "simt_step.yield"(%[[CASE0_VAL]], {{.*}}, {{.*}}, %[[CASE0_DONE]]) : (i32, i1, i1, i1) -> ()
// MLIR:       }, {
// MLIR:         "simt_step.yield"(%[[CASE0_VAL]], %[[CASE0_HAS]], %[[CASE0_EXEC]], %[[CASE0_DONE]]) : (i32, i1, i1, i1) -> ()
// MLIR:       }) : (i1) -> (i32, i1, i1, i1)
// MLIR:       "simt_step.yield"(%[[CASE0_IF]]#0, %[[CASE0_IF]]#1, %[[CASE0_IF]]#2, %[[CASE0_IF]]#3) : (i32, i1, i1, i1) -> ()
// MLIR:     ^bb1(%[[CASE1_VAL:.*]]: i32, %[[CASE1_HAS:.*]]: i1, %[[CASE1_EXEC:.*]]: i1, %[[CASE1_DONE:.*]]: i1):
// MLIR:       %[[CASE1_NOT_MATCHED:.*]] = "arith.cmpi"(%[[CASE1_HAS]], %[[FALSE:.*]])
// MLIR:       %[[CASE1_EQ:.*]] = "arith.cmpi"(%[[SELECT]], %[[ONE:.*]])
// MLIR:       %[[CASE1_IF:.*]]:4 = "simt_step.if"(%{{.*}}) ({
// MLIR:         %[[CASE1_TRUE1:.*]] = "arith.constant"() <{value = true}>
// MLIR:         %[[CASE1_FALSE:.*]] = "arith.constant"() <{value = false}>
// MLIR:         %[[CASE1_TRUE2:.*]] = "arith.constant"() <{value = true}>
// MLIR:         %[[CONST2:.*]] = "arith.constant"() <{value = 2 : i32}>
// MLIR:         "simt_step.yield"(%[[CONST2]], %[[CASE1_TRUE1]], %[[CASE1_FALSE]], %[[CASE1_TRUE2]]) : (i32, i1, i1, i1) -> ()
// MLIR:       }, {
// MLIR:         "simt_step.yield"(%[[CASE1_VAL]], %[[CASE1_HAS]], %[[CASE1_EXEC]], %[[CASE1_DONE]]) : (i32, i1, i1, i1) -> ()
// MLIR:       }) : (i1) -> (i32, i1, i1, i1)
// MLIR:       "simt_step.yield"(%[[CASE1_IF]]#0, %[[CASE1_IF]]#1, %[[CASE1_IF]]#2, %[[CASE1_IF]]#3) : (i32, i1, i1, i1) -> ()
// MLIR:     ^bb2(%[[CASE2_VAL:.*]]: i32, %[[CASE2_HAS:.*]]: i1, %[[CASE2_EXEC:.*]]: i1, %[[CASE2_DONE:.*]]: i1):
// MLIR:       %[[CASE2_FALSE:.*]] = "arith.constant"() <{value = false}>
// MLIR:       %[[CASE2_TRUE:.*]] = "arith.constant"() <{value = true}>
// MLIR:       %[[CASE2_NOT_DONE:.*]] = "arith.cmpi"(%[[CASE2_DONE]], %[[CASE2_FALSE]])
// MLIR:       %[[CASE2_NOT_MATCHED:.*]] = "arith.cmpi"(%[[CASE2_HAS]], %[[CASE2_FALSE]])
// MLIR:       %[[CASE2_EQ:.*]] = "arith.cmpi"(%[[SELECT]], %[[TWO:.*]])
// MLIR:       %[[CASE2_CAN:.*]] = "arith.andi"(%[[CASE2_NOT_MATCHED]], %[[CASE2_NOT_DONE]])
// MLIR:       %[[CASE2_MATCH_BASE:.*]] = "arith.andi"(%[[CASE2_CAN]], %[[CASE2_EQ]])
// MLIR:       %[[CASE2_EXEC_OR:.*]] = "arith.ori"(%[[CASE2_EXEC]], %[[CASE2_MATCH_BASE]])
// MLIR:       %[[CASE2_ENTER:.*]] = "arith.andi"(%[[CASE2_EXEC_OR]], %[[CASE2_NOT_DONE]])
// MLIR:       %[[CASE2_IF:.*]]:4 = "simt_step.if"(%[[CASE2_ENTER]]) ({
// MLIR:         %[[CONST5:.*]] = "arith.constant"() <{value = 5 : i32}>
// MLIR:         %[[CASE2_OR:.*]] = "arith.ori"(%[[CASE2_HAS]], %[[CASE2_MATCH_BASE]])
// MLIR:         "simt_step.yield"(%[[CONST5]], %[[CASE2_OR]], %[[CASE2_TRUE]], %[[CASE2_DONE]]) : (i32, i1, i1, i1) -> ()
// MLIR:       }, {
// MLIR:         "simt_step.yield"(%[[CASE2_VAL]], %[[CASE2_HAS]], %[[CASE2_EXEC]], %[[CASE2_DONE]]) : (i32, i1, i1, i1) -> ()
// MLIR:       }) : (i1) -> (i32, i1, i1, i1)
// MLIR:       "simt_step.yield"(%[[CASE2_IF]]#0, %[[CASE2_IF]]#1, %[[CASE2_IF]]#2, %[[CASE2_IF]]#3) : (i32, i1, i1, i1) -> ()
// MLIR:     ^bb3(%[[DEF_VAL:.*]]: i32, %[[DEF_HAS:.*]]: i1, %[[DEF_EXEC:.*]]: i1, %[[DEF_DONE:.*]]: i1):
// MLIR:       %[[DEF_FALSE:.*]] = "arith.constant"() <{value = false}>
// MLIR:       %[[DEF_TRUE:.*]] = "arith.constant"() <{value = true}>
// MLIR:       %[[DEF_NOT_DONE:.*]] = "arith.cmpi"(%[[DEF_DONE]], %[[DEF_FALSE]])
// MLIR:       %[[DEF_NOT_MATCHED:.*]] = "arith.cmpi"(%[[DEF_HAS]], %[[DEF_FALSE]])
// MLIR:       %[[DEF_BASE:.*]] = "arith.andi"(%[[DEF_NOT_MATCHED]], %[[DEF_NOT_DONE]])
// MLIR:       %[[DEF_ENTER:.*]] = "arith.andi"(%{{.*}}, %[[DEF_NOT_DONE]])
// MLIR:       %[[DEF_IF:.*]]:4 = "simt_step.if"(%[[DEF_ENTER]]) ({
// MLIR:         %[[DEF_TRUE2:.*]] = "arith.constant"() <{value = true}>
// MLIR:         %[[DEF_FALSE2:.*]] = "arith.constant"() <{value = false}>
// MLIR:         %[[DEF_TRUE3:.*]] = "arith.constant"() <{value = true}>
// MLIR:         %[[CONST7:.*]] = "arith.constant"() <{value = 7 : i32}>
// MLIR:         "simt_step.yield"(%[[CONST7]], %[[DEF_TRUE2]], %[[DEF_FALSE2]], %[[DEF_TRUE3]]) : (i32, i1, i1, i1) -> ()
// MLIR:       }, {
// MLIR:         "simt_step.yield"(%[[DEF_VAL]], %[[DEF_HAS]], %[[DEF_EXEC]], %[[DEF_DONE]]) : (i32, i1, i1, i1) -> ()
// MLIR:       }) : (i1) -> (i32, i1, i1, i1)
// MLIR:   }) {case_values = array<i64: 0, 1, 2, 0>} : (i32, i32, i1, i1, i1) -> (i32, i1, i1, i1)
// MLIR:   "func.return"() : () -> ()
