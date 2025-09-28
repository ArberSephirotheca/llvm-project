// MLIR-LABEL: "builtin.module"()
// MLIR: "func.func"() <{function_type = (i32) -> (), sym_name = "main"}> ({
// MLIR:   %[[SWITCH:.*]]:4 = "simt_step.switch"(%[[SELECT:.*]], %[[SUM0:.*]], %[[HAS0:.*]], %[[EXEC0:.*]], %[[DONE0:.*]]) ({
// MLIR-COUNT: 4 "simt_step.if"
// MLIR:     ^bb0(%[[CASE0_VAL:.*]]: i32, %[[CASE0_HAS:.*]]: i1, %[[CASE0_EXEC:.*]]: i1, %[[CASE0_DONE:.*]]: i1):
// MLIR:       %[[CASE0_IF:.*]]:4 = "simt_step.if"(%{{.*}}) ({
// MLIR:         %[[ONE:.*]] = "arith.constant"() <{value = 1 : i32}>
// MLIR:         %[[ADD1:.*]] = "arith.addi"(%[[CASE0_VAL]], %[[ONE]])
// MLIR:         "simt_step.yield"(%[[ADD1]], {{.*}}, {{.*}}, {{.*}})
// MLIR:       }, {
// MLIR:         "simt_step.yield"(%[[CASE0_VAL]], %[[CASE0_HAS]], %[[CASE0_EXEC]], %[[CASE0_DONE]])
// MLIR:       }) : (i1) -> (i32, i1, i1, i1)
// MLIR:       "simt_step.yield"(%[[CASE0_IF]]#0, %[[CASE0_IF]]#1, %[[CASE0_IF]]#2, %[[CASE0_IF]]#3)
// MLIR:     ^bb1(%[[CASE1_VAL:.*]]: i32, %[[CASE1_HAS:.*]]: i1, %[[CASE1_EXEC:.*]]: i1, %[[CASE1_DONE:.*]]: i1):
// MLIR-NEXT:       %[[CASE1_FALSE:.*]] = "arith.constant"() <{value = false}>
// MLIR-NEXT:       %[[CASE1_TRUE:.*]] = "arith.constant"() <{value = true}>
// MLIR-NEXT:       %[[CASE1_NOT_DONE:.*]] = "arith.cmpi"(%[[CASE1_DONE]], %[[CASE1_FALSE]])
// MLIR-NEXT:       %[[CASE1_NOT_MATCHED:.*]] = "arith.cmpi"(%[[CASE1_HAS]], %[[CASE1_FALSE]])
// MLIR-NEXT:       %[[CASE1_SEL1:.*]] = "arith.constant"() <{value = 1 : i32}>
// MLIR-NEXT:       %[[CASE1_EQ:.*]] = "arith.cmpi"(%[[SELECT]], %[[CASE1_SEL1]])
// MLIR-NEXT:       %[[CASE1_AND0:.*]] = "arith.andi"(%[[CASE1_NOT_MATCHED]], %[[CASE1_NOT_DONE]])
// MLIR-NEXT:       %[[CASE1_MATCH_BASE:.*]] = "arith.andi"(%[[CASE1_AND0]], %[[CASE1_EQ]])
// MLIR-NEXT:       %[[CASE1_ENTER_OR:.*]] = "arith.ori"(%[[CASE1_EXEC]], %[[CASE1_MATCH_BASE]])
// MLIR-NEXT:       %[[CASE1_ENTER:.*]] = "arith.andi"(%[[CASE1_ENTER_OR]], %[[CASE1_NOT_DONE]])
// MLIR-NEXT:       %[[CASE1_IF:.*]]:4 = "simt_step.if"(%[[CASE1_ENTER]]) ({
// MLIR-NEXT:         %[[CASE1_TRUE1:.*]] = "arith.constant"() <{value = true}>
// MLIR-NEXT:         %[[CASE1_FALSE:.*]] = "arith.constant"() <{value = false}>
// MLIR-NEXT:         %[[CASE1_TRUE2:.*]] = "arith.constant"() <{value = true}>
// MLIR-NEXT:         %[[CASE1_TWO:.*]] = "arith.constant"() <{value = 2 : i32}>
// MLIR-NEXT:         %[[ADD2:.*]] = "arith.addi"(%[[CASE1_VAL]], %[[CASE1_TWO]])
// MLIR-NEXT:         "simt_step.yield"(%[[ADD2]], %[[CASE1_TRUE1]], %[[CASE1_FALSE]], %[[CASE1_TRUE2]])
// MLIR:       }, {
// MLIR:         "simt_step.yield"(%[[CASE1_VAL]], %[[CASE1_HAS]], %[[CASE1_EXEC]], %[[CASE1_DONE]])
// MLIR:       }) : (i1) -> (i32, i1, i1, i1)
// MLIR:       "simt_step.yield"(%[[CASE1_IF]]#0, %[[CASE1_IF]]#1, %[[CASE1_IF]]#2, %[[CASE1_IF]]#3)
// MLIR:     ^bb2(%[[CASE2_VAL:.*]]: i32, %[[CASE2_HAS:.*]]: i1, %[[CASE2_EXEC:.*]]: i1, %[[CASE2_DONE:.*]]: i1):
// MLIR-NEXT:       %[[CASE2_FALSE:.*]] = "arith.constant"() <{value = false}>
// MLIR-NEXT:       %[[CASE2_TRUE:.*]] = "arith.constant"() <{value = true}>
// MLIR-NEXT:       %[[CASE2_NOT_DONE:.*]] = "arith.cmpi"(%[[CASE2_DONE]], %[[CASE2_FALSE]])
// MLIR-NEXT:       %[[CASE2_NOT_MATCHED:.*]] = "arith.cmpi"(%[[CASE2_HAS]], %[[CASE2_FALSE]])
// MLIR-NEXT:       %[[CASE2_SEL2:.*]] = "arith.constant"() <{value = 2 : i32}>
// MLIR-NEXT:       %[[CASE2_EQ:.*]] = "arith.cmpi"(%[[SELECT]], %[[CASE2_SEL2]])
// MLIR-NEXT:       %[[CASE2_AND0:.*]] = "arith.andi"(%[[CASE2_NOT_MATCHED]], %[[CASE2_NOT_DONE]])
// MLIR-NEXT:       %[[CASE2_MATCH_BASE:.*]] = "arith.andi"(%[[CASE2_AND0]], %[[CASE2_EQ]])
// MLIR-NEXT:       %[[CASE2_ENTER_OR:.*]] = "arith.ori"(%[[CASE2_EXEC]], %[[CASE2_MATCH_BASE]])
// MLIR-NEXT:       %[[CASE2_ENTER:.*]] = "arith.andi"(%[[CASE2_ENTER_OR]], %[[CASE2_NOT_DONE]])
// MLIR-NEXT:       %[[CASE2_IF:.*]]:4 = "simt_step.if"(%[[CASE2_ENTER]]) ({
// MLIR-NEXT:         %[[CASE2_TRUE_IF:.*]] = "arith.constant"() <{value = true}>
// MLIR-NEXT:         %[[CASE2_FALSE_IF:.*]] = "arith.constant"() <{value = false}>
// MLIR-NEXT:         %[[CASE2_TRUE_IF2:.*]] = "arith.constant"() <{value = true}>
// MLIR-NEXT:         %[[CASE2_FOUR:.*]] = "arith.constant"() <{value = 4 : i32}>
// MLIR-NEXT:         %[[ADD4:.*]] = "arith.addi"(%[[CASE2_VAL]], %[[CASE2_FOUR]])
// MLIR-NEXT:         %[[CASE2_NEWEXEC:.*]] = "arith.ori"(%[[CASE2_HAS]], %[[CASE2_MATCH_BASE]])
// MLIR-NEXT:         "simt_step.yield"(%[[ADD4]], %[[CASE2_NEWEXEC]], %[[CASE2_TRUE]], %[[CASE2_DONE]])
// MLIR:       }, {
// MLIR:         "simt_step.yield"(%[[CASE2_VAL]], %[[CASE2_HAS]], %[[CASE2_EXEC]], %[[CASE2_DONE]])
// MLIR:       }) : (i1) -> (i32, i1, i1, i1)
// MLIR:       "simt_step.yield"(%[[CASE2_IF]]#0, %[[CASE2_IF]]#1, %[[CASE2_IF]]#2, %[[CASE2_IF]]#3)
// MLIR:     ^bb3(%[[DEF_VAL:.*]]: i32, %[[DEF_HAS:.*]]: i1, %[[DEF_EXEC:.*]]: i1, %[[DEF_DONE:.*]]: i1):
// MLIR-NEXT:       %[[DEF_FALSE:.*]] = "arith.constant"() <{value = false}>
// MLIR-NEXT:       %[[DEF_TRUE:.*]] = "arith.constant"() <{value = true}>
// MLIR-NEXT:       %[[DEF_NOT_DONE:.*]] = "arith.cmpi"(%[[DEF_DONE]], %[[DEF_FALSE]])
// MLIR-NEXT:       %[[DEF_NOT_MATCHED:.*]] = "arith.cmpi"(%[[DEF_HAS]], %[[DEF_FALSE]])
// MLIR-NEXT:       %[[DEF_AND0:.*]] = "arith.andi"(%[[DEF_NOT_MATCHED]], %[[DEF_NOT_DONE]])
// MLIR-NEXT:       %[[DEF_ENTER_OR:.*]] = "arith.ori"(%[[DEF_EXEC]], %[[DEF_AND0]])
// MLIR-NEXT:       %[[DEF_ENTER:.*]] = "arith.andi"(%[[DEF_ENTER_OR]], %[[DEF_NOT_DONE]])
// MLIR-NEXT:       %[[DEF_IF:.*]]:4 = "simt_step.if"(%[[DEF_ENTER]]) ({
// MLIR-NEXT:         %[[DEF_TRUE1:.*]] = "arith.constant"() <{value = true}>
// MLIR-NEXT:         %[[DEF_FALSE2:.*]] = "arith.constant"() <{value = false}>
// MLIR-NEXT:         %[[DEF_TRUE2:.*]] = "arith.constant"() <{value = true}>
// MLIR-NEXT:         %[[DEF_EIGHT:.*]] = "arith.constant"() <{value = 8 : i32}>
// MLIR-NEXT:         %[[ADD8:.*]] = "arith.addi"(%[[DEF_VAL]], %[[DEF_EIGHT]])
// MLIR-NEXT:         "simt_step.yield"(%[[ADD8]], %[[DEF_TRUE1]], %[[DEF_FALSE2]], %[[DEF_TRUE2]])
// MLIR:       }, {
// MLIR:         "simt_step.yield"(%[[DEF_VAL]], %[[DEF_HAS]], %[[DEF_EXEC]], %[[DEF_DONE]])
// MLIR:       }) : (i1) -> (i32, i1, i1, i1)
// MLIR:   }) {case_values = array<i64: 0, 1, 2, 0>} : (i32, i32, i1, i1, i1) -> (i32, i1, i1, i1)
// MLIR:   "func.return"()
