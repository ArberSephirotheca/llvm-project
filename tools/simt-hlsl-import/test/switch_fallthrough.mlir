// MLIR-LABEL: "builtin.module"()
// MLIR: "func.func"() <{function_type = (i32) -> (), sym_name = "main"}> ({
// MLIR: %[[SWITCH:.*]]:4 = "simt_step.switch"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) ({
// MLIR:   %[[CASE0:.*]]:4 = "simt_step.if"(%{{.*}}) ({
// MLIR:     %[[ONE:.*]] = "arith.constant"() <{value = 1 : i32}>
// MLIR:     %[[ADD1:.*]] = "arith.addi"(%{{.*}}, %[[ONE]])
// MLIR:     "simt_step.yield"(%[[ADD1]], {{.*}}) : (i32, i1, i1, i1) -> ()
// MLIR:   }, {
// MLIR:     "simt_step.yield"({{.*}}) : (i32, i1, i1, i1) -> ()
// MLIR:   }) : (i1) -> (i32, i1, i1, i1)
// MLIR:   %[[CASE1:.*]]:4 = "simt_step.if"(%{{.*}}) ({
// MLIR:     %[[TWO:.*]] = "arith.constant"() <{value = 2 : i32}>
// MLIR:     %[[ADD2:.*]] = "arith.addi"(%{{.*}}, %[[TWO]])
// MLIR:     "simt_step.yield"(%[[ADD2]], {{.*}}) : (i32, i1, i1, i1) -> ()
// MLIR:   }, {
// MLIR:     "simt_step.yield"({{.*}}) : (i32, i1, i1, i1) -> ()
// MLIR:   }) : (i1) -> (i32, i1, i1, i1)
// MLIR:   %[[CASE2:.*]]:4 = "simt_step.if"(%{{.*}}) ({
// MLIR:     %[[FOUR:.*]] = "arith.constant"() <{value = 4 : i32}>
// MLIR:     %[[ADD4:.*]] = "arith.addi"(%{{.*}}, %[[FOUR]])
// MLIR:     "simt_step.yield"(%[[ADD4]], {{.*}}) : (i32, i1, i1, i1) -> ()
// MLIR:   }, {
// MLIR:     "simt_step.yield"({{.*}}) : (i32, i1, i1, i1) -> ()
// MLIR:   }) : (i1) -> (i32, i1, i1, i1)
// MLIR:   %[[DEFAULT:.*]]:4 = "simt_step.if"(%{{.*}}) ({
// MLIR:     %[[EIGHT:.*]] = "arith.constant"() <{value = 8 : i32}>
// MLIR:     %[[ADD8:.*]] = "arith.addi"(%{{.*}}, %[[EIGHT]])
// MLIR:     "simt_step.yield"(%[[ADD8]], {{.*}}) : (i32, i1, i1, i1) -> ()
// MLIR:   }, {
// MLIR:     "simt_step.yield"({{.*}}) : (i32, i1, i1, i1) -> ()
// MLIR:   }) : (i1) -> (i32, i1, i1, i1)
// MLIR: }) {case_values = array<i64: 0, 1, 2, 0>} : (i32, i32, i1, i1, i1) -> (i32, i1, i1, i1)
// MLIR: return
