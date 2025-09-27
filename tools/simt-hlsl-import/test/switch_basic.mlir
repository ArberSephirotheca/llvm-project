// MLIR-LABEL: "builtin.module"()
// MLIR: "func.func"() <{function_type = (i32) -> (), sym_name = "main"}> ({
// MLIR: %[[SWITCH:.*]]:4 = "simt_step.switch"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) ({
// MLIR:   %[[CASE1:.*]]:4 = "simt_step.if"(%{{.*}}) ({
// MLIR:     %[[C2:.*]] = "arith.constant"() <{value = 2 : i32}>
// MLIR:     "simt_step.yield"(%[[C2]], {{.*}}, {{.*}}, {{.*}}) : (i32, i1, i1, i1) -> ()
// MLIR:   }, {
// MLIR:     "simt_step.yield"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i32, i1, i1, i1) -> ()
// MLIR:   }) : (i1) -> (i32, i1, i1, i1)
// MLIR:   %[[CASE2:.*]]:4 = "simt_step.if"(%{{.*}}) ({
// MLIR:     %[[C5:.*]] = "arith.constant"() <{value = 5 : i32}>
// MLIR:     "simt_step.yield"(%[[C5]], {{.*}}, {{.*}}, {{.*}}) : (i32, i1, i1, i1) -> ()
// MLIR:   }, {
// MLIR:     "simt_step.yield"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i32, i1, i1, i1) -> ()
// MLIR:   }) : (i1) -> (i32, i1, i1, i1)
// MLIR:   %[[DEFAULT:.*]]:4 = "simt_step.if"(%{{.*}}) ({
// MLIR:     %[[C7:.*]] = "arith.constant"() <{value = 7 : i32}>
// MLIR:     "simt_step.yield"(%[[C7]], {{.*}}, {{.*}}, {{.*}}) : (i32, i1, i1, i1) -> ()
// MLIR:   }, {
// MLIR:     "simt_step.yield"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i32, i1, i1, i1) -> ()
// MLIR:   }) : (i1) -> (i32, i1, i1, i1)
// MLIR: }) {case_values = array<i64: 0, 1, 2, 0>} : (i32, i32, i1, i1, i1) -> (i32, i1, i1, i1)
// MLIR: return
