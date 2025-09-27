// MLIR-LABEL: "builtin.module"()
// MLIR: "func.func"() <{function_type = (i32) -> (), sym_name = "main"}> ({
// MLIR:   %[[SWITCH:.*]]:4 = "simt_step.switch"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) ({
// MLIR:     "simt_step.if"
// MLIR:     "simt_step.switch"
// MLIR:   }) {case_values = array<i64: 0, 1, 0, 4>} : (i32, i32, i1, i1, i1) -> (i32, i1, i1, i1)
// MLIR:   "simt_step.loop"
// MLIR-NOT: "simt_step.break"
// MLIR: })
