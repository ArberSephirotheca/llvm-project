// MLIR-LABEL: "builtin.module"()
// MLIR:   "func.func"() <{function_type = (vector<3xi32>, vector<3xi32>, i32, vector<3xi32>) -> (), sym_name = "main"}>
// MLIR:     %[[MASK:.*]] = "simt_step.active_mask"() : () -> i64
// MLIR:     %[[LOCAL:.*]] = "simt_step.group_thread_id"() : () -> vector<3xi32>
// MLIR:     %[[GROUP:.*]] = "simt_step.group_id"() : () -> vector<3xi32>
// MLIR:     %[[INDEX:.*]] = "simt_step.group_index"() : () -> i32
// MLIR:     %[[DISPATCH:.*]] = "simt_step.dispatch_thread_id"() : () -> vector<3xi32>
// MLIR:     "func.return"() : () -> ()
