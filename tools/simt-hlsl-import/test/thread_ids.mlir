// MLIR-LABEL: module {
// MLIR:   func.func @main(%arg0: i32) attributes {simt.num_threads = array<i64: 1, 1, 1>} {
// MLIR:     %[[LOCAL:.*]] = simt_step.group_thread_id : vector<3xi32>
// MLIR:     %[[GROUP:.*]] = simt_step.group_id : vector<3xi32>
// MLIR:     %[[INDEX:.*]] = simt_step.group_index : i32
// MLIR:     %[[DISPATCH:.*]] = simt_step.dispatch_thread_id : vector<3xi32>
// MLIR:     func.return
