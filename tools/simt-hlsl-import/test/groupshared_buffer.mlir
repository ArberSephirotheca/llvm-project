// MLIR-LABEL: module {
// MLIR: func.func @main(
// MLIR-SAME: %[[SHARED:arg1]]: !simt_step.resource<Shared, i32>
// MLIR: %[[TID:.*]] = simt_step.dispatch_thread_id : i32
// MLIR: "simt_step.buffer.store"(%[[SHARED]], %[[TID]], %[[TID]]) : (!simt_step.resource<Shared, i32>, i32, i32) -> ()
// MLIR: %[[LOAD:.*]] = "simt_step.buffer.load"(%[[SHARED]], %[[TID]]) : (!simt_step.resource<Shared, i32>, i32) -> i32
// MLIR: return
