// MLIR-LABEL: func.func @main(
// MLIR-SAME: %[[TID:arg0]]: i32, %[[SHARED:arg1]]: !simt_step.resource<Shared, i32>
// MLIR: "simt_step.buffer.store"(%[[SHARED]], %[[TID]], %[[TID]]) : (!simt_step.resource<Shared, i32>, i32, i32) -> ()
// MLIR: %[[LOAD:.*]] = "simt_step.buffer.load"(%[[SHARED]], %[[TID]]) : (!simt_step.resource<Shared, i32>, i32) -> i32
// MLIR: return
