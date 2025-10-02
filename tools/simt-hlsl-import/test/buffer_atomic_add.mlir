// MLIR-LABEL: func.func @main(
// MLIR-SAME: %[[BUF:arg1]]: !simt_step.resource<Global, i32>
// MLIR: %[[TID:.*]] = simt_step.dispatch_thread_id : i32
// MLIR: %c1_i32 = arith.constant 1 : i32
// MLIR: %[[OLD:.*]] = "simt_step.buffer.atomic_add"(%[[BUF]], %[[TID]], %c1_i32) : (!simt_step.resource<Global, i32>, i32, i32) -> i32
// MLIR: return
