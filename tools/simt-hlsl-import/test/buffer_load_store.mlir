// MLIR-LABEL: func.func @main(
// MLIR-SAME: %[[TID:arg0]]: i32, %[[BUF:arg1]]: !simt_step.resource<Global, i32>
// MLIR: %[[LOAD:.*]] = "simt_step.buffer.load"(%[[BUF]], %[[TID]]) : (!simt_step.resource<Global, i32>, i32) -> i32
// MLIR: %[[INC:.*]] = arith.addi %[[LOAD]], %c1_i32 : i32
// MLIR: "simt_step.buffer.store"(%[[BUF]], %[[TID]], %[[INC]]) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
// MLIR: return
