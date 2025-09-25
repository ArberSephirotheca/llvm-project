// MLIR-LABEL: func.func @main(
// MLIR-SAME: %[[TID:arg0]]: i32, %[[BUF:arg1]]: !simt_step.resource<Global, i32>
// MLIR: %[[MASK:.*]] = "simt_step.active_mask"() : () -> i64
// MLIR: %c2_i32 = arith.constant 2 : i32
// MLIR: %[[EXCH:.*]] = "simt_step.buffer.atomic_exchange"(%[[BUF]], %[[TID]], %c2_i32) : (!simt_step.resource<Global, i32>, i32, i32) -> i32
// MLIR: %c3_i32 = arith.constant 3 : i32
// MLIR: %[[OFFSET:.*]] = arith.addi %[[TID]], %c3_i32 : i32
// MLIR: %[[CMP:.*]] = "simt_step.buffer.atomic_compare_exchange"(%[[BUF]], %[[TID]], %[[TID]], %[[OFFSET]]) : (!simt_step.resource<Global, i32>, i32, i32, i32) -> i32
// MLIR: %[[MIN:.*]] = "simt_step.buffer.atomic_min"(%[[BUF]], %[[TID]], %[[TID]]) : (!simt_step.resource<Global, i32>, i32, i32) -> i32
// MLIR: %[[MAX:.*]] = "simt_step.buffer.atomic_max"(%[[BUF]], %[[TID]], %[[TID]]) : (!simt_step.resource<Global, i32>, i32, i32) -> i32
// MLIR: %[[AND:.*]] = "simt_step.buffer.atomic_and"(%[[BUF]], %[[TID]], %[[TID]]) : (!simt_step.resource<Global, i32>, i32, i32) -> i32
// MLIR: %[[OR:.*]] = "simt_step.buffer.atomic_or"(%[[BUF]], %[[TID]], %[[TID]]) : (!simt_step.resource<Global, i32>, i32, i32) -> i32
// MLIR: %[[XOR:.*]] = "simt_step.buffer.atomic_xor"(%[[BUF]], %[[TID]], %[[TID]]) : (!simt_step.resource<Global, i32>, i32, i32) -> i32
// MLIR: return
