// MLIR-LABEL: func.func @main(
// MLIR-SAME: %[[TID:arg0]]: i32)
// MLIR: attributes {simt.num_threads = array<i64: 8, 4, 1>}
// MLIR: %[[MASK:.*]] = "simt_step.active_mask"() : () -> i64
// MLIR: %[[CONST_A:.*]] = arith.constant 42 : i32
// MLIR: %[[CONST_B:.*]] = arith.constant 13 : i32
// MLIR: %[[CONST_C:.*]] = arith.constant 2 : i32
// MLIR: %[[MUL:.*]] = arith.muli %[[CONST_B]], %[[CONST_C]] : i32
// MLIR: %[[ADD:.*]] = arith.addi %[[CONST_A]], %[[MUL]] : i32
// MLIR: %[[RES:.*]] = arith.subi %[[ADD]], %[[TID]] : i32
// MLIR: return
