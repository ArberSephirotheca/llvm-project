// MLIR-LABEL: func.func @main(
// MLIR-SAME: %[[TID:arg0]]: i32)
// MLIR: attributes {simt.num_threads = dense<[8, 4, 1]> : vector<3xi64>}
// MLIR: %[[A:.*]] = arith.constant 42 : i32
// MLIR: %[[B:.*]] = arith.constant 13 : i32
// MLIR: %[[TWO:.*]] = arith.constant 2 : i32
// MLIR: %[[TMP:.*]] = arith.muli %[[B]], %[[TWO]] : i32
// MLIR: %[[SUM:.*]] = arith.addi %[[A]], %[[TMP]] : i32
// MLIR: %[[RES:.*]] = arith.subi %[[SUM]], %[[TID]] : i32
// MLIR: func.return
*** End Patch
