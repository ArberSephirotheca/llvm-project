// MLIR-LABEL: func.func @main(
// MLIR-SAME: %[[TID:arg0]]: i32)
// MLIR: %[[ZERO:.*]] = arith.constant 0 : i32
// MLIR: simt_step.if %[[COND:.*]] {
// MLIR:   simt_step.yield
// MLIR: } else {
// MLIR:   simt_step.yield
// MLIR: }
// MLIR: func.return
