// MLIR-LABEL: func.func @main(
// MLIR: %[[MASK:.*]] = "simt_step.active_mask"() : () -> i64
// MLIR: simt_step.dispatch_thread_id : i32
// MLIR: return
