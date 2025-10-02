// MLIR-LABEL: module {
// MLIR: func.func @main(
// MLIR: %[[MASK:.*]] = "simt_step.active_mask"() : () -> i64
// MLIR: simt_step.dispatch_thread_id : i32
// MLIR: return
