// MLIR-LABEL: module {
// MLIR:   func.func @main(%arg0: i32) attributes {simt.num_threads = array<i64: 1, 1, 1>} {
// MLIR:     %[[TID:.*]] = simt_step.dispatch_thread_id : i32
// MLIR:     %[[ZERO:.*]] = arith.constant 0 : i32
// MLIR:     %[[PRED:.*]] = arith.cmpi eq, %[[TID]], %[[ZERO]] : i32
// MLIR:     %[[COUNT:.*]] = "simt_step.wave_ballot"(%[[PRED]]) : (i1) -> i32
// MLIR:     func.return
// MLIR:   }
// MLIR: }
