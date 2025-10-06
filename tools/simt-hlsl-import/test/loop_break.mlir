// MLIR-LABEL: module {
// MLIR:   func.func @main(%arg0: i32) attributes {simt.num_threads = array<i64: 1, 1, 1>} {
// MLIR:     %[[MASK:.*]] = "simt_step.active_mask"() : () -> i64
// MLIR:     %{{.*}} = simt_step.dispatch_thread_id : i32
// MLIR:     %[[ZERO:.*]] = arith.constant 0 : i32
// MLIR:     %[[LOOP:.*]]:2 = "simt_step.loop"(%{{.*}}, %[[ZERO]]) ({
// MLIR:       ^bb0(%[[VALUE:.*]]: i32, %[[ITER:.*]]: i32):
// MLIR:         %[[LIMIT:.*]] = arith.constant 1 : i32
// MLIR:         %[[COND:.*]] = arith.cmpi slt, %[[ITER]], %[[LIMIT]] : i32
// MLIR:         "simt_step.condition"(%[[COND]], %[[VALUE]], %[[ITER]]) : (i1, i32, i32) -> ()
// MLIR:     }, {
// MLIR:       ^bb0(%[[BODY_VALUE:.*]]: i32, %[[BODY_ITER:.*]]: i32):
// MLIR:         "simt_step.break"(%[[BODY_VALUE]], %[[BODY_ITER]]) : (i32, i32) -> ()
// MLIR:     }) : (i32, i32) -> (i32, i32)
// MLIR:     return
// MLIR:   }
// MLIR: }
