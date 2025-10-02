// MLIR-LABEL: module {
// MLIR:   func.func @main(%arg0: i32) attributes {simt.num_threads = array<i64: 1, 1, 1>} {
// MLIR:     %[[MASK:.*]] = "simt_step.active_mask"() : () -> i64
// MLIR:     %{{.*}} = simt_step.dispatch_thread_id : i32
// MLIR:     %[[ACC_INIT:.*]] = arith.constant 0 : i32
// MLIR:     %[[I_INIT:.*]] = arith.constant 0 : i32
// MLIR:     %[[LOOP:.*]]:2 = "simt_step.loop"(%[[ACC_INIT]], %[[I_INIT]]) ({
// MLIR:       ^bb0(%[[ACC_PH:.*]]: i32, %[[I_PH:.*]]: i32):
// MLIR:         %[[LIMIT:.*]] = arith.constant 3 : i32
// MLIR:         %[[COND:.*]] = arith.cmpi slt, %[[I_PH]], %[[LIMIT]] : i32
// MLIR:         "simt_step.condition"(%[[COND]], %[[ACC_PH]], %[[I_PH]]) : (i1, i32, i32) -> ()
// MLIR:     }, {
// MLIR:       ^bb0(%[[ACC_BODY:.*]]: i32, %[[I_BODY:.*]]: i32):
// MLIR:         %[[ACC_NEXT:.*]] = arith.addi %[[ACC_BODY]], %[[I_BODY]] : i32
// MLIR:         %[[ONE:.*]] = arith.constant 1 : i32
// MLIR:         %[[I_NEXT:.*]] = arith.addi %[[I_BODY]], %[[ONE]] : i32
// MLIR:         "simt_step.yield"(%[[ACC_NEXT]], %[[I_NEXT]]) : (i32, i32) -> ()
// MLIR:     }) : (i32, i32) -> (i32, i32)
// MLIR:     return
// MLIR:   }
