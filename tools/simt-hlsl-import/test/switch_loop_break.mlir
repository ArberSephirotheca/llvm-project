// MLIR-LABEL: module {
// MLIR:   func.func @main(%arg0: i32) attributes {simt.num_threads = array<i64: 1, 1, 1>} {
// MLIR:     %[[TID:.*]] = simt_step.dispatch_thread_id : i32
// MLIR:     %[[SWITCH:.*]]:4 = "simt_step.switch"(%[[TID]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) ({
// MLIR-COUNT: 2 "simt_step.if"
// MLIR:       ^bb0(%[[CASE0_VAL:.*]]: i32, %[[CASE0_HAS:.*]]: i1, %[[CASE0_EXEC:.*]]: i1, %[[CASE0_DONE:.*]]: i1):
// MLIR:         %[[CASE0_IF:.*]]:4 = "simt_step.if"(%{{.*}})
// MLIR:         "simt_step.yield"(%[[CASE0_IF]]#0, %[[CASE0_IF]]#1, %[[CASE0_IF]]#2, %[[CASE0_IF]]#3)
// MLIR:       ^bb1(%[[DEF_VAL:.*]]: i32, %[[DEF_HAS:.*]]: i1, %[[DEF_EXEC:.*]]: i1, %[[DEF_DONE:.*]]: i1):
// MLIR:         %[[DEF_IF:.*]]:4 = "simt_step.if"(%{{.*}})
// MLIR:     }) {case_values = array<i64: 0, 0>} : (i32, i32, i1, i1, i1) -> (i32, i1, i1, i1)
// MLIR:     return
// MLIR:   }
// MLIR: }
