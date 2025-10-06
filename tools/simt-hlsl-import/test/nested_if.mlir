// MLIR-LABEL: module {
// MLIR:   func.func @main(%{{.*}}: i32) attributes {simt.num_threads = array<i64: 1, 1, 1>} {
// MLIR:     %{{.*}} = "simt_step.active_mask"()
// MLIR:     %[[DISPATCH:.*]] = simt_step.dispatch_thread_id : i32
// MLIR:     %[[COND0:.*]] = arith.cmpi ne, {{.*}} : i32
// MLIR:     %[[COND1:.*]] = arith.cmpi ne, {{.*}} : i32
// MLIR:     %[[OUTER:.*]] = "simt_step.if"(%[[COND0]]) ({
// MLIR:       %[[INNER:.*]] = "simt_step.if"(%[[COND1]]) ({
// MLIR:         %{{.*}} = arith.constant 1 : i32
// MLIR:         "simt_step.yield"(%{{.*}}) : (i32) -> ()
// MLIR:       }, {
// MLIR:         %{{.*}} = arith.constant 2 : i32
// MLIR:         "simt_step.yield"(%{{.*}}) : (i32) -> ()
// MLIR:       }) : (i1) -> i32
// MLIR:       "simt_step.yield"(%[[INNER]]) : (i32) -> ()
// MLIR:     }, {
// MLIR:       %{{.*}} = arith.constant 0 : i32
// MLIR:       "simt_step.yield"(%{{.*}}) : (i32) -> ()
// MLIR:     }) : (i1) -> i32
// MLIR:     return
// MLIR:   }
// MLIR: }
