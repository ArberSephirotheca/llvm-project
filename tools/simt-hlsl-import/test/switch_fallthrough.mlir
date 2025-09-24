// MLIR-LABEL: func.func @main(
// MLIR: %[[MASK:.*]] = "simt_step.active_mask"() : () -> i64
// MLIR: %[[CASE0:.*]]:4 = "simt_step.if"(%{{.*}}) ({
// MLIR:   %[[ADD1:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
// MLIR:   "simt_step.yield"(%[[ADD1]], %{{.*}}, %{{.*}}, %{{.*}}) : (i32, i1, i1, i1) -> ()
// MLIR: }, {
// MLIR:   "simt_step.yield"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i32, i1, i1, i1) -> ()
// MLIR: }) : (i1) -> (i32, i1, i1, i1)
// MLIR: %[[CASE1:.*]]:4 = "simt_step.if"(%{{.*}}) ({
// MLIR:   %[[ADD2:.*]] = arith.addi %[[CASE0]]#0, %{{.*}} : i32
// MLIR:   "simt_step.yield"(%[[ADD2]], %{{.*}}, %{{.*}}, %{{.*}}) : (i32, i1, i1, i1) -> ()
// MLIR: }, {
// MLIR:   "simt_step.yield"(%[[CASE0]]#0, %[[CASE0]]#1, %[[CASE0]]#2, %[[CASE0]]#3) : (i32, i1, i1, i1) -> ()
// MLIR: }) : (i1) -> (i32, i1, i1, i1)
// MLIR: %[[CASE2:.*]]:4 = "simt_step.if"(%{{.*}}) ({
// MLIR:   %[[ADD4:.*]] = arith.addi %[[CASE1]]#0, %{{.*}} : i32
// MLIR:   "simt_step.yield"(%[[ADD4]], %{{.*}}, %{{.*}}, %[[CASE1]]#3) : (i32, i1, i1, i1) -> ()
// MLIR: }, {
// MLIR:   "simt_step.yield"(%[[CASE1]]#0, %[[CASE1]]#1, %[[CASE1]]#2, %[[CASE1]]#3) : (i32, i1, i1, i1) -> ()
// MLIR: }) : (i1) -> (i32, i1, i1, i1)
// MLIR: %[[DEFAULT:.*]]:4 = "simt_step.if"(%{{.*}}) ({
// MLIR:   %[[ADD8:.*]] = arith.addi %[[CASE2]]#0, %{{.*}} : i32
// MLIR:   "simt_step.yield"(%[[ADD8]], %{{.*}}, %{{.*}}, %{{.*}}) : (i32, i1, i1, i1) -> ()
// MLIR: }, {
// MLIR:   "simt_step.yield"(%[[CASE2]]#0, %[[CASE2]]#1, %[[CASE2]]#2, %[[CASE2]]#3) : (i32, i1, i1, i1) -> ()
// MLIR: }) : (i1) -> (i32, i1, i1, i1)
// MLIR: return
