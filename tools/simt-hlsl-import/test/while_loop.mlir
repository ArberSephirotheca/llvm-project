// MLIR-LABEL: "func.func"() <{function_type = (i32) -> (), sym_name = "main"}>
// MLIR: %[[MASK:.*]] = "simt_step.active_mask"() : () -> i64
// MLIR: %{{.*}} = "simt_step.dispatch_thread_id"() : () -> i32
// MLIR: %[[INIT0:.*]] = "arith.constant"() <{value = 0 : i32}> : () -> i32
// MLIR: %[[INIT1:.*]] = "arith.constant"() <{value = 0 : i32}> : () -> i32
// MLIR: %[[LOOP:.*]]:2 = "simt_step.loop"(%[[INIT0]], %[[INIT1]]) ({
// MLIR:   ^bb0(%[[PH0:.*]]: i32, %[[PH1:.*]]: i32):
// MLIR:     %[[LIMIT:.*]] = "arith.constant"() <{value = 4 : i32}> : () -> i32
// MLIR:     %[[COND:.*]] = "arith.cmpi"(%{{.*}}, %[[LIMIT]]) <{predicate = 2 : i64}> : (i32, i32) -> i1
// MLIR:     "simt_step.condition"(%[[COND]], %[[PH0]], %[[PH1]]) : (i1, i32, i32) -> ()
// MLIR: }, {
// MLIR:   ^bb0(%[[BODY0:.*]]: i32, %[[BODY1:.*]]: i32):
// MLIR:     %[[NEXT_ACC:.*]] = "arith.addi"(%[[BODY1]], %[[BODY0]]) {{.*}}
// MLIR:     %[[ONE:.*]] = "arith.constant"() <{value = 1 : i32}> : () -> i32
// MLIR:     %[[NEXT_I:.*]] = "arith.addi"(%[[BODY0]], %[[ONE]]) {{.*}}
// MLIR:     "simt_step.yield"(%[[NEXT_I]], %[[NEXT_ACC]]) : (i32, i32) -> ()
// MLIR: }) : (i32, i32) -> (i32, i32)
// MLIR: "func.return"() : () -> ()
