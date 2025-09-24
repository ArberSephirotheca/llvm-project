// MLIR-LABEL: "func.func"() <{function_type = (i32) -> (), sym_name = "main"}>
// MLIR: %[[MASK:.*]] = "simt_step.active_mask"() : () -> i64
// MLIR: %[[ACC_INIT:.*]] = "arith.constant"() <{value = 0 : i32}> : () -> i32
// MLIR: %[[I_INIT:.*]] = "arith.constant"() <{value = 0 : i32}> : () -> i32
// MLIR: %[[LOOP:.*]]:2 = "simt_step.loop"(%[[ACC_INIT]], %[[I_INIT]]) ({
// MLIR:   ^bb0(%[[ACC_PH:.*]]: i32, %[[I_PH:.*]]: i32):
// MLIR:     %[[LIMIT:.*]] = "arith.constant"() <{value = 3 : i32}> : () -> i32
// MLIR:     %[[COND:.*]] = "arith.cmpi"(%[[I_PH]], %[[LIMIT]]) <{predicate = 2 : i64}> : (i32, i32) -> i1
// MLIR:     "simt_step.condition"(%[[COND]], %[[ACC_PH]], %[[I_PH]]) : (i1, i32, i32) -> ()
// MLIR: }, {
// MLIR:   ^bb0(%[[ACC_BODY:.*]]: i32, %[[I_BODY:.*]]: i32):
// MLIR:     %[[ACC_NEXT:.*]] = "arith.addi"(%[[ACC_BODY]], %[[I_BODY]]) {{.*}}
// MLIR:     %[[ONE:.*]] = "arith.constant"() <{value = 1 : i32}> : () -> i32
// MLIR:     %[[I_NEXT:.*]] = "arith.addi"(%[[I_BODY]], %[[ONE]]) {{.*}}
// MLIR:     "simt_step.yield"(%[[ACC_NEXT]], %[[I_NEXT]]) : (i32, i32) -> ()
// MLIR: }) : (i32, i32) -> (i32, i32)
// MLIR: "func.return"() : () -> ()
