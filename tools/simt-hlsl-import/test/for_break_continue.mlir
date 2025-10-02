// MLIR-LABEL: "func.func"() <{function_type = (i32) -> (), sym_name = "main"}>
// MLIR: %[[MASK:.*]] = "simt_step.active_mask"() : () -> i64
// MLIR: %{{.*}} = "simt_step.dispatch_thread_id"() : () -> i32
// MLIR: %[[ACC_INIT:.*]] = "arith.constant"() <{value = 0 : i32}> : () -> i32
// MLIR: %[[I_INIT:.*]] = "arith.constant"() <{value = 0 : i32}> : () -> i32
// MLIR: %[[LOOP:.*]]:2 = "simt_step.loop"(%[[ACC_INIT]], %[[I_INIT]]) ({
// MLIR:   ^bb0(%[[ACC_PH:.*]]: i32, %[[I_PH:.*]]: i32):
// MLIR:     %[[LIMIT:.*]] = "arith.constant"() <{value = 4 : i32}> : () -> i32
// MLIR:     %[[COND:.*]] = "arith.cmpi"(%[[I_PH]], %[[LIMIT]]) <{predicate = 2 : i64}> : (i32, i32) -> i1
// MLIR:     "simt_step.condition"(%[[COND]], %[[ACC_PH]], %[[I_PH]]) : (i1, i32, i32) -> ()
// MLIR: }, {
// MLIR:   ^bb0(%[[ACC_BODY:.*]]: i32, %[[I_BODY:.*]]: i32):
// MLIR:     %[[ONE:.*]] = "arith.constant"() <{value = 1 : i32}> : () -> i32
// MLIR:     %[[IS_CONT:.*]] = "arith.cmpi"(%[[I_BODY]], %[[ONE]]) <{predicate = 0 : i64}> : (i32, i32) -> i1
// MLIR:     %[[CONT:.*]]:2 = "simt_step.if"(%[[IS_CONT]]) ({
// MLIR:       "simt_step.continue"(%[[ACC_BODY]], %[[I_BODY]]) : (i32, i32) -> ()
// MLIR:     }, {
// MLIR:       "simt_step.yield"(%[[ACC_BODY]], %[[I_BODY]]) : (i32, i32) -> ()
// MLIR:     }) : (i1) -> (i32, i32)
// MLIR:     %[[THREE:.*]] = "arith.constant"() <{value = 3 : i32}> : () -> i32
// MLIR:     %[[IS_BREAK:.*]] = "arith.cmpi"(%[[CONT]]#1, %[[THREE]]) <{predicate = 0 : i64}> : (i32, i32) -> i1
// MLIR:     %[[AFTER_BREAK:.*]]:2 = "simt_step.if"(%[[IS_BREAK]]) ({
// MLIR:       "simt_step.break"(%[[CONT]]#0, %[[CONT]]#1) : (i32, i32) -> ()
// MLIR:     }, {
// MLIR:       "simt_step.yield"(%[[CONT]]#0, %[[CONT]]#1) : (i32, i32) -> ()
// MLIR:     }) : (i1) -> (i32, i32)
// MLIR:     %[[ACC_NEXT:.*]] = "arith.addi"(%[[AFTER_BREAK]]#0, %[[AFTER_BREAK]]#1) {{.*}}
// MLIR:     %[[I_NEXT:.*]] = "arith.addi"(%[[AFTER_BREAK]]#1, %{{.*}}) {{.*}}
// MLIR:     "simt_step.yield"(%[[ACC_NEXT]], %[[I_NEXT]]) : (i32, i32) -> ()
// MLIR: }) : (i32, i32) -> (i32, i32)
// MLIR: "func.return"() : () -> ()
