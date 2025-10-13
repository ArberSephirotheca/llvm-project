// MLIR-LABEL: "builtin.module"()
// MLIR:   "func.func"() <{function_type = (i32) -> (), sym_name = "main"}>
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
// MLIR:     %[[CONT_STAGE1:.*]]:3 = "simt_step.if"(%[[IS_CONT]]) ({
// MLIR:       %[[INC_ONE:.*]] = "arith.constant"() <{value = 1 : i32}> : () -> i32
// MLIR:       %[[I_CONT_NEXT:.*]] = "arith.addi"(%[[I_BODY]], %[[INC_ONE]]) {{.*}}
// MLIR:       %[[CONT_TRUE:.*]] = "arith.constant"() <{value = true}> : () -> i1
// MLIR:       "simt_step.yield"(%[[CONT_TRUE]], %[[ACC_BODY]], %[[I_CONT_NEXT]]) : (i1, i32, i32) -> ()
// MLIR:     }, {
// MLIR:       %[[CONT_FALSE:.*]] = "arith.constant"() <{value = false}> : () -> i1
// MLIR:       "simt_step.yield"(%[[CONT_FALSE]], %[[ACC_BODY]], %[[I_BODY]]) : (i1, i32, i32) -> ()
// MLIR:     }) : (i1) -> (i1, i32, i32)
// MLIR:     "simt_step.if"(%[[CONT_STAGE1]]#0) ({
// MLIR:       "simt_step.continue"(%[[CONT_STAGE1]]#1, %[[CONT_STAGE1]]#2) : (i32, i32) -> ()
// MLIR:     }, {
// MLIR:       "simt_step.yield"() : () -> ()
// MLIR:     }) {simt.normalized.loop_terminators} : (i1) -> ()
// MLIR:     %[[THREE:.*]] = "arith.constant"() <{value = 3 : i32}> : () -> i32
// MLIR:     %[[IS_BREAK:.*]] = "arith.cmpi"(%[[CONT_STAGE1]]#2, %[[THREE]]) <{predicate = 0 : i64}> : (i32, i32) -> i1
// MLIR:     %[[BREAK_STAGE1:.*]]:3 = "simt_step.if"(%[[IS_BREAK]]) ({
// MLIR:       %[[BREAK_TRUE:.*]] = "arith.constant"() <{value = true}> : () -> i1
// MLIR:       "simt_step.yield"(%[[BREAK_TRUE]], %[[CONT_STAGE1]]#1, %[[CONT_STAGE1]]#2) : (i1, i32, i32) -> ()
// MLIR:     }, {
// MLIR:       %[[BREAK_FALSE:.*]] = "arith.constant"() <{value = false}> : () -> i1
// MLIR:       "simt_step.yield"(%[[BREAK_FALSE]], %[[CONT_STAGE1]]#1, %[[CONT_STAGE1]]#2) : (i1, i32, i32) -> ()
// MLIR:     }) : (i1) -> (i1, i32, i32)
// MLIR:     "simt_step.if"(%[[BREAK_STAGE1]]#0) ({
// MLIR:       "simt_step.break"(%[[BREAK_STAGE1]]#1, %[[BREAK_STAGE1]]#2) : (i32, i32) -> ()
// MLIR:     }, {
// MLIR:       "simt_step.yield"() : () -> ()
// MLIR:     }) {simt.normalized.loop_terminators} : (i1) -> ()
// MLIR:     %[[ACC_NEXT:.*]] = "arith.addi"(%[[CONT_STAGE1]]#1, %[[CONT_STAGE1]]#2) {{.*}}
// MLIR:     %[[I_NEXT:.*]] = "arith.addi"(%[[CONT_STAGE1]]#2, %{{.*}}) {{.*}}
// MLIR:     "simt_step.yield"(%[[ACC_NEXT]], %[[I_NEXT]]) : (i32, i32) -> ()
// MLIR: }) : (i32, i32) -> (i32, i32)
// MLIR:   "func.return"() : () -> ()
