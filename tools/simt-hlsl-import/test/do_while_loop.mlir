// MLIR-LABEL: "builtin.module"()
// MLIR:   "func.func"() <{function_type = (i32) -> (), sym_name = "main"}>
// MLIR:     %{{.*}} = "simt_step.dispatch_thread_id"() : () -> i32
// MLIR:     %[[INIT0:.*]] = "arith.constant"() <{value = 0 : i32}> : () -> i32
// MLIR:     %[[INIT1:.*]] = "arith.constant"() <{value = 0 : i32}> : () -> i32
// MLIR:     %[[FLAG_INIT:.*]] = "arith.constant"() <{value = true}> : () -> i1
// MLIR:     %[[LOOP:.*]]:3 = "simt_step.loop"(%[[INIT0]], %[[INIT1]], %[[FLAG_INIT]]) ({
// MLIR:   ^bb0(%[[PH0:.*]]: i32, %[[PH1:.*]]: i32, %[[PHFLAG:.*]]: i1):
// MLIR:     %[[COND_IF:.*]]:3 = "simt_step.if"(%[[PHFLAG]]) ({
// MLIR:       %[[TRUE:.*]] = "arith.constant"() <{value = true}> : () -> i1
// MLIR:       "simt_step.yield"(%[[TRUE]], %[[PH0]], %[[PH1]]) : (i1, i32, i32) -> ()
// MLIR:     }, {
// MLIR:       %[[LIMIT:.*]] = "arith.constant"() <{value = 5 : i32}> : () -> i32
// MLIR:       %[[TEST:.*]] = "arith.cmpi"(%[[PH0]], %[[LIMIT]]) <{predicate = 2 : i64}> : (i32, i32) -> i1
// MLIR:       "simt_step.yield"(%[[TEST]], %[[PH0]], %[[PH1]]) : (i1, i32, i32) -> ()
// MLIR:     }) : (i1) -> (i1, i32, i32)
// MLIR:     %[[FALSE:.*]] = "arith.constant"() <{value = false}> : () -> i1
// MLIR:     "simt_step.condition"(%[[COND_IF]]#0, %[[COND_IF]]#1, %[[COND_IF]]#2, %[[FALSE]]) : (i1, i32, i32, i1) -> ()
// MLIR: }, {
// MLIR:   ^bb0(%[[BODY0:.*]]: i32, %[[BODY1:.*]]: i32, %[[BODYFLAG:.*]]: i1):
// MLIR:     %[[FALSE2:.*]] = "arith.constant"() <{value = false}> : () -> i1
// MLIR:     %[[ONE:.*]] = "arith.constant"() <{value = 1 : i32}> : () -> i32
// MLIR:     %[[IS_CONT:.*]] = "arith.cmpi"(%[[BODY0]], %[[ONE]]) <{predicate = 0 : i64}> : (i32, i32) -> i1
// MLIR:     %[[CONT_STAGE1:.*]]:4 = "simt_step.if"(%[[IS_CONT]]) ({
// MLIR:       %[[INC:.*]] = "arith.constant"() <{value = 1 : i32}> : () -> i32
// MLIR:       %[[NEXT_I_CONT:.*]] = "arith.addi"(%[[BODY0]], %[[INC]]) {{.*}}
// MLIR:       %[[CONT_TRUE:.*]] = "arith.constant"() <{value = true}> : () -> i1
// MLIR:       "simt_step.yield"(%[[CONT_TRUE]], %[[NEXT_I_CONT]], %[[BODY1]], %[[FALSE2]]) : (i1, i32, i32, i1) -> ()
// MLIR:     }, {
// MLIR:       %[[CONT_FALSE:.*]] = "arith.constant"() <{value = false}> : () -> i1
// MLIR:       "simt_step.yield"(%[[CONT_FALSE]], %[[BODY0]], %[[BODY1]], %{{.*}}) : (i1, i32, i32, i1) -> ()
// MLIR:     }) : (i1) -> (i1, i32, i32, i1)
// MLIR:     %[[CONT_STAGE2:.*]]:2 = "simt_step.if"(%[[CONT_STAGE1]]#0) ({
// MLIR:       "simt_step.continue"(%[[CONT_STAGE1]]#1, %[[CONT_STAGE1]]#2, %[[CONT_STAGE1]]#3) : (i32, i32, i1) -> ()
// MLIR:     }, {
// MLIR:       "simt_step.yield"(%[[CONT_STAGE1]]#1, %[[CONT_STAGE1]]#2) : (i32, i32) -> ()
// MLIR:     }) {simt.normalized.loop_terminators} : (i1) -> (i32, i32)
// MLIR:     %[[THREE:.*]] = "arith.constant"() <{value = 3 : i32}> : () -> i32
// MLIR:     %[[IS_BREAK:.*]] = "arith.cmpi"(%[[CONT_STAGE2]]#0, %[[THREE]]) <{predicate = 0 : i64}> : (i32, i32) -> i1
// MLIR:     %[[BREAK_STAGE1:.*]]:4 = "simt_step.if"(%[[IS_BREAK]]) ({
// MLIR:       %[[BREAK_TRUE:.*]] = "arith.constant"() <{value = true}> : () -> i1
// MLIR:       "simt_step.yield"(%[[BREAK_TRUE]], %[[CONT_STAGE2]]#0, %[[CONT_STAGE2]]#1, %[[FALSE2]]) : (i1, i32, i32, i1) -> ()
// MLIR:     }, {
// MLIR:       %[[BREAK_FALSE:.*]] = "arith.constant"() <{value = false}> : () -> i1
// MLIR:       "simt_step.yield"(%[[BREAK_FALSE]], %[[CONT_STAGE2]]#0, %[[CONT_STAGE2]]#1, %{{.*}}) : (i1, i32, i32, i1) -> ()
// MLIR:     }) : (i1) -> (i1, i32, i32, i1)
// MLIR:     %[[BREAK_STAGE2:.*]]:2 = "simt_step.if"(%[[BREAK_STAGE1]]#0) ({
// MLIR:       "simt_step.break"(%[[BREAK_STAGE1]]#1, %[[BREAK_STAGE1]]#2, %[[BREAK_STAGE1]]#3) : (i32, i32, i1) -> ()
// MLIR:     }, {
// MLIR:       "simt_step.yield"(%[[BREAK_STAGE1]]#1, %[[BREAK_STAGE1]]#2) : (i32, i32) -> ()
// MLIR:     }) {simt.normalized.loop_terminators} : (i1) -> (i32, i32)
// MLIR:     %[[ACC_NEXT:.*]] = "arith.addi"(%[[BREAK_STAGE2]]#1, %[[BREAK_STAGE2]]#0) {{.*}}
// MLIR:     %[[INC_MAIN:.*]] = "arith.constant"() <{value = 1 : i32}> : () -> i32
// MLIR:     %[[I_NEXT:.*]] = "arith.addi"(%[[BREAK_STAGE2]]#0, %[[INC_MAIN]]) {{.*}}
// MLIR:     "simt_step.yield"(%[[I_NEXT]], %[[ACC_NEXT]], %[[FALSE2]]) : (i32, i32, i1) -> ()
// MLIR:     }) : (i32, i32, i1) -> (i32, i32, i1)
// MLIR:     "func.return"() : () -> ()
