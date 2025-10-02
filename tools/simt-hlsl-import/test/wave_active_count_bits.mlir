// MLIR-LABEL: "builtin.module"()
// MLIR:   "func.func"() <{function_type = (i32) -> (), sym_name = "main"}>
// MLIR:     %[[MASK:.*]] = "simt_step.active_mask"() : () -> i64
// MLIR:     %[[TID:.*]] = "simt_step.dispatch_thread_id"() : () -> i32
// MLIR:     %[[ZERO:.*]] = "arith.constant"() <{value = 0 : i32}> : () -> i32
// MLIR:     %[[PRED:.*]] = "arith.cmpi"(%[[TID]], %[[ZERO]]) <{predicate = 0 : i64}> : (i32, i32) -> i1
// MLIR:     %[[BALLOT:.*]] = "simt_step.wave_ballot"(%[[PRED]]) : (i1) -> i64
// MLIR:     %[[CTPOP:.*]] = "math.ctpop"(%[[BALLOT]]) : (i64) -> i64
// MLIR:     "arith.trunci"(%[[CTPOP]]) <{overflowFlags = #arith.overflow<none>}> : (i64) -> i32
// MLIR:     "func.return"() : () -> ()
