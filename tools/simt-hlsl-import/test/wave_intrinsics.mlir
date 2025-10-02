// MLIR-LABEL: "func.func"() <{function_type = (i32) -> (), sym_name = "main"}>
// MLIR: %[[MASK:.*]] = "simt_step.active_mask"() : () -> i64
// MLIR: %[[TID:.*]] = "simt_step.dispatch_thread_id"() : () -> i32
// MLIR: %[[ZERO_EQ:.*]] = "arith.constant"() <{value = 0 : i32}> : () -> i32
// MLIR: %[[EQ:.*]] = "arith.cmpi"(%[[TID]], %[[ZERO_EQ]]) <{predicate = 0 : i64}> : (i32, i32) -> i1
// MLIR: %[[ALL:.*]] = "simt_step.wave_all"(%[[EQ]]) : (i1) -> i1
// MLIR: %[[ZERO_NE:.*]] = "arith.constant"() <{value = 0 : i32}> : () -> i32
// MLIR: %[[NE:.*]] = "arith.cmpi"(%[[TID]], %[[ZERO_NE]]) <{predicate = 1 : i64}> : (i32, i32) -> i1
// MLIR: %[[ANY:.*]] = "simt_step.wave_any"(%[[NE]]) : (i1) -> i1
// MLIR: %[[LANE:.*]] = "simt_step.lane_id"() : () -> index
// MLIR: "arith.index_cast"(%[[LANE]]) : (index) -> i32
