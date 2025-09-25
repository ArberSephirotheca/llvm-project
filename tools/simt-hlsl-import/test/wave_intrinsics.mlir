// MLIR: "simt_step.wave_all"(%{{.*}}) : (i1) -> i1
// MLIR: "simt_step.wave_any"(%{{.*}}) : (i1) -> i1
// MLIR: "simt_step.lane_id"() : () -> index
// MLIR: "arith.index_cast"(%{{.*}}) : (index) -> i32
