// MLIR: %[[BALLOT:.*]] = "simt_step.wave_ballot"(%{{.*}}) : (i1) -> i64
// MLIR: %[[CTPOP:.*]] = "math.ctpop"(%[[BALLOT]]) : (i64) -> i64
// MLIR: %[[TRUNC:.*]] = "arith.trunci"(%[[CTPOP]]) {{.*}} : (i64) -> i32
