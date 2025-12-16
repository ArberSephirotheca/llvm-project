"builtin.module"() ({
  "func.func"() <{function_type = (vector<3xi32>) -> (), sym_name = "main"}> ({
  ^bb0(%arg0: vector<3xi32>):
    %0 = "simt_step.dispatch_thread_id"() : () -> vector<3xi32>
    %1 = "vector.extract"(%0) <{static_position = array<i64: 0>}> : (vector<3xi32>) -> i32
    %2 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %3 = "arith.cmpi"(%1, %2) <{predicate = 0 : i64}> : (i32, i32) -> i1
    %4 = "simt_step.wave_count_bits"(%3) : (i1) -> i32
    %5 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %6 = "arith.cmpi"(%4, %5) <{predicate = 0 : i64}> : (i32, i32) -> i1
    "simt_step.if"(%6) ({
      "simt_step.return"() : () -> ()
    }, {
    }) : (i1) -> ()
  }) {simt.num_threads = array<i64: 1, 1, 1>} : () -> ()
}) : () -> ()