module {
  func.func @main(%arg0: i32) attributes {simt.num_threads = array<i64: 1, 1, 1>} {
    %0 = "simt_step.dispatch_thread_id"() : () -> i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %1 = arith.andi %0, %c1_i32 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %2 = arith.cmpi ne, %1, %c0_i32_0 : i32
    %c2_i32 = arith.constant 2 : i32
    %3 = arith.andi %0, %c2_i32 : i32
    %c0_i32_1 = arith.constant 0 : i32
    %4 = arith.cmpi ne, %3, %c0_i32_1 : i32
    %5 = "simt_step.if"(%2) ({
      %6 = "simt_step.if"(%4) ({
        %c1_i32_2 = arith.constant 1 : i32
        "simt_step.yield"(%c1_i32_2) : (i32) -> ()
      }, {
        %c2_i32_2 = arith.constant 2 : i32
        "simt_step.yield"(%c2_i32_2) : (i32) -> ()
      }) : (i1) -> i32
      "simt_step.yield"(%6) : (i32) -> ()
    }, {
      %c0_i32_2 = arith.constant 0 : i32
      "simt_step.yield"(%c0_i32_2) : (i32) -> ()
    }) : (i1) -> i32
    return
  }
}