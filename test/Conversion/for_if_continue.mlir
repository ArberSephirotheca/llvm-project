module {
  func.func @main(%arg0: i32) attributes {simt.num_threads = array<i64: 1, 1, 1>} {
    %0 = simt_step.dispatch_thread_id : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %1:2 = "simt_step.loop"(%c0_i32, %c0_i32_0) ({
    ^bb0(%arg1: i32, %arg2: i32):
      %c4_i32 = arith.constant 4 : i32
      %2 = arith.cmpi slt, %arg2, %c4_i32 : i32
      "simt_step.condition"(%2, %arg1, %arg2) : (i1, i32, i32) -> ()
    }, {
    ^bb0(%arg1: i32, %arg2: i32):
      %c1_i32 = arith.constant 1 : i32
      %2 = arith.cmpi eq, %arg2, %c1_i32 : i32
      %3:3 = "simt_step.if"(%2) ({
        %c1_i32_2 = arith.constant 1 : i32
        %6 = arith.addi %arg2, %c1_i32_2 : i32
        %true = arith.constant true
        "simt_step.yield"(%true, %arg1, %6) : (i1, i32, i32) -> ()
      }, {
        %false = arith.constant false
        "simt_step.yield"(%false, %arg1, %arg2) : (i1, i32, i32) -> ()
      }) : (i1) -> (i1, i32, i32)
      "simt_step.if"(%3#0) ({
        "simt_step.continue"(%3#1, %3#2) : (i32, i32) -> ()
      }, {
        "simt_step.yield"(%3#1, %3#2) : (i32, i32) -> ()
      }) {simt.normalized.loop_terminators} : (i1) -> ()
      %4 = arith.addi %3#1, %3#2 : i32
      %c1_i32_1 = arith.constant 1 : i32
      %5 = arith.addi %3#2, %c1_i32_1 : i32
      "simt_step.yield"(%4, %5) : (i32, i32) -> ()
    }) : (i32, i32) -> (i32, i32)
    func.return
  }
}
