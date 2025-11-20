module {
  func.func @main(%arg0: i32) attributes {simt.num_threads = array<i64: 1, 1, 1>} {
    %tid = "simt_step.dispatch_thread_id"() : () -> i32
    %c0 = arith.constant 0 : i32
    %cond = arith.cmpi eq, %tid, %c0 : i32
    %result = "simt_step.if"(%cond) ({
      %one = arith.constant 1 : i32
      "simt_step.yield"(%one) : (i32) -> ()
    }, {
      %zero = arith.constant 0 : i32
      "simt_step.yield"(%zero) : (i32) -> ()
    }) : (i1) -> i32
    func.return
  }
}
