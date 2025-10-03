builtin.module {
  func.func @loop_switch(%seed: i32) {
    %zero = arith.constant 0 : i32
    %false = arith.constant false
    %one = arith.constant 1 : i32
    %loop:2 = "simt_step.loop"(%zero, %zero) (
      {
      ^bb0(%iter: i32, %acc: i32):
        %limit = arith.constant 3 : i32
        %cond = arith.cmpi slt, %iter, %limit : i32
        "simt_step.condition"(%cond, %iter, %acc) : (i1, i32, i32) -> ()
      },
      {
      ^bb0(%iter_body: i32, %acc_body: i32):
        %switch_false = arith.constant false
        %switch:4 = "simt_step.switch"(%iter_body, %acc_body, %switch_false, %switch_false, %switch_false) ({
        ^bb0(%case0_val: i32, %case0_has: i1, %case0_exec: i1, %case0_done: i1):
          %one_local = arith.constant 1 : i32
          %case_inc = arith.addi %case0_val, %one_local : i32
          "simt_step.yield"(%case_inc, %case0_has, %case0_exec, %case0_done)
            : (i32, i1, i1, i1) -> ()
        ^bb1(%case1_val: i32, %case1_has: i1, %case1_exec: i1, %case1_done: i1):
          %two = arith.constant 2 : i32
          %case_seed = arith.addi %case1_val, %two : i32
          "simt_step.yield"(%case_seed, %case1_has, %case1_exec, %case1_done)
            : (i32, i1, i1, i1) -> ()
        }) {case_values = array<i64: 0, 1>} : (i32, i32, i1, i1, i1) -> (i32, i1, i1, i1)
        "simt_step.yield"(%iter_body, %switch#0) : (i32, i32) -> ()
      }
    ) : (i32, i32) -> (i32, i32)
    %final = arith.addi %loop#1, %zero : i32
    func.return
  }
}

// XFAIL: *
// TODO: Nested switch inside loop currently triggers invalid carried-argument wiring.
// RUN: %simt-opt --simt-step-to-structured %s | %mlir-file-check %s

// CHECK-LABEL: func.func @loop_switch
// CHECK: "simt_struct.block"() ({
// CHECK:   ^bb0(%[[RET_MASK:.*]]: i64, %[[RET_ITER:.*]]: i32, %[[RET_ACC:.*]]: i32):
// CHECK:     %[[ADD:.*]] = arith.addi %[[RET_ACC]], %{{.*}} : i32
// CHECK:     "simt_struct.return"() : () -> ()
// CHECK: }) {sym_name = "block{{[0-9]+}}"}
