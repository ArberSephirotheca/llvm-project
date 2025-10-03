builtin.module {
  func.func @switch_loop(%sel: i32) {
    %zero = arith.constant 0 : i32
    %false = arith.constant false
    %switch:4 = "simt_step.switch"(%sel, %zero, %false, %false, %false) ({
    ^bb0(%case0_val: i32, %case0_has: i1, %case0_exec: i1, %case0_done: i1):
      %init = arith.constant 0 : i32
      %loop:2 = "simt_step.loop"(%init, %case0_val) ({
      ^bb0(%iter: i32, %acc: i32):
        %limit = arith.constant 1 : i32
        %cond = arith.cmpi sle, %iter, %limit : i32
        "simt_step.condition"(%cond, %iter, %acc) : (i1, i32, i32) -> ()
      }, {
      ^bb0(%iter_body: i32, %acc_body: i32):
        %one = arith.constant 1 : i32
        %next_iter = arith.addi %iter_body, %one : i32
        %next_acc = arith.addi %acc_body, %one : i32
        "simt_step.yield"(%next_iter, %next_acc) : (i32, i32) -> ()
      }) : (i32, i32) -> (i32, i32)
      "simt_step.yield"(%loop#1, %case0_has, %case0_exec, %case0_done)
        : (i32, i1, i1, i1) -> ()
    ^bb1(%case1_val: i32, %case1_has: i1, %case1_exec: i1, %case1_done: i1):
      %two = arith.constant 2 : i32
      %bump = arith.addi %case1_val, %two : i32
      "simt_step.yield"(%bump, %case1_has, %case1_exec, %case1_done)
        : (i32, i1, i1, i1) -> ()
    }) {case_values = array<i64: 0, 1>} : (i32, i32, i1, i1, i1) -> (i32, i1, i1, i1)
    %sum = arith.addi %switch#0, %zero : i32
    func.return
  }
}

// XFAIL: *
// TODO: Loop inside switch currently exposes gaps in carried argument propagation.
// RUN: %simt-opt --simt-step-to-structured %s | %mlir-file-check %s

// CHECK-LABEL: func.func @switch_loop
// CHECK: "simt_struct.block"() ({
// CHECK:   ^bb0(%[[RET_MASK:.*]]: i64, %[[RET_VAL:.*]]: i32, %[[RET_HAS:.*]]: i1, %[[RET_EXEC:.*]]: i1, %[[RET_DONE:.*]]: i1):
// CHECK:     %[[ADD:.*]] = arith.addi %[[RET_VAL]], %{{.*}} : i32
// CHECK:     "simt_struct.return"() : () -> ()
// CHECK: }) {sym_name = "block{{[0-9]+}}"}
