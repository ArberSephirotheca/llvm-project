builtin.module {
  func.func @simple_switch(%sel: i32) {
    %zero = arith.constant 0 : i32
    %false = arith.constant false
    %result:4 = "simt_step.switch"(%sel, %zero, %false, %false, %false) ({
    ^bb0(%case0_val: i32, %case0_has: i1, %case0_exec: i1, %case0_done: i1):
      "simt_step.yield"(%case0_val, %case0_has, %case0_exec, %case0_done)
        : (i32, i1, i1, i1) -> ()
    ^bb1(%case1_val: i32, %case1_has: i1, %case1_exec: i1, %case1_done: i1):
      "simt_step.yield"(%case1_val, %case1_has, %case1_exec, %case1_done)
        : (i32, i1, i1, i1) -> ()
    }) {case_values = array<i64: 0, 0>} : (i32, i32, i1, i1, i1) -> (i32, i1, i1, i1)
    func.return
  }
}

// RUN: %simt-opt --simt-step-to-structured %s | %mlir-file-check %s

// CHECK-LABEL: func.func @simple_switch
// CHECK: "simt_struct.block"() ({
// CHECK:   ^bb0(%[[ENTRY_MASK:.*]]: i64, %[[ENTRY_SEL:.*]]: i32):
// CHECK:     %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK:     %[[FALSE:.*]] = arith.constant false
// CHECK:     "simt_struct.branch"(%[[ENTRY_MASK]], %[[ZERO]], %[[FALSE]], %[[FALSE]], %[[FALSE]]) {target = @block1}
// CHECK: }) {sym_name = "entry"}
// CHECK: "simt_struct.block"() ({
// CHECK:   ^bb0(%[[HDR_MASK:.*]]: i64, %[[HDR_VAL:.*]]: i32, %[[HDR_HAS:.*]]: i1, %[[HDR_EXEC:.*]]: i1, %[[HDR_DONE:.*]]: i1):
// CHECK:     "simt_struct.mask_push"(%[[HDR_MASK]]) {merge_target = @block5}
// CHECK:     "simt_struct.branch"(%[[HDR_MASK]], %[[HDR_VAL]], %[[HDR_HAS]], %[[HDR_EXEC]], %[[HDR_DONE]]) {target = @block2}
// CHECK: }) {merge_target = @block5, sym_name = "block1"}
// CHECK: "simt_struct.block"() ({
// CHECK:   ^bb0(%[[CASE0_MASK:.*]]: i64, %[[CASE0_VAL:.*]]: i32, %[[CASE0_HAS:.*]]: i1, %[[CASE0_EXEC:.*]]: i1, %[[CASE0_DONE:.*]]: i1):
// CHECK:     %[[CASE0_POP:.*]] = "simt_struct.mask_pop"()
// CHECK:     %[[CASE0_MERGE:.*]] = "simt_struct.mask_merge"(%[[CASE0_MASK]])
// CHECK:     %[[FALSE0:.*]] = arith.constant false
// CHECK:     %[[NOT_DONE0:.*]] = arith.cmpi eq, %[[CASE0_DONE]], %[[FALSE0]]
// CHECK:     %[[HEAD_EXEC0:.*]] = arith.andi %[[CASE0_EXEC]], %[[NOT_DONE0]]
// CHECK:     "simt_struct.cond_branch"(%[[HEAD_EXEC0]], %[[CASE0_MERGE]], %[[CASE0_MERGE]], %[[CASE0_VAL]], %[[CASE0_HAS]], %[[CASE0_EXEC]], %[[CASE0_DONE]], %[[CASE0_VAL]], %[[CASE0_HAS]], %[[CASE0_EXEC]], %[[CASE0_DONE]])
// CHECK-SAME: {false_target = @block5,
// CHECK-SAME: true_target = @block3}
// CHECK: }) {merge_target = @block5, sym_name = "block2"}
// CHECK: "simt_struct.block"() ({
// CHECK:   ^bb0(%[[FALL_MASK:.*]]: i64, %[[FALL_VAL:.*]]: i32, %[[FALL_HAS:.*]]: i1, %[[FALL_EXEC:.*]]: i1, %[[FALL_DONE:.*]]: i1):
// CHECK:     "simt_struct.mask_push"(%[[FALL_MASK]]) {merge_target = @block5}
// CHECK:     "simt_struct.branch"(%[[FALL_MASK]], %[[FALL_VAL]], %[[FALL_HAS]], %[[FALL_EXEC]], %[[FALL_DONE]]) {target = @block4}
// CHECK: }) {merge_target = @block5, sym_name = "block3"}
// CHECK: "simt_struct.block"() ({
// CHECK:   ^bb0(%[[CASE1_MASK:.*]]: i64, %[[CASE1_VAL:.*]]: i32, %[[CASE1_HAS:.*]]: i1, %[[CASE1_EXEC:.*]]: i1, %[[CASE1_DONE:.*]]: i1):
// CHECK:     %[[CASE1_POP:.*]] = "simt_struct.mask_pop"()
// CHECK:     %[[CASE1_MERGE:.*]] = "simt_struct.mask_merge"(%[[CASE1_MASK]])
// CHECK:     "simt_struct.cond_branch"(%{{.*}}, %[[CASE1_MERGE]], %[[CASE1_MERGE]], %[[CASE1_VAL]], %[[CASE1_HAS]], %[[CASE1_EXEC]], %[[CASE1_DONE]], %[[CASE1_VAL]], %[[CASE1_HAS]], %[[CASE1_EXEC]], %[[CASE1_DONE]])
// CHECK-SAME: {false_target = @block5
// CHECK-SAME: true_target = @block5}
// CHECK: }) {merge_target = @block5, sym_name = "block4"}
// CHECK: "simt_struct.block"() ({
// CHECK:   ^bb0(%[[EXIT_MASK:.*]]: i64, %[[EXIT_VAL:.*]]: i32, %[[EXIT_HAS:.*]]: i1, %[[EXIT_EXEC:.*]]: i1, %[[EXIT_DONE:.*]]: i1):
// CHECK:     %[[EXIT_POP:.*]] = "simt_struct.mask_pop"()
// CHECK:     %[[EXIT_MERGE:.*]] = "simt_struct.mask_merge"(%[[EXIT_MASK]])
// CHECK:     "simt_struct.branch"(%[[EXIT_MERGE]], %[[EXIT_VAL]], %[[EXIT_HAS]], %[[EXIT_EXEC]], %[[EXIT_DONE]]) {target = @block6}
// CHECK: }) {sym_name = "block5"}
// CHECK: "simt_struct.block"() ({
// CHECK:   ^bb0(%[[RET_MASK:.*]]: i64, %[[RET_VAL:.*]]: i32, %[[RET_HAS:.*]]: i1, %[[RET_EXEC:.*]]: i1, %[[RET_DONE:.*]]: i1):
// CHECK:     "simt_struct.return"() : () -> ()
// CHECK: }) {sym_name = "block6"}
