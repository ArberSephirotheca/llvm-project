builtin.module {
  func.func @switch_use(%sel: i32) {
    %zero = arith.constant 0 : i32
    %false = arith.constant false
    %result:4 = "simt_step.switch"(%sel, %zero, %false, %false, %false) ({
    ^bb0(%case0_val: i32, %case0_has: i1, %case0_exec: i1, %case0_done: i1):
      %case_zero = arith.constant 0 : i32
      %inc0 = arith.addi %case0_val, %case_zero : i32
      "simt_step.yield"(%inc0, %case0_has, %case0_exec, %case0_done)
        : (i32, i1, i1, i1) -> ()
    ^bb1(%case1_val: i32, %case1_has: i1, %case1_exec: i1, %case1_done: i1):
      %inc1 = arith.addi %case1_val, %case1_val : i32
      "simt_step.yield"(%inc1, %case1_has, %case1_exec, %case1_done)
        : (i32, i1, i1, i1) -> ()
    }) {case_values = array<i64: 0, 1>} : (i32, i32, i1, i1, i1) -> (i32, i1, i1, i1)
    %after_cst = arith.constant 4 : i32
    %use = arith.addi %result#0, %after_cst : i32
    %shuf = arith.addi %use, %result#0 : i32
    func.return
  }
}

// RUN: %simt-opt --simt-step-to-structured %s | %mlir-file-check %s

// CHECK-LABEL: func.func @switch_use
// CHECK: "simt_struct.block"() ({
// CHECK:   "simt_struct.branch"(%[[ENTRY_MASK:.*]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {target = @block1}
// CHECK: }) {sym_name = "entry"}
// CHECK: "simt_struct.block"() ({
// CHECK:   ^bb0(%[[CASE0_MASK:.*]]: i64, %[[CASE0_SEL:.*]]: i32, %[[CASE0_VAL:.*]]: i32,
// CHECK-SAME: %[[CASE0_HAS:.*]]: i1, %[[CASE0_EXEC:.*]]: i1, %[[CASE0_DONE:.*]]: i1)
// CHECK:     %[[POP0:.*]] = "simt_struct.mask_pop"()
// CHECK:     %[[MERGE0:.*]] = "simt_struct.mask_merge"(%[[CASE0_MASK]])
// CHECK:     %[[INC0:.*]] = arith.addi %[[CASE0_SEL]], %{{.*}} : i32
// CHECK:     "simt_struct.cond_branch"(%{{.*}}, %[[MERGE0]], %[[MERGE0]], %[[CASE0_SEL]], %[[INC0]], %[[CASE0_VAL]], %[[CASE0_HAS]], %[[CASE0_EXEC]],
// CHECK-SAME: %[[CASE0_SEL]], %[[INC0]], %[[CASE0_VAL]], %[[CASE0_HAS]], %[[CASE0_EXEC]]) {false_target = @block5
// CHECK: }) {merge_target = @block5, sym_name = "block2"}
// CHECK: "simt_struct.block"() ({
// CHECK:   ^bb0(%[[EXIT_MASK:.*]]: i64, %[[EXIT_SEL:.*]]: i32, %[[EXIT_VAL:.*]]: i32,
// CHECK-SAME: %[[EXIT_HAS:.*]]: i1, %[[EXIT_EXEC:.*]]: i1, %[[EXIT_DONE:.*]]: i1)
// CHECK:     %[[POP_EXIT:.*]] = "simt_struct.mask_pop"()
// CHECK:     %[[MERGE_EXIT:.*]] = "simt_struct.mask_merge"(%[[EXIT_MASK]])
// CHECK:     "simt_struct.branch"(%[[MERGE_EXIT]], %[[EXIT_SEL]], %[[EXIT_VAL]], %[[EXIT_HAS]], %[[EXIT_EXEC]], %[[EXIT_DONE]]) {target = @block6}
// CHECK: }) {sym_name = "block5"}
// CHECK: "simt_struct.block"() ({
// CHECK:   ^bb0(%[[RET_MASK:.*]]: i64, %[[RET_SEL:.*]]: i32, %[[RET_VAL:.*]]: i32,
// CHECK-SAME: %[[RET_HAS:.*]]: i1, %[[RET_EXEC:.*]]: i1, %[[RET_DONE:.*]]: i1)
// CHECK:     %[[C4:.*]] = arith.constant 4 : i32
// CHECK:     %[[USE:.*]] = arith.addi %[[RET_VAL]], %[[C4]] : i32
// CHECK:     %[[SHUF:.*]] = arith.addi %[[USE]], %[[RET_VAL]] : i32
// CHECK:     "simt_struct.return"() : () -> ()
// CHECK: }) {sym_name = "block6"}
