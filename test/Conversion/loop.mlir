builtin.module {
  func.func @loop(%init: i32) {
    %zero = arith.constant 0 : i32
    %sum:2 = "simt_step.loop"(%init, %zero) (
      {
      ^bb0(%iter: i32, %acc: i32):
        %limit = arith.constant 4 : i32
        %cmp = arith.cmpi slt, %iter, %limit : i32
        "simt_step.condition"(%cmp, %iter, %acc) : (i1, i32, i32) -> ()
      },
      {
      ^bb0(%iter: i32, %acc: i32):
        %one = arith.constant 1 : i32
        %new = arith.addi %acc, %iter : i32
        %next = arith.addi %iter, %one : i32
        "simt_step.yield"(%next, %new) : (i32, i32) -> ()
      }
    ) : (i32, i32) -> (i32, i32)
    func.return
  }
}

// RUN: %simt-opt --simt-step-to-structured %s | %mlir-file-check %s

// CHECK-LABEL: func.func @loop
// CHECK: "simt_struct.block"() ({
// CHECK:   ^bb0(%[[ENTRY_MASK:.*]]: i64, %[[ENTRY_VAL:.*]]: i32):
// CHECK:     "simt_struct.branch"(%[[ENTRY_MASK]], %[[ENTRY_VAL]], %{{.*}}) {target = @block1}
// CHECK: }) {sym_name = "entry"}
// CHECK: "simt_struct.block"() ({
// CHECK:   ^bb0(%[[HDR_MASK:.*]]: i64, %[[ITER:.*]]: i32, %[[ACC:.*]]: i32):
// CHECK:     %[[POP_HDR:.*]] = "simt_struct.mask_pop"() : () -> i64
// CHECK:     %[[MERGE_HDR:.*]] = "simt_struct.mask_merge"(%[[HDR_MASK]]) : (i64) -> i64
// CHECK:     "simt_struct.mask_push"(%[[MERGE_HDR]]) {continue_target = @block1, merge_target = @block3}
// CHECK:     %[[CMP:.*]] = arith.cmpi
// CHECK:     "simt_struct.cond_branch"(%[[CMP]], %[[MERGE_HDR]], %[[MERGE_HDR]], %[[ITER]], %[[ACC]], %[[ITER]], %[[ACC]]) {false_target = @block3
// CHECK-SAME: operandSegmentSizes = array<i32: 1, 1, 1, 2, 2>, true_target = @block2}
// CHECK: }) {continue_target = @block1, merge_target = @block3, sym_name = "block1"}
// CHECK: "simt_struct.block"() ({
// CHECK:   ^bb0(%[[BODY_MASK:.*]]: i64, %[[BODY_ITER:.*]]: i32, %[[BODY_ACC:.*]]: i32):
// CHECK:     %[[POP_BODY:.*]] = "simt_struct.mask_pop"() : () -> i64
// CHECK:     %[[MERGED_BODY:.*]] = "simt_struct.mask_merge"(%[[BODY_MASK]]) : (i64) -> i64
// CHECK:     "simt_struct.mask_push"(%[[MERGED_BODY]]) {continue_target = @block1}
// CHECK:     %[[NEW_ACC:.*]] = arith.addi %[[BODY_ACC]], %[[BODY_ITER]] : i32
// CHECK:     %[[NEXT:.*]] = arith.addi %[[BODY_ITER]], %{{.*}} : i32
// CHECK:     "simt_struct.branch"(%[[MERGED_BODY]], %[[NEXT]], %[[NEW_ACC]]) {target = @block1}
// CHECK: }) {continue_target = @block1, sym_name = "block2"}
// CHECK: "simt_struct.block"() ({
// CHECK:   ^bb0(%[[EXIT_MASK:.*]]: i64, %[[EXIT_ITER:.*]]: i32, %[[EXIT_ACC:.*]]: i32):
// CHECK:     %[[POP_EXIT:.*]] = "simt_struct.mask_pop"() : () -> i64
// CHECK:     %[[MERGE_EXIT:.*]] = "simt_struct.mask_merge"(%[[EXIT_MASK]]) : (i64) -> i64
// CHECK:     "simt_struct.return"() : () -> ()
// CHECK: }) {sym_name = "block3"}
