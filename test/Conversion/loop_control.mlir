builtin.module {
  func.func @loop_break(%input: i32) {
    %zero = arith.constant 0 : i32
    %loop:2 = "simt_step.loop"(%zero, %input) (
      {
      ^bb0(%iter: i32, %value: i32):
        %limit = arith.constant 1 : i32
        %cond = arith.cmpi slt, %iter, %limit : i32
        "simt_step.condition"(%cond, %iter, %value) : (i1, i32, i32) -> ()
      },
      {
      ^bb0(%iter_body: i32, %val_body: i32):
        "simt_step.break"(%iter_body, %val_body) : (i32, i32) -> ()
      }
    ) : (i32, i32) -> (i32, i32)
    func.return
  }
}

// RUN: %simt-opt --simt-step-to-structured %s | %mlir-file-check %s

// CHECK-LABEL: func.func @loop_break
// CHECK: "simt_struct.block"() ({
// CHECK:   ^bb0(%[[ENTRY_MASK:.*]]: i64, %[[ENTRY_VAL:.*]]: i32):
// CHECK:     %[[ENTRY_ZERO:.*]] = arith.constant 0 : i32
// CHECK:     "simt_struct.branch"(%[[ENTRY_MASK]], %[[ENTRY_ZERO]], %[[ENTRY_VAL]]) {target = @[[HEADER:[0-9A-Za-z_\.]+]]}
// CHECK: "simt_struct.block"() ({
// CHECK:   ^bb0(%[[MERGE_MASK:.*]]: i64, %[[MERGE_ITER:.*]]: i32, %[[MERGE_VAL:.*]]: i32):
// CHECK:     "simt_struct.return"() : () -> ()
// CHECK: "simt_struct.block"() ({
// CHECK:   ^bb0(%[[BODY_MASK:.*]]: i64, %[[BODY_ITER:.*]]: i32, %[[BODY_VAL:.*]]: i32):
// CHECK:     "simt_struct.branch"(%[[BODY_MASK]], %[[BODY_ITER]], %[[BODY_VAL]]) {target = @[[MERGESYM:[0-9A-Za-z_\.]+]]}
// CHECK: "simt_struct.block"() ({
// CHECK:   ^bb0(%[[HDR_MASK:.*]]: i64, %[[HDR_ITER:.*]]: i32, %[[HDR_VAL:.*]]: i32):
// CHECK:     "simt_struct.mask_push"(%{{.*}}) {continue_target = @[[HEADER]], merge_target = @[[MERGESYM]]}
// CHECK:     "simt_struct.cond_branch"(%{{.*}}) {false_target = @[[MERGESYM]], {{.*}}true_target = @[[BODY:[0-9A-Za-z_\.]+]]}
