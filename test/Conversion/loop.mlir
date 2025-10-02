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
// CHECK:   "simt_struct.branch"
// CHECK: "simt_struct.block"() ({
// CHECK:   "simt_struct.mask_push"(%{{.*}}) {continue_target = @block1, merge_target = @block3}
// CHECK:   %{{.*}} = arith.cmpi
// CHECK:   "simt_struct.mask_pop"
// CHECK:   "simt_struct.cond_branch"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {false_target = @block3, operandSegmentSizes = array<i32: 1, 1, 1, 2, 2>, true_target = @block2}
// CHECK: "simt_struct.block"() ({
// CHECK:   "simt_struct.mask_push"(%{{.*}}) {continue_target = @block1}
// CHECK:   %{{.*}} = arith.addi
// CHECK:   "simt_struct.mask_pop"
// CHECK:   "simt_struct.branch"(%{{.*}}, %{{.*}}, %{{.*}}) {target = @block1}
// CHECK: "simt_struct.block"() ({
// CHECK:   "simt_struct.return"
