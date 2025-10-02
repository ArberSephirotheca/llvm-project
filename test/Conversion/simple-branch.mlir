builtin.module {
  func.func @branches(%arg0: i32) {
    %mask = "simt_step.active_mask"() : () -> i64
    %c1 = arith.constant 1 : i32
    %cmp = arith.cmpi eq, %arg0, %c1 : i32
    cf.cond_br %cmp, ^bb1(%c1 : i32), ^bb2(%arg0 : i32)
  ^bb1(%x: i32):
    cf.br ^bb3(%x : i32)
  ^bb2(%y: i32):
    cf.br ^bb3(%y : i32)
  ^bb3(%z: i32):
    func.return
  }
}

// RUN: %simt-opt --simt-step-to-structured %s | %mlir-file-check %s

// CHECK-LABEL: func.func @branches
// CHECK:     "simt_struct.block"() ({
// CHECK:       ^bb0(%[[ENTRY_ARG:.*]]: i32):
// CHECK:         %[[ENTRY_MASK:.*]] = "simt_step.active_mask"() : () -> i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : i32
// CHECK:         %[[CMP:.*]] = arith.cmpi eq, %[[ENTRY_ARG]], %[[C1]] : i32
// CHECK:         "simt_struct.cond_branch"(%[[CMP]], %[[ENTRY_MASK]], %[[ENTRY_MASK]], %[[C1]], %[[ENTRY_ARG]])
// CHECK-SAME: {false_target = @block2,
// CHECK-SAME: operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>,
// CHECK-SAME: true_target = @block1}
// CHECK:     }) {sym_name = "entry"}
// CHECK:     "simt_struct.block"() ({
// CHECK:       ^bb0(%[[BLOCK1_ARG:.*]]: i32):
// CHECK:         %[[BLOCK1_MASK:.*]] = "simt_step.active_mask"() : () -> i64
// CHECK:         "simt_struct.branch"(%[[BLOCK1_MASK]], %[[BLOCK1_ARG]]) {target = @block3}
// CHECK:     }) {sym_name = "block1"}
// CHECK:     "simt_struct.block"() ({
// CHECK:       ^bb0(%[[BLOCK2_ARG:.*]]: i32):
// CHECK:         %[[BLOCK2_MASK:.*]] = "simt_step.active_mask"() : () -> i64
// CHECK:         "simt_struct.branch"(%[[BLOCK2_MASK]], %[[BLOCK2_ARG]]) {target = @block3}
// CHECK:     }) {sym_name = "block2"}
// CHECK:     "simt_struct.block"() ({
// CHECK:       ^bb0(%[[BLOCK3_ARG:.*]]: i32):
// CHECK:         "simt_struct.return"() : () -> ()
// CHECK:     }) {sym_name = "block3"}
