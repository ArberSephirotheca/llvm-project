module {
  func.func @main(%arg0: i32) attributes {simt.num_threads = array<i64: 1, 1, 1>} {
    %0 = simt_step.dispatch_thread_id : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %1:2 = "simt_step.loop"(%c0_i32, %c0_i32_0) ({
    ^bb0(%arg1: i32, %arg2: i32):
      %c3_i32 = arith.constant 3 : i32
      %2 = arith.cmpi slt, %arg2, %c3_i32 : i32
      "simt_step.condition"(%2, %arg1, %arg2) : (i1, i32, i32) -> ()
    }, {
    ^bb0(%arg1: i32, %arg2: i32):
      %2 = arith.addi %arg1, %arg2 : i32
      %c1_i32 = arith.constant 1 : i32
      %3 = arith.addi %arg2, %c1_i32 : i32
      "simt_step.yield"(%2, %3) : (i32, i32) -> ()
    }) : (i32, i32) -> (i32, i32)
    func.return
  }
}

module {
  func.func @main(%arg0: i32) attributes {simt.num_threads = array<i64: 1, 1, 1>} {
    "simt_struct.block"() ({
    ^bb0(%arg1: i64, %arg2: i32):
      %0 = simt_step.dispatch_thread_id : i32
      %c0_i32 = arith.constant 0 : i32
      %c0_i32_0 = arith.constant 0 : i32
      "simt_struct.branch"(%arg1, %c0_i32, %c0_i32_0) {target = @block4} : (i64, i32, i32) -> ()
    }) {sym_name = "entry"} : () -> ()

    "simt_struct.block"() ({
    ^bb0(%arg1: i64, %arg2: i32, %arg3: i32):
      "simt_struct.return"() : () -> ()
    }) {sym_name = "block1"} : () -> ()

    "simt_struct.block"() ({
    ^bb0(%arg1: i64, %arg2: i32, %arg3: i32):
      %0 = arith.addi %arg2, %arg3 : i32
      %c1_i32 = arith.constant 1 : i32
      %1 = arith.addi %arg3, %c1_i32 : i32
      "simt_struct.branch"(%arg1, %0, %1) {target = @block4} : (i64, i32, i32) -> ()
    }) {sym_name = "block2"} : () -> ()
    
    "simt_struct.block"() ({
    ^bb0(%arg1: i64, %arg2: i32, %arg3: i32):
      %c3_i32 = arith.constant 3 : i32
      %0 = arith.cmpi slt, %arg3, %c3_i32 : i32
      %1 = "simt_struct.mask_not"(%0) : (i1) -> i1
      "simt_struct.cond_branch"(%0, %0, %1, %arg2, %arg3, %arg2, %arg3) {false_target = @block1, operandSegmentSizes = array<i32: 1, 1, 1, 2, 2>, true_target = @block2} : (i1, i1, i1, i32, i32, i32, i32) -> ()
    }) {continue_target = @block4, merge_target = @block1, sym_name = "block4"} : () -> ()
    func.return
  }
}


// RUN: %simt-opt --simt-step-to-structured %s | %mlir-file-check %s

// CHECK-LABEL: func.func @loop
// CHECK: "simt_struct.block"() ({
// CHECK:   ^bb0(%[[ENTRY_MASK:.*]]: i64, %[[INIT:.*]]: i32):
// CHECK:     %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK:     "simt_struct.branch"(%[[ENTRY_MASK]], %[[INIT]], %[[ZERO]]) {target = @[[PREP:block[0-9]+]]}
// CHECK: }) {sym_name = "entry"}
// CHECK: "simt_struct.block"() ({
// CHECK:   ^bb0(%[[PREP_MASK:.*]]: i64, %[[ITER:.*]]: i32, %[[ACC:.*]]: i32):
// CHECK:     %[[LIMIT:.*]] = arith.constant 4 : i32
// CHECK:     %[[CMP:.*]] = arith.cmpi slt, %[[ITER]], %[[LIMIT]] : i32
// CHECK:     %[[NOTCMP:.*]] = "simt_struct.mask_not"(%[[CMP]]) : (i1) -> i1
// CHECK:     "simt_struct.cond_branch"(%[[CMP]], %[[CMP]], %[[NOTCMP]], %[[ITER]], %[[ACC]], %[[ITER]], %[[ACC]]) {false_target = @[[MERGE:block[0-9]+]], true_target = @[[BODY:block[0-9]+]]}
// CHECK: }) {sym_name = "[[PREP]]"}
// CHECK: "simt_struct.block"() ({
// CHECK:   ^bb0(%[[BODY_MASK:.*]]: i64, %[[BODY_ITER:.*]]: i32, %[[BODY_ACC:.*]]: i32):
// CHECK:     %[[ONE:.*]] = arith.constant 1 : i32
// CHECK:     %[[NEW_ACC:.*]] = arith.addi %[[BODY_ACC]], %[[BODY_ITER]] : i32
// CHECK:     %[[NEXT:.*]] = arith.addi %[[BODY_ITER]], %[[ONE]] : i32
// CHECK:     "simt_struct.branch"(%[[BODY_MASK]], %[[NEXT]], %[[NEW_ACC]]) {target = @[[PREP]]}
// CHECK: }) {sym_name = "[[BODY]]"}
// CHECK: "simt_struct.block"() ({
// CHECK:   ^bb0(%[[MERGE_MASK:.*]]: i64, %[[MERGE_ITER:.*]]: i32, %[[MERGE_ACC:.*]]: i32):
// CHECK:     "simt_struct.return"() : () -> ()
// CHECK: }) {sym_name = "[[MERGE]]"}
