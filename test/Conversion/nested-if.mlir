builtin.module {
  func.func @nested_if(%cond0: i1, %cond1: i1) {
    %zero = arith.constant 0 : i32
    %outer:1 = "simt_step.if"(%cond0) ({
    ^bb0:
      %inner:1 = "simt_step.if"(%cond1) ({
      ^bb0:
        %one = arith.constant 1 : i32
        "simt_step.yield"(%one) : (i32) -> ()
      }, {
      ^bb0:
        %two = arith.constant 2 : i32
        "simt_step.yield"(%two) : (i32) -> ()
      }) : (i1) -> (i32)
      "simt_step.yield"(%inner#0) : (i32) -> ()
    }, {
    ^bb0:
      "simt_step.yield"(%zero) : (i32) -> ()
    }) : (i1) -> (i32)
    func.return
  }
}

// RUN: %simt-opt --simt-step-to-structured %s | %mlir-file-check %s

// CHECK-LABEL: func.func @nested_if
// CHECK: "simt_struct.block"() ({
// CHECK:   ^bb0(%[[MASK:.*]]: i64, %[[COND0:.*]]: i1, %[[COND1:.*]]: i1):
// CHECK:     "simt_struct.cond_branch"(%[[COND0]], %[[MASK]], %[[MASK]], %[[COND0]], %[[COND1]]) {false_target = @[[ELSE:.*]], true_target = @[[THEN:.*]]}
// CHECK: }) {sym_name = "entry"}
// CHECK: "simt_struct.block"() ({
// CHECK:   ^bb0(%[[THEN_MASK:.*]]: i64, %[[THEN_COND0:.*]]: i1, %[[THEN_COND1:.*]]: i1):
// CHECK:     "simt_struct.cond_branch"(%[[THEN_COND1]], %[[THEN_MASK]], %[[THEN_MASK]], %{{.*}}, %{{.*}}) {false_target = @[[INNER_ELSE:.*]], true_target = @[[INNER_THEN:.*]]}
// CHECK: }) {merge_target = @[[MERGE:.*]], sym_name = "[[THEN]]"}
// CHECK: "simt_struct.block"() ({
// CHECK:   ^bb0(%[[INNER_THEN_MASK:.*]]: i64, %{{.*}}: i1, %{{.*}}: i1):
// CHECK:     %[[ONE:.*]] = arith.constant 1 : i32
// CHECK:     "simt_struct.branch"(%[[INNER_THEN_MASK]], %[[ONE]]) {target = @[[MERGE]]}
// CHECK: }) {sym_name = "[[INNER_THEN]]"}
// CHECK: "simt_struct.block"() ({
// CHECK:   ^bb0(%[[INNER_ELSE_MASK:.*]]: i64, %{{.*}}: i1, %{{.*}}: i1):
// CHECK:     %[[TWO:.*]] = arith.constant 2 : i32
// CHECK:     "simt_struct.branch"(%[[INNER_ELSE_MASK]], %[[TWO]]) {target = @[[MERGE]]}
// CHECK: }) {sym_name = "[[INNER_ELSE]]"}
// CHECK: "simt_struct.block"() ({
// CHECK:   ^bb0(%[[MERGE_MASK:.*]]: i64, %{{.*}}: i1, %{{.*}}: i1, %[[MERGED_VAL:.*]]: i32):
// CHECK:     "simt_struct.branch"(%[[MERGE_MASK]], %[[MERGED_VAL]]) {target = @[[EXIT:.*]]}
// CHECK: }) {sym_name = "[[MERGE]]"}
// CHECK: "simt_struct.block"() ({
// CHECK:   ^bb0(%[[ELSE_MASK:.*]]: i64, %{{.*}}: i1, %{{.*}}: i1):
// CHECK:     %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK:     "simt_struct.branch"(%[[ELSE_MASK]], %[[ZERO]]) {target = @[[EXIT]]}
// CHECK: }) {sym_name = "[[ELSE]]"}
// CHECK: "simt_struct.block"() ({
// CHECK:   ^bb0(%[[EXIT_MASK:.*]]: i64, %{{.*}}: i1, %{{.*}}: i1, %[[EXIT_VAL:.*]]: i32):
// CHECK:     "simt_struct.return"() : () -> ()
// CHECK: }) {sym_name = "[[EXIT]]"}
