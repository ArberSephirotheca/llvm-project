module {
  func.func @main(%arg0: i32) attributes {simt.num_threads = array<i64: 1, 1, 1>} {
    %0 = "simt_step.dispatch_thread_id"() : () -> i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %1 = arith.andi %0, %c1_i32 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %2 = arith.cmpi ne, %1, %c0_i32_0 : i32
    %c2_i32 = arith.constant 2 : i32
    %3 = arith.andi %0, %c2_i32 : i32
    %c0_i32_1 = arith.constant 0 : i32
    %4 = arith.cmpi ne, %3, %c0_i32_1 : i32
    %5 = "simt_step.if"(%2) ({
      %6 = "simt_step.if"(%4) ({
        %c1_i32_2 = arith.constant 1 : i32
        "simt_step.yield"(%c1_i32_2) : (i32) -> ()
      }, {
        %c2_i32_2 = arith.constant 2 : i32
        "simt_step.yield"(%c2_i32_2) : (i32) -> ()
      }) : (i1) -> i32
      "simt_step.yield"(%6) : (i32) -> ()
    }, {
      %c0_i32_2 = arith.constant 0 : i32
      "simt_step.yield"(%c0_i32_2) : (i32) -> ()
    }) : (i1) -> i32
    return
  }
}
// RUN: %simt-opt --simt-step-to-structured %s | %mlir-file-check %s

//=== func anchor ===//
// CHECK-LABEL: func.func @nested_if

//=== entry (outer header) ===//
// CHECK-LABEL: "simt_struct.block"() ({
// CHECK:   ^bb0(
// CHECK:   %{{.*}} = arith.constant 0 : i32
// (mask ops noise ok)
// CHECK:   "simt_struct.mask_push"(
// CHECK:   "simt_struct.mask_pop"() : () -> i64
// outer cond_branch: (cond0, mask, mask, cond1, zero)
// CHECK:   "simt_struct.cond_branch"(
// CHECK-SAME: ) {{.*}} : (i1, i64, i64, i1, i32) -> ()
// CHECK: }) {merge_target = @block1, sym_name = "entry"} : () -> ()

//=== block1 (outer exit / merge) ===//
// CHECK-LABEL: "simt_struct.block"() ({
// CHECK:   ^bb0(%{{.*}}: i64, %{{.*}}: i32):
// CHECK:   "simt_struct.return"() : () -> ()
// CHECK: }) {sym_name = "block1"} : () -> ()

//=== block2 (outer else arm) ===//
// CHECK-LABEL: "simt_struct.block"() ({
// CHECK:   ^bb0(%{{.*}}: i64, %{{.*}}: i32):
// CHECK:   "simt_struct.branch"(%{{.*}}, %{{.*}}) {target = @block1} : (i64, i32) -> ()
// CHECK: }) {sym_name = "block2"} : () -> ()

//=== block3 (inner header) ===//
// CHECK-LABEL: "simt_struct.block"() ({
// CHECK:   ^bb0(%{{.*}}: i64, %{{.*}}: i1):
// (mask ops noise ok)
// CHECK:   "simt_struct.mask_push"(
// CHECK:   "simt_struct.mask_pop"() : () -> i64
// inner cond_branch: (cond1_local, mask, mask)
// CHECK:   "simt_struct.cond_branch"(
// CHECK-SAME: ) {{.*}} : (i1, i64, i64) -> ()
// CHECK: }) {merge_target = @block3.merge, sym_name = "block3"} : () -> ()

//=== block3.merge (inner merge) ===//
// CHECK-LABEL: "simt_struct.block"() ({
// CHECK:   ^bb0(%{{.*}}: i64, %{{.*}}: i32):
// CHECK:   "simt_struct.branch"(%{{.*}}, %{{.*}}) {target = @block1} : (i64, i32) -> ()
// CHECK: }) {sym_name = "block3.merge"} : () -> ()

//=== block4 (inner else arm) ===//
// CHECK-LABEL: "simt_struct.block"() ({
// CHECK:   ^bb0(%{{.*}}: i64):
// CHECK:   %{{.*}} = arith.constant 2 : i32
// CHECK:   "simt_struct.branch"(%{{.*}}, %{{.*}}) {target = @block3.merge} : (i64, i32) -> ()
// CHECK: }) {sym_name = "block4"} : () -> ()

//=== block7 (inner then arm) ===//
// CHECK-LABEL: "simt_struct.block"() ({
// CHECK:   ^bb0(%{{.*}}: i64):
// CHECK:   %{{.*}} = arith.constant 1 : i32
// CHECK:   "simt_struct.branch"(%{{.*}}, %{{.*}}) {target = @block3.merge} : (i64, i32) -> ()
// CHECK: }) {sym_name = "block7"} : () -> ()
