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
// CHECK: simt_struct.block @entry
// CHECK:   %[[MASK:.*]] = "simt_step.active_mask"() : () -> i64
// CHECK:   simt_struct.cond_branch %{{.*}}, %[[MASK]], %[[MASK]]
// CHECK: simt_struct.block @block1
// CHECK:   simt_struct.branch %[[MASK]]
// CHECK: simt_struct.block @block2
// CHECK:   simt_struct.branch %[[MASK]]
// CHECK: simt_struct.block @block3
// CHECK:   simt_struct.return
