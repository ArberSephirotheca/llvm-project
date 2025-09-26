// CHECK-LABEL: func.func @main(%arg0: i32) attributes {simt.num_threads = array<i64: 1, 1, 1>}
// CHECK: simt_step.break
// CHECK: return
