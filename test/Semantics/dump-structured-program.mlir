builtin.module {
  func.func @kernel(%arg0: i32) {
    %mask = simt_step.active_mask : i64
    cf.br ^bb1
  ^bb1:
    func.return
  }
}

// RUN: %simt-opt --simt-step-to-structured --simt-dump-structured-program %s | %mlir-file-check %s

// CHECK: entry: entry
// CHECK: block entry args=0
// CHECK: block block1 args=0
