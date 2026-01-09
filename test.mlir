module {
  func.func @helper0(%arg0: i32) -> i32 {
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.addi %arg0, %c1_i32 : i32
    %c4_i32 = arith.constant 4 : i32
    %1 = arith.remsi %0, %c4_i32 : i32
    %c0_i32 = arith.constant 0 : i32
    %2 = arith.addi %1, %c0_i32 : i32
    return %2 : i32
  }
  func.func @main(%arg0: !simt_step.resource<Global, i32>) attributes {simt.num_threads = [4, 1, 1]} {
    %0 = "simt_step.dispatch_thread_id"() : () -> i32
    %1 = call @helper0(%0) : (i32) -> i32
    %c128_i32 = arith.constant 128 : i32
    %2 = arith.addi %c128_i32, %0 : i32
    "simt_step.buffer.store"(%arg0, %2, %1) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
    %c2_i32 = arith.constant 2 : i32
    %3 = arith.remsi %0, %c2_i32 : i32
    %4 = "simt_step.switch"(%3, %0) ({
    ^bb0(%arg1: i32):
      %c0_i32_1 = arith.constant 0 : i32
      %c0_i32_2 = arith.constant 0 : i32
      %10:2 = "simt_step.loop"(%c0_i32_1, %c0_i32_2) ({
      ^bb0(%arg2: i32, %arg3: i32):
        %c4_i32 = arith.constant 4 : i32
        %20 = arith.remsi %0, %c4_i32 : i32
        %c1_i32 = arith.constant 1 : i32
        %21 = arith.addi %20, %c1_i32 : i32
        %22 = arith.cmpi slt, %arg3, %21 : i32
        "simt_step.condition"(%22, %arg2, %arg3) : (i1, i32, i32) -> ()
      }, {
      ^bb0(%arg2: i32, %arg3: i32):
        %c3_i32_11 = arith.constant 3 : i32
        %20 = arith.cmpi slt, %0, %c3_i32_11 : i32
        %21 = "simt_step.if"(%20) ({
          "simt_step.yield"(%0) : (i32) -> ()
        }, {
          %c2_i32_16 = arith.constant 2 : i32
          %32 = arith.addi %0, %c2_i32_16 : i32
          "simt_step.yield"(%32) : (i32) -> ()
        }) : (i1) -> i32
        %c0_i32_12 = arith.constant 0 : i32
        %22 = arith.addi %c0_i32_12, %0 : i32
        %true_13 = arith.constant true
        %23 = "simt_step.wave_count_bits"(%true_13) : (i1) -> i32
        "simt_step.buffer.store"(%arg0, %22, %23) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
        %24 = arith.addi %arg2, %21 : i32
        %c1_i32 = arith.constant 1 : i32
        %25 = arith.addi %arg3, %c1_i32 : i32
        %true_14 = arith.constant true
        %c16_i32 = arith.constant 16 : i32
        %26 = arith.addi %arg3, %arg3 : i32
        %27 = arith.addi %26, %arg3 : i32
        %28 = arith.addi %27, %arg3 : i32
        %29 = arith.addi %c16_i32, %28 : i32
        %30 = arith.addi %29, %0 : i32
        %true_15 = arith.constant true
        %31 = "simt_step.wave_count_bits"(%true_15) : (i1) -> i32
        "simt_step.buffer.store"(%arg0, %30, %31) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
        "simt_step.continue"(%24, %25) : (i32, i32) -> ()
      }) : (i32, i32) -> (i32, i32)
      "simt_step.yield"(%10#0) {fallthrough = false} : (i32) -> ()
    ^bb1(%11: i32):  // no predecessors
      %c0_i32_3 = arith.constant 0 : i32
      %c0_i32_4 = arith.constant 0 : i32
      %12:2 = "simt_step.loop"(%c0_i32_3, %c0_i32_4) ({
      ^bb0(%arg2: i32, %arg3: i32):
        %c2_i32_11 = arith.constant 2 : i32
        %20 = arith.remsi %0, %c2_i32_11 : i32
        %c1_i32 = arith.constant 1 : i32
        %21 = arith.addi %20, %c1_i32 : i32
        %22 = arith.cmpi slt, %arg3, %21 : i32
        "simt_step.condition"(%22, %arg2, %arg3) : (i1, i32, i32) -> ()
      }, {
      ^bb0(%arg2: i32, %arg3: i32):
        %c2_i32_11 = arith.constant 2 : i32
        %20 = arith.remsi %0, %c2_i32_11 : i32
        %c3_i32_12 = arith.constant 3 : i32
        %21 = arith.addi %0, %c3_i32_12 : i32
        %22 = "simt_step.switch"(%20, %21) ({
        ^bb0(%arg4: i32):
          %c0_i32_15 = arith.constant 0 : i32
          %31 = arith.addi %0, %c0_i32_15 : i32
          "simt_step.yield"(%31) {fallthrough = false} : (i32) -> ()
        ^bb1(%32: i32):  // no predecessors
          %c3_i32_16 = arith.constant 3 : i32
          %33 = arith.addi %0, %c3_i32_16 : i32
          %true_17 = arith.constant true
          %c32_i32 = arith.constant 32 : i32
          %34 = arith.addi %c32_i32, %0 : i32
          %true_18 = arith.constant true
          %35 = "simt_step.wave_count_bits"(%true_18) : (i1) -> i32
          "simt_step.buffer.store"(%arg0, %34, %35) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
          "simt_step.yield"(%33) {fallthrough = false} : (i32) -> ()
        }) {case_values = array<i64: 0>, default_index = 1 : i64} : (i32, i32) -> i32
        %23 = arith.addi %arg2, %22 : i32
        %c1_i32 = arith.constant 1 : i32
        %24 = arith.addi %arg3, %c1_i32 : i32
        %true_13 = arith.constant true
        %c48_i32 = arith.constant 48 : i32
        %25 = arith.addi %arg3, %arg3 : i32
        %26 = arith.addi %25, %arg3 : i32
        %27 = arith.addi %26, %arg3 : i32
        %28 = arith.addi %c48_i32, %27 : i32
        %29 = arith.addi %28, %0 : i32
        %true_14 = arith.constant true
        %30 = "simt_step.wave_count_bits"(%true_14) : (i1) -> i32
        "simt_step.buffer.store"(%arg0, %29, %30) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
        "simt_step.yield"(%23, %24) : (i32, i32) -> ()
      }) : (i32, i32) -> (i32, i32)
      %true = arith.constant true
      %c64_i32 = arith.constant 64 : i32
      %13 = arith.addi %c64_i32, %0 : i32
      %true_5 = arith.constant true
      %14 = "simt_step.wave_count_bits"(%true_5) : (i1) -> i32
      "simt_step.buffer.store"(%arg0, %13, %14) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
      "simt_step.yield"(%12#0) {fallthrough = false} : (i32) -> ()
    ^bb2(%15: i32):  // no predecessors
      %c3_i32_6 = arith.constant 3 : i32
      %16 = arith.remsi %0, %c3_i32_6 : i32
      %c2_i32_7 = arith.constant 2 : i32
      %17 = "simt_step.switch"(%16, %c2_i32_7) ({
      ^bb0(%arg2: i32):
        %c0_i32_11 = arith.constant 0 : i32
        %c0_i32_12 = arith.constant 0 : i32
        %20:2 = "simt_step.loop"(%c0_i32_11, %c0_i32_12) ({
        ^bb0(%arg3: i32, %arg4: i32):
          %c3_i32_17 = arith.constant 3 : i32
          %26 = arith.remsi %0, %c3_i32_17 : i32
          %c1_i32 = arith.constant 1 : i32
          %27 = arith.addi %26, %c1_i32 : i32
          %28 = arith.cmpi slt, %arg4, %27 : i32
          "simt_step.condition"(%28, %arg3, %arg4) : (i1, i32, i32) -> ()
        }, {
        ^bb0(%arg3: i32, %arg4: i32):
          %26 = arith.addi %arg3, %arg4 : i32
          %c1_i32 = arith.constant 1 : i32
          %27 = arith.addi %arg4, %c1_i32 : i32
          %true_17 = arith.constant true
          %c80_i32 = arith.constant 80 : i32
          %28 = arith.addi %arg4, %arg4 : i32
          %29 = arith.addi %28, %arg4 : i32
          %30 = arith.addi %29, %arg4 : i32
          %31 = arith.addi %c80_i32, %30 : i32
          %32 = arith.addi %31, %0 : i32
          %true_18 = arith.constant true
          %33 = "simt_step.wave_count_bits"(%true_18) : (i1) -> i32
          "simt_step.buffer.store"(%arg0, %32, %33) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
          "simt_step.break"(%26, %27) : (i32, i32) -> ()
        }) : (i32, i32) -> (i32, i32)
        "simt_step.yield"(%20#0) {fallthrough = true} : (i32) -> ()
      ^bb1(%21: i32):  // no predecessors
        %c0_i32_13 = arith.constant 0 : i32
        %c0_i32_14 = arith.constant 0 : i32
        %22:2 = "simt_step.loop"(%c0_i32_13, %c0_i32_14) ({
        ^bb0(%arg3: i32, %arg4: i32):
          %c3_i32_17 = arith.constant 3 : i32
          %26 = arith.remsi %0, %c3_i32_17 : i32
          %c1_i32 = arith.constant 1 : i32
          %27 = arith.addi %26, %c1_i32 : i32
          %28 = arith.cmpi slt, %arg4, %27 : i32
          "simt_step.condition"(%28, %arg3, %arg4) : (i1, i32, i32) -> ()
        }, {
        ^bb0(%arg3: i32, %arg4: i32):
          %26 = arith.addi %arg3, %arg4 : i32
          %c1_i32 = arith.constant 1 : i32
          %27 = arith.addi %arg4, %c1_i32 : i32
          %true_17 = arith.constant true
          %c96_i32 = arith.constant 96 : i32
          %28 = arith.addi %arg4, %arg4 : i32
          %29 = arith.addi %28, %arg4 : i32
          %30 = arith.addi %29, %arg4 : i32
          %31 = arith.addi %c96_i32, %30 : i32
          %32 = arith.addi %31, %0 : i32
          %true_18 = arith.constant true
          %33 = "simt_step.wave_count_bits"(%true_18) : (i1) -> i32
          "simt_step.buffer.store"(%arg0, %32, %33) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
          "simt_step.yield"(%26, %27) : (i32, i32) -> ()
        }) : (i32, i32) -> (i32, i32)
        "simt_step.yield"(%22#0) {fallthrough = true} : (i32) -> ()
      ^bb2(%23: i32):  // no predecessors
        %true_15 = arith.constant true
        %c112_i32 = arith.constant 112 : i32
        %24 = arith.addi %c112_i32, %0 : i32
        %true_16 = arith.constant true
        %25 = "simt_step.wave_count_bits"(%true_16) : (i1) -> i32
        "simt_step.buffer.store"(%arg0, %24, %25) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
        "simt_step.yield"(%0) {fallthrough = false} : (i32) -> ()
      }) {case_values = array<i64: 0, 1>, default_index = 2 : i64} : (i32, i32) -> i32
      %true_8 = arith.constant true
      %c128_i32_9 = arith.constant 128 : i32
      %18 = arith.addi %c128_i32_9, %0 : i32
      %true_10 = arith.constant true
      %19 = "simt_step.wave_count_bits"(%true_10) : (i1) -> i32
      "simt_step.buffer.store"(%arg0, %18, %19) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
      "simt_step.yield"(%17) {fallthrough = false} : (i32) -> ()
    }) {case_values = array<i64: 0, 1>, default_index = 2 : i64} : (i32, i32) -> i32
    %c3_i32 = arith.constant 3 : i32
    %5 = arith.remsi %0, %c3_i32 : i32
    %6 = "simt_step.switch"(%5, %0) ({
    ^bb0(%arg1: i32):
      %c3_i32_1 = arith.constant 3 : i32
      "simt_step.yield"(%c3_i32_1) {fallthrough = false} : (i32) -> ()
    ^bb1(%10: i32):  // no predecessors
      %c0_i32_2 = arith.constant 0 : i32
      %c0_i32_3 = arith.constant 0 : i32
      %11:2 = "simt_step.loop"(%c0_i32_2, %c0_i32_3) ({
      ^bb0(%arg2: i32, %arg3: i32):
        %c3_i32_7 = arith.constant 3 : i32
        %22 = arith.remsi %0, %c3_i32_7 : i32
        %c1_i32_8 = arith.constant 1 : i32
        %23 = arith.addi %22, %c1_i32_8 : i32
        %24 = arith.cmpi slt, %arg3, %23 : i32
        "simt_step.condition"(%24, %arg2, %arg3) : (i1, i32, i32) -> ()
      }, {
      ^bb0(%arg2: i32, %arg3: i32):
        %c2_i32_7 = arith.constant 2 : i32
        %22 = arith.remsi %0, %c2_i32_7 : i32
        %c0_i32_8 = arith.constant 0 : i32
        %23 = arith.cmpi eq, %22, %c0_i32_8 : i32
        %24 = "simt_step.if"(%23) ({
          %c4_i32 = arith.constant 4 : i32
          %35 = arith.addi %0, %c4_i32 : i32
          "simt_step.yield"(%35) : (i32) -> ()
        }, {
          "simt_step.yield"(%0) : (i32) -> ()
        }) : (i1) -> i32
        %c144_i32 = arith.constant 144 : i32
        %25 = arith.addi %c144_i32, %0 : i32
        %true_9 = arith.constant true
        %26 = "simt_step.wave_count_bits"(%true_9) : (i1) -> i32
        "simt_step.buffer.store"(%arg0, %25, %26) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
        %27 = arith.addi %arg2, %24 : i32
        %c1_i32_10 = arith.constant 1 : i32
        %28 = arith.addi %arg3, %c1_i32_10 : i32
        %true_11 = arith.constant true
        %c160_i32 = arith.constant 160 : i32
        %29 = arith.addi %arg3, %arg3 : i32
        %30 = arith.addi %29, %arg3 : i32
        %31 = arith.addi %30, %arg3 : i32
        %32 = arith.addi %c160_i32, %31 : i32
        %33 = arith.addi %32, %0 : i32
        %true_12 = arith.constant true
        %34 = "simt_step.wave_count_bits"(%true_12) : (i1) -> i32
        "simt_step.buffer.store"(%arg0, %33, %34) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
        "simt_step.break"(%27, %28) : (i32, i32) -> ()
      }) : (i32, i32) -> (i32, i32)
      %true = arith.constant true
      %c176_i32 = arith.constant 176 : i32
      %12 = arith.addi %c176_i32, %0 : i32
      %true_4 = arith.constant true
      %13 = "simt_step.wave_count_bits"(%true_4) : (i1) -> i32
      "simt_step.buffer.store"(%arg0, %12, %13) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
      "simt_step.yield"(%11#0) {fallthrough = false} : (i32) -> ()
    ^bb2(%14: i32):  // no predecessors
      %c2_i32_5 = arith.constant 2 : i32
      %15 = arith.remsi %0, %c2_i32_5 : i32
      %16 = "simt_step.switch"(%15, %0) ({
      ^bb0(%arg2: i32):
        %c0_i32_7 = arith.constant 0 : i32
        %c0_i32_8 = arith.constant 0 : i32
        %22:2 = "simt_step.loop"(%c0_i32_7, %c0_i32_8) ({
        ^bb0(%arg3: i32, %arg4: i32):
          %c4_i32 = arith.constant 4 : i32
          %27 = arith.remsi %0, %c4_i32 : i32
          %c1_i32_13 = arith.constant 1 : i32
          %28 = arith.addi %27, %c1_i32_13 : i32
          %29 = arith.cmpi slt, %arg4, %28 : i32
          "simt_step.condition"(%29, %arg3, %arg4) : (i1, i32, i32) -> ()
        }, {
        ^bb0(%arg3: i32, %arg4: i32):
          %27 = arith.addi %arg3, %arg4 : i32
          %c1_i32_13 = arith.constant 1 : i32
          %28 = arith.addi %arg4, %c1_i32_13 : i32
          %true_14 = arith.constant true
          %c192_i32 = arith.constant 192 : i32
          %29 = arith.addi %arg4, %arg4 : i32
          %30 = arith.addi %29, %arg4 : i32
          %31 = arith.addi %30, %arg4 : i32
          %32 = arith.addi %c192_i32, %31 : i32
          %33 = arith.addi %32, %0 : i32
          %true_15 = arith.constant true
          %34 = "simt_step.wave_count_bits"(%true_15) : (i1) -> i32
          "simt_step.buffer.store"(%arg0, %33, %34) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
          "simt_step.break"(%27, %28) : (i32, i32) -> ()
        }) : (i32, i32) -> (i32, i32)
        "simt_step.yield"(%22#0) {fallthrough = true} : (i32) -> ()
      ^bb1(%23: i32):  // no predecessors
        %c0_i32_9 = arith.constant 0 : i32
        %c0_i32_10 = arith.constant 0 : i32
        %24:2 = "simt_step.loop"(%c0_i32_9, %c0_i32_10) ({
        ^bb0(%arg3: i32, %arg4: i32):
          %c2_i32_13 = arith.constant 2 : i32
          %27 = arith.remsi %0, %c2_i32_13 : i32
          %c1_i32_14 = arith.constant 1 : i32
          %28 = arith.addi %27, %c1_i32_14 : i32
          %29 = arith.cmpi slt, %arg4, %28 : i32
          "simt_step.condition"(%29, %arg3, %arg4) : (i1, i32, i32) -> ()
        }, {
        ^bb0(%arg3: i32, %arg4: i32):
          %27 = arith.addi %arg3, %arg4 : i32
          %c1_i32_13 = arith.constant 1 : i32
          %28 = arith.addi %arg4, %c1_i32_13 : i32
          %true_14 = arith.constant true
          %c208_i32 = arith.constant 208 : i32
          %29 = arith.addi %arg4, %arg4 : i32
          %30 = arith.addi %29, %arg4 : i32
          %31 = arith.addi %30, %arg4 : i32
          %32 = arith.addi %c208_i32, %31 : i32
          %33 = arith.addi %32, %0 : i32
          %true_15 = arith.constant true
          %34 = "simt_step.wave_count_bits"(%true_15) : (i1) -> i32
          "simt_step.buffer.store"(%arg0, %33, %34) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
          "simt_step.yield"(%27, %28) : (i32, i32) -> ()
        }) : (i32, i32) -> (i32, i32)
        %true_11 = arith.constant true
        %c224_i32 = arith.constant 224 : i32
        %25 = arith.addi %c224_i32, %0 : i32
        %true_12 = arith.constant true
        %26 = "simt_step.wave_count_bits"(%true_12) : (i1) -> i32
        "simt_step.buffer.store"(%arg0, %25, %26) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
        "simt_step.yield"(%24#0) {fallthrough = false} : (i32) -> ()
      }) {case_values = array<i64: 0>, default_index = 1 : i64} : (i32, i32) -> i32
      "simt_step.yield"(%16) {fallthrough = false} : (i32) -> ()
    ^bb3(%17: i32):  // no predecessors
      %c1_i32 = arith.constant 1 : i32
      %18 = arith.cmpi slt, %0, %c1_i32 : i32
      %19 = "simt_step.if"(%18) ({
        %c0_i32_7 = arith.constant 0 : i32
        %c0_i32_8 = arith.constant 0 : i32
        %22:2 = "simt_step.loop"(%c0_i32_7, %c0_i32_8) ({
        ^bb0(%arg2: i32, %arg3: i32):
          %c2_i32_9 = arith.constant 2 : i32
          %23 = arith.remsi %0, %c2_i32_9 : i32
          %c1_i32_10 = arith.constant 1 : i32
          %24 = arith.addi %23, %c1_i32_10 : i32
          %25 = arith.cmpi slt, %arg3, %24 : i32
          "simt_step.condition"(%25, %arg2, %arg3) : (i1, i32, i32) -> ()
        }, {
        ^bb0(%arg2: i32, %arg3: i32):
          %23 = arith.addi %arg2, %arg3 : i32
          %c1_i32_9 = arith.constant 1 : i32
          %24 = arith.addi %arg3, %c1_i32_9 : i32
          %true_10 = arith.constant true
          %c240_i32 = arith.constant 240 : i32
          %25 = arith.addi %arg3, %arg3 : i32
          %26 = arith.addi %25, %arg3 : i32
          %27 = arith.addi %26, %arg3 : i32
          %28 = arith.addi %c240_i32, %27 : i32
          %29 = arith.addi %28, %0 : i32
          %true_11 = arith.constant true
          %30 = "simt_step.wave_count_bits"(%true_11) : (i1) -> i32
          "simt_step.buffer.store"(%arg0, %29, %30) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
          "simt_step.continue"(%23, %24) : (i32, i32) -> ()
        }) : (i32, i32) -> (i32, i32)
        "simt_step.yield"(%22#0) : (i32) -> ()
      }, {
        %c0_i32_7 = arith.constant 0 : i32
        %c0_i32_8 = arith.constant 0 : i32
        %22:2 = "simt_step.loop"(%c0_i32_7, %c0_i32_8) ({
        ^bb0(%arg2: i32, %arg3: i32):
          %c2_i32_9 = arith.constant 2 : i32
          %23 = arith.remsi %0, %c2_i32_9 : i32
          %c1_i32_10 = arith.constant 1 : i32
          %24 = arith.addi %23, %c1_i32_10 : i32
          %25 = arith.cmpi slt, %arg3, %24 : i32
          "simt_step.condition"(%25, %arg2, %arg3) : (i1, i32, i32) -> ()
        }, {
        ^bb0(%arg2: i32, %arg3: i32):
          %23 = arith.addi %arg2, %arg3 : i32
          %c1_i32_9 = arith.constant 1 : i32
          %24 = arith.addi %arg3, %c1_i32_9 : i32
          %true_10 = arith.constant true
          %c256_i32 = arith.constant 256 : i32
          %25 = arith.addi %arg3, %arg3 : i32
          %26 = arith.addi %25, %arg3 : i32
          %27 = arith.addi %26, %arg3 : i32
          %28 = arith.addi %c256_i32, %27 : i32
          %29 = arith.addi %28, %0 : i32
          %true_11 = arith.constant true
          %30 = "simt_step.wave_count_bits"(%true_11) : (i1) -> i32
          "simt_step.buffer.store"(%arg0, %29, %30) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
          "simt_step.continue"(%23, %24) : (i32, i32) -> ()
        }) : (i32, i32) -> (i32, i32)
        "simt_step.yield"(%22#0) : (i32) -> ()
      }) : (i1) -> i32
      %c272_i32 = arith.constant 272 : i32
      %20 = arith.addi %c272_i32, %0 : i32
      %true_6 = arith.constant true
      %21 = "simt_step.wave_count_bits"(%true_6) : (i1) -> i32
      "simt_step.buffer.store"(%arg0, %20, %21) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
      "simt_step.yield"(%19) {fallthrough = false} : (i32) -> ()
    }) {case_values = array<i64: 0, 1, 2>, default_index = 3 : i64} : (i32, i32) -> i32
    %c2_i32_0 = arith.constant 2 : i32
    %7 = arith.remsi %0, %c2_i32_0 : i32
    %c0_i32 = arith.constant 0 : i32
    %8 = arith.addi %0, %c0_i32 : i32
    %9 = "simt_step.switch"(%7, %8) ({
    ^bb0(%arg1: i32):
      %c0_i32_1 = arith.constant 0 : i32
      %10 = arith.addi %0, %c0_i32_1 : i32
      %true = arith.constant true
      %c288_i32 = arith.constant 288 : i32
      %11 = arith.addi %c288_i32, %0 : i32
      %true_2 = arith.constant true
      %12 = "simt_step.wave_count_bits"(%true_2) : (i1) -> i32
      "simt_step.buffer.store"(%arg0, %11, %12) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
      "simt_step.yield"(%10) {fallthrough = false} : (i32) -> ()
    ^bb1(%13: i32):  // no predecessors
      %c2_i32_3 = arith.constant 2 : i32
      %14 = arith.remsi %0, %c2_i32_3 : i32
      %15 = "simt_step.switch"(%14, %0) ({
      ^bb0(%arg2: i32):
        %c1_i32_6 = arith.constant 1 : i32
        %21 = arith.cmpi slt, %0, %c1_i32_6 : i32
        %22 = "simt_step.if"(%21) ({
          %c4_i32_12 = arith.constant 4 : i32
          %29 = arith.addi %0, %c4_i32_12 : i32
          "simt_step.yield"(%29) : (i32) -> ()
        }, {
          %c3_i32_12 = arith.constant 3 : i32
          "simt_step.yield"(%c3_i32_12) : (i32) -> ()
        }) : (i1) -> i32
        %c304_i32 = arith.constant 304 : i32
        %23 = arith.addi %c304_i32, %0 : i32
        %true_7 = arith.constant true
        %24 = "simt_step.wave_count_bits"(%true_7) : (i1) -> i32
        "simt_step.buffer.store"(%arg0, %23, %24) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
        %true_8 = arith.constant true
        %c320_i32 = arith.constant 320 : i32
        %25 = arith.addi %c320_i32, %0 : i32
        %true_9 = arith.constant true
        %26 = "simt_step.wave_count_bits"(%true_9) : (i1) -> i32
        "simt_step.buffer.store"(%arg0, %25, %26) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
        "simt_step.yield"(%22) {fallthrough = false} : (i32) -> ()
      ^bb1(%27: i32):  // no predecessors
        %c0_i32_10 = arith.constant 0 : i32
        %c0_i32_11 = arith.constant 0 : i32
        %28:2 = "simt_step.loop"(%c0_i32_10, %c0_i32_11) ({
        ^bb0(%arg3: i32, %arg4: i32):
          %c4_i32_12 = arith.constant 4 : i32
          %29 = arith.remsi %0, %c4_i32_12 : i32
          %c1_i32_13 = arith.constant 1 : i32
          %30 = arith.addi %29, %c1_i32_13 : i32
          %31 = arith.cmpi slt, %arg4, %30 : i32
          "simt_step.condition"(%31, %arg3, %arg4) : (i1, i32, i32) -> ()
        }, {
        ^bb0(%arg3: i32, %arg4: i32):
          %29 = arith.addi %arg3, %arg4 : i32
          %c1_i32_12 = arith.constant 1 : i32
          %30 = arith.addi %arg4, %c1_i32_12 : i32
          %true_13 = arith.constant true
          %c336_i32 = arith.constant 336 : i32
          %31 = arith.addi %arg4, %arg4 : i32
          %32 = arith.addi %31, %arg4 : i32
          %33 = arith.addi %32, %arg4 : i32
          %34 = arith.addi %c336_i32, %33 : i32
          %35 = arith.addi %34, %0 : i32
          %true_14 = arith.constant true
          %36 = "simt_step.wave_count_bits"(%true_14) : (i1) -> i32
          "simt_step.buffer.store"(%arg0, %35, %36) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
          "simt_step.yield"(%29, %30) : (i32, i32) -> ()
        }) : (i32, i32) -> (i32, i32)
        "simt_step.yield"(%28#0) {fallthrough = false} : (i32) -> ()
      }) {case_values = array<i64: 0>, default_index = 0 : i64} : (i32, i32) -> i32
      "simt_step.yield"(%15) {fallthrough = false} : (i32) -> ()
    ^bb2(%16: i32):  // no predecessors
      %c4_i32 = arith.constant 4 : i32
      %17 = arith.remsi %0, %c4_i32 : i32
      %c1_i32 = arith.constant 1 : i32
      %18 = "simt_step.switch"(%17, %c1_i32) ({
      ^bb0(%arg2: i32):
        %c2_i32_6 = arith.constant 2 : i32
        %21 = arith.remsi %0, %c2_i32_6 : i32
        %c2_i32_7 = arith.constant 2 : i32
        %22 = arith.addi %0, %c2_i32_7 : i32
        %23 = "simt_step.switch"(%21, %22) ({
        ^bb0(%arg3: i32):
          %c2_i32_15 = arith.constant 2 : i32
          %true_16 = arith.constant true
          %c352_i32 = arith.constant 352 : i32
          %35 = arith.addi %c352_i32, %0 : i32
          %true_17 = arith.constant true
          %36 = "simt_step.wave_count_bits"(%true_17) : (i1) -> i32
          "simt_step.buffer.store"(%arg0, %35, %36) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
          "simt_step.yield"(%c2_i32_15) {fallthrough = false} : (i32) -> ()
        ^bb1(%37: i32):  // no predecessors
          "simt_step.yield"(%0) {fallthrough = false} : (i32) -> ()
        }) {case_values = array<i64: 0>, default_index = 0 : i64} : (i32, i32) -> i32
        %true_8 = arith.constant true
        %c368_i32 = arith.constant 368 : i32
        %24 = arith.addi %c368_i32, %0 : i32
        %true_9 = arith.constant true
        %25 = "simt_step.wave_count_bits"(%true_9) : (i1) -> i32
        "simt_step.buffer.store"(%arg0, %24, %25) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
        "simt_step.yield"(%23) {fallthrough = false} : (i32) -> ()
      ^bb1(%26: i32):  // no predecessors
        %c2_i32_10 = arith.constant 2 : i32
        %27 = arith.cmpi slt, %0, %c2_i32_10 : i32
        %28 = "simt_step.if"(%27) ({
          %c1_i32_15 = arith.constant 1 : i32
          "simt_step.yield"(%c1_i32_15) : (i32) -> ()
        }, {
          %c1_i32_15 = arith.constant 1 : i32
          %35 = arith.addi %0, %c1_i32_15 : i32
          "simt_step.yield"(%35) : (i32) -> ()
        }) : (i1) -> i32
        %c384_i32 = arith.constant 384 : i32
        %29 = arith.addi %c384_i32, %0 : i32
        %true_11 = arith.constant true
        %30 = "simt_step.wave_count_bits"(%true_11) : (i1) -> i32
        "simt_step.buffer.store"(%arg0, %29, %30) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
        "simt_step.yield"(%28) {fallthrough = true} : (i32) -> ()
      ^bb2(%31: i32):  // no predecessors
        %c2_i32_12 = arith.constant 2 : i32
        "simt_step.yield"(%c2_i32_12) {fallthrough = false} : (i32) -> ()
      ^bb3(%32: i32):  // no predecessors
        %c4_i32_13 = arith.constant 4 : i32
        %33 = arith.remsi %0, %c4_i32_13 : i32
        %c0_i32_14 = arith.constant 0 : i32
        %34 = "simt_step.switch"(%33, %c0_i32_14) ({
        ^bb0(%arg3: i32):
          %c3_i32_15 = arith.constant 3 : i32
          "simt_step.yield"(%c3_i32_15) {fallthrough = false} : (i32) -> ()
        ^bb1(%35: i32):  // no predecessors
          %c0_i32_16 = arith.constant 0 : i32
          %36 = arith.addi %0, %c0_i32_16 : i32
          %true_17 = arith.constant true
          %c400_i32 = arith.constant 400 : i32
          %37 = arith.addi %c400_i32, %0 : i32
          %true_18 = arith.constant true
          %38 = "simt_step.wave_count_bits"(%true_18) : (i1) -> i32
          "simt_step.buffer.store"(%arg0, %37, %38) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
          "simt_step.yield"(%36) {fallthrough = false} : (i32) -> ()
        ^bb2(%39: i32):  // no predecessors
          "simt_step.yield"(%0) {fallthrough = false} : (i32) -> ()
        ^bb3(%40: i32):  // no predecessors
          %c0_i32_19 = arith.constant 0 : i32
          %41 = arith.addi %0, %c0_i32_19 : i32
          "simt_step.yield"(%41) {fallthrough = false} : (i32) -> ()
        }) {case_values = array<i64: 0, 1, 2>, default_index = 0 : i64} : (i32, i32) -> i32
        "simt_step.yield"(%34) {fallthrough = false} : (i32) -> ()
      }) {case_values = array<i64: 0, 1, 2>, default_index = 0 : i64} : (i32, i32) -> i32
      %true_4 = arith.constant true
      %c416_i32 = arith.constant 416 : i32
      %19 = arith.addi %c416_i32, %0 : i32
      %true_5 = arith.constant true
      %20 = "simt_step.wave_count_bits"(%true_5) : (i1) -> i32
      "simt_step.buffer.store"(%arg0, %19, %20) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
      "simt_step.yield"(%18) {fallthrough = false} : (i32) -> ()
    }) {case_values = array<i64: 0, 1>, default_index = 2 : i64} : (i32, i32) -> i32
    return
  }
}