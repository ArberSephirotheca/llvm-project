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
  func.func @main(%arg0: !simt_step.resource<Global, i32>) attributes {simt.num_threads = [8, 1, 1], simt.subgroup_width = 4 : i64} {
    %0 = "simt_step.dispatch_thread_id"() : () -> i32
    %1 = call @helper0(%0) : (i32) -> i32
    %c128_i32 = arith.constant 128 : i32
    %2 = arith.addi %c128_i32, %0 : i32
    "simt_step.buffer.store"(%arg0, %2, %1) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
    %c2_i32 = arith.constant 2 : i32
    %3 = arith.remsi %0, %c2_i32 : i32
    %4 = "simt_step.switch"(%3, %0) ({
    ^bb0(%arg1: i32):
      %c0_i32_3 = arith.constant 0 : i32
      %c0_i32_4 = arith.constant 0 : i32
      %11:2 = "simt_step.loop"(%c0_i32_3, %c0_i32_4) ({
      ^bb0(%arg2: i32, %arg3: i32):
        %c4_i32 = arith.constant 4 : i32
        %22 = arith.remsi %0, %c4_i32 : i32
        %c1_i32 = arith.constant 1 : i32
        %23 = arith.addi %22, %c1_i32 : i32
        %24 = arith.cmpi slt, %arg3, %23 : i32
        "simt_step.condition"(%24, %arg2, %arg3) : (i1, i32, i32) -> ()
      }, {
      ^bb0(%arg2: i32, %arg3: i32):
        %c6_i32 = arith.constant 6 : i32
        %22 = arith.cmpi slt, %0, %c6_i32 : i32
        %23 = "simt_step.if"(%22) ({
          %c4_i32 = arith.constant 4 : i32
          %38 = arith.addi %0, %c4_i32 : i32
          "simt_step.yield"(%38) : (i32) -> ()
        }, {
          %c2_i32_13 = arith.constant 2 : i32
          %38 = arith.addi %0, %c2_i32_13 : i32
          "simt_step.yield"(%38) : (i32) -> ()
        }) : (i1) -> i32
        %c0_i32_9 = arith.constant 0 : i32
        %24 = arith.addi %c0_i32_9, %0 : i32
        %true_10 = arith.constant true
        %25 = "simt_step.wave_count_bits"(%true_10) : (i1) -> i32
        "simt_step.buffer.store"(%arg0, %24, %25) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
        %26 = arith.addi %arg2, %23 : i32
        %c1_i32 = arith.constant 1 : i32
        %27 = arith.addi %arg3, %c1_i32 : i32
        %true_11 = arith.constant true
        %c32_i32 = arith.constant 32 : i32
        %28 = arith.addi %arg3, %arg3 : i32
        %29 = arith.addi %28, %arg3 : i32
        %30 = arith.addi %29, %arg3 : i32
        %31 = arith.addi %30, %arg3 : i32
        %32 = arith.addi %31, %arg3 : i32
        %33 = arith.addi %32, %arg3 : i32
        %34 = arith.addi %33, %arg3 : i32
        %35 = arith.addi %c32_i32, %34 : i32
        %36 = arith.addi %35, %0 : i32
        %true_12 = arith.constant true
        %37 = "simt_step.wave_count_bits"(%true_12) : (i1) -> i32
        "simt_step.buffer.store"(%arg0, %36, %37) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
        "simt_step.yield"(%26, %27) : (i32, i32) -> ()
      }) : (i32, i32) -> (i32, i32)
      "simt_step.yield"(%11#0) {fallthrough = false} : (i32) -> ()
    ^bb1(%12: i32):  // no predecessors
      %true_5 = arith.constant true
      %c64_i32 = arith.constant 64 : i32
      %13 = arith.addi %c64_i32, %0 : i32
      %true_6 = arith.constant true
      %14 = "simt_step.wave_count_bits"(%true_6) : (i1) -> i32
      "simt_step.buffer.store"(%arg0, %13, %14) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
      "simt_step.yield"(%0) {fallthrough = false} : (i32) -> ()
    ^bb2(%15: i32):  // no predecessors
      %c3_i32 = arith.constant 3 : i32
      %16 = arith.remsi %0, %c3_i32 : i32
      %17 = simt_step.lane_id
      %18 = arith.index_cast %17 : index to i32
      %19 = "simt_step.switch"(%16, %18) ({
      ^bb0(%arg2: i32):
        %22 = simt_step.subgroup_id
        %23 = arith.index_cast %22 : index to i32
        %true_9 = arith.constant true
        %c96_i32 = arith.constant 96 : i32
        %24 = arith.addi %c96_i32, %0 : i32
        %true_10 = arith.constant true
        %25 = "simt_step.wave_count_bits"(%true_10) : (i1) -> i32
        "simt_step.buffer.store"(%arg0, %24, %25) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
        "simt_step.yield"(%23) {fallthrough = false} : (i32) -> ()
      ^bb1(%26: i32):  // no predecessors
        %c0_i32_11 = arith.constant 0 : i32
        %c0_i32_12 = arith.constant 0 : i32
        %27:2 = "simt_step.loop"(%c0_i32_11, %c0_i32_12) ({
        ^bb0(%arg3: i32, %arg4: i32):
          %c3_i32_18 = arith.constant 3 : i32
          %36 = arith.remsi %0, %c3_i32_18 : i32
          %c1_i32 = arith.constant 1 : i32
          %37 = arith.addi %36, %c1_i32 : i32
          %38 = arith.cmpi slt, %arg4, %37 : i32
          "simt_step.condition"(%38, %arg3, %arg4) : (i1, i32, i32) -> ()
        }, {
        ^bb0(%arg3: i32, %arg4: i32):
          %36 = arith.addi %arg3, %arg4 : i32
          %c1_i32 = arith.constant 1 : i32
          %37 = arith.addi %arg4, %c1_i32 : i32
          %true_18 = arith.constant true
          %c128_i32_19 = arith.constant 128 : i32
          %38 = arith.addi %arg4, %arg4 : i32
          %39 = arith.addi %38, %arg4 : i32
          %40 = arith.addi %39, %arg4 : i32
          %41 = arith.addi %40, %arg4 : i32
          %42 = arith.addi %41, %arg4 : i32
          %43 = arith.addi %42, %arg4 : i32
          %44 = arith.addi %43, %arg4 : i32
          %45 = arith.addi %c128_i32_19, %44 : i32
          %46 = arith.addi %45, %0 : i32
          %true_20 = arith.constant true
          %47 = "simt_step.wave_count_bits"(%true_20) : (i1) -> i32
          "simt_step.buffer.store"(%arg0, %46, %47) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
          "simt_step.continue"(%36, %37) : (i32, i32) -> ()
        }) : (i32, i32) -> (i32, i32)
        "simt_step.yield"(%27#0) {fallthrough = true} : (i32) -> ()
      ^bb2(%28: i32):  // no predecessors
        %c4_i32 = arith.constant 4 : i32
        %29 = arith.remsi %0, %c4_i32 : i32
        %c3_i32_13 = arith.constant 3 : i32
        %30 = arith.addi %0, %c3_i32_13 : i32
        %31 = "simt_step.switch"(%29, %30) ({
        ^bb0(%arg3: i32):
          %c2_i32_18 = arith.constant 2 : i32
          %true_19 = arith.constant true
          %c160_i32 = arith.constant 160 : i32
          %36 = arith.addi %c160_i32, %0 : i32
          %true_20 = arith.constant true
          %37 = "simt_step.wave_count_bits"(%true_20) : (i1) -> i32
          "simt_step.buffer.store"(%arg0, %36, %37) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
          "simt_step.yield"(%c2_i32_18) {fallthrough = false} : (i32) -> ()
        ^bb1(%38: i32):  // no predecessors
          %39 = simt_step.lane_id
          %40 = arith.index_cast %39 : index to i32
          %true_21 = arith.constant true
          %c192_i32 = arith.constant 192 : i32
          %41 = arith.addi %c192_i32, %0 : i32
          %true_22 = arith.constant true
          %42 = "simt_step.wave_count_bits"(%true_22) : (i1) -> i32
          "simt_step.buffer.store"(%arg0, %41, %42) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
          "simt_step.yield"(%40) {fallthrough = false} : (i32) -> ()
        ^bb2(%43: i32):  // no predecessors
          %44 = simt_step.lane_id
          %45 = arith.index_cast %44 : index to i32
          %true_23 = arith.constant true
          %c224_i32 = arith.constant 224 : i32
          %46 = arith.addi %c224_i32, %0 : i32
          %true_24 = arith.constant true
          %47 = "simt_step.wave_count_bits"(%true_24) : (i1) -> i32
          "simt_step.buffer.store"(%arg0, %46, %47) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
          "simt_step.yield"(%45) {fallthrough = false} : (i32) -> ()
        ^bb3(%48: i32):  // no predecessors
          %49 = simt_step.lane_id
          %50 = arith.index_cast %49 : index to i32
          "simt_step.yield"(%50) {fallthrough = false} : (i32) -> ()
        }) {case_values = array<i64: 0, 1, 2>, default_index = 0 : i64} : (i32, i32) -> i32
        "simt_step.yield"(%31) {fallthrough = true} : (i32) -> ()
      ^bb3(%32: i32):  // no predecessors
        %c0_i32_14 = arith.constant 0 : i32
        %c0_i32_15 = arith.constant 0 : i32
        %33:2 = "simt_step.loop"(%c0_i32_14, %c0_i32_15) ({
        ^bb0(%arg3: i32, %arg4: i32):
          %c2_i32_18 = arith.constant 2 : i32
          %36 = arith.remsi %0, %c2_i32_18 : i32
          %c1_i32 = arith.constant 1 : i32
          %37 = arith.addi %36, %c1_i32 : i32
          %38 = arith.cmpi slt, %arg4, %37 : i32
          "simt_step.condition"(%38, %arg3, %arg4) : (i1, i32, i32) -> ()
        }, {
        ^bb0(%arg3: i32, %arg4: i32):
          %36 = arith.addi %arg3, %arg4 : i32
          %c1_i32 = arith.constant 1 : i32
          %37 = arith.addi %arg4, %c1_i32 : i32
          %true_18 = arith.constant true
          %c256_i32 = arith.constant 256 : i32
          %38 = arith.addi %arg4, %arg4 : i32
          %39 = arith.addi %38, %arg4 : i32
          %40 = arith.addi %39, %arg4 : i32
          %41 = arith.addi %40, %arg4 : i32
          %42 = arith.addi %41, %arg4 : i32
          %43 = arith.addi %42, %arg4 : i32
          %44 = arith.addi %43, %arg4 : i32
          %45 = arith.addi %c256_i32, %44 : i32
          %46 = arith.addi %45, %0 : i32
          %true_19 = arith.constant true
          %47 = "simt_step.wave_count_bits"(%true_19) : (i1) -> i32
          "simt_step.buffer.store"(%arg0, %46, %47) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
          "simt_step.yield"(%36, %37) : (i32, i32) -> ()
        }) : (i32, i32) -> (i32, i32)
        %true_16 = arith.constant true
        %c288_i32 = arith.constant 288 : i32
        %34 = arith.addi %c288_i32, %0 : i32
        %true_17 = arith.constant true
        %35 = "simt_step.wave_count_bits"(%true_17) : (i1) -> i32
        "simt_step.buffer.store"(%arg0, %34, %35) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
        "simt_step.yield"(%33#0) {fallthrough = false} : (i32) -> ()
      }) {case_values = array<i64: 0, 1, 2>, default_index = 2 : i64} : (i32, i32) -> i32
      %true_7 = arith.constant true
      %c320_i32 = arith.constant 320 : i32
      %20 = arith.addi %c320_i32, %0 : i32
      %true_8 = arith.constant true
      %21 = "simt_step.wave_count_bits"(%true_8) : (i1) -> i32
      "simt_step.buffer.store"(%arg0, %20, %21) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
      "simt_step.yield"(%19) {fallthrough = false} : (i32) -> ()
    }) {case_values = array<i64: 0, 1>, default_index = 2 : i64} : (i32, i32) -> i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %5:2 = "simt_step.loop"(%c0_i32, %c0_i32_0) ({
    ^bb0(%arg1: i32, %arg2: i32):
      %c4_i32 = arith.constant 4 : i32
      %11 = arith.remsi %0, %c4_i32 : i32
      %c1_i32 = arith.constant 1 : i32
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi slt, %arg2, %12 : i32
      "simt_step.condition"(%13, %arg1, %arg2) : (i1, i32, i32) -> ()
    }, {
    ^bb0(%arg1: i32, %arg2: i32):
      %11 = arith.addi %arg1, %arg2 : i32
      %c1_i32 = arith.constant 1 : i32
      %12 = arith.addi %arg2, %c1_i32 : i32
      %true_3 = arith.constant true
      %c352_i32 = arith.constant 352 : i32
      %13 = arith.addi %arg2, %arg2 : i32
      %14 = arith.addi %13, %arg2 : i32
      %15 = arith.addi %14, %arg2 : i32
      %16 = arith.addi %15, %arg2 : i32
      %17 = arith.addi %16, %arg2 : i32
      %18 = arith.addi %17, %arg2 : i32
      %19 = arith.addi %18, %arg2 : i32
      %20 = arith.addi %c352_i32, %19 : i32
      %21 = arith.addi %20, %0 : i32
      %true_4 = arith.constant true
      %22 = "simt_step.wave_count_bits"(%true_4) : (i1) -> i32
      "simt_step.buffer.store"(%arg0, %21, %22) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
      "simt_step.yield"(%11, %12) : (i32, i32) -> ()
    }) : (i32, i32) -> (i32, i32)
    %c2_i32_1 = arith.constant 2 : i32
    %6 = arith.remsi %0, %c2_i32_1 : i32
    %c0_i32_2 = arith.constant 0 : i32
    %7 = arith.cmpi eq, %6, %c0_i32_2 : i32
    %8 = "simt_step.if"(%7) ({
      %c4_i32 = arith.constant 4 : i32
      %11 = arith.cmpi slt, %0, %c4_i32 : i32
      %12 = "simt_step.if"(%11) ({
        "simt_step.yield"(%0) : (i32) -> ()
      }, {
        %c0_i32_4 = arith.constant 0 : i32
        %c0_i32_5 = arith.constant 0 : i32
        %15:2 = "simt_step.loop"(%c0_i32_4, %c0_i32_5) ({
        ^bb0(%arg1: i32, %arg2: i32):
          %c2_i32_6 = arith.constant 2 : i32
          %16 = arith.remsi %0, %c2_i32_6 : i32
          %c1_i32 = arith.constant 1 : i32
          %17 = arith.addi %16, %c1_i32 : i32
          %18 = arith.cmpi slt, %arg2, %17 : i32
          "simt_step.condition"(%18, %arg1, %arg2) : (i1, i32, i32) -> ()
        }, {
        ^bb0(%arg1: i32, %arg2: i32):
          %16 = arith.addi %arg1, %arg2 : i32
          %c1_i32 = arith.constant 1 : i32
          %17 = arith.addi %arg2, %c1_i32 : i32
          %true_6 = arith.constant true
          %c384_i32 = arith.constant 384 : i32
          %18 = arith.addi %arg2, %arg2 : i32
          %19 = arith.addi %18, %arg2 : i32
          %20 = arith.addi %19, %arg2 : i32
          %21 = arith.addi %20, %arg2 : i32
          %22 = arith.addi %21, %arg2 : i32
          %23 = arith.addi %22, %arg2 : i32
          %24 = arith.addi %23, %arg2 : i32
          %25 = arith.addi %c384_i32, %24 : i32
          %26 = arith.addi %25, %0 : i32
          %true_7 = arith.constant true
          %27 = "simt_step.wave_count_bits"(%true_7) : (i1) -> i32
          "simt_step.buffer.store"(%arg0, %26, %27) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
          "simt_step.break"(%16, %17) : (i32, i32) -> ()
        }) : (i32, i32) -> (i32, i32)
        "simt_step.yield"(%15#0) : (i32) -> ()
      }) : (i1) -> i32
      %c416_i32 = arith.constant 416 : i32
      %13 = arith.addi %c416_i32, %0 : i32
      %true_3 = arith.constant true
      %14 = "simt_step.wave_count_bits"(%true_3) : (i1) -> i32
      "simt_step.buffer.store"(%arg0, %13, %14) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
      "simt_step.yield"(%12) : (i32) -> ()
    }, {
      %c2_i32_3 = arith.constant 2 : i32
      %11 = arith.remsi %0, %c2_i32_3 : i32
      %c0_i32_4 = arith.constant 0 : i32
      %12 = arith.cmpi eq, %11, %c0_i32_4 : i32
      %13 = "simt_step.if"(%12) ({
        %c4_i32 = arith.constant 4 : i32
        %16 = arith.remsi %0, %c4_i32 : i32
        %17 = simt_step.subgroup_id
        %18 = arith.index_cast %17 : index to i32
        %19 = "simt_step.switch"(%16, %18) ({
        ^bb0(%arg1: i32):
          %c3_i32 = arith.constant 3 : i32
          %20 = arith.addi %0, %c3_i32 : i32
          %true_6 = arith.constant true
          %c448_i32 = arith.constant 448 : i32
          %21 = arith.addi %c448_i32, %0 : i32
          %true_7 = arith.constant true
          %22 = "simt_step.wave_count_bits"(%true_7) : (i1) -> i32
          "simt_step.buffer.store"(%arg0, %21, %22) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
          "simt_step.yield"(%20) {fallthrough = true} : (i32) -> ()
        ^bb1(%23: i32):  // no predecessors
          %24 = simt_step.subgroup_id
          %25 = arith.index_cast %24 : index to i32
          "simt_step.yield"(%25) {fallthrough = false} : (i32) -> ()
        ^bb2(%26: i32):  // no predecessors
          %27 = simt_step.lane_id
          %28 = arith.index_cast %27 : index to i32
          %true_8 = arith.constant true
          %c480_i32 = arith.constant 480 : i32
          %29 = arith.addi %c480_i32, %0 : i32
          %true_9 = arith.constant true
          %30 = "simt_step.wave_count_bits"(%true_9) : (i1) -> i32
          "simt_step.buffer.store"(%arg0, %29, %30) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
          "simt_step.yield"(%28) {fallthrough = true} : (i32) -> ()
        ^bb3(%31: i32):  // no predecessors
          %32 = simt_step.lane_id
          %33 = arith.index_cast %32 : index to i32
          %true_10 = arith.constant true
          %c512_i32 = arith.constant 512 : i32
          %34 = arith.addi %c512_i32, %0 : i32
          %true_11 = arith.constant true
          %35 = "simt_step.wave_count_bits"(%true_11) : (i1) -> i32
          "simt_step.buffer.store"(%arg0, %34, %35) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
          "simt_step.yield"(%33) {fallthrough = false} : (i32) -> ()
        }) {case_values = array<i64: 0, 1, 2>, default_index = 3 : i64} : (i32, i32) -> i32
        "simt_step.yield"(%19) : (i32) -> ()
      }, {
        %c2_i32_6 = arith.constant 2 : i32
        %16 = arith.remsi %0, %c2_i32_6 : i32
        %17 = simt_step.lane_id
        %18 = arith.index_cast %17 : index to i32
        %19 = "simt_step.switch"(%16, %18) ({
        ^bb0(%arg1: i32):
          %20 = simt_step.lane_id
          %21 = arith.index_cast %20 : index to i32
          %true_7 = arith.constant true
          %c544_i32 = arith.constant 544 : i32
          %22 = arith.addi %c544_i32, %0 : i32
          %true_8 = arith.constant true
          %23 = "simt_step.wave_count_bits"(%true_8) : (i1) -> i32
          "simt_step.buffer.store"(%arg0, %22, %23) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
          "simt_step.yield"(%21) {fallthrough = false} : (i32) -> ()
        ^bb1(%24: i32):  // no predecessors
          "simt_step.yield"(%0) {fallthrough = false} : (i32) -> ()
        ^bb2(%25: i32):  // no predecessors
          %26 = simt_step.lane_id
          %27 = arith.index_cast %26 : index to i32
          "simt_step.yield"(%27) {fallthrough = false} : (i32) -> ()
        }) {case_values = array<i64: 0, 1>, default_index = 0 : i64} : (i32, i32) -> i32
        "simt_step.yield"(%19) : (i32) -> ()
      }) : (i1) -> i32
      %c576_i32 = arith.constant 576 : i32
      %14 = arith.addi %c576_i32, %0 : i32
      %true_5 = arith.constant true
      %15 = "simt_step.wave_count_bits"(%true_5) : (i1) -> i32
      "simt_step.buffer.store"(%arg0, %14, %15) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
      "simt_step.yield"(%13) : (i32) -> ()
    }) : (i1) -> i32
    %c608_i32 = arith.constant 608 : i32
    %9 = arith.addi %c608_i32, %0 : i32
    %true = arith.constant true
    %10 = "simt_step.wave_count_bits"(%true) : (i1) -> i32
    "simt_step.buffer.store"(%arg0, %9, %10) : (!simt_step.resource<Global, i32>, i32, i32) -> ()
    return
  }
}