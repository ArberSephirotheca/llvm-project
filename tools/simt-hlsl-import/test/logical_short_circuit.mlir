// MLIR-LABEL: module {
// MLIR: func.func @main(
// MLIR: %[[MASK:.*]] = "simt_step.active_mask"() : () -> i64
// MLIR: %[[TID:.*]] = simt_step.dispatch_thread_id : i32
// MLIR: %[[TMP0:.*]] = arith.constant 0 : i32
// MLIR: %[[COND_AND:.*]] = arith.cmpi eq, %[[TID]], %{{.*}} : i32
// MLIR: %[[AND:.*]]:2 = "simt_step.if"(%[[COND_AND]]) ({
// MLIR:   %[[ASSIGN1:.*]] = arith.constant 1 : i32
// MLIR:   %[[ASSIGN1_DUP:.*]] = arith.constant 1 : i32
// MLIR:   %[[EQ1:.*]] = arith.cmpi eq, %[[ASSIGN1]], %[[ASSIGN1_DUP]] : i32
// MLIR:   "simt_step.yield"(%[[EQ1]], %[[ASSIGN1]]) : (i1, i32) -> ()
// MLIR: }, {
// MLIR:   %[[FALSE:.*]] = arith.constant false
// MLIR:   "simt_step.yield"(%[[FALSE]], %[[TMP0]]) : (i1, i32) -> ()
// MLIR: }) : (i1) -> (i1, i32)
// MLIR: %[[COND_OR:.*]] = arith.cmpi eq, %[[TID]], %{{.*}} : i32
// MLIR: %[[OR:.*]]:2 = "simt_step.if"(%[[COND_OR]]) ({
// MLIR:   %[[TRUE:.*]] = arith.constant true
// MLIR:   "simt_step.yield"(%[[TRUE]], %[[AND]]#1) : (i1, i32) -> ()
// MLIR: }, {
// MLIR:   %[[ASSIGN2:.*]] = arith.constant 2 : i32
// MLIR:   %[[ASSIGN2_DUP:.*]] = arith.constant 2 : i32
// MLIR:   %[[EQ2:.*]] = arith.cmpi eq, %[[ASSIGN2]], %[[ASSIGN2_DUP]] : i32
// MLIR:   "simt_step.yield"(%[[EQ2]], %[[ASSIGN2]]) : (i1, i32) -> ()
// MLIR: }) : (i1) -> (i1, i32)
