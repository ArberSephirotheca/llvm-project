// MLIR-LABEL: func.func @main(
// MLIR: %[[TID:.*]] = simt_step.dispatch_thread_id : vector<3xi32>
// MLIR: %[[EX0:.*]] = vector.extract %[[TID]][0] : i32 from vector<3xi32>
// MLIR: %[[ZERO:.*]] = arith.constant dense<0> : vector<2xi32>
// MLIR: %[[EX1:.*]] = vector.extract %[[TID]][1] : i32 from vector<3xi32>
// MLIR: %[[INS0:.*]] = vector.insert %[[EX1]], %[[ZERO]] [0] : i32 into vector<2xi32>
// MLIR: %[[EX2:.*]] = vector.extract %[[TID]][2] : i32 from vector<3xi32>
// MLIR: %[[INS1:.*]] = vector.insert %[[EX2]], %[[INS0]] [1] : i32 into vector<2xi32>
// MLIR: return
