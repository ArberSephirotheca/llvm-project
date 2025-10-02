// MLIR-LABEL: module {
// MLIR: func.func @main(
// MLIR: simt_step.dispatch_thread_id : i32
// MLIR: simt_step.fence {scope = #simt_step.scope<Workgroup>, memsem = #simt_step.memsem<AcqRel>, memspace = #simt_step.memspace<Shared>}
// MLIR: simt_step.barrier {scope = #simt_step.scope<Workgroup>, memsem = #simt_step.memsem<AcqRel>}
