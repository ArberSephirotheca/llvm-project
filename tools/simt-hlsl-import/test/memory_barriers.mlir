// MLIR-LABEL: func.func @main(
// MLIR: simt_step.fence {scope = #simt_step.scope<Workgroup>, memsem = #simt_step.memsem<AcqRel>, memspace = #simt_step.memspace<Shared>}
// MLIR: simt_step.fence {scope = #simt_step.scope<Workgroup>, memsem = #simt_step.memsem<AcqRel>, memspace = #simt_step.memspace<Shared>}
// MLIR: simt_step.barrier {scope = #simt_step.scope<Workgroup>, memsem = #simt_step.memsem<AcqRel>}
// MLIR: simt_step.fence {scope = #simt_step.scope<Workgroup>, memsem = #simt_step.memsem<AcqRel>, memspace = #simt_step.memspace<Global>}
// MLIR: simt_step.fence {scope = #simt_step.scope<Workgroup>, memsem = #simt_step.memsem<AcqRel>, memspace = #simt_step.memspace<Global>}
// MLIR: simt_step.barrier {scope = #simt_step.scope<Workgroup>, memsem = #simt_step.memsem<AcqRel>}
// MLIR: simt_step.fence {scope = #simt_step.scope<Workgroup>, memsem = #simt_step.memsem<AcqRel>, memspace = #simt_step.memspace<Generic>}
// MLIR: simt_step.fence {scope = #simt_step.scope<Workgroup>, memsem = #simt_step.memsem<AcqRel>, memspace = #simt_step.memspace<Generic>}
// MLIR: simt_step.barrier {scope = #simt_step.scope<Workgroup>, memsem = #simt_step.memsem<AcqRel>}
