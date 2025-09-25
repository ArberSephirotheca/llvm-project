// MLIR-LABEL: func.func @main(
// MLIR: "simt_step.fence"() {{.*}}memsem = #simt_step.memsem<AcqRel>, memspace = #simt_step.memspace<Shared>, scope = #simt_step.scope<Workgroup>{{.*}} : () -> ()
// MLIR: "simt_step.barrier"() {{.*}}memsem = #simt_step.memsem<AcqRel>, scope = #simt_step.scope<Workgroup>{{.*}} : () -> ()
