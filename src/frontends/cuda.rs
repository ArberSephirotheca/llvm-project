//! CUDA importer.
//! - Recognize __ballot_sync, __all_sync, __syncthreads -> simt_step ops.
//! - Recognize plugin intrinsics via `simt_ext.cuh` -> simt.custom.

use anyhow::Result;
use melior::Context;
use melior::ir::{Module, Location};

pub fn import_cuda_to_mlir(ctx: &Context, _source: &str) -> Result<Module> {
    let loc = Location::unknown(ctx);
    let module = Module::new(loc);
    // TODO: Clang/NVVM path or symbol mapping to simt_step/simt.custom
    Ok(module)
}
