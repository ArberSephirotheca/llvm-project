//! HLSL importer.
//! - Recognize Wave intrinsics -> simt_step ops.
//! - Recognize plugin intrinsics from `simt_ext.hlsli` -> simt.custom.

use anyhow::Result;
use melior::Context;
use melior::ir::{Module, Location, r#type};
use serde_json::json;

pub fn import_hlsl_to_mlir(_ctx: &Context, _source: &str) -> Result<Module> {
    // MVP stub: build a minimal module with one simt.custom op to prove pipeline.
    let ctx = _ctx;
    let loc = Location::unknown(ctx);
    let module = Module::new(loc);
    // TODO: parse DXIL or AST, then emit simt_step and simt.custom ops.
    Ok(module)
}
