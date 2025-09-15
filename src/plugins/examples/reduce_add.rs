use serde_json::json;
use crate::plugins::registry::{InterpreterHandler, InstrId, Params};
use crate::semantics::SemanticsContext;

pub struct ReduceAdd;
impl InterpreterHandler for ReduceAdd {
    fn interpret(&self, _id: InstrId, _p: &Params, _inputs: &[serde_json::Value],
                 outs: &mut Vec<serde_json::Value>, ctx: &mut SemanticsContext) -> bool {
        // Demo: sum 1 across active lanes
        let mut sum = 0u32;
        for lane in 0..ctx.subgroup_width { if (ctx.active_mask & (1u64<<lane)) != 0 { sum += 1; } }
        outs.push(json!(sum));
        true
    }
}
