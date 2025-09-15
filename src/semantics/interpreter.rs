use anyhow::{Result, anyhow};
use melior::ir::OperationRef;
use crate::plugins::registry::{Registry, Params};
use crate::SemanticsContext;

pub struct Interpreter<'a> { pub registry: &'a Registry, pub model: &'a str }

impl<'a> Interpreter<'a> {
    pub fn run_custom(&self, op: &OperationRef, ctx: &mut SemanticsContext) -> Result<serde_json::Value> {
        let instr_attr = op.attribute("instr").ok_or_else(|| anyhow!("missing instr"))?;
        let params_attr = op.attribute("params").ok_or_else(|| anyhow!("missing params"))?;
        let instr = instr_attr.to_string(); // naive; replace with StringAttribute downcast
        let params_json = params_attr.to_string();
        let params: Params = serde_json::from_str(params_json.trim_matches('"'))?;
        let id = self.registry.lookup(instr.trim_matches('"')).ok_or_else(|| anyhow!("unknown instr"))?;
        let h = self.registry.handler(self.model, id).ok_or_else(|| anyhow!("no handler for model"))?;
        let mut outs = vec![];
        let ok = h.interpret(id, &params, &[], &mut outs, ctx);
        if !ok { return Err(anyhow!("handler failed")); }
        Ok(outs.into_iter().next().unwrap_or(serde_json::json!(null)))
    }
}
