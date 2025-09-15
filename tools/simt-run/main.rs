use anyhow::Result;
use clap::Parser;
use melior::Context;
use simt_step::plugins::registry::{InstructionSpec, OperandDesc, ResultDesc, Registry};
use simt_step::semantics::{interpreter::Interpreter, SemanticsContext};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value="baseline")]
    model: String,
    #[arg(long, default_value="32")]
    w: u32,
}

fn main() -> Result<()> {
    // Demo runner: create a registry and run a toy handler on a synthetic op.
    let _ctx = Context::new();
    let mut reg = Registry::new();
    let id = reg.register_instruction(InstructionSpec{
        name: "reduce_add".into(),
        operands: vec![OperandDesc{name:"v".into(), ty:"i32".into()}],
        results: vec![ResultDesc{name:"sum".into(), ty:"i32".into()}],
        has_scope:true, has_sync:true, has_memsem:false,
        needs_subgroup:true, needs_shared_mem:false,
    });
    // TODO: register a real handler (from plugins::examples)
    // reg.register_handler("baseline", id, Box::new(simt_step::plugins::examples::reduce_add::ReduceAdd));

    let mut ctx = SemanticsContext{ subgroup_width: 32, active_mask: 0xFFFF_FFFF_FFFF_FFFFu64 };
    let interp = Interpreter{ registry: &reg, model: "baseline" };
    // TODO: build/load MLIR and walk ops; for now, just print context.
    println!("(demo) subgroup_width={}, active_mask=0x{:x}", ctx.subgroup_width, ctx.active_mask);
    Ok(())
}
