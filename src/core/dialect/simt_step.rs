//! MLIR `simt_step` dialect surface (Rust-side construction helpers).
//! Ops: wave.all/any/ballot/shuffle, barrier, fence. Attributes: scope, mem.

use melior::dialect::{arith, cf, func};
use melior::ir::{attribute, r#type, Location, Operation, Identifier, Region, Type, Value, Attribute};
use melior::Context;

#[derive(Clone, Copy, Debug)]
pub enum Scope { Thread, Subgroup, Workgroup }
#[derive(Clone, Copy, Debug)]
pub enum MemSem { None, Acquire, Release, AcqRel }

pub fn scope_attr(ctx: &Context, s: Scope) -> Attribute {
    let s = match s { Scope::Thread=>"Thread", Scope::Subgroup=>"Subgroup", Scope::Workgroup=>"Workgroup" };
    attribute::StringAttribute::new(ctx, s).into()
}
pub fn memsem_attr(ctx: &Context, m: MemSem) -> Attribute {
    let m = match m { MemSem::None=>"None", MemSem::Acquire=>"Acquire", MemSem::Release=>"Release", MemSem::AcqRel=>"AcqRel" };
    attribute::StringAttribute::new(ctx, m).into()
}

// Example: build a wave.all op
pub fn wave_all(ctx: &Context, pred: Value, loc: Location) -> Operation {
    let i1: Type = r#type::IntegerType::new(ctx, 1).into();
    Operation::new(
        ctx,
        Identifier::new(ctx, "simt_step.wave_all"),
        &[pred],
        &[i1],
        &[],
        Region::new(),
        loc,
    )
}
