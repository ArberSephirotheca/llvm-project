//! Generic extension op `simt.custom` with (instr, params JSON).

use melior::ir::{Attribute, Identifier, Location, Operation, Region, Type, Value};
use melior::ir::attribute::StringAttribute;
use melior::Context;

pub fn custom(ctx: &Context, instr: &str, params_json: &str, operands: &[Value], result_tys: &[Type], loc: Location) -> Operation {
    let attrs: &[(&str, Attribute)] = &[
        ("instr", StringAttribute::new(ctx, instr).into()),
        ("params", StringAttribute::new(ctx, params_json).into())
    ];
    Operation::new(ctx, Identifier::new(ctx, "simt.custom"), operands, result_tys, attrs, Region::new(), loc)
}
