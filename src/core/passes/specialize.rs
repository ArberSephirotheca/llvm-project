//! SpecializeSimt: bake subgroup width (W), masks, indices, seeds.
//! TODO: walk ops, turn constant cases into constants, DCE.

pub struct SpecializeConfig {
    pub subgroup_width: u32,
    pub active_mask: u64,
}

impl Default for SpecializeConfig {
    fn default() -> Self { Self { subgroup_width: 32, active_mask: !0 } }
}
