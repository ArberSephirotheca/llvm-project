pub mod interpreter;
pub mod runtime;

#[derive(Default)]
pub struct SemanticsContext {
    pub subgroup_width: u32,
    pub active_mask: u64,
}

#[derive(Clone)]
pub struct Model(pub String); // "baseline" | "vendorish" | ...
