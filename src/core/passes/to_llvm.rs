//! SimtToLLVM: lower simt_step ops to calls into simt_runtime intrinsics.
//! This file will hold melior-based rewrites when you wire the compiled oracle.

pub struct ToLlvmConfig {
    pub model: String, // baseline | vendorish | ...
}
