pub mod core;
pub mod frontends;
pub mod plugins;
pub mod semantics;

// Re-exports for convenience
pub use core::{dialect, passes};
pub use plugins::registry::{InstructionSpec, Registry};
