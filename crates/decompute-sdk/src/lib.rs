//! Public in-process Decompute SDK.
//!
//! Runtime integrations are added behind engine features. This crate owns the
//! stable request, result, and actor-facing API rather than network transport.

mod generation;
mod local;

pub use decompute_core::*;
pub use generation::*;
pub use local::*;
