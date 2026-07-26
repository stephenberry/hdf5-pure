//! Helpers shared by the integration tests. Not a test target itself (Cargo only
//! builds `tests/*.rs`), so it is compiled into each binary that declares
//! `mod common;`.
//!
//! Items in [`heap`] are pure Rust and available everywhere. Everything at this
//! level touches the reference C library, which is a dev-dependency gated to
//! 64-bit-pointer targets, so it compiles out on 32-bit rather than forcing every
//! including file to carry the gate.
#![allow(dead_code)]

pub mod heap;

#[cfg(not(target_pointer_width = "32"))]
mod c_library;
// Not every including binary uses the C helpers — some want only `heap` — and an
// unused re-export is a warning that `allow(dead_code)` does not cover.
#[cfg(not(target_pointer_width = "32"))]
#[allow(unused_imports)]
pub use c_library::*;
