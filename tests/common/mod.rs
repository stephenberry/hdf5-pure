//! Helpers shared by the integration tests. Not a test target itself (Cargo only
//! builds `tests/*.rs`), so it is compiled into each binary that declares
//! `mod common;`.
//!
//! Items in [`heap`] are pure Rust and available everywhere. Everything at this
//! level touches the reference C library, which is a dev-dependency gated to
//! 64-bit little-endian targets, so it compiles out on `i686` and `s390x`
//! rather than forcing every including file to carry the gate.
#![allow(dead_code)]

pub mod heap;

#[cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
mod c_library;
// Not every including binary uses the C helpers — some want only `heap` — and an
// unused re-export is a warning that `allow(dead_code)` does not cover.
#[cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#[allow(unused_imports)]
pub use c_library::*;
