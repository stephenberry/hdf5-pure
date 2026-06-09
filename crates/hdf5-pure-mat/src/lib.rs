//! MATLAB v7.3 (`.mat`) serde (de)serialization for the `hdf5-pure` workspace.
//!
//! This is an internal implementation crate. Depend on [`hdf5-pure`] with the
//! `serde` feature instead; its public API is the supported, semver-tracked
//! surface.
//!
//! [`hdf5-pure`]: https://docs.rs/hdf5-pure

// Re-exported from lower crates so in-crate `crate::<module>` paths resolve.
pub use hdf5_pure_api::{reader, types, writer};
pub use hdf5_pure_engine::{error, file_writer, type_builders};

pub mod mat;
