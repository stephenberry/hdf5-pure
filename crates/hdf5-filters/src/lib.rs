//! HDF5 filter pipeline for the `hdf5-pure` workspace.
//!
//! This is an internal implementation crate. Depend on [`hdf5-pure`] instead;
//! its public API is the supported, semver-tracked surface.
//!
//! [`hdf5-pure`]: https://docs.rs/hdf5-pure
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

// Re-exported from lower crates so in-crate `crate::<module>` paths resolve.
pub use hdf5_core::{convert, error};
pub use hdf5_format::datatype;

pub mod filter_pipeline;
pub mod filters;
pub mod scaleoffset;

#[cfg(feature = "zfp")]
pub mod zfp;
