//! Core primitives shared across the `hdf5-pure` workspace.
//!
//! This is an internal implementation crate. Depend on [`hdf5-pure`] instead;
//! its public API is the supported, semver-tracked surface.
//!
//! [`hdf5-pure`]: https://docs.rs/hdf5-pure
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod checksum;
pub mod convert;
pub mod error;
pub mod message_type;
pub mod signature;
pub mod source;

#[cfg(not(feature = "std"))]
pub mod nosync;
