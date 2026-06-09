//! HDF5 on-disk format structures for the `hdf5-pure` workspace.
//!
//! This is an internal implementation crate. Depend on [`hdf5-pure`] instead;
//! its public API is the supported, semver-tracked surface.
//!
//! [`hdf5-pure`]: https://docs.rs/hdf5-pure
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

// Re-exported from lower crates so in-crate `crate::<module>` paths resolve.
pub use hdf5_pure_core::{checksum, convert, error, message_type, signature, source};

pub mod attribute_info;
pub mod btree_v1;
pub mod btree_v2;
pub mod data_layout;
pub mod dataspace;
pub mod datatype;
pub mod fractal_heap;
pub mod global_heap;
pub mod group_info;
pub mod link_info;
pub mod link_message;
pub mod local_heap;
pub mod object_header;
pub mod object_header_writer;
pub mod shared_message;
pub mod superblock;
pub mod symbol_table;
pub mod vl_data;
