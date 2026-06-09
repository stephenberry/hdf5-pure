//! HDF5 data engine for the `hdf5-pure` workspace.
//!
//! This crate is the strongly-connected core of the format: chunked read and
//! write, the chunk index structures, attributes, and group traversal are
//! mutually recursive and therefore live together in one crate.
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

#[cfg(not(feature = "std"))]
pub use hdf5_pure_core::nosync;

#[cfg(feature = "zfp")]
pub use hdf5_pure_filters::zfp;
pub use hdf5_pure_filters::{filter_pipeline, filters, scaleoffset};

pub use hdf5_pure_format::{
    attribute_info, btree_v1, btree_v2, data_layout, dataspace, datatype, fractal_heap,
    global_heap, group_info, link_info, link_message, local_heap, object_header,
    object_header_writer, shared_message, superblock, symbol_table, vl_data,
};

pub mod attribute;
pub mod chunk_cache;
pub mod chunked_read;
pub mod chunked_write;
pub mod data_read;
pub mod extensible_array;
pub mod file_writer;
pub mod fixed_array;
pub mod group_v1;
pub mod group_v2;
pub mod metadata_index;
pub mod type_builders;

#[cfg(feature = "ndarray")]
pub mod ndarray_support;

#[cfg(feature = "parallel")]
pub mod lane_partition;
#[cfg(feature = "parallel")]
pub mod parallel_read;

#[cfg(feature = "provenance")]
pub mod provenance;
