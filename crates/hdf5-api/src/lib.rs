//! High-level HDF5 reader/writer API for the `hdf5-pure` workspace.
//!
//! This is an internal implementation crate. Depend on [`hdf5-pure`] instead;
//! its public API is the supported, semver-tracked surface.
//!
//! [`hdf5-pure`]: https://docs.rs/hdf5-pure

// Re-exported from lower crates so in-crate `crate::<module>` paths resolve.
// `hdf5-engine` re-exports the core/format/filters modules at its own root, so
// every lower-layer module this crate names is reachable through it.
pub use hdf5_engine::{
    attribute, checksum, chunk_cache, chunked_write, convert, data_layout, data_read, dataspace,
    datatype, error, extensible_array, file_writer, filter_pipeline, group_v1, group_v2,
    message_type, object_header, signature, source, superblock, type_builders, vl_data,
};

pub mod reader;
pub mod swmr_writer;
pub mod types;
pub mod writer;

#[cfg(feature = "ndarray")]
pub mod ndarray_support;
