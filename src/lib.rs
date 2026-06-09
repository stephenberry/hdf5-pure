//! Pure-Rust HDF5 file reading and writing library.
//!
//! `hdf5-pure` is a zero-C-dependency crate for creating and reading HDF5 files.
//! It is WASM-compatible and supports `no_std` environments with `alloc`.
//!
//! # Writing files
//!
//! ```rust
//! use hdf5_pure::{FileBuilder, AttrValue};
//!
//! let mut builder = FileBuilder::new();
//! builder.create_dataset("data")
//!     .with_f64_data(&[1.0, 2.0, 3.0])
//!     .with_shape(&[3])
//!     .set_attr("unit", AttrValue::String("m/s".into()));
//! let bytes = builder.finish().unwrap();
//! ```
//!
//! # Reading files
//!
//! ```rust,no_run
//! use hdf5_pure::File;
//!
//! let file = File::from_bytes(std::fs::read("output.h5").unwrap()).unwrap();
//! let ds = file.dataset("data").unwrap();
//! let values = ds.read_f64().unwrap();
//! ```
//!
//! # Streaming large files
//!
//! [`File::open`] reads the whole file into memory. To read a file too large to
//! buffer (for example a multi-gigabyte file on a 32-bit host, where it exceeds
//! the address space), use [`File::open_streaming`], which fetches metadata and
//! dataset chunks from the file on demand instead of buffering it whole. The
//! reading API is identical; only the backing store differs. Reads of
//! contiguous, compact, and chunked datasets are supported; the streaming
//! backend currently resolves only latest-format (v2) groups and does not yet
//! read attributes.
//!
//! ```rust,no_run
//! use hdf5_pure::File;
//!
//! let file = File::open_streaming("huge.h5").unwrap();
//! let values = file.dataset("signal").unwrap().read_f64().unwrap();
//! ```
//!
//! # N-dimensional arrays (`ndarray` feature)
//!
//! With the `ndarray` feature, datasets can be written from and read back into
//! [`ndarray`] arrays of any rank, in row-major (C) order:
//!
//! ```
//! # #[cfg(feature = "ndarray")] {
//! use hdf5_pure::{File, FileBuilder};
//! use ndarray::{array, Array2};
//!
//! let a: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
//! let mut fb = FileBuilder::new();
//! fb.create_dataset("m").with_ndarray(&a);
//! let bytes = fb.finish().unwrap();
//!
//! let file = File::from_bytes(bytes).unwrap();
//! let back: Array2<f64> = file.dataset("m").unwrap().read_array().unwrap();
//! assert_eq!(a, back);
//! # }
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

// ---------------------------------------------------------------------------
// Format-level modules (from rustyhdf5-format)
// ---------------------------------------------------------------------------

// Re-exported from `hdf5-pure-core` (keeps the `hdf5_pure::<module>` paths and the
// in-crate `crate::<module>` references resolving unchanged). `#[doc(inline)]`
// makes rustdoc render and emit the inlined contents under `hdf5_pure::*` so
// the documented and semver-tracked surface matches the pre-split crate.
#[doc(inline)]
pub use hdf5_pure_core::{checksum, convert, error, message_type, signature, source};

// Re-exported from `hdf5-pure-filters`.
#[cfg(feature = "zfp")]
#[doc(inline)]
pub use hdf5_pure_filters::zfp;
#[doc(inline)]
pub use hdf5_pure_filters::{filter_pipeline, filters, scaleoffset};

// Re-exported from `hdf5-pure-format`.
#[doc(inline)]
pub use hdf5_pure_format::{
    attribute_info, btree_v1, btree_v2, data_layout, dataspace, datatype, fractal_heap,
    global_heap, group_info, link_info, link_message, local_heap, object_header,
    object_header_writer, shared_message, superblock, symbol_table, vl_data,
};

// Re-exported from `hdf5-pure-engine`.
#[doc(inline)]
pub use hdf5_pure_engine::{
    attribute, chunk_cache, chunked_read, chunked_write, data_read, extensible_array, file_writer,
    fixed_array, group_v1, group_v2, metadata_index, type_builders,
};
#[cfg(feature = "parallel")]
#[doc(inline)]
pub use hdf5_pure_engine::{lane_partition, parallel_read};

#[cfg(feature = "provenance")]
#[doc(inline)]
pub use hdf5_pure_engine::provenance;

// ---------------------------------------------------------------------------
// High-level modules
// ---------------------------------------------------------------------------

// Re-exported from `hdf5-pure-api` (the std-only high-level surface).
#[cfg(feature = "std")]
#[doc(inline)]
pub use hdf5_pure_api::{reader, swmr_writer, types, writer};

// Re-exported from `hdf5-pure-mat`. The MAT builder API is available under `std`;
// the serde-based (de)serialization parts inside it are additionally gated by
// the `serde` feature (forwarded to `hdf5-mat/serde`).
#[cfg(feature = "std")]
#[doc(inline)]
pub use hdf5_pure_mat::mat;

#[cfg(feature = "ndarray")]
#[doc(inline)]
pub use hdf5_pure_api::ndarray_support;

// ---------------------------------------------------------------------------
// Public API re-exports
// ---------------------------------------------------------------------------

#[cfg(feature = "std")]
pub use error::Error;
pub use error::FormatError;

#[cfg(feature = "std")]
pub use reader::{Dataset, File, Group};

#[cfg(feature = "std")]
pub use types::{AttrValue, DType};

#[cfg(feature = "std")]
pub use writer::FileBuilder;

#[cfg(feature = "std")]
pub use swmr_writer::SwmrWriter;

#[cfg(feature = "ndarray")]
pub use ndarray_support::H5Element;

pub use scaleoffset::ScaleOffset;

pub use type_builders::{
    CompoundTypeBuilder, EnumTypeBuilder, make_f32_type, make_f64_type, make_i8_type,
    make_i16_type, make_i32_type, make_i64_type, make_u8_type, make_u16_type, make_u32_type,
    make_u64_type,
};
