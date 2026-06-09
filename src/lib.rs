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

// Re-exported from `hdf5-core` (keeps the `hdf5_pure::<module>` paths and the
// in-crate `crate::<module>` references resolving unchanged).
pub use hdf5_core::{checksum, convert, error, message_type, signature, source};

#[cfg(not(feature = "std"))]
pub(crate) use hdf5_core::nosync;

// Re-exported from `hdf5-filters`.
#[cfg(feature = "zfp")]
pub use hdf5_filters::zfp;
pub use hdf5_filters::{filter_pipeline, filters, scaleoffset};

// Re-exported from `hdf5-format`.
pub use hdf5_format::{
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
#[cfg(feature = "parallel")]
pub mod lane_partition;
pub mod metadata_index;
#[cfg(feature = "parallel")]
pub mod parallel_read;
pub mod type_builders;

#[cfg(feature = "provenance")]
pub mod provenance;

// ---------------------------------------------------------------------------
// High-level modules
// ---------------------------------------------------------------------------

#[cfg(feature = "std")]
pub mod reader;
#[cfg(feature = "std")]
pub mod swmr_writer;
#[cfg(feature = "std")]
pub mod types;
#[cfg(feature = "std")]
pub mod writer;

#[cfg(feature = "std")]
pub mod mat;

#[cfg(feature = "ndarray")]
pub mod ndarray_support;

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
