//! Pure-Rust HDF5 file reading, writing, and in-place editing library.
//!
//! `hdf5-pure` is a zero-C-dependency crate for creating, reading, and editing
//! HDF5 files. It is WASM-compatible and supports `no_std` environments with
//! `alloc`.
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
//! # Editing files in place
//!
//! [`EditSession`] opens an existing file and adds, deletes, or copies objects
//! without reading it all in and rewriting it. New data and rebuilt object
//! headers are appended at end-of-file and the superblock is repointed last, so
//! the cost is proportional to what changes rather than to the file size. It
//! edits files written by this crate, the reference HDF5 C library, and h5py
//! across all of their on-disk formats, and refuses — rather than silently
//! degrade the file — anything it cannot reproduce faithfully.
//!
//! ```rust,no_run
//! use hdf5_pure::EditSession;
//!
//! let mut session = EditSession::open("output.h5").unwrap();
//! session.create_dataset("extra").with_f64_data(&[4.0, 5.0]);
//! session.delete("data");            // H5Ldelete
//! session.commit().unwrap();
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
// In `no_std` builds the high-level entry points that consume the parsing and
// serialization machinery — the `reader`, `writer`, and `swmr_writer` modules —
// are `std`-gated and therefore absent, leaving much of that machinery without
// an in-crate consumer. It is deliberately kept available for future `no_std`
// readers/writers rather than cfg-gating every module to `std`, so `dead_code`
// is allowed only in `no_std` builds. `std` builds (the primary target) keep
// full `dead_code` enforcement.
#![cfg_attr(not(feature = "std"), allow(dead_code))]

#[cfg(not(feature = "std"))]
extern crate alloc;

// ---------------------------------------------------------------------------
// Internal modules (encapsulated as `pub(crate)`; the curated public surface is
// re-exported at the bottom of this file).
// ---------------------------------------------------------------------------

pub(crate) mod attribute;
pub(crate) mod attribute_info;
pub(crate) mod btree_v1;
pub(crate) mod btree_v2;
pub(crate) mod checksum;
pub(crate) mod chunk_cache;
pub(crate) mod chunked_read;
pub(crate) mod chunked_write;
pub(crate) mod convert;
pub(crate) mod data_layout;
pub(crate) mod data_read;
pub(crate) mod dataspace;
pub(crate) mod datatype;
pub(crate) mod error;
pub(crate) mod extensible_array;
pub(crate) mod file_writer;
pub(crate) mod filter_pipeline;
pub(crate) mod filters;
pub(crate) mod fixed_array;
pub(crate) mod fractal_heap;
pub(crate) mod global_heap;
pub(crate) mod group_v1;
pub(crate) mod group_v2;
#[cfg(feature = "parallel")]
pub(crate) mod lane_partition;
pub(crate) mod libver;
pub(crate) mod link_info;
pub(crate) mod link_message;
pub(crate) mod local_heap;
pub(crate) mod message_type;
pub(crate) mod object_header;
pub(crate) mod object_header_writer;
#[cfg(feature = "parallel")]
pub(crate) mod parallel_read;
pub(crate) mod scaleoffset;
pub(crate) mod shared_message;
pub(crate) mod signature;
pub(crate) mod source;
pub(crate) mod superblock;
pub(crate) mod symbol_table;
pub(crate) mod type_builders;
pub(crate) mod vl_data;
#[cfg(feature = "zfp")]
pub(crate) mod zfp;

#[cfg(feature = "provenance")]
pub(crate) mod provenance;

#[cfg(not(feature = "std"))]
pub(crate) mod nosync;

// ---------------------------------------------------------------------------
// High-level modules
// ---------------------------------------------------------------------------

#[cfg(feature = "std")]
pub(crate) mod edit;
#[cfg(feature = "std")]
pub(crate) mod free_space;
#[cfg(feature = "std")]
pub(crate) mod reader;
#[cfg(feature = "std")]
pub(crate) mod repack;
#[cfg(feature = "std")]
pub(crate) mod swmr_writer;
#[cfg(feature = "std")]
pub(crate) mod types;
#[cfg(feature = "std")]
pub(crate) mod writer;

#[cfg(feature = "std")]
pub mod mat;

#[cfg(feature = "ndarray")]
pub(crate) mod ndarray_support;

// ---------------------------------------------------------------------------
// Public API re-exports
// ---------------------------------------------------------------------------

#[cfg(feature = "std")]
pub use error::Error;
pub use error::FormatError;

#[cfg(feature = "std")]
pub use reader::{Dataset, File, Group, is_hdf5, is_hdf5_bytes};

pub use libver::LibVer;

#[cfg(all(feature = "std", feature = "provenance"))]
pub use provenance::VerifyResult;

#[cfg(feature = "std")]
pub use types::{AttrValue, DType};

#[cfg(feature = "std")]
pub use writer::FileBuilder;

#[cfg(feature = "std")]
pub use swmr_writer::SwmrWriter;

#[cfg(feature = "std")]
pub use edit::EditSession;

#[cfg(feature = "std")]
pub use repack::{RepackOptions, repack};

#[cfg(feature = "ndarray")]
pub use ndarray_support::H5Element;

pub use scaleoffset::ScaleOffset;

// The HDF5 datatype handle returned by the `make_*_type` constructors and
// accepted by the compound/enum builders and `DatasetBuilder::with_dtype`.
pub use datatype::Datatype;

pub use type_builders::{
    CompoundTypeBuilder, DatasetBuilder, EnumTypeBuilder, FinishedGroup, GroupBuilder,
    make_f32_type, make_f64_type, make_i8_type, make_i16_type, make_i32_type, make_i64_type,
    make_u8_type, make_u16_type, make_u32_type, make_u64_type,
};
