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

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

// ---------------------------------------------------------------------------
// Format-level modules (from rustyhdf5-format)
// ---------------------------------------------------------------------------

pub mod attribute;
pub mod attribute_info;
pub mod chunk_cache;
pub mod chunked_read;
pub mod chunked_write;
pub mod file_writer;
pub mod metadata_index;
pub mod object_header_writer;
pub mod type_builders;
pub mod btree_v1;
pub mod checksum;
pub mod btree_v2;
pub mod fractal_heap;
pub mod group_info;
pub mod group_v2;
pub mod link_info;
pub mod link_message;
pub mod data_layout;
pub mod data_read;
pub mod filter_pipeline;
pub mod extensible_array;
pub mod fixed_array;
pub mod filters;
#[cfg(feature = "parallel")]
pub mod lane_partition;
#[cfg(feature = "parallel")]
pub mod parallel_read;
pub mod dataspace;
pub mod datatype;
pub mod error;
pub mod global_heap;
pub mod group_v1;
pub mod local_heap;
pub mod message_type;
pub mod object_header;
pub mod shared_message;
pub mod signature;
pub mod superblock;
pub mod symbol_table;
pub mod vl_data;
#[cfg(feature = "zfp")]
pub mod zfp;

#[cfg(feature = "provenance")]
pub mod provenance;

#[cfg(not(feature = "std"))]
pub(crate) mod nosync;

// ---------------------------------------------------------------------------
// High-level modules
// ---------------------------------------------------------------------------

#[cfg(feature = "std")]
pub mod reader;
#[cfg(feature = "std")]
pub mod types;
#[cfg(feature = "std")]
pub mod writer;

#[cfg(feature = "std")]
pub mod mat;

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

pub use type_builders::{
    CompoundTypeBuilder, EnumTypeBuilder,
    make_f32_type, make_f64_type, make_i8_type, make_i16_type, make_i32_type, make_i64_type,
    make_u8_type, make_u16_type, make_u32_type, make_u64_type,
};
