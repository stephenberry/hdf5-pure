//! MATLAB v7.3 (`.mat`) file conventions on top of HDF5.
//!
//! MAT v7.3 files are HDF5 files with MATLAB conventions:
//! - A 512-byte userblock starting with the `MATLAB 7.3 MAT-file...` signature
//! - Every dataset and group carries a `MATLAB_class` attribute (`"double"`,
//!   `"char"`, `"struct"`, ...)
//! - Strings are stored as `uint16` datasets encoding UTF-16LE (legacy `char`
//!   class) or as opaque-class objects via `#subsystem#/MCOS` (modern `string`
//!   class)
//! - 2-D arrays are laid out column-major (Fortran order); HDF5 shape is
//!   `[cols, rows]` so that MATLAB sees the intended `[rows, cols]`
//!
//! This module exposes three layers of API:
//!
//! 1. **Low-level conventions helpers** (always available): [`class::MatClass`],
//!    [`userblock`], [`utf16`], [`identifier`], [`dims`], [`string_object`].
//! 2. **Mid-level builder** (always available): [`builder::MatBuilder`] wraps
//!    [`crate::FileBuilder`] and applies MATLAB conventions automatically.
//!    Use this for one-pass writers that walk a custom value tree (e.g. a
//!    BEVE wire-format walker).
//! 3. **High-level serde** (gated on `feature = "serde"`): [`to_file`],
//!    [`to_bytes`], [`from_file`], [`from_bytes`] for `#[derive(Serialize,
//!    Deserialize)]` types.
//!
//! ```no_run
//! # #[cfg(feature = "serde")] {
//! use hdf5_pure::mat;
//! use serde::{Serialize, Deserialize};
//!
//! #[derive(Serialize, Deserialize, Debug, PartialEq)]
//! struct Experiment {
//!     name: String,
//!     trial: u32,
//!     samples: Vec<f64>,
//! }
//!
//! let e = Experiment {
//!     name: "run1".into(),
//!     trial: 3,
//!     samples: vec![1.0, 2.0, 3.0],
//! };
//!
//! let bytes = mat::to_bytes(&e).unwrap();
//! let back: Experiment = mat::from_bytes(&bytes).unwrap();
//! assert_eq!(back, e);
//! # }
//! ```

pub mod class;
pub mod error;
pub mod userblock;
pub mod utf16;

// Convention helpers and builder. Available without serde so downstream
// crates can drive a MatBuilder directly.
pub mod builder;
pub mod dims;
pub mod identifier;
pub mod options;
pub mod string_object;

// Complex/Matrix types and the serde-driven (de)serializer. Only meaningful
// when serde is in scope.
#[cfg(feature = "serde")]
pub mod complex;
#[cfg(feature = "serde")]
pub mod matrix;
#[cfg(feature = "serde")]
pub(crate) mod value;

#[cfg(feature = "serde")]
pub mod de;
#[cfg(feature = "serde")]
pub mod ser;

pub use builder::{CellWriter, MatBuilder, StructWriter};
pub use class::MatClass;
pub use error::MatError;
pub use options::{
    Compression, EmptyMarkerEncoding, InvalidNamePolicy, NullPolicy, OneDimensionalMode, Options,
    RowMajorPolicy, StringClass, UnsupportedPolicy,
};

#[cfg(feature = "serde")]
pub use complex::{Complex32, Complex64};
#[cfg(feature = "serde")]
pub use matrix::Matrix;

#[cfg(feature = "serde")]
use serde::Serialize;
#[cfg(feature = "serde")]
use serde::de::DeserializeOwned;

/// Serialize `value` to a MAT v7.3 byte vector.
///
/// The root value must be a struct with named fields. Each field becomes a
/// top-level MATLAB variable.
///
/// # Sequence handling
///
/// Numeric sequences whose elements share a class collapse to row/column vectors or 2-D matrices as before. Any sequence that doesn't (e.g. `Vec<MyStruct>`, `Vec<Option<T>>` with `None` interspersed, ragged `Vec<Vec<f64>>`, or mixed numeric tags) lowers to a MATLAB cell array; each element is interned under the conventional `#refs#` group and the parent dataset stores object references with `MATLAB_class="cell"`. `Option::None` inside a sequence becomes `struct([])` so each cell slot has a defined MATLAB type. Cases that previously errored with `MatError::RaggedMatrix` or `MatError::MixedSequenceElementTypes` now succeed via this fallback.
#[cfg(feature = "serde")]
pub fn to_bytes<T: Serialize + ?Sized>(value: &T) -> Result<Vec<u8>, MatError> {
    ser::to_bytes(value)
}

/// Serialize `value` to the given filesystem path as a MAT v7.3 file. See
/// [`to_bytes`] for details on how heterogeneous sequences are encoded.
#[cfg(feature = "serde")]
pub fn to_file<T: Serialize + ?Sized, P: AsRef<std::path::Path>>(
    value: &T,
    path: P,
) -> Result<(), MatError> {
    let bytes = to_bytes(value)?;
    std::fs::write(path, bytes).map_err(MatError::Io)
}

/// Like [`to_bytes`] but with explicit options. Use for opting into the
/// modern `string` class, name sanitization, compression, etc.
#[cfg(feature = "serde")]
pub fn to_bytes_with_options<T: Serialize + ?Sized>(
    value: &T,
    options: &Options,
) -> Result<Vec<u8>, MatError> {
    ser::to_bytes_with_options(value, options)
}

/// Like [`to_file`] but with explicit options.
#[cfg(feature = "serde")]
pub fn to_file_with_options<T: Serialize + ?Sized, P: AsRef<std::path::Path>>(
    value: &T,
    path: P,
    options: &Options,
) -> Result<(), MatError> {
    let bytes = to_bytes_with_options(value, options)?;
    std::fs::write(path, bytes).map_err(MatError::Io)
}

/// Deserialize a MAT v7.3 file from a byte slice.
#[cfg(feature = "serde")]
pub fn from_bytes<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, MatError> {
    de::from_bytes(bytes)
}

/// Deserialize a MAT v7.3 file from the filesystem.
#[cfg(feature = "serde")]
pub fn from_file<T: DeserializeOwned, P: AsRef<std::path::Path>>(path: P) -> Result<T, MatError> {
    let bytes = std::fs::read(path).map_err(MatError::Io)?;
    from_bytes(&bytes)
}
