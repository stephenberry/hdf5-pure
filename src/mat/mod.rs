//! Serde-based reading and writing of MATLAB v7.3 `.mat` files.
//!
//! MAT v7.3 files are HDF5 files with MATLAB conventions:
//! - A 512-byte userblock starting with the `MATLAB 7.3 MAT-file...` signature
//! - Every dataset and group carries a `MATLAB_class` attribute (`"double"`,
//!   `"char"`, `"struct"`, ...)
//! - Strings are stored as `uint16` datasets encoding UTF-16LE
//! - 2-D arrays are laid out column-major (Fortran order); HDF5 shape is
//!   `[cols, rows]` so that MATLAB sees the intended `[rows, cols]`
//!
//! # Required top-level shape
//!
//! The outer Rust value must be a **struct with named fields**. Each field
//! becomes a top-level MATLAB variable. This matches `scipy.io.savemat` and
//! MATLAB's workspace model.
//!
//! ```no_run
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
//! ```
//!
//! # Data model
//!
//! | Rust | HDF5 / MATLAB |
//! |---|---|
//! | `f64`, `f32`, `i*`, `u*` | scalar dataset `[1,1]`, `MATLAB_class` = `"double"` etc. |
//! | `bool` | `uint8` scalar, `MATLAB_class = "logical"` |
//! | `String` | `uint16` `[1, N]` UTF-16LE, `MATLAB_class = "char"` |
//! | `Vec<T>` of numeric `T` | `[1, N]` row vector |
//! | [`Matrix`]`<T>` or `Vec<Vec<T>>` | column-major 2-D dataset |
//! | [`Complex32`] / [`Complex64`] | compound `{real, imag}` dataset |
//! | nested struct | group with child datasets, `MATLAB_class = "struct"` |
//! | `Option<T>` | field is omitted when `None` |
//! | unit enum variants | UTF-16 char dataset containing the variant name |
//!
//! See the crate-level README for a fuller description.

pub mod class;
pub mod complex;
pub mod error;
pub mod matrix;
pub mod userblock;
pub mod utf16;
pub(crate) mod value;

pub mod de;
pub mod ser;

pub use class::MatClass;
pub use complex::{Complex32, Complex64};
pub use error::MatError;
pub use matrix::Matrix;

use serde::de::DeserializeOwned;
use serde::Serialize;

/// Serialize `value` to a MAT v7.3 byte vector.
///
/// The root value must be a struct with named fields. Each field becomes a
/// top-level MATLAB variable.
pub fn to_bytes<T: Serialize + ?Sized>(value: &T) -> Result<Vec<u8>, MatError> {
    ser::to_bytes(value)
}

/// Serialize `value` to the given filesystem path as a MAT v7.3 file.
pub fn to_file<T: Serialize + ?Sized, P: AsRef<std::path::Path>>(
    value: &T,
    path: P,
) -> Result<(), MatError> {
    let bytes = to_bytes(value)?;
    std::fs::write(path, bytes).map_err(MatError::Io)
}

/// Deserialize a MAT v7.3 file from a byte slice.
pub fn from_bytes<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, MatError> {
    de::from_bytes(bytes)
}

/// Deserialize a MAT v7.3 file from the filesystem.
pub fn from_file<T: DeserializeOwned, P: AsRef<std::path::Path>>(path: P) -> Result<T, MatError> {
    let bytes = std::fs::read(path).map_err(MatError::Io)?;
    from_bytes(&bytes)
}
