//! Error type for MATLAB v7.3 serde (de)serialization.

use core::fmt;

use crate::error::{Error as Hdf5Error, FormatError};

/// Errors that can occur when (de)serializing `.mat` v7.3 files.
#[derive(Debug)]
pub enum MatError {
    /// Underlying HDF5 I/O or format error.
    Hdf5(Hdf5Error),
    /// Underlying HDF5 format parse error.
    Format(FormatError),
    /// I/O error when reading or writing a file path.
    Io(std::io::Error),
    /// Top-level must be a struct with named fields (each field becomes a MATLAB variable).
    RootMustBeStruct,
    /// The requested Rust type has no MATLAB v7.3 encoding in this crate.
    UnsupportedType(&'static str),
    /// A sequence contained elements of different primitive types.
    MixedSequenceElementTypes,
    /// A 2-D matrix had inconsistent row lengths.
    RaggedMatrix {
        /// Expected row length (first row).
        expected: usize,
        /// The row that differed.
        got: usize,
    },
    /// A dataset's on-disk shape didn't match the Rust type.
    ShapeMismatch {
        /// The Rust side's expectation.
        expected: String,
        /// What the file contained.
        actual: String,
    },
    /// A required struct field was missing from the file.
    MissingField(String),
    /// A `MATLAB_class` attribute value wasn't recognized.
    UnknownClass(String),
    /// UTF-16 decoding of a `char` dataset failed.
    Utf16Decode(String),
    /// A generic serde-originated error (from `Error::custom`).
    Custom(String),
}

impl fmt::Display for MatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatError::Hdf5(e) => write!(f, "HDF5 error: {e}"),
            MatError::Format(e) => write!(f, "HDF5 format error: {e}"),
            MatError::Io(e) => write!(f, "I/O error: {e}"),
            MatError::RootMustBeStruct => write!(
                f,
                "top-level value must be a struct with named fields; each field becomes a MATLAB variable"
            ),
            MatError::UnsupportedType(t) => write!(f, "unsupported Rust type for MAT v7.3: {t}"),
            MatError::MixedSequenceElementTypes => write!(
                f,
                "sequence elements have mixed primitive types; all elements of a numeric array must share a type"
            ),
            MatError::RaggedMatrix { expected, got } => write!(
                f,
                "ragged 2-D matrix: expected row length {expected}, got {got}"
            ),
            MatError::ShapeMismatch { expected, actual } => {
                write!(f, "shape mismatch: expected {expected}, got {actual}")
            }
            MatError::MissingField(name) => write!(f, "missing required field: {name}"),
            MatError::UnknownClass(c) => write!(f, "unknown MATLAB_class: {c:?}"),
            MatError::Utf16Decode(msg) => write!(f, "UTF-16 decode: {msg}"),
            MatError::Custom(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for MatError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            MatError::Hdf5(e) => Some(e),
            MatError::Format(e) => Some(e),
            MatError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<Hdf5Error> for MatError {
    fn from(e: Hdf5Error) -> Self {
        MatError::Hdf5(e)
    }
}

impl From<FormatError> for MatError {
    fn from(e: FormatError) -> Self {
        MatError::Format(e)
    }
}

impl From<std::io::Error> for MatError {
    fn from(e: std::io::Error) -> Self {
        MatError::Io(e)
    }
}

impl serde::ser::Error for MatError {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        MatError::Custom(msg.to_string())
    }
}

impl serde::de::Error for MatError {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        MatError::Custom(msg.to_string())
    }
}
