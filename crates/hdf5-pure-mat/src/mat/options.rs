//! Configuration knobs for MAT v7.3 writers.
//!
//! [`Options`] groups every policy a MATLAB writer might need to flex on:
//! string class (`char` vs `string`), 1-D vector orientation, name validation
//! behavior, compression, and more.
//!
//! The defaults match the historical hdf5-pure serde writer (`String -> char`,
//! no name sanitization, no compression, column vectors). Callers wanting the
//! richer "modern MATLAB" output should construct an `Options` with explicit
//! `string_class: StringClass::String`, `invalid_name_policy: Sanitize`, etc.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// MATLAB class to use when emitting Rust `String` (or BEVE string) values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum StringClass {
    /// Encode as `char` (UTF-16 row vector). Compatible with `strcmp`, but
    /// not with the `==` operator.
    Char,
    /// Encode as the modern MATLAB `string` class via `mxOPAQUE_CLASS`.
    /// Supports `==` semantics. Costs a `#refs#` payload and a
    /// `#subsystem#/MCOS` entry per writer.
    String,
}

/// Encoding for 1-D vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum OneDimensionalMode {
    /// MATLAB shape `[N, 1]`.
    ColumnVector,
    /// MATLAB shape `[1, N]`.
    RowVector,
}

/// Behavior for `null` / `Option::None` values inside sequences and at the
/// dataset root.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum NullPolicy {
    /// Map `None` to MATLAB `struct([])` (an empty struct array).
    EmptyStructArray,
    /// Reject `None` with an error.
    Error,
}

/// Behavior when a Rust struct field name or BEVE key is not a valid MATLAB
/// identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum InvalidNamePolicy {
    /// Return an error with the offending name.
    Error,
    /// Rewrite the name into a valid identifier and deduplicate.
    Sanitize,
}

/// Behavior for BEVE values that have no direct MATLAB encoding (bf16, f16,
/// 128-bit integers, unknown extensions).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum UnsupportedPolicy {
    /// Reject unsupported values with an error.
    Error,
    /// Convert unsupported scalar values to their string representation and
    /// emit them as MATLAB `string` objects.
    StringFallback,
    /// Widen low-precision floats (bf16, f16) to MATLAB `single`.
    LossyNumericWidening,
}

/// HDF5 dataset compression settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum Compression {
    /// No compression (no chunking).
    None,
    /// HDF5 deflate compression.
    Deflate {
        /// zlib level (0-9).
        level: u8,
        /// Apply HDF5 byte-shuffle filter before deflate.
        shuffle: bool,
    },
}

/// Behavior for row-major matrix payloads (e.g. BEVE `MatrixLayout::Right`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum RowMajorPolicy {
    /// Reorder row-major payloads into MATLAB column-major layout.
    ReorderToColumnMajor,
    /// Return an error rather than reordering.
    Error,
}

/// Marker encoding to use for empty values (`Vec<T>` of length 0,
/// `struct([])`, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum EmptyMarkerEncoding {
    /// Zero-element dataset of shape `[0, 0]` with `MATLAB_empty=1`.
    /// (hdf5-pure 0.3 historical default.)
    ZeroElement,
    /// One-element-per-dim `uint64` dataset whose payload is the dimension
    /// vector, with `MATLAB_empty=1`. (beve historical default.)
    DataAsDims,
}

/// Aggregated options for MAT v7.3 writers.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct Options {
    /// MATLAB class for string values.
    pub string_class: StringClass,
    /// HDF5 dataset compression settings.
    pub compression: Compression,
    /// Policy for invalid MATLAB names.
    pub invalid_name_policy: InvalidNamePolicy,
    /// Policy for `null`.
    pub null_policy: NullPolicy,
    /// Policy for unsupported numeric/BEVE types.
    pub unsupported_policy: UnsupportedPolicy,
    /// 1-D vector orientation.
    pub one_dimensional_mode: OneDimensionalMode,
    /// Row-major matrix payload handling.
    pub row_major_policy: RowMajorPolicy,
    /// Empty marker encoding.
    pub empty_marker_encoding: EmptyMarkerEncoding,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            string_class: StringClass::Char,
            compression: Compression::None,
            invalid_name_policy: InvalidNamePolicy::Error,
            null_policy: NullPolicy::EmptyStructArray,
            unsupported_policy: UnsupportedPolicy::Error,
            one_dimensional_mode: OneDimensionalMode::ColumnVector,
            row_major_policy: RowMajorPolicy::ReorderToColumnMajor,
            empty_marker_encoding: EmptyMarkerEncoding::ZeroElement,
        }
    }
}

impl Options {
    /// Construct options that emit the modern MATLAB `string` class via
    /// `mxOPAQUE_CLASS` and use the data-as-dims empty marker encoding.
    /// Matches what real MATLAB's `save -v7.3` produces (and what the BEVE
    /// → MAT walker has historically used).
    pub fn with_modern_strings() -> Self {
        Self {
            string_class: StringClass::String,
            empty_marker_encoding: EmptyMarkerEncoding::DataAsDims,
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_legacy_serde_writer() {
        let o = Options::default();
        assert_eq!(o.string_class, StringClass::Char);
        assert_eq!(o.invalid_name_policy, InvalidNamePolicy::Error);
        assert_eq!(o.empty_marker_encoding, EmptyMarkerEncoding::ZeroElement);
    }

    #[test]
    fn modern_strings_constructor() {
        let o = Options::with_modern_strings();
        assert_eq!(o.string_class, StringClass::String);
        assert_eq!(o.empty_marker_encoding, EmptyMarkerEncoding::DataAsDims);
    }
}
