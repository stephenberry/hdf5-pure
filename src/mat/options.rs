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
    /// Map `None` to MATLAB `struct([])` (an empty struct array). The field
    /// stays present, so MATLAB code can reference it unconditionally and
    /// test it with `isempty`.
    EmptyStructArray,
    /// Drop the field from its parent struct entirely, so `isfield` reports
    /// `false` and `s.field` raises. This is what the serde writer did
    /// unconditionally before 0.30.
    Omit,
    /// Reject `None` with an error.
    Error,
}

/// Encoding for a fieldless (unit) enum variant.
///
/// Only the serde *writer* consults this: serde hands the serializer both the
/// variant's index and its name, so either encoding is reachable. The BEVE
/// walker has no such choice, since BEVE records whichever one the producing
/// encoder chose. Reading is not conditioned on it either — the deserializer
/// resolves a variant from a name or from an index whichever way the file was
/// written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum UnitVariantEncoding {
    /// Emit the variant's name, honoring `#[serde(rename)]`. Self-describing
    /// in MATLAB, and stable across a reordering of the enum.
    Name,
    /// Emit the variant's zero-based declaration index as a `uint32`.
    ///
    /// This is the index serde reports, which counts variants from zero and
    /// ignores any explicit discriminant: `enum E { A = 5 }` writes `0`.
    ///
    /// Prefer [`Name`](Self::Name) unless a reader already expects the
    /// integer. An index means nothing without the schema that fixes the
    /// ordering, so inserting or reordering a variant silently changes what
    /// existing files decode to, and in MATLAB it reads as a magic number
    /// rather than something `strcmp` can check. This exists for files whose
    /// layout is already fixed, not as a default worth choosing.
    Index,
}

/// Encoding for a sequence that turned out to have no elements.
///
/// Only the serde writer consults this. An empty `serialize_seq` carries no
/// element type, so there is nothing to infer a MATLAB class from; the BEVE
/// walker does not need the hint, because a BEVE typed array names its element
/// type even when the array is empty.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum EmptySequencePolicy {
    /// Emit an empty `double` array, MATLAB's universal `[]`.
    DoubleArray,
    /// Emit an empty cell array, `{}`. Right when the sequence would have held
    /// structs or mixed types had it been populated.
    Cell,
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
///
/// Two writers consume this: the serde writer in this crate, and the BEVE
/// walker that downstream crates build on [`MatBuilder`](crate::mat::MatBuilder).
/// Most knobs apply to both. The ones that cannot are marked in their field
/// docs, and they are asymmetric only because the two input formats carry
/// different type information, never as an oversight.
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
    /// Policy for `null` / `Option::None`.
    pub null_policy: NullPolicy,
    /// Encoding for fieldless enum variants. Serde writer only.
    pub unit_variant_encoding: UnitVariantEncoding,
    /// Encoding for sequences that turned out to be empty. Serde writer only.
    pub empty_sequence_policy: EmptySequencePolicy,
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
            unit_variant_encoding: UnitVariantEncoding::Name,
            empty_sequence_policy: EmptySequencePolicy::DoubleArray,
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
    ///
    /// One shape is this crate's rather than MATLAB's, and only if you also
    /// select [`EmptySequencePolicy::Cell`]: an empty cell array is written
    /// `0x1`, following the `[n, 1]` rule for 1-D cells, where MATLAB's `{}`
    /// is `0x0`. `isempty` holds for both.
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

    /// The encoding defaults have not moved since the serde writer was the
    /// only consumer. `null_policy` has (see the 0.30 changelog), so this is
    /// no longer a statement about the whole struct.
    #[test]
    fn encoding_defaults_match_the_legacy_serde_writer() {
        let o = Options::default();
        assert_eq!(o.string_class, StringClass::Char);
        assert_eq!(o.invalid_name_policy, InvalidNamePolicy::Error);
        assert_eq!(o.empty_marker_encoding, EmptyMarkerEncoding::ZeroElement);
    }

    /// The two knobs added in 0.30 default to what the writer did before them,
    /// so only `null_policy` changes an existing caller's output.
    #[test]
    fn serde_only_knobs_default_to_previous_behavior() {
        let o = Options::default();
        assert_eq!(o.unit_variant_encoding, UnitVariantEncoding::Name);
        assert_eq!(o.empty_sequence_policy, EmptySequencePolicy::DoubleArray);
    }

    #[test]
    fn modern_strings_constructor() {
        let o = Options::with_modern_strings();
        assert_eq!(o.string_class, StringClass::String);
        assert_eq!(o.empty_marker_encoding, EmptyMarkerEncoding::DataAsDims);
    }
}
