//! Public, curated introspection of a dataset's on-disk storage layout and
//! filter pipeline (issue #149).
//!
//! These types are decoded from the HDF5 data-layout and filter-pipeline
//! messages but deliberately omit on-disk encoding artifacts (message and layout
//! version numbers, chunk-index root addresses, and the single-chunk
//! filtered-size sidecar fields), so the public surface is not welded to the
//! internal parse representation. Obtain them from the [`Dataset`] accessors
//! [`layout`], [`chunk_index`], [`chunks`], and [`filter_pipeline`].
//!
//! [`Dataset`]: crate::Dataset
//! [`layout`]: crate::Dataset::layout
//! [`chunk_index`]: crate::Dataset::chunk_index
//! [`chunks`]: crate::Dataset::chunks
//! [`filter_pipeline`]: crate::Dataset::filter_pipeline

#[cfg(not(feature = "std"))]
use alloc::{format, string::String, vec::Vec};

use core::fmt;

use crate::datatype::Dims;
use crate::error::FormatError;

/// How a dataset's raw data is arranged on disk.
///
/// The curated analogue of HDF5's layout class (`H5Pget_layout`), enriched with
/// the per-class facts needed to locate or size the data without decoding it.
/// Obtain it with [`Dataset::layout`](crate::Dataset::layout).
///
/// Use it to choose a reading strategy: a [`Contiguous`](Layout::Contiguous)
/// dataset is a single seek-and-read, while a [`Chunked`](Layout::Chunked)
/// dataset is read (and, for an appendable index, grown) one chunk at a time —
/// enumerate its chunks with [`Dataset::chunks`](crate::Dataset::chunks).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Layout {
    /// Stored inline in the dataset's object header, as used for tiny datasets.
    /// The bytes are already resident once the header is read, so there is no
    /// separate file region to seek to; `size` is the inline byte count.
    Compact {
        /// The number of raw bytes stored inline.
        size: u64,
    },
    /// Stored as one contiguous run of bytes.
    Contiguous {
        /// Absolute file offset of the first byte, or `None` when storage has
        /// not been allocated yet (a fixed-shape dataset that was never
        /// written). In that case `size` is the extent that *would* be written.
        address: Option<u64>,
        /// The length of the run in bytes.
        size: u64,
    },
    /// Stored as a grid of independently located (and optionally filtered)
    /// chunks. Filtered datasets are always chunked.
    Chunked {
        /// The chunk edge lengths, one per dataset dimension, in the same order
        /// as [`shape`](crate::Dataset::shape). This is the value returned by
        /// [`chunk_shape`](crate::Dataset::chunk_shape); the on-disk
        /// element-size dimension is stripped.
        chunk_shape: Vec<u64>,
        /// The index that maps chunk coordinates to file addresses, which
        /// governs append eligibility (see [`ChunkIndex`]).
        index: ChunkIndex,
    },
    /// A virtual dataset whose data is mapped from other datasets. Only the
    /// classification is exposed; the source mappings are not decoded.
    Virtual,
}

/// The kind of index a chunked dataset uses to locate its chunks.
///
/// The curated, named form of HDF5's chunk-index type. The index kind is fixed
/// at dataset creation by the shape and its extensibility, and it determines
/// whether the dataset can be grown in place: see
/// [`supports_inplace_append`](ChunkIndex::supports_inplace_append).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ChunkIndex {
    /// A version-1 B-tree indexes the chunks — the classic layout, used for any
    /// rank and any number of unlimited dimensions in older files.
    BTreeV1,
    /// A single chunk holds the entire dataset; there is no separate index
    /// structure.
    SingleChunk,
    /// Chunk addresses are computed arithmetically from each chunk's position
    /// (a fixed dataspace with every chunk allocated); there is no separate
    /// index structure.
    Implicit,
    /// A fixed array indexes a fixed number of chunks (a non-extensible
    /// dataspace with more than one chunk).
    FixedArray,
    /// An extensible array indexes chunks along a single unlimited dimension.
    /// This is the index [`Dataset::append`](crate::Dataset::append) and
    /// [`Dataset::append_staged`](crate::Dataset::append_staged) grow
    /// in place.
    ExtensibleArray,
    /// A version-2 B-tree indexes the chunks (several unlimited dimensions). A
    /// dataset with this index is classified here, but enumerating its chunks
    /// with [`Dataset::chunks`](crate::Dataset::chunks) is not yet supported.
    BTreeV2,
}

impl ChunkIndex {
    /// Whether a dataset with this index kind can be grown in place with
    /// [`Dataset::append`](crate::Dataset::append) — true only for
    /// [`ExtensibleArray`](ChunkIndex::ExtensibleArray).
    ///
    /// This reflects the index *structure* alone; an actual append also requires
    /// the dataset's first maximum dimension to be unlimited (see
    /// [`Dataset::maxshape`](crate::Dataset::maxshape)).
    #[must_use]
    pub const fn supports_inplace_append(self) -> bool {
        matches!(self, ChunkIndex::ExtensibleArray)
    }

    /// Map an internal `(layout version, chunk index type)` pair to a public
    /// index kind. Version-3 layouts always use a version-1 B-tree; version-4
    /// layouts carry an explicit index type (1..=5).
    pub(crate) fn from_layout(version: u8, index_type: Option<u8>) -> Result<Self, FormatError> {
        Ok(match (version, index_type) {
            (3, _) => ChunkIndex::BTreeV1,
            (4, Some(1)) => ChunkIndex::SingleChunk,
            (4, Some(2)) => ChunkIndex::Implicit,
            (4, Some(3)) => ChunkIndex::FixedArray,
            (4, Some(4)) => ChunkIndex::ExtensibleArray,
            (4, Some(5)) => ChunkIndex::BTreeV2,
            (v, Some(idx)) => {
                return Err(FormatError::ChunkedReadError(format!(
                    "unrecognized chunk index (layout version={v}, index type={idx})"
                )));
            }
            (v, None) => {
                return Err(FormatError::ChunkedReadError(format!(
                    "unrecognized chunk index (layout version={v}, no index type)"
                )));
            }
        })
    }
}

/// The location and on-disk footprint of one stored chunk.
///
/// A `Chunk` is a lightweight record: enumerating chunks reads only the chunk
/// index, never the chunk data. To read one chunk, seek to
/// [`address`](Self::address), read exactly [`storage_size`](Self::storage_size)
/// bytes, then invert the dataset's
/// [`filter_pipeline`](crate::Dataset::filter_pipeline) in *reverse* order
/// (skipping the filters marked in [`filter_mask`](Self::filter_mask)). The
/// curated analogue of `H5Dget_chunk_info`; obtain these from
/// [`Dataset::chunks`](crate::Dataset::chunks).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Chunk {
    /// The logical offset of this chunk's first element within the dataset, one
    /// coordinate per dataset dimension (row-major, in elements). The origin
    /// chunk is all zeros.
    pub offset: Vec<u64>,
    /// The absolute file offset of this chunk's stored bytes.
    pub address: u64,
    /// The number of bytes stored at [`address`](Self::address): the filtered
    /// (compressed) size for a filtered dataset, or the raw chunk byte size
    /// otherwise.
    pub storage_size: u64,
    /// Per-filter skip mask: if bit *i* is set, the *i*-th filter of the
    /// pipeline was not applied to this chunk. `0` means every filter applies.
    pub filter_mask: u32,
}

/// One filter in a dataset's pipeline.
///
/// The curated per-filter analogue of `H5Pget_filter2`. Obtain the ordered
/// pipeline with [`Dataset::filter_pipeline`](crate::Dataset::filter_pipeline);
/// [`Dataset::filters`](crate::Dataset::filters) stays the lighter call when
/// only the identifiers are needed.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Filter {
    /// The registered HDF5 filter identifier — e.g. 1 = deflate, 2 = shuffle,
    /// 3 = fletcher32, 6 = scale-offset, 32000 = lzf, 32013 = zfp. The same
    /// numbering returned by [`Dataset::filters`](crate::Dataset::filters).
    pub id: u16,
    /// The filter's recorded name, when the file stores one. Absent for most
    /// built-in filters, which are identified by [`id`](Self::id) alone.
    pub name: Option<String>,
    /// Whether the filter is optional. When `true`, a reader that cannot apply
    /// the filter may skip it; a mandatory filter (`false`) must be applied for
    /// the data to decode correctly.
    pub is_optional: bool,
    /// The filter's client data (`cd_values`): the auxiliary parameters stored
    /// with it — for deflate, one value, the compression level. The meaning is
    /// filter-specific.
    pub client_data: Vec<u32>,
}

// ---- Display ----
//
// This is the introspection surface a caller prints to describe a dataset, so
// `Display` is the one-line form. `Debug` stays the full record.

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compact { size } => write!(f, "compact ({size} bytes)"),
            Self::Contiguous {
                address: Some(address),
                size,
            } => write!(f, "contiguous ({size} bytes at 0x{address:x})"),
            Self::Contiguous {
                address: None,
                size,
            } => write!(f, "contiguous ({size} bytes, unallocated)"),
            Self::Chunked { chunk_shape, index } => {
                write!(f, "chunked ({}, {index} index)", Dims(chunk_shape))
            }
            Self::Virtual => f.write_str("virtual"),
        }
    }
}

impl fmt::Display for ChunkIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::BTreeV1 => "B-tree v1",
            Self::SingleChunk => "single chunk",
            Self::Implicit => "implicit",
            Self::FixedArray => "fixed array",
            Self::ExtensibleArray => "extensible array",
            Self::BTreeV2 => "B-tree v2",
        })
    }
}

impl fmt::Display for Filter {
    /// The registered name of the filter, falling back to its identifier, and
    /// the client data when the filter carries any.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match well_known_filter_name(self.id) {
            Some(name) => f.write_str(name)?,
            None => match &self.name {
                Some(name) => write!(f, "{name} ({})", self.id)?,
                None => write!(f, "filter {}", self.id)?,
            },
        }
        if !self.client_data.is_empty() {
            f.write_str("(")?;
            for (i, value) in self.client_data.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{value}")?;
            }
            f.write_str(")")?;
        }
        if self.is_optional {
            f.write_str(" [optional]")?;
        }
        Ok(())
    }
}

/// The name of a filter this crate knows by identifier.
///
/// The file may record a name of its own, but most built-in filters store none,
/// which would otherwise leave a bare number in the output.
fn well_known_filter_name(id: u16) -> Option<&'static str> {
    Some(match id {
        1 => "deflate",
        2 => "shuffle",
        3 => "fletcher32",
        4 => "szip",
        5 => "nbit",
        6 => "scaleoffset",
        32000 => "lzf",
        32013 => "zfp",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_index_from_layout_maps_every_kind() {
        assert_eq!(
            ChunkIndex::from_layout(3, None).unwrap(),
            ChunkIndex::BTreeV1
        );
        assert_eq!(
            ChunkIndex::from_layout(3, Some(4)).unwrap(),
            ChunkIndex::BTreeV1,
            "v3 is always a v1 B-tree regardless of the index-type byte"
        );
        assert_eq!(
            ChunkIndex::from_layout(4, Some(1)).unwrap(),
            ChunkIndex::SingleChunk
        );
        assert_eq!(
            ChunkIndex::from_layout(4, Some(2)).unwrap(),
            ChunkIndex::Implicit
        );
        assert_eq!(
            ChunkIndex::from_layout(4, Some(3)).unwrap(),
            ChunkIndex::FixedArray
        );
        assert_eq!(
            ChunkIndex::from_layout(4, Some(4)).unwrap(),
            ChunkIndex::ExtensibleArray
        );
        assert_eq!(
            ChunkIndex::from_layout(4, Some(5)).unwrap(),
            ChunkIndex::BTreeV2
        );
    }

    #[test]
    fn chunk_index_from_layout_rejects_unknown() {
        assert!(ChunkIndex::from_layout(4, Some(9)).is_err());
        assert!(ChunkIndex::from_layout(4, None).is_err());
        assert!(ChunkIndex::from_layout(2, Some(1)).is_err());
    }

    #[test]
    fn only_extensible_array_supports_inplace_append() {
        assert!(ChunkIndex::ExtensibleArray.supports_inplace_append());
        for idx in [
            ChunkIndex::BTreeV1,
            ChunkIndex::SingleChunk,
            ChunkIndex::Implicit,
            ChunkIndex::FixedArray,
            ChunkIndex::BTreeV2,
        ] {
            assert!(!idx.supports_inplace_append());
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod display_tests {
    use super::*;

    #[test]
    fn a_layout_reads_as_one_line() {
        assert_eq!(
            Layout::Compact { size: 40 }.to_string(),
            "compact (40 bytes)"
        );
        assert_eq!(
            Layout::Contiguous {
                address: Some(0x2a0),
                size: 128,
            }
            .to_string(),
            "contiguous (128 bytes at 0x2a0)"
        );
        assert_eq!(
            Layout::Chunked {
                chunk_shape: vec![4, 8],
                index: ChunkIndex::ExtensibleArray,
            }
            .to_string(),
            "chunked (4x8, extensible array index)"
        );
    }

    /// An unallocated dataset says so, rather than printing `None`.
    #[test]
    fn an_unallocated_contiguous_dataset_says_so() {
        let layout = Layout::Contiguous {
            address: None,
            size: 64,
        };
        let shown = layout.to_string();
        assert_eq!(shown, "contiguous (64 bytes, unallocated)");
        assert!(!shown.contains("None"));
    }

    /// Most built-in filters record no name, which would leave a bare number.
    #[test]
    fn a_filter_is_named_by_its_identifier_when_the_file_records_none() {
        let deflate = Filter {
            id: 1,
            name: None,
            is_optional: false,
            client_data: vec![6],
        };
        assert_eq!(deflate.to_string(), "deflate(6)");

        let lzf = Filter {
            id: 32000,
            name: None,
            is_optional: false,
            client_data: vec![],
        };
        assert_eq!(lzf.to_string(), "lzf");
    }

    #[test]
    fn an_unregistered_filter_falls_back_to_its_recorded_name_then_its_id() {
        let named = Filter {
            id: 40000,
            name: Some("custom".into()),
            is_optional: true,
            client_data: vec![],
        };
        assert_eq!(named.to_string(), "custom (40000) [optional]");

        let anonymous = Filter {
            id: 40001,
            name: None,
            is_optional: false,
            client_data: vec![],
        };
        assert_eq!(anonymous.to_string(), "filter 40001");
    }

    /// The error used to print `Some(9)` and `None` from a `Debug` of the
    /// index-type byte.
    #[test]
    fn an_unrecognized_index_error_has_no_rust_option_in_it() {
        let with_type = ChunkIndex::from_layout(4, Some(9)).unwrap_err().to_string();
        assert!(with_type.contains("index type=9"), "{with_type}");
        assert!(!with_type.contains("Some"), "{with_type}");

        let without_type = ChunkIndex::from_layout(9, None).unwrap_err().to_string();
        assert!(without_type.contains("no index type"), "{without_type}");
        assert!(!without_type.contains("None"), "{without_type}");
    }
}
