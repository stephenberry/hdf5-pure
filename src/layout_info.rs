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
    /// This is the index [`AppendWriter`](crate::AppendWriter) and
    /// [`EditSession::append_dataset`](crate::EditSession::append_dataset) grow
    /// in place.
    ExtensibleArray,
    /// A version-2 B-tree indexes the chunks (several unlimited dimensions). A
    /// dataset with this index is classified here, but enumerating its chunks
    /// with [`Dataset::chunks`](crate::Dataset::chunks) is not yet supported.
    BTreeV2,
}

impl ChunkIndex {
    /// Whether a dataset with this index kind can be grown in place with
    /// [`AppendWriter`](crate::AppendWriter) — true only for
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
            (v, idx) => {
                return Err(FormatError::ChunkedReadError(format!(
                    "unrecognized chunk index (layout version={v}, index type={idx:?})"
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
    /// 3 = fletcher32, 6 = scale-offset, 32013 = zfp. The same numbering
    /// returned by [`Dataset::filters`](crate::Dataset::filters).
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
