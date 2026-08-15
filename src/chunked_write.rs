//! Chunked dataset writing: chunk splitting, compression, index building.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};

use crate::checksum::jenkins_lookup3;
use core::num::NonZeroUsize;

use crate::convert::{TryToUsize, nonzero_usize_from};
use crate::error::FormatError;
use crate::extensible_array::{EaGeometry, ExtensibleArrayHeader};
#[cfg(feature = "zfp")]
use crate::filter_pipeline::FILTER_ZFP;
use crate::filter_pipeline::{
    FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_LZF, FILTER_SCALEOFFSET, FILTER_SHUFFLE,
    FilterDescription, FilterPipeline,
};
use crate::filters::{ChunkContext, ZfpElementTypeWhenEnabled, compress_chunk};
use crate::scaleoffset::{ScaleOffset, ScaleOffsetType, build_cd_values};

/// Log2 of the Fixed Array data-block page size (`2^10 = 1024` elements).
///
/// Single source of truth for the page exponent the writer emits: it is both
/// the `max_nelmts_bits` field stored in the Fixed Array header (FAHD) and the
/// `max_dblk_page_nelmts_bits` field in the v4 chunked layout message, which the
/// HDF5 spec requires to be equal. Above this many chunks the writer switches to
/// the paged data-block layout. The value mirrors the HDF5 C library's
/// `H5D_FARRAY_MAX_DBLK_PAGE_NELMTS_BITS`. The reader does not use this constant:
/// it honors whatever page size a file declares in its FAHD.
pub(crate) const FIXED_ARRAY_PAGE_BITS: u8 = 10;

/// Options for chunked dataset creation.
#[derive(Debug, Clone, Default)]
pub struct ChunkOptions {
    /// Chunk dimensions (one per dataset dimension).
    pub chunk_dims: Option<Vec<u64>>,
    /// Deflate compression level (0-9), None = no deflate.
    pub deflate_level: Option<u32>,
    /// Whether to apply shuffle filter before compression.
    pub shuffle: bool,
    /// Whether to apply the h5py LZF filter (id 32000). Mutually exclusive
    /// with deflate; ignored when ZFP is active (ZFP replaces byte
    /// compressors).
    pub lzf: bool,
    /// Whether to apply fletcher32 checksum.
    pub fletcher32: bool,
    /// ZFP fixed-rate compression (bits per value), None = no ZFP.
    /// When set, takes priority over shuffle + deflate.
    #[cfg(feature = "zfp")]
    pub zfp_rate: Option<f64>,
    /// Scale-offset compression mode, None = no scale-offset. When set it is
    /// the primary transform (mutually exclusive with ZFP, replaces shuffle)
    /// and may be followed by deflate.
    pub scale_offset: Option<ScaleOffset>,
}

impl ChunkOptions {
    /// Whether any chunking option is enabled.
    pub fn is_chunked(&self) -> bool {
        self.chunk_dims.is_some()
            || self.deflate_level.is_some()
            || self.shuffle
            || self.lzf
            || self.fletcher32
            || self.zfp_enabled()
            || self.scale_offset.is_some()
    }

    #[cfg(feature = "zfp")]
    #[inline]
    fn zfp_enabled(&self) -> bool {
        self.zfp_rate.is_some()
    }

    #[cfg(not(feature = "zfp"))]
    #[inline]
    fn zfp_enabled(&self) -> bool {
        false
    }

    /// Refuse a combination of filters where honoring one means discarding
    /// another.
    ///
    /// Two filters here are *primary transforms* that consume the raw elements
    /// and hand on something else: ZFP, and scale-offset. Each displaces
    /// whatever it sits on top of, so a request naming a displaced filter as
    /// well is a contradiction — the caller asked for something the file cannot
    /// end up containing.
    ///
    /// Every such contradiction is an error. Dropping the loser silently is the
    /// one option a caller cannot detect: nothing in the resulting file records
    /// that a filter was requested, so `with_shuffle().with_zfp(16.0)` produced
    /// an unshuffled dataset and no way to tell that from `with_zfp(16.0)`
    /// alone. Documented precedence is not a substitute, because a precedence
    /// rule still has to be read to be obeyed and there is nothing to read it
    /// against at the call site.
    ///
    /// Checked in one place, before any filter is built, so which contradiction
    /// gets reported does not depend on the order the pipeline happens to be
    /// assembled in, and so a filter added later inherits the rule rather than
    /// having to restate it.
    fn refuse_conflicting_filters(&self) -> Result<(), FormatError> {
        let clash = |a: &str, b: &str| {
            Err(FormatError::FilterError(format!(
                "{a} and {b} cannot be combined on one dataset"
            )))
        };

        if self.zfp_enabled() {
            if self.scale_offset.is_some() {
                return clash("scale-offset", "ZFP");
            }
            if self.shuffle {
                return clash("shuffle", "ZFP");
            }
            if self.lzf {
                return clash("lzf", "ZFP");
            }
            if self.deflate_level.is_some() {
                return clash("deflate", "ZFP");
            }
        }
        if self.scale_offset.is_some() && self.shuffle {
            return clash("shuffle", "scale-offset");
        }
        // Not a primary transform, but the same shape: LZF and deflate fill one
        // byte-compressor slot, and stacking two of them is never useful.
        if self.lzf && self.deflate_level.is_some() {
            return clash("lzf", "deflate");
        }
        Ok(())
    }

    /// Build a FilterPipeline from the options.
    ///
    /// `chunk_dims` and `zfp_element_type` are only consulted when the ZFP
    /// filter is active — they're embedded into the ZFP cd_values so the
    /// resulting file is readable by the reference H5Z-ZFP plugin.
    ///
    /// Returns [`FormatError::UnsupportedZfp`] when ZFP was requested but
    /// `zfp_element_type` is `None` (e.g. the dataset's datatype isn't one of
    /// f32/f64/i32/i64), or the chunk rank is outside 1..=4, and
    /// [`FormatError::FilterError`] for a combination of filters where one
    /// would displace another — see [`refuse_conflicting_filters`].
    ///
    /// [`refuse_conflicting_filters`]: Self::refuse_conflicting_filters
    pub fn build_pipeline(
        &self,
        element_size: u32,
        chunk_dims: &[u64],
        zfp_element_type: Option<ZfpElementTypeWhenEnabled>,
        scale_offset_type: Option<ScaleOffsetType>,
    ) -> Result<Option<FilterPipeline>, FormatError> {
        self.refuse_conflicting_filters()?;

        let mut filters = Vec::new();
        let _ = zfp_element_type; // used only under the `zfp` feature below

        // ZFP is a standalone compressor: `refuse_conflicting_filters` has
        // already established that nothing it would displace was asked for.
        #[cfg(feature = "zfp")]
        if let Some(rate) = self.zfp_rate {
            let elem_ty = zfp_element_type.ok_or_else(|| {
                FormatError::UnsupportedZfp(
                    "ZFP compression requires the dataset's datatype to be one \
                     of f32, f64, i32, or i64"
                        .into(),
                )
            })?;
            filters.push(FilterDescription {
                filter_id: FILTER_ZFP,
                name: Some("zfp".into()),
                flags: 0,
                client_data: crate::zfp::zfp_cd_values_rate(rate, elem_ty, chunk_dims)?,
            });
        }

        // Scale-offset is also a primary transform: it displaces shuffle, but
        // may be followed by a byte compressor (pushed first so the pipeline
        // order is [scaleoffset, lzf|deflate]).
        if let Some(mode) = self.scale_offset {
            let ty = scale_offset_type.ok_or_else(|| {
                FormatError::FilterError(
                    "scale-offset requires an integer or floating-point scalar \
                     datatype with a definite (little/big endian) byte order"
                        .into(),
                )
            })?;
            let nelmts = u32::try_from(chunk_dims.iter().product::<u64>()).map_err(|_| {
                FormatError::FilterError("scale-offset: chunk has too many elements".into())
            })?;
            filters.push(FilterDescription {
                filter_id: FILTER_SCALEOFFSET,
                name: None,
                flags: 0,
                client_data: build_cd_values(mode, ty, element_size, nelmts)?,
            });
        }

        if self.shuffle {
            filters.push(FilterDescription {
                filter_id: FILTER_SHUFFLE,
                name: None,
                flags: 0,
                client_data: vec![element_size],
            });
        }

        // LZF fills the same byte-compressor slot as deflate; h5py's convention
        // is shuffle then lzf.
        if self.lzf {
            filters.push(FilterDescription {
                filter_id: FILTER_LZF,
                // Ids >= 256 serialize a name; "lzf" is h5py's registered name.
                name: Some("lzf".into()),
                // Optional (bit 0), which is what h5py records. Unlike every
                // other filter here, LZF *can* fail: liblzf returns 0 for a
                // chunk it cannot shrink, and h5py's filter relies on the
                // optional flag to store that chunk raw with its filter-mask
                // bit set. A mandatory LZF makes that a hard error, so h5py
                // cannot write incompressible data back into a file we wrote.
                // Our own writer still applies LZF unconditionally (a grown
                // stream is a valid stream), so it never sets a mask bit; the
                // flag exists for the writers that come after us.
                flags: 1,
                client_data: crate::lzf::h5py_cd_values(element_size, chunk_dims).to_vec(),
            });
        }

        if let Some(level) = self.deflate_level {
            filters.push(FilterDescription {
                filter_id: FILTER_DEFLATE,
                name: None,
                flags: 0,
                client_data: vec![level],
            });
        }

        if self.fletcher32 {
            filters.push(FilterDescription {
                filter_id: FILTER_FLETCHER32,
                name: None,
                flags: 0,
                client_data: vec![],
            });
        }

        // Note: h5py marks every filter optional (flags=0x0001); we match it
        // only on LZF, the one filter here whose compressor can decline a
        // chunk. For a filter that cannot fail the flag is unobservable, and
        // leaving those at 0 keeps our bytes stable against existing fixtures.

        if filters.is_empty() {
            Ok(None)
        } else {
            Ok(Some(FilterPipeline {
                version: 2,
                filters,
            }))
        }
    }

    /// Determine chunk dimensions, using user-specified or auto-computing.
    pub fn resolve_chunk_dims(&self, shape: &[u64]) -> Vec<u64> {
        if let Some(ref dims) = self.chunk_dims {
            dims.clone()
        } else {
            // Auto chunk: use the full dataset shape (single chunk)
            shape.to_vec()
        }
    }

    /// Validate the chunk geometry of a dataset that will use chunked storage,
    /// against its `shape` and optional `maxshape`. Returns a static reason on
    /// the first problem; callers map it to their own error type. Only
    /// meaningful when the dataset is actually chunked
    /// ([`is_chunked`](Self::is_chunked) or a `maxshape` is set).
    ///
    /// These checks turn what would otherwise be a panic deep in the chunk
    /// splitter ([`split_into_chunks`], which indexes `chunk_dims` by the shape's
    /// rank and divides by each chunk dimension) — or a silently corrupt,
    /// unreadable dataset — into an up-front, descriptive refusal. A
    /// zero-element shape (e.g. `[0]` for an empty extensible dataset) is allowed:
    /// it is not scalar and produces zero chunks, which is well-formed.
    pub fn validate_geometry(
        &self,
        shape: &[u64],
        maxshape: Option<&[u64]>,
    ) -> Result<(), &'static str> {
        if shape.is_empty() {
            return Err("a scalar dataset cannot be chunked, filtered, or extensible");
        }
        // Explicit chunk dimensions must match the shape's rank and be non-zero;
        // a zero would divide-by-zero when counting chunks per dimension, and a
        // rank mismatch would index past the end of `chunk_dims`.
        if let Some(dims) = self.chunk_dims.as_deref() {
            if dims.len() != shape.len() {
                return Err("chunk dimensions must have the same rank as the dataset shape");
            }
            if dims.contains(&0) {
                return Err("chunk dimensions must all be non-zero");
            }
        }
        // A maximum shape must match the rank and bound the current shape in
        // every dimension (an unlimited dimension, `u64::MAX`, bounds anything).
        if let Some(ms) = maxshape {
            if ms.len() != shape.len() {
                return Err("maxshape must have the same rank as the dataset shape");
            }
            if ms.iter().zip(shape).any(|(&m, &d)| m != u64::MAX && m < d) {
                return Err("maxshape must be at least the current shape in every dimension");
            }
        }
        Ok(())
    }
}

/// A chunk that has been written to the file buffer.
#[derive(Debug, Clone)]
pub struct WrittenChunk {
    /// Address within the file where chunk data starts.
    pub address: u64,
    /// Size of the (possibly compressed) chunk data in bytes.
    pub compressed_size: u64,
    /// Original uncompressed size in bytes.
    pub raw_size: u64,
    /// Filter mask (0 = all filters applied).
    pub filter_mask: u32,
}

/// Result of building a chunked dataset.
pub struct ChunkedDataResult {
    /// Raw bytes containing all chunk data + index structures.
    pub data_bytes: Vec<u8>,
    /// The DataLayout v4 message bytes.
    pub layout_message: Vec<u8>,
    /// The FilterPipeline message bytes, if any.
    pub pipeline_message: Option<Vec<u8>>,
}

/// Split raw data into chunk-sized pieces based on shape and chunk dimensions.
/// Returns a Vec of (chunk_offset_per_dim, chunk_raw_bytes).
pub fn split_into_chunks(
    raw_data: &[u8],
    shape: &[u64],
    chunk_dims: &[u64],
    element_size: NonZeroUsize,
) -> Vec<(Vec<u64>, Vec<u8>)> {
    let rank = shape.len();
    if rank == 0 {
        return vec![(vec![], raw_data.to_vec())];
    }

    // Compute number of chunks per dimension
    let mut num_chunks_per_dim = Vec::with_capacity(rank);
    for d in 0..rank {
        num_chunks_per_dim.push(shape[d].div_ceil(chunk_dims[d]));
    }
    let total_chunks: u64 = num_chunks_per_dim.iter().product();

    // Dataset strides (row-major)
    let mut ds_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "dataset dimension derived from the in-memory write request; bounded by addressable memory"
        )]
        let dim = shape[i + 1] as usize;
        ds_strides[i] = ds_strides[i + 1] * dim;
    }

    // Chunk strides
    let mut chunk_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "chunk dimension derived from the in-memory write request; bounded by addressable memory"
        )]
        let dim = chunk_dims[i + 1] as usize;
        chunk_strides[i] = chunk_strides[i + 1] * dim;
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "chunk/dataset dimensions derived from the in-memory write request; bounded by addressable memory"
    )]
    let (chunk_dims_us, shape_us): (Vec<usize>, Vec<usize>) = (
        chunk_dims.iter().map(|&d| d as usize).collect(),
        shape.iter().map(|&d| d as usize).collect(),
    );
    let chunk_total_elements: usize = chunk_dims_us.iter().product();

    #[expect(
        clippy::cast_possible_truncation,
        reason = "total_chunks derived from the in-memory write request; bounded by addressable memory"
    )]
    let mut result = Vec::with_capacity(total_chunks as usize);

    // Innermost dimension is contiguous in both the dataset (`raw_data`) and the
    // chunk buffer, so each in-bounds row is gathered with a single
    // `copy_from_slice`. Only the outer `rank - 1` dims are walked (odometer),
    // matching the read-side `copy_chunk_to_output` kernel.
    let inner = rank - 1;
    let mut coord = vec![0usize; inner];

    for linear_idx in 0..total_chunks {
        // Convert linear index to chunk grid coordinates and the chunk's
        // dataset-space offset.
        let mut chunk_grid_coords = vec![0u64; rank];
        let mut remaining = linear_idx;
        for d in (0..rank).rev() {
            chunk_grid_coords[d] = remaining % num_chunks_per_dim[d];
            remaining /= num_chunks_per_dim[d];
        }
        let offsets: Vec<u64> = (0..rank)
            .map(|d| chunk_grid_coords[d] * chunk_dims[d])
            .collect();
        #[expect(
            clippy::cast_possible_truncation,
            reason = "chunk offset derived from the in-memory write request; bounded by addressable memory"
        )]
        let offsets_us: Vec<usize> = offsets.iter().map(|&o| o as usize).collect();

        let mut chunk_bytes = vec![0u8; chunk_total_elements * element_size.get()];

        // In-bounds run length along the contiguous innermost dimension.
        let inner_row_len =
            chunk_dims_us[inner].min(shape_us[inner].saturating_sub(offsets_us[inner]));
        if inner_row_len > 0 {
            let row_bytes = inner_row_len * element_size.get();
            let inner_src = offsets_us[inner] * ds_strides[inner];
            let outer_total: usize = chunk_dims_us[..inner].iter().product();
            for c in coord.iter_mut() {
                *c = 0;
            }
            for _ in 0..outer_total {
                let mut dst_base = 0usize;
                let mut src_base = inner_src;
                let mut in_bounds = true;
                for d in 0..inner {
                    dst_base += coord[d] * chunk_strides[d];
                    let global = offsets_us[d] + coord[d];
                    if global >= shape_us[d] {
                        in_bounds = false;
                        break;
                    }
                    src_base += global * ds_strides[d];
                }

                if in_bounds {
                    let src = src_base * element_size.get();
                    let dst = dst_base * element_size.get();
                    let mut avail = row_bytes.min(raw_data.len().saturating_sub(src));
                    avail -= avail % element_size;
                    if avail > 0 {
                        chunk_bytes[dst..dst + avail].copy_from_slice(&raw_data[src..src + avail]);
                    }
                }

                for d in (0..inner).rev() {
                    coord[d] += 1;
                    if coord[d] < chunk_dims_us[d] {
                        break;
                    }
                    coord[d] = 0;
                }
            }
        }

        result.push((offsets, chunk_bytes));
    }

    result
}

/// Serialize a v4 single chunk layout message.
fn serialize_v4_single_chunk(
    chunk_dims: &[u32],
    chunk_address: u64,
    filtered_size: Option<u64>,
    filter_mask: Option<u32>,
    offset_size: u8,
    element_size: u32,
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.push(4); // version
    buf.push(2); // class = chunked

    // flags: bit 0 = unknown meaning in some files, bit 1 = filters for single chunk
    let flags: u8 = if filtered_size.is_some() { 0x02 } else { 0x00 };
    buf.push(flags);

    // dimensionality = rank + 1 (chunk dims + element size dim)
    #[expect(
        clippy::cast_possible_truncation,
        reason = "rank written into the 1-byte dimensionality field selected for this file"
    )]
    let ndims = chunk_dims.len() as u8 + 1;
    buf.push(ndims);

    // dim_size_encoded_length: how many bytes per dimension
    // We need to figure out the minimum encoding width
    let max_dim = chunk_dims
        .iter()
        .map(|&d| d as u64)
        .chain(core::iter::once(element_size as u64))
        .max()
        .unwrap_or(1);
    let dim_encoded_len: u8 = if max_dim <= 0xFF {
        1
    } else if max_dim <= 0xFFFF {
        2
    } else {
        4
    };
    buf.push(dim_encoded_len);

    // dimension sizes (chunk dims + element size)
    for &d in chunk_dims {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "dimension written into the on-disk encoding width selected for this file"
        )]
        match dim_encoded_len {
            1 => buf.push(d as u8),
            2 => buf.extend_from_slice(&(d as u16).to_le_bytes()),
            4 => buf.extend_from_slice(&d.to_le_bytes()),
            _ => {}
        }
    }
    // Element size dimension
    #[expect(
        clippy::cast_possible_truncation,
        reason = "element size written into the on-disk encoding width selected for this file"
    )]
    match dim_encoded_len {
        1 => buf.push(element_size as u8),
        2 => buf.extend_from_slice(&(element_size as u16).to_le_bytes()),
        4 => buf.extend_from_slice(&element_size.to_le_bytes()),
        _ => {}
    }

    // chunk index type = 1 (single chunk)
    buf.push(1);

    // Index-specific fields
    if let (Some(fs), Some(fm)) = (filtered_size, filter_mask) {
        // filtered_size (length_size bytes)
        buf.extend_from_slice(&fs.to_le_bytes()); // 8 bytes for length_size=8
        buf.extend_from_slice(&fm.to_le_bytes()); // 4 bytes
    }

    // chunk address
    #[expect(
        clippy::cast_possible_truncation,
        reason = "chunk address written into the on-disk offset width selected for this file"
    )]
    match offset_size {
        4 => buf.extend_from_slice(&(chunk_address as u32).to_le_bytes()),
        8 => buf.extend_from_slice(&chunk_address.to_le_bytes()),
        _ => {}
    }

    buf
}

/// Serialize a v4 Fixed Array layout message.
fn serialize_v4_fixed_array(
    chunk_dims: &[u32],
    fixed_array_address: u64,
    offset_size: u8,
    element_size: u32,
    max_bits: u8,
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.push(4); // version
    buf.push(2); // class = chunked

    let flags: u8 = 0x00;
    buf.push(flags);

    #[expect(
        clippy::cast_possible_truncation,
        reason = "rank written into the 1-byte dimensionality field selected for this file"
    )]
    let ndims = chunk_dims.len() as u8 + 1;
    buf.push(ndims);

    let max_dim = chunk_dims
        .iter()
        .map(|&d| d as u64)
        .chain(core::iter::once(element_size as u64))
        .max()
        .unwrap_or(1);
    let dim_encoded_len: u8 = if max_dim <= 0xFF {
        1
    } else if max_dim <= 0xFFFF {
        2
    } else {
        4
    };
    buf.push(dim_encoded_len);

    for &d in chunk_dims {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "dimension written into the on-disk encoding width selected for this file"
        )]
        match dim_encoded_len {
            1 => buf.push(d as u8),
            2 => buf.extend_from_slice(&(d as u16).to_le_bytes()),
            4 => buf.extend_from_slice(&d.to_le_bytes()),
            _ => {}
        }
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "element size written into the on-disk encoding width selected for this file"
    )]
    match dim_encoded_len {
        1 => buf.push(element_size as u8),
        2 => buf.extend_from_slice(&(element_size as u16).to_le_bytes()),
        4 => buf.extend_from_slice(&element_size.to_le_bytes()),
        _ => {}
    }

    // chunk index type = 3 (Fixed Array)
    buf.push(3);

    // max_dblk_page_nelmts_bits — must match FAHD max_nelmts_bits
    buf.push(max_bits);

    // Fixed Array header address
    #[expect(
        clippy::cast_possible_truncation,
        reason = "fixed array header address written into the on-disk offset width selected for this file"
    )]
    match offset_size {
        4 => buf.extend_from_slice(&(fixed_array_address as u32).to_le_bytes()),
        8 => buf.extend_from_slice(&fixed_array_address.to_le_bytes()),
        _ => {}
    }

    buf
}

/// Build a complete Fixed Array at a known absolute address.
pub fn build_fixed_array_at(
    chunks: &[WrittenChunk],
    offset_size: u8,
    length_size: u8,
    has_filters: bool,
    fa_base_address: u64,
) -> Vec<u8> {
    let os = offset_size as usize;
    let num_elements = chunks.len();

    // For filtered chunks, compute chunk_size encoding width.
    // Must match the HDF5 C library's H5D_FARRAY_FILT_COMPUTE_CHUNK_SIZE_LEN macro:
    //   chunk_size_len = 1 + ((H5VM_log2_gen(chunk.size) + 8) / 8)
    // where chunk.size is the unfiltered chunk size in bytes (product of all chunk dims).
    let chunk_size_bytes: usize = if has_filters {
        let max_raw = chunks.iter().map(|c| c.raw_size).max().unwrap_or(1);
        let log2_val = if max_raw <= 1 {
            0
        } else {
            63 - max_raw.leading_zeros()
        };
        let len = 1 + ((log2_val + 8) / 8) as usize;
        len.min(8)
    } else {
        0
    };

    let elem_size = if has_filters {
        os + chunk_size_bytes + 4
    } else {
        os
    };

    let client_id: u8 = if has_filters { 1 } else { 0 };

    // FAHD total size
    let nelmts_field_size = length_size as usize;
    let fahd_total_size = 4 + 1 + 1 + 1 + 1 + nelmts_field_size + os + 4;
    let fadb_address = fa_base_address + fahd_total_size as u64;

    // Build FAHD
    let mut fahd = Vec::with_capacity(fahd_total_size);
    fahd.extend_from_slice(b"FAHD");
    fahd.push(0); // version
    fahd.push(client_id);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "element record size written into the 1-byte FAHD field selected for this file"
    )]
    fahd.push(elem_size as u8);

    let max_bits = FIXED_ARRAY_PAGE_BITS;
    fahd.push(max_bits);

    #[expect(
        clippy::cast_possible_truncation,
        reason = "element count written into the on-disk length width selected for this file"
    )]
    match length_size {
        4 => fahd.extend_from_slice(&(num_elements as u32).to_le_bytes()),
        8 => fahd.extend_from_slice(&(num_elements as u64).to_le_bytes()),
        _ => fahd.extend_from_slice(&(num_elements as u64).to_le_bytes()),
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "FADB address written into the on-disk offset width selected for this file"
    )]
    match offset_size {
        4 => fahd.extend_from_slice(&(fadb_address as u32).to_le_bytes()),
        8 => fahd.extend_from_slice(&fadb_address.to_le_bytes()),
        _ => fahd.extend_from_slice(&fadb_address.to_le_bytes()),
    }

    // Checksum
    let checksum = jenkins_lookup3(&fahd);
    fahd.extend_from_slice(&checksum.to_le_bytes());

    debug_assert_eq!(fahd.len(), fahd_total_size);

    // Append one element record (chunk address, plus filtered size + mask).
    let write_element = |buf: &mut Vec<u8>, chunk: &WrittenChunk| {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "chunk address written into the on-disk offset width selected for this file"
        )]
        match offset_size {
            4 => buf.extend_from_slice(&(chunk.address as u32).to_le_bytes()),
            _ => buf.extend_from_slice(&chunk.address.to_le_bytes()),
        }
        if has_filters {
            // Compressed size, written using the variable chunk_size_bytes width.
            let cs_bytes = chunk.compressed_size.to_le_bytes();
            buf.extend_from_slice(&cs_bytes[..chunk_size_bytes]);
            buf.extend_from_slice(&chunk.filter_mask.to_le_bytes());
        }
    };

    // Build FADB prefix: signature + version + client_id + header address.
    let mut fadb = Vec::new();
    fadb.extend_from_slice(b"FADB");
    fadb.push(0); // version
    fadb.push(client_id);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "fixed array base address written into the on-disk offset width selected for this file"
    )]
    match offset_size {
        4 => fadb.extend_from_slice(&(fa_base_address as u32).to_le_bytes()),
        _ => fadb.extend_from_slice(&fa_base_address.to_le_bytes()),
    }

    let page_size = 1usize << max_bits;
    if num_elements <= page_size {
        // Non-paged: elements stored directly, then a single checksum.
        for chunk in chunks {
            write_element(&mut fadb, chunk);
        }
        let fadb_checksum = jenkins_lookup3(&fadb);
        fadb.extend_from_slice(&fadb_checksum.to_le_bytes());
    } else {
        // Paged: a page-init bitmap and checksum follow the prefix, then each
        // page stores its elements followed by its own checksum. We write every
        // chunk densely, so all pages are initialized.
        let npages = num_elements.div_ceil(page_size);
        let bitmap_size = npages.div_ceil(8);
        let mut bitmap = vec![0u8; bitmap_size];
        for page in 0..npages {
            // Most-significant-bit-first ordering, matching H5VM_bit_set.
            bitmap[page / 8] |= 1 << (7 - (page % 8));
        }
        fadb.extend_from_slice(&bitmap);
        let prefix_checksum = jenkins_lookup3(&fadb);
        fadb.extend_from_slice(&prefix_checksum.to_le_bytes());

        for page in 0..npages {
            let start = page * page_size;
            let end = core::cmp::min(start + page_size, num_elements);
            let mut page_buf = Vec::with_capacity((end - start) * elem_size);
            for chunk in &chunks[start..end] {
                write_element(&mut page_buf, chunk);
            }
            let page_checksum = jenkins_lookup3(&page_buf);
            page_buf.extend_from_slice(&page_checksum.to_le_bytes());
            fadb.extend_from_slice(&page_buf);
        }
    }

    let mut combined = fahd;
    combined.extend_from_slice(&fadb);
    combined
}

/// Serialize a v4 Extensible Array layout message.
pub(crate) fn serialize_v4_extensible_array(
    chunk_dims: &[u32],
    ea_address: u64,
    offset_size: u8,
    element_size: u32,
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.push(4); // version
    buf.push(2); // class = chunked
    buf.push(0x00); // flags

    #[expect(
        clippy::cast_possible_truncation,
        reason = "rank written into the 1-byte dimensionality field selected for this file"
    )]
    let ndims = chunk_dims.len() as u8 + 1;
    buf.push(ndims);

    let max_dim = chunk_dims
        .iter()
        .map(|&d| d as u64)
        .chain(core::iter::once(element_size as u64))
        .max()
        .unwrap_or(1);
    let dim_encoded_len: u8 = if max_dim <= 0xFF {
        1
    } else if max_dim <= 0xFFFF {
        2
    } else {
        4
    };
    buf.push(dim_encoded_len);

    for &d in chunk_dims {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "dimension written into the on-disk encoding width selected for this file"
        )]
        match dim_encoded_len {
            1 => buf.push(d as u8),
            2 => buf.extend_from_slice(&(d as u16).to_le_bytes()),
            4 => buf.extend_from_slice(&d.to_le_bytes()),
            _ => {}
        }
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "element size written into the on-disk encoding width selected for this file"
    )]
    match dim_encoded_len {
        1 => buf.push(element_size as u8),
        2 => buf.extend_from_slice(&(element_size as u16).to_le_bytes()),
        4 => buf.extend_from_slice(&element_size.to_le_bytes()),
        _ => {}
    }

    // chunk index type = 4 (Extensible Array)
    buf.push(4);

    // EA creation parameters (must match AEHD and HDF5 C library defaults)
    buf.push(32); // max_nelmts_bits
    buf.push(4); // idx_blk_elmts
    buf.push(4); // super_blk_min_data_ptrs
    buf.push(16); // data_blk_min_elmts
    buf.push(10); // max_dblk_page_nelmts_bits

    // EA header address
    #[expect(
        clippy::cast_possible_truncation,
        reason = "extensible array header address written into the on-disk offset width selected for this file"
    )]
    match offset_size {
        4 => buf.extend_from_slice(&(ea_address as u32).to_le_bytes()),
        8 => buf.extend_from_slice(&ea_address.to_le_bytes()),
        _ => {}
    }

    buf
}

/// Write an offset-sized address (little-endian) to `buf`.
pub(crate) fn write_ea_addr(buf: &mut Vec<u8>, val: u64, offset_size: u8) {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "address written into the on-disk offset width selected for this file"
    )]
    match offset_size {
        4 => buf.extend_from_slice(&(val as u32).to_le_bytes()),
        _ => buf.extend_from_slice(&val.to_le_bytes()),
    }
}

/// Build a single Extensible Array Data Block (`EADB`) holding the chunk
/// elements for `[elem_start, elem_start + dblk_nelmts)`. Slots whose absolute
/// element index reaches `num_elements` are written as undefined.
///
/// When `dblk_nelmts` exceeds the page size the block is *paged*: the header
/// carries its own checksum and the elements are split into contiguous pages of
/// `page_nelmts` slots, each followed by a checksum. Returns the block bytes and
/// the number of leading pages that contain at least one real element (used by
/// the owning super block to build its page-init bitmap). For non-paged blocks
/// the returned page count is 0.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_eadb(
    chunks: &[WrittenChunk],
    num_elements: usize,
    elem_start: usize,
    dblk_nelmts: usize,
    block_offset_rel: u64,
    ea_base_address: u64,
    offset_size: u8,
    has_filters: bool,
    chunk_size_bytes: usize,
    client_id: u8,
    page_nelmts: usize,
    blk_off_size: usize,
) -> (Vec<u8>, usize) {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"EADB");
    buf.push(0); // version
    buf.push(client_id);
    write_ea_addr(&mut buf, ea_base_address, offset_size);
    buf.extend_from_slice(&block_offset_rel.to_le_bytes()[..blk_off_size]);

    if dblk_nelmts <= page_nelmts {
        // Non-paged: elements inline, single checksum.
        for slot in 0..dblk_nelmts {
            let idx = elem_start + slot;
            if idx < num_elements {
                write_chunk_element(
                    &mut buf,
                    &chunks[idx],
                    offset_size,
                    has_filters,
                    chunk_size_bytes,
                );
            } else {
                write_undefined_element(&mut buf, offset_size, has_filters, chunk_size_bytes);
            }
        }
        let cks = jenkins_lookup3(&buf);
        buf.extend_from_slice(&cks.to_le_bytes());
        (buf, 0)
    } else {
        // Paged: the header has its own checksum, then full pages follow. We
        // reserve every page (matching the C library's allocation) and report
        // how many leading pages hold real data so the super block can mark
        // them initialized in its bitmap.
        let header_cks = jenkins_lookup3(&buf);
        buf.extend_from_slice(&header_cks.to_le_bytes());

        let npages = dblk_nelmts / page_nelmts;
        let mut pages_init = 0usize;
        for page in 0..npages {
            let page_start = elem_start + page * page_nelmts;
            let mut page_buf = Vec::new();
            let mut has_real = false;
            for slot in 0..page_nelmts {
                let idx = page_start + slot;
                if idx < num_elements {
                    write_chunk_element(
                        &mut page_buf,
                        &chunks[idx],
                        offset_size,
                        has_filters,
                        chunk_size_bytes,
                    );
                    has_real = true;
                } else {
                    write_undefined_element(
                        &mut page_buf,
                        offset_size,
                        has_filters,
                        chunk_size_bytes,
                    );
                }
            }
            let page_cks = jenkins_lookup3(&page_buf);
            page_buf.extend_from_slice(&page_cks.to_le_bytes());
            buf.extend_from_slice(&page_buf);
            if has_real {
                pages_init += 1;
            }
        }
        (buf, pages_init)
    }
}

/// Build an Extensible Array Super (secondary) Block (`EASB`) referencing
/// `dblk_addrs`. When `page_bitmap` is non-empty the block's data blocks are
/// paged and the bitmap (already populated by the caller) is written between
/// the block offset and the data block addresses.
pub(crate) fn build_aesb(
    ea_base_address: u64,
    block_offset_rel: u64,
    page_bitmap: &[u8],
    dblk_addrs: &[u64],
    offset_size: u8,
    blk_off_size: usize,
    client_id: u8,
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"EASB");
    buf.push(0); // version
    buf.push(client_id);
    write_ea_addr(&mut buf, ea_base_address, offset_size);
    buf.extend_from_slice(&block_offset_rel.to_le_bytes()[..blk_off_size]);
    buf.extend_from_slice(page_bitmap);
    for &addr in dblk_addrs {
        write_ea_addr(&mut buf, addr, offset_size);
    }
    let cks = jenkins_lookup3(&buf);
    buf.extend_from_slice(&cks.to_le_bytes());
    buf
}

/// On-disk byte size of an Extensible Array index block (`EAIB`): the prefix
/// (signature, version, client id, header address), the always-written inline
/// element slots, the direct data-block and super-block address pointers, and a
/// trailing checksum. The single source of truth shared by the bulk writer
/// ([`build_extensible_array_at`]) and the in-place editor's reclaim walk, so
/// the two cannot disagree on how many bytes the index block occupies.
pub(crate) fn aeib_size(
    offset_size: u8,
    inline_elmts: usize,
    elem_size: usize,
    ndblk_addrs: usize,
    nsblk_addrs: usize,
) -> usize {
    let os = offset_size as usize;
    4 + 1 + 1 + os // signature + version + client id + header address
        + inline_elmts * elem_size // inline element slots (always all written)
        + ndblk_addrs * os // direct data-block addresses
        + nsblk_addrs * os // super-block addresses
        + 4 // checksum
}

/// The six Extensible Array header statistics, in the C library's stored order.
/// Read by the incremental append writer, and by [`ea_layout`], for which two of
/// them add up to the array's body length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EaStats {
    pub nsuper_blks: u64,
    pub super_blk_size: u64,
    pub ndata_blks: u64,
    pub data_blk_size: u64,
    pub max_idx_set: u64,
    pub nelmts: u64,
}

/// On-disk byte size of one non-paged Extensible Array data block (`EADB`)
/// holding `dblk_nelmts` element slots.
pub(crate) fn eadb_size(
    dblk_nelmts: u64,
    elem_size: usize,
    page_nelmts: u64,
    offset_size: u8,
    blk_off_size: usize,
) -> u64 {
    let prefix = 4 + 1 + 1 + offset_size as usize + blk_off_size;
    if dblk_nelmts <= page_nelmts {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "data block element count derived from the in-memory write request; bounded by addressable memory"
        )]
        let nelmts = dblk_nelmts as usize;
        (prefix + nelmts * elem_size + 4) as u64
    } else {
        // Paged: header carries its own checksum, then full pages follow.
        let npages = dblk_nelmts / page_nelmts;
        (prefix + 4) as u64 + npages * (page_nelmts * elem_size as u64 + 4)
    }
}

/// On-disk byte size of one Extensible Array super block (`EASB`) with `ndblks`
/// data-block pointers and (when its data blocks are paged) a page-init bitmap.
pub(crate) fn aesb_size(
    ndblks: u64,
    dblk_nelmts: u64,
    page_nelmts: u64,
    offset_size: u8,
    blk_off_size: usize,
) -> u64 {
    let os = offset_size as usize;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "data block and page counts derived from the in-memory write request; bounded by addressable memory"
    )]
    let bitmap = if dblk_nelmts > page_nelmts {
        let npages = dblk_nelmts / page_nelmts;
        ndblks as usize * npages.div_ceil(8) as usize
    } else {
        0
    };
    #[expect(
        clippy::cast_possible_truncation,
        reason = "data block count derived from the in-memory write request; bounded by addressable memory"
    )]
    let ndblks_usize = ndblks as usize;
    (4 + 1 + 1 + os + blk_off_size + bitmap + ndblks_usize * os + 4) as u64
}

/// Compute the six Extensible Array header statistics for an array holding
/// `num_elements` densely-filled elements. Mirrors the allocation performed by
/// [`build_extensible_array_at`] so the bulk writer and the incremental append
/// writer always agree (asserted by a unit test).
///
/// The two size statistics are also what [`extensible_array_len`] reports, since
/// the array's body is its allocated blocks and nothing else — so this walk
/// decides a reservation as well as a set of header fields.
pub(crate) fn ea_compute_stats(
    geom: &EaGeometry,
    idx_blk_elmts: u64,
    elem_size: usize,
    page_nelmts: u64,
    offset_size: u8,
    blk_off_size: usize,
    num_elements: u64,
) -> EaStats {
    let mut s = EaStats {
        nsuper_blks: 0,
        super_blk_size: 0,
        ndata_blks: 0,
        data_blk_size: 0,
        max_idx_set: num_elements,
        nelmts: idx_blk_elmts,
    };
    let mut elem = idx_blk_elmts;
    for &dn in &geom.direct_dblk_nelmts {
        if elem < num_elements {
            s.ndata_blks += 1;
            s.data_blk_size += eadb_size(dn, elem_size, page_nelmts, offset_size, blk_off_size);
            s.nelmts += dn;
        }
        elem += dn;
    }
    for j in 0..geom.nsblk_addrs {
        let (ndblks, dn) = geom.sblks[geom.first_indirect_sblk + j];
        let span = ndblks * dn;
        if elem < num_elements {
            s.nsuper_blks += 1;
            s.super_blk_size += aesb_size(ndblks, dn, page_nelmts, offset_size, blk_off_size);
            let mut le = elem;
            for _ in 0..ndblks {
                if le < num_elements {
                    s.ndata_blks += 1;
                    s.data_blk_size +=
                        eadb_size(dn, elem_size, page_nelmts, offset_size, blk_off_size);
                    s.nelmts += dn;
                }
                le += dn;
            }
        }
        elem += span;
    }
    s
}

/// Everything about an Extensible Array that does not depend on where it is
/// placed: the element encoding, the block geometry, and every byte count.
///
/// [`build_extensible_array_at`] writes an array's bytes at a chosen base
/// address, but no term of this layout is that address — each block's size, and
/// so the array's total length, is the same for every base. That is what lets
/// [`extensible_array_len`] answer the length without emitting the array.
///
/// The builder shares this layout's *geometry*, so no second derivation of the
/// block sizes can drift from what is written. It does not share the walk that
/// decides which of those blocks a given element count allocates: that is
/// written once here (through [`ea_compute_stats`]) and once in the builder's
/// own body. Those two are what the length assertion at the end of the builder,
/// and `extensible_array_len_matches_what_it_builds`, hold together.
struct EaLayout {
    /// Byte size of one element record: an address, plus the compressed size and
    /// filter mask when the dataset is filtered.
    elem_size: usize,
    /// Width of the compressed-size field inside a filtered element record, sized
    /// to the largest raw chunk. Zero when the dataset is unfiltered.
    chunk_size_bytes: usize,
    client_id: u8,
    /// EA creation parameters — these must match the HDF5 C library defaults
    /// exactly, and are held here so the header writer and the size computation
    /// read the same values.
    max_nelmts_bits: u8,
    idx_blk_elmts: u8,
    min_dblk_nelmts: u8,
    super_blk_min_nelmts: u8,
    max_dblk_nelmts_bits: u8,
    geom: EaGeometry,
    page_nelmts: usize,
    blk_off_size: usize,
    /// Element slots held inline in the index block (`idx_blk_elmts`).
    inline: usize,
    aehd_size: usize,
    aeib_size: usize,
    /// The six header statistics, two of which (`data_blk_size` and
    /// `super_blk_size`) are exactly the body's byte length.
    stats: EaStats,
    /// Header + index block + body: the whole array.
    total_len: u64,
}

/// Lay out the Extensible Array that would hold `chunks`, without building it.
fn ea_layout(
    chunks: &[WrittenChunk],
    offset_size: u8,
    length_size: u8,
    has_filters: bool,
) -> EaLayout {
    let os = offset_size as usize;

    // Compute element encoding size (same logic as Fixed Array)
    let chunk_size_bytes: usize = if has_filters {
        let max_raw = chunks.iter().map(|c| c.raw_size).max().unwrap_or(1);
        let log2_val = if max_raw <= 1 {
            0
        } else {
            63 - max_raw.leading_zeros()
        };
        let len = 1 + ((log2_val + 8) / 8) as usize;
        len.min(8)
    } else {
        0
    };

    let elem_size = if has_filters {
        os + chunk_size_bytes + 4
    } else {
        os
    };

    let client_id: u8 = if has_filters { 1 } else { 0 };

    // EA creation parameters — must match the HDF5 C library defaults exactly.
    let max_nelmts_bits: u8 = 32;
    let idx_blk_elmts: u8 = 4;
    let min_dblk_nelmts: u8 = 16;
    let super_blk_min_nelmts: u8 = 4;
    let max_dblk_nelmts_bits: u8 = 10;

    // Derive the block-size geometry from the shared helper (single source of
    // truth shared with the reader).
    #[expect(
        clippy::cast_possible_truncation,
        reason = "element record size written into the 1-byte EA header field selected for this file"
    )]
    let geom_header = ExtensibleArrayHeader {
        client_id,
        element_size: elem_size as u8,
        max_nelmts_bits,
        idx_blk_elmts,
        min_dblk_nelmts,
        super_blk_min_nelmts,
        max_dblk_nelmts_bits,
        num_elements: 0,
        index_block_address: 0,
    };
    let geom = EaGeometry::from_header(&geom_header);
    let page_nelmts = 1usize << max_dblk_nelmts_bits;
    let blk_off_size = (max_nelmts_bits as usize).div_ceil(8);
    let inline = idx_blk_elmts as usize;

    let aehd_size = ExtensibleArrayHeader::serialized_size(offset_size, length_size);
    let aeib_size = aeib_size(
        offset_size,
        inline,
        elem_size,
        geom.direct_dblk_nelmts.len(),
        geom.nsblk_addrs,
    );

    // The body is the allocated data blocks and super blocks, concatenated with
    // nothing between them, so the two size statistics are its byte length.
    let stats = ea_compute_stats(
        &geom,
        idx_blk_elmts as u64,
        elem_size,
        page_nelmts as u64,
        offset_size,
        blk_off_size,
        chunks.len() as u64,
    );
    let total_len = (aehd_size + aeib_size) as u64 + stats.data_blk_size + stats.super_blk_size;

    EaLayout {
        elem_size,
        chunk_size_bytes,
        client_id,
        max_nelmts_bits,
        idx_blk_elmts,
        min_dblk_nelmts,
        super_blk_min_nelmts,
        max_dblk_nelmts_bits,
        geom,
        page_nelmts,
        blk_off_size,
        inline,
        aehd_size,
        aeib_size,
        stats,
        total_len,
    }
}

/// The byte length [`build_extensible_array_at`] would produce for `chunks`,
/// without building it.
///
/// A caller that has to reserve space for the array before it exists — the
/// in-place editor placing one into freed space — needs the length first. It
/// comes from the same [`EaLayout`] the builder emits from, so no second
/// derivation of the block geometry can drift away from what is written.
pub(crate) fn extensible_array_len(
    chunks: &[WrittenChunk],
    offset_size: u8,
    length_size: u8,
    has_filters: bool,
) -> u64 {
    ea_layout(chunks, offset_size, length_size, has_filters).total_len
}

/// Build a complete Extensible Array at a known absolute address.
///
/// Lays out the header (`EAHD`), index block (`EAIB`), and — for datasets with
/// more than `idx_blk_elmts + sum(direct data blocks)` chunks — the on-disk
/// super blocks (`EASB`) and their data blocks (`EADB`, paged when large). The
/// super-block / data-block size progression comes from the shared
/// [`EaGeometry`], so the writer and reader cannot drift. Byte-for-byte
/// compatible with the reference HDF5 C library across inline, direct, super
/// block, and paged ranges (verified by crosscheck tests).
pub fn build_extensible_array_at(
    chunks: &[WrittenChunk],
    offset_size: u8,
    length_size: u8,
    has_filters: bool,
    ea_base_address: u64,
) -> Result<Vec<u8>, FormatError> {
    let num_elements = chunks.len();

    let layout = ea_layout(chunks, offset_size, length_size, has_filters);
    let EaLayout {
        elem_size,
        chunk_size_bytes,
        client_id,
        max_nelmts_bits,
        idx_blk_elmts,
        min_dblk_nelmts,
        super_blk_min_nelmts,
        max_dblk_nelmts_bits,
        ref geom,
        page_nelmts,
        blk_off_size,
        inline,
        aehd_size,
        aeib_size,
        ..
    } = layout;

    let aeib_address = ea_base_address + aehd_size as u64;
    let body_base = aeib_address + aeib_size as u64;

    let undef_addr: u64 = match offset_size {
        4 => 0xFFFF_FFFF,
        _ => u64::MAX,
    };

    // ---- Build the body (direct data blocks, then super blocks) -----------
    // Addresses are absolute, computed from `body_base`, so the body can be
    // built before the index block that references it.
    let mut body: Vec<u8> =
        Vec::with_capacity((layout.stats.data_blk_size + layout.stats.super_blk_size).to_usize()?);
    let mut direct_addrs: Vec<u64> = Vec::with_capacity(geom.direct_dblk_nelmts.len());
    let mut sblk_addrs: Vec<u64> = Vec::with_capacity(geom.nsblk_addrs);

    // Stats (match the C library's EAHD fields exactly).
    let mut ndata_blks: u64 = 0;
    let mut data_blk_size: u64 = 0;
    let mut nsuper_blks: u64 = 0;
    let mut super_blk_size: u64 = 0;
    let mut alloc_slots: u64 = inline as u64; // nelmts: idx slots + every allocated data block

    // Absolute element index past the inline slots. This walks the extensible
    // array's theoretical element space (up to `2^max_nelmts_bits` slots), which
    // exceeds a 32-bit `usize`, so it and the per-block spans are tracked in
    // `u64`; only the bounded, real-data values handed to the block builders are
    // narrowed (checked) to `usize`.
    let mut elem_cursor: u64 = inline as u64;

    // Direct data blocks: addresses stored directly in the index block.
    for &dblk_nelmts in &geom.direct_dblk_nelmts {
        if elem_cursor >= num_elements as u64 {
            direct_addrs.push(undef_addr);
            elem_cursor += dblk_nelmts;
            continue;
        }
        let addr = body_base + body.len() as u64;
        let (db_bytes, _) = build_eadb(
            chunks,
            num_elements,
            elem_cursor.to_usize()?,
            dblk_nelmts.to_usize()?,
            elem_cursor - inline as u64,
            ea_base_address,
            offset_size,
            has_filters,
            chunk_size_bytes,
            client_id,
            page_nelmts,
            blk_off_size,
        );
        ndata_blks += 1;
        data_blk_size += db_bytes.len() as u64;
        alloc_slots += dblk_nelmts;
        body.extend_from_slice(&db_bytes);
        direct_addrs.push(addr);
        elem_cursor += dblk_nelmts;
    }

    // Super blocks: addresses stored in the index block; super-block pointer `j`
    // refers to super block `first_indirect_sblk + j`.
    for j in 0..geom.nsblk_addrs {
        let sblk_idx = geom.first_indirect_sblk + j;
        // `ndblks` and `dblk_nelmts` are u64 element counts from the EA geometry.
        // Their product (this super block's element span) and the running cursor
        // walk the array's theoretical address space and can exceed a 32-bit
        // usize, so they stay in u64; only bounded, real-data counts are narrowed.
        let (ndblks, dblk_nelmts) = geom.sblks[sblk_idx];
        let sb_span = ndblks * dblk_nelmts;
        if elem_cursor >= num_elements as u64 {
            sblk_addrs.push(undef_addr);
            elem_cursor += sb_span;
            continue;
        }

        // Past the early-out this super block holds real data, so its block
        // counts are bounded by the (usize) chunk count and narrow safely.
        let is_paged = dblk_nelmts > page_nelmts as u64;
        let npages = if is_paged {
            dblk_nelmts / page_nelmts as u64
        } else {
            0
        };
        let sb_block_offset = elem_cursor - inline as u64;
        let bitmap_size = if is_paged {
            (ndblks * npages.div_ceil(8)).to_usize()?
        } else {
            0
        };
        let mut page_bitmap = vec![0u8; bitmap_size];

        let mut sb_dblk_addrs: Vec<u64> = Vec::with_capacity(ndblks.to_usize()?);
        let mut local_elem = elem_cursor;
        for db_local in 0..ndblks {
            if local_elem >= num_elements as u64 {
                sb_dblk_addrs.push(undef_addr);
                local_elem += dblk_nelmts;
                continue;
            }
            let addr = body_base + body.len() as u64;
            let (db_bytes, pages_init) = build_eadb(
                chunks,
                num_elements,
                local_elem.to_usize()?,
                dblk_nelmts.to_usize()?,
                local_elem - inline as u64,
                ea_base_address,
                offset_size,
                has_filters,
                chunk_size_bytes,
                client_id,
                page_nelmts,
                blk_off_size,
            );
            ndata_blks += 1;
            data_blk_size += db_bytes.len() as u64;
            alloc_slots += dblk_nelmts;
            body.extend_from_slice(&db_bytes);
            sb_dblk_addrs.push(addr);
            if is_paged {
                for p in 0..pages_init {
                    let global_page = (db_local * npages).to_usize()? + p;
                    page_bitmap[global_page / 8] |= 0x80 >> (global_page % 8);
                }
            }
            local_elem += dblk_nelmts;
        }

        let aesb_addr = body_base + body.len() as u64;
        let aesb = build_aesb(
            ea_base_address,
            sb_block_offset,
            &page_bitmap,
            &sb_dblk_addrs,
            offset_size,
            blk_off_size,
            client_id,
        );
        nsuper_blks += 1;
        super_blk_size += aesb.len() as u64;
        body.extend_from_slice(&aesb);
        sblk_addrs.push(aesb_addr);

        elem_cursor += sb_span;
    }

    // ---- Build the header (EAHD) ------------------------------------------
    #[expect(
        clippy::cast_possible_truncation,
        reason = "statistic written into the on-disk length width selected for this file"
    )]
    let write_length = |buf: &mut Vec<u8>, val: u64| match length_size {
        4 => buf.extend_from_slice(&(val as u32).to_le_bytes()),
        _ => buf.extend_from_slice(&val.to_le_bytes()),
    };

    let mut aehd = Vec::with_capacity(aehd_size);
    aehd.extend_from_slice(b"EAHD");
    aehd.push(0); // version
    aehd.push(client_id);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "element record size written into the 1-byte EA header field selected for this file"
    )]
    aehd.push(elem_size as u8);
    aehd.push(max_nelmts_bits);
    aehd.push(idx_blk_elmts);
    aehd.push(min_dblk_nelmts);
    aehd.push(super_blk_min_nelmts);
    aehd.push(max_dblk_nelmts_bits);

    // 6 statistics, in the C library's order:
    //   [0] nsuper_blks   [1] super_blk_size   [2] ndata_blks
    //   [3] data_blk_size [4] max_idx_set      [5] nelmts
    write_length(&mut aehd, nsuper_blks);
    write_length(&mut aehd, super_blk_size);
    write_length(&mut aehd, ndata_blks);
    write_length(&mut aehd, data_blk_size);
    write_length(&mut aehd, num_elements as u64); // max_idx_set (dense fill)
    write_length(&mut aehd, alloc_slots); // nelmts (allocated slots)

    write_ea_addr(&mut aehd, aeib_address, offset_size);

    let aehd_checksum = jenkins_lookup3(&aehd);
    aehd.extend_from_slice(&aehd_checksum.to_le_bytes());
    debug_assert_eq!(aehd.len(), aehd_size);

    // ---- Build the index block (EAIB) -------------------------------------
    let mut aeib = Vec::with_capacity(aeib_size);
    aeib.extend_from_slice(b"EAIB");
    aeib.push(0); // version
    aeib.push(client_id);
    write_ea_addr(&mut aeib, ea_base_address, offset_size);

    // Inline elements (always write idx_blk_elmts slots; fill unused as undefined).
    #[allow(clippy::needless_range_loop)]
    for i in 0..inline {
        if i < num_elements {
            write_chunk_element(
                &mut aeib,
                &chunks[i],
                offset_size,
                has_filters,
                chunk_size_bytes,
            );
        } else {
            write_undefined_element(&mut aeib, offset_size, has_filters, chunk_size_bytes);
        }
    }
    // Direct data block addresses, then super block addresses.
    for &addr in &direct_addrs {
        write_ea_addr(&mut aeib, addr, offset_size);
    }
    for &addr in &sblk_addrs {
        write_ea_addr(&mut aeib, addr, offset_size);
    }

    let aeib_checksum = jenkins_lookup3(&aeib);
    aeib.extend_from_slice(&aeib_checksum.to_le_bytes());
    debug_assert_eq!(aeib.len(), aeib_size);

    let mut combined = aehd;
    combined.extend_from_slice(&aeib);
    combined.extend_from_slice(&body);
    // The length `extensible_array_len` promises a caller reserving space for
    // this array, checked against the bytes actually produced. A reservation
    // that disagrees with the emission would place the next object on top of
    // this one, so pin it where it is emitted as well as in a test.
    debug_assert_eq!(
        combined.len() as u64,
        layout.total_len,
        "an extensible array must fill the length its layout promised"
    );
    Ok(combined)
}

fn write_chunk_element(
    buf: &mut Vec<u8>,
    chunk: &WrittenChunk,
    offset_size: u8,
    has_filters: bool,
    chunk_size_bytes: usize,
) {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "chunk address written into the on-disk offset width selected for this file"
    )]
    match offset_size {
        4 => buf.extend_from_slice(&(chunk.address as u32).to_le_bytes()),
        8 => buf.extend_from_slice(&chunk.address.to_le_bytes()),
        _ => buf.extend_from_slice(&chunk.address.to_le_bytes()),
    }
    if has_filters {
        let cs_bytes = chunk.compressed_size.to_le_bytes();
        buf.extend_from_slice(&cs_bytes[..chunk_size_bytes]);
        buf.extend_from_slice(&chunk.filter_mask.to_le_bytes());
    }
}

fn write_undefined_element(
    buf: &mut Vec<u8>,
    offset_size: u8,
    has_filters: bool,
    chunk_size_bytes: usize,
) {
    let os = offset_size as usize;
    buf.extend_from_slice(&vec![0xFF; os]);
    if has_filters {
        buf.extend_from_slice(&vec![0x00; chunk_size_bytes]);
        buf.extend_from_slice(&0u32.to_le_bytes());
    }
}

/// A chunked dataset's chunks already split and compressed — the expensive,
/// **address-independent** half of building a chunked layout. The compressed
/// bytes, the chunk-index choice, and the pipeline message do not depend on
/// where the data lands in the file; only the absolute addresses embedded in the
/// chunk index do. The writer sizes a dataset's object header in one pass and
/// emits its data in a later pass (it needs every prior object's size to know
/// this object's address), so it computes this set once and feeds it to
/// [`assemble_chunked_at`] twice — sizing at a dummy address, then emitting at
/// the real one — instead of recompressing the whole dataset each pass.
pub(crate) struct CompressedChunkSet {
    /// Per-chunk compressed bytes, in dense row-major grid order.
    compressed: Vec<Vec<u8>>,
    /// Per-chunk uncompressed size (a full chunk is stored at full size, edge
    /// overhang zero-filled, so these are all equal in practice).
    raw_sizes: Vec<u64>,
    chunk_dims_u32: Vec<u32>,
    element_size: NonZeroUsize,
    has_filters: bool,
    use_extensible: bool,
    pipeline_message: Option<Vec<u8>>,
}

/// Split `raw_data` into chunks and compress each one, producing the
/// address-independent [`CompressedChunkSet`]. This performs the dataset's only
/// pass of the filter pipeline (shuffle/deflate/ZFP/…); [`assemble_chunked_at`]
/// then lays the result out at a concrete address without recompressing.
pub(crate) fn compress_chunks(
    raw_data: &[u8],
    shape: &[u64],
    ctx: ChunkContext<'_>,
    options: &ChunkOptions,
    maxshape: Option<&[u64]>,
) -> Result<CompressedChunkSet, FormatError> {
    let chunk_dims = ctx.chunk_dims;
    let element_size = nonzero_usize_from(ctx.element_size)?;
    let pipeline = options.build_pipeline(
        ctx.element_size.get(),
        chunk_dims,
        ctx.element_type,
        ctx.scale_offset_type,
    )?;

    let chunks = split_into_chunks(raw_data, shape, chunk_dims, element_size);
    let num_chunks = chunks.len();
    let has_filters = pipeline.is_some();

    let mut compressed = Vec::with_capacity(num_chunks);
    let mut raw_sizes = Vec::with_capacity(num_chunks);
    for (_offsets, chunk_bytes) in chunks {
        raw_sizes.push(chunk_bytes.len() as u64);
        let c = if let Some(ref pl) = pipeline {
            compress_chunk(&chunk_bytes, pl, ctx)?
        } else {
            // No pipeline: the split already produced an owned chunk buffer;
            // move it into the set instead of cloning.
            chunk_bytes
        };
        compressed.push(c);
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "chunk dimensions written into the on-disk u32 dimension fields selected for this file"
    )]
    let chunk_dims_u32: Vec<u32> = chunk_dims.iter().map(|&d| d as u32).collect();

    Ok(CompressedChunkSet {
        compressed,
        raw_sizes,
        chunk_dims_u32,
        element_size,
        has_filters,
        use_extensible: maxshape.is_some_and(|ms| ms.contains(&u64::MAX)),
        pipeline_message: pipeline.as_ref().map(|pl| pl.serialize()),
    })
}

/// Where each of a chunk set's chunks lands when the set is laid out at
/// `base_address` — they are stored back to back from there — and the address the
/// chunk index follows them at.
fn plan_chunk_slots(set: &CompressedChunkSet, base_address: u64) -> (Vec<WrittenChunk>, u64) {
    let mut cursor = base_address;
    let mut written_chunks = Vec::with_capacity(set.compressed.len());
    for (chunk, &raw_size) in set.compressed.iter().zip(set.raw_sizes.iter()) {
        written_chunks.push(WrittenChunk {
            address: cursor,
            compressed_size: chunk.len() as u64,
            raw_size,
            filter_mask: 0,
        });
        cursor += chunk.len() as u64;
    }
    (written_chunks, cursor)
}

/// Build the chunk index that follows a chunk set's data at `index_address`,
/// returning the index bytes — empty for the single-chunk layout, whose chunk
/// address lives in the layout message instead — and the v4 data-layout message
/// naming it.
///
/// Every address either one embeds sits in a fixed-width field, so the returned
/// bytes have the same *length* for every `index_address`. That is what lets
/// [`chunked_data_len`] size a dataset before its address has been chosen, and
/// what lets the writer size an object header in one pass and emit it in a later
/// one.
fn chunk_index_bytes(
    set: &CompressedChunkSet,
    written_chunks: &[WrittenChunk],
    index_address: u64,
) -> Result<(Vec<u8>, Vec<u8>), FormatError> {
    let offset_size: u8 = 8;
    let length_size: u8 = 8;
    let has_filters = set.has_filters;

    #[expect(
        clippy::cast_possible_truncation,
        reason = "element size written into the on-disk u32 dimension field selected for this file"
    )]
    let (index, layout_message) = if set.use_extensible {
        let ea_bytes = build_extensible_array_at(
            written_chunks,
            offset_size,
            length_size,
            has_filters,
            index_address,
        )?;
        let layout = serialize_v4_extensible_array(
            &set.chunk_dims_u32,
            index_address,
            offset_size,
            set.element_size.get() as u32,
        );
        (ea_bytes, layout)
    } else if written_chunks.len() == 1 {
        let chunk = &written_chunks[0];
        let filtered_size = has_filters.then_some(chunk.compressed_size);
        let filter_mask = has_filters.then_some(0u32);
        let layout = serialize_v4_single_chunk(
            &set.chunk_dims_u32,
            chunk.address,
            filtered_size,
            filter_mask,
            offset_size,
            set.element_size.get() as u32,
        );
        (Vec::new(), layout)
    } else {
        let fa_bytes = build_fixed_array_at(
            written_chunks,
            offset_size,
            length_size,
            has_filters,
            index_address,
        );
        let layout = serialize_v4_fixed_array(
            &set.chunk_dims_u32,
            index_address,
            offset_size,
            set.element_size.get() as u32,
            FIXED_ARRAY_PAGE_BITS,
        );
        (fa_bytes, layout)
    };
    Ok((index, layout_message))
}

/// The exact byte length [`assemble_chunked_at`] produces for `set` — the same
/// at every base address, since the layout depends on the chunk sizes and the
/// index shape alone.
///
/// Sizing without assembling is what lets a caller pick the dataset's address
/// *first*: the in-place editor asks its free-space list for a region this long
/// and, if it gets one, assembles the set straight into it rather than growing
/// the file (issue #261).
pub(crate) fn chunked_data_len(set: &CompressedChunkSet) -> Result<u64, FormatError> {
    let (written_chunks, index_address) = plan_chunk_slots(set, 0);
    let (index, _layout) = chunk_index_bytes(set, &written_chunks, index_address)?;
    Ok(index_address + index.len() as u64)
}

/// Lay an already-[`compress`ed](compress_chunks) chunk set out at `base_address`,
/// producing the on-disk data region (chunk bytes followed by the chunk index)
/// and the v4 data-layout message. Cheap: this only concatenates and builds the
/// index, so it can be run more than once (different addresses) without
/// repeating the dataset's compression.
pub(crate) fn assemble_chunked_at(
    set: &CompressedChunkSet,
    base_address: u64,
) -> Result<ChunkedDataResult, FormatError> {
    let (written_chunks, index_address) = plan_chunk_slots(set, base_address);
    let (index, layout_message) = chunk_index_bytes(set, &written_chunks, index_address)?;

    // One exact allocation for chunks plus index: the buffer is filled to its
    // capacity, never doubled and copied.
    let chunk_bytes_total: usize = set.compressed.iter().map(Vec::len).sum();
    let mut data_buf = Vec::with_capacity(chunk_bytes_total + index.len());
    for chunk in &set.compressed {
        data_buf.extend_from_slice(chunk);
    }
    data_buf.extend_from_slice(&index);

    Ok(ChunkedDataResult {
        data_bytes: data_buf,
        layout_message,
        pipeline_message: set.pipeline_message.clone(),
    })
}

/// Build chunked data with absolute addresses and optional maxshape.
///
/// Convenience composition of [`compress_chunks`] + [`assemble_chunked_at`] for
/// tests that build a single chunked dataset at a known address in one shot.
/// Production callers keep the [`CompressedChunkSet`] between passes instead, so
/// they compress each dataset only once: the file writer sizes an object header
/// before it emits the data, and the in-place editor sizes the dataset
/// ([`chunked_data_len`]) before it chooses the address to assemble it at.
///
/// `ctx` carries chunk_dims, element_size, and (for type-aware filters like
/// ZFP) the scalar element type. Build it via [`ChunkContext::from_datatype`]
/// when a `Datatype` is in scope.
#[cfg(test)]
pub fn build_chunked_data_at_ext(
    raw_data: &[u8],
    shape: &[u64],
    ctx: ChunkContext<'_>,
    options: &ChunkOptions,
    base_address: u64,
    maxshape: Option<&[u64]>,
) -> Result<ChunkedDataResult, FormatError> {
    let set = compress_chunks(raw_data, shape, ctx, options, maxshape)?;
    assemble_chunked_at(&set, base_address)
}

/// Per-chunk metadata in dense row-major grid order — enough to compute the
/// destination layout (chunk addresses and index structures) *without* the
/// chunk bytes. `compressed_size` is the exact byte count the
/// matching [`ChunkProvider::chunk_bytes`] call must return.
#[derive(Debug, Clone)]
pub(crate) struct ChunkMeta {
    /// Compressed on-disk size of this chunk, in bytes.
    pub(crate) compressed_size: u64,
    /// The chunk's filter mask from the source index, carried through verbatim.
    pub(crate) filter_mask: u32,
}

/// Yields one chunk's already-compressed bytes on demand. Called once per grid
/// slot, in ascending slot order, during the streaming assembly pass — so a
/// repacked dataset never holds more than a single chunk's bytes at a time.
///
/// `Send + Sync` is required so that a [`DatasetBuilder`](crate::type_builders::DatasetBuilder)
/// holding a boxed provider — and thus the public `FileBuilder` — keeps its
/// `Send`/`Sync` auto-traits. Real providers own an `Arc<File>`, which is both.
pub(crate) trait ChunkProvider: Send + Sync {
    /// Append grid slot `index`'s compressed bytes to `out`, which the emitter
    /// hands over empty. It is the same buffer on every call, so an
    /// implementation that appends costs one allocation for the whole dataset
    /// rather than one per chunk. The resulting length must equal the matching
    /// [`ChunkMeta::compressed_size`]; the emitter checks it.
    fn chunk_bytes(&self, index: usize, out: &mut Vec<u8>) -> Result<(), FormatError>;
}

/// A minimal byte sink so the verbatim chunk emitter works against both an
/// in-memory `Vec<u8>` (the buffered / `no_std` path) and a streaming
/// `std::io::Write` (the out-of-core path), without pulling `std::io` into
/// `no_std` builds.
pub(crate) trait ByteSink {
    /// Append `bytes` to the output.
    fn put(&mut self, bytes: &[u8]) -> Result<(), FormatError>;
    /// Append `n` zero bytes.
    fn put_zeros(&mut self, n: usize) -> Result<(), FormatError>;
    /// Total bytes written so far (used to assert layout addresses on a
    /// non-seekable sink).
    fn position(&self) -> u64;
    /// Hint that `additional` more bytes are about to be written. Lets a buffered
    /// (`Vec`) sink preallocate the whole file in one shot, as the writer did
    /// before streaming. A no-op for sinks that do not benefit (e.g. a streaming
    /// `Write`).
    fn reserve(&mut self, _additional: usize) {}
}

impl ByteSink for Vec<u8> {
    fn put(&mut self, bytes: &[u8]) -> Result<(), FormatError> {
        self.extend_from_slice(bytes);
        Ok(())
    }
    fn put_zeros(&mut self, n: usize) -> Result<(), FormatError> {
        self.resize(self.len() + n, 0u8);
        Ok(())
    }
    fn position(&self) -> u64 {
        self.len() as u64
    }
    fn reserve(&mut self, additional: usize) {
        Vec::reserve(self, additional);
    }
}

/// One grid slot's placement in the data region. Slots are stored back to back,
/// so a slot's own compressed byte count is its whole placement: the next slot
/// begins where this one ends.
pub(crate) struct ChunkSlotPlan {
    pub(crate) compressed_size: u64,
}

/// The full destination layout of a verbatim chunked dataset's data region,
/// computed from chunk *sizes* alone (no chunk bytes). Feeds both the object
/// header (via the separately returned layout/pipeline messages) and the
/// streaming emit ([`emit_chunked_data_verbatim`]).
pub(crate) struct VerbatimPlan {
    /// One entry per grid slot, in ascending address order.
    pub(crate) slots: Vec<ChunkSlotPlan>,
    /// The serialized chunk-index structure (Fixed Array / Extensible Array),
    /// emitted after the chunk bytes. Empty for the single-chunk layout (whose
    /// address is embedded in the layout message instead).
    pub(crate) index_tail: Vec<u8>,
    /// Total byte length of the data region (chunks + index tail).
    pub(crate) total_len: u64,
}

/// The result of planning a verbatim chunked dataset: the data-region
/// [`VerbatimPlan`] (for the streaming emit) plus the object-header messages it
/// implies (the v4 layout message and the verbatim pipeline message).
pub(crate) struct VerbatimLayout {
    pub(crate) plan: VerbatimPlan,
    pub(crate) layout_message: Vec<u8>,
    pub(crate) pipeline_message: Option<Vec<u8>>,
}

/// Compute the [`VerbatimLayout`] (data-region plan plus the v4 layout and
/// verbatim pipeline messages) for a dense, grid-ordered set of chunks, from
/// their sizes and filter masks alone — no chunk bytes. The byte layout is
/// identical whether the chunks are later buffered or streamed.
pub(crate) fn plan_chunked_data_verbatim(
    meta: &[ChunkMeta],
    chunk_dims: &[u64],
    element_size: NonZeroUsize,
    raw_size: u64,
    pipeline_message: Option<&[u8]>,
    base_address: u64,
    maxshape: Option<&[u64]>,
) -> Result<VerbatimLayout, FormatError> {
    if meta.is_empty() {
        return Err(FormatError::ChunkedReadError(
            "a verbatim chunked dataset requires at least one chunk".into(),
        ));
    }
    let num_chunks = meta.len();
    let has_filters = pipeline_message.is_some();

    // Walk a running cursor instead of pushing bytes; each address is a pure
    // function of the preceding chunk sizes, mirroring the buffered builder.
    let mut cursor: u64 = 0;
    let mut slots = Vec::with_capacity(num_chunks);
    let mut written_chunks = Vec::with_capacity(num_chunks);

    for m in meta {
        let address = base_address + cursor;
        let compressed_size = m.compressed_size;
        slots.push(ChunkSlotPlan { compressed_size });
        written_chunks.push(WrittenChunk {
            address,
            compressed_size,
            raw_size,
            filter_mask: m.filter_mask,
        });
        cursor += compressed_size;
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "chunk dimensions written into the on-disk u32 dimension fields selected for this file"
    )]
    let chunk_dims_u32: Vec<u32> = chunk_dims.iter().map(|&d| d as u32).collect();
    let offset_size: u8 = 8;
    let length_size: u8 = 8;

    let use_extensible = maxshape.is_some_and(|ms| ms.contains(&u64::MAX));

    let mut index_tail = Vec::new();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "element size written into the on-disk u32 dimension field selected for this file"
    )]
    let layout_message = if use_extensible {
        let ea_address = base_address + cursor;
        let ea_bytes = build_extensible_array_at(
            &written_chunks,
            offset_size,
            length_size,
            has_filters,
            ea_address,
        )?;
        cursor += ea_bytes.len() as u64;
        index_tail = ea_bytes;
        serialize_v4_extensible_array(
            &chunk_dims_u32,
            ea_address,
            offset_size,
            element_size.get() as u32,
        )
    } else if num_chunks == 1 {
        let chunk_addr = written_chunks[0].address;
        let filtered_size = if has_filters {
            Some(written_chunks[0].compressed_size)
        } else {
            None
        };
        let filter_mask = if has_filters {
            Some(written_chunks[0].filter_mask)
        } else {
            None
        };
        serialize_v4_single_chunk(
            &chunk_dims_u32,
            chunk_addr,
            filtered_size,
            filter_mask,
            offset_size,
            element_size.get() as u32,
        )
    } else {
        let fa_address = base_address + cursor;
        let fa_bytes = build_fixed_array_at(
            &written_chunks,
            offset_size,
            length_size,
            has_filters,
            fa_address,
        );
        cursor += fa_bytes.len() as u64;
        index_tail = fa_bytes;
        serialize_v4_fixed_array(
            &chunk_dims_u32,
            fa_address,
            offset_size,
            element_size.get() as u32,
            FIXED_ARRAY_PAGE_BITS,
        )
    };

    Ok(VerbatimLayout {
        plan: VerbatimPlan {
            slots,
            index_tail,
            total_len: cursor,
        },
        layout_message,
        pipeline_message: pipeline_message.map(<[u8]>::to_vec),
    })
}

/// Stream a planned verbatim dataset's data region to `sink`, pulling each
/// chunk's bytes from `provider` one at a time. The emitted bytes are identical
/// to the concatenation [`plan_chunked_data_verbatim`] describes, so a streamed
/// file and a buffered file are byte-for-byte equal.
pub(crate) fn emit_chunked_data_verbatim<S: ByteSink>(
    sink: &mut S,
    plan: &VerbatimPlan,
    provider: &dyn ChunkProvider,
) -> Result<(), FormatError> {
    // One buffer for the whole dataset: it grows to the largest chunk and is
    // reused, so the streaming path's allocation count does not scale with the
    // chunk count.
    let mut chunk = Vec::new();
    for (i, slot) in plan.slots.iter().enumerate() {
        chunk.clear();
        provider.chunk_bytes(i, &mut chunk)?;
        if chunk.len() as u64 != slot.compressed_size {
            return Err(FormatError::ChunkedReadError(
                "verbatim chunk provider returned a chunk whose size differs from the \
                 planned size"
                    .into(),
            ));
        }
        sink.put(&chunk)?;
    }
    sink.put(&plan.index_tail)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunked_read::read_chunked_data;
    use crate::convert::nz;
    use crate::data_layout::DataLayout;
    use crate::dataspace::{Dataspace, DataspaceType};
    use crate::datatype::{Datatype, DatatypeByteOrder};

    fn make_f64_type() -> Datatype {
        Datatype::FloatingPoint {
            size: 8,
            byte_order: DatatypeByteOrder::LittleEndian,
            bit_offset: 0,
            bit_precision: 64,
            exponent_location: 52,
            exponent_size: 11,
            mantissa_location: 0,
            mantissa_size: 52,
            exponent_bias: 1023,
        }
    }

    fn f64_to_bytes(data: &[f64]) -> Vec<u8> {
        let mut b = Vec::with_capacity(data.len() * 8);
        for &v in data {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b
    }

    fn bytes_to_f64(data: &[u8]) -> Vec<f64> {
        data.chunks(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    /// Helper: build a chunked file blob and read it back using read_chunked_data
    fn roundtrip_chunked(
        values: &[f64],
        shape: &[u64],
        chunk_dims: &[u64],
        options: &ChunkOptions,
    ) -> Vec<f64> {
        let raw = f64_to_bytes(values);
        let base_address = 0x1000u64;
        let ctx = ChunkContext::basic(chunk_dims, 8);
        let result =
            build_chunked_data_at_ext(&raw, shape, ctx, options, base_address, None).unwrap();

        // Build a fake file buffer
        let file_size = base_address as usize + result.data_bytes.len();
        let mut file_data = vec![0u8; file_size];
        file_data[base_address as usize..].copy_from_slice(&result.data_bytes);

        // Parse layout
        let layout = DataLayout::parse(&result.layout_message, 8, 8).unwrap();
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: shape.len() as u8,
            dimensions: shape.to_vec(),
            max_dimensions: None,
        };
        let datatype = make_f64_type();

        // Parse pipeline if present
        let pipeline = result
            .pipeline_message
            .as_ref()
            .map(|pm| crate::filter_pipeline::FilterPipeline::parse(pm).unwrap());

        let output = read_chunked_data(
            &file_data,
            &layout,
            &dataspace,
            &datatype,
            pipeline.as_ref(),
            8,
            8,
        )
        .unwrap();

        bytes_to_f64(&output)
    }

    #[test]
    fn split_1d_single_chunk() {
        let data = f64_to_bytes(&[1.0, 2.0, 3.0]);
        let result = split_into_chunks(&data, &[3], &[3], nz(8));
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, vec![0]);
        assert_eq!(bytes_to_f64(&result[0].1), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn split_1d_multiple_chunks() {
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let data = f64_to_bytes(&values);
        let result = split_into_chunks(&data, &[10], &[4], nz(8));
        assert_eq!(result.len(), 3); // ceil(10/4) = 3
        assert_eq!(result[0].0, vec![0]);
        assert_eq!(result[1].0, vec![4]);
        assert_eq!(result[2].0, vec![8]);
        assert_eq!(bytes_to_f64(&result[0].1), vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(bytes_to_f64(&result[1].1), vec![4.0, 5.0, 6.0, 7.0]);
        // Last chunk: 2 valid + 2 padding zeros
        assert_eq!(bytes_to_f64(&result[2].1), vec![8.0, 9.0, 0.0, 0.0]);
    }

    #[test]
    fn split_2d_chunks() {
        // 4x4 dataset, 2x2 chunks -> 4 chunks
        let values: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let data = f64_to_bytes(&values);
        let result = split_into_chunks(&data, &[4, 4], &[2, 2], nz(8));
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].0, vec![0, 0]);
        assert_eq!(result[1].0, vec![0, 2]);
        assert_eq!(result[2].0, vec![2, 0]);
        assert_eq!(result[3].0, vec![2, 2]);
        // chunk (0,0): elements [0,1,4,5]
        assert_eq!(bytes_to_f64(&result[0].1), vec![0.0, 1.0, 4.0, 5.0]);
        // chunk (0,2): elements [2,3,6,7]
        assert_eq!(bytes_to_f64(&result[1].1), vec![2.0, 3.0, 6.0, 7.0]);
    }

    #[test]
    fn roundtrip_1d_single_chunk_no_compression() {
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let options = ChunkOptions {
            chunk_dims: Some(vec![10]),
            ..Default::default()
        };
        let result = roundtrip_chunked(&values, &[10], &[10], &options);
        assert_eq!(result, values);
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn roundtrip_1d_single_chunk_deflate() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let options = ChunkOptions {
            chunk_dims: Some(vec![100]),
            deflate_level: Some(6),
            ..Default::default()
        };
        let result = roundtrip_chunked(&values, &[100], &[100], &options);
        assert_eq!(result, values);
    }

    #[test]
    fn roundtrip_1d_multi_chunk_no_compression() {
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let options = ChunkOptions {
            chunk_dims: Some(vec![8]),
            ..Default::default()
        };
        let result = roundtrip_chunked(&values, &[20], &[8], &options);
        assert_eq!(result, values);
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn roundtrip_1d_multi_chunk_deflate() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let options = ChunkOptions {
            chunk_dims: Some(vec![20]),
            deflate_level: Some(6),
            ..Default::default()
        };
        let result = roundtrip_chunked(&values, &[100], &[20], &options);
        assert_eq!(result, values);
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn roundtrip_1d_shuffle_deflate() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let options = ChunkOptions {
            chunk_dims: Some(vec![50]),
            deflate_level: Some(6),
            shuffle: true,
            ..Default::default()
        };
        let result = roundtrip_chunked(&values, &[100], &[50], &options);
        assert_eq!(result, values);
    }

    #[test]
    fn roundtrip_2d_chunks() {
        // 6x4 dataset, 3x2 chunks
        let values: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let options = ChunkOptions {
            chunk_dims: Some(vec![3, 2]),
            ..Default::default()
        };
        let result = roundtrip_chunked(&values, &[6, 4], &[3, 2], &options);
        assert_eq!(result, values);
    }

    /// Chunks are stored back to back, and the index begins where the last
    /// chunk ends. The chunk size here (7 f64 = 56 bytes) is deliberately not a
    /// multiple of any cache line, so padding at *either* site — between chunks
    /// or before the index — would move bytes this pins.
    #[test]
    fn chunks_are_stored_back_to_back() {
        let values: Vec<f64> = (0..21).map(|i| i as f64).collect();
        let raw = f64_to_bytes(&values);
        let options = ChunkOptions {
            chunk_dims: Some(vec![7]),
            ..Default::default()
        };
        let dims = [7u64];
        let ctx = ChunkContext::basic(&dims, 8);
        let result = build_chunked_data_at_ext(&raw, &[21], ctx, &options, 0x1000, None).unwrap();

        // Unfiltered chunks are stored verbatim, so the data region opens with
        // the raw bytes in order.
        assert_eq!(
            &result.data_bytes[..raw.len()],
            &raw[..],
            "the three chunks must concatenate with nothing between them"
        );
        // And the Fixed Array header starts in the very next byte. Asserting the
        // signature's position, rather than only the chunk prefix, is what keeps
        // padding from creeping back in ahead of the index.
        assert_eq!(
            &result.data_bytes[raw.len()..raw.len() + 4],
            b"FAHD",
            "the chunk index must begin where the last chunk ends"
        );
    }

    /// The same rule for the streaming path, stated where it is computed: the
    /// data region is exactly the chunks plus the index, so a plan that inserted
    /// padding would make `total_len` exceed the sum.
    #[test]
    fn a_verbatim_plan_reserves_only_the_chunks_and_the_index() {
        let meta: Vec<ChunkMeta> = [37u64, 111, 5]
            .into_iter()
            .map(|compressed_size| ChunkMeta {
                compressed_size,
                filter_mask: 0,
            })
            .collect();
        let layout =
            plan_chunked_data_verbatim(&meta, &[7], nz(8), 56, Some(&[]), 0x1000, None).unwrap();

        let chunk_bytes: u64 = meta.iter().map(|m| m.compressed_size).sum();
        assert_eq!(
            layout.plan.total_len,
            chunk_bytes + layout.plan.index_tail.len() as u64
        );
        let planned: Vec<u64> = layout
            .plan
            .slots
            .iter()
            .map(|s| s.compressed_size)
            .collect();
        assert_eq!(planned, vec![37, 111, 5]);
    }

    /// A chunk-less plan has no first chunk to anchor the index against, so it
    /// is refused rather than planned as an empty region.
    #[test]
    fn a_verbatim_plan_with_no_chunks_is_refused() {
        let result = plan_chunked_data_verbatim(&[], &[7], nz(8), 56, None, 0x1000, None);
        assert!(
            matches!(result, Err(FormatError::ChunkedReadError(_))),
            "a chunk-less plan must be refused"
        );
    }

    #[test]
    fn chunk_options_auto_dims() {
        let options = ChunkOptions {
            chunk_dims: None,
            deflate_level: Some(6),
            ..Default::default()
        };
        let dims = options.resolve_chunk_dims(&[100, 50]);
        assert_eq!(dims, vec![100, 50]);
    }

    #[test]
    fn chunk_options_pipeline_deflate() {
        let options = ChunkOptions {
            deflate_level: Some(6),
            ..Default::default()
        };
        let pl = options.build_pipeline(8, &[], None, None).unwrap().unwrap();
        assert_eq!(pl.filters.len(), 1);
        assert_eq!(pl.filters[0].filter_id, FILTER_DEFLATE);
    }

    #[test]
    fn chunk_options_pipeline_shuffle_deflate_fletcher32() {
        let options = ChunkOptions {
            deflate_level: Some(6),
            shuffle: true,
            fletcher32: true,
            ..Default::default()
        };
        let pl = options.build_pipeline(8, &[], None, None).unwrap().unwrap();
        assert_eq!(pl.filters.len(), 3);
        assert_eq!(pl.filters[0].filter_id, FILTER_SHUFFLE);
        assert_eq!(pl.filters[1].filter_id, FILTER_DEFLATE);
        assert_eq!(pl.filters[2].filter_id, FILTER_FLETCHER32);
    }

    /// Every request naming two filters where one would displace the other is
    /// refused, and the error names both (#233).
    ///
    /// The refusal is the observable part: a dropped filter leaves nothing in
    /// the file to distinguish `with_shuffle().with_zfp(16.0)` from
    /// `with_zfp(16.0)`, so a caller who wrote the first and got the second has
    /// no way to find out. Asserting on the message rather than only on
    /// `is_err()` is what keeps a refusal from being reported as some unrelated
    /// failure that happens to also be an error.
    #[test]
    fn conflicting_filter_requests_are_refused() {
        let so = ScaleOffset::FloatDScale(2);
        let cases: &[(&str, &str, ChunkOptions)] = &[
            (
                "lzf",
                "deflate",
                ChunkOptions {
                    lzf: true,
                    deflate_level: Some(6),
                    ..Default::default()
                },
            ),
            (
                "shuffle",
                "scale-offset",
                ChunkOptions {
                    shuffle: true,
                    scale_offset: Some(so),
                    ..Default::default()
                },
            ),
            #[cfg(feature = "zfp")]
            (
                "scale-offset",
                "ZFP",
                ChunkOptions {
                    scale_offset: Some(so),
                    zfp_rate: Some(16.0),
                    ..Default::default()
                },
            ),
            #[cfg(feature = "zfp")]
            (
                "shuffle",
                "ZFP",
                ChunkOptions {
                    shuffle: true,
                    zfp_rate: Some(16.0),
                    ..Default::default()
                },
            ),
            #[cfg(feature = "zfp")]
            (
                "lzf",
                "ZFP",
                ChunkOptions {
                    lzf: true,
                    zfp_rate: Some(16.0),
                    ..Default::default()
                },
            ),
            #[cfg(feature = "zfp")]
            (
                "deflate",
                "ZFP",
                ChunkOptions {
                    deflate_level: Some(6),
                    zfp_rate: Some(16.0),
                    ..Default::default()
                },
            ),
        ];

        for (a, b, options) in cases {
            // Arguments a valid ZFP or scale-offset request would need. The
            // clash has to be reported whether or not they are satisfiable, so
            // pass ones that are: an error raised only because the datatype was
            // also wrong would pass an `is_err()` check while leaving the
            // combination itself unrefused.
            let err = options
                .build_pipeline(
                    8,
                    &[64],
                    zfp_f64_type(),
                    Some(
                        crate::scaleoffset::scale_offset_type_from_datatype(&make_f64_type())
                            .expect("f64 is a scale-offset type"),
                    ),
                )
                .expect_err("{a} + {b} was accepted");
            let FormatError::FilterError(msg) = &err else {
                panic!("{a} + {b}: expected a filter error, got {err}");
            };
            assert!(msg.contains(a) && msg.contains(b), "{a} + {b}: {msg}");
        }
    }

    /// The type a ZFP request needs, when the feature is on.
    #[cfg(feature = "zfp")]
    fn zfp_f64_type() -> Option<ZfpElementTypeWhenEnabled> {
        crate::filters::zfp_element_type_from_datatype(&make_f64_type())
    }

    #[cfg(not(feature = "zfp"))]
    fn zfp_f64_type() -> Option<ZfpElementTypeWhenEnabled> {
        None
    }

    /// Chaining a primary transform with a filter it does *not* displace still
    /// builds, so the refusal above is a rule about conflicts rather than a
    /// blanket ban on combining filters.
    #[test]
    fn compatible_filter_requests_still_build() {
        let so = Some(ScaleOffset::FloatDScale(2));
        let so_ty = crate::scaleoffset::scale_offset_type_from_datatype(&make_f64_type());
        let cases: &[(ChunkOptions, &[u16])] = &[
            (
                ChunkOptions {
                    shuffle: true,
                    deflate_level: Some(6),
                    ..Default::default()
                },
                &[FILTER_SHUFFLE, FILTER_DEFLATE],
            ),
            (
                ChunkOptions {
                    shuffle: true,
                    lzf: true,
                    ..Default::default()
                },
                &[FILTER_SHUFFLE, FILTER_LZF],
            ),
            (
                ChunkOptions {
                    scale_offset: so,
                    deflate_level: Some(6),
                    ..Default::default()
                },
                &[FILTER_SCALEOFFSET, FILTER_DEFLATE],
            ),
            (
                ChunkOptions {
                    scale_offset: so,
                    lzf: true,
                    fletcher32: true,
                    ..Default::default()
                },
                &[FILTER_SCALEOFFSET, FILTER_LZF, FILTER_FLETCHER32],
            ),
        ];

        for (options, expected) in cases {
            let pl = options
                .build_pipeline(8, &[64], zfp_f64_type(), so_ty)
                .unwrap()
                .unwrap();
            let ids: Vec<u16> = pl.filters.iter().map(|f| f.filter_id).collect();
            assert_eq!(&ids, expected);
        }
    }

    #[test]
    fn serialize_v4_single_chunk_no_filters_roundtrip() {
        let msg = serialize_v4_single_chunk(&[20], 0x1000, None, None, 8, 8);
        let layout = DataLayout::parse(&msg, 8, 8).unwrap();
        match layout {
            DataLayout::Chunked {
                chunk_dimensions,
                btree_address,
                version,
                chunk_index_type,
                single_chunk_filtered_size,
                single_chunk_filter_mask,
            } => {
                assert_eq!(version, 4);
                assert_eq!(chunk_index_type, Some(1));
                assert_eq!(chunk_dimensions, vec![20, 8]);
                assert_eq!(btree_address, Some(0x1000));
                assert_eq!(single_chunk_filtered_size, None);
                assert_eq!(single_chunk_filter_mask, None);
            }
            _ => panic!("expected chunked layout"),
        }
    }

    #[test]
    fn serialize_v4_single_chunk_with_filters_roundtrip() {
        let msg = serialize_v4_single_chunk(&[100], 0x2000, Some(500), Some(0), 8, 8);
        let layout = DataLayout::parse(&msg, 8, 8).unwrap();
        match layout {
            DataLayout::Chunked {
                btree_address,
                single_chunk_filtered_size,
                single_chunk_filter_mask,
                ..
            } => {
                assert_eq!(btree_address, Some(0x2000));
                assert_eq!(single_chunk_filtered_size, Some(500));
                assert_eq!(single_chunk_filter_mask, Some(0));
            }
            _ => panic!("expected chunked layout"),
        }
    }

    #[test]
    fn serialize_v4_fixed_array_roundtrip() {
        let msg = serialize_v4_fixed_array(&[20], 0x3000, 8, 8, 4);
        let layout = DataLayout::parse(&msg, 8, 8).unwrap();
        match layout {
            DataLayout::Chunked {
                version,
                chunk_index_type,
                btree_address,
                chunk_dimensions,
                ..
            } => {
                assert_eq!(version, 4);
                assert_eq!(chunk_index_type, Some(3));
                assert_eq!(btree_address, Some(0x3000));
                assert_eq!(chunk_dimensions, vec![20, 8]);
            }
            _ => panic!("expected chunked layout"),
        }
    }

    #[test]
    fn build_fixed_array_valid_structure() {
        let chunks = vec![
            WrittenChunk {
                address: 0x1000,
                compressed_size: 160,
                raw_size: 160,
                filter_mask: 0,
            },
            WrittenChunk {
                address: 0x10A0,
                compressed_size: 160,
                raw_size: 160,
                filter_mask: 0,
            },
        ];
        let fa = build_fixed_array_at(&chunks, 8, 8, false, 0x2000);
        // Should start with FAHD
        assert_eq!(&fa[0..4], b"FAHD");
        // FAHD size = 4+1+1+1+1+8+8+4 = 28
        // FADB starts at offset 28
        assert_eq!(&fa[28..32], b"FADB");
    }

    // ---- Extensible Array tests ----

    #[test]
    fn serialize_v4_extensible_array_roundtrip() {
        let msg = serialize_v4_extensible_array(&[10], 0x4000, 8, 8);
        let layout = DataLayout::parse(&msg, 8, 8).unwrap();
        match layout {
            DataLayout::Chunked {
                version,
                chunk_index_type,
                btree_address,
                chunk_dimensions,
                ..
            } => {
                assert_eq!(version, 4);
                assert_eq!(chunk_index_type, Some(4));
                assert_eq!(btree_address, Some(0x4000));
                assert_eq!(chunk_dimensions, vec![10, 8]);
            }
            _ => panic!("expected chunked layout"),
        }
    }

    #[test]
    fn build_extensible_array_valid_structure() {
        let chunks = vec![
            WrittenChunk {
                address: 0x1000,
                compressed_size: 80,
                raw_size: 80,
                filter_mask: 0,
            },
            WrittenChunk {
                address: 0x1050,
                compressed_size: 80,
                raw_size: 80,
                filter_mask: 0,
            },
        ];
        let ea = build_extensible_array_at(&chunks, 8, 8, false, 0x2000).unwrap();
        assert_eq!(&ea[0..4], b"EAHD");
        // Find EAIB after EAHD: 12 fixed + 6*8 stats + 8 addr + 4 checksum = 72
        let aehd_size = 4 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 6 * 8 + 8 + 4;
        assert_eq!(&ea[aehd_size..aehd_size + 4], b"EAIB");
    }

    /// Helper: roundtrip with EA (maxshape)
    fn roundtrip_ea(
        values: &[f64],
        shape: &[u64],
        chunk_dims: &[u64],
        maxshape: &[u64],
    ) -> Vec<f64> {
        let raw = f64_to_bytes(values);
        let base_address = 0x1000u64;
        let options = ChunkOptions {
            chunk_dims: Some(chunk_dims.to_vec()),
            ..Default::default()
        };
        let ctx = ChunkContext::basic(chunk_dims, 8);
        let result =
            build_chunked_data_at_ext(&raw, shape, ctx, &options, base_address, Some(maxshape))
                .unwrap();

        let file_size = base_address as usize + result.data_bytes.len();
        let mut file_data = vec![0u8; file_size];
        file_data[base_address as usize..].copy_from_slice(&result.data_bytes);

        let layout = DataLayout::parse(&result.layout_message, 8, 8).unwrap();
        // Verify it uses EA index
        match &layout {
            DataLayout::Chunked {
                chunk_index_type, ..
            } => {
                assert_eq!(*chunk_index_type, Some(4), "expected EA index type");
            }
            _ => panic!("expected chunked layout"),
        }

        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: shape.len() as u8,
            dimensions: shape.to_vec(),
            max_dimensions: Some(maxshape.to_vec()),
        };
        let datatype = make_f64_type();

        let output =
            read_chunked_data(&file_data, &layout, &dataspace, &datatype, None, 8, 8).unwrap();

        bytes_to_f64(&output)
    }

    #[test]
    fn ea_roundtrip_1d_inline_only() {
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let result = roundtrip_ea(&values, &[10], &[10], &[u64::MAX]);
        assert_eq!(result, values);
    }

    #[test]
    fn ea_roundtrip_1d_multi_chunks() {
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let result = roundtrip_ea(&values, &[20], &[5], &[u64::MAX]);
        assert_eq!(result, values);
    }

    #[test]
    fn ea_roundtrip_1d_many_chunks() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let result = roundtrip_ea(&values, &[100], &[10], &[u64::MAX]);
        assert_eq!(result, values);
    }

    /// One chunk per element across the inline, direct-data-block, and
    /// super-block ranges. Before the geometry fix these silently corrupted
    /// past 20 chunks (4 inline + the first 16-element direct block).
    #[test]
    fn ea_roundtrip_super_block_sizes() {
        for &n in &[245u64, 300, 2000, 50000] {
            let values: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let result = roundtrip_ea(&values, &[n], &[1], &[u64::MAX]);
            assert_eq!(result.len(), n as usize, "length mismatch at n={n}");
            assert_eq!(result, values, "data mismatch at n={n}");
        }
    }

    /// Cross the paging boundary (131060 = 4 inline + 240 direct + super blocks
    /// SB4..SB12), exercising paged data blocks in super block 13 (the first
    /// whose data blocks exceed 1024 elements) on both write and read.
    #[test]
    fn ea_roundtrip_paged_data_blocks() {
        let n: u64 = 132_000;
        let values: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let result = roundtrip_ea(&values, &[n], &[1], &[u64::MAX]);
        assert_eq!(result.len(), n as usize);
        assert_eq!(result, values);
    }

    /// `ea_compute_stats` must reproduce the EAHD statistics that
    /// `build_extensible_array_at` actually writes (these feed the in-place
    /// append writer, so any drift would corrupt appended files).
    #[cfg(feature = "std")]
    #[test]
    fn ea_compute_stats_matches_builder() {
        use crate::extensible_array::{EaGeometry, ExtensibleArrayHeader};
        let geom_header = ExtensibleArrayHeader {
            client_id: 0,
            element_size: 8,
            max_nelmts_bits: 32,
            idx_blk_elmts: 4,
            min_dblk_nelmts: 16,
            super_blk_min_nelmts: 4,
            max_dblk_nelmts_bits: 10,
            num_elements: 0,
            index_block_address: 0,
        };
        let geom = EaGeometry::from_header(&geom_header);
        for &n in &[1u64, 4, 20, 100, 244, 300, 2000, 50000, 131056, 140000] {
            let chunks: Vec<WrittenChunk> = (0..n)
                .map(|i| WrittenChunk {
                    address: 0x1000 + i * 8,
                    compressed_size: 8,
                    raw_size: 8,
                    filter_mask: 0,
                })
                .collect();
            let ea = build_extensible_array_at(&chunks, 8, 8, false, 0x100000).unwrap();
            // Parse the 6 stats from the EAHD (12-byte fixed prefix, then 6 * ls).
            let stat =
                |k: usize| u64::from_le_bytes(ea[12 + k * 8..12 + k * 8 + 8].try_into().unwrap());
            let built = super::EaStats {
                nsuper_blks: stat(0),
                super_blk_size: stat(1),
                ndata_blks: stat(2),
                data_blk_size: stat(3),
                max_idx_set: stat(4),
                nelmts: stat(5),
            };
            let computed = super::ea_compute_stats(&geom, 4, 8, 1024, 8, 4, n);
            assert_eq!(computed, built, "stats mismatch at n={n}");
        }
    }

    /// `extensible_array_len` is the span the in-place editor reserves for an
    /// array before a byte of it exists, so it has to equal the length
    /// `build_extensible_array_at` goes on to emit. A reservation that came out
    /// short would place the next object on top of the array.
    ///
    /// The small counts are swept *contiguously* rather than at hand-picked
    /// boundaries: which blocks an element count allocates is decided twice over
    /// — once by `ea_compute_stats`, which this length comes from, and once by
    /// the builder's own body — and a contiguous sweep crosses every transition
    /// between those two walks without anyone having to work out where the
    /// transitions are. It covers the inline slots, all six direct data blocks,
    /// and the first on-disk super block. The larger counts then reach the deeper
    /// super blocks and, at 131,061, the first *paged* data block.
    #[test]
    fn extensible_array_len_matches_what_it_builds() {
        fn check(n: u64, raw_size: u64, offset_size: u8, length_size: u8, has_filters: bool) {
            let chunks: Vec<WrittenChunk> = (0..n)
                .map(|i| WrittenChunk {
                    address: 0x1000 + i * 8,
                    compressed_size: 8,
                    raw_size,
                    filter_mask: 0,
                })
                .collect();
            let planned = extensible_array_len(&chunks, offset_size, length_size, has_filters);
            let built = build_extensible_array_at(
                &chunks,
                offset_size,
                length_size,
                has_filters,
                0x10_0000,
            )
            .unwrap();
            assert_eq!(
                planned,
                built.len() as u64,
                "planned length must match the emitted array at n={n}, raw_size={raw_size}, \
                 offset_size={offset_size}, has_filters={has_filters}"
            );
        }

        for &(offset_size, length_size) in &[(8u8, 8u8), (4u8, 4u8)] {
            for &has_filters in &[false, true] {
                // Contiguous across the inline, direct-block and first
                // super-block ranges.
                for n in 0..=250u64 {
                    check(n, 8, offset_size, length_size, has_filters);
                }
                // The deeper super blocks, and the paged boundary: 131,060 is the
                // last element the unpaged super block 12 holds, 131,061 the first
                // that allocates a paged data block.
                for &n in &[300u64, 2_000, 50_000, 131_060, 131_061, 140_000] {
                    check(n, 8, offset_size, length_size, has_filters);
                }
            }
        }

        // A filtered element record carries the chunk's compressed size in a field
        // sized to the largest *raw* chunk, so the record width — and with it the
        // index block and every data block — changes with that size.
        const RAW_SIZES: [u64; 4] = [8, 300, 100_000, 1 << 32];
        for &raw_size in &RAW_SIZES {
            for &n in &[1u64, 5, 244, 300, 2_000] {
                check(n, raw_size, 8, 8, true);
            }
        }
        // Those four raw sizes have to select four *different* field widths, or
        // the loop above is one fixture written four times. Asserted as
        // distinctness rather than as four literals: the rule is that the width
        // tracks the raw size, not that it takes any particular value.
        let widths: Vec<usize> = RAW_SIZES
            .iter()
            .map(|&raw_size| {
                let chunks = [WrittenChunk {
                    address: 0,
                    compressed_size: 8,
                    raw_size,
                    filter_mask: 0,
                }];
                super::ea_layout(&chunks, 8, 8, true).chunk_size_bytes
            })
            .collect();
        let mut distinct = widths.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(
            distinct.len(),
            RAW_SIZES.len(),
            "each raw size must select a different compressed-size field width, got {widths:?}"
        );
    }

    // ---- h5py round-trip tests for chunked writes ----

    // Runs `script` under python3, passing the HDF5 file path as `sys.argv[1]`
    // so the script can open it without interpolating the path into the source.
    // Interpolating a Windows path (with backslashes) into a Python string
    // literal breaks the parser (e.g. `\U` triggers a unicode-escape error).
    #[cfg(feature = "std")]
    fn h5py_run(path: &std::path::Path, script: &str) -> Option<String> {
        let o = std::process::Command::new("python3")
            .args(["-c", script, &path.to_string_lossy()])
            .output()
            .ok()?;
        if !o.status.success() {
            let err = String::from_utf8_lossy(&o.stderr);
            if err.contains("No module named") {
                return None; // h5py not installed — skip
            }
            panic!("h5py: {err}");
        }
        Some(String::from_utf8(o.stdout).unwrap().trim().to_string())
    }

    #[cfg(feature = "std")]
    #[test]
    fn h5py_reads_multiple_chunked_datasets() {
        use crate::file_writer::FileWriter;
        let mut fw = FileWriter::new();
        let data1: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let data2: Vec<f64> = (0..30).map(|i| (i * 10) as f64).collect();
        fw.create_dataset("a")
            .with_f64_data(&data1)
            .with_shape(&[50])
            .with_chunks(&[25]);
        fw.create_dataset("b")
            .with_f64_data(&data2)
            .with_shape(&[30])
            .with_chunks(&[10]);
        let bytes = fw.finish().unwrap();
        let path = std::env::temp_dir().join("rustyhdf5_chunked_multi.h5");
        std::fs::write(&path, &bytes).unwrap();
        let script = "import sys,h5py,json; f=h5py.File(sys.argv[1],'r'); print(json.dumps({'a':f['a'][:].tolist(),'b':f['b'][:].tolist()}))";
        let Some(out) = h5py_run(&path, script) else {
            return;
        };
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        let va: Vec<f64> = serde_json::from_value(v["a"].clone()).unwrap();
        let vb: Vec<f64> = serde_json::from_value(v["b"].clone()).unwrap();
        assert_eq!(va, data1);
        assert_eq!(vb, data2);
    }

    #[cfg(feature = "std")]
    #[test]
    fn h5py_reads_chunked_with_attrs() {
        use crate::file_writer::{AttrValue, FileWriter};
        let mut fw = FileWriter::new();
        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
        fw.create_dataset("data")
            .with_f64_data(&data)
            .with_shape(&[50])
            .with_chunks(&[25])
            .set_attr("units", AttrValue::String("meters".to_string()));
        let bytes = fw.finish().unwrap();
        let path = std::env::temp_dir().join("rustyhdf5_chunked_attrs.h5");
        std::fs::write(&path, &bytes).unwrap();
        let script = "import sys,h5py,json; f=h5py.File(sys.argv[1],'r'); d=f['data']; print(json.dumps({'values':d[:].tolist(),'units':d.attrs['units'].decode() if isinstance(d.attrs['units'],bytes) else str(d.attrs['units'])}))";
        let Some(out) = h5py_run(&path, script) else {
            return;
        };
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        let values: Vec<f64> = serde_json::from_value(v["values"].clone()).unwrap();
        assert_eq!(values, data);
        assert_eq!(v["units"], serde_json::json!("meters"));
    }
}
