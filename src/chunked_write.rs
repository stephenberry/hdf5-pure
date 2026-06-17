//! Chunked dataset writing: chunk splitting, compression, index building.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::checksum::jenkins_lookup3;
use crate::chunk_cache::align_to_cache_line;
use crate::convert::TryToUsize;
use crate::error::FormatError;
use crate::extensible_array::{EaGeometry, ExtensibleArrayHeader};
#[cfg(feature = "zfp")]
use crate::filter_pipeline::FILTER_ZFP;
use crate::filter_pipeline::{
    FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_SCALEOFFSET, FILTER_SHUFFLE, FilterDescription,
    FilterPipeline,
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

    /// Build a FilterPipeline from the options.
    ///
    /// `chunk_dims` and `zfp_element_type` are only consulted when the ZFP
    /// filter is active — they're embedded into the ZFP cd_values so the
    /// resulting file is readable by the reference H5Z-ZFP plugin.
    ///
    /// Returns [`FormatError::UnsupportedZfp`] when ZFP was requested but
    /// `zfp_element_type` is `None` (e.g. the dataset's datatype isn't one of
    /// f32/f64/i32/i64), or the chunk rank is outside 1..=4.
    pub fn build_pipeline(
        &self,
        element_size: u32,
        chunk_dims: &[u64],
        zfp_element_type: Option<ZfpElementTypeWhenEnabled>,
        scale_offset_type: Option<ScaleOffsetType>,
    ) -> Result<Option<FilterPipeline>, FormatError> {
        let mut filters = Vec::new();
        let _ = zfp_element_type; // used only under the `zfp` feature below

        // ZFP is a standalone compressor — it replaces shuffle + deflate.
        #[cfg(feature = "zfp")]
        let zfp_active = if let Some(rate) = self.zfp_rate {
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
            true
        } else {
            false
        };
        #[cfg(not(feature = "zfp"))]
        let zfp_active = false;

        // Scale-offset is also a primary transform: mutually exclusive with
        // ZFP, replaces shuffle, but may be followed by deflate (pushed first
        // so the pipeline order is [scaleoffset, deflate]).
        let scaleoffset_active = if let Some(mode) = self.scale_offset {
            if zfp_active {
                return Err(FormatError::FilterError(
                    "scale-offset and ZFP cannot be combined on one dataset".into(),
                ));
            }
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
            true
        } else {
            false
        };

        if !zfp_active && !scaleoffset_active && self.shuffle {
            filters.push(FilterDescription {
                filter_id: FILTER_SHUFFLE,
                name: None,
                flags: 0,
                client_data: vec![element_size],
            });
        }

        if !zfp_active && let Some(level) = self.deflate_level {
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

        // Note: h5py sets flags=0x0001 (optional) on filters, but this is not required
        // for read compatibility.

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
    element_size: usize,
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
        reason = "chunk dimensions derived from the in-memory write request; bounded by addressable memory"
    )]
    let chunk_total_elements: usize = chunk_dims.iter().map(|&d| d as usize).product();

    #[expect(
        clippy::cast_possible_truncation,
        reason = "total_chunks derived from the in-memory write request; bounded by addressable memory"
    )]
    let mut result = Vec::with_capacity(total_chunks as usize);

    for linear_idx in 0..total_chunks {
        // Convert linear index to chunk grid coordinates
        let mut chunk_grid_coords = vec![0u64; rank];
        let mut remaining = linear_idx;
        for d in (0..rank).rev() {
            chunk_grid_coords[d] = remaining % num_chunks_per_dim[d];
            remaining /= num_chunks_per_dim[d];
        }

        // Chunk offset in dataset space
        let offsets: Vec<u64> = (0..rank)
            .map(|d| chunk_grid_coords[d] * chunk_dims[d])
            .collect();

        // Extract chunk data
        let mut chunk_bytes = vec![0u8; chunk_total_elements * element_size];

        for flat_idx in 0..chunk_total_elements {
            let mut remaining_idx = flat_idx;
            let mut ds_flat = 0usize;
            let mut out_of_bounds = false;

            for d in 0..rank {
                let coord_in_chunk = remaining_idx / chunk_strides[d];
                remaining_idx %= chunk_strides[d];

                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "chunk offset derived from the in-memory write request; bounded by addressable memory"
                )]
                let global_coord = offsets[d] as usize + coord_in_chunk;
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "dataset dimension derived from the in-memory write request; bounded by addressable memory"
                )]
                let dim_extent = shape[d] as usize;
                if global_coord >= dim_extent {
                    out_of_bounds = true;
                    break;
                }
                ds_flat += global_coord * ds_strides[d];
            }

            if out_of_bounds {
                // Zero-filled (already initialized)
                continue;
            }

            let src_start = ds_flat * element_size;
            let dst_start = flat_idx * element_size;

            if src_start + element_size <= raw_data.len() {
                chunk_bytes[dst_start..dst_start + element_size]
                    .copy_from_slice(&raw_data[src_start..src_start + element_size]);
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

    assert_eq!(fahd.len(), fahd_total_size);

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
fn serialize_v4_extensible_array(
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
/// Used by the SWMR append writer (`std` only).
#[cfg(feature = "std")]
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
#[cfg(feature = "std")]
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
#[cfg(feature = "std")]
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
#[cfg(feature = "std")]
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
    let os = offset_size as usize;
    let num_elements = chunks.len();

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
    let aeib_address = ea_base_address + aehd_size as u64;

    let ndblk_addrs = geom.direct_dblk_nelmts.len();
    let nsblk_addrs = geom.nsblk_addrs;
    let aeib_size = aeib_size(offset_size, inline, elem_size, ndblk_addrs, nsblk_addrs);
    let body_base = aeib_address + aeib_size as u64;

    let undef_addr: u64 = match offset_size {
        4 => 0xFFFF_FFFF,
        _ => u64::MAX,
    };

    // ---- Build the body (direct data blocks, then super blocks) -----------
    // Addresses are absolute, computed from `body_base`, so the body can be
    // built before the index block that references it.
    let mut body: Vec<u8> = Vec::new();
    let mut direct_addrs: Vec<u64> = Vec::with_capacity(ndblk_addrs);
    let mut sblk_addrs: Vec<u64> = Vec::with_capacity(nsblk_addrs);

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
    for j in 0..nsblk_addrs {
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

/// Build chunked data with absolute addresses and optional maxshape.
///
/// `ctx` carries chunk_dims, element_size, and (for type-aware filters like
/// ZFP) the scalar element type. Build it via [`ChunkContext::from_datatype`]
/// when a `Datatype` is in scope.
pub fn build_chunked_data_at_ext(
    raw_data: &[u8],
    shape: &[u64],
    ctx: ChunkContext<'_>,
    options: &ChunkOptions,
    base_address: u64,
    maxshape: Option<&[u64]>,
) -> Result<ChunkedDataResult, FormatError> {
    let chunk_dims = ctx.chunk_dims;
    let element_size = ctx.element_size as usize;
    let pipeline = options.build_pipeline(
        ctx.element_size,
        chunk_dims,
        ctx.element_type,
        ctx.scale_offset_type,
    )?;

    let chunks = split_into_chunks(raw_data, shape, chunk_dims, element_size);
    let num_chunks = chunks.len();
    let has_filters = pipeline.is_some();

    // Compress each chunk, padding to cache-line boundaries for aligned access
    let mut data_buf = Vec::new();
    let mut written_chunks = Vec::with_capacity(num_chunks);

    for (_offsets, chunk_bytes) in &chunks {
        let compressed = if let Some(ref pl) = pipeline {
            compress_chunk(chunk_bytes, pl, ctx)?
        } else {
            chunk_bytes.clone()
        };

        // Pad current position to cache-line boundary
        let aligned_offset = align_to_cache_line(data_buf.len());
        if aligned_offset > data_buf.len() {
            data_buf.resize(aligned_offset, 0u8);
        }

        let address = base_address + data_buf.len() as u64;
        let compressed_size = compressed.len() as u64;
        let raw_size = chunk_bytes.len() as u64;

        data_buf.extend_from_slice(&compressed);

        written_chunks.push(WrittenChunk {
            address,
            compressed_size,
            raw_size,
            filter_mask: 0,
        });
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "chunk dimensions written into the on-disk u32 dimension fields selected for this file"
    )]
    let chunk_dims_u32: Vec<u32> = chunk_dims.iter().map(|&d| d as u32).collect();
    let offset_size: u8 = 8;
    let length_size: u8 = 8;

    // Determine if we should use Extensible Array (resizable datasets)
    let use_extensible = maxshape.is_some_and(|ms| ms.contains(&u64::MAX));

    // Pad before index structures so they are also cache-line aligned
    let aligned_idx = align_to_cache_line(data_buf.len());
    if aligned_idx > data_buf.len() {
        data_buf.resize(aligned_idx, 0u8);
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "element size written into the on-disk u32 dimension field selected for this file"
    )]
    let layout_message = if use_extensible {
        let ea_address = base_address + data_buf.len() as u64;

        let ea_bytes = build_extensible_array_at(
            &written_chunks,
            offset_size,
            length_size,
            has_filters,
            ea_address,
        )?;
        data_buf.extend_from_slice(&ea_bytes);

        serialize_v4_extensible_array(
            &chunk_dims_u32,
            ea_address,
            offset_size,
            element_size as u32,
        )
    } else if num_chunks == 1 {
        let chunk_addr = written_chunks[0].address;
        let filtered_size = if has_filters {
            Some(written_chunks[0].compressed_size)
        } else {
            None
        };
        let filter_mask = if has_filters { Some(0u32) } else { None };
        serialize_v4_single_chunk(
            &chunk_dims_u32,
            chunk_addr,
            filtered_size,
            filter_mask,
            offset_size,
            element_size as u32,
        )
    } else {
        let fa_address = base_address + data_buf.len() as u64;

        let fa_bytes = build_fixed_array_at(
            &written_chunks,
            offset_size,
            length_size,
            has_filters,
            fa_address,
        );
        data_buf.extend_from_slice(&fa_bytes);

        serialize_v4_fixed_array(
            &chunk_dims_u32,
            fa_address,
            offset_size,
            element_size as u32,
            FIXED_ARRAY_PAGE_BITS,
        )
    };

    let pipeline_message = pipeline.as_ref().map(|pl| pl.serialize());

    Ok(ChunkedDataResult {
        data_bytes: data_buf,
        layout_message,
        pipeline_message,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunked_read::read_chunked_data;
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
        let result = split_into_chunks(&data, &[3], &[3], 8);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, vec![0]);
        assert_eq!(bytes_to_f64(&result[0].1), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn split_1d_multiple_chunks() {
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let data = f64_to_bytes(&values);
        let result = split_into_chunks(&data, &[10], &[4], 8);
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
        let result = split_into_chunks(&data, &[4, 4], &[2, 2], 8);
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

    #[test]
    fn chunk_addresses_are_cache_aligned() {
        use super::align_to_cache_line;
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let raw = f64_to_bytes(&values);
        let base_address = 0x1000u64;
        // Ensure base is aligned for this test
        let base_address = align_to_cache_line(base_address as usize) as u64;
        let options = ChunkOptions {
            chunk_dims: Some(vec![20]),
            ..Default::default()
        };
        let dims = [20u64];
        let ctx = ChunkContext::basic(&dims, 8);
        let result =
            build_chunked_data_at_ext(&raw, &[100], ctx, &options, base_address, None).unwrap();

        // Parse layout to get chunk addresses (via roundtrip read)
        let file_size = base_address as usize + result.data_bytes.len();
        let mut file_data = vec![0u8; file_size];
        file_data[base_address as usize..].copy_from_slice(&result.data_bytes);

        let layout = DataLayout::parse(&result.layout_message, 8, 8).unwrap();
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 1,
            dimensions: vec![100],
            max_dimensions: None,
        };
        let datatype = make_f64_type();

        // Verify data roundtrips correctly
        let output =
            read_chunked_data(&file_data, &layout, &dataspace, &datatype, None, 8, 8).unwrap();
        assert_eq!(bytes_to_f64(&output), values);
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
