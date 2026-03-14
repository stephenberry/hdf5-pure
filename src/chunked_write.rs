//! Chunked dataset writing: chunk splitting, compression, index building.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::checksum::jenkins_lookup3;
use crate::chunk_cache::{CACHE_LINE_SIZE, align_to_cache_line};
use crate::error::FormatError;
use crate::filter_pipeline::{
    FilterDescription, FilterPipeline, FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_SHUFFLE,
};
use crate::filters::compress_chunk;

/// Round a file offset up to the next cache-line boundary.
///
/// This ensures chunk data starts at an address that is a multiple of the
/// architecture's cache line size, enabling aligned loads in SIMD paths.
#[inline]
pub fn align_chunk_offset(offset: u64) -> u64 {
    let align = CACHE_LINE_SIZE as u64;
    (offset + align - 1) & !(align - 1)
}

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
}

impl ChunkOptions {
    /// Whether any chunking option is enabled.
    pub fn is_chunked(&self) -> bool {
        self.chunk_dims.is_some()
            || self.deflate_level.is_some()
            || self.shuffle
            || self.fletcher32
    }

    /// Build a FilterPipeline from the options.
    pub fn build_pipeline(&self, element_size: u32) -> Option<FilterPipeline> {
        let mut filters = Vec::new();

        if self.shuffle {
            filters.push(FilterDescription {
                filter_id: FILTER_SHUFFLE,
                name: None,
                flags: 0,
                client_data: vec![element_size],
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

        // Note: h5py sets flags=0x0001 (optional) on filters, but this is not required
        // for read compatibility.

        if filters.is_empty() {
            None
        } else {
            Some(FilterPipeline {
                version: 2,
                filters,
            })
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
        ds_strides[i] = ds_strides[i + 1] * shape[i + 1] as usize;
    }

    // Chunk strides
    let mut chunk_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        chunk_strides[i] = chunk_strides[i + 1] * chunk_dims[i + 1] as usize;
    }

    let chunk_total_elements: usize = chunk_dims.iter().map(|&d| d as usize).product();

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

                let global_coord = offsets[d] as usize + coord_in_chunk;
                if global_coord >= shape[d] as usize {
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

/// Build the complete chunked dataset blob (chunk data + index) and return
/// layout/pipeline messages. `base_address` is where the blob will be placed in the file.
/// Serialize a v4 single chunk layout message (public for OH size estimation).
pub fn serialize_v4_single_chunk_pub(
    chunk_dims: &[u32],
    chunk_address: u64,
    filtered_size: Option<u64>,
    filter_mask: Option<u32>,
    offset_size: u8,
    element_size: u32,
) -> Vec<u8> {
    serialize_v4_single_chunk(
        chunk_dims, chunk_address, filtered_size, filter_mask, offset_size, element_size,
    )
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
        match dim_encoded_len {
            1 => buf.push(d as u8),
            2 => buf.extend_from_slice(&(d as u16).to_le_bytes()),
            4 => buf.extend_from_slice(&d.to_le_bytes()),
            _ => {}
        }
    }
    // Element size dimension
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
        match dim_encoded_len {
            1 => buf.push(d as u8),
            2 => buf.extend_from_slice(&(d as u16).to_le_bytes()),
            4 => buf.extend_from_slice(&d.to_le_bytes()),
            _ => {}
        }
    }
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
        let log2_val = if max_raw <= 1 { 0 } else { 63 - max_raw.leading_zeros() };
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
    fahd.push(elem_size as u8);

    // max_nelmts_bits: use 10 as default (page_size = 1024), matching h5py convention
    let max_bits: u8 = 10;
    fahd.push(max_bits);

    match length_size {
        4 => fahd.extend_from_slice(&(num_elements as u32).to_le_bytes()),
        8 => fahd.extend_from_slice(&(num_elements as u64).to_le_bytes()),
        _ => fahd.extend_from_slice(&(num_elements as u64).to_le_bytes()),
    }

    match offset_size {
        4 => fahd.extend_from_slice(&(fadb_address as u32).to_le_bytes()),
        8 => fahd.extend_from_slice(&fadb_address.to_le_bytes()),
        _ => fahd.extend_from_slice(&fadb_address.to_le_bytes()),
    }

    // Checksum
    let checksum = jenkins_lookup3(&fahd);
    fahd.extend_from_slice(&checksum.to_le_bytes());

    assert_eq!(fahd.len(), fahd_total_size);

    // Build FADB
    let mut fadb = Vec::new();
    fadb.extend_from_slice(b"FADB");
    fadb.push(0); // version
    fadb.push(client_id);

    // header address
    match offset_size {
        4 => fadb.extend_from_slice(&(fa_base_address as u32).to_le_bytes()),
        8 => fadb.extend_from_slice(&fa_base_address.to_le_bytes()),
        _ => fadb.extend_from_slice(&fa_base_address.to_le_bytes()),
    }

    // Element data
    for chunk in chunks {
        match offset_size {
            4 => fadb.extend_from_slice(&(chunk.address as u32).to_le_bytes()),
            8 => fadb.extend_from_slice(&chunk.address.to_le_bytes()),
            _ => fadb.extend_from_slice(&chunk.address.to_le_bytes()),
        }
        if has_filters {
            // Write compressed size using chunk_size_bytes (variable width)
            let cs_bytes = chunk.compressed_size.to_le_bytes();
            fadb.extend_from_slice(&cs_bytes[..chunk_size_bytes]);
            fadb.extend_from_slice(&chunk.filter_mask.to_le_bytes());
        }
    }

    // FADB checksum
    let fadb_checksum = jenkins_lookup3(&fadb);
    fadb.extend_from_slice(&fadb_checksum.to_le_bytes());

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
        match dim_encoded_len {
            1 => buf.push(d as u8),
            2 => buf.extend_from_slice(&(d as u16).to_le_bytes()),
            4 => buf.extend_from_slice(&d.to_le_bytes()),
            _ => {}
        }
    }
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
    match offset_size {
        4 => buf.extend_from_slice(&(ea_address as u32).to_le_bytes()),
        8 => buf.extend_from_slice(&ea_address.to_le_bytes()),
        _ => {}
    }

    buf
}

/// Build a complete Extensible Array at a known absolute address.
///
/// For simplicity, we put all elements inline in the index block when the
/// number of chunks is small (up to idx_blk_elmts), otherwise use inline +
/// direct data blocks.
pub fn build_extensible_array_at(
    chunks: &[WrittenChunk],
    offset_size: u8,
    length_size: u8,
    has_filters: bool,
    ea_base_address: u64,
) -> Vec<u8> {
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

    // EA creation parameters — must match HDF5 C library defaults exactly
    let max_nelmts_bits: u8 = 32;
    let idx_blk_elmts: u8 = 4;
    let min_dblk_nelmts: u8 = 16;
    let super_blk_min_nelmts: u8 = 4;
    let max_dblk_nelmts_bits: u8 = 10;

    // EAHD size: fixed(12) + 6 stats(6*length_size) + addr(offset_size) + checksum(4)
    let aehd_size = 4 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1
        + 6 * length_size as usize + os + 4;
    let aeib_address = ea_base_address + aehd_size as u64;

    // Determine how many elements go inline vs data blocks
    let n_inline = (idx_blk_elmts as usize).min(num_elements);
    let remaining_after_inline = num_elements.saturating_sub(n_inline);

    // Compute super block layout per HDF5 spec:
    // nsblks = floor(log2((2^max_nelmts_bits - idx_blk_elmts) / data_blk_min_elmts)) + 1
    // For each super block i:
    //   ndblks_in_sblk = 2^floor(i/2)
    //   dblk_nelmts = data_blk_min_elmts * 2^ceil(i/2)
    // Super blocks 0..sup_blk_min_data_ptrs-1 have their data block addrs in EAIB directly.
    // Super blocks sup_blk_min_data_ptrs..nsblks-1 get super block addresses.
    let sblk_min = super_blk_min_nelmts as usize; // sup_blk_min_data_ptrs
    // nsblks = log2(2^max_nelmts_bits / data_blk_min_elmts) + 1
    //        = max_nelmts_bits - log2(data_blk_min_elmts) + 1
    let log2_dblk_min = if min_dblk_nelmts <= 1 { 0 } else { (min_dblk_nelmts as u32).trailing_zeros() as usize };
    let nsblks = (max_nelmts_bits as usize).saturating_sub(log2_dblk_min) + 1;

    // Direct data block addresses (from super blocks 0..sblk_min-1)
    let mut dblk_sizes: Vec<usize> = Vec::new();
    for sblk_idx in 0..sblk_min.min(nsblks) {
        let ndblks = 1usize << (sblk_idx / 2);
        let dblk_nelmts = (min_dblk_nelmts as usize) * (1 << sblk_idx.div_ceil(2));
        for _ in 0..ndblks {
            dblk_sizes.push(dblk_nelmts);
        }
    }
    let n_direct_dblks = dblk_sizes.len();

    // Super block addresses (for super blocks sblk_min..nsblks-1)
    let n_sblk_addrs = nsblks.saturating_sub(sblk_min);

    // EAIB size: header + inline elements + direct dblk addresses + sblk addresses + checksum
    let aeib_size = 4 + 1 + 1 + os // sig+ver+client+hdr_addr
        + idx_blk_elmts as usize * elem_size // inline elements (always all slots)
        + n_direct_dblks * os // direct data block addresses
        + n_sblk_addrs * os // super block addresses
        + 4; // checksum

    // Build AEHD
    let mut aehd = Vec::with_capacity(aehd_size);
    aehd.extend_from_slice(b"EAHD");
    aehd.push(0); // version
    aehd.push(client_id);
    aehd.push(elem_size as u8);
    aehd.push(max_nelmts_bits);
    aehd.push(idx_blk_elmts);
    aehd.push(min_dblk_nelmts);
    aehd.push(super_blk_min_nelmts);
    aehd.push(max_dblk_nelmts_bits);

    // 6 stats fields matching HDF5 C library:
    // [0] = 0 (unknown/reserved), [1] = 0 (unknown/reserved),
    // [2] = ndata_blks, [3] = data_blk_total_size (computed after building data blocks),
    // [4] = nelmts, [5] = max_idx_set
    // We'll compute ndata_blks and data_blk_size below, and fill with placeholder for now.
    // Actually, we need to compute these before writing EAHD.
    // Count data blocks that will have chunks:
    let n_active_dblks: u64 = if remaining_after_inline > 0 {
        let mut count = 0u64;
        let mut ci = n_inline;
        for &sz in &dblk_sizes {
            if ci < num_elements {
                count += 1;
                ci += sz;
            }
        }
        count
    } else {
        0
    };
    // Compute total data block size (we'll update after building, but estimate here)
    // For now, compute the AEDB size per block: sig(4) + ver(1) + cid(1) + hdr_addr(os) + nelmts*elem_size + checksum(4)
    let aedb_header_overhead = 4 + 1 + 1 + os + 4;
    let data_blk_total_size: u64 = if remaining_after_inline > 0 {
        let mut total = 0u64;
        let mut ci = n_inline;
        for &sz in &dblk_sizes {
            if ci < num_elements {
                total += (aedb_header_overhead + sz * elem_size) as u64;
                ci += sz;
            }
        }
        total
    } else {
        0
    };
    // max_idx_set: idx_blk_elmts + sum of data block sizes for active blocks
    let max_idx_set: u64 = if remaining_after_inline > 0 {
        let mut max_set = idx_blk_elmts as u64;
        let mut ci = n_inline;
        for &sz in &dblk_sizes {
            if ci < num_elements {
                max_set += sz as u64;
                ci += sz;
            }
        }
        max_set
    } else {
        idx_blk_elmts as u64
    };

    let write_length = |buf: &mut Vec<u8>, val: u64| {
        match length_size {
            4 => buf.extend_from_slice(&(val as u32).to_le_bytes()),
            _ => buf.extend_from_slice(&val.to_le_bytes()),
        }
    };
    let write_addr = |buf: &mut Vec<u8>, val: u64| {
        match offset_size {
            4 => buf.extend_from_slice(&(val as u32).to_le_bytes()),
            _ => buf.extend_from_slice(&val.to_le_bytes()),
        }
    };

    write_length(&mut aehd, 0); // stat[0]: reserved/unknown
    write_length(&mut aehd, 0); // stat[1]: reserved/unknown
    write_length(&mut aehd, n_active_dblks); // stat[2]: ndata_blks
    write_length(&mut aehd, data_blk_total_size); // stat[3]: data_blk_total_size
    write_length(&mut aehd, num_elements as u64); // stat[4]: nelmts
    write_length(&mut aehd, max_idx_set); // stat[5]: max_idx_set

    write_addr(&mut aehd, aeib_address);

    let aehd_checksum = jenkins_lookup3(&aehd);
    aehd.extend_from_slice(&aehd_checksum.to_le_bytes());
    debug_assert_eq!(aehd.len(), aehd_size);

    // Build AEIB
    let mut aeib = Vec::with_capacity(aeib_size);
    aeib.extend_from_slice(b"EAIB");
    aeib.push(0); // version
    aeib.push(client_id);

    // header address
    match offset_size {
        4 => aeib.extend_from_slice(&(ea_base_address as u32).to_le_bytes()),
        8 => aeib.extend_from_slice(&ea_base_address.to_le_bytes()),
        _ => aeib.extend_from_slice(&ea_base_address.to_le_bytes()),
    }

    // Inline elements (always write idx_blk_elmts slots, fill unused with undefined)
    #[allow(clippy::needless_range_loop)]
    for i in 0..idx_blk_elmts as usize {
        if i < n_inline {
            write_chunk_element(&mut aeib, &chunks[i], offset_size, has_filters, chunk_size_bytes);
        } else {
            write_undefined_element(&mut aeib, offset_size, has_filters, chunk_size_bytes);
        }
    }

    // Data block addresses + build data blocks
    let mut data_blocks_buf = Vec::new();
    let dblks_base = aeib_address + aeib_size as u64;
    let mut dblk_cursor = dblks_base;
    let mut chunk_idx = n_inline;

    for &nelmts in &dblk_sizes {
        if chunk_idx >= num_elements {
            // No more chunks — write undefined address
            match offset_size {
                4 => aeib.extend_from_slice(&u32::MAX.to_le_bytes()),
                8 => aeib.extend_from_slice(&u64::MAX.to_le_bytes()),
                _ => aeib.extend_from_slice(&u64::MAX.to_le_bytes()),
            }
            continue;
        }

        // Write this data block's address
        match offset_size {
            4 => aeib.extend_from_slice(&(dblk_cursor as u32).to_le_bytes()),
            8 => aeib.extend_from_slice(&dblk_cursor.to_le_bytes()),
            _ => aeib.extend_from_slice(&dblk_cursor.to_le_bytes()),
        }

        // Build EADB
        let mut aedb = Vec::new();
        aedb.extend_from_slice(b"EADB");
        aedb.push(0); // version
        aedb.push(client_id);
        match offset_size {
            4 => aedb.extend_from_slice(&(ea_base_address as u32).to_le_bytes()),
            8 => aedb.extend_from_slice(&ea_base_address.to_le_bytes()),
            _ => aedb.extend_from_slice(&ea_base_address.to_le_bytes()),
        }

        // Block offset: encoded in ceil(max_nelmts_bits/8) bytes
        // This is the EA-relative index of the first element in this data block
        let blk_off_size = (max_nelmts_bits as usize).div_ceil(8);
        let blk_off_val = (chunk_idx - n_inline) as u64;
        aedb.extend_from_slice(&blk_off_val.to_le_bytes()[..blk_off_size]);

        // Write elements (fill all nelmts slots, use undefined for empty)
        for slot in 0..nelmts {
            if chunk_idx + slot < num_elements {
                write_chunk_element(
                    &mut aedb,
                    &chunks[chunk_idx + slot],
                    offset_size,
                    has_filters,
                    chunk_size_bytes,
                );
            } else {
                // Undefined slot
                write_undefined_element(&mut aedb, offset_size, has_filters, chunk_size_bytes);
            }
        }

        let aedb_checksum = jenkins_lookup3(&aedb);
        aedb.extend_from_slice(&aedb_checksum.to_le_bytes());

        dblk_cursor += aedb.len() as u64;
        data_blocks_buf.extend_from_slice(&aedb);
        chunk_idx += nelmts;
    }

    // Super block addresses (all undefined for now — we don't create super blocks)
    for _ in 0..n_sblk_addrs {
        match offset_size {
            4 => aeib.extend_from_slice(&u32::MAX.to_le_bytes()),
            8 => aeib.extend_from_slice(&u64::MAX.to_le_bytes()),
            _ => aeib.extend_from_slice(&u64::MAX.to_le_bytes()),
        }
    }

    // AEIB checksum
    let aeib_checksum = jenkins_lookup3(&aeib);
    aeib.extend_from_slice(&aeib_checksum.to_le_bytes());
    debug_assert_eq!(aeib.len(), aeib_size);

    let mut combined = aehd;
    combined.extend_from_slice(&aeib);
    combined.extend_from_slice(&data_blocks_buf);
    combined
}

fn write_chunk_element(
    buf: &mut Vec<u8>,
    chunk: &WrittenChunk,
    offset_size: u8,
    has_filters: bool,
    chunk_size_bytes: usize,
) {
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

/// Build chunked data with absolute addresses.
/// If `maxshape` has unlimited dims, uses Extensible Array index.
pub fn build_chunked_data_at(
    raw_data: &[u8],
    shape: &[u64],
    chunk_dims: &[u64],
    element_size: usize,
    options: &ChunkOptions,
    base_address: u64,
) -> Result<ChunkedDataResult, FormatError> {
    build_chunked_data_at_ext(raw_data, shape, chunk_dims, element_size, options, base_address, None)
}

/// Build chunked data with absolute addresses and optional maxshape.
pub fn build_chunked_data_at_ext(
    raw_data: &[u8],
    shape: &[u64],
    chunk_dims: &[u64],
    element_size: usize,
    options: &ChunkOptions,
    base_address: u64,
    maxshape: Option<&[u64]>,
) -> Result<ChunkedDataResult, FormatError> {
    let pipeline = options.build_pipeline(element_size as u32);

    let chunks = split_into_chunks(raw_data, shape, chunk_dims, element_size);
    let num_chunks = chunks.len();
    let has_filters = pipeline.is_some();

    // Compress each chunk, padding to cache-line boundaries for aligned access
    let mut data_buf = Vec::new();
    let mut written_chunks = Vec::with_capacity(num_chunks);

    for (_offsets, chunk_bytes) in &chunks {
        let compressed = if let Some(ref pl) = pipeline {
            compress_chunk(chunk_bytes, pl, element_size as u32)?
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

    let layout_message = if use_extensible {
        let ea_address = base_address + data_buf.len() as u64;

        let ea_bytes = build_extensible_array_at(
            &written_chunks,
            offset_size,
            length_size,
            has_filters,
            ea_address,
        );
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
        let max_bits: u8 = 10;

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
            max_bits,
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
        let result =
            build_chunked_data_at(&raw, shape, chunk_dims, 8, options, base_address).unwrap();

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
    fn align_chunk_offset_values() {
        use super::align_chunk_offset;
        use super::CACHE_LINE_SIZE;
        let cl = CACHE_LINE_SIZE as u64;
        assert_eq!(align_chunk_offset(0), 0);
        assert_eq!(align_chunk_offset(1), cl);
        assert_eq!(align_chunk_offset(cl), cl);
        assert_eq!(align_chunk_offset(cl + 1), cl * 2);
        assert_eq!(align_chunk_offset(cl * 10), cl * 10);
    }

    #[test]
    fn chunk_addresses_are_cache_aligned() {
        use super::align_chunk_offset;
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let raw = f64_to_bytes(&values);
        let base_address = 0x1000u64;
        // Ensure base is aligned for this test
        let base_address = align_chunk_offset(base_address);
        let options = ChunkOptions {
            chunk_dims: Some(vec![20]),
            ..Default::default()
        };
        let result =
            build_chunked_data_at(&raw, &[100], &[20], 8, &options, base_address).unwrap();

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
        let output = read_chunked_data(
            &file_data, &layout, &dataspace, &datatype, None, 8, 8,
        ).unwrap();
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
        let pl = options.build_pipeline(8).unwrap();
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
        let pl = options.build_pipeline(8).unwrap();
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
        let ea = build_extensible_array_at(&chunks, 8, 8, false, 0x2000);
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
        let result = build_chunked_data_at_ext(
            &raw, shape, chunk_dims, 8, &options, base_address, Some(maxshape),
        )
        .unwrap();

        let file_size = base_address as usize + result.data_bytes.len();
        let mut file_data = vec![0u8; file_size];
        file_data[base_address as usize..].copy_from_slice(&result.data_bytes);

        let layout = DataLayout::parse(&result.layout_message, 8, 8).unwrap();
        // Verify it uses EA index
        match &layout {
            DataLayout::Chunked { chunk_index_type, .. } => {
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

        let output = read_chunked_data(&file_data, &layout, &dataspace, &datatype, None, 8, 8)
            .unwrap();

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

    // ---- h5py round-trip tests for chunked writes ----

    #[cfg(feature = "std")]
    fn h5py_run(script: &str) -> Option<String> {
        let o = std::process::Command::new("python3")
            .args(["-c", script])
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
        fw.create_dataset("a").with_f64_data(&data1).with_shape(&[50]).with_chunks(&[25]);
        fw.create_dataset("b").with_f64_data(&data2).with_shape(&[30]).with_chunks(&[10]);
        let bytes = fw.finish().unwrap();
        let path = std::env::temp_dir().join("rustyhdf5_chunked_multi.h5");
        std::fs::write(&path, &bytes).unwrap();
        let script = format!(
            "import h5py,json; f=h5py.File('{}','r'); print(json.dumps({{'a':f['a'][:].tolist(),'b':f['b'][:].tolist()}}))",
            path.display()
        );
        let Some(out) = h5py_run(&script) else { return };
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        let va: Vec<f64> = serde_json::from_value(v["a"].clone()).unwrap();
        let vb: Vec<f64> = serde_json::from_value(v["b"].clone()).unwrap();
        assert_eq!(va, data1);
        assert_eq!(vb, data2);
    }

    #[cfg(feature = "std")]
    #[test]
    fn h5py_reads_chunked_with_attrs() {
        use crate::file_writer::{FileWriter, AttrValue};
        let mut fw = FileWriter::new();
        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
        fw.create_dataset("data").with_f64_data(&data).with_shape(&[50]).with_chunks(&[25])
            .set_attr("units", AttrValue::String("meters".to_string()));
        let bytes = fw.finish().unwrap();
        let path = std::env::temp_dir().join("rustyhdf5_chunked_attrs.h5");
        std::fs::write(&path, &bytes).unwrap();
        let script = format!(
            "import h5py,json; f=h5py.File('{}','r'); d=f['data']; print(json.dumps({{'values':d[:].tolist(),'units':d.attrs['units'].decode() if isinstance(d.attrs['units'],bytes) else str(d.attrs['units'])}}))",
            path.display()
        );
        let Some(out) = h5py_run(&script) else { return };
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        let values: Vec<f64> = serde_json::from_value(v["values"].clone()).unwrap();
        assert_eq!(values, data);
        assert_eq!(v["units"], serde_json::json!("meters"));
    }
}
