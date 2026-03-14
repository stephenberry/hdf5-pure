//! Chunked dataset reading: B-tree v1 type 1 traversal and chunk assembly.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};

use crate::chunk_cache::{ChunkCache, CacheAlignedBuffer};
use crate::data_layout::DataLayout;
use crate::dataspace::Dataspace;
use crate::datatype::Datatype;
use crate::error::FormatError;
use crate::filter_pipeline::FilterPipeline;
use crate::filters::decompress_chunk;
use crate::extensible_array::{ExtensibleArrayHeader, read_extensible_array_chunks};
use crate::fixed_array::{FixedArrayHeader, read_fixed_array_chunks};

#[cfg(feature = "parallel")]
use crate::parallel_read;

#[cfg(feature = "parallel")]
use crate::lane_partition::PartitionStats;

/// Decompress all chunks into cache-line-aligned buffers, using lane-partitioned
/// parallel decompression when the `parallel` feature is enabled and the chunk
/// count exceeds the threshold.
fn decompress_all_chunks(
    file_data: &[u8],
    chunks: &[ChunkInfo],
    pipeline: Option<&FilterPipeline>,
    chunk_total_bytes: usize,
    element_size: u32,
) -> Result<Vec<CacheAlignedBuffer>, FormatError> {
    #[cfg(feature = "parallel")]
    {
        if let Some(pl) = pipeline {
            if parallel_read::should_use_parallel(chunks.len()) {
                // Seed from the first chunk's address and count for determinism.
                let seed = chunks.first()
                    .map(|c| c.address)
                    .unwrap_or(0)
                    ^ (chunks.len() as u64);
                let (data, _stats) = parallel_read::decompress_chunks_lane_partitioned(
                    file_data,
                    chunks,
                    pl,
                    chunk_total_bytes,
                    element_size,
                    seed,
                    None, // auto-detect lane count
                )?;
                return Ok(data.into_iter().map(CacheAlignedBuffer::from_vec).collect());
            }
        }
    }

    // Sequential fallback — allocate into aligned buffers
    let mut result = Vec::with_capacity(chunks.len());
    for chunk_info in chunks {
        let c_addr = chunk_info.address as usize;
        let size = chunk_info.chunk_size as usize;
        if c_addr + size > file_data.len() {
            return Err(FormatError::UnexpectedEof {
                expected: c_addr + size,
                available: file_data.len(),
            });
        }
        let raw_chunk = &file_data[c_addr..c_addr + size];

        let decompressed = if let Some(pl) = pipeline {
            if chunk_info.filter_mask == 0 {
                decompress_chunk(raw_chunk, pl, chunk_total_bytes, element_size)?
            } else {
                raw_chunk.to_vec()
            }
        } else {
            raw_chunk.to_vec()
        };
        result.push(CacheAlignedBuffer::from_vec(decompressed));
    }
    Ok(result)
}

/// Decompress all chunks with lane-partitioned parallelism and return
/// per-lane diagnostics.
///
/// This is the stats-returning variant for callers who want to inspect
/// the partition balance.  Only available with the `parallel` feature.
#[cfg(feature = "parallel")]
pub fn decompress_all_chunks_with_stats(
    file_data: &[u8],
    chunks: &[ChunkInfo],
    pipeline: &FilterPipeline,
    chunk_total_bytes: usize,
    element_size: u32,
    seed: u64,
    num_lanes: Option<usize>,
) -> Result<(Vec<Vec<u8>>, PartitionStats), FormatError> {
    parallel_read::decompress_chunks_lane_partitioned(
        file_data,
        chunks,
        pipeline,
        chunk_total_bytes,
        element_size,
        seed,
        num_lanes,
    )
}

/// Information about a single chunk in a chunked dataset.
#[derive(Debug, Clone)]
pub struct ChunkInfo {
    /// Size of chunk data in the file (after compression).
    pub chunk_size: u32,
    /// Bitmask of filters that were NOT applied (0 = all applied).
    pub filter_mask: u32,
    /// N-dimensional offset of this chunk in dataset space.
    pub offsets: Vec<u64>,
    /// File address of the chunk data.
    pub address: u64,
}

fn read_offset(data: &[u8], pos: usize, size: u8) -> Result<u64, FormatError> {
    let s = size as usize;
    if pos + s > data.len() {
        return Err(FormatError::UnexpectedEof {
            expected: pos + s,
            available: data.len(),
        });
    }
    let slice = &data[pos..pos + s];
    Ok(match size {
        2 => u16::from_le_bytes([slice[0], slice[1]]) as u64,
        4 => u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]) as u64,
        8 => u64::from_le_bytes([
            slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
        ]),
        _ => return Err(FormatError::InvalidOffsetSize(size)),
    })
}

/// Traverse B-tree v1 type 1 to collect all chunk locations.
///
/// `ndims` is the number of offset dimensions in each key, which equals
/// `chunk_dimensions.len()` from the DataLayout::Chunked message (rank+1).
pub fn collect_chunk_info(
    file_data: &[u8],
    btree_address: u64,
    ndims: usize,
    offset_size: u8,
    _length_size: u8,
) -> Result<Vec<ChunkInfo>, FormatError> {
    let offset = btree_address as usize;
    let os = offset_size as usize;

    // Parse B-tree v1 header
    let header_size = 8 + os * 2;
    if offset + header_size > file_data.len() {
        return Err(FormatError::UnexpectedEof {
            expected: offset + header_size,
            available: file_data.len(),
        });
    }

    if &file_data[offset..offset + 4] != b"TREE" {
        return Err(FormatError::InvalidBTreeSignature);
    }

    let node_type = file_data[offset + 4];
    if node_type != 1 {
        return Err(FormatError::InvalidBTreeNodeType(node_type));
    }

    let node_level = file_data[offset + 5];
    let entries_used =
        u16::from_le_bytes([file_data[offset + 6], file_data[offset + 7]]) as usize;

    let mut pos = offset + 8 + os * 2; // skip left/right sibling

    // Key size: chunk_size(4) + filter_mask(4) + ndims * offset_size
    let key_size = 4 + 4 + ndims * os;

    if node_level == 0 {
        // Leaf node: keys and children interleaved
        // key[0], child[0], key[1], child[1], ..., key[N-1], child[N-1], key[N]
        let needed = entries_used * (key_size + os) + key_size;
        if pos + needed > file_data.len() {
            return Err(FormatError::UnexpectedEof {
                expected: pos + needed,
                available: file_data.len(),
            });
        }

        let mut chunks = Vec::with_capacity(entries_used);
        for _ in 0..entries_used {
            // Parse key
            let chunk_size = u32::from_le_bytes([
                file_data[pos],
                file_data[pos + 1],
                file_data[pos + 2],
                file_data[pos + 3],
            ]);
            let filter_mask = u32::from_le_bytes([
                file_data[pos + 4],
                file_data[pos + 5],
                file_data[pos + 6],
                file_data[pos + 7],
            ]);
            let mut offsets = Vec::with_capacity(ndims);
            let mut kp = pos + 8;
            for _ in 0..ndims {
                offsets.push(read_offset(file_data, kp, offset_size)?);
                kp += os;
            }
            pos += key_size;

            // Parse child address
            let address = read_offset(file_data, pos, offset_size)?;
            pos += os;

            chunks.push(ChunkInfo {
                chunk_size,
                filter_mask,
                offsets,
                address,
            });
        }
        // Skip final key
        Ok(chunks)
    } else {
        // Internal node: recurse into children
        let needed = entries_used * (key_size + os) + key_size;
        if pos + needed > file_data.len() {
            return Err(FormatError::UnexpectedEof {
                expected: pos + needed,
                available: file_data.len(),
            });
        }

        let mut child_addrs = Vec::with_capacity(entries_used);
        for _ in 0..entries_used {
            pos += key_size; // skip key
            let child_addr = read_offset(file_data, pos, offset_size)?;
            child_addrs.push(child_addr);
            pos += os;
        }

        let mut all_chunks = Vec::new();
        for child_addr in child_addrs {
            let child_chunks =
                collect_chunk_info(file_data, child_addr, ndims, offset_size, _length_size)?;
            all_chunks.extend(child_chunks);
        }
        Ok(all_chunks)
    }
}

/// Generate ChunkInfo entries for an implicit index (v4 index type 2).
///
/// Chunks are stored contiguously starting at `base_address`. No stored index;
/// addresses are computed from the chunk position.
pub fn generate_implicit_chunks(
    base_address: u64,
    dataset_dims: &[u64],
    chunk_dimensions: &[u32],
    element_size: u32,
) -> Vec<ChunkInfo> {
    let rank = chunk_dimensions.len();
    let chunk_byte_size: u64 = chunk_dimensions.iter().map(|&d| d as u64).product::<u64>()
        * element_size as u64;

    let mut num_chunks_per_dim = Vec::with_capacity(rank);
    for d in 0..rank {
        let ds = dataset_dims[d];
        let ch = chunk_dimensions[d] as u64;
        num_chunks_per_dim.push(ds.div_ceil(ch));
    }
    let total_chunks: u64 = num_chunks_per_dim.iter().product();

    let mut chunks = Vec::with_capacity(total_chunks as usize);
    for linear_idx in 0..total_chunks {
        let mut offsets = vec![0u64; rank];
        let mut remaining = linear_idx;
        for d in (0..rank).rev() {
            let nchunks = num_chunks_per_dim[d];
            let chunk_idx = remaining % nchunks;
            remaining /= nchunks;
            offsets[d] = chunk_idx * chunk_dimensions[d] as u64;
        }

        chunks.push(ChunkInfo {
            chunk_size: chunk_byte_size as u32,
            filter_mask: 0,
            offsets,
            address: base_address + linear_idx * chunk_byte_size,
        });
    }

    chunks
}

/// Read a chunked dataset, decompressing chunks as needed.
pub fn read_chunked_data(
    file_data: &[u8],
    layout: &DataLayout,
    dataspace: &Dataspace,
    datatype: &Datatype,
    pipeline: Option<&FilterPipeline>,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<u8>, FormatError> {
    let (chunk_dimensions, version, chunk_index_type, addr_opt,
         single_filtered_size, single_filter_mask) = match layout {
        DataLayout::Chunked {
            chunk_dimensions,
            btree_address,
            version,
            chunk_index_type,
            single_chunk_filtered_size,
            single_chunk_filter_mask,
        } => (chunk_dimensions, *version, *chunk_index_type, *btree_address,
              *single_chunk_filtered_size, *single_chunk_filter_mask),
        _ => {
            return Err(FormatError::ChunkedReadError(
                "expected chunked layout".into(),
            ))
        }
    };

    let addr = addr_opt.ok_or_else(|| {
        FormatError::ChunkedReadError("no address for chunked layout".into())
    })?;

    let elem_size = datatype.type_size() as usize;

    // Both v3 and v4 include element size as last dim (rank+1)
    let ndims = chunk_dimensions.len();
    let rank = ndims - 1;
    let chunk_dims: Vec<usize> = chunk_dimensions[..rank]
        .iter()
        .map(|&d| d as usize)
        .collect();

    let ds_dims: Vec<usize> = dataspace.dimensions.iter().map(|&d| d as usize).collect();
    if ds_dims.len() != rank {
        return Err(FormatError::ChunkedReadError(format!(
            "rank mismatch: dataspace has {} dims, layout has {} chunk dims (rank={})",
            ds_dims.len(),
            chunk_dimensions.len(),
            rank
        )));
    }

    // Collect chunks based on version and index type
    let chunks = match (version, chunk_index_type) {
        (3, _) => {
            let ndims = chunk_dimensions.len(); // rank+1
            collect_chunk_info(file_data, addr, ndims, offset_size, length_size)?
        }
        (4, Some(1)) => {
            // Single chunk — one chunk covering the entire dataset
            let chunk_byte_size: usize = chunk_dims.iter().product::<usize>() * elem_size;
            let (csize, fmask) = if let Some(fs) = single_filtered_size {
                (fs as u32, single_filter_mask.unwrap_or(0))
            } else {
                (chunk_byte_size as u32, 0)
            };
            vec![ChunkInfo {
                chunk_size: csize,
                filter_mask: fmask,
                offsets: vec![0u64; rank],
                address: addr,
            }]
        }
        (4, Some(2)) => {
            // Implicit index — use spatial chunk dims only
            let spatial_chunk_dims: Vec<u32> = chunk_dimensions[..rank].to_vec();
            generate_implicit_chunks(
                addr,
                &dataspace.dimensions,
                &spatial_chunk_dims,
                elem_size as u32,
            )
        }
        (4, Some(3)) => {
            // Fixed Array — use spatial chunk dims only
            let spatial_chunk_dims: Vec<u32> = chunk_dimensions[..rank].to_vec();
            let header = FixedArrayHeader::parse(file_data, addr as usize, offset_size, length_size)?;
            read_fixed_array_chunks(
                file_data, &header, &dataspace.dimensions, &spatial_chunk_dims,
                elem_size as u32, offset_size, length_size,
            )?
        }
        (4, Some(4)) => {
            // Extensible Array — use spatial chunk dims only
            let spatial_chunk_dims: Vec<u32> = chunk_dimensions[..rank].to_vec();
            let header = ExtensibleArrayHeader::parse(
                file_data, addr as usize, offset_size, length_size,
            )?;
            read_extensible_array_chunks(
                file_data, &header, &dataspace.dimensions, &spatial_chunk_dims,
                elem_size as u32, offset_size, length_size,
            )?
        }
        (v, idx) => {
            return Err(FormatError::ChunkedReadError(format!(
                "unsupported chunked layout version={v}, index_type={idx:?}"
            )))
        }
    };

    // Assemble output
    let total_elements = dataspace.num_elements() as usize;
    let total_bytes = total_elements * elem_size;
    let mut output = vec![0u8; total_bytes];

    let mut ds_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        ds_strides[i] = ds_strides[i + 1] * ds_dims[i + 1];
    }

    let mut chunk_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        chunk_strides[i] = chunk_strides[i + 1] * chunk_dims[i + 1];
    }

    let chunk_total_elements: usize = chunk_dims.iter().product();
    let chunk_total_bytes = chunk_total_elements * elem_size;

    // Decompress all chunks (parallel when beneficial, sequential otherwise)
    let decompressed_chunks = decompress_all_chunks(
        file_data,
        &chunks,
        pipeline,
        chunk_total_bytes,
        elem_size as u32,
    )?;

    for (chunk_info, decompressed) in chunks.iter().zip(decompressed_chunks.iter()) {
        // B-tree v1 (v3) offsets have rank+1 dims; v4 index offsets have rank dims
        let chunk_offsets: Vec<usize> = chunk_info.offsets.iter()
            .take(rank)
            .map(|&o| o as usize)
            .collect();

        if rank == 0 {
            let copy_len = decompressed.len().min(output.len());
            output[..copy_len].copy_from_slice(&decompressed[..copy_len]);
        } else {
            copy_chunk_to_output(
                decompressed,
                &mut output,
                &chunk_offsets,
                &chunk_dims,
                &ds_dims,
                &ds_strides,
                &chunk_strides,
                elem_size,
                rank,
            );
        }
    }

    Ok(output)
}

/// Read a chunked dataset with caching support.
///
/// On the first call, scans the chunk index (B-tree / fixed array / etc.) once
/// and populates the cache's hash index.  Subsequent calls skip the index scan
/// entirely.  Decompressed chunk data is also cached with LRU eviction.
pub fn read_chunked_data_cached(
    file_data: &[u8],
    layout: &DataLayout,
    dataspace: &Dataspace,
    datatype: &Datatype,
    pipeline: Option<&FilterPipeline>,
    offset_size: u8,
    length_size: u8,
    cache: &ChunkCache,
) -> Result<Vec<u8>, FormatError> {
    let (chunk_dimensions, version, chunk_index_type, addr_opt,
         single_filtered_size, single_filter_mask) = match layout {
        DataLayout::Chunked {
            chunk_dimensions,
            btree_address,
            version,
            chunk_index_type,
            single_chunk_filtered_size,
            single_chunk_filter_mask,
        } => (chunk_dimensions, *version, *chunk_index_type, *btree_address,
              *single_chunk_filtered_size, *single_chunk_filter_mask),
        _ => {
            return Err(FormatError::ChunkedReadError(
                "expected chunked layout".into(),
            ))
        }
    };

    let addr = addr_opt.ok_or_else(|| {
        FormatError::ChunkedReadError("no address for chunked layout".into())
    })?;

    let elem_size = datatype.type_size() as usize;
    let ndims = chunk_dimensions.len();
    let rank = ndims - 1;
    let chunk_dims: Vec<usize> = chunk_dimensions[..rank]
        .iter()
        .map(|&d| d as usize)
        .collect();

    let ds_dims: Vec<usize> = dataspace.dimensions.iter().map(|&d| d as usize).collect();
    if ds_dims.len() != rank {
        return Err(FormatError::ChunkedReadError(format!(
            "rank mismatch: dataspace has {} dims, layout has {} chunk dims (rank={})",
            ds_dims.len(), chunk_dimensions.len(), rank
        )));
    }

    // Populate chunk index on first access
    if !cache.has_index() {
        let chunks = match (version, chunk_index_type) {
            (3, _) => {
                collect_chunk_info(file_data, addr, ndims, offset_size, length_size)?
            }
            (4, Some(1)) => {
                let chunk_byte_size: usize = chunk_dims.iter().product::<usize>() * elem_size;
                let (csize, fmask) = if let Some(fs) = single_filtered_size {
                    (fs as u32, single_filter_mask.unwrap_or(0))
                } else {
                    (chunk_byte_size as u32, 0)
                };
                vec![ChunkInfo {
                    chunk_size: csize,
                    filter_mask: fmask,
                    offsets: vec![0u64; rank],
                    address: addr,
                }]
            }
            (4, Some(2)) => {
                let spatial_chunk_dims: Vec<u32> = chunk_dimensions[..rank].to_vec();
                generate_implicit_chunks(addr, &dataspace.dimensions, &spatial_chunk_dims, elem_size as u32)
            }
            (4, Some(3)) => {
                let spatial_chunk_dims: Vec<u32> = chunk_dimensions[..rank].to_vec();
                let header = FixedArrayHeader::parse(file_data, addr as usize, offset_size, length_size)?;
                read_fixed_array_chunks(
                    file_data, &header, &dataspace.dimensions, &spatial_chunk_dims,
                    elem_size as u32, offset_size, length_size,
                )?
            }
            (4, Some(4)) => {
                let spatial_chunk_dims: Vec<u32> = chunk_dimensions[..rank].to_vec();
                let header = ExtensibleArrayHeader::parse(file_data, addr as usize, offset_size, length_size)?;
                read_extensible_array_chunks(
                    file_data, &header, &dataspace.dimensions, &spatial_chunk_dims,
                    elem_size as u32, offset_size, length_size,
                )?
            }
            (v, idx) => {
                return Err(FormatError::ChunkedReadError(format!(
                    "unsupported chunked layout version={v}, index_type={idx:?}"
                )))
            }
        };
        cache.populate_index(&chunks, rank);
    }

    let chunks = cache.all_indexed_chunks().unwrap_or_default();

    // Assemble output
    let total_elements = dataspace.num_elements() as usize;
    let total_bytes = total_elements * elem_size;
    let mut output = vec![0u8; total_bytes];

    let mut ds_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        ds_strides[i] = ds_strides[i + 1] * ds_dims[i + 1];
    }

    let mut chunk_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        chunk_strides[i] = chunk_strides[i + 1] * chunk_dims[i + 1];
    }

    let chunk_total_elements: usize = chunk_dims.iter().product();
    let chunk_total_bytes = chunk_total_elements * elem_size;

    for chunk_info in &chunks {
        let coord: Vec<u64> = chunk_info.offsets.iter().take(rank).copied().collect();

        // Try decompressed cache first
        let decompressed = if let Some(cached) = cache.get_decompressed(&coord) {
            cached
        } else {
            // Decompress from file
            let c_addr = chunk_info.address as usize;
            let size = chunk_info.chunk_size as usize;
            if c_addr + size > file_data.len() {
                return Err(FormatError::UnexpectedEof {
                    expected: c_addr + size,
                    available: file_data.len(),
                });
            }
            let raw_chunk = &file_data[c_addr..c_addr + size];
            let dec = if let Some(pl) = pipeline {
                if chunk_info.filter_mask == 0 {
                    decompress_chunk(raw_chunk, pl, chunk_total_bytes, elem_size as u32)?
                } else {
                    raw_chunk.to_vec()
                }
            } else {
                raw_chunk.to_vec()
            };
            cache.put_decompressed(coord, dec.clone());
            dec
        };

        let chunk_offsets: Vec<usize> = chunk_info.offsets.iter()
            .take(rank)
            .map(|&o| o as usize)
            .collect();

        if rank == 0 {
            let copy_len = decompressed.len().min(output.len());
            output[..copy_len].copy_from_slice(&decompressed[..copy_len]);
        } else {
            copy_chunk_to_output(
                &decompressed,
                &mut output,
                &chunk_offsets,
                &chunk_dims,
                &ds_dims,
                &ds_strides,
                &chunk_strides,
                elem_size,
                rank,
            );
        }
    }

    Ok(output)
}

/// Sweep context passed into `read_chunked_data_sweep` to enable adaptive
/// prefetching based on detected access patterns.
///
/// The caller is responsible for maintaining the `SweepContext` across
/// multiple reads on the same dataset. After each read, the context will
/// contain updated sweep detection state and any predicted next-chunk
/// coordinates.
pub struct SweepContext {
    /// Sliding window of recent chunk coordinates.
    pub history: Vec<Vec<u64>>,
    /// Maximum window size.
    pub window_size: usize,
    /// Currently detected sweep direction label.
    pub direction: &'static str,
    /// How many chunks ahead to predict.
    pub prefetch_count: usize,
    /// Predicted next chunk coordinates (populated after each read).
    pub predicted_next: Vec<Vec<u64>>,
}

impl SweepContext {
    /// Create a new sweep context with the given window size and prefetch count.
    pub fn new(window_size: usize, prefetch_count: usize) -> Self {
        Self {
            history: Vec::with_capacity(window_size),
            window_size,
            direction: "random",
            prefetch_count,
            predicted_next: Vec::new(),
        }
    }

    /// Create with default settings (window=12, prefetch=4).
    pub fn with_defaults() -> Self {
        Self::new(12, 4)
    }

    /// Record a chunk coordinate access and update predictions.
    fn record(&mut self, coord: Vec<u64>, ndims: usize) {
        if self.history.len() >= self.window_size {
            self.history.remove(0);
        }
        self.history.push(coord);

        if self.history.len() < 3 || ndims == 0 {
            self.direction = "random";
            self.predicted_next.clear();
            return;
        }

        // Inline sweep detection matching the algorithm in rustyhdf5-io/sweep.rs
        let num_deltas = self.history.len() - 1;
        let mut changing = vec![0usize; ndims];
        for i in 0..num_deltas {
            let prev = &self.history[i];
            let curr = &self.history[i + 1];
            if prev.len() < ndims || curr.len() < ndims {
                self.direction = "random";
                self.predicted_next.clear();
                return;
            }
            for d in 0..ndims {
                if curr[d] != prev[d] {
                    changing[d] += 1;
                }
            }
        }

        let threshold = (num_deltas + 1) / 2;
        let (max_dim, max_changes) = changing.iter().enumerate()
            .max_by_key(|(_, c)| *c).unwrap();

        if *max_changes < threshold {
            self.direction = "random";
            self.predicted_next.clear();
            return;
        }

        let others_max = changing.iter().enumerate()
            .filter(|(d, _)| *d != max_dim)
            .map(|(_, c)| *c)
            .max()
            .unwrap_or(0);

        if others_max > 0 && *max_changes < others_max * 2 {
            self.direction = "random";
            self.predicted_next.clear();
            return;
        }

        self.direction = if max_dim == ndims - 1 {
            "row_major"
        } else if max_dim == 0 {
            "column_major"
        } else {
            "slice_major"
        };

        // Predict next chunks
        let sweep_dim = max_dim;
        let mut total_step: i64 = 0;
        let mut step_count: usize = 0;
        for i in 1..self.history.len() {
            let prev = self.history[i - 1][sweep_dim] as i64;
            let curr = self.history[i][sweep_dim] as i64;
            let diff = curr - prev;
            if diff != 0 {
                total_step += diff;
                step_count += 1;
            }
        }

        if step_count == 0 {
            self.predicted_next.clear();
            return;
        }

        let avg_step = total_step / step_count as i64;
        if avg_step == 0 {
            self.predicted_next.clear();
            return;
        }

        let last = self.history.last().unwrap();
        self.predicted_next.clear();
        for i in 1..=self.prefetch_count {
            let mut pred = last.clone();
            let new_val = last[sweep_dim] as i64 + avg_step * i as i64;
            if new_val < 0 {
                break;
            }
            pred[sweep_dim] = new_val as u64;
            self.predicted_next.push(pred);
        }
    }
}

/// Read a chunked dataset with caching and sweep-aware prefetching.
///
/// Extends `read_chunked_data_cached` by feeding each chunk coordinate to a
/// [`SweepContext`]. When a sweep pattern is detected, predicted next-chunk
/// coordinates are pre-populated in the cache index via `prefetch_hint`.
#[allow(clippy::too_many_arguments)]
pub fn read_chunked_data_sweep(
    file_data: &[u8],
    layout: &DataLayout,
    dataspace: &Dataspace,
    datatype: &Datatype,
    pipeline: Option<&FilterPipeline>,
    offset_size: u8,
    length_size: u8,
    cache: &ChunkCache,
    sweep: &mut SweepContext,
) -> Result<Vec<u8>, FormatError> {
    let (chunk_dimensions, version, chunk_index_type, addr_opt,
         single_filtered_size, single_filter_mask) = match layout {
        DataLayout::Chunked {
            chunk_dimensions,
            btree_address,
            version,
            chunk_index_type,
            single_chunk_filtered_size,
            single_chunk_filter_mask,
        } => (chunk_dimensions, *version, *chunk_index_type, *btree_address,
              *single_chunk_filtered_size, *single_chunk_filter_mask),
        _ => {
            return Err(FormatError::ChunkedReadError(
                "expected chunked layout".into(),
            ))
        }
    };

    let addr = addr_opt.ok_or_else(|| {
        FormatError::ChunkedReadError("no address for chunked layout".into())
    })?;

    let elem_size = datatype.type_size() as usize;
    let ndims = chunk_dimensions.len();
    let rank = ndims - 1;
    let chunk_dims: Vec<usize> = chunk_dimensions[..rank]
        .iter()
        .map(|&d| d as usize)
        .collect();

    let ds_dims: Vec<usize> = dataspace.dimensions.iter().map(|&d| d as usize).collect();
    if ds_dims.len() != rank {
        return Err(FormatError::ChunkedReadError(format!(
            "rank mismatch: dataspace has {} dims, layout has {} chunk dims (rank={})",
            ds_dims.len(), chunk_dimensions.len(), rank
        )));
    }

    // Populate chunk index on first access
    if !cache.has_index() {
        let chunks = match (version, chunk_index_type) {
            (3, _) => {
                collect_chunk_info(file_data, addr, ndims, offset_size, length_size)?
            }
            (4, Some(1)) => {
                let chunk_byte_size: usize = chunk_dims.iter().product::<usize>() * elem_size;
                let (csize, fmask) = if let Some(fs) = single_filtered_size {
                    (fs as u32, single_filter_mask.unwrap_or(0))
                } else {
                    (chunk_byte_size as u32, 0)
                };
                vec![ChunkInfo {
                    chunk_size: csize,
                    filter_mask: fmask,
                    offsets: vec![0u64; rank],
                    address: addr,
                }]
            }
            (4, Some(2)) => {
                let spatial_chunk_dims: Vec<u32> = chunk_dimensions[..rank].to_vec();
                generate_implicit_chunks(addr, &dataspace.dimensions, &spatial_chunk_dims, elem_size as u32)
            }
            (4, Some(3)) => {
                let spatial_chunk_dims: Vec<u32> = chunk_dimensions[..rank].to_vec();
                let header = FixedArrayHeader::parse(file_data, addr as usize, offset_size, length_size)?;
                read_fixed_array_chunks(
                    file_data, &header, &dataspace.dimensions, &spatial_chunk_dims,
                    elem_size as u32, offset_size, length_size,
                )?
            }
            (4, Some(4)) => {
                let spatial_chunk_dims: Vec<u32> = chunk_dimensions[..rank].to_vec();
                let header = ExtensibleArrayHeader::parse(file_data, addr as usize, offset_size, length_size)?;
                read_extensible_array_chunks(
                    file_data, &header, &dataspace.dimensions, &spatial_chunk_dims,
                    elem_size as u32, offset_size, length_size,
                )?
            }
            (v, idx) => {
                return Err(FormatError::ChunkedReadError(format!(
                    "unsupported chunked layout version={v}, index_type={idx:?}"
                )))
            }
        };
        cache.populate_index(&chunks, rank);
    }

    let chunks = cache.all_indexed_chunks().unwrap_or_default();

    // Assemble output
    let total_elements = dataspace.num_elements() as usize;
    let total_bytes = total_elements * elem_size;
    let mut output = vec![0u8; total_bytes];

    let mut ds_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        ds_strides[i] = ds_strides[i + 1] * ds_dims[i + 1];
    }

    let mut chunk_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        chunk_strides[i] = chunk_strides[i + 1] * chunk_dims[i + 1];
    }

    let chunk_total_elements: usize = chunk_dims.iter().product();
    let chunk_total_bytes = chunk_total_elements * elem_size;

    for chunk_info in &chunks {
        let coord: Vec<u64> = chunk_info.offsets.iter().take(rank).copied().collect();

        // Feed coordinate to sweep detector
        sweep.record(coord.clone(), rank);

        // Issue prefetch hint for predicted next chunks
        if !sweep.predicted_next.is_empty() {
            cache.prefetch_hint(&sweep.predicted_next);
            cache.set_sweep_direction(sweep.direction);
        }

        // Try decompressed cache first
        let decompressed = if let Some(cached) = cache.get_decompressed(&coord) {
            cached
        } else {
            // Decompress from file
            let c_addr = chunk_info.address as usize;
            let size = chunk_info.chunk_size as usize;
            if c_addr + size > file_data.len() {
                return Err(FormatError::UnexpectedEof {
                    expected: c_addr + size,
                    available: file_data.len(),
                });
            }
            let raw_chunk = &file_data[c_addr..c_addr + size];
            let dec = if let Some(pl) = pipeline {
                if chunk_info.filter_mask == 0 {
                    decompress_chunk(raw_chunk, pl, chunk_total_bytes, elem_size as u32)?
                } else {
                    raw_chunk.to_vec()
                }
            } else {
                raw_chunk.to_vec()
            };
            cache.put_decompressed(coord, dec.clone());
            dec
        };

        let chunk_offsets: Vec<usize> = chunk_info.offsets.iter()
            .take(rank)
            .map(|&o| o as usize)
            .collect();

        if rank == 0 {
            let copy_len = decompressed.len().min(output.len());
            output[..copy_len].copy_from_slice(&decompressed[..copy_len]);
        } else {
            copy_chunk_to_output(
                &decompressed,
                &mut output,
                &chunk_offsets,
                &chunk_dims,
                &ds_dims,
                &ds_strides,
                &chunk_strides,
                elem_size,
                rank,
            );
        }
    }

    Ok(output)
}

/// Copy chunk data into the output buffer at the correct N-D position.
#[allow(clippy::too_many_arguments)]
fn copy_chunk_to_output(
    chunk_data: &[u8],
    output: &mut [u8],
    chunk_offsets: &[usize],
    chunk_dims: &[usize],
    ds_dims: &[usize],
    ds_strides: &[usize],
    chunk_strides: &[usize],
    elem_size: usize,
    rank: usize,
) {
    // Iterate over all elements in the chunk using a flat index
    let chunk_total: usize = chunk_dims.iter().product();
    for flat_idx in 0..chunk_total {
        // Convert flat index to N-D chunk-local coordinates
        let mut remaining = flat_idx;
        let mut ds_flat = 0usize;
        let mut out_of_bounds = false;

        for d in 0..rank {
            let coord_in_chunk = remaining / chunk_strides[d];
            remaining %= chunk_strides[d];

            let global_coord = chunk_offsets[d] + coord_in_chunk;
            if global_coord >= ds_dims[d] {
                out_of_bounds = true;
                break;
            }
            ds_flat += global_coord * ds_strides[d];
        }

        if out_of_bounds {
            continue;
        }

        let src_start = flat_idx * elem_size;
        let dst_start = ds_flat * elem_size;

        if src_start + elem_size <= chunk_data.len() && dst_start + elem_size <= output.len() {
            output[dst_start..dst_start + elem_size]
                .copy_from_slice(&chunk_data[src_start..src_start + elem_size]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_offset(buf: &mut Vec<u8>, val: u64, size: u8) {
        match size {
            4 => buf.extend_from_slice(&(val as u32).to_le_bytes()),
            8 => buf.extend_from_slice(&val.to_le_bytes()),
            _ => panic!("unsupported offset size in test"),
        }
    }

    /// Build a B-tree v1 type 1 leaf node with given chunk infos.
    fn build_chunk_btree_leaf(
        chunks: &[ChunkInfo],
        ndims: usize,
        offset_size: u8,
    ) -> Vec<u8> {
        let _os = offset_size as usize;
        let entries_used = chunks.len() as u16;
        let mut buf = Vec::new();

        // Header
        buf.extend_from_slice(b"TREE");
        buf.push(1); // node_type = 1 (raw data chunks)
        buf.push(0); // node_level = 0 (leaf)
        buf.extend_from_slice(&entries_used.to_le_bytes());

        // Left/right sibling = undefined
        let undef: u64 = if offset_size == 4 {
            0xFFFFFFFF
        } else {
            0xFFFFFFFFFFFFFFFF
        };
        write_offset(&mut buf, undef, offset_size);
        write_offset(&mut buf, undef, offset_size);

        // Entries: key[i], child[i] pairs, then final key
        for chunk in chunks {
            // Key: chunk_size(4) + filter_mask(4) + ndims offsets
            buf.extend_from_slice(&chunk.chunk_size.to_le_bytes());
            buf.extend_from_slice(&chunk.filter_mask.to_le_bytes());
            for d in 0..ndims {
                let off = if d < chunk.offsets.len() {
                    chunk.offsets[d]
                } else {
                    0
                };
                write_offset(&mut buf, off, offset_size);
            }
            // Child: address
            write_offset(&mut buf, chunk.address, offset_size);
        }

        // Final key (dummy)
        buf.extend_from_slice(&0u32.to_le_bytes()); // chunk_size
        buf.extend_from_slice(&0u32.to_le_bytes()); // filter_mask
        for _ in 0..ndims {
            write_offset(&mut buf, u64::MAX, offset_size);
        }

        buf
    }

    // --- ChunkInfo collection tests ---

    #[test]
    fn collect_two_chunks_from_leaf() {
        let ndims = 2; // rank+1 for 1D dataset
        let os: u8 = 8;

        let chunks = vec![
            ChunkInfo {
                chunk_size: 80,
                filter_mask: 0,
                offsets: vec![0, 0],
                address: 0x1000,
            },
            ChunkInfo {
                chunk_size: 80,
                filter_mask: 0,
                offsets: vec![10, 0],
                address: 0x2000,
            },
        ];

        let btree = build_chunk_btree_leaf(&chunks, ndims, os);
        let mut file_data = vec![0u8; 0x3000];
        file_data[..btree.len()].copy_from_slice(&btree);

        let result = collect_chunk_info(&file_data, 0, ndims, os, os).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].address, 0x1000);
        assert_eq!(result[0].offsets, vec![0, 0]);
        assert_eq!(result[0].chunk_size, 80);
        assert_eq!(result[1].address, 0x2000);
        assert_eq!(result[1].offsets, vec![10, 0]);
    }

    #[test]
    fn collect_three_chunks() {
        let ndims = 2;
        let os: u8 = 8;

        let chunks = vec![
            ChunkInfo {
                chunk_size: 40,
                filter_mask: 0,
                offsets: vec![0, 0],
                address: 0x100,
            },
            ChunkInfo {
                chunk_size: 40,
                filter_mask: 0,
                offsets: vec![5, 0],
                address: 0x200,
            },
            ChunkInfo {
                chunk_size: 40,
                filter_mask: 0,
                offsets: vec![10, 0],
                address: 0x300,
            },
        ];

        let btree = build_chunk_btree_leaf(&chunks, ndims, os);
        let mut file_data = vec![0u8; 0x1000];
        file_data[..btree.len()].copy_from_slice(&btree);

        let result = collect_chunk_info(&file_data, 0, ndims, os, os).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].address, 0x100);
        assert_eq!(result[1].address, 0x200);
        assert_eq!(result[2].address, 0x300);
    }

    #[test]
    fn collect_empty_btree() {
        let ndims = 2;
        let os: u8 = 8;
        let btree = build_chunk_btree_leaf(&[], ndims, os);
        let mut file_data = vec![0u8; 0x1000];
        file_data[..btree.len()].copy_from_slice(&btree);

        let result = collect_chunk_info(&file_data, 0, ndims, os, os).unwrap();
        assert_eq!(result.len(), 0);
    }

    // --- Chunked read tests (synthetic) ---

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

    fn make_f32_type() -> Datatype {
        Datatype::FloatingPoint {
            size: 4,
            byte_order: DatatypeByteOrder::LittleEndian,
            bit_offset: 0,
            bit_precision: 32,
            exponent_location: 23,
            exponent_size: 8,
            mantissa_location: 0,
            mantissa_size: 23,
            exponent_bias: 127,
        }
    }

    /// Build a synthetic file with a B-tree and chunk data for a 1D uncompressed dataset.
    fn build_1d_chunked_file(
        values: &[f64],
        chunk_size_elems: usize,
    ) -> (Vec<u8>, DataLayout, Dataspace) {
        let os: u8 = 8;
        let elem_size = 8usize;
        let ndims = 2; // rank(1) + 1
        let total = values.len();

        // Place chunk data starting at offset 0x2000
        let mut file_data = vec![0u8; 0x10000];
        let mut chunk_infos = Vec::new();
        let mut data_offset = 0x2000usize;

        let mut start = 0;
        while start < total {
            let end = (start + chunk_size_elems).min(total);
            let chunk_bytes = chunk_size_elems * elem_size; // full chunk allocation

            // Write chunk data (full chunk size, padding with zeros)
            for i in start..end {
                let byte_offset = data_offset + (i - start) * elem_size;
                file_data[byte_offset..byte_offset + 8]
                    .copy_from_slice(&values[i].to_le_bytes());
            }

            chunk_infos.push(ChunkInfo {
                chunk_size: chunk_bytes as u32,
                filter_mask: 0,
                offsets: vec![start as u64, 0],
                address: data_offset as u64,
            });

            data_offset += chunk_bytes;
            start += chunk_size_elems;
        }

        // Build B-tree at offset 0x100
        let btree = build_chunk_btree_leaf(&chunk_infos, ndims, os);
        let btree_addr = 0x100usize;
        file_data[btree_addr..btree_addr + btree.len()].copy_from_slice(&btree);

        let layout = DataLayout::Chunked {
            chunk_dimensions: vec![chunk_size_elems as u32, elem_size as u32],
            btree_address: Some(btree_addr as u64),
            version: 3,
            chunk_index_type: None,
            single_chunk_filtered_size: None,
            single_chunk_filter_mask: None,
        };

        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 1,
            dimensions: vec![total as u64],
            max_dimensions: None,
        };

        (file_data, layout, dataspace)
    }

    #[test]
    fn read_1d_two_chunks_no_compression() {
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let (file_data, layout, dataspace) = build_1d_chunked_file(&values, 10);
        let datatype = make_f64_type();

        let raw = read_chunked_data(&file_data, &layout, &dataspace, &datatype, None, 8, 8)
            .unwrap();
        assert_eq!(raw.len(), 20 * 8);

        // Verify values
        for i in 0..20 {
            let val = f64::from_le_bytes(raw[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(val, i as f64);
        }
    }

    #[test]
    fn read_1d_three_chunks_partial_last() {
        // 25 elements, chunk size 10 => 3 chunks, last has only 5 valid
        let values: Vec<f64> = (0..25).map(|i| i as f64).collect();
        let (file_data, layout, dataspace) = build_1d_chunked_file(&values, 10);
        let datatype = make_f64_type();

        let raw = read_chunked_data(&file_data, &layout, &dataspace, &datatype, None, 8, 8)
            .unwrap();
        assert_eq!(raw.len(), 25 * 8);

        for i in 0..25 {
            let val = f64::from_le_bytes(raw[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(val, i as f64, "mismatch at index {i}");
        }
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn read_1d_two_chunks_with_deflate() {
        use crate::filter_pipeline::{FilterDescription, FilterPipeline, FILTER_DEFLATE};
        use crate::filters::compress_chunk;

        let os: u8 = 8;
        let elem_size = 8usize;
        let ndims = 2;
        let chunk_elems = 10usize;
        let total = 20usize;

        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![FilterDescription {
                filter_id: FILTER_DEFLATE,
                name: None,
                flags: 0,
                client_data: vec![6],
            }],
        };

        let values: Vec<f64> = (0..total).map(|i| i as f64).collect();
        let mut file_data = vec![0u8; 0x10000];
        let mut chunk_infos = Vec::new();
        let mut data_offset = 0x2000usize;

        for chunk_idx in 0..2 {
            let start = chunk_idx * chunk_elems;
            let mut chunk_bytes = Vec::new();
            for i in start..start + chunk_elems {
                chunk_bytes.extend_from_slice(&values[i].to_le_bytes());
            }
            let compressed = compress_chunk(&chunk_bytes, &pipeline, elem_size as u32).unwrap();

            file_data[data_offset..data_offset + compressed.len()]
                .copy_from_slice(&compressed);

            chunk_infos.push(ChunkInfo {
                chunk_size: compressed.len() as u32,
                filter_mask: 0,
                offsets: vec![start as u64, 0],
                address: data_offset as u64,
            });

            data_offset += compressed.len() + 16; // some padding
        }

        let btree = build_chunk_btree_leaf(&chunk_infos, ndims, os);
        let btree_addr = 0x100usize;
        file_data[btree_addr..btree_addr + btree.len()].copy_from_slice(&btree);

        let layout = DataLayout::Chunked {
            chunk_dimensions: vec![chunk_elems as u32, elem_size as u32],
            btree_address: Some(btree_addr as u64),
            version: 3,
            chunk_index_type: None,
            single_chunk_filtered_size: None,
            single_chunk_filter_mask: None,
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 1,
            dimensions: vec![total as u64],
            max_dimensions: None,
        };
        let datatype = make_f64_type();

        let raw = read_chunked_data(
            &file_data,
            &layout,
            &dataspace,
            &datatype,
            Some(&pipeline),
            8,
            8,
        )
        .unwrap();

        for i in 0..total {
            let val = f64::from_le_bytes(raw[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(val, i as f64, "mismatch at index {i}");
        }
    }

    #[test]
    fn read_2d_four_chunks() {
        // 4x6 dataset with chunk size 2x3 => 4 chunks
        let os: u8 = 8;
        let elem_size = 4usize; // f32
        let ndims = 3; // rank(2) + 1
        let ds_dims = [4usize, 6];
        let chunk_dims = [2usize, 3];

        let values: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let mut file_data = vec![0u8; 0x10000];
        let mut chunk_infos = Vec::new();
        let mut data_offset = 0x2000usize;

        // Generate chunks: (0,0), (0,3), (2,0), (2,3)
        for row_start in (0..ds_dims[0]).step_by(chunk_dims[0]) {
            for col_start in (0..ds_dims[1]).step_by(chunk_dims[1]) {
                let mut chunk_bytes = Vec::new();
                for r in 0..chunk_dims[0] {
                    for c in 0..chunk_dims[1] {
                        let gr = row_start + r;
                        let gc = col_start + c;
                        let val = if gr < ds_dims[0] && gc < ds_dims[1] {
                            values[gr * ds_dims[1] + gc]
                        } else {
                            0.0
                        };
                        chunk_bytes.extend_from_slice(&val.to_le_bytes());
                    }
                }

                let chunk_size = chunk_bytes.len();
                file_data[data_offset..data_offset + chunk_size]
                    .copy_from_slice(&chunk_bytes);

                chunk_infos.push(ChunkInfo {
                    chunk_size: chunk_size as u32,
                    filter_mask: 0,
                    offsets: vec![row_start as u64, col_start as u64, 0],
                    address: data_offset as u64,
                });

                data_offset += chunk_size + 8;
            }
        }

        let btree = build_chunk_btree_leaf(&chunk_infos, ndims, os);
        let btree_addr = 0x100usize;
        file_data[btree_addr..btree_addr + btree.len()].copy_from_slice(&btree);

        let layout = DataLayout::Chunked {
            chunk_dimensions: vec![chunk_dims[0] as u32, chunk_dims[1] as u32, elem_size as u32],
            btree_address: Some(btree_addr as u64),
            version: 3,
            chunk_index_type: None,
            single_chunk_filtered_size: None,
            single_chunk_filter_mask: None,
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 2,
            dimensions: vec![ds_dims[0] as u64, ds_dims[1] as u64],
            max_dimensions: None,
        };
        let datatype = make_f32_type();

        let raw = read_chunked_data(&file_data, &layout, &dataspace, &datatype, None, 8, 8)
            .unwrap();
        assert_eq!(raw.len(), 24 * 4);

        for i in 0..24 {
            let val = f32::from_le_bytes(raw[i * 4..(i + 1) * 4].try_into().unwrap());
            assert_eq!(val, i as f32, "mismatch at element {i}");
        }
    }

    #[test]
    fn wrong_node_type_error() {
        // Build a type-0 B-tree and try to collect chunk info
        let mut buf = Vec::new();
        buf.extend_from_slice(b"TREE");
        buf.push(0); // type 0, not 1
        buf.push(0);
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&[0xFF; 16]); // siblings
        // final key
        buf.extend_from_slice(&[0u8; 24]);

        let mut file_data = vec![0u8; 512];
        file_data[..buf.len()].copy_from_slice(&buf);

        let err = collect_chunk_info(&file_data, 0, 2, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidBTreeNodeType(0));
    }

    // --- Implicit chunk generation tests ---

    #[test]
    fn implicit_chunks_1d_five_chunks() {
        let chunks = generate_implicit_chunks(
            0x1000,
            &[100],
            &[20],
            8, // f64
        );
        assert_eq!(chunks.len(), 5);
        let chunk_byte_size = 20 * 8;
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.address, 0x1000 + i as u64 * chunk_byte_size as u64);
            assert_eq!(c.offsets, vec![i as u64 * 20]);
            assert_eq!(c.filter_mask, 0);
            assert_eq!(c.chunk_size, chunk_byte_size as u32);
        }
    }

    #[test]
    fn implicit_chunks_2d() {
        // 10x6 dataset, 4x3 chunks => ceil(10/4)=3, ceil(6/3)=2 => 6 chunks
        let chunks = generate_implicit_chunks(
            0x2000,
            &[10, 6],
            &[4, 3],
            4, // f32
        );
        assert_eq!(chunks.len(), 6);
        let chunk_byte_size = 4 * 3 * 4;
        // Row-major: (0,0), (0,3), (4,0), (4,3), (8,0), (8,3)
        assert_eq!(chunks[0].offsets, vec![0, 0]);
        assert_eq!(chunks[1].offsets, vec![0, 3]);
        assert_eq!(chunks[2].offsets, vec![4, 0]);
        assert_eq!(chunks[3].offsets, vec![4, 3]);
        assert_eq!(chunks[4].offsets, vec![8, 0]);
        assert_eq!(chunks[5].offsets, vec![8, 3]);
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.address, 0x2000 + i as u64 * chunk_byte_size as u64);
        }
    }

    #[test]
    fn implicit_chunks_partial_last() {
        // 25 elements, chunk size 10 => 3 chunks (last partial)
        let chunks = generate_implicit_chunks(0x0, &[25], &[10], 8);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].offsets, vec![0]);
        assert_eq!(chunks[1].offsets, vec![10]);
        assert_eq!(chunks[2].offsets, vec![20]);
    }

    // --- V4 single chunk synthetic test ---

    #[test]
    fn read_v4_single_chunk_synthetic() {
        // Build a synthetic v4 single chunk dataset (no filters)
        let values: Vec<f64> = vec![10.0, 20.0, 30.0];
        let elem_size = 8usize;
        let chunk_elems = 3usize;

        let mut file_data = vec![0u8; 0x2000];
        let data_addr = 0x1000usize;
        for (i, &v) in values.iter().enumerate() {
            file_data[data_addr + i * elem_size..data_addr + (i + 1) * elem_size]
                .copy_from_slice(&v.to_le_bytes());
        }

        let layout = DataLayout::Chunked {
            chunk_dimensions: vec![chunk_elems as u32, elem_size as u32],
            btree_address: Some(data_addr as u64),
            version: 4,
            chunk_index_type: Some(1),
            single_chunk_filtered_size: None,
            single_chunk_filter_mask: None,
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 1,
            dimensions: vec![3],
            max_dimensions: None,
        };
        let datatype = make_f64_type();

        let raw = read_chunked_data(&file_data, &layout, &dataspace, &datatype, None, 8, 8)
            .unwrap();
        assert_eq!(raw.len(), 24);
        for i in 0..3 {
            let val = f64::from_le_bytes(raw[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(val, values[i]);
        }
    }

    // --- Cached read tests ---

    use crate::chunk_cache::ChunkCache;

    #[test]
    fn cached_read_populates_index_and_returns_correct_data() {
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let (file_data, layout, dataspace) = build_1d_chunked_file(&values, 10);
        let datatype = make_f64_type();
        let cache = ChunkCache::new();

        assert!(!cache.has_index());
        let raw = read_chunked_data_cached(
            &file_data, &layout, &dataspace, &datatype, None, 8, 8, &cache,
        ).unwrap();
        assert!(cache.has_index());
        assert_eq!(raw.len(), 20 * 8);
        for i in 0..20 {
            let val = f64::from_le_bytes(raw[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(val, i as f64);
        }
    }

    #[test]
    fn cached_read_second_call_uses_cache() {
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let (file_data, layout, dataspace) = build_1d_chunked_file(&values, 10);
        let datatype = make_f64_type();
        let cache = ChunkCache::new();

        // First read — populates index + decompressed cache
        let raw1 = read_chunked_data_cached(
            &file_data, &layout, &dataspace, &datatype, None, 8, 8, &cache,
        ).unwrap();
        assert!(cache.has_index());
        assert!(cache.cached_chunk_count() > 0);

        // Second read — should hit the decompressed cache
        let raw2 = read_chunked_data_cached(
            &file_data, &layout, &dataspace, &datatype, None, 8, 8, &cache,
        ).unwrap();
        assert_eq!(raw1, raw2);
    }

    #[test]
    fn cached_read_with_partial_last_chunk() {
        let values: Vec<f64> = (0..25).map(|i| i as f64).collect();
        let (file_data, layout, dataspace) = build_1d_chunked_file(&values, 10);
        let datatype = make_f64_type();
        let cache = ChunkCache::new();

        let raw = read_chunked_data_cached(
            &file_data, &layout, &dataspace, &datatype, None, 8, 8, &cache,
        ).unwrap();
        assert_eq!(raw.len(), 25 * 8);
        for i in 0..25 {
            let val = f64::from_le_bytes(raw[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(val, i as f64, "mismatch at index {i}");
        }
    }

    // --- Sweep-aware read tests ---

    #[test]
    fn sweep_read_returns_correct_data() {
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let (file_data, layout, dataspace) = build_1d_chunked_file(&values, 10);
        let datatype = make_f64_type();
        let cache = ChunkCache::new();
        let mut sweep = SweepContext::with_defaults();

        let raw = read_chunked_data_sweep(
            &file_data, &layout, &dataspace, &datatype, None, 8, 8, &cache, &mut sweep,
        ).unwrap();
        assert_eq!(raw.len(), 20 * 8);
        for i in 0..20 {
            let val = f64::from_le_bytes(raw[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(val, i as f64);
        }
    }

    #[test]
    fn sweep_read_populates_sweep_context() {
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let (file_data, layout, dataspace) = build_1d_chunked_file(&values, 10);
        let datatype = make_f64_type();
        let cache = ChunkCache::new();
        let mut sweep = SweepContext::with_defaults();

        read_chunked_data_sweep(
            &file_data, &layout, &dataspace, &datatype, None, 8, 8, &cache, &mut sweep,
        ).unwrap();

        // After reading 2 chunks (offsets [0] and [10]), history should be populated
        assert!(!sweep.history.is_empty());
    }

    #[test]
    fn sweep_context_unit_test() {
        let mut ctx = SweepContext::with_defaults();
        ctx.record(vec![0, 0], 2);
        ctx.record(vec![0, 10], 2);
        ctx.record(vec![0, 20], 2);
        assert_eq!(ctx.direction, "row_major");
        assert!(!ctx.predicted_next.is_empty());
        assert_eq!(ctx.predicted_next[0], vec![0, 30]);
    }

    #[test]
    fn sweep_context_random() {
        let mut ctx = SweepContext::with_defaults();
        ctx.record(vec![0, 0], 2);
        ctx.record(vec![30, 20], 2);
        ctx.record(vec![10, 0], 2);
        assert_eq!(ctx.direction, "random");
        assert!(ctx.predicted_next.is_empty());
    }

    #[test]
    fn sweep_read_access_stats() {
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let (file_data, layout, dataspace) = build_1d_chunked_file(&values, 10);
        let datatype = make_f64_type();
        let cache = ChunkCache::new();
        let mut sweep = SweepContext::with_defaults();

        read_chunked_data_sweep(
            &file_data, &layout, &dataspace, &datatype, None, 8, 8, &cache, &mut sweep,
        ).unwrap();

        let stats = cache.access_stats();
        // We accessed 2 chunks; the second should be sequential to the first
        assert!(stats.sequential_count > 0 || stats.random_count > 0);
    }
}
