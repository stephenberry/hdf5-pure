//! Chunked dataset reading: B-tree v1 type 1 traversal and chunk assembly.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};

use crate::btree_v1::btree_v1_node_header_size;
use crate::chunk_cache::ChunkCache;
use crate::convert::{TryToUsize, slice_range, u32_from};
use crate::data_layout::DataLayout;
use crate::dataspace::Dataspace;
use crate::datatype::Datatype;
use crate::error::FormatError;
use crate::extensible_array::{
    ExtensibleArrayHeader, read_extensible_array_chunks, read_extensible_array_chunks_from_source,
};
use crate::filter_pipeline::FilterPipeline;
use crate::filters::{ChunkContext, decompress_chunk};
use crate::fixed_array::{
    FixedArrayHeader, read_fixed_array_chunks, read_fixed_array_chunks_from_source,
};
use crate::source::FileSource;

#[cfg(feature = "parallel")]
use crate::parallel_read;

/// Decompress all chunks, using lane-partitioned parallel decompression when the
/// `parallel` feature is enabled and the chunk count exceeds the threshold.
fn decompress_all_chunks(
    file_data: &[u8],
    chunks: &[ChunkInfo],
    pipeline: Option<&FilterPipeline>,
    ctx: ChunkContext<'_>,
) -> Result<Vec<Vec<u8>>, FormatError> {
    #[cfg(feature = "parallel")]
    {
        if let Some(pl) = pipeline {
            if parallel_read::should_use_parallel(chunks.len()) {
                // Seed from the first chunk's address and count for determinism.
                let seed = chunks.first().map(|c| c.address).unwrap_or(0) ^ (chunks.len() as u64);
                let (data, _stats) = parallel_read::decompress_chunks_lane_partitioned(
                    file_data, chunks, pl, ctx, seed, None, // auto-detect lane count
                )?;
                return Ok(data);
            }
        }
    }

    let mut result = Vec::with_capacity(chunks.len());
    for chunk_info in chunks {
        let r = slice_range(chunk_info.address, u64::from(chunk_info.chunk_size))?;
        if r.end > file_data.len() {
            return Err(FormatError::UnexpectedEof {
                expected: r.end,
                available: file_data.len(),
            });
        }
        let raw_chunk = &file_data[r];

        let decompressed = if let Some(pl) = pipeline {
            decompress_chunk(raw_chunk, pl, ctx, chunk_info.filter_mask)?
        } else {
            raw_chunk.to_vec()
        };
        result.push(decompressed);
    }
    Ok(result)
}

/// Decompress all chunks, reading each chunk's bytes from a [`FileSource`].
///
/// Streaming counterpart of [`decompress_all_chunks`]. Sequential only: the
/// lane-partitioned parallel path borrows a whole-file `&[u8]` inside a `Send`
/// rayon closure, and a `ReadSeekSource` serializes on its mutex anyway, so a
/// parallel streaming variant is a separate follow-up.
fn decompress_all_chunks_from_source<S: FileSource + ?Sized>(
    source: &S,
    chunks: &[ChunkInfo],
    pipeline: Option<&FilterPipeline>,
    ctx: ChunkContext<'_>,
) -> Result<Vec<Vec<u8>>, FormatError> {
    let mut result = Vec::with_capacity(chunks.len());
    for chunk_info in chunks {
        let raw_chunk =
            source.read_exact_at(chunk_info.address, chunk_info.chunk_size.to_usize()?)?;
        let decompressed = if let Some(pl) = pipeline {
            decompress_chunk(&raw_chunk, pl, ctx, chunk_info.filter_mask)?
        } else {
            raw_chunk
        };
        result.push(decompressed);
    }
    Ok(result)
}

/// Information about a single chunk in a chunked dataset.
#[derive(Debug, Clone, PartialEq, Eq)]
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
    if s > data.len() || pos > data.len() - s {
        return Err(FormatError::UnexpectedEof {
            expected: pos.saturating_add(s),
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

/// Size of a single key in a version 1 B-tree of type 1 (raw data chunks):
/// `chunk_size(4) + filter_mask(4) + ndims * offset_size`. The `ndims` trailing
/// offsets are the per-dimension scaled chunk coordinates (rank + 1 values, the
/// extra one being the element-offset dimension). (HDF5 format spec, "Disk
/// Format: Level 1A1 — Version 1 B-trees", type 1 key.)
const fn chunk_record_key_size(ndims: usize, offset_size: usize) -> usize {
    4 + 4 + ndims * offset_size
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
    length_size: u8,
) -> Result<Vec<ChunkInfo>, FormatError> {
    collect_chunk_info_inner(file_data, btree_address, ndims, offset_size, length_size, 0)
}

/// Depth-tracking core of [`collect_chunk_info`]. The `depth` guard mirrors
/// [`collect_chunk_btree_node_spans_inner`]: a cyclic or pathologically deep
/// internal node in a foreign file errors out instead of recursing until the
/// stack overflows (which would abort the process uncatchably).
fn collect_chunk_info_inner(
    file_data: &[u8],
    btree_address: u64,
    ndims: usize,
    offset_size: u8,
    _length_size: u8,
    depth: u32,
) -> Result<Vec<ChunkInfo>, FormatError> {
    if depth > MAX_CHUNK_BTREE_DEPTH {
        return Err(FormatError::ChunkedReadError(
            "chunk B-tree nested too deeply".into(),
        ));
    }
    let offset = btree_address.to_usize()?;
    let os = offset_size as usize;

    // Parse B-tree v1 header
    let header_size = btree_v1_node_header_size(offset_size);
    if header_size > file_data.len() || offset > file_data.len() - header_size {
        return Err(FormatError::UnexpectedEof {
            expected: offset.saturating_add(header_size),
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
    let entries_used = u16::from_le_bytes([file_data[offset + 6], file_data[offset + 7]]) as usize;

    let mut pos = offset + header_size; // first key, past signature/siblings

    let key_size = chunk_record_key_size(ndims, os);

    if node_level == 0 {
        // Leaf node: keys and children interleaved
        // key[0], child[0], key[1], child[1], ..., key[N-1], child[N-1], key[N]
        let needed = entries_used * (key_size + os) + key_size;
        if needed > file_data.len() || pos > file_data.len() - needed {
            return Err(FormatError::UnexpectedEof {
                expected: pos.saturating_add(needed),
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
        if needed > file_data.len() || pos > file_data.len() - needed {
            return Err(FormatError::UnexpectedEof {
                expected: pos.saturating_add(needed),
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
            let child_chunks = collect_chunk_info_inner(
                file_data,
                child_addr,
                ndims,
                offset_size,
                _length_size,
                depth + 1,
            )?;
            all_chunks.extend(child_chunks);
        }
        Ok(all_chunks)
    }
}

/// Recursion-depth cap for the chunk B-tree node-span walk, guarding against a
/// stack overflow on a cyclic or pathological index in a foreign file. A real
/// chunk B-tree is only a few levels deep, so this is far beyond any valid tree.
const MAX_CHUNK_BTREE_DEPTH: u32 = 64;

/// Enumerate the on-disk byte spans of every node in a version 1 B-tree of
/// type 1 (the raw-data chunk index), so a deleted chunked dataset's index can
/// be reclaimed. Returns one `(addr, len)` per `TREE` node (internal and leaf);
/// the chunk *data* blocks the leaves point at are enumerated separately by
/// [`collect_chunk_info`].
///
/// Each node's length is sized from `entries_used` — the region the reader
/// consumes (`header + entries_used * (key_size + offset_size) + key_size`).
/// HDF5 allocates each node at its full `2K` capacity, so this is a lower bound:
/// a partially-filled node's trailing slack is left unreclaimed rather than
/// guessed at, honoring the editor's "under-reclaim, never over-reclaim" rule.
///
/// `ndims` is the number of offset dimensions in each key (rank + 1), the same
/// value [`collect_chunk_info`] takes. Errors (a malformed node, an out-of-bounds
/// child, or a tree deeper than [`MAX_CHUNK_BTREE_DEPTH`]) propagate so the
/// caller can leave the whole index unreclaimed.
pub(crate) fn collect_chunk_btree_node_spans(
    file_data: &[u8],
    btree_address: u64,
    ndims: usize,
    offset_size: u8,
) -> Result<Vec<(u64, u64)>, FormatError> {
    let mut out = Vec::new();
    collect_chunk_btree_node_spans_inner(
        file_data,
        btree_address,
        ndims,
        offset_size,
        0,
        &mut out,
    )?;
    Ok(out)
}

fn collect_chunk_btree_node_spans_inner(
    file_data: &[u8],
    btree_address: u64,
    ndims: usize,
    offset_size: u8,
    depth: u32,
    out: &mut Vec<(u64, u64)>,
) -> Result<(), FormatError> {
    if depth > MAX_CHUNK_BTREE_DEPTH {
        return Err(FormatError::ChunkedReadError(
            "chunk B-tree nested too deeply".into(),
        ));
    }
    let offset = btree_address.to_usize()?;
    let os = offset_size as usize;
    let header_size = btree_v1_node_header_size(offset_size);
    if header_size > file_data.len() || offset > file_data.len() - header_size {
        return Err(FormatError::UnexpectedEof {
            expected: offset.saturating_add(header_size),
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
    let entries_used = u16::from_le_bytes([file_data[offset + 6], file_data[offset + 7]]) as usize;
    let key_size = chunk_record_key_size(ndims, os);

    // The reader-consumed body: `entries_used` (key, child) pairs and a trailing
    // key. This is the conservative node extent (see the doc comment).
    let body = entries_used
        .checked_mul(key_size + os)
        .and_then(|b| b.checked_add(key_size))
        .ok_or(FormatError::OffsetOverflow {
            offset: entries_used as u64,
            length: (key_size + os) as u64,
        })?;
    let node_len = header_size + body;
    if node_len > file_data.len() || offset > file_data.len() - node_len {
        return Err(FormatError::UnexpectedEof {
            expected: offset.saturating_add(node_len),
            available: file_data.len(),
        });
    }
    out.push((btree_address, node_len as u64));

    // Internal nodes (level > 0) hold child node addresses, not chunk records;
    // recurse so their nodes are reclaimed too.
    if node_level != 0 {
        let mut pos = offset + header_size;
        for _ in 0..entries_used {
            pos += key_size; // skip the key
            let child_addr = read_offset(file_data, pos, offset_size)?;
            pos += os;
            collect_chunk_btree_node_spans_inner(
                file_data,
                child_addr,
                ndims,
                offset_size,
                depth + 1,
                out,
            )?;
        }
    }
    Ok(())
}

/// Traverse B-tree v1 type 1 from a [`FileSource`], reading each node on demand.
///
/// The streaming counterpart of [`collect_chunk_info`]: it reads the node header
/// and the key/child region as two bounded windows via `read_at`, recursing into
/// child nodes by reading their regions, so the index can be walked without a
/// whole-file buffer.
pub fn collect_chunk_info_from_source<S: FileSource + ?Sized>(
    source: &S,
    btree_address: u64,
    ndims: usize,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<ChunkInfo>, FormatError> {
    collect_chunk_info_from_source_inner(source, btree_address, ndims, offset_size, length_size, 0)
}

/// Depth-tracking core of [`collect_chunk_info_from_source`]; see
/// [`collect_chunk_info_inner`] for why the bound is required.
fn collect_chunk_info_from_source_inner<S: FileSource + ?Sized>(
    source: &S,
    btree_address: u64,
    ndims: usize,
    offset_size: u8,
    _length_size: u8,
    depth: u32,
) -> Result<Vec<ChunkInfo>, FormatError> {
    if depth > MAX_CHUNK_BTREE_DEPTH {
        return Err(FormatError::ChunkedReadError(
            "chunk B-tree nested too deeply".into(),
        ));
    }
    let os = offset_size as usize;
    let header_size = btree_v1_node_header_size(offset_size);

    // Node header: signature(4) + type(1) + level(1) + entries_used(2) + 2 siblings.
    let header = source.read_metadata_at(btree_address, header_size)?;
    if &header[0..4] != b"TREE" {
        return Err(FormatError::InvalidBTreeSignature);
    }
    let node_type = header[4];
    if node_type != 1 {
        return Err(FormatError::InvalidBTreeNodeType(node_type));
    }
    let node_level = header[5];
    let entries_used = u16::from_le_bytes([header[6], header[7]]) as usize;

    // Key/child region begins right after the header (siblings already included).
    let key_size = chunk_record_key_size(ndims, os);
    let needed = entries_used * (key_size + os) + key_size;
    let body_addr =
        btree_address
            .checked_add(header_size as u64)
            .ok_or(FormatError::OffsetOverflow {
                offset: btree_address,
                length: header_size as u64,
            })?;
    let body = source.read_metadata_at(body_addr, needed)?;
    let mut pos = 0usize;

    if node_level == 0 {
        let mut chunks = Vec::with_capacity(entries_used);
        for _ in 0..entries_used {
            let chunk_size =
                u32::from_le_bytes([body[pos], body[pos + 1], body[pos + 2], body[pos + 3]]);
            let filter_mask =
                u32::from_le_bytes([body[pos + 4], body[pos + 5], body[pos + 6], body[pos + 7]]);
            let mut offsets = Vec::with_capacity(ndims);
            let mut kp = pos + 8;
            for _ in 0..ndims {
                offsets.push(read_offset(&body, kp, offset_size)?);
                kp += os;
            }
            pos += key_size;
            let address = read_offset(&body, pos, offset_size)?;
            pos += os;
            chunks.push(ChunkInfo {
                chunk_size,
                filter_mask,
                offsets,
                address,
            });
        }
        Ok(chunks)
    } else {
        let mut child_addrs = Vec::with_capacity(entries_used);
        for _ in 0..entries_used {
            pos += key_size; // skip key
            child_addrs.push(read_offset(&body, pos, offset_size)?);
            pos += os;
        }
        let mut all_chunks = Vec::new();
        for child_addr in child_addrs {
            all_chunks.extend(collect_chunk_info_from_source_inner(
                source,
                child_addr,
                ndims,
                offset_size,
                _length_size,
                depth + 1,
            )?);
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
    let chunk_byte_size: u64 =
        chunk_dimensions.iter().map(|&d| d as u64).product::<u64>() * element_size as u64;

    let mut num_chunks_per_dim = Vec::with_capacity(rank);
    for d in 0..rank {
        let ds = dataset_dims[d];
        let ch = chunk_dimensions[d] as u64;
        num_chunks_per_dim.push(ds.div_ceil(ch));
    }
    let total_chunks: u64 = num_chunks_per_dim.iter().product();

    #[expect(
        clippy::cast_possible_truncation,
        reason = "with_capacity hint only; total_chunks is bounded by the dataset's chunk \
                  grid and a truncated hint merely under-reserves (the Vec still grows)"
    )]
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

        #[expect(
            clippy::cast_possible_truncation,
            reason = "chunk byte size is stored in the 32-bit ChunkInfo.chunk_size field \
                      (HDF5 caps a chunk at 4 GiB)"
        )]
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
    let (
        chunk_dimensions,
        version,
        chunk_index_type,
        addr_opt,
        single_filtered_size,
        single_filter_mask,
    ) = match layout {
        DataLayout::Chunked {
            chunk_dimensions,
            btree_address,
            version,
            chunk_index_type,
            single_chunk_filtered_size,
            single_chunk_filter_mask,
        } => (
            chunk_dimensions,
            *version,
            *chunk_index_type,
            *btree_address,
            *single_chunk_filtered_size,
            *single_chunk_filter_mask,
        ),
        _ => {
            return Err(FormatError::ChunkedReadError(
                "expected chunked layout".into(),
            ));
        }
    };

    let addr = addr_opt
        .ok_or_else(|| FormatError::ChunkedReadError("no address for chunked layout".into()))?;

    let elem_size = datatype.type_size() as usize;

    // Both v3 and v4 include element size as last dim (rank+1)
    let ndims = chunk_dimensions.len();
    let rank = ndims
        .checked_sub(1)
        .ok_or_else(|| FormatError::ChunkedReadError("chunked layout has no dimensions".into()))?;
    let chunk_dims: Vec<usize> = chunk_dimensions[..rank]
        .iter()
        .map(|&d| d as usize)
        .collect();

    let ds_dims: Vec<usize> = dataspace
        .dimensions
        .iter()
        .map(|&d| d.to_usize())
        .collect::<Result<_, _>>()?;
    if ds_dims.len() != rank {
        return Err(FormatError::ChunkedReadError(format!(
            "rank mismatch: dataspace has {} dims, layout has {} chunk dims (rank={})",
            ds_dims.len(),
            chunk_dimensions.len(),
            rank
        )));
    }

    // Collect chunks based on version and index type
    #[expect(
        clippy::cast_possible_truncation,
        reason = "chunk byte sizes and the datatype element size are encoded into 32-bit \
                  chunk-info fields; both stay well below u32::MAX (HDF5 caps a chunk at 4 GiB)"
    )]
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
            let header =
                FixedArrayHeader::parse(file_data, addr.to_usize()?, offset_size, length_size)?;
            read_fixed_array_chunks(
                file_data,
                &header,
                &dataspace.dimensions,
                &spatial_chunk_dims,
                elem_size as u32,
                offset_size,
                length_size,
            )?
        }
        (4, Some(4)) => {
            // Extensible Array — use spatial chunk dims only
            let spatial_chunk_dims: Vec<u32> = chunk_dimensions[..rank].to_vec();
            let header = ExtensibleArrayHeader::parse(
                file_data,
                addr.to_usize()?,
                offset_size,
                length_size,
            )?;
            read_extensible_array_chunks(
                file_data,
                &header,
                &dataspace.dimensions,
                &spatial_chunk_dims,
                elem_size as u32,
                offset_size,
                length_size,
            )?
        }
        (v, idx) => {
            return Err(FormatError::ChunkedReadError(format!(
                "unsupported chunked layout version={v}, index_type={idx:?}"
            )));
        }
    };

    // Decompress all chunks (parallel when beneficial, sequential otherwise)
    let chunk_dims_u64: Vec<u64> = chunk_dims.iter().map(|&d| d as u64).collect();
    let ctx = ChunkContext::from_datatype(&chunk_dims_u64, datatype);
    let decompressed_chunks = decompress_all_chunks(file_data, &chunks, pipeline, ctx)?;

    let total_bytes = dataspace.num_elements().to_usize()? * elem_size;
    Ok(assemble_chunks(
        &chunks,
        &decompressed_chunks,
        rank,
        &chunk_dims,
        &ds_dims,
        elem_size,
        total_bytes,
    ))
}

/// Read a chunked dataset from a [`FileSource`], reading the chunk index and
/// each chunk's bytes on demand via `read_at`.
///
/// Streaming counterpart of [`read_chunked_data`], with the same chunk-index
/// coverage: the B-tree v1 (v3) index and the v4 single-chunk, implicit,
/// Fixed-Array, and Extensible-Array indexes (index types 1-4). A v4 index
/// type 5 (version-2 B-tree) is not supported, matching the buffered reader.
/// The decompression is sequential.
pub fn read_chunked_data_from_source<S: FileSource + ?Sized>(
    source: &S,
    layout: &DataLayout,
    dataspace: &Dataspace,
    datatype: &Datatype,
    pipeline: Option<&FilterPipeline>,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<u8>, FormatError> {
    let (
        chunk_dimensions,
        version,
        chunk_index_type,
        addr_opt,
        single_filtered_size,
        single_filter_mask,
    ) = match layout {
        DataLayout::Chunked {
            chunk_dimensions,
            btree_address,
            version,
            chunk_index_type,
            single_chunk_filtered_size,
            single_chunk_filter_mask,
        } => (
            chunk_dimensions,
            *version,
            *chunk_index_type,
            *btree_address,
            *single_chunk_filtered_size,
            *single_chunk_filter_mask,
        ),
        _ => {
            return Err(FormatError::ChunkedReadError(
                "expected chunked layout".into(),
            ));
        }
    };

    let addr = addr_opt
        .ok_or_else(|| FormatError::ChunkedReadError("no address for chunked layout".into()))?;

    let elem_size = datatype.type_size() as usize;
    let ndims = chunk_dimensions.len();
    let rank = ndims
        .checked_sub(1)
        .ok_or_else(|| FormatError::ChunkedReadError("chunked layout has no dimensions".into()))?;
    // `chunk_dimensions` are u32, so this widens; the dataspace dims are u64 and
    // are narrowed with a checked conversion.
    let chunk_dims: Vec<usize> = chunk_dimensions[..rank]
        .iter()
        .map(|&d| d as usize)
        .collect();
    let ds_dims: Vec<usize> = dataspace
        .dimensions
        .iter()
        .map(|&d| d.to_usize())
        .collect::<Result<_, _>>()?;
    if ds_dims.len() != rank {
        return Err(FormatError::ChunkedReadError(format!(
            "rank mismatch: dataspace has {} dims, layout has {} chunk dims (rank={})",
            ds_dims.len(),
            chunk_dimensions.len(),
            rank
        )));
    }

    let chunks = collect_chunks_for_layout_from_source(
        source,
        version,
        chunk_index_type,
        addr,
        single_filtered_size,
        single_filter_mask,
        chunk_dimensions,
        dataspace,
        elem_size,
        offset_size,
        length_size,
    )?;

    let chunk_dims_u64: Vec<u64> = chunk_dims.iter().map(|&d| d as u64).collect();
    let ctx = ChunkContext::from_datatype(&chunk_dims_u64, datatype);
    let decompressed_chunks = decompress_all_chunks_from_source(source, &chunks, pipeline, ctx)?;
    let total_bytes = dataspace.num_elements().to_usize()? * elem_size;
    Ok(assemble_chunks(
        &chunks,
        &decompressed_chunks,
        rank,
        &chunk_dims,
        &ds_dims,
        elem_size,
        total_bytes,
    ))
}

/// Read a chunked dataset from a [`FileSource`] with parsed-index and
/// decompressed-chunk caching.
///
/// This is the streaming counterpart of [`read_chunked_data_cached`].
#[allow(clippy::too_many_arguments)]
pub fn read_chunked_data_cached_from_source<S: FileSource + ?Sized>(
    source: &S,
    layout: &DataLayout,
    dataspace: &Dataspace,
    datatype: &Datatype,
    pipeline: Option<&FilterPipeline>,
    offset_size: u8,
    length_size: u8,
    cache: &ChunkCache,
) -> Result<Vec<u8>, FormatError> {
    let (
        chunk_dimensions,
        version,
        chunk_index_type,
        addr_opt,
        single_filtered_size,
        single_filter_mask,
    ) = match layout {
        DataLayout::Chunked {
            chunk_dimensions,
            btree_address,
            version,
            chunk_index_type,
            single_chunk_filtered_size,
            single_chunk_filter_mask,
        } => (
            chunk_dimensions,
            *version,
            *chunk_index_type,
            *btree_address,
            *single_chunk_filtered_size,
            *single_chunk_filter_mask,
        ),
        _ => {
            return Err(FormatError::ChunkedReadError(
                "expected chunked layout".into(),
            ));
        }
    };

    let addr = addr_opt
        .ok_or_else(|| FormatError::ChunkedReadError("no address for chunked layout".into()))?;

    let elem_size = datatype.type_size() as usize;
    let ndims = chunk_dimensions.len();
    let rank = ndims
        .checked_sub(1)
        .ok_or_else(|| FormatError::ChunkedReadError("chunked layout has no dimensions".into()))?;
    let chunk_dims: Vec<usize> = chunk_dimensions[..rank]
        .iter()
        .map(|&d| d as usize)
        .collect();
    let ds_dims: Vec<usize> = dataspace
        .dimensions
        .iter()
        .map(|&d| d.to_usize())
        .collect::<Result<_, _>>()?;
    if ds_dims.len() != rank {
        return Err(FormatError::ChunkedReadError(format!(
            "rank mismatch: dataspace has {} dims, layout has {} chunk dims (rank={})",
            ds_dims.len(),
            chunk_dimensions.len(),
            rank
        )));
    }

    let chunks = if let Some(chunks) = cache.all_indexed_chunks() {
        chunks
    } else {
        let chunks = collect_chunks_for_layout_from_source(
            source,
            version,
            chunk_index_type,
            addr,
            single_filtered_size,
            single_filter_mask,
            chunk_dimensions,
            dataspace,
            elem_size,
            offset_size,
            length_size,
        )?;
        cache.populate_index(&chunks, rank);
        chunks
    };

    let total_elements = dataspace.num_elements().to_usize()?;
    let total_bytes = total_elements
        .checked_mul(elem_size)
        .ok_or(FormatError::OffsetOverflow {
            offset: total_elements as u64,
            length: elem_size as u64,
        })?;
    let mut output = vec![0u8; total_bytes];

    let mut ds_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        ds_strides[i] = ds_strides[i + 1] * ds_dims[i + 1];
    }

    let mut chunk_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        chunk_strides[i] = chunk_strides[i + 1] * chunk_dims[i + 1];
    }

    let chunk_dims_u64: Vec<u64> = chunk_dims.iter().map(|&d| d as u64).collect();
    let ctx = ChunkContext::from_datatype(&chunk_dims_u64, datatype);

    for chunk_info in &chunks {
        let coord: Vec<u64> = chunk_info.offsets.iter().take(rank).copied().collect();
        let chunk_offsets: Vec<usize> = chunk_info
            .offsets
            .iter()
            .take(rank)
            .map(|&o| o.to_usize())
            .collect::<Result<_, _>>()?;

        // Scatter straight from the cached chunk under the lock (no copy out).
        let hit = cache.with_decompressed(&coord, |bytes| {
            place_chunk(
                bytes,
                &mut output,
                &chunk_offsets,
                &chunk_dims,
                &ds_dims,
                &ds_strides,
                &chunk_strides,
                elem_size,
                rank,
            );
        });
        if hit.is_some() {
            continue;
        }

        // Cache miss: fetch the chunk's bytes from the source (already owned).
        let raw_chunk =
            source.read_exact_at(chunk_info.address, chunk_info.chunk_size.to_usize()?)?;
        let dec = if let Some(pl) = pipeline {
            decompress_chunk(&raw_chunk, pl, ctx, chunk_info.filter_mask)?
        } else {
            raw_chunk
        };
        place_chunk(
            &dec,
            &mut output,
            &chunk_offsets,
            &chunk_dims,
            &ds_dims,
            &ds_strides,
            &chunk_strides,
            elem_size,
            rank,
        );
        cache.put_decompressed(coord, dec); // move; dropped if not admitted
    }

    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn collect_chunks_for_layout_from_source<S: FileSource + ?Sized>(
    source: &S,
    version: u8,
    chunk_index_type: Option<u8>,
    addr: u64,
    single_filtered_size: Option<u64>,
    single_filter_mask: Option<u32>,
    chunk_dimensions: &[u32],
    dataspace: &Dataspace,
    elem_size: usize,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<ChunkInfo>, FormatError> {
    let ndims = chunk_dimensions.len();
    let rank = ndims
        .checked_sub(1)
        .ok_or_else(|| FormatError::ChunkedReadError("chunked layout has no dimensions".into()))?;
    match (version, chunk_index_type) {
        (3, _) => collect_chunk_info_from_source(source, addr, ndims, offset_size, length_size),
        (4, Some(1)) => {
            let chunk_byte_size: usize = chunk_dimensions[..rank]
                .iter()
                .map(|&d| d as usize)
                .product::<usize>()
                * elem_size;
            let (csize, fmask) = if let Some(fs) = single_filtered_size {
                (u32_from(fs)?, single_filter_mask.unwrap_or(0))
            } else {
                (u32_from(chunk_byte_size as u64)?, 0)
            };
            Ok(vec![ChunkInfo {
                chunk_size: csize,
                filter_mask: fmask,
                offsets: vec![0u64; rank],
                address: addr,
            }])
        }
        (4, Some(2)) => {
            let spatial_chunk_dims: Vec<u32> = chunk_dimensions[..rank].to_vec();
            Ok(generate_implicit_chunks(
                addr,
                &dataspace.dimensions,
                &spatial_chunk_dims,
                u32_from(elem_size as u64)?,
            ))
        }
        (4, Some(3)) => {
            let spatial_chunk_dims: Vec<u32> = chunk_dimensions[..rank].to_vec();
            let header =
                FixedArrayHeader::parse_from_source(source, addr, offset_size, length_size)?;
            read_fixed_array_chunks_from_source(
                source,
                &header,
                &dataspace.dimensions,
                &spatial_chunk_dims,
                u32_from(elem_size as u64)?,
                offset_size,
                length_size,
            )
        }
        (4, Some(4)) => {
            let spatial_chunk_dims: Vec<u32> = chunk_dimensions[..rank].to_vec();
            let header =
                ExtensibleArrayHeader::parse_from_source(source, addr, offset_size, length_size)?;
            read_extensible_array_chunks_from_source(
                source,
                &header,
                &dataspace.dimensions,
                &spatial_chunk_dims,
                u32_from(elem_size as u64)?,
                offset_size,
                length_size,
            )
        }
        (v, idx) => Err(FormatError::ChunkedReadError(format!(
            "unsupported chunked layout version={v}, index_type={idx:?}"
        ))),
    }
}

/// Enumerate every allocated chunk of a chunked dataset from an in-memory file
/// image, one [`ChunkInfo`] per chunk (file address, on-disk stored size, filter
/// mask, logical offsets). A buffered convenience wrapper over
/// [`collect_chunks_for_layout_from_source`] for callers that hold the whole
/// file as a byte slice (the in-place editor): it accepts a parsed
/// [`DataLayout::Chunked`] and the dataset's [`Dataspace`] and reads through a
/// [`BytesSource`](crate::source::BytesSource).
///
/// Returns an empty vector when the index address is undefined (a chunked
/// dataset with no storage allocated yet). Errors — propagated from the index
/// walkers — for a version-2 B-tree (index type 5) or any unknown index type,
/// which have no walker.
#[cfg(feature = "std")]
pub(crate) fn enumerate_chunks_buffered(
    file_data: &[u8],
    layout: &DataLayout,
    dataspace: &Dataspace,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<ChunkInfo>, FormatError> {
    let DataLayout::Chunked {
        chunk_dimensions,
        btree_address,
        version,
        chunk_index_type,
        single_chunk_filtered_size,
        single_chunk_filter_mask,
    } = layout
    else {
        return Err(FormatError::ChunkedReadError(
            "enumerate_chunks_buffered called on a non-chunked layout".into(),
        ));
    };
    // An undefined index address means no storage is allocated yet.
    let Some(index_addr) = *btree_address else {
        return Ok(Vec::new());
    };
    if chunk_dimensions.is_empty() {
        return Err(FormatError::ChunkedReadError(
            "chunked layout has no dimensions".into(),
        ));
    }
    // `chunk_dimensions` is rank + 1 entries; the last is the element size.
    let rank = chunk_dimensions
        .len()
        .checked_sub(1)
        .ok_or_else(|| FormatError::ChunkedReadError("chunked layout has no dimensions".into()))?;
    let elem_size = chunk_dimensions[rank] as usize;
    if elem_size == 0 {
        return Err(FormatError::ChunkedReadError(
            "chunked layout has a zero element size".into(),
        ));
    }
    let source = crate::source::BytesSource::new(file_data);
    collect_chunks_for_layout_from_source(
        &source,
        *version,
        *chunk_index_type,
        index_addr,
        *single_chunk_filtered_size,
        *single_chunk_filter_mask,
        chunk_dimensions,
        dataspace,
        elem_size,
        offset_size,
        length_size,
    )
}

/// A chunked dataset's chunks mapped onto a dense logical grid: every grid slot
/// filled exactly once, ordered row-major (last dimension fastest) so the order
/// matches [`split_into_chunks`](crate::chunked_write::split_into_chunks) and the
/// verbatim layout planner.
#[cfg(feature = "std")]
pub(crate) struct DenseChunkGrid {
    /// One [`ChunkInfo`] per grid slot, in row-major grid order.
    pub(crate) grid_order: Vec<ChunkInfo>,
}

/// Map enumerated `infos` onto the logical chunk grid implied by `dims` (the
/// dataspace shape) and `chunk_dims` (the rank-only spatial chunk dimensions),
/// returning each slot's [`ChunkInfo`] in row-major order. Returns `None` when
/// the grid is *not* dense — a hole, a duplicate, a misaligned or out-of-range
/// chunk offset, a zero chunk dimension, or a chunk count other than the full
/// grid — so a caller that needs every slot filled (a verbatim copy, a
/// whole-dataset overwrite) can fall back or refuse. No chunk bytes are read.
#[cfg(feature = "std")]
pub(crate) fn plan_dense_grid(
    infos: Vec<ChunkInfo>,
    dims: &[u64],
    chunk_dims: &[u64],
) -> Option<DenseChunkGrid> {
    let rank = dims.len();
    if chunk_dims.len() < rank {
        return None;
    }
    let mut num_chunks_per_dim = Vec::with_capacity(rank);
    for d in 0..rank {
        if chunk_dims[d] == 0 {
            return None;
        }
        num_chunks_per_dim.push(dims[d].div_ceil(chunk_dims[d]));
    }
    let total: u64 = num_chunks_per_dim.iter().product();
    if infos.len() as u64 != total {
        // A different chunk count than the full grid means holes (or duplicates).
        return None;
    }
    let total_us = usize::try_from(total).ok()?;

    // Map each chunk to its linear grid slot; detect any hole or duplicate.
    let mut slots: Vec<Option<ChunkInfo>> = (0..total_us).map(|_| None).collect();
    for info in infos {
        if info.offsets.len() < rank {
            return None;
        }
        let mut linear: u64 = 0;
        for d in 0..rank {
            if !info.offsets[d].is_multiple_of(chunk_dims[d]) {
                return None;
            }
            let grid_coord = info.offsets[d] / chunk_dims[d];
            if grid_coord >= num_chunks_per_dim[d] {
                return None;
            }
            linear = linear * num_chunks_per_dim[d] + grid_coord;
        }
        let idx = usize::try_from(linear).ok()?;
        if slots[idx].is_some() {
            return None; // duplicate offset
        }
        slots[idx] = Some(info);
    }

    // Every slot must be filled for a dense grid.
    let mut grid_order = Vec::with_capacity(total_us);
    for slot in slots {
        grid_order.push(slot?);
    }
    Some(DenseChunkGrid { grid_order })
}

/// Every on-disk byte span `(addr, len)` a chunked dataset owns: each allocated
/// chunk data block plus the chunk index's own structure blocks. This is the
/// single place that maps a chunked layout to the regions it occupies on disk;
/// the in-place editor uses it to reclaim that space on delete (issue #77).
///
/// `layout` must be a [`DataLayout::Chunked`] (anything else is an error).
/// Returns an empty vector when the index address is undefined (an empty or
/// never-written dataset owns no storage). Errors — propagated from the index
/// walkers — mean the layout could not be enumerated exhaustively (a version 2
/// B-tree chunk index, or a malformed structure); the caller then leaves the
/// dataset's bytes in place rather than free a region it cannot fully account
/// for. The spans are not pre-checked for mutual disjointness or file bounds:
/// the editor validates that before handing them to the free list.
///
/// Chunk data spans come from the same index walkers the reader uses
/// ([`collect_chunks_for_layout_from_source`]); index-structure spans come from
/// [`collect_chunk_index_spans`]. `offset_size`/`length_size` are the
/// superblock's address and length widths.
#[cfg(feature = "std")]
pub(crate) fn collect_chunked_storage_spans(
    file_data: &[u8],
    layout: &DataLayout,
    dataspace: &Dataspace,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<(u64, u64)>, FormatError> {
    let DataLayout::Chunked {
        chunk_dimensions,
        btree_address,
        version,
        chunk_index_type,
        single_chunk_filtered_size,
        single_chunk_filter_mask,
    } = layout
    else {
        return Err(FormatError::ChunkedReadError(
            "collect_chunked_storage_spans called on a non-chunked layout".into(),
        ));
    };
    // An undefined index address means no storage is allocated yet.
    let Some(index_addr) = *btree_address else {
        return Ok(Vec::new());
    };
    if chunk_dimensions.is_empty() {
        return Err(FormatError::ChunkedReadError(
            "chunked layout has no dimensions".into(),
        ));
    }
    // `chunk_dimensions` is rank + 1 entries; the last is the element size.
    let rank = chunk_dimensions
        .len()
        .checked_sub(1)
        .ok_or_else(|| FormatError::ChunkedReadError("chunked layout has no dimensions".into()))?;
    let elem_size = chunk_dimensions[rank] as usize;
    if elem_size == 0 {
        return Err(FormatError::ChunkedReadError(
            "chunked layout has a zero element size".into(),
        ));
    }

    let source = crate::source::BytesSource::new(file_data);
    let mut spans: Vec<(u64, u64)> = Vec::new();

    // Chunk data blocks (the walkers omit unallocated chunks).
    for ci in collect_chunks_for_layout_from_source(
        &source,
        *version,
        *chunk_index_type,
        index_addr,
        *single_chunk_filtered_size,
        *single_chunk_filter_mask,
        chunk_dimensions,
        dataspace,
        elem_size,
        offset_size,
        length_size,
    )? {
        if ci.chunk_size != 0 {
            spans.push((ci.address, ci.chunk_size as u64));
        }
    }

    // The chunk index's own structure blocks.
    spans.extend(collect_chunk_index_spans(
        file_data,
        *version,
        *chunk_index_type,
        index_addr,
        chunk_dimensions.len(),
        offset_size,
        length_size,
    )?);
    Ok(spans)
}

/// On-disk byte spans `(addr, len)` of a chunked dataset's index *structure*
/// (not its chunk data): the B-tree v1 nodes, fixed-array header and data block,
/// or extensible-array header, index, super, and data blocks, by index type.
/// Single-chunk and implicit indexes have no separate structure (their footprint
/// is the chunk data alone), so they return an empty vector. An index type with
/// no walker (a version 2 B-tree, index type 5) is an error.
///
/// `ndims` is `chunk_dimensions.len()` (rank + 1), as [`collect_chunk_info`]
/// takes. Shared by [`collect_chunked_storage_spans`]; kept separate so the
/// per-type dispatch lives next to the chunk-data dispatch it mirrors.
#[cfg(feature = "std")]
fn collect_chunk_index_spans(
    file_data: &[u8],
    version: u8,
    chunk_index_type: Option<u8>,
    index_addr: u64,
    ndims: usize,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<(u64, u64)>, FormatError> {
    match (version, chunk_index_type) {
        (3, _) => collect_chunk_btree_node_spans(file_data, index_addr, ndims, offset_size),
        // Single chunk and implicit indexes have no separate index structure.
        (4, Some(1)) | (4, Some(2)) => Ok(Vec::new()),
        (4, Some(3)) => crate::fixed_array::fixed_array_index_spans(
            file_data,
            index_addr,
            offset_size,
            length_size,
        ),
        (4, Some(4)) => crate::extensible_array::extensible_array_index_spans(
            file_data,
            index_addr,
            offset_size,
            length_size,
        ),
        (v, idx) => Err(FormatError::ChunkedReadError(format!(
            "chunk index has no reclaim walker: version={v}, index_type={idx:?}"
        ))),
    }
}

/// The on-disk byte spans of a chunked dataset's index *structure* (not its chunk
/// data), from an in-memory file image and a parsed [`DataLayout::Chunked`]. A
/// buffered convenience wrapper over [`collect_chunk_index_spans`], used by the
/// in-place editor to test whether a chunked dataset's index occupies a single
/// contiguous region it can rebuild in place. Returns an empty vector for an
/// undefined index address or a single-chunk / implicit index (no separate
/// structure); errors for an index type with no walker (a version-2 B-tree).
#[cfg(feature = "std")]
pub(crate) fn chunk_index_spans_buffered(
    file_data: &[u8],
    layout: &DataLayout,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<(u64, u64)>, FormatError> {
    let DataLayout::Chunked {
        chunk_dimensions,
        btree_address,
        version,
        chunk_index_type,
        ..
    } = layout
    else {
        return Err(FormatError::ChunkedReadError(
            "chunk_index_spans_buffered called on a non-chunked layout".into(),
        ));
    };
    let Some(index_addr) = *btree_address else {
        return Ok(Vec::new());
    };
    collect_chunk_index_spans(
        file_data,
        *version,
        *chunk_index_type,
        index_addr,
        chunk_dimensions.len(),
        offset_size,
        length_size,
    )
}

/// Scatter decompressed chunks into the dense output buffer. Pure (no file
/// access): shared by the buffered and streaming chunked readers.
fn assemble_chunks(
    chunks: &[ChunkInfo],
    decompressed: &[Vec<u8>],
    rank: usize,
    chunk_dims: &[usize],
    ds_dims: &[usize],
    elem_size: usize,
    total_bytes: usize,
) -> Vec<u8> {
    let mut output = vec![0u8; total_bytes];

    let mut ds_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        ds_strides[i] = ds_strides[i + 1] * ds_dims[i + 1];
    }
    let mut chunk_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        chunk_strides[i] = chunk_strides[i + 1] * chunk_dims[i + 1];
    }

    for (chunk_info, decompressed) in chunks.iter().zip(decompressed.iter()) {
        // B-tree v1 (v3) offsets have rank+1 dims; v4 index offsets have rank dims
        #[expect(
            clippy::cast_possible_truncation,
            reason = "chunk coordinate offsets are bounded by ds_dims, which already fit usize"
        )]
        let chunk_offsets: Vec<usize> = chunk_info
            .offsets
            .iter()
            .take(rank)
            .map(|&o| o as usize)
            .collect();

        place_chunk(
            decompressed,
            &mut output,
            &chunk_offsets,
            chunk_dims,
            ds_dims,
            &ds_strides,
            &chunk_strides,
            elem_size,
            rank,
        );
    }

    output
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
    let (
        chunk_dimensions,
        version,
        chunk_index_type,
        addr_opt,
        single_filtered_size,
        single_filter_mask,
    ) = match layout {
        DataLayout::Chunked {
            chunk_dimensions,
            btree_address,
            version,
            chunk_index_type,
            single_chunk_filtered_size,
            single_chunk_filter_mask,
        } => (
            chunk_dimensions,
            *version,
            *chunk_index_type,
            *btree_address,
            *single_chunk_filtered_size,
            *single_chunk_filter_mask,
        ),
        _ => {
            return Err(FormatError::ChunkedReadError(
                "expected chunked layout".into(),
            ));
        }
    };

    let addr = addr_opt
        .ok_or_else(|| FormatError::ChunkedReadError("no address for chunked layout".into()))?;

    let elem_size = datatype.type_size() as usize;
    let ndims = chunk_dimensions.len();
    let rank = ndims
        .checked_sub(1)
        .ok_or_else(|| FormatError::ChunkedReadError("chunked layout has no dimensions".into()))?;
    let chunk_dims: Vec<usize> = chunk_dimensions[..rank]
        .iter()
        .map(|&d| d as usize)
        .collect();

    let ds_dims: Vec<usize> = dataspace
        .dimensions
        .iter()
        .map(|&d| d.to_usize())
        .collect::<Result<_, _>>()?;
    if ds_dims.len() != rank {
        return Err(FormatError::ChunkedReadError(format!(
            "rank mismatch: dataspace has {} dims, layout has {} chunk dims (rank={})",
            ds_dims.len(),
            chunk_dimensions.len(),
            rank
        )));
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "chunk byte sizes and the datatype element size are encoded into 32-bit \
                  chunk-info fields; both stay well below u32::MAX (HDF5 caps a chunk at 4 GiB)"
    )]
    let chunks = if let Some(chunks) = cache.all_indexed_chunks() {
        chunks
    } else {
        let chunks = match (version, chunk_index_type) {
            (3, _) => collect_chunk_info(file_data, addr, ndims, offset_size, length_size)?,
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
                generate_implicit_chunks(
                    addr,
                    &dataspace.dimensions,
                    &spatial_chunk_dims,
                    elem_size as u32,
                )
            }
            (4, Some(3)) => {
                let spatial_chunk_dims: Vec<u32> = chunk_dimensions[..rank].to_vec();
                let header =
                    FixedArrayHeader::parse(file_data, addr.to_usize()?, offset_size, length_size)?;
                read_fixed_array_chunks(
                    file_data,
                    &header,
                    &dataspace.dimensions,
                    &spatial_chunk_dims,
                    elem_size as u32,
                    offset_size,
                    length_size,
                )?
            }
            (4, Some(4)) => {
                let spatial_chunk_dims: Vec<u32> = chunk_dimensions[..rank].to_vec();
                let header = ExtensibleArrayHeader::parse(
                    file_data,
                    addr.to_usize()?,
                    offset_size,
                    length_size,
                )?;
                read_extensible_array_chunks(
                    file_data,
                    &header,
                    &dataspace.dimensions,
                    &spatial_chunk_dims,
                    elem_size as u32,
                    offset_size,
                    length_size,
                )?
            }
            (v, idx) => {
                return Err(FormatError::ChunkedReadError(format!(
                    "unsupported chunked layout version={v}, index_type={idx:?}"
                )));
            }
        };
        cache.populate_index(&chunks, rank);
        chunks
    };

    // Assemble output
    let total_elements = dataspace.num_elements().to_usize()?;
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

    let chunk_dims_u64: Vec<u64> = chunk_dims.iter().map(|&d| d as u64).collect();
    let ctx = ChunkContext::from_datatype(&chunk_dims_u64, datatype);

    for chunk_info in &chunks {
        let coord: Vec<u64> = chunk_info.offsets.iter().take(rank).copied().collect();

        #[expect(
            clippy::cast_possible_truncation,
            reason = "chunk coordinate offsets are bounded by ds_dims, which already fit usize"
        )]
        let chunk_offsets: Vec<usize> = chunk_info
            .offsets
            .iter()
            .take(rank)
            .map(|&o| o as usize)
            .collect();

        // Scatter straight from the cached chunk under the lock (no copy out).
        let hit = cache.with_decompressed(&coord, |bytes| {
            place_chunk(
                bytes,
                &mut output,
                &chunk_offsets,
                &chunk_dims,
                &ds_dims,
                &ds_strides,
                &chunk_strides,
                elem_size,
                rank,
            );
        });
        if hit.is_some() {
            continue;
        }

        // Cache miss: read the chunk's bytes from the file.
        let r = slice_range(chunk_info.address, u64::from(chunk_info.chunk_size))?;
        if r.end > file_data.len() {
            return Err(FormatError::UnexpectedEof {
                expected: r.end,
                available: file_data.len(),
            });
        }
        let raw_chunk = &file_data[r];
        if let Some(pl) = pipeline {
            let dec = decompress_chunk(raw_chunk, pl, ctx, chunk_info.filter_mask)?;
            place_chunk(
                &dec,
                &mut output,
                &chunk_offsets,
                &chunk_dims,
                &ds_dims,
                &ds_strides,
                &chunk_strides,
                elem_size,
                rank,
            );
            cache.put_decompressed(coord, dec); // move; dropped if not admitted
        } else {
            // No pipeline: scatter directly from the file buffer, and copy into
            // the cache only if it would actually be retained.
            place_chunk(
                raw_chunk,
                &mut output,
                &chunk_offsets,
                &chunk_dims,
                &ds_dims,
                &ds_strides,
                &chunk_strides,
                elem_size,
                rank,
            );
            cache.put_decompressed_slice(coord, raw_chunk);
        }
    }

    Ok(output)
}

/// Place one decompressed chunk into the dense output buffer, handling the
/// scalar (`rank == 0`) case and delegating the N-D case to the row-copy kernel.
/// Shared by the buffered, cached, and streaming chunked readers so they all use
/// the same scatter logic.
#[allow(clippy::too_many_arguments)]
fn place_chunk(
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
    if rank == 0 {
        let copy_len = chunk_data.len().min(output.len());
        output[..copy_len].copy_from_slice(&chunk_data[..copy_len]);
    } else {
        copy_chunk_to_output(
            chunk_data,
            output,
            chunk_offsets,
            chunk_dims,
            ds_dims,
            ds_strides,
            chunk_strides,
            elem_size,
            rank,
        );
    }
}

/// Copy chunk data into the output buffer at the correct N-D position.
///
/// The innermost dimension is contiguous in both the chunk and the dataset (both
/// have stride 1 there in a row-major layout), so each in-bounds row of the
/// chunk is moved with a single `copy_from_slice` instead of one tiny copy per
/// element. Only the outer `rank - 1` dimensions are walked (with an odometer),
/// which removes the per-element flat→N-D coordinate division/modulo that
/// dominated the old kernel. Edge chunks that overhang the dataset are clamped in
/// the innermost dimension (a shortened row) and skipped per-row in the outer
/// dimensions (a row whose outer global coordinate is past the dataset bound),
/// reproducing the old per-element out-of-bounds skip exactly. The per-row length
/// is also clamped to the bytes actually available on both sides, so a malformed
/// (short) chunk copies only its valid prefix and is never an out-of-bounds
/// access — matching the old per-element guard's silent-skip behavior.
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
    debug_assert!(rank >= 1, "rank == 0 is handled by the callers");
    // Row contiguity (the whole optimization) relies on a unit innermost stride
    // in both layouts, which the row-major stride construction guarantees.
    debug_assert_eq!(chunk_strides[rank - 1], 1);
    debug_assert_eq!(ds_strides[rank - 1], 1);

    let inner = rank - 1;

    // In-bounds run length along the contiguous innermost dimension: the chunk
    // is stored at full size but may hang over the dataset's edge.
    let inner_row_len = chunk_dims[inner].min(ds_dims[inner].saturating_sub(chunk_offsets[inner]));
    if inner_row_len == 0 {
        return; // the chunk lies entirely past the inner edge
    }
    let row_bytes = inner_row_len * elem_size;
    // Destination offset contributed by the innermost dimension (stride 1).
    let inner_dst = chunk_offsets[inner] * ds_strides[inner];

    // Walk the outer dimensions with an odometer (`coord` is the chunk-local
    // coordinate of each outer dim), copying one contiguous row per step.
    let outer_total: usize = chunk_dims[..inner].iter().product();
    let mut coord = vec![0usize; inner];

    for _ in 0..outer_total {
        let mut chunk_base = 0usize;
        let mut ds_base = inner_dst;
        let mut in_bounds = true;
        for d in 0..inner {
            chunk_base += coord[d] * chunk_strides[d];
            let global = chunk_offsets[d] + coord[d];
            if global >= ds_dims[d] {
                in_bounds = false;
                break;
            }
            ds_base += global * ds_strides[d];
        }

        if in_bounds {
            let src = chunk_base * elem_size;
            let dst = ds_base * elem_size;
            // Clamp to the bytes available on each side, on an element boundary.
            let mut avail = row_bytes
                .min(chunk_data.len().saturating_sub(src))
                .min(output.len().saturating_sub(dst));
            avail -= avail % elem_size;
            if avail > 0 {
                output[dst..dst + avail].copy_from_slice(&chunk_data[src..src + avail]);
            }
        }

        // Advance the odometer over the outer dims (last outer dim varies fastest).
        for d in (0..inner).rev() {
            coord[d] += 1;
            if coord[d] < chunk_dims[d] {
                break;
            }
            coord[d] = 0;
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
    fn build_chunk_btree_leaf(chunks: &[ChunkInfo], ndims: usize, offset_size: u8) -> Vec<u8> {
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

    /// Build a B-tree v1 type 1 internal node pointing at the given child node
    /// addresses (keys are dummies — the node-span walk reads only children).
    fn build_chunk_btree_internal(
        level: u8,
        child_addrs: &[u64],
        ndims: usize,
        offset_size: u8,
    ) -> Vec<u8> {
        let entries_used = child_addrs.len() as u16;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"TREE");
        buf.push(1); // node_type = 1 (raw data chunks)
        buf.push(level); // node_level > 0 (internal)
        buf.extend_from_slice(&entries_used.to_le_bytes());
        let undef: u64 = if offset_size == 4 {
            0xFFFFFFFF
        } else {
            0xFFFFFFFFFFFFFFFF
        };
        write_offset(&mut buf, undef, offset_size);
        write_offset(&mut buf, undef, offset_size);
        for &addr in child_addrs {
            buf.extend_from_slice(&0u32.to_le_bytes()); // key: chunk_size
            buf.extend_from_slice(&0u32.to_le_bytes()); // key: filter_mask
            for _ in 0..ndims {
                write_offset(&mut buf, 0, offset_size);
            }
            write_offset(&mut buf, addr, offset_size); // child node address
        }
        // Final key.
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        for _ in 0..ndims {
            write_offset(&mut buf, u64::MAX, offset_size);
        }
        buf
    }

    #[test]
    fn btree_node_spans_single_leaf() {
        // A leaf's reclaim span is the whole node: its built byte length.
        let ndims = 2;
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
        let leaf = build_chunk_btree_leaf(&chunks, ndims, os);
        let at = 0x40usize;
        let mut file_data = vec![0u8; 0x3000];
        file_data[at..at + leaf.len()].copy_from_slice(&leaf);

        let spans = collect_chunk_btree_node_spans(&file_data, at as u64, ndims, os).unwrap();
        assert_eq!(spans, vec![(at as u64, leaf.len() as u64)]);
    }

    #[test]
    fn btree_node_spans_two_level_recurses() {
        // A 2-level tree yields one span per node: the internal root and both
        // leaves, each sized to its built byte length, pairwise disjoint.
        let ndims = 2;
        let os: u8 = 8;
        let leaf_chunks = vec![ChunkInfo {
            chunk_size: 40,
            filter_mask: 0,
            offsets: vec![0, 0],
            address: 0x100,
        }];
        let leaf0 = build_chunk_btree_leaf(&leaf_chunks, ndims, os);
        let leaf1 = build_chunk_btree_leaf(&leaf_chunks, ndims, os);
        let (l0, l1, root) = (0x1000usize, 0x2000usize, 0x3000usize);
        let internal = build_chunk_btree_internal(1, &[l0 as u64, l1 as u64], ndims, os);

        let mut file = vec![0u8; 0x4000];
        file[l0..l0 + leaf0.len()].copy_from_slice(&leaf0);
        file[l1..l1 + leaf1.len()].copy_from_slice(&leaf1);
        file[root..root + internal.len()].copy_from_slice(&internal);

        let mut spans = collect_chunk_btree_node_spans(&file, root as u64, ndims, os).unwrap();
        spans.sort_unstable();
        assert_eq!(
            spans,
            vec![
                (l0 as u64, leaf0.len() as u64),
                (l1 as u64, leaf1.len() as u64),
                (root as u64, internal.len() as u64),
            ]
        );
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

    #[test]
    fn collect_chunk_info_rejects_cyclic_btree() {
        // An internal node whose only child points back at itself. Without a
        // depth guard this recurses forever and overflows the stack (an
        // uncatchable process abort); the guard must turn it into an error.
        let ndims = 2;
        let os: u8 = 8;
        let node = build_chunk_btree_internal(1, &[0u64], ndims, os);
        let mut file_data = vec![0u8; 0x1000];
        file_data[..node.len()].copy_from_slice(&node);

        let err = collect_chunk_info(&file_data, 0, ndims, os, os).unwrap_err();
        assert!(matches!(err, FormatError::ChunkedReadError(_)));
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
                file_data[byte_offset..byte_offset + 8].copy_from_slice(&values[i].to_le_bytes());
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

        let raw =
            read_chunked_data(&file_data, &layout, &dataspace, &datatype, None, 8, 8).unwrap();
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

        let raw =
            read_chunked_data(&file_data, &layout, &dataspace, &datatype, None, 8, 8).unwrap();
        assert_eq!(raw.len(), 25 * 8);

        for i in 0..25 {
            let val = f64::from_le_bytes(raw[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(val, i as f64, "mismatch at index {i}");
        }
    }

    /// Assert the streaming chunked reader matches the buffered one over both an
    /// in-memory source and a lazy `Read+Seek` source.
    #[cfg(feature = "std")]
    fn assert_chunked_streams_match(
        file_data: &[u8],
        layout: &DataLayout,
        dataspace: &Dataspace,
        datatype: &Datatype,
        pipeline: Option<&FilterPipeline>,
    ) {
        use crate::source::{BytesSource, ReadSeekSource};
        let buffered =
            read_chunked_data(file_data, layout, dataspace, datatype, pipeline, 8, 8).unwrap();
        let from_mem = read_chunked_data_from_source(
            &BytesSource::new(file_data),
            layout,
            dataspace,
            datatype,
            pipeline,
            8,
            8,
        )
        .unwrap();
        let from_seek = read_chunked_data_from_source(
            &ReadSeekSource::new(std::io::Cursor::new(file_data.to_vec())).unwrap(),
            layout,
            dataspace,
            datatype,
            pipeline,
            8,
            8,
        )
        .unwrap();
        assert_eq!(buffered, from_mem, "BytesSource mismatch");
        assert_eq!(buffered, from_seek, "ReadSeekSource mismatch");
    }

    #[cfg(feature = "std")]
    #[test]
    fn streaming_chunked_btree_v1_matches_buffered() {
        let values: Vec<f64> = (0..25).map(|i| i as f64).collect();
        let (file_data, layout, dataspace) = build_1d_chunked_file(&values, 10);
        assert_chunked_streams_match(&file_data, &layout, &dataspace, &make_f64_type(), None);
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn read_1d_two_chunks_with_deflate() {
        use crate::filter_pipeline::{FILTER_DEFLATE, FilterDescription, FilterPipeline};
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
            let dims_u64 = [chunk_elems as u64];
            let ctx = crate::filters::ChunkContext::basic(&dims_u64, elem_size as u32);
            let compressed = compress_chunk(&chunk_bytes, &pipeline, ctx).unwrap();

            file_data[data_offset..data_offset + compressed.len()].copy_from_slice(&compressed);

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

        // The streaming reader must reproduce the same decompressed bytes.
        assert_chunked_streams_match(&file_data, &layout, &dataspace, &datatype, Some(&pipeline));
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
                file_data[data_offset..data_offset + chunk_size].copy_from_slice(&chunk_bytes);

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

        let raw =
            read_chunked_data(&file_data, &layout, &dataspace, &datatype, None, 8, 8).unwrap();
        assert_eq!(raw.len(), 24 * 4);

        for i in 0..24 {
            let val = f32::from_le_bytes(raw[i * 4..(i + 1) * 4].try_into().unwrap());
            assert_eq!(val, i as f32, "mismatch at element {i}");
        }

        #[cfg(feature = "std")]
        assert_chunked_streams_match(&file_data, &layout, &dataspace, &datatype, None);
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

        let raw =
            read_chunked_data(&file_data, &layout, &dataspace, &datatype, None, 8, 8).unwrap();
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

        assert!(!cache.stats().index_loaded());
        let raw = read_chunked_data_cached(
            &file_data, &layout, &dataspace, &datatype, None, 8, 8, &cache,
        )
        .unwrap();
        assert!(cache.stats().index_loaded());
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
        )
        .unwrap();
        assert!(cache.stats().index_loaded());
        assert!(cache.stats().cached_chunks() > 0);

        // Second read — should hit the decompressed cache
        let raw2 = read_chunked_data_cached(
            &file_data, &layout, &dataspace, &datatype, None, 8, 8, &cache,
        )
        .unwrap();
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
        )
        .unwrap();
        assert_eq!(raw.len(), 25 * 8);
        for i in 0..25 {
            let val = f64::from_le_bytes(raw[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(val, i as f64, "mismatch at index {i}");
        }
    }
}
