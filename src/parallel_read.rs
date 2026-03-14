//! Parallel chunk decompression using rayon with lane partitioning.
//!
//! When reading a chunked+compressed dataset with many chunks, this module
//! uses lane-partitioned parallel decompression: each thread receives a
//! deterministic, disjoint subset of chunks — no overlap, no coordination.
//!
//! The lane assignment is seeded by dataset metadata so repeated reads of
//! the same region produce identical partitions (cache-friendly, reproducible).

use crate::chunked_read::ChunkInfo;
use crate::error::FormatError;
use crate::filter_pipeline::FilterPipeline;
use crate::filters::decompress_chunk;
use crate::lane_partition::{self, LaneStats, PartitionStats};

/// Threshold: only use parallel decompression when chunk count exceeds this.
const PARALLEL_THRESHOLD: usize = 4;

/// Result of decompressing a single chunk, tagged with its index for ordering.
struct DecompressedChunk {
    index: usize,
    data: Vec<u8>,
}

/// Returns `true` if the parallel path should be used for the given chunk count.
pub fn should_use_parallel(chunk_count: usize) -> bool {
    chunk_count > PARALLEL_THRESHOLD
}

/// Decompress chunks in parallel using lane-partitioned assignment.
///
/// Instead of naive `par_iter`, chunks are deterministically assigned to lanes
/// (threads) using a seeded pseudorandom permutation.  Each lane processes
/// only its assigned chunks — no redundant work, no coordination.
///
/// # Arguments
///
/// * `seed` - Seed for the partition permutation (e.g. dataset address + chunk range hash).
/// * `num_lanes` - Number of parallel lanes.  Pass `None` to auto-detect from available cores.
///
/// # Errors
///
/// Returns the first error encountered by any worker thread.
pub fn decompress_chunks_lane_partitioned(
    file_data: &[u8],
    chunks: &[ChunkInfo],
    pipeline: &FilterPipeline,
    chunk_total_bytes: usize,
    element_size: u32,
    seed: u64,
    num_lanes: Option<usize>,
) -> Result<(Vec<Vec<u8>>, PartitionStats), FormatError> {
    use rayon::prelude::*;

    let lanes = num_lanes.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });

    let assignments = lane_partition::partition_chunks(chunks.len(), lanes, seed);
    let num_lanes = assignments.len();

    // Each lane processes its assigned chunks and returns results + stats.
    let lane_results: Result<Vec<(Vec<DecompressedChunk>, LaneStats)>, FormatError> = assignments
        .into_par_iter()
        .map(|indices| {
            let mut results = Vec::with_capacity(indices.len());
            let mut stats = LaneStats::default();

            for &index in &indices {
                let chunk_info = &chunks[index];
                let c_addr = chunk_info.address as usize;
                let size = chunk_info.chunk_size as usize;

                if c_addr + size > file_data.len() {
                    return Err(FormatError::UnexpectedEof {
                        expected: c_addr + size,
                        available: file_data.len(),
                    });
                }
                let raw_chunk = &file_data[c_addr..c_addr + size];

                let decompressed = if chunk_info.filter_mask == 0 {
                    decompress_chunk(raw_chunk, pipeline, chunk_total_bytes, element_size)?
                } else {
                    raw_chunk.to_vec()
                };

                stats.chunks_processed += 1;
                stats.compressed_bytes += size as u64;
                stats.decompressed_bytes += decompressed.len() as u64;

                results.push(DecompressedChunk { index, data: decompressed });
            }

            Ok((results, stats))
        })
        .collect();

    let lane_results = lane_results?;

    // Aggregate stats
    let mut partition_stats = PartitionStats::new(num_lanes);
    partition_stats.total_chunks = chunks.len();
    for (lane_idx, (_, stats)) in lane_results.iter().enumerate() {
        partition_stats.per_lane[lane_idx] = stats.clone();
    }

    // Flatten and sort by original index to restore order
    let mut all_chunks: Vec<DecompressedChunk> = lane_results
        .into_iter()
        .flat_map(|(chunks, _)| chunks)
        .collect();
    all_chunks.sort_by_key(|dc| dc.index);

    let ordered = all_chunks.into_iter().map(|dc| dc.data).collect();
    Ok((ordered, partition_stats))
}

/// Decompress chunks in parallel using rayon (legacy par_iter path).
///
/// Each chunk is read from `file_data` at the address in the corresponding
/// `ChunkInfo`, decompressed through `pipeline`, and collected in order.
///
/// # Errors
///
/// Returns the first error encountered by any worker thread.
pub fn decompress_chunks_parallel(
    file_data: &[u8],
    chunks: &[ChunkInfo],
    pipeline: &FilterPipeline,
    chunk_total_bytes: usize,
    element_size: u32,
) -> Result<Vec<Vec<u8>>, FormatError> {
    use rayon::prelude::*;

    let results: Result<Vec<DecompressedChunk>, FormatError> = chunks
        .par_iter()
        .enumerate()
        .map(|(index, chunk_info)| {
            let c_addr = chunk_info.address as usize;
            let size = chunk_info.chunk_size as usize;
            if c_addr + size > file_data.len() {
                return Err(FormatError::UnexpectedEof {
                    expected: c_addr + size,
                    available: file_data.len(),
                });
            }
            let raw_chunk = &file_data[c_addr..c_addr + size];

            let decompressed = if chunk_info.filter_mask == 0 {
                decompress_chunk(raw_chunk, pipeline, chunk_total_bytes, element_size)?
            } else {
                raw_chunk.to_vec()
            };

            Ok(DecompressedChunk { index, data: decompressed })
        })
        .collect();

    let mut result_vec = results?;
    result_vec.sort_by_key(|dc| dc.index);
    Ok(result_vec.into_iter().map(|dc| dc.data).collect())
}

/// Decompress chunks sequentially (fallback when parallel is not warranted).
pub fn decompress_chunks_sequential(
    file_data: &[u8],
    chunks: &[ChunkInfo],
    pipeline: Option<&FilterPipeline>,
    chunk_total_bytes: usize,
    element_size: u32,
) -> Result<Vec<Vec<u8>>, FormatError> {
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
        result.push(decompressed);
    }
    Ok(result)
}
