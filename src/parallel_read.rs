//! Parallel chunk decompression using rayon with lane partitioning.
//!
//! When reading a chunked+compressed dataset with many chunks, this module
//! uses lane-partitioned parallel decompression: each thread receives a
//! deterministic, disjoint subset of chunks — no overlap, no coordination.
//!
//! The lane assignment is seeded by dataset metadata so repeated reads of
//! the same region produce identical partitions (cache-friendly, reproducible).

use crate::chunked_read::ChunkInfo;
use crate::convert::slice_range;
use crate::error::FormatError;
use crate::filter_pipeline::FilterPipeline;
use crate::filters::{ChunkContext, decompress_chunk};
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
    ctx: ChunkContext<'_>,
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
                let r = slice_range(chunk_info.address, u64::from(chunk_info.chunk_size))?;

                if r.end > file_data.len() {
                    return Err(FormatError::UnexpectedEof {
                        expected: r.end,
                        available: file_data.len(),
                    });
                }
                let raw_chunk = &file_data[r];

                let decompressed = decompress_chunk(raw_chunk, pipeline, ctx, chunk_info.filter_mask)?;

                stats.chunks_processed += 1;
                stats.compressed_bytes += u64::from(chunk_info.chunk_size);
                stats.decompressed_bytes += decompressed.len() as u64;

                results.push(DecompressedChunk {
                    index,
                    data: decompressed,
                });
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a file buffer holding `n` chunks of `chunk_len` bytes each, laid
    /// out back to back, where chunk `i` is filled with the byte `i as u8`.
    /// Returns the buffer and the matching `ChunkInfo` list (in index order).
    fn build_chunks(n: usize, chunk_len: usize) -> (Vec<u8>, Vec<ChunkInfo>) {
        let mut file_data = Vec::with_capacity(n * chunk_len);
        let mut chunks = Vec::with_capacity(n);
        for i in 0..n {
            let address = file_data.len() as u64;
            #[allow(clippy::cast_possible_truncation)]
            file_data.extend(std::iter::repeat_n(i as u8, chunk_len));
            chunks.push(ChunkInfo {
                chunk_size: chunk_len as u32,
                filter_mask: 0,
                offsets: vec![i as u64],
                address,
            });
        }
        (file_data, chunks)
    }

    /// Serial reference decode: read each chunk in index order with an empty
    /// pipeline (identity), mirroring what the parallel path must reproduce.
    fn serial_decode(
        file_data: &[u8],
        chunks: &[ChunkInfo],
        pipeline: &FilterPipeline,
        ctx: ChunkContext<'_>,
    ) -> Vec<Vec<u8>> {
        chunks
            .iter()
            .map(|c| {
                let r = slice_range(c.address, u64::from(c.chunk_size)).unwrap();
                decompress_chunk(&file_data[r], pipeline, ctx, c.filter_mask).unwrap()
            })
            .collect()
    }

    #[test]
    fn lane_partitioned_restores_chunk_order_like_serial() {
        // A multi-chunk input comfortably above PARALLEL_THRESHOLD so the
        // partitioner actually spreads chunks across lanes; with several lanes
        // the per-lane flatten order is not the global index order, so this
        // exercises the `sort_by_key` order-restoration specifically.
        let n = 64;
        assert!(should_use_parallel(n));
        let chunk_len = 17; // non-power-of-two to catch any stride confusion
        let (file_data, chunks) = build_chunks(n, chunk_len);

        let pipeline = FilterPipeline {
            version: 2,
            filters: Vec::new(),
        };
        let chunk_dims = [chunk_len as u64];
        let ctx = ChunkContext::basic(&chunk_dims, 1);

        let expected = serial_decode(&file_data, &chunks, &pipeline, ctx);

        // Several distinct seeds: order restoration must hold regardless of how
        // the seeded permutation assigns chunks to lanes.
        for seed in [0u64, 1, 7, 0xDEAD_BEEF, u64::MAX] {
            let (ordered, stats) = decompress_chunks_lane_partitioned(
                &file_data,
                &chunks,
                &pipeline,
                ctx,
                seed,
                Some(8),
            )
            .expect("parallel decode");

            assert_eq!(ordered.len(), n, "all chunks returned for seed {seed}");
            // Every chunk lands at its original index, byte-for-byte, matching
            // the serial decode. This is the property `sort_by_key` guarantees.
            assert_eq!(ordered, expected, "order restored for seed {seed}");
            // And each decoded chunk really is its own distinct content (guards
            // against a degenerate all-equal buffer making the check vacuous).
            #[allow(clippy::cast_possible_truncation)]
            for (i, chunk) in ordered.iter().enumerate() {
                assert_eq!(chunk.as_slice(), &vec![i as u8; chunk_len][..]);
            }
            // Stats account for every chunk exactly once across all lanes.
            assert_eq!(stats.total_chunks, n);
            let processed: usize = stats.per_lane.iter().map(|l| l.chunks_processed).sum();
            assert_eq!(processed, n, "each chunk processed once for seed {seed}");
        }
    }
}
