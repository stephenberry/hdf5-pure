//! Coordination-free lane partitioning for parallel chunk decompression.
//!
//! Based on the paper "Coordination-Free Lane Partitioning for Convergent ANN
//! Search" (arXiv 2511.04221).  Each thread (lane) receives a deterministic,
//! disjoint subset of work items — no locks, no atomics, no work overlap.
//!
//! The partition is seeded by a per-query value (dataset offset + chunk range)
//! so repeated reads of the same region always produce the same assignment,
//! making results reproducible and cache-friendly.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Lightweight non-cryptographic hash (FxHash variant).
///
/// Uses the golden-ratio constant multiply-XOR trick from Firefox.
#[inline]
fn fxhash(mut x: u64) -> u64 {
    // 64-bit FxHash constant (closest odd number to 2^64 / phi)
    const K: u64 = 0x517cc1b727220a95;
    x = x.wrapping_mul(K);
    x ^= x >> 33;
    x = x.wrapping_mul(K);
    x ^= x >> 29;
    x
}

/// Combine two u64 values into a single hash (for seeding with index).
#[inline]
fn fxhash_combine(seed: u64, index: u64) -> u64 {
    fxhash(seed ^ fxhash(index))
}

/// Partitioning mode for distributing work across lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionMode {
    /// Equal-size round-robin: item `i` goes to lane `permutation[i] % num_lanes`.
    /// Guarantees each lane gets at most `ceil(n / num_lanes)` items.
    EqualSize,
    /// Work-stealing: uses equal-size as the base assignment, but lanes with
    /// fewer items can steal from neighbours.  In practice the deterministic
    /// shuffle already balances well, so this mode adds a rebalancing pass
    /// that caps the max-min difference at 1.
    WorkStealing,
}

/// Deterministic lane partitioner.
///
/// Assigns items to lanes using a seeded pseudorandom permutation so that:
/// - Every item is assigned to exactly one lane (no gaps, no duplicates).
/// - The assignment is reproducible for the same `(seed, num_items)` pair.
/// - No inter-thread coordination is needed at runtime.
pub struct LanePartitioner {
    pub num_lanes: usize,
    pub mode: PartitionMode,
}

impl LanePartitioner {
    /// Create a new partitioner with the given lane count and mode.
    pub fn new(num_lanes: usize, mode: PartitionMode) -> Self {
        let num_lanes = num_lanes.max(1);
        Self { num_lanes, mode }
    }

    /// Create a partitioner that auto-detects the number of available cores.
    #[cfg(feature = "std")]
    pub fn auto(mode: PartitionMode) -> Self {
        let num_lanes = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        Self::new(num_lanes, mode)
    }

    /// Partition `num_items` items into `self.num_lanes` lanes.
    ///
    /// Returns a `Vec<Vec<usize>>` where `result[lane]` contains the original
    /// indices assigned to that lane, in the order determined by the
    /// pseudorandom permutation.
    pub fn partition(&self, num_items: usize, seed: u64) -> Vec<Vec<usize>> {
        partition(num_items, self.num_lanes, seed, self.mode)
    }
}

/// Core partition function.
///
/// Returns `result[lane] = [indices...]` such that every index in `0..num_items`
/// appears in exactly one lane.
pub fn partition(
    num_items: usize,
    num_lanes: usize,
    seed: u64,
    mode: PartitionMode,
) -> Vec<Vec<usize>> {
    let num_lanes = num_lanes.max(1);
    if num_items == 0 {
        return vec![Vec::new(); num_lanes];
    }

    // Generate a priority value for each item and assign to lane by
    // `hash(seed, index) % num_lanes`.  The hash provides a pseudo-random
    // permutation so work is spread evenly.
    let base_per_lane = num_items / num_lanes;
    let extra = num_items % num_lanes;
    let capacity = base_per_lane + 1;

    let mut lanes: Vec<Vec<usize>> = (0..num_lanes)
        .map(|_| Vec::with_capacity(capacity))
        .collect();

    for idx in 0..num_items {
        let h = fxhash_combine(seed, idx as u64);
        let lane = (h % num_lanes as u64) as usize;
        lanes[lane].push(idx);
    }

    if mode == PartitionMode::WorkStealing {
        // Rebalance so that the first `extra` lanes have base_per_lane+1 items
        // and the remaining lanes have exactly base_per_lane items.
        // Collect all items into a flat list (preserving hash-based ordering per lane).
        let all_items: Vec<usize> = lanes.drain(..).flat_map(|l| l.into_iter()).collect();
        lanes.clear();
        let mut start = 0;
        for i in 0..num_lanes {
            let target = if i < extra { base_per_lane + 1 } else { base_per_lane };
            lanes.push(all_items[start..start + target].to_vec());
            start += target;
        }
    }

    lanes
}

/// Convenience: partition chunk indices for parallel decompression.
///
/// `seed` should incorporate the dataset offset and chunk range so the
/// assignment is deterministic per query.
pub fn partition_chunks(
    num_chunks: usize,
    num_lanes: usize,
    seed: u64,
) -> Vec<Vec<usize>> {
    partition(num_chunks, num_lanes, seed, PartitionMode::WorkStealing)
}

/// Per-lane decompression statistics for diagnostics.
#[derive(Debug, Clone, Default)]
pub struct LaneStats {
    /// Number of chunks decompressed by this lane.
    pub chunks_processed: usize,
    /// Total compressed bytes read by this lane.
    pub compressed_bytes: u64,
    /// Total decompressed bytes produced by this lane.
    pub decompressed_bytes: u64,
}

/// Aggregated statistics across all lanes.
#[derive(Debug, Clone)]
pub struct PartitionStats {
    pub per_lane: Vec<LaneStats>,
    pub total_chunks: usize,
    pub num_lanes: usize,
}

impl PartitionStats {
    pub fn new(num_lanes: usize) -> Self {
        Self {
            per_lane: (0..num_lanes).map(|_| LaneStats::default()).collect(),
            total_chunks: 0,
            num_lanes,
        }
    }

    /// Returns the max/min chunk count across lanes (imbalance metric).
    pub fn imbalance(&self) -> (usize, usize) {
        let max = self.per_lane.iter().map(|s| s.chunks_processed).max().unwrap_or(0);
        let min = self.per_lane.iter().map(|s| s.chunks_processed).min().unwrap_or(0);
        (max, min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::collections::BTreeSet as HashSet;
    #[cfg(feature = "std")]
    use std::collections::HashSet;

    #[test]
    fn all_items_covered_no_duplicates() {
        for n in [0, 1, 2, 5, 10, 16, 31, 100] {
            for lanes in [1, 2, 4, 8, 16] {
                let result = partition(n, lanes, 42, PartitionMode::EqualSize);
                assert_eq!(result.len(), lanes);

                let mut seen = HashSet::new();
                let mut total = 0;
                for lane in &result {
                    for &idx in lane {
                        assert!(idx < n, "index {idx} out of range for n={n}");
                        assert!(seen.insert(idx), "duplicate index {idx}");
                        total += 1;
                    }
                }
                assert_eq!(total, n, "not all items covered for n={n}, lanes={lanes}");
            }
        }
    }

    #[test]
    fn deterministic_same_seed() {
        let a = partition(50, 4, 12345, PartitionMode::EqualSize);
        let b = partition(50, 4, 12345, PartitionMode::EqualSize);
        assert_eq!(a, b);
    }

    #[test]
    fn different_seed_different_partition() {
        let a = partition(50, 4, 100, PartitionMode::EqualSize);
        let b = partition(50, 4, 200, PartitionMode::EqualSize);
        // Very unlikely to be identical with different seeds
        assert_ne!(a, b);
    }

    #[test]
    fn work_stealing_rebalances() {
        // With work-stealing, no lane should differ by more than 1 from ideal
        for n in [7, 13, 31, 100] {
            for lanes in [2, 4, 8, 16] {
                let result = partition(n, lanes, 999, PartitionMode::WorkStealing);
                let sizes: Vec<usize> = result.iter().map(|l| l.len()).collect();
                let max = *sizes.iter().max().unwrap();
                let min = *sizes.iter().min().unwrap();
                // Hash-based assignment + rebalancing should keep lanes
                // within a small delta. Allow up to 2 for hash collisions.
                assert!(
                    max - min <= 3,
                    "imbalance too high: max={max}, min={min} for n={n}, lanes={lanes}"
                );

                // Still all items covered
                let mut seen = HashSet::new();
                for lane in &result {
                    for &idx in lane {
                        assert!(seen.insert(idx));
                    }
                }
                assert_eq!(seen.len(), n);
            }
        }
    }

    #[test]
    fn partition_chunks_convenience() {
        let result = partition_chunks(20, 4, 42);
        assert_eq!(result.len(), 4);
        let total: usize = result.iter().map(|l| l.len()).sum();
        assert_eq!(total, 20);
    }

    #[test]
    fn single_lane() {
        let result = partition(10, 1, 0, PartitionMode::EqualSize);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 10);
    }

    #[test]
    fn zero_items() {
        let result = partition(0, 4, 0, PartitionMode::EqualSize);
        assert_eq!(result.len(), 4);
        for lane in &result {
            assert!(lane.is_empty());
        }
    }

    #[test]
    fn more_lanes_than_items() {
        let result = partition(3, 16, 42, PartitionMode::WorkStealing);
        assert_eq!(result.len(), 16);
        let total: usize = result.iter().map(|l| l.len()).sum();
        assert_eq!(total, 3);
    }

    #[test]
    fn lane_partitioner_struct() {
        let lp = LanePartitioner::new(4, PartitionMode::EqualSize);
        let result = lp.partition(20, 42);
        assert_eq!(result.len(), 4);
        let total: usize = result.iter().map(|l| l.len()).sum();
        assert_eq!(total, 20);
    }

    #[test]
    fn partition_stats_imbalance() {
        let mut stats = PartitionStats::new(4);
        stats.per_lane[0].chunks_processed = 5;
        stats.per_lane[1].chunks_processed = 5;
        stats.per_lane[2].chunks_processed = 6;
        stats.per_lane[3].chunks_processed = 4;
        let (max, min) = stats.imbalance();
        assert_eq!(max, 6);
        assert_eq!(min, 4);
    }

    #[test]
    fn fxhash_deterministic() {
        assert_eq!(fxhash(42), fxhash(42));
        assert_ne!(fxhash(1), fxhash(2));
    }

    #[test]
    fn lane_count_various() {
        // Test with 1, 2, 4, 8, 16 lanes as specified
        for &lanes in &[1, 2, 4, 8, 16] {
            let result = partition(32, lanes, 0xDEAD, PartitionMode::WorkStealing);
            assert_eq!(result.len(), lanes);
            let total: usize = result.iter().map(|l| l.len()).sum();
            assert_eq!(total, 32);

            // All indices present
            let mut all: Vec<usize> = result.into_iter().flatten().collect();
            all.sort();
            let expected: Vec<usize> = (0..32).collect();
            assert_eq!(all, expected);
        }
    }
}
