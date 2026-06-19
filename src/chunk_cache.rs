//! Chunk cache with hash-based index and LRU eviction.
//!
//! The [`ChunkCache`] avoids re-traversing B-trees on repeated reads of chunked
//! datasets.  On first access it scans the B-tree once and builds a
//! `HashMap<ChunkCoord, ChunkInfo>` (the *chunk index*).  Decompressed chunk
//! data is cached with LRU eviction controlled by a byte-budget.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
use crate::nosync::Mutex;
#[cfg(feature = "std")]
use std::sync::Mutex;

#[cfg(not(feature = "std"))]
use alloc::collections::BTreeMap;
#[cfg(feature = "std")]
use std::collections::HashMap;

use crate::chunked_read::ChunkInfo;

// ---------------------------------------------------------------------------
// Cache-line alignment
// ---------------------------------------------------------------------------

/// Cache line size in bytes for the target architecture.
///
/// ARM64 uses 128-byte cache lines; x86_64 uses 64-byte. The writer aligns
/// chunk data blocks to this boundary on disk so a chunk read lands on a
/// cache-line-aligned file offset.
#[cfg(target_arch = "aarch64")]
pub const CACHE_LINE_SIZE: usize = 128;

#[cfg(target_arch = "x86_64")]
pub const CACHE_LINE_SIZE: usize = 64;

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub const CACHE_LINE_SIZE: usize = 64;

/// Round `size` up to the next multiple of [`CACHE_LINE_SIZE`].
#[inline]
pub fn align_to_cache_line(size: usize) -> usize {
    (size + CACHE_LINE_SIZE - 1) & !(CACHE_LINE_SIZE - 1)
}

/// Coordinate key for a chunk — the N-dimensional offset vector.
pub type ChunkCoord = Vec<u64>;

/// Default maximum bytes of decompressed chunk data to cache.
pub const DEFAULT_CACHE_BYTES: usize = 1024 * 1024; // 1 MiB

/// Default maximum number of cached decompressed chunks.
pub const DEFAULT_MAX_SLOTS: usize = 16;

/// Configuration for a per-dataset chunk cache.
///
/// The byte and slot limits are the `hdf5-pure` counterpart of the
/// `rdcc_nbytes` and `rdcc_nslots` raw-data chunk-cache settings from HDF5's
/// `H5Pset_cache`. They apply to decompressed raw chunk data. The optional
/// chunk-index cache controls whether `hdf5-pure` retains the parsed chunk
/// address index between reads of the same [`crate::Dataset`]. Disabling the
/// index cache lowers retained metadata memory at the cost of re-scanning the
/// on-disk chunk index for repeated reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkCacheConfig {
    max_bytes: usize,
    max_slots: usize,
    cache_index: bool,
}

impl ChunkCacheConfig {
    /// Create a config matching the historical defaults: 1 MiB of decompressed
    /// chunks, 16 slots, and retained parsed chunk indexes.
    pub const fn new() -> Self {
        Self {
            max_bytes: DEFAULT_CACHE_BYTES,
            max_slots: DEFAULT_MAX_SLOTS,
            cache_index: true,
        }
    }

    /// Create a config from HDF5 `H5Pset_cache` raw data chunk-cache values.
    ///
    /// `rdcc_nslots` maps to the maximum retained chunk slots and
    /// `rdcc_nbytes` maps to the maximum retained decompressed chunk bytes.
    /// Modern HDF5 ignores `H5Pset_cache`'s `mdc_nelmts`; use
    /// [`crate::MetadataCacheConfig`] for the metadata-cache budget. The
    /// `rdcc_w0` preemption policy has no direct equivalent because this
    /// read-only cache uses strict LRU eviction.
    pub const fn from_h5p_cache(rdcc_nslots: usize, rdcc_nbytes: usize) -> Self {
        Self {
            max_bytes: rdcc_nbytes,
            max_slots: rdcc_nslots,
            cache_index: true,
        }
    }

    /// Disable retained decompressed chunks and parsed chunk indexes.
    pub const fn disabled() -> Self {
        Self {
            max_bytes: 0,
            max_slots: 0,
            cache_index: false,
        }
    }

    /// Set the maximum decompressed chunk bytes retained per dataset.
    pub const fn with_max_bytes(mut self, max_bytes: usize) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    /// Set the maximum number of decompressed chunk slots retained per dataset.
    pub const fn with_max_slots(mut self, max_slots: usize) -> Self {
        self.max_slots = max_slots;
        self
    }

    /// Enable or disable retaining the parsed chunk index between reads.
    pub const fn with_index_cache(mut self, enabled: bool) -> Self {
        self.cache_index = enabled;
        self
    }

    /// Return the maximum decompressed chunk bytes retained per dataset.
    pub const fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Return the maximum decompressed chunk slots retained per dataset.
    pub const fn max_slots(&self) -> usize {
        self.max_slots
    }

    /// Return whether parsed chunk indexes are retained between reads.
    pub const fn index_cache_enabled(&self) -> bool {
        self.cache_index
    }
}

impl Default for ChunkCacheConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// A read-only snapshot of a dataset's chunk-cache occupancy.
///
/// Returned by [`crate::Dataset::chunk_cache_stats`]. Use it to confirm a
/// chunk-cache configuration is taking effect: after reading a chunked dataset,
/// an enabled cache reports a loaded index and retained chunks, a disabled one
/// (or one over its byte/slot budget) reports fewer or none. The counts are a
/// point-in-time view and change as further reads populate or evict chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ChunkCacheStats {
    index_loaded: bool,
    cached_chunks: usize,
    cached_bytes: usize,
}

impl ChunkCacheStats {
    /// Whether the parsed chunk index is currently held in memory.
    pub const fn index_loaded(&self) -> bool {
        self.index_loaded
    }

    /// Number of decompressed chunks currently retained.
    pub const fn cached_chunks(&self) -> usize {
        self.cached_chunks
    }

    /// Total bytes of decompressed chunk data currently retained.
    pub const fn cached_bytes(&self) -> usize {
        self.cached_bytes
    }
}

// ---------------------------------------------------------------------------
// LRU entry
// ---------------------------------------------------------------------------

struct CachedChunk {
    coord: ChunkCoord,
    data: Vec<u8>,
    /// Monotonically increasing access counter for LRU ordering.
    last_access: u64,
}

// ---------------------------------------------------------------------------
// ChunkCache
// ---------------------------------------------------------------------------

/// A per-dataset chunk cache with hash-based index and LRU eviction.
///
/// # Usage
///
/// ```ignore
/// let cache = ChunkCache::new();
/// // Pass &cache to read_chunked_data — it will populate the index lazily.
/// ```
///
/// The cache is wrapped in `Mutex` internally so it can be mutated through
/// shared references (thread-safe).
pub struct ChunkCache {
    inner: Mutex<CacheInner>,
}

struct CacheInner {
    /// Hash index: chunk coordinate → ChunkInfo (offset + size in file).
    /// Populated once per dataset on first access.
    #[cfg(feature = "std")]
    index: Option<HashMap<ChunkCoord, ChunkInfo>>,
    #[cfg(not(feature = "std"))]
    index: Option<BTreeMap<ChunkCoord, ChunkInfo>>,

    /// LRU cache of decompressed chunk data.
    slots: Vec<CachedChunk>,

    /// Current total bytes of cached decompressed data.
    current_bytes: usize,

    /// Maximum bytes of decompressed data to cache.
    max_bytes: usize,

    /// Maximum number of slots.
    max_slots: usize,

    /// Monotonic counter for LRU ordering.
    tick: u64,

    /// Whether the parsed chunk index should be retained between reads.
    cache_index: bool,
}

impl ChunkCache {
    /// Create a new chunk cache with default limits (1 MiB, 16 slots).
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CACHE_BYTES, DEFAULT_MAX_SLOTS)
    }

    /// Create a new chunk cache with custom byte budget and slot count.
    pub fn with_capacity(max_bytes: usize, max_slots: usize) -> Self {
        Self::with_config(
            ChunkCacheConfig::new()
                .with_max_bytes(max_bytes)
                .with_max_slots(max_slots),
        )
    }

    /// Create a new chunk cache from a full configuration.
    pub fn with_config(config: ChunkCacheConfig) -> Self {
        Self {
            inner: Mutex::new(CacheInner {
                index: None,
                slots: Vec::with_capacity(config.max_slots.min(64)),
                current_bytes: 0,
                max_bytes: config.max_bytes,
                max_slots: config.max_slots,
                tick: 0,
                cache_index: config.cache_index,
            }),
        }
    }

    /// Snapshot the current chunk-cache occupancy (index loaded, retained
    /// chunk count, retained bytes).
    ///
    /// This is the public, read-only way to observe whether a chunk-cache
    /// configuration is taking effect. It locks the cache briefly to read a
    /// consistent snapshot.
    pub fn stats(&self) -> ChunkCacheStats {
        let inner = self.inner.lock().unwrap();
        ChunkCacheStats {
            index_loaded: inner.index.is_some(),
            cached_chunks: inner.slots.len(),
            cached_bytes: inner.current_bytes,
        }
    }

    // ----- Index operations -----

    /// Build the chunk index from a pre-collected list of `ChunkInfo`.
    ///
    /// The `rank` parameter is used to truncate offsets to spatial dims only
    /// (B-tree v1 stores rank+1 offsets).
    pub fn populate_index(&self, chunks: &[ChunkInfo], rank: usize) {
        let mut inner = self.inner.lock().unwrap();
        if !inner.cache_index {
            return;
        }
        if inner.index.is_some() {
            return; // already populated
        }
        #[cfg(feature = "std")]
        let mut map = HashMap::with_capacity(chunks.len());
        #[cfg(not(feature = "std"))]
        let mut map = BTreeMap::new();

        for ci in chunks {
            let coord: ChunkCoord = ci.offsets.iter().take(rank).copied().collect();
            map.insert(coord, ci.clone());
        }
        inner.index = Some(map);
    }

    /// Return all indexed chunks as a `Vec<ChunkInfo>` (order unspecified).
    pub fn all_indexed_chunks(&self) -> Option<Vec<ChunkInfo>> {
        let inner = self.inner.lock().unwrap();
        inner.index.as_ref().map(|m| m.values().cloned().collect())
    }

    // ----- Decompressed data cache (LRU) -----

    /// Run `f` over a borrowed view of a cached chunk's decompressed bytes, if
    /// present, returning its result.
    ///
    /// The closure runs while the cache lock is held, which lets the caller copy
    /// the chunk straight into its output buffer with no intermediate `Vec`
    /// allocation or clone. The closure must not touch this cache (it would
    /// deadlock); the chunk-assembly scatter it is used for does not.
    pub fn with_decompressed<R>(&self, coord: &[u64], f: impl FnOnce(&[u8]) -> R) -> Option<R> {
        let mut inner = self.inner.lock().unwrap();
        inner.tick += 1;
        let tick = inner.tick;
        for slot in inner.slots.iter_mut() {
            if slot.coord.as_slice() == coord {
                slot.last_access = tick;
                return Some(f(&slot.data));
            }
        }
        None
    }

    /// Whether a decompressed chunk of `data_len` bytes would be admitted to the
    /// cache (cache enabled and the chunk within the per-chunk byte budget). Used
    /// to skip copying a chunk into an owned buffer when it would be rejected.
    fn accepts_decompressed_len(&self, data_len: usize) -> bool {
        let inner = self.inner.lock().unwrap();
        inner.max_bytes != 0 && inner.max_slots != 0 && data_len <= inner.max_bytes
    }

    /// Insert decompressed chunk data into the LRU cache, taking ownership of the
    /// buffer (no copy). A chunk too large for the budget, or a disabled cache,
    /// drops the buffer instead of storing it.
    pub fn put_decompressed(&self, coord: ChunkCoord, data: Vec<u8>) {
        let mut inner = self.inner.lock().unwrap();
        let data_len = data.len();

        // Don't cache if disabled or if a single chunk exceeds the budget.
        if inner.max_bytes == 0 || inner.max_slots == 0 || data_len > inner.max_bytes {
            return;
        }

        // Check if already present
        inner.tick += 1;
        let tick = inner.tick;
        for slot in inner.slots.iter_mut() {
            if slot.coord == coord {
                slot.last_access = tick;
                return; // already cached
            }
        }

        // Evict until we have room
        while inner.slots.len() >= inner.max_slots
            || (inner.current_bytes + data_len > inner.max_bytes && !inner.slots.is_empty())
        {
            // Find LRU slot
            let lru_idx = inner
                .slots
                .iter()
                .enumerate()
                .min_by_key(|(_, s)| s.last_access)
                .map(|(i, _)| i)
                .unwrap();
            let removed = inner.slots.swap_remove(lru_idx);
            inner.current_bytes -= removed.data.len();
        }

        inner.current_bytes += data_len;
        inner.slots.push(CachedChunk {
            coord,
            data,
            last_access: tick,
        });
    }

    /// Insert a copy of `data` into the LRU cache, but only if it would actually
    /// be admitted. This lets the unfiltered read path scatter directly from the
    /// file buffer and copy into the cache only when caching is enabled and the
    /// chunk fits the budget (avoiding a throwaway copy otherwise).
    pub fn put_decompressed_slice(&self, coord: ChunkCoord, data: &[u8]) {
        if !self.accepts_decompressed_len(data.len()) {
            return;
        }
        self.put_decompressed(coord, data.to_vec());
    }

    /// Clear the entire cache (index + decompressed data).
    ///
    /// Currently only exercised by unit tests; gated so it is not shipped as
    /// dead code.
    #[cfg(test)]
    pub fn clear(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.index = None;
        inner.slots.clear();
        inner.current_bytes = 0;
        inner.tick = 0;
    }
}

impl Default for ChunkCache {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunk(offsets: Vec<u64>, address: u64, size: u32) -> ChunkInfo {
        ChunkInfo {
            chunk_size: size,
            filter_mask: 0,
            offsets,
            address,
        }
    }

    #[test]
    fn index_populate_and_lookup() {
        let cache = ChunkCache::new();
        let chunks = vec![
            make_chunk(vec![0, 0, 0], 0x1000, 80),
            make_chunk(vec![10, 0, 0], 0x2000, 80),
        ];
        cache.populate_index(&chunks, 2); // rank=2, truncate to [0,0] and [10,0]
        assert!(cache.stats().index_loaded());

        let mut addrs: Vec<u64> = cache
            .all_indexed_chunks()
            .unwrap()
            .iter()
            .map(|c| c.address)
            .collect();
        addrs.sort_unstable();
        assert_eq!(addrs, vec![0x1000, 0x2000]);
    }

    /// Test helper: clone a cached chunk's bytes if present (the production
    /// path uses `with_decompressed` to avoid this copy).
    fn get_decompressed(cache: &ChunkCache, coord: &[u64]) -> Option<Vec<u8>> {
        cache.with_decompressed(coord, <[u8]>::to_vec)
    }

    #[test]
    fn decompressed_cache_hit() {
        let cache = ChunkCache::new();
        cache.put_decompressed(vec![0, 0], vec![1, 2, 3, 4]);
        let got = get_decompressed(&cache, &[0, 0]).unwrap();
        assert_eq!(got, vec![1, 2, 3, 4]);
    }

    #[test]
    fn lru_eviction_by_slots() {
        let cache = ChunkCache::with_capacity(1024 * 1024, 2); // max 2 slots

        cache.put_decompressed(vec![0], vec![1; 10]);
        cache.put_decompressed(vec![1], vec![2; 10]);
        assert_eq!(cache.stats().cached_chunks(), 2);

        // Access slot 0 to make it more recent
        get_decompressed(&cache, &[0]);

        // Insert slot 2 — should evict slot 1 (LRU)
        cache.put_decompressed(vec![2], vec![3; 10]);
        assert_eq!(cache.stats().cached_chunks(), 2);

        assert!(get_decompressed(&cache, &[0]).is_some());
        assert!(get_decompressed(&cache, &[1]).is_none()); // evicted
        assert!(get_decompressed(&cache, &[2]).is_some());
    }

    #[test]
    fn lru_eviction_by_bytes() {
        let cache = ChunkCache::with_capacity(50, 100); // 50 bytes max

        cache.put_decompressed(vec![0], vec![0; 20]);
        cache.put_decompressed(vec![1], vec![0; 20]);
        assert_eq!(cache.stats().cached_bytes(), 40);

        // This needs 20 bytes but only 10 free — evict LRU
        cache.put_decompressed(vec![2], vec![0; 20]);
        assert!(cache.stats().cached_bytes() <= 50);
        assert!(get_decompressed(&cache, &[0]).is_none()); // evicted (LRU)
    }

    #[test]
    fn put_decompressed_slice_only_copies_when_admitted() {
        // Disabled cache: the slice is not copied or stored.
        let cache = ChunkCache::with_config(ChunkCacheConfig::disabled());
        cache.put_decompressed_slice(vec![0], &[1, 2, 3]);
        assert_eq!(cache.stats().cached_chunks(), 0);

        // Enabled cache within budget: stored.
        let cache = ChunkCache::with_capacity(1024, 16);
        cache.put_decompressed_slice(vec![0], &[1, 2, 3, 4]);
        assert_eq!(get_decompressed(&cache, &[0]).unwrap(), vec![1, 2, 3, 4]);

        // Over the per-chunk budget: not stored.
        let cache = ChunkCache::with_capacity(2, 16);
        cache.put_decompressed_slice(vec![0], &[1, 2, 3, 4]);
        assert_eq!(cache.stats().cached_chunks(), 0);
    }

    #[test]
    fn oversized_chunk_not_cached() {
        let cache = ChunkCache::with_capacity(10, 16);
        cache.put_decompressed(vec![0], vec![0; 100]); // too big
        assert_eq!(cache.stats().cached_chunks(), 0);
    }

    #[test]
    fn disabled_cache_retains_no_index_or_chunks() {
        let cache = ChunkCache::with_config(ChunkCacheConfig::disabled());
        let chunks = vec![make_chunk(vec![0, 0], 0x1000, 80)];
        cache.populate_index(&chunks, 1);
        assert!(!cache.stats().index_loaded());

        cache.put_decompressed(vec![0], vec![1, 2, 3]);
        assert_eq!(cache.stats().cached_chunks(), 0);
        assert_eq!(cache.stats().cached_bytes(), 0);
    }

    #[test]
    fn h5p_cache_constructor_maps_raw_data_chunk_settings() {
        let config = ChunkCacheConfig::from_h5p_cache(521, 2 * 1024 * 1024);
        assert_eq!(config.max_slots(), 521);
        assert_eq!(config.max_bytes(), 2 * 1024 * 1024);
        assert!(config.index_cache_enabled());
    }

    #[test]
    fn clear_resets_everything() {
        let cache = ChunkCache::new();
        let chunks = vec![make_chunk(vec![0, 0], 0x1000, 80)];
        cache.populate_index(&chunks, 1);
        cache.put_decompressed(vec![0], vec![1, 2, 3]);

        cache.clear();
        assert!(!cache.stats().index_loaded());
        assert_eq!(cache.stats().cached_chunks(), 0);
        assert_eq!(cache.stats().cached_bytes(), 0);
    }

    #[test]
    fn duplicate_insert_is_noop() {
        let cache = ChunkCache::new();
        cache.put_decompressed(vec![0], vec![1, 2, 3]);
        cache.put_decompressed(vec![0], vec![1, 2, 3]); // duplicate
        assert_eq!(cache.stats().cached_chunks(), 1);
        assert_eq!(cache.stats().cached_bytes(), 3);
    }

    #[test]
    fn align_to_cache_line_values() {
        assert_eq!(align_to_cache_line(0), 0);
        assert_eq!(align_to_cache_line(1), CACHE_LINE_SIZE);
        assert_eq!(align_to_cache_line(CACHE_LINE_SIZE), CACHE_LINE_SIZE);
        assert_eq!(
            align_to_cache_line(CACHE_LINE_SIZE + 1),
            CACHE_LINE_SIZE * 2
        );
    }
}
