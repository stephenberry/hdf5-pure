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
    /// The read pass that stored it, so that pass cannot evict it again.
    stored_by: u64,
}

/// One read's pass over a set of chunks, as far as cache admission is concerned.
///
/// Every read path in this crate visits each of its chunks exactly once. Within
/// one such pass, evicting a chunk to make room for another is work with no
/// upside *to that pass*: the evicted chunk has already been placed and will not
/// be asked for again, and neither will the one that displaced it. A whole read
/// of a dataset larger than the cache did exactly that — 2,048 chunks offered to
/// 16 slots, 2,032 of them evicted by the same read that stored them, an
/// allocator round trip each and, on the unfiltered path, a copy of the chunk as
/// well (issue #228).
///
/// So a pass fills the cache and then stops offering, and what it leaves behind
/// is the chunks it reached first rather than the ones it reached last.
///
/// # Which half is worth keeping is the caller's question, not this type's
///
/// That last sentence is the whole trade, and it does not go the same way for
/// every read. Keeping the *tail* is only possible by offering every chunk and
/// evicting, which is the cost this exists to remove — so a read that wants the
/// tail asks for [`CachePass::LRU`] and pays for it.
///
/// A read of a whole dataset does not want it: a caller who reads it again
/// starts at the beginning, so a retained prefix is worth at least as much as a
/// retained suffix, and it costs a fraction as much to keep. A *lone* row window
/// does want it, because its successor is the adjacent window and the chunk they
/// share is the one this read finished on — which is why
/// [`Dataset::read_raw_rows`](crate::Dataset::read_raw_rows) asks for `LRU` while
/// the two whole-dataset loops open a real pass.
///
/// The windowed reader itself takes the pass from its caller rather than
/// choosing, because the same window means different things to different
/// callers: a sweep of a whole dataset in windows opens one real pass for all of
/// them, since it asks for each chunk exactly once and has no more use for the
/// last window's chunks than for the first's.
///
/// Across passes nothing changes either way: a later read still evicts what an
/// earlier one left, so the cache goes on tracking the most recent access
/// pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CachePass(u64);

impl CachePass {
    /// Admission by the plain LRU rule this cache had before passes existed:
    /// every slot is evictable, including one this same identity stored.
    ///
    /// It is recognized by *being* this value rather than by its number. Zero is
    /// outside the range [`ChunkCache::begin_pass`] hands out, which keeps a real
    /// pass from ever being mistaken for it — but that alone would not be enough
    /// in the other direction, since every `LRU` insert records the same
    /// `stored_by` and would then look like its own pass's work.
    pub const LRU: CachePass = CachePass(0);
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

    /// Monotonic counter handing out [`CachePass`] identities.
    pass: u64,

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
                pass: 0,
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
        self.indexed_chunks_matching(|_| true)
    }

    /// Return the indexed chunks `keep` accepts, as a `Vec<ChunkInfo>` (order
    /// unspecified).
    ///
    /// A row window wants the chunks its rows overlap and nothing else, and the
    /// difference is not a nicety: each [`ChunkInfo`] owns a coordinate `Vec`, so
    /// taking the whole index and discarding most of it costs an allocation per
    /// chunk *of the dataset* per window. A sweep of a dataset in windows paid
    /// that as a product — 8 windows over 2,048 chunks, 16,384 allocations to
    /// visit 2,048 chunks (issue #289). The filter runs while the lock is held,
    /// over borrowed entries, so a rejected chunk costs no allocation at all.
    pub fn indexed_chunks_matching(
        &self,
        keep: impl Fn(&ChunkInfo) -> bool,
    ) -> Option<Vec<ChunkInfo>> {
        let inner = self.inner.lock().unwrap();
        inner
            .index
            .as_ref()
            .map(|m| m.values().filter(|ci| keep(ci)).cloned().collect())
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

    /// Opens a read pass. See [`CachePass`] for what one is and why it exists.
    pub fn begin_pass(&self) -> CachePass {
        let mut inner = self.inner.lock().unwrap();
        inner.pass += 1;
        CachePass(inner.pass)
    }

    /// Makes room for a `data_len`-byte chunk at `coord`, reporting whether the
    /// caller should go on to store it.
    ///
    /// The single place the admission policy lives, so the owned and borrowed
    /// entry points below cannot drift — and so the borrowed one learns it has
    /// nowhere to put the chunk *before* copying it rather than after.
    ///
    /// **On `true` the caller must push a slot**: the byte total has already been
    /// charged for it, and a caller that returned instead would leave the cache
    /// believing it holds bytes nothing occupies. On `false` nothing was changed
    /// beyond the LRU tick.
    fn reserve(inner: &mut CacheInner, pass: CachePass, coord: &[u64], data_len: usize) -> bool {
        // Don't cache if disabled or if a single chunk exceeds the budget.
        if inner.max_bytes == 0 || inner.max_slots == 0 || data_len > inner.max_bytes {
            return false;
        }

        // Check if already present
        inner.tick += 1;
        let tick = inner.tick;
        for slot in inner.slots.iter_mut() {
            if slot.coord == coord {
                slot.last_access = tick;
                return false; // already cached
            }
        }

        // A chunk this same pass stored is not taken back: that trades a chunk
        // nobody will ask for again for another one nobody will ask for again.
        //
        // Unless this is [`CachePass::LRU`], which asks for the plain rule and
        // must therefore be allowed to evict what it stored itself. Testing that
        // by identity rather than leaning on `LRU`'s number is the whole of it:
        // every `LRU` insert records the same `stored_by`, so an identity
        // comparison alone would make the second one see the first as its own and
        // refuse — turning the plain rule into fill-once for the life of the
        // cache. `a_pass_marked_lru_evicts_its_own_chunks` is that bug's test.
        let evicts_its_own = pass == CachePass::LRU;
        let reclaimable = |slot: &&CachedChunk| evicts_its_own || slot.stored_by != pass.0;

        // Whether the reclaimable slots can make room *at all*, decided before
        // anything is removed. Evicting some and then finding the rest untouchable
        // would leave the cache holding less and storing nothing — a chunk given
        // up for no one. Removing every reclaimable slot is the most room there is
        // to be had, so the test is the loop's own exit condition evaluated
        // against that state.
        let (freed_slots, freed_bytes) = inner
            .slots
            .iter()
            .filter(reclaimable)
            .fold((0usize, 0usize), |(n, b), slot| {
                (n + 1, b + slot.data.len())
            });
        let (least_slots, least_bytes) = (
            inner.slots.len() - freed_slots,
            inner.current_bytes - freed_bytes,
        );
        if least_slots >= inner.max_slots
            || (least_bytes + data_len > inner.max_bytes && least_slots > 0)
        {
            return false;
        }

        // Evict in LRU order until there is room. The check above proves a
        // reclaimable slot exists for as long as this condition holds, so the
        // `else` below cannot be reached; it returns rather than storing over
        // budget in case that reasoning is ever made false.
        while inner.slots.len() >= inner.max_slots
            || (inner.current_bytes + data_len > inner.max_bytes && !inner.slots.is_empty())
        {
            // The LRU slot among those an earlier pass stored.
            let lru_idx = inner
                .slots
                .iter()
                .enumerate()
                .filter(|(_, s)| reclaimable(s))
                .min_by_key(|(_, s)| s.last_access)
                .map(|(i, _)| i);
            let Some(lru_idx) = lru_idx else {
                debug_assert!(
                    false,
                    "the feasibility check above admitted a chunk this pass cannot make room for"
                );
                return false;
            };
            let removed = inner.slots.swap_remove(lru_idx);
            inner.current_bytes -= removed.data.len();
        }

        inner.current_bytes += data_len;
        true
    }

    /// Insert decompressed chunk data into the LRU cache, taking ownership of the
    /// buffer (no copy). A chunk too large for the budget, a disabled cache, or a
    /// pass that has already filled the cache drops the buffer instead of storing
    /// it.
    ///
    /// `coord` is borrowed and copied only on the path that stores it, so a
    /// caller in a loop needs no owned coordinate per chunk.
    pub fn put_decompressed(&self, pass: CachePass, coord: &[u64], data: Vec<u8>) {
        let mut inner = self.inner.lock().unwrap();
        if !Self::reserve(&mut inner, pass, coord, data.len()) {
            return;
        }
        let last_access = inner.tick;
        inner.slots.push(CachedChunk {
            coord: coord.to_vec(),
            data,
            last_access,
            stored_by: pass.0,
        });
    }

    /// The coordinates of every chunk whose decompressed bytes this cache
    /// currently holds, in no particular order.
    ///
    /// A reader planning coalesced reads (see [`crate::chunk_span`]) uses this
    /// to leave the chunks it already has out of the plan: a span covering a
    /// chunk the read will skip fetches those bytes for nothing. It answers in
    /// one lock over at most [`ChunkCacheConfig::max_slots`] entries, where
    /// probing per chunk would take a lock apiece.
    ///
    /// The answer is a snapshot. A chunk named here can be evicted before the
    /// read reaches it — by this very read, admitting later chunks — which
    /// costs the coalescing for that chunk and nothing else: a chunk in no span
    /// is read directly.
    pub fn decompressed_coords(&self) -> Vec<ChunkCoord> {
        let inner = self.inner.lock().unwrap();
        inner.slots.iter().map(|s| s.coord.clone()).collect()
    }

    /// Insert a copy of `data` into the LRU cache, but only if it will actually
    /// be kept. This lets the unfiltered read path scatter directly from the file
    /// buffer and copy into the cache only when the chunk is going to stay there
    /// — no copy at all when caching is off, when the chunk is over the budget,
    /// or when this pass has already filled the cache.
    pub fn put_decompressed_slice(&self, pass: CachePass, coord: &[u64], data: &[u8]) {
        let mut inner = self.inner.lock().unwrap();
        if !Self::reserve(&mut inner, pass, coord, data.len()) {
            return;
        }
        let last_access = inner.tick;
        inner.slots.push(CachedChunk {
            coord: coord.to_vec(),
            data: data.to_vec(),
            last_access,
            stored_by: pass.0,
        });
    }

    /// Clear the entire cache (index + decompressed data).
    ///
    /// Called after a mutation through the owning [`Dataset`](crate::Dataset)
    /// handle: an append relocates the trailing chunk and adds new index
    /// entries, so both the cached chunk index and any retained decompressed
    /// chunks may be stale.
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
        cache.put_decompressed(cache.begin_pass(), &[0, 0], vec![1, 2, 3, 4]);
        let got = get_decompressed(&cache, &[0, 0]).unwrap();
        assert_eq!(got, vec![1, 2, 3, 4]);
    }

    #[test]
    fn lru_eviction_by_slots() {
        let cache = ChunkCache::with_capacity(1024 * 1024, 2); // max 2 slots

        cache.put_decompressed(cache.begin_pass(), &[0], vec![1; 10]);
        cache.put_decompressed(cache.begin_pass(), &[1], vec![2; 10]);
        assert_eq!(cache.stats().cached_chunks(), 2);

        // Access slot 0 to make it more recent
        get_decompressed(&cache, &[0]);

        // Insert slot 2 — should evict slot 1 (LRU)
        cache.put_decompressed(cache.begin_pass(), &[2], vec![3; 10]);
        assert_eq!(cache.stats().cached_chunks(), 2);

        assert!(get_decompressed(&cache, &[0]).is_some());
        assert!(get_decompressed(&cache, &[1]).is_none()); // evicted
        assert!(get_decompressed(&cache, &[2]).is_some());
    }

    #[test]
    fn lru_eviction_by_bytes() {
        let cache = ChunkCache::with_capacity(50, 100); // 50 bytes max

        cache.put_decompressed(cache.begin_pass(), &[0], vec![0; 20]);
        cache.put_decompressed(cache.begin_pass(), &[1], vec![0; 20]);
        assert_eq!(cache.stats().cached_bytes(), 40);

        // This needs 20 bytes but only 10 free — evict LRU
        cache.put_decompressed(cache.begin_pass(), &[2], vec![0; 20]);
        assert!(cache.stats().cached_bytes() <= 50);
        assert!(get_decompressed(&cache, &[0]).is_none()); // evicted (LRU)
    }

    #[test]
    fn put_decompressed_slice_only_copies_when_admitted() {
        // Disabled cache: the slice is not copied or stored.
        let cache = ChunkCache::with_config(ChunkCacheConfig::disabled());
        cache.put_decompressed_slice(cache.begin_pass(), &[0], &[1, 2, 3]);
        assert_eq!(cache.stats().cached_chunks(), 0);

        // Enabled cache within budget: stored.
        let cache = ChunkCache::with_capacity(1024, 16);
        cache.put_decompressed_slice(cache.begin_pass(), &[0], &[1, 2, 3, 4]);
        assert_eq!(get_decompressed(&cache, &[0]).unwrap(), vec![1, 2, 3, 4]);

        // Over the per-chunk budget: not stored.
        let cache = ChunkCache::with_capacity(2, 16);
        cache.put_decompressed_slice(cache.begin_pass(), &[0], &[1, 2, 3, 4]);
        assert_eq!(cache.stats().cached_chunks(), 0);
    }

    /// The rule [`CachePass`] exists for: one pass fills the cache and then
    /// stops, rather than spending a copy per chunk to evict what it just
    /// stored. A later pass is free to replace all of it.
    #[test]
    fn a_pass_fills_the_cache_and_then_stops_evicting_itself() {
        let cache = ChunkCache::with_capacity(1024 * 1024, 2);

        // One pass over four chunks, as a read of a four-chunk dataset makes.
        let pass = cache.begin_pass();
        for c in 0..4u64 {
            cache.put_decompressed(pass, &[c], vec![c as u8; 10]);
        }

        // The two it reached first are the two it kept: chunks 2 and 3 were
        // never copied, and chunks 0 and 1 were not evicted to make room for
        // them.
        assert_eq!(cache.stats().cached_chunks(), 2);
        assert!(get_decompressed(&cache, &[0]).is_some());
        assert!(get_decompressed(&cache, &[1]).is_some());
        assert!(get_decompressed(&cache, &[2]).is_none());
        assert!(get_decompressed(&cache, &[3]).is_none());

        // A second read is a second pass, and it may take both slots back.
        let next = cache.begin_pass();
        cache.put_decompressed(next, &[9], vec![9; 10]);
        cache.put_decompressed(next, &[8], vec![8; 10]);
        assert_eq!(cache.stats().cached_chunks(), 2);
        assert!(get_decompressed(&cache, &[9]).is_some());
        assert!(get_decompressed(&cache, &[8]).is_some());
    }

    /// [`CachePass::LRU`] is a sentinel: it works only because a real pass is
    /// never numbered zero. A `begin_pass` that started counting at zero would
    /// silently turn the windowed reader's plain-LRU admission into fill-once and
    /// lose it the boundary chunk its successor window needs.
    #[test]
    fn a_pass_marked_lru_evicts_its_own_chunks() {
        let cache = ChunkCache::with_capacity(1024 * 1024, 2);

        for c in 0..4u64 {
            cache.put_decompressed(CachePass::LRU, &[c], vec![c as u8; 10]);
        }

        // The last two, where a fill-once pass would have kept the first two.
        assert_eq!(cache.stats().cached_chunks(), 2);
        assert!(get_decompressed(&cache, &[2]).is_some());
        assert!(get_decompressed(&cache, &[3]).is_some());
        assert!(get_decompressed(&cache, &[0]).is_none());

        // The property that makes the sentinel sound, asserted rather than
        // assumed: no real pass can collide with it.
        assert_ne!(cache.begin_pass(), CachePass::LRU);
    }

    /// A pass that gives up must not have taken anything with it. Reclaiming some
    /// slots and then finding the rest untouchable would leave the cache holding
    /// less and storing nothing — a chunk dropped for no one.
    #[test]
    fn a_pass_that_cannot_make_room_evicts_nothing() {
        // 100 bytes, plenty of slots: only the byte budget can bite.
        let cache = ChunkCache::with_capacity(100, 16);

        let first = cache.begin_pass();
        cache.put_decompressed(first, &[0], vec![0; 10]);

        let second = cache.begin_pass();
        cache.put_decompressed(second, &[1], vec![1; 80]);
        assert_eq!(cache.stats().cached_chunks(), 2);

        // 80 bytes more will not fit even with the 10-byte chunk from `first`
        // reclaimed, and the 80-byte one belongs to this pass. The old code
        // evicted the reclaimable chunk first and gave up afterwards.
        cache.put_decompressed(second, &[2], vec![2; 80]);
        assert_eq!(cache.stats().cached_chunks(), 2);
        assert_eq!(cache.stats().cached_bytes(), 90);
        assert!(get_decompressed(&cache, &[0]).is_some());
    }

    /// The same rule on the borrowed entry point, where it also decides whether
    /// the chunk is copied at all.
    #[test]
    fn a_full_pass_does_not_copy_the_chunk_it_cannot_store() {
        let cache = ChunkCache::with_capacity(1024 * 1024, 1);
        let pass = cache.begin_pass();

        cache.put_decompressed_slice(pass, &[0], &[1; 10]);
        cache.put_decompressed_slice(pass, &[1], &[2; 10]);

        assert_eq!(cache.stats().cached_chunks(), 1);
        assert_eq!(cache.stats().cached_bytes(), 10);
        assert!(get_decompressed(&cache, &[0]).is_some());
    }

    #[test]
    fn oversized_chunk_not_cached() {
        let cache = ChunkCache::with_capacity(10, 16);
        cache.put_decompressed(cache.begin_pass(), &[0], vec![0; 100]); // too big
        assert_eq!(cache.stats().cached_chunks(), 0);
    }

    #[test]
    fn disabled_cache_retains_no_index_or_chunks() {
        let cache = ChunkCache::with_config(ChunkCacheConfig::disabled());
        let chunks = vec![make_chunk(vec![0, 0], 0x1000, 80)];
        cache.populate_index(&chunks, 1);
        assert!(!cache.stats().index_loaded());

        cache.put_decompressed(cache.begin_pass(), &[0], vec![1, 2, 3]);
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
        cache.put_decompressed(cache.begin_pass(), &[0], vec![1, 2, 3]);

        cache.clear();
        assert!(!cache.stats().index_loaded());
        assert_eq!(cache.stats().cached_chunks(), 0);
        assert_eq!(cache.stats().cached_bytes(), 0);
    }

    #[test]
    fn duplicate_insert_is_noop() {
        let cache = ChunkCache::new();
        cache.put_decompressed(cache.begin_pass(), &[0], vec![1, 2, 3]);
        cache.put_decompressed(cache.begin_pass(), &[0], vec![1, 2, 3]); // duplicate
        assert_eq!(cache.stats().cached_chunks(), 1);
        assert_eq!(cache.stats().cached_bytes(), 3);
    }
}
