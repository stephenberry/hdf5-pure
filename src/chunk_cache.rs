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
use alloc::alloc::{alloc_zeroed, dealloc, handle_alloc_error};
#[cfg(feature = "std")]
use std::alloc::{alloc_zeroed, dealloc, handle_alloc_error};

#[cfg(not(feature = "std"))]
use crate::nosync::Mutex;
#[cfg(feature = "std")]
use std::sync::Mutex;

use core::ops::{Deref, DerefMut};

#[cfg(not(feature = "std"))]
use alloc::collections::BTreeMap;
#[cfg(feature = "std")]
use std::collections::HashMap;

use crate::chunked_read::ChunkInfo;

// ---------------------------------------------------------------------------
// Cache-line alignment constants (TVL — Tensor Virtualization Layout)
// ---------------------------------------------------------------------------

/// Cache line size in bytes for the target architecture.
///
/// ARM64 uses 128-byte cache lines; x86_64 uses 64-byte. We align all chunk
/// buffers to this boundary so SIMD operations can assume aligned input.
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

// ---------------------------------------------------------------------------
// CacheAlignedBuffer
// ---------------------------------------------------------------------------

/// A byte buffer whose data pointer is aligned to [`CACHE_LINE_SIZE`].
///
/// This enables SIMD operations to use aligned loads/stores when processing
/// chunk data, avoiding the penalty of misaligned memory accesses.
///
/// The buffer is backed by `std::alloc::Layout`-controlled allocation. It
/// dereferences to `&[u8]` / `&mut [u8]` for seamless use.
pub struct CacheAlignedBuffer {
    ptr: *mut u8,
    len: usize,
    capacity: usize,
}

// SAFETY: The raw pointer is exclusively owned — no aliasing.
unsafe impl Send for CacheAlignedBuffer {}
unsafe impl Sync for CacheAlignedBuffer {}

impl CacheAlignedBuffer {
    /// Allocate a new cache-line-aligned buffer of exactly `len` bytes,
    /// initialized to zero.
    pub fn zeroed(len: usize) -> Self {
        if len == 0 {
            return Self {
                ptr: core::ptr::NonNull::dangling().as_ptr(),
                len: 0,
                capacity: 0,
            };
        }
        let capacity = align_to_cache_line(len);
        let layout = core::alloc::Layout::from_size_align(capacity, CACHE_LINE_SIZE)
            .expect("invalid layout");
        // SAFETY: layout has non-zero size.
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        Self { ptr, len, capacity }
    }

    /// Create a cache-line-aligned copy of an existing byte slice.
    pub fn from_slice(data: &[u8]) -> Self {
        let mut buf = Self::zeroed(data.len());
        buf.as_mut_slice()[..data.len()].copy_from_slice(data);
        buf
    }

    /// Create from an existing `Vec<u8>`, copying into an aligned allocation.
    pub fn from_vec(v: Vec<u8>) -> Self {
        Self::from_slice(&v)
    }

    /// The length of the valid data (may be less than capacity).
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    #[inline]
    #[allow(dead_code)] // exercised by unit tests; companion to `len`
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Borrow as a byte slice.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: ptr is valid for `len` bytes and properly aligned.
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Borrow as a mutable byte slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.len == 0 {
            return &mut [];
        }
        // SAFETY: ptr is valid for `len` bytes and properly aligned.
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Convert to a `Vec<u8>` (copies data into a standard allocation).
    pub fn to_vec(&self) -> Vec<u8> {
        self.as_slice().to_vec()
    }

    /// Returns `true` if the data pointer is aligned to `CACHE_LINE_SIZE`.
    #[inline]
    pub fn is_aligned(&self) -> bool {
        self.len == 0 || (self.ptr as usize).is_multiple_of(CACHE_LINE_SIZE)
    }
}

impl Drop for CacheAlignedBuffer {
    fn drop(&mut self) {
        if self.capacity > 0 {
            let layout = core::alloc::Layout::from_size_align(self.capacity, CACHE_LINE_SIZE)
                .expect("invalid layout");
            // SAFETY: ptr was allocated with this layout.
            unsafe { dealloc(self.ptr, layout) };
        }
    }
}

impl Clone for CacheAlignedBuffer {
    fn clone(&self) -> Self {
        Self::from_slice(self.as_slice())
    }
}

impl Deref for CacheAlignedBuffer {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl DerefMut for CacheAlignedBuffer {
    #[inline]
    fn deref_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

impl core::fmt::Debug for CacheAlignedBuffer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CacheAlignedBuffer")
            .field("len", &self.len)
            .field("capacity", &self.capacity)
            .field("aligned", &self.is_aligned())
            .finish()
    }
}

/// Coordinate key for a chunk — the N-dimensional offset vector.
pub type ChunkCoord = Vec<u64>;

/// Default maximum bytes of decompressed chunk data to cache.
pub const DEFAULT_CACHE_BYTES: usize = 1024 * 1024; // 1 MiB

/// Default maximum number of cached decompressed chunks.
pub const DEFAULT_MAX_SLOTS: usize = 16;

/// Configuration for a per-dataset chunk cache.
///
/// The byte and slot limits apply to decompressed raw chunk data. The optional
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

// ---------------------------------------------------------------------------
// LRU entry
// ---------------------------------------------------------------------------

struct CachedChunk {
    coord: ChunkCoord,
    data: CacheAlignedBuffer,
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

    // ----- Index operations -----

    /// Returns `true` if the chunk index has been built.
    #[cfg(test)]
    pub fn has_index(&self) -> bool {
        self.inner.lock().unwrap().index.is_some()
    }

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

    /// Try to get cached decompressed data for a chunk coordinate.
    ///
    /// Returns a clone of the cache-line-aligned buffer.
    pub fn get_decompressed(&self, coord: &[u64]) -> Option<Vec<u8>> {
        let mut inner = self.inner.lock().unwrap();
        inner.tick += 1;
        let tick = inner.tick;

        for slot in inner.slots.iter_mut() {
            if slot.coord.as_slice() == coord {
                slot.last_access = tick;
                return Some(slot.data.to_vec());
            }
        }
        None
    }

    /// Try to get a reference-counted clone of the aligned buffer for a chunk.
    ///
    /// Currently only exercised by unit tests; gated so it is not shipped as
    /// dead code.
    #[cfg(test)]
    pub fn get_decompressed_aligned(&self, coord: &[u64]) -> Option<CacheAlignedBuffer> {
        let mut inner = self.inner.lock().unwrap();
        inner.tick += 1;
        let tick = inner.tick;
        for slot in inner.slots.iter_mut() {
            if slot.coord.as_slice() == coord {
                slot.last_access = tick;
                return Some(slot.data.clone());
            }
        }
        None
    }

    /// Insert decompressed chunk data into the LRU cache.
    ///
    /// The data is stored in a [`CacheAlignedBuffer`] so subsequent reads
    /// return cache-line-aligned memory.
    pub fn put_decompressed(&self, coord: ChunkCoord, data: Vec<u8>) {
        let aligned = CacheAlignedBuffer::from_slice(&data);
        self.put_decompressed_aligned(coord, aligned);
    }

    /// Insert an already-aligned buffer into the LRU cache.
    pub fn put_decompressed_aligned(&self, coord: ChunkCoord, data: CacheAlignedBuffer) {
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

    /// Number of decompressed chunks currently cached.
    ///
    /// Currently only exercised by unit tests; gated so it is not shipped as
    /// dead code.
    #[cfg(test)]
    pub fn cached_chunk_count(&self) -> usize {
        self.inner.lock().unwrap().slots.len()
    }

    /// Total bytes of decompressed data currently cached.
    ///
    /// Currently only exercised by unit tests; gated so it is not shipped as
    /// dead code.
    #[cfg(test)]
    pub fn cached_bytes(&self) -> usize {
        self.inner.lock().unwrap().current_bytes
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
        assert!(cache.has_index());

        let mut addrs: Vec<u64> = cache
            .all_indexed_chunks()
            .unwrap()
            .iter()
            .map(|c| c.address)
            .collect();
        addrs.sort_unstable();
        assert_eq!(addrs, vec![0x1000, 0x2000]);
    }

    #[test]
    fn decompressed_cache_hit() {
        let cache = ChunkCache::new();
        cache.put_decompressed(vec![0, 0], vec![1, 2, 3, 4]);
        let got = cache.get_decompressed(&[0, 0]).unwrap();
        assert_eq!(got, vec![1, 2, 3, 4]);
    }

    #[test]
    fn lru_eviction_by_slots() {
        let cache = ChunkCache::with_capacity(1024 * 1024, 2); // max 2 slots

        cache.put_decompressed(vec![0], vec![1; 10]);
        cache.put_decompressed(vec![1], vec![2; 10]);
        assert_eq!(cache.cached_chunk_count(), 2);

        // Access slot 0 to make it more recent
        cache.get_decompressed(&[0]);

        // Insert slot 2 — should evict slot 1 (LRU)
        cache.put_decompressed(vec![2], vec![3; 10]);
        assert_eq!(cache.cached_chunk_count(), 2);

        assert!(cache.get_decompressed(&[0]).is_some());
        assert!(cache.get_decompressed(&[1]).is_none()); // evicted
        assert!(cache.get_decompressed(&[2]).is_some());
    }

    #[test]
    fn lru_eviction_by_bytes() {
        let cache = ChunkCache::with_capacity(50, 100); // 50 bytes max

        cache.put_decompressed(vec![0], vec![0; 20]);
        cache.put_decompressed(vec![1], vec![0; 20]);
        assert_eq!(cache.cached_bytes(), 40);

        // This needs 20 bytes but only 10 free — evict LRU
        cache.put_decompressed(vec![2], vec![0; 20]);
        assert!(cache.cached_bytes() <= 50);
        assert!(cache.get_decompressed(&[0]).is_none()); // evicted (LRU)
    }

    #[test]
    fn oversized_chunk_not_cached() {
        let cache = ChunkCache::with_capacity(10, 16);
        cache.put_decompressed(vec![0], vec![0; 100]); // too big
        assert_eq!(cache.cached_chunk_count(), 0);
    }

    #[test]
    fn disabled_cache_retains_no_index_or_chunks() {
        let cache = ChunkCache::with_config(ChunkCacheConfig::disabled());
        let chunks = vec![make_chunk(vec![0, 0], 0x1000, 80)];
        cache.populate_index(&chunks, 1);
        assert!(!cache.has_index());

        cache.put_decompressed(vec![0], vec![1, 2, 3]);
        assert_eq!(cache.cached_chunk_count(), 0);
        assert_eq!(cache.cached_bytes(), 0);
    }

    #[test]
    fn clear_resets_everything() {
        let cache = ChunkCache::new();
        let chunks = vec![make_chunk(vec![0, 0], 0x1000, 80)];
        cache.populate_index(&chunks, 1);
        cache.put_decompressed(vec![0], vec![1, 2, 3]);

        cache.clear();
        assert!(!cache.has_index());
        assert_eq!(cache.cached_chunk_count(), 0);
        assert_eq!(cache.cached_bytes(), 0);
    }

    #[test]
    fn duplicate_insert_is_noop() {
        let cache = ChunkCache::new();
        cache.put_decompressed(vec![0], vec![1, 2, 3]);
        cache.put_decompressed(vec![0], vec![1, 2, 3]); // duplicate
        assert_eq!(cache.cached_chunk_count(), 1);
        assert_eq!(cache.cached_bytes(), 3);
    }

    // --- CacheAlignedBuffer tests ---

    #[test]
    fn aligned_buffer_basic() {
        let buf = CacheAlignedBuffer::zeroed(256);
        assert_eq!(buf.len(), 256);
        assert!(buf.is_aligned());
        assert_eq!(&buf[..4], &[0, 0, 0, 0]);
    }

    #[test]
    fn aligned_buffer_from_slice() {
        let data = vec![1u8, 2, 3, 4, 5];
        let buf = CacheAlignedBuffer::from_slice(&data);
        assert_eq!(buf.len(), 5);
        assert!(buf.is_aligned());
        assert_eq!(buf.to_vec(), data);
    }

    #[test]
    fn aligned_buffer_from_vec() {
        let data = vec![42u8; 1024];
        let buf = CacheAlignedBuffer::from_vec(data.clone());
        assert!(buf.is_aligned());
        assert_eq!(buf.to_vec(), data);
    }

    #[test]
    fn aligned_buffer_empty() {
        let buf = CacheAlignedBuffer::zeroed(0);
        assert!(buf.is_empty());
        assert!(buf.is_aligned());
        assert_eq!(buf.to_vec(), Vec::<u8>::new());
    }

    #[test]
    fn aligned_buffer_clone_is_aligned() {
        let buf = CacheAlignedBuffer::from_slice(&[1, 2, 3, 4]);
        let cloned = buf.clone();
        assert!(cloned.is_aligned());
        assert_eq!(buf.to_vec(), cloned.to_vec());
    }

    #[test]
    fn aligned_buffer_deref_works() {
        let buf = CacheAlignedBuffer::from_slice(&[10, 20, 30]);
        assert_eq!(buf[0], 10);
        assert_eq!(buf[1], 20);
        assert_eq!(buf[2], 30);
    }

    #[test]
    fn aligned_buffer_various_sizes() {
        // Test alignment for various sizes including non-power-of-two
        for size in [1, 7, 63, 64, 65, 127, 128, 129, 255, 256, 1000, 4096] {
            let buf = CacheAlignedBuffer::zeroed(size);
            assert!(buf.is_aligned(), "not aligned for size {size}");
            assert_eq!(buf.len(), size);
        }
    }

    #[test]
    fn cached_data_is_aligned() {
        let cache = ChunkCache::new();
        cache.put_decompressed(vec![0, 0], vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let aligned = cache.get_decompressed_aligned(&[0, 0]).unwrap();
        assert!(aligned.is_aligned());
        assert_eq!(aligned.to_vec(), vec![1, 2, 3, 4, 5, 6, 7, 8]);
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
