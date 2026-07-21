//! Random-access byte sources for the reader: the [`Source`] trait and its
//! backends.
//!
//! # Why this exists
//!
//! Today the reader holds the **entire file** in one `Vec<u8>` ([`crate::File`])
//! and threads a `&[u8]` of that whole buffer through every parser, indexing it
//! by absolute offset. That is simple and fast, but it has a hard ceiling: a
//! file larger than the process address space cannot be loaded at all. On a
//! 32-bit host (`usize` is 32 bits, ~4 GiB of usable address space) a 20 GiB
//! HDF5 file produced on a 64-bit machine simply cannot be `read()` into a
//! `Vec`, no matter how carefully offsets are converted (see [`crate::convert`],
//! which makes the *narrowing* safe but cannot conjure address space). This is
//! the core of issue #27.
//!
//! HDF5 metadata (superblock, object headers, B-trees, heaps) is tiny relative
//! to the dataset payload, and the format is designed for random access by
//! absolute file offset. So the durable fix is to read **on demand** from a
//! seekable source instead of materializing the whole file: keep only a small
//! working set (the metadata being parsed, plus the data chunks currently being
//! decompressed) resident at any time.
//!
//! [`Source`] is that abstraction. It is deliberately minimal and
//! `no_std`/`alloc`-friendly (the trait and the in-memory backends need no
//! `std`), so it works on the same constrained targets the rest of the crate
//! supports.
//!
//! # Backends
//!
//! - [`BytesSource`] — wraps any owned-or-borrowed byte buffer (`Vec<u8>`,
//!   `&[u8]`, `Box<[u8]>`, `Arc<[u8]>`, …). This is the in-memory model the
//!   current [`crate::File`] uses; it is always available, including on WASM and
//!   `no_std`.
//! - [`ReadSeekSource`] (`std` only) — wraps any `Read + Seek` (a
//!   [`std::fs::File`], a `Cursor`, etc.) and reads bytes lazily via
//!   `seek` + `read`. This is the backend that lets a 32-bit host read a file
//!   far larger than its address space, because it never holds more than the
//!   bytes a single `read_at` requests.
//!
//! A windowed `mmap` backend (an optional, `std`-plus-OS feature pulling a crate
//! like `memmap2`) is a natural future addition behind this same trait. Note
//! that a *whole-file* mmap does **not** solve the 32-bit problem — mapping
//! 20 GiB still needs 20 GiB of virtual address space — so only a *windowed*
//! mmap (map/unmap sub-ranges) or plain `Read + Seek` works there. It is left
//! out for now rather than adding a dependency speculatively.
//!
//! # Migration plan (this is the first increment)
//!
//! The reader is not yet ported onto `Source`; that is a staged effort
//! tracked by issue #27. This module is the foundation the later stages build
//! on. The intended path, smallest-risk first:
//!
//! 1. **Foundation (this commit).** Land the trait + in-memory and `Read+Seek`
//!    backends + tests. Nothing in the existing reader changes, so there is no
//!    risk to the current in-memory path.
//! 2. **A cursor.** Introduce a small `Cursor<'a>` over a `&'a dyn Source`
//!    that offers the `read_offset` / `read_length` / "give me bytes at
//!    `[off, off+len)`" idioms the parsers already use, with the checked
//!    [`crate::convert`] conversions built in. The ~15 duplicated per-module
//!    `read_offset` helpers collapse into it.
//! 3. **Bulk path first.** Port the contiguous and chunked **data** readers
//!    (`data_read`, `chunked_read`, `parallel_read`) to fetch each chunk via
//!    [`Source::read_at`] instead of slicing the whole-file buffer. This is
//!    self-contained (a chunk is already `{address, size}`) and captures most of
//!    the memory win, since the data payload is what is actually large. The
//!    zero-copy `&'a [u8]` return of `data_read::read_raw_data_zerocopy` becomes
//!    an owned `Vec<u8>` / `Cow` here, since a window may be evicted.
//! 4. **Metadata parsers.** Migrate the remaining ~56 functions that take a
//!    whole-file `&[u8]` to borrow the cursor, reading each bounded structure
//!    into a small buffer on demand.
//! 5. **Entry point.** Add `File::open_streaming` / `File::from_source` that
//!    construct a [`crate::File`] backed by a [`ReadSeekSource`], plus SWMR
//!    `refresh` over a live streaming handle (the consistent-snapshot semantics
//!    need care over a source that is being appended to).
//!
//! Until step 5 lands, opening a file still buffers it; this module is the
//! building block that makes the staged migration possible without a single
//! risky rewrite.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::convert::TryToUsize;
use crate::error::FormatError;

/// Default maximum size of one entry admitted to a streaming metadata cache.
pub const DEFAULT_METADATA_CACHE_MAX_ENTRY_BYTES: usize = 64 * 1024;

/// Initial metadata-cache settings for streaming file access.
///
/// This is the `hdf5-pure` counterpart to the memory-budget portion of HDF5's
/// `H5Pset_mdc_config`: it bounds the bytes retained for parsed metadata reads
/// while a file is opened through [`crate::File::open_streaming_with_options`].
/// Raw dataset payload reads use `Source::read_exact_at` and are not
/// admitted to this cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetadataCacheConfig {
    max_bytes: usize,
    max_entry_bytes: usize,
}

impl MetadataCacheConfig {
    /// Create a metadata cache with the given total byte budget.
    ///
    /// Individual cached reads are capped at
    /// `DEFAULT_METADATA_CACHE_MAX_ENTRY_BYTES` (64 KiB) by default so one large
    /// heap or index block cannot monopolize the cache. Use
    /// [`with_max_entry_bytes`](Self::with_max_entry_bytes) to change that.
    pub const fn new(max_bytes: usize) -> Self {
        let max_entry_bytes = if max_bytes < DEFAULT_METADATA_CACHE_MAX_ENTRY_BYTES {
            max_bytes
        } else {
            DEFAULT_METADATA_CACHE_MAX_ENTRY_BYTES
        };
        Self {
            max_bytes,
            max_entry_bytes,
        }
    }

    /// Disable metadata read caching.
    pub const fn disabled() -> Self {
        Self {
            max_bytes: 0,
            max_entry_bytes: 0,
        }
    }

    /// Set the maximum size of a single metadata read admitted to the cache.
    pub const fn with_max_entry_bytes(mut self, max_entry_bytes: usize) -> Self {
        self.max_entry_bytes = max_entry_bytes;
        self
    }

    /// Return the total metadata-cache byte budget.
    pub const fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Return the maximum size of one cached metadata entry.
    pub const fn max_entry_bytes(&self) -> usize {
        self.max_entry_bytes
    }

    /// Whether metadata read caching is enabled.
    pub const fn is_enabled(&self) -> bool {
        self.max_bytes > 0 && self.max_entry_bytes > 0
    }
}

impl Default for MetadataCacheConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

/// A random-access, read-only source of the bytes of an HDF5 file.
///
/// Offsets are `u64` (HDF5's native address width); lengths of individual reads
/// are `usize` (they must fit in a caller-provided buffer). Implementations must
/// either fill the whole request or return an error — a short read is always an
/// error, never silently truncated.
pub trait Source {
    /// Total number of bytes the source can supply.
    fn len(&self) -> u64;

    /// Whether the source is empty (zero bytes).
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read exactly `buf.len()` bytes starting at absolute offset `offset`,
    /// filling `buf`.
    ///
    /// Returns [`FormatError::UnexpectedEof`] if fewer than `buf.len()` bytes are
    /// available at `offset`, [`FormatError::OffsetOverflow`] if
    /// `offset + buf.len()` overflows, [`FormatError::ValueTooLargeForPlatform`]
    /// if `offset` does not fit this platform's `usize` (for in-memory
    /// backends), or [`FormatError::Source`] for a backend I/O failure.
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError>;

    /// Read `len` bytes starting at `offset` into a freshly allocated `Vec`.
    ///
    /// Convenience wrapper over [`read_at`](Source::read_at) for callers that
    /// want an owned buffer; the lazy backends keep no more than this resident.
    ///
    /// The request is bounds-checked against [`len`](Source::len) *before* the
    /// buffer is allocated. The metadata parsers feed `len` values straight from
    /// the file (a chunk-0 body size, a continuation-block length, a heap object
    /// size), so a malformed file could otherwise name a multi-gigabyte length
    /// and make this reserve `vec![0u8; len]` up front only for the read to fail
    /// EOF anyway — a cheap denial of service. Rejecting an out-of-range request
    /// before allocating avoids that; the error returned is identical to the one
    /// the underlying [`read_at`](Source::read_at) would have produced.
    fn read_exact_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        let end = offset
            .checked_add(len as u64)
            .ok_or(FormatError::OffsetOverflow {
                offset,
                length: len as u64,
            })?;
        if end > self.len() {
            return Err(FormatError::UnexpectedEof {
                expected: end.to_usize().unwrap_or(usize::MAX),
                available: self.len().to_usize().unwrap_or(usize::MAX),
            });
        }
        let mut buf = vec![0u8; len];
        self.read_at(offset, &mut buf)?;
        Ok(buf)
    }

    /// Read metadata bytes, allowing source implementations to apply a bounded
    /// metadata cache.
    ///
    /// The default implementation performs an uncached exact read. Raw dataset
    /// payload readers intentionally call [`read_exact_at`](Self::read_exact_at)
    /// instead, so a metadata cache does not retain user data chunks.
    fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        self.read_exact_at(offset, len)
    }
}

// Forward `Source` through references and boxes so `&S`, `&dyn Source`,
// and `Box<dyn Source>` are all usable wherever an `S: Source` is.
impl<S: Source + ?Sized> Source for &S {
    fn len(&self) -> u64 {
        (**self).len()
    }
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        (**self).read_at(offset, buf)
    }

    fn read_exact_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        (**self).read_exact_at(offset, len)
    }

    fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        (**self).read_metadata_at(offset, len)
    }
}

#[cfg(feature = "std")]
impl<S: Source + ?Sized> Source for std::boxed::Box<S> {
    fn len(&self) -> u64 {
        (**self).len()
    }
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        (**self).read_at(offset, buf)
    }

    fn read_exact_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        (**self).read_exact_at(offset, len)
    }

    fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        (**self).read_metadata_at(offset, len)
    }
}

// ---------------------------------------------------------------------------
// In-memory backend
// ---------------------------------------------------------------------------

/// A [`Source`] over an in-memory byte buffer: anything that is
/// `AsRef<[u8]>` (`Vec<u8>`, `&[u8]`, `Box<[u8]>`, `Arc<[u8]>`, …).
///
/// This is the always-available backend that mirrors the crate's current
/// in-memory model, usable on WASM and `no_std`.
#[derive(Debug, Clone, Copy)]
pub struct BytesSource<T>(pub T);

impl<T: AsRef<[u8]>> BytesSource<T> {
    /// Wrap an in-memory byte buffer.
    pub fn new(bytes: T) -> Self {
        BytesSource(bytes)
    }
}

impl<T: AsRef<[u8]>> Source for BytesSource<T> {
    fn len(&self) -> u64 {
        self.0.as_ref().len() as u64
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        let bytes = self.0.as_ref();
        let start = offset.to_usize()?;
        let end = start
            .checked_add(buf.len())
            .ok_or(FormatError::OffsetOverflow {
                offset,
                length: buf.len() as u64,
            })?;
        if end > bytes.len() {
            return Err(FormatError::UnexpectedEof {
                expected: end,
                available: bytes.len(),
            });
        }
        buf.copy_from_slice(&bytes[start..end]);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Metadata-caching wrapper (std)
// ---------------------------------------------------------------------------

#[cfg(feature = "std")]
struct CachedMetadataRead {
    offset: u64,
    len: usize,
    bytes: Vec<u8>,
    last_access: u64,
}

/// The bounded LRU store behind [`MetadataCachingSource`], also embedded
/// directly by the bounded read-write backend (`crate::bounded`), which must
/// invalidate entries that overlap an in-place write.
#[cfg(feature = "std")]
pub(crate) struct MetadataReadCache {
    entries: Vec<CachedMetadataRead>,
    current_bytes: usize,
    tick: u64,
}

#[cfg(feature = "std")]
impl MetadataReadCache {
    pub(crate) fn new() -> Self {
        Self {
            entries: Vec::new(),
            current_bytes: 0,
            tick: 0,
        }
    }

    /// Drop every cached entry that overlaps `[offset, offset + len)`, so a
    /// read after an in-place write never observes stale bytes.
    pub(crate) fn invalidate_overlapping(&mut self, offset: u64, len: usize) {
        if len == 0 {
            return;
        }
        let end = offset.saturating_add(len as u64);
        let mut removed = 0usize;
        self.entries.retain(|entry| {
            let entry_end = entry.offset.saturating_add(entry.len as u64);
            let overlaps = entry.offset < end && offset < entry_end;
            if overlaps {
                removed += entry.bytes.len();
            }
            !overlaps
        });
        self.current_bytes -= removed;
    }

    pub(crate) fn get(&mut self, offset: u64, len: usize) -> Option<Vec<u8>> {
        self.tick = self.tick.wrapping_add(1);
        let tick = self.tick;
        for entry in &mut self.entries {
            if entry.offset == offset && entry.len == len {
                entry.last_access = tick;
                return Some(entry.bytes.clone());
            }
        }
        None
    }

    pub(crate) fn insert(&mut self, offset: u64, len: usize, bytes: Vec<u8>, max_bytes: usize) {
        if len == 0 || bytes.len() > max_bytes {
            return;
        }

        self.tick = self.tick.wrapping_add(1);
        let tick = self.tick;

        for entry in &mut self.entries {
            if entry.offset == offset && entry.len == len {
                self.current_bytes = self.current_bytes - entry.bytes.len() + bytes.len();
                entry.bytes = bytes;
                entry.last_access = tick;
                self.evict_to_budget(max_bytes);
                return;
            }
        }

        self.current_bytes += bytes.len();
        self.entries.push(CachedMetadataRead {
            offset,
            len,
            bytes,
            last_access: tick,
        });
        self.evict_to_budget(max_bytes);
    }

    fn evict_to_budget(&mut self, max_bytes: usize) {
        while self.current_bytes > max_bytes && !self.entries.is_empty() {
            let lru_idx = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.last_access)
                .map(|(idx, _)| idx)
                .unwrap();
            let removed = self.entries.swap_remove(lru_idx);
            self.current_bytes -= removed.bytes.len();
        }
    }
}

/// A [`Source`] wrapper with a bounded cache for metadata reads.
///
/// The wrapper only caches calls to [`Source::read_metadata_at`]. Plain
/// [`Source::read_exact_at`] calls still go directly to the inner source,
/// which keeps raw dataset payloads out of the metadata cache.
#[cfg(feature = "std")]
pub struct MetadataCachingSource<S> {
    inner: S,
    config: MetadataCacheConfig,
    cache: std::sync::Mutex<MetadataReadCache>,
}

#[cfg(feature = "std")]
impl<S> MetadataCachingSource<S> {
    /// Wrap a source with the supplied metadata-cache configuration.
    pub fn new(inner: S, config: MetadataCacheConfig) -> Self {
        Self {
            inner,
            config,
            cache: std::sync::Mutex::new(MetadataReadCache::new()),
        }
    }
}

#[cfg(feature = "std")]
impl<S: Source> Source for MetadataCachingSource<S> {
    fn len(&self) -> u64 {
        self.inner.len()
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        self.inner.read_at(offset, buf)
    }

    fn read_exact_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        self.inner.read_exact_at(offset, len)
    }

    fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        if !self.config.is_enabled()
            || len == 0
            || len > self.config.max_entry_bytes
            || len > self.config.max_bytes
        {
            return self.inner.read_metadata_at(offset, len);
        }

        if let Some(bytes) = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(offset, len)
        {
            return Ok(bytes);
        }

        let bytes = self.inner.read_metadata_at(offset, len)?;
        self.cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(offset, len, bytes.clone(), self.config.max_bytes);
        Ok(bytes)
    }
}

// ---------------------------------------------------------------------------
// Read + Seek backend (std)
// ---------------------------------------------------------------------------

/// A lazy [`Source`] over any [`std::io::Read`] + [`std::io::Seek`] (a
/// [`std::fs::File`], an in-memory `Cursor`, etc.).
///
/// Each [`read_at`](Source::read_at) performs a `seek` + `read_exact`, so no
/// more than the requested bytes are ever held in memory. This is the backend
/// that lets a 32-bit host read a file larger than its address space: the
/// metadata and one working chunk fit even when the whole file does not.
///
/// The reader is wrapped in a [`std::sync::Mutex`] so the source is `Sync` and
/// `read_at` can take `&self` (seeking needs `&mut` access). This serializes
/// concurrent reads, which is correct though not maximally parallel; a future
/// backend can use positioned reads (`pread`/`seek_read`) to avoid the lock.
#[cfg(feature = "std")]
pub struct ReadSeekSource<R> {
    inner: std::sync::Mutex<R>,
    len: u64,
}

#[cfg(feature = "std")]
impl<R: std::io::Read + std::io::Seek> ReadSeekSource<R> {
    /// Wrap a `Read + Seek`, measuring its length by seeking to the end (then
    /// restoring nothing — every `read_at` seeks absolutely anyway).
    pub fn new(mut reader: R) -> Result<Self, FormatError> {
        let len = reader
            .seek(std::io::SeekFrom::End(0))
            .map_err(|e| FormatError::Source(format_io(&e)))?;
        Ok(ReadSeekSource {
            inner: std::sync::Mutex::new(reader),
            len,
        })
    }
}

#[cfg(feature = "std")]
impl<R: std::io::Read + std::io::Seek> Source for ReadSeekSource<R> {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        // Bound-check up front so a request past EOF is a clean error rather
        // than a backend-specific short read.
        let end = offset
            .checked_add(buf.len() as u64)
            .ok_or(FormatError::OffsetOverflow {
                offset,
                length: buf.len() as u64,
            })?;
        if end > self.len {
            return Err(FormatError::UnexpectedEof {
                // `expected`/`available` are byte counts; report them as the
                // best `usize` we can without truncating on a 32-bit host.
                expected: end.to_usize().unwrap_or(usize::MAX),
                available: self.len.to_usize().unwrap_or(usize::MAX),
            });
        }
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard
            .seek(std::io::SeekFrom::Start(offset))
            .map_err(|e| FormatError::Source(format_io(&e)))?;
        guard
            .read_exact(buf)
            .map_err(|e| FormatError::Source(format_io(&e)))?;
        Ok(())
    }
}

/// Render an `std::io::Error` to a short owned string for [`FormatError::Source`]
/// (which is `no_std`-friendly and cannot hold the error itself).
#[cfg(feature = "std")]
fn format_io(e: &std::io::Error) -> std::string::String {
    std::format!("{e}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "std"))]
    use alloc::vec;

    #[test]
    fn bytes_source_reads_and_reports_len() {
        let data = (0u8..=255).collect::<Vec<u8>>();
        let src = BytesSource::new(data.clone());
        assert_eq!(src.len(), 256);
        assert!(!src.is_empty());

        let mut buf = [0u8; 4];
        src.read_at(10, &mut buf).unwrap();
        assert_eq!(buf, [10, 11, 12, 13]);

        let owned = src.read_exact_at(250, 6).unwrap();
        assert_eq!(owned, vec![250, 251, 252, 253, 254, 255]);
    }

    #[test]
    fn bytes_source_short_read_is_eof() {
        let src = BytesSource::new(vec![1u8, 2, 3]);
        let mut buf = [0u8; 4];
        let err = src.read_at(0, &mut buf).unwrap_err();
        assert!(matches!(err, FormatError::UnexpectedEof { .. }));
        // Reading exactly to the end is fine.
        let mut ok = [0u8; 3];
        src.read_at(0, &mut ok).unwrap();
        assert_eq!(ok, [1, 2, 3]);
    }

    #[test]
    fn bytes_source_offset_past_end_is_eof() {
        let src = BytesSource::new(vec![0u8; 8]);
        let mut buf = [0u8; 1];
        assert!(matches!(
            src.read_at(8, &mut buf).unwrap_err(),
            FormatError::UnexpectedEof { .. }
        ));
        // Zero-length read at EOF succeeds.
        src.read_at(8, &mut []).unwrap();
    }

    #[test]
    fn read_exact_at_rejects_oversized_len_without_allocating() {
        // A length far larger than the source must error cleanly rather than
        // attempt to reserve the buffer first. Before the pre-allocation bounds
        // check, this called `vec![0u8; usize::MAX]` and aborted the process.
        let src = BytesSource::new(vec![1u8, 2, 3, 4]);
        assert!(matches!(
            src.read_exact_at(0, usize::MAX).unwrap_err(),
            FormatError::UnexpectedEof { .. }
        ));
        // A read that fits is unaffected.
        assert_eq!(src.read_exact_at(1, 3).unwrap(), vec![2, 3, 4]);
    }

    #[test]
    fn empty_source() {
        let src = BytesSource::new(Vec::<u8>::new());
        assert_eq!(src.len(), 0);
        assert!(src.is_empty());
    }

    #[test]
    fn forwarding_through_reference() {
        let src = BytesSource::new(vec![9u8, 8, 7]);
        let r: &dyn Source = &src;
        let mut buf = [0u8; 2];
        r.read_at(1, &mut buf).unwrap();
        assert_eq!(buf, [8, 7]);
    }

    #[test]
    fn forwarding_through_reference_preserves_metadata_reads() {
        use core::cell::Cell;

        struct MetadataSource {
            metadata_reads: Cell<usize>,
        }

        impl Source for MetadataSource {
            fn len(&self) -> u64 {
                16
            }

            fn read_at(&self, _offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
                buf.fill(0);
                Ok(())
            }

            fn read_metadata_at(&self, _offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
                self.metadata_reads.set(self.metadata_reads.get() + 1);
                Ok(vec![0xAB; len])
            }
        }

        fn read_metadata_via_trait<T: Source>(source: T) -> Vec<u8> {
            source.read_metadata_at(4, 3).unwrap()
        }

        let source = MetadataSource {
            metadata_reads: Cell::new(0),
        };

        assert_eq!(read_metadata_via_trait(&source), vec![0xAB; 3]);
        assert_eq!(source.metadata_reads.get(), 1);
    }

    #[cfg(feature = "std")]
    #[test]
    fn metadata_cache_caches_only_metadata_reads() {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };

        struct CountingSource {
            data: Vec<u8>,
            reads: Arc<AtomicUsize>,
        }

        impl Source for CountingSource {
            fn len(&self) -> u64 {
                self.data.len() as u64
            }

            fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
                self.reads.fetch_add(1, Ordering::SeqCst);
                BytesSource::new(&self.data).read_at(offset, buf)
            }
        }

        let reads = Arc::new(AtomicUsize::new(0));
        let source = MetadataCachingSource::new(
            CountingSource {
                data: (0u8..16).collect(),
                reads: Arc::clone(&reads),
            },
            MetadataCacheConfig::new(16),
        );

        assert_eq!(source.read_metadata_at(4, 4).unwrap(), vec![4, 5, 6, 7]);
        assert_eq!(source.read_metadata_at(4, 4).unwrap(), vec![4, 5, 6, 7]);
        assert_eq!(reads.load(Ordering::SeqCst), 1);

        assert_eq!(source.read_exact_at(4, 4).unwrap(), vec![4, 5, 6, 7]);
        assert_eq!(source.read_exact_at(4, 4).unwrap(), vec![4, 5, 6, 7]);
        assert_eq!(reads.load(Ordering::SeqCst), 3);
    }

    #[cfg(feature = "std")]
    #[test]
    fn read_seek_source_matches_in_memory() {
        use std::io::Cursor;
        let data = (0u8..200).collect::<Vec<u8>>();
        let mem = BytesSource::new(data.clone());
        let seek = ReadSeekSource::new(Cursor::new(data.clone())).unwrap();
        assert_eq!(seek.len(), mem.len());

        // Every read_at against the lazy source matches the in-memory source.
        for &(off, len) in &[(0u64, 1usize), (5, 10), (199, 1), (100, 50)] {
            let a = mem.read_exact_at(off, len).unwrap();
            let b = seek.read_exact_at(off, len).unwrap();
            assert_eq!(a, b, "mismatch at offset {off} len {len}");
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn read_seek_source_past_end_is_error() {
        use std::io::Cursor;
        let seek = ReadSeekSource::new(Cursor::new(vec![1u8, 2, 3, 4])).unwrap();
        let mut buf = [0u8; 3];
        assert!(matches!(
            seek.read_at(2, &mut buf).unwrap_err(),
            FormatError::UnexpectedEof { .. }
        ));
    }

    #[cfg(feature = "std")]
    #[test]
    fn read_seek_source_is_sync() {
        // Compile-time assertion that the std backend is Send + Sync so it can
        // back a parallel reader.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ReadSeekSource<std::io::Cursor<Vec<u8>>>>();
    }
}
