//! The byte image a mutating session edits: the [`FileImage`] trait and its
//! whole-file mirror backend.
//!
//! # Why this exists
//!
//! [`Source`] describes how the *reader* gets bytes out of a file. This is its
//! write-side counterpart: what a [`WriteEngine`](crate::edit::WriteEngine)
//! needs of the bytes it is editing — the same random-access reads, plus the
//! four primitives a commit performs (append at end-of-file, overwrite in
//! place, truncate, and force durability).
//!
//! Separating it from the engine is what lets one engine drive two very
//! different backings (issue #198). Historically `File::open_rw` held the whole
//! file in a `Vec<u8>` mirror and carried the full staged-edit vocabulary,
//! while the bounded read-write open held no mirror and could therefore only read
//! and append. *That* limitation is a property of where the bytes live, not of
//! what edits are expressible, so it belongs behind this trait rather than in
//! two engines.
//!
//! The trait abstracts residency and nothing else. Which files each backing can
//! open at all still differs, the bounded one refusing a v0/v1 superblock, 4-byte
//! offsets, and any userblock, but that is no longer a difference between two
//! entry points: `File::open_rw` picks a backing per file and mirrors what the
//! bounded one turns down (issue #198, step 4).
//!
//! # Reads
//!
//! [`FileImage`] extends [`Source`], so every parser the edit engine drives
//! reads through the same interface it already uses. [`as_slice`](FileImage::as_slice)
//! is the one concession to backing: a mirror can hand out its whole buffer,
//! letting a slice-walking parser borrow rather than copy, and a file-backed
//! image cannot. Callers that can exploit a slice should, and must have a
//! `Source` path for when there is none.
//!
//! # The end-of-file cursor
//!
//! [`Source::len`] is the image's end-of-file: [`append`](FileImage::append)
//! places bytes there and extends it by exactly their length. For the mirror
//! that is the buffer length; for a file-backed image it is a counter seeded
//! from the file's real length at open.
//!
//! Callers depend on the exact-extension part, not just on monotonicity: the
//! engine reads `len()`, builds a blob whose interior addresses assume it will
//! land at that address, and only then appends. An implementation that inserted
//! padding inside `append` would silently relocate every such blob, so padding
//! belongs in a layer above this one.

use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};

use crate::convert::TryToUsize;
use crate::error::{Error, FormatError};
use crate::source::{BytesSource, MetadataCacheConfig, MetadataReadCache, Source};

/// The file bytes a mutating session works on.
///
/// Implementors keep the image and the file on disk consistent, and are free to
/// choose the order in which they update them; see [`MirrorImage`] for why the
/// mirror writes to disk first.
pub(crate) trait FileImage: Source + Send + Sync {
    /// Append `bytes` at end-of-file, returning the absolute address they were
    /// written at — which is the pre-call [`Source::len`]. Extends `len` by
    /// exactly `bytes.len()`; see the module docs for why callers rely on that.
    ///
    /// An implementation that caches reads must invalidate any entry covering
    /// the appended range: a preceding [`truncate`](Self::truncate) can make an
    /// address readable, then cached, then appended over. An implementation that
    /// also refuses reads past end-of-file satisfies this through its `truncate`
    /// alone, since nothing above the original end could have been cached; the
    /// obligation is stated here because it belongs to the contract rather than
    /// to that policy.
    fn append(&mut self, bytes: &[u8]) -> Result<u64, Error>;

    /// Overwrite `[offset, offset + bytes.len())` in place. The range must
    /// already exist; the engine computes it from its own allocation, so a
    /// range past end-of-file is a bug rather than a bad file, and
    /// implementations may assert it.
    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error>;

    /// Shrink the image to `len` bytes, physically shortening the file. `len`
    /// must not exceed the current [`Source::len`]: this cannot grow an image,
    /// and implementations may assert that rather than define a growth
    /// semantics no caller wants.
    ///
    /// The same cache-invalidation obligation as [`append`](Self::append)
    /// applies to the discarded range.
    fn truncate(&mut self, len: u64) -> Result<(), Error>;

    /// Flush buffered writes and force the file's *data* to durable storage.
    fn sync_data(&mut self) -> Result<(), Error>;

    /// Flush buffered writes and force the file's data **and metadata** to
    /// durable storage. Distinct from [`sync_data`](Self::sync_data) because a
    /// commit changes the file's length, which lives in that metadata.
    ///
    /// The distinction is not observable in this crate's test suite, which
    /// simulates a crash by copying the file rather than by losing the page
    /// cache. Weakening a call site from `sync_all` to `sync_data` would pass
    /// every test and lose a committed file's length on power loss, so the
    /// choice at each call site has to be preserved by inspection.
    fn sync_all(&mut self) -> Result<(), Error>;

    /// The whole image as one slice, for parsers that walk bytes directly.
    ///
    /// `Some` only for a backing that already holds the file in memory. A
    /// caller must always have a [`Source`] path for the `None` case; this is a
    /// fast path, not a capability check.
    fn as_slice(&self) -> Option<&[u8]> {
        None
    }
}

/// A whole-file in-memory mirror plus the read/write handle it mirrors, kept
/// byte-for-byte in sync. This is the backing
/// [`File::open_rw`](crate::File::open_rw) falls back to for a file the bounded
/// engine cannot edit, and the one
/// [`MemoryStrategy::Mirrored`](crate::MemoryStrategy) always takes: reads are
/// slice accesses and never touch the disk, at the cost of holding the entire
/// file resident.
///
/// Every mutation writes to disk *before* updating the mirror, so a failed
/// write can leave the mirror behind the file but never ahead of it. That
/// direction is the safe one: the session re-reads its own mirror to plan
/// later edits, and planning against bytes that are not yet on disk would
/// commit a structure pointing at content that does not exist.
pub(crate) struct MirrorImage {
    handle: fs::File,
    data: Vec<u8>,
}

impl MirrorImage {
    /// Wrap an open read/write `handle` and the `data` already read from it.
    /// The caller is responsible for the two agreeing.
    pub(crate) fn new(handle: fs::File, data: Vec<u8>) -> Self {
        Self { handle, data }
    }
}

impl Source for MirrorImage {
    fn len(&self) -> u64 {
        self.data.len() as u64
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        BytesSource::new(&self.data[..]).read_at(offset, buf)
    }
}

impl FileImage for MirrorImage {
    fn append(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        let addr = self.data.len() as u64;
        self.handle.seek(SeekFrom::Start(addr)).map_err(Error::Io)?;
        self.handle.write_all(bytes).map_err(Error::Io)?;
        self.data.extend_from_slice(bytes);
        Ok(addr)
    }

    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
        debug_assert!(
            offset.saturating_add(bytes.len() as u64) <= self.len(),
            "write_at past end-of-file: {offset}+{} > {}",
            bytes.len(),
            self.len()
        );
        // Convert before touching the file so a failure leaves both sides
        // untouched rather than the mirror behind the disk.
        let offset_usize = offset.to_usize()?;
        self.handle
            .seek(SeekFrom::Start(offset))
            .map_err(Error::Io)?;
        self.handle.write_all(bytes).map_err(Error::Io)?;
        self.data[offset_usize..offset_usize + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    fn truncate(&mut self, len: u64) -> Result<(), Error> {
        debug_assert!(
            len <= self.len(),
            "truncate would grow the image: {len} > {}",
            self.len()
        );
        // `set_len` grows a file where `Vec::truncate` no-ops, and a failed
        // conversion after `set_len` would leave the mirror longer than the
        // file. Convert first so neither side moves unless both can.
        let len_usize = len.to_usize()?;
        self.handle.set_len(len).map_err(Error::Io)?;
        self.data.truncate(len_usize);
        Ok(())
    }

    fn sync_data(&mut self) -> Result<(), Error> {
        self.handle.flush().map_err(Error::Io)?;
        self.handle.sync_data().map_err(Error::Io)?;
        Ok(())
    }

    fn sync_all(&mut self) -> Result<(), Error> {
        // `Write::flush` is a no-op for `fs::File`, so this matches the bare
        // `sync_all` it replaced; it is here so the trait's documented "flush
        // buffered writes" holds for an implementation that does buffer.
        self.handle.flush().map_err(Error::Io)?;
        self.handle.sync_all().map_err(Error::Io)?;
        Ok(())
    }

    fn as_slice(&self) -> Option<&[u8]> {
        Some(&self.data)
    }
}

/// Read exactly `buf.len()` bytes at `offset` from a shared file handle,
/// bounds-checked against `len` (mirroring `ReadSeekSource`). Uses the
/// `Read`/`Seek` impls on `&fs::File`, so it can serve a `&self` read: callers
/// serialize access through the session's engine lock, and the shared cursor is
/// never raced.
pub(crate) fn read_at_handle(
    handle: &fs::File,
    len: u64,
    offset: u64,
    buf: &mut [u8],
) -> Result<(), FormatError> {
    let end = offset
        .checked_add(buf.len() as u64)
        .ok_or(FormatError::OffsetOverflow {
            offset,
            length: buf.len() as u64,
        })?;
    if end > len {
        return Err(FormatError::UnexpectedEof {
            expected: end.to_usize().unwrap_or(usize::MAX),
            available: len.to_usize().unwrap_or(usize::MAX),
        });
    }
    let mut h = handle;
    h.seek(SeekFrom::Start(offset))
        .map_err(|e| FormatError::Source(std::format!("{e}")))?;
    h.read_exact(buf)
        .map_err(|e| FormatError::Source(std::format!("{e}")))?;
    Ok(())
}

/// A [`Source`] over a *borrowed* open handle, for the reads an open has to make
/// before it decides which image will own that handle.
///
/// It exists so a read-write open can locate and validate the superblock — a few
/// bounded windows — before building an image that might read the whole file.
/// Refusing after that build costs `O(file size)` on a file that is then
/// rejected. It moves the handle's shared cursor, as everything using
/// [`read_at_handle`] does, so a caller that later reads sequentially from the
/// same handle must position it itself ([`MirrorImage`] does).
pub(crate) struct BorrowedHandle<'a> {
    handle: &'a fs::File,
    len: u64,
}

impl<'a> BorrowedHandle<'a> {
    pub(crate) fn new(handle: &'a fs::File, len: u64) -> Self {
        Self { handle, len }
    }
}

impl Source for BorrowedHandle<'_> {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        read_at_handle(self.handle, self.len, offset, buf)
    }
}

/// A file-backed image that holds no whole-file mirror: reads are positioned I/O
/// against the handle, served through a bounded metadata cache when one is
/// configured. This is the backing a bounded read-write open uses,
/// and the reason [`FileImage`] exists — resident memory is the cache budget
/// plus whatever the caller is parsing, independent of the file's size.
///
/// The end-of-file cursor is explicit ([`len`](Source::len)), seeded from the
/// file's real length at open and advanced by [`append`](FileImage::append). It
/// is not re-read from the filesystem, so it stays the authority even if the
/// handle's own cursor moves.
///
/// Every mutation invalidates the cache entries it overlaps *before* the write
/// reaches the disk, so a concurrent-looking read can miss a fresh byte but
/// never return a stale one.
pub(crate) struct HandleImage {
    handle: fs::File,
    /// Logical end-of-file: the real file length at open, moved by `append` and
    /// `truncate` and by nothing else.
    len: u64,
    /// Bounded read cache for metadata-sized reads; `None` when disabled.
    metadata_cache: Option<(MetadataCacheConfig, std::sync::Mutex<MetadataReadCache>)>,
}

impl HandleImage {
    /// Wrap an open read/write `handle` whose current length is `len`, caching
    /// metadata reads under `cache` (see [`MetadataCacheConfig::disabled`] to
    /// opt out).
    pub(crate) fn new(handle: fs::File, len: u64, cache: MetadataCacheConfig) -> Self {
        Self {
            handle,
            len,
            metadata_cache: cache
                .is_enabled()
                .then(|| (cache, std::sync::Mutex::new(MetadataReadCache::new()))),
        }
    }

    /// Drop every cached read overlapping `[offset, offset + len)`. Called before
    /// each mutation; a no-op when caching is off or the range is empty.
    fn invalidate(&self, offset: u64, len: u64) {
        let Some((_, cache)) = &self.metadata_cache else {
            return;
        };
        cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .invalidate_overlapping(offset, len.to_usize().unwrap_or(usize::MAX));
    }
}

impl Source for HandleImage {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        read_at_handle(&self.handle, self.len, offset, buf)
    }

    fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        let Some((config, cache)) = &self.metadata_cache else {
            return self.read_exact_at(offset, len);
        };
        if len == 0 || len > config.max_entry_bytes() || len > config.max_bytes() {
            return self.read_exact_at(offset, len);
        }
        if let Some(bytes) = cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(offset, len)
        {
            return Ok(bytes);
        }
        let bytes = self.read_exact_at(offset, len)?;
        cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(offset, len, bytes.clone(), config.max_bytes());
        Ok(bytes)
    }
}

impl FileImage for HandleImage {
    fn append(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        let addr = self.len;
        // The trait requires this. It is belt-and-braces for *this* image, whose
        // reads past end-of-file are refused: the only way a cached entry can
        // cover an address an append reuses is a preceding `truncate`, and that
        // already invalidated everything it discarded. It is kept because the
        // obligation belongs to the contract, not to this implementation's
        // read-bounds policy.
        self.invalidate(addr, bytes.len() as u64);
        self.handle.seek(SeekFrom::Start(addr)).map_err(Error::Io)?;
        self.handle.write_all(bytes).map_err(Error::Io)?;
        self.len += bytes.len() as u64;
        Ok(addr)
    }

    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
        // The mirror image fails loudly on this in every build — it indexes its
        // buffer — so this one must too rather than extend the file behind the
        // `len` cursor and leave a later `append` overwriting what was written.
        // A caller bug that is a panic on one backing and silent corruption on the
        // other is exactly what the two being interchangeable has to rule out.
        let end = offset
            .checked_add(bytes.len() as u64)
            .filter(|&e| e <= self.len)
            .ok_or(Error::Format(FormatError::UnexpectedEof {
                expected: offset.to_usize().unwrap_or(usize::MAX),
                available: self.len.to_usize().unwrap_or(usize::MAX),
            }))?;
        debug_assert!(end <= self.len);
        self.invalidate(offset, bytes.len() as u64);
        self.handle
            .seek(SeekFrom::Start(offset))
            .map_err(Error::Io)?;
        self.handle.write_all(bytes).map_err(Error::Io)?;
        Ok(())
    }

    fn truncate(&mut self, len: u64) -> Result<(), Error> {
        debug_assert!(
            len <= self.len,
            "truncate would grow the image: {len} > {}",
            self.len
        );
        self.invalidate(len, self.len.saturating_sub(len));
        self.handle.set_len(len).map_err(Error::Io)?;
        self.len = len;
        Ok(())
    }

    fn sync_data(&mut self) -> Result<(), Error> {
        self.handle.flush().map_err(Error::Io)?;
        self.handle.sync_data().map_err(Error::Io)?;
        Ok(())
    }

    fn sync_all(&mut self) -> Result<(), Error> {
        self.handle.flush().map_err(Error::Io)?;
        self.handle.sync_all().map_err(Error::Io)?;
        Ok(())
    }

    // No `as_slice`: withholding the whole-file slice is what this image is for.
}

/// A [`FileImage`] that counts the bytes read through it, so a test can measure
/// how much of a file an operation actually touches.
///
/// It deliberately does *not* forward `read_metadata_at`, taking [`Source`]'s
/// default instead, so every read funnels through this one `read_at` and is
/// counted. Callers build the inner image with the metadata cache disabled, so
/// nothing is bypassed and the count is exact rather than an upper bound.
#[cfg(test)]
pub(crate) struct CountingImage {
    inner: Box<dyn FileImage>,
    read_bytes: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

#[cfg(test)]
impl CountingImage {
    pub(crate) fn new(
        inner: Box<dyn FileImage>,
        read_bytes: std::sync::Arc<std::sync::atomic::AtomicU64>,
    ) -> Self {
        Self { inner, read_bytes }
    }
}

#[cfg(test)]
impl Source for CountingImage {
    fn len(&self) -> u64 {
        self.inner.len()
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        self.read_bytes
            .fetch_add(buf.len() as u64, std::sync::atomic::Ordering::Relaxed);
        self.inner.read_at(offset, buf)
    }
}

#[cfg(test)]
impl FileImage for CountingImage {
    fn append(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        self.inner.append(bytes)
    }

    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
        self.inner.write_at(offset, bytes)
    }

    fn truncate(&mut self, len: u64) -> Result<(), Error> {
        self.inner.truncate(len)
    }

    fn sync_data(&mut self) -> Result<(), Error> {
        self.inner.sync_data()
    }

    fn sync_all(&mut self) -> Result<(), Error> {
        self.inner.sync_all()
    }

    fn as_slice(&self) -> Option<&[u8]> {
        self.inner.as_slice()
    }
}

/// A [`MirrorImage`] that withholds its slice, so every read routed through it
/// takes the [`Source`] path instead of the whole-file fast path.
///
/// This exists to make the mirrorless read paths reachable before a mirrorless
/// backing does. Each read the engine serves has two forms — one walking a
/// borrowed slice, one going through `Source` — and until the bounded read-write open
/// runs on this engine (issue #198), only the first would ever execute. Opening
/// a file through this image runs the same tests down the other form and lets
/// them be compared, which is the only thing that keeps the two from drifting.
#[cfg(test)]
pub(crate) struct SourceOnlyImage(MirrorImage);

#[cfg(test)]
impl SourceOnlyImage {
    pub(crate) fn new(inner: MirrorImage) -> Self {
        Self(inner)
    }
}

#[cfg(test)]
impl Source for SourceOnlyImage {
    fn len(&self) -> u64 {
        self.0.len()
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        self.0.read_at(offset, buf)
    }
}

#[cfg(test)]
impl FileImage for SourceOnlyImage {
    fn append(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        self.0.append(bytes)
    }

    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
        self.0.write_at(offset, bytes)
    }

    fn truncate(&mut self, len: u64) -> Result<(), Error> {
        self.0.truncate(len)
    }

    fn sync_data(&mut self) -> Result<(), Error> {
        self.0.sync_data()
    }

    fn sync_all(&mut self) -> Result<(), Error> {
        self.0.sync_all()
    }

    // Deliberately inherits the `None` default: that is the whole point.
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two backings [`FileImage`] exists to make interchangeable. Every case
    /// below runs against both, because a primitive that holds for one and not
    /// the other is exactly the divergence this seam would otherwise hide.
    #[derive(Clone, Copy, Debug)]
    enum Backing {
        Mirror,
        Handle,
    }

    const BACKINGS: [Backing; 2] = [Backing::Mirror, Backing::Handle];

    /// Open a fresh file holding `initial`, wrapped in the given backing. The
    /// handle image caches metadata reads, so the cache is exercised rather
    /// than configured away.
    fn image(
        dir: &std::path::Path,
        initial: &[u8],
        backing: Backing,
    ) -> (std::path::PathBuf, Box<dyn FileImage>) {
        let path = dir.join(std::format!("{backing:?}.bin"));
        std::fs::write(&path, initial).unwrap();
        let handle = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let img: Box<dyn FileImage> = match backing {
            Backing::Mirror => Box::new(MirrorImage::new(handle, initial.to_vec())),
            Backing::Handle => Box::new(HandleImage::new(
                handle,
                initial.len() as u64,
                MetadataCacheConfig::new(64 * 1024),
            )),
        };
        (path, img)
    }

    /// The whole image read back through [`Source`], which is the interface
    /// every parser above this layer uses.
    fn bytes(img: &dyn FileImage) -> Vec<u8> {
        let mut buf = vec![0u8; img.len().to_usize().unwrap()];
        img.read_at(0, &mut buf).unwrap();
        buf
    }

    /// The invariant the layer exists to hold: after any primitive, what the
    /// image reports and what is on disk agree. A primitive that updates only
    /// one side is the defect that would otherwise surface as a corrupt file
    /// much later.
    fn assert_in_sync(path: &std::path::Path, img: &dyn FileImage, backing: Backing) {
        let on_disk = std::fs::read(path).unwrap();
        assert_eq!(
            img.len(),
            on_disk.len() as u64,
            "{backing:?}: end-of-file disagrees with the file"
        );
        assert_eq!(
            bytes(img),
            on_disk,
            "{backing:?}: reads disagree with the file"
        );
        if let Some(slice) = img.as_slice() {
            assert_eq!(
                slice,
                &on_disk[..],
                "{backing:?}: the slice disagrees with the file"
            );
        }
    }

    #[test]
    fn append_returns_the_pre_append_end_and_extends_by_exactly_the_length() {
        let dir = tempfile::tempdir().unwrap();
        for backing in BACKINGS {
            let (path, mut img) = image(dir.path(), b"abcd", backing);

            let addr = img.append(b"XYZ").unwrap();

            assert_eq!(addr, 4, "{backing:?}: append must report where it wrote");
            assert_eq!(
                img.len(),
                7,
                "{backing:?}: append must extend len by exactly bytes.len()"
            );
            assert_in_sync(&path, img.as_ref(), backing);
        }
    }

    #[test]
    fn write_at_overwrites_both_sides_in_place() {
        let dir = tempfile::tempdir().unwrap();
        for backing in BACKINGS {
            let (path, mut img) = image(dir.path(), b"abcdef", backing);

            img.write_at(2, b"ZZ").unwrap();

            assert_eq!(bytes(img.as_ref()), b"abZZef");
            assert_eq!(
                img.len(),
                6,
                "{backing:?}: an in-place write must not move end-of-file"
            );
            assert_in_sync(&path, img.as_ref(), backing);
        }
    }

    #[test]
    fn truncate_shrinks_both_sides() {
        let dir = tempfile::tempdir().unwrap();
        for backing in BACKINGS {
            let (path, mut img) = image(dir.path(), b"abcdef", backing);

            img.truncate(2).unwrap();

            assert_eq!(bytes(img.as_ref()), b"ab");
            assert_eq!(img.len(), 2, "{backing:?}");
            assert_in_sync(&path, img.as_ref(), backing);
        }
    }

    /// A write following a truncate must land at the shortened end-of-file, not
    /// wherever the previous operation left the handle's cursor.
    #[test]
    fn append_after_truncate_lands_at_the_new_end() {
        let dir = tempfile::tempdir().unwrap();
        for backing in BACKINGS {
            let (path, mut img) = image(dir.path(), b"abcdef", backing);

            img.truncate(3).unwrap();
            let addr = img.append(b"Z").unwrap();

            assert_eq!(addr, 3, "{backing:?}");
            assert_eq!(bytes(img.as_ref()), b"abcZ");
            assert_in_sync(&path, img.as_ref(), backing);
        }
    }

    /// Reads go through `Source`, so they must observe writes immediately —
    /// the engine plans its next edit against bytes it just wrote.
    #[test]
    fn reads_observe_writes_immediately() {
        let dir = tempfile::tempdir().unwrap();
        for backing in BACKINGS {
            let (_path, mut img) = image(dir.path(), b"abcdef", backing);
            img.write_at(0, b"ZY").unwrap();
            img.append(b"!").unwrap();

            let mut buf = [0u8; 3];
            img.read_at(0, &mut buf).unwrap();
            assert_eq!(&buf, b"ZYc", "{backing:?}");
            img.read_at(6, &mut buf[..1]).unwrap();
            assert_eq!(buf[0], b'!', "{backing:?}");
        }
    }

    /// The same, through the *cached* read path: a metadata read taken before a
    /// write must not survive it. This is the handle image's own hazard — the
    /// mirror has no cache to go stale — but it runs on both so the case is
    /// stated once.
    #[test]
    fn cached_reads_observe_writes_immediately() {
        let dir = tempfile::tempdir().unwrap();
        for backing in BACKINGS {
            let (_path, mut img) = image(dir.path(), b"abcdef", backing);

            assert_eq!(img.read_metadata_at(0, 4).unwrap(), b"abcd", "{backing:?}");
            img.write_at(1, b"ZZ").unwrap();

            assert_eq!(
                img.read_metadata_at(0, 4).unwrap(),
                b"aZZd",
                "{backing:?}: a cached read outlived the write that overwrote it"
            );
        }
    }

    /// Truncate makes a range unreadable, a later append reuses those addresses,
    /// and a read cached before the truncate must not resurface.
    ///
    /// What this pins is `truncate`'s invalidation, not `append`'s: for an image
    /// that refuses reads past end-of-file the two overlap, and deleting the call
    /// in `append` leaves this green. See the trait's `append` contract for why
    /// the call stays anyway.
    #[test]
    fn a_cached_read_does_not_survive_being_truncated_and_appended_over() {
        let dir = tempfile::tempdir().unwrap();
        for backing in BACKINGS {
            let (_path, mut img) = image(dir.path(), b"abcdef", backing);

            assert_eq!(img.read_metadata_at(4, 2).unwrap(), b"ef", "{backing:?}");
            img.truncate(4).unwrap();
            img.append(b"ZZ").unwrap();

            assert_eq!(
                img.read_metadata_at(4, 2).unwrap(),
                b"ZZ",
                "{backing:?}: a read cached before the truncate survived the append"
            );
        }
    }

    #[test]
    fn reads_past_end_of_file_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        for backing in BACKINGS {
            let (_path, img) = image(dir.path(), b"abcd", backing);

            let mut buf = [0u8; 2];
            assert!(img.read_at(3, &mut buf).is_err(), "{backing:?}");
        }
    }

    /// Only the mirror lends its buffer out; the handle image withholds it, and
    /// that difference is the one the read paths branch on.
    #[test]
    fn only_the_mirror_offers_a_whole_file_slice() {
        let dir = tempfile::tempdir().unwrap();
        let (_p1, mirror) = image(dir.path(), b"abcdef", Backing::Mirror);
        let (_p2, handle) = image(dir.path(), b"abcdef", Backing::Handle);

        assert_eq!(mirror.as_slice(), Some(&b"abcdef"[..]));
        assert!(handle.as_slice().is_none());
    }

    /// `SourceOnlyImage` must differ from the mirror in exactly one respect.
    #[test]
    fn source_only_withholds_the_slice_but_reads_the_same() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("source_only.bin");
        std::fs::write(&path, b"abcdef").unwrap();
        let handle = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let mut only = SourceOnlyImage::new(MirrorImage::new(handle, b"abcdef".to_vec()));

        assert!(only.as_slice().is_none(), "the slice must be withheld");

        only.write_at(1, b"Z").unwrap();
        only.append(b"gh").unwrap();
        assert_eq!(only.len(), 8);

        let mut buf = [0u8; 8];
        only.read_at(0, &mut buf).unwrap();
        assert_eq!(&buf, b"aZcdefgh");
    }
}
