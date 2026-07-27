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
//! while `File::open_rw_bounded` held no mirror and could therefore only read
//! and append. *That* limitation is a property of where the bytes live, not of
//! what edits are expressible, so it belongs behind this trait rather than in
//! two engines.
//!
//! The trait abstracts residency and nothing else. The two backends still
//! differ in ways this seam does not reach and does not claim to: which files
//! they will open at all (the bounded backend refuses a v0/v1 superblock,
//! 4-byte offsets, and any userblock), and when they rewrite persisted
//! free-space managers. Those are separate reconciliations.
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
use std::io::{Seek, SeekFrom, Write};

use crate::convert::TryToUsize;
use crate::error::{Error, FormatError};
use crate::source::{BytesSource, Source};

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
    /// address readable, then cached, then appended over.
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
/// byte-for-byte in sync. This is the backing behind [`File::open_rw`](crate::File::open_rw):
/// reads are slice accesses and never touch the disk, at the cost of holding
/// the entire file resident.
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

/// A [`MirrorImage`] that withholds its slice, so every read routed through it
/// takes the [`Source`] path instead of the whole-file fast path.
///
/// This exists to make the mirrorless read paths reachable before a mirrorless
/// backing does. Each read the engine serves has two forms — one walking a
/// borrowed slice, one going through `Source` — and until `File::open_rw_bounded`
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

    /// Open a fresh file holding `initial`, wrapped as a mirror image.
    fn image(dir: &std::path::Path, initial: &[u8]) -> (std::path::PathBuf, MirrorImage) {
        let path = dir.join("image.bin");
        std::fs::write(&path, initial).unwrap();
        let handle = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        (path, MirrorImage::new(handle, initial.to_vec()))
    }

    /// The invariant the whole type exists to hold: after any primitive, the
    /// mirror and the bytes on disk agree. Every case below asserts it, because
    /// a primitive that updates only one side is exactly the defect that would
    /// otherwise surface as a corrupt file much later.
    fn assert_in_sync(path: &std::path::Path, img: &MirrorImage) {
        let on_disk = std::fs::read(path).unwrap();
        assert_eq!(
            img.as_slice().unwrap(),
            &on_disk[..],
            "mirror and file disagree"
        );
        assert_eq!(img.len(), on_disk.len() as u64, "len disagrees with file");
    }

    #[test]
    fn append_returns_the_pre_append_end_and_extends_by_exactly_the_length() {
        let dir = tempfile::tempdir().unwrap();
        let (path, mut img) = image(dir.path(), b"abcd");

        let addr = img.append(b"XYZ").unwrap();

        assert_eq!(addr, 4, "append must report the address it wrote at");
        assert_eq!(
            img.len(),
            7,
            "append must extend len by exactly bytes.len()"
        );
        assert_in_sync(&path, &img);
    }

    #[test]
    fn write_at_overwrites_both_sides_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let (path, mut img) = image(dir.path(), b"abcdef");

        img.write_at(2, b"ZZ").unwrap();

        assert_eq!(img.as_slice().unwrap(), b"abZZef");
        assert_eq!(img.len(), 6, "an in-place write must not move end-of-file");
        assert_in_sync(&path, &img);
    }

    #[test]
    fn truncate_shrinks_both_sides() {
        let dir = tempfile::tempdir().unwrap();
        let (path, mut img) = image(dir.path(), b"abcdef");

        img.truncate(2).unwrap();

        assert_eq!(img.as_slice().unwrap(), b"ab");
        assert_eq!(img.len(), 2);
        assert_in_sync(&path, &img);
    }

    /// A write following a truncate must land at the shortened end-of-file, not
    /// wherever the previous operation left the handle's cursor.
    #[test]
    fn append_after_truncate_lands_at_the_new_end() {
        let dir = tempfile::tempdir().unwrap();
        let (path, mut img) = image(dir.path(), b"abcdef");

        img.truncate(3).unwrap();
        let addr = img.append(b"Z").unwrap();

        assert_eq!(addr, 3);
        assert_eq!(img.as_slice().unwrap(), b"abcZ");
        assert_in_sync(&path, &img);
    }

    /// Reads go through `Source`, so they must observe writes immediately —
    /// the engine plans its next edit against bytes it just wrote.
    #[test]
    fn reads_observe_writes_immediately() {
        let dir = tempfile::tempdir().unwrap();
        let (_path, mut img) = image(dir.path(), b"abcdef");
        img.write_at(0, b"ZY").unwrap();
        img.append(b"!").unwrap();

        let mut buf = [0u8; 3];
        img.read_at(0, &mut buf).unwrap();
        assert_eq!(&buf, b"ZYc");
        img.read_at(6, &mut buf[..1]).unwrap();
        assert_eq!(buf[0], b'!');
    }

    #[test]
    fn reads_past_end_of_file_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let (_path, img) = image(dir.path(), b"abcd");

        let mut buf = [0u8; 2];
        assert!(img.read_at(3, &mut buf).is_err());
    }

    /// `SourceOnlyImage` must differ from the mirror in exactly one respect.
    #[test]
    fn source_only_withholds_the_slice_but_reads_the_same() {
        let dir = tempfile::tempdir().unwrap();
        let (_path, mirror) = image(dir.path(), b"abcdef");
        let mut only = SourceOnlyImage::new(mirror);

        assert!(only.as_slice().is_none(), "the slice must be withheld");

        only.write_at(1, b"Z").unwrap();
        only.append(b"gh").unwrap();
        assert_eq!(only.len(), 8);

        let mut buf = [0u8; 8];
        only.read_at(0, &mut buf).unwrap();
        assert_eq!(&buf, b"aZcdefgh");
    }
}
