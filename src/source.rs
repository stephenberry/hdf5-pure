//! Random-access byte sources for the reader: the [`FileSource`] trait and its
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
//! [`FileSource`] is that abstraction. It is deliberately minimal and
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
//! The reader is not yet ported onto `FileSource`; that is a staged effort
//! tracked by issue #27. This module is the foundation the later stages build
//! on. The intended path, smallest-risk first:
//!
//! 1. **Foundation (this commit).** Land the trait + in-memory and `Read+Seek`
//!    backends + tests. Nothing in the existing reader changes, so there is no
//!    risk to the current in-memory path.
//! 2. **A cursor.** Introduce a small `Cursor<'a>` over a `&'a dyn FileSource`
//!    that offers the `read_offset` / `read_length` / "give me bytes at
//!    `[off, off+len)`" idioms the parsers already use, with the checked
//!    [`crate::convert`] conversions built in. The ~15 duplicated per-module
//!    `read_offset` helpers collapse into it.
//! 3. **Bulk path first.** Port the contiguous and chunked **data** readers
//!    (`data_read`, `chunked_read`, `parallel_read`) to fetch each chunk via
//!    [`FileSource::read_at`] instead of slicing the whole-file buffer. This is
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

/// A random-access, read-only source of the bytes of an HDF5 file.
///
/// Offsets are `u64` (HDF5's native address width); lengths of individual reads
/// are `usize` (they must fit in a caller-provided buffer). Implementations must
/// either fill the whole request or return an error — a short read is always an
/// error, never silently truncated.
pub trait FileSource {
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
    /// Convenience wrapper over [`read_at`](FileSource::read_at) for callers that
    /// want an owned buffer; the lazy backends keep no more than this resident.
    fn read_exact_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        let mut buf = vec![0u8; len];
        self.read_at(offset, &mut buf)?;
        Ok(buf)
    }
}

// Forward `FileSource` through references and boxes so `&S`, `&dyn FileSource`,
// and `Box<dyn FileSource>` are all usable wherever an `S: FileSource` is.
impl<S: FileSource + ?Sized> FileSource for &S {
    fn len(&self) -> u64 {
        (**self).len()
    }
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        (**self).read_at(offset, buf)
    }
}

#[cfg(feature = "std")]
impl<S: FileSource + ?Sized> FileSource for std::boxed::Box<S> {
    fn len(&self) -> u64 {
        (**self).len()
    }
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        (**self).read_at(offset, buf)
    }
}

// ---------------------------------------------------------------------------
// In-memory backend
// ---------------------------------------------------------------------------

/// A [`FileSource`] over an in-memory byte buffer: anything that is
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

    /// Borrow the underlying bytes.
    pub fn as_bytes(&self) -> &[u8] {
        self.0.as_ref()
    }
}

impl<T: AsRef<[u8]>> FileSource for BytesSource<T> {
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
// Read + Seek backend (std)
// ---------------------------------------------------------------------------

/// A lazy [`FileSource`] over any [`std::io::Read`] + [`std::io::Seek`] (a
/// [`std::fs::File`], an in-memory `Cursor`, etc.).
///
/// Each [`read_at`](FileSource::read_at) performs a `seek` + `read_exact`, so no
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

    /// Consume the source and return the wrapped reader.
    pub fn into_inner(self) -> R {
        self.inner
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

#[cfg(feature = "std")]
impl<R: std::io::Read + std::io::Seek> FileSource for ReadSeekSource<R> {
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

// ---------------------------------------------------------------------------
// Cursor
// ---------------------------------------------------------------------------

/// A checked reader over a [`FileSource`], tracking a current position.
///
/// `Cursor` is the seam the metadata parsers migrate onto: it provides the same
/// little-endian fixed-width and variable-width (`size`-byte offset/length)
/// reads the per-module `read_offset` / `read_length` helpers do today, but
/// against a `FileSource` rather than a borrowed whole-file `&[u8]`, and with
/// every offset/length checked via [`crate::convert`]. Both random access
/// (`*_at`, no position change) and sequential reads (advancing `position`) are
/// supported, mirroring how the parsers walk a structure from a base address.
///
/// Reads return owned data; a streaming source cannot hand out a borrow into a
/// window that may later be evicted. The bytes pulled per call are small
/// (metadata fields), so the allocations are negligible.
pub struct Cursor<'s> {
    source: &'s dyn FileSource,
    pos: u64,
}

impl<'s> Cursor<'s> {
    /// Create a cursor at offset 0.
    pub fn new(source: &'s dyn FileSource) -> Self {
        Cursor { source, pos: 0 }
    }

    /// Create a cursor positioned at `pos`.
    pub fn at(source: &'s dyn FileSource, pos: u64) -> Self {
        Cursor { source, pos }
    }

    /// The underlying source.
    pub fn source(&self) -> &'s dyn FileSource {
        self.source
    }

    /// Total length of the source in bytes.
    pub fn len(&self) -> u64 {
        self.source.len()
    }

    /// Whether the source is empty.
    pub fn is_empty(&self) -> bool {
        self.source.is_empty()
    }

    /// The current read position.
    pub fn position(&self) -> u64 {
        self.pos
    }

    /// Move the read position to `pos` (absolute).
    pub fn seek(&mut self, pos: u64) {
        self.pos = pos;
    }

    /// Advance the read position by `delta` bytes, checking for `u64` overflow.
    pub fn advance(&mut self, delta: u64) -> Result<(), FormatError> {
        self.pos = self
            .pos
            .checked_add(delta)
            .ok_or(FormatError::OffsetOverflow {
                offset: self.pos,
                length: delta,
            })?;
        Ok(())
    }

    /// Bytes remaining between the current position and the end of the source.
    pub fn remaining(&self) -> u64 {
        self.source.len().saturating_sub(self.pos)
    }

    /// Read `n` bytes at absolute `offset` into an owned buffer (no position
    /// change).
    pub fn bytes_at(&self, offset: u64, n: usize) -> Result<Vec<u8>, FormatError> {
        self.source.read_exact_at(offset, n)
    }

    /// Read `n` bytes at the current position into an owned buffer, advancing.
    pub fn read_bytes(&mut self, n: usize) -> Result<Vec<u8>, FormatError> {
        let out = self.source.read_exact_at(self.pos, n)?;
        self.advance(n as u64)?;
        Ok(out)
    }

    /// Read a single byte at the current position, advancing.
    pub fn read_u8(&mut self) -> Result<u8, FormatError> {
        let mut b = [0u8; 1];
        self.source.read_at(self.pos, &mut b)?;
        self.advance(1)?;
        Ok(b[0])
    }

    /// Read a little-endian `u16` at the current position, advancing.
    pub fn read_u16(&mut self) -> Result<u16, FormatError> {
        let mut b = [0u8; 2];
        self.source.read_at(self.pos, &mut b)?;
        self.advance(2)?;
        Ok(u16::from_le_bytes(b))
    }

    /// Read a little-endian `u32` at the current position, advancing.
    pub fn read_u32(&mut self) -> Result<u32, FormatError> {
        let mut b = [0u8; 4];
        self.source.read_at(self.pos, &mut b)?;
        self.advance(4)?;
        Ok(u32::from_le_bytes(b))
    }

    /// Read a little-endian `u64` at the current position, advancing.
    pub fn read_u64(&mut self) -> Result<u64, FormatError> {
        let mut b = [0u8; 8];
        self.source.read_at(self.pos, &mut b)?;
        self.advance(8)?;
        Ok(u64::from_le_bytes(b))
    }

    /// Read a little-endian variable-width unsigned integer of `size` bytes
    /// (1, 2, 4, or 8 — HDF5's "size of offsets" / "size of lengths") at the
    /// current position, widening to `u64` and advancing.
    ///
    /// This is the [`Cursor`] equivalent of the `read_offset` / `read_length`
    /// helpers duplicated across the parser modules.
    pub fn read_uint(&mut self, size: u8) -> Result<u64, FormatError> {
        let v = match size {
            1 => u64::from(self.read_u8()?),
            2 => u64::from(self.read_u16()?),
            4 => u64::from(self.read_u32()?),
            8 => self.read_u64()?,
            _ => return Err(FormatError::InvalidOffsetSize(size)),
        };
        Ok(v)
    }
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
    fn empty_source() {
        let src = BytesSource::new(Vec::<u8>::new());
        assert_eq!(src.len(), 0);
        assert!(src.is_empty());
    }

    #[test]
    fn forwarding_through_reference() {
        let src = BytesSource::new(vec![9u8, 8, 7]);
        let r: &dyn FileSource = &src;
        let mut buf = [0u8; 2];
        r.read_at(1, &mut buf).unwrap();
        assert_eq!(buf, [8, 7]);
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

    #[test]
    fn cursor_sequential_little_endian_reads() {
        // 1 + 2 + 4 + 8 = 15 bytes of known little-endian values.
        let mut data = Vec::new();
        data.push(0xABu8);
        data.extend_from_slice(&0x1234u16.to_le_bytes());
        data.extend_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        data.extend_from_slice(&0x0102_0304_0506_0708u64.to_le_bytes());
        let src = BytesSource::new(data);
        let mut c = Cursor::new(&src);
        assert_eq!(c.position(), 0);
        assert_eq!(c.read_u8().unwrap(), 0xAB);
        assert_eq!(c.read_u16().unwrap(), 0x1234);
        assert_eq!(c.read_u32().unwrap(), 0xDEAD_BEEF);
        assert_eq!(c.read_u64().unwrap(), 0x0102_0304_0506_0708);
        assert_eq!(c.position(), 15);
        assert_eq!(c.remaining(), 0);
    }

    #[test]
    fn cursor_read_uint_widths() {
        let data = vec![0x01u8, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let src = BytesSource::new(data);

        let mut c = Cursor::new(&src);
        assert_eq!(c.read_uint(1).unwrap(), 0x01);
        let mut c = Cursor::new(&src);
        assert_eq!(c.read_uint(2).unwrap(), 0x0201);
        let mut c = Cursor::new(&src);
        assert_eq!(c.read_uint(4).unwrap(), 0x0403_0201);
        let mut c = Cursor::new(&src);
        assert_eq!(c.read_uint(8).unwrap(), 0x0807_0605_0403_0201);

        // An invalid offset/length size is rejected, not silently mis-read.
        let mut c = Cursor::new(&src);
        assert!(matches!(
            c.read_uint(3).unwrap_err(),
            FormatError::InvalidOffsetSize(3)
        ));
    }

    #[test]
    fn cursor_random_access_and_seek() {
        let data = (0u8..32).collect::<Vec<u8>>();
        let src = BytesSource::new(data);
        let mut c = Cursor::new(&src);

        // Random access does not move the position.
        assert_eq!(c.bytes_at(10, 3).unwrap(), vec![10, 11, 12]);
        assert_eq!(c.position(), 0);

        // Seek + sequential read.
        c.seek(20);
        assert_eq!(c.read_bytes(4).unwrap(), vec![20, 21, 22, 23]);
        assert_eq!(c.position(), 24);
    }

    #[test]
    fn cursor_eof_is_reported() {
        let src = BytesSource::new(vec![1u8, 2, 3]);
        let mut c = Cursor::at(&src, 2);
        assert_eq!(c.read_u8().unwrap(), 3);
        assert!(matches!(
            c.read_u8().unwrap_err(),
            FormatError::UnexpectedEof { .. }
        ));
    }

    #[cfg(feature = "std")]
    #[test]
    fn cursor_drives_the_same_bytes_over_any_backend() {
        // The whole point: a Cursor yields identical results whether the source
        // is in memory or a lazy Read+Seek.
        let data = (0u8..=255).collect::<Vec<u8>>();
        let mem = BytesSource::new(data.clone());
        let seek = ReadSeekSource::new(std::io::Cursor::new(data)).unwrap();

        let mut cm = Cursor::at(&mem, 100);
        let mut cs = Cursor::at(&seek, 100);
        assert_eq!(cm.read_u32().unwrap(), cs.read_u32().unwrap());
        assert_eq!(cm.read_uint(8).unwrap(), cs.read_uint(8).unwrap());
        assert_eq!(cm.bytes_at(0, 16).unwrap(), cs.bytes_at(0, 16).unwrap());
    }
}
