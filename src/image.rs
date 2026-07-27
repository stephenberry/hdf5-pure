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
//! and append. The difference between them is entirely a property of *where the
//! bytes live*, not of what edits are expressible, so it belongs behind this
//! trait rather than in two engines.
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
//! [`Source::len`] is the image's *logical* length, and appends extend it. For
//! the mirror that is the buffer length; for a file-backed image it is a
//! counter, which may be shorter than the real file if a previous session died
//! mid-append (its trailing bytes are unreferenced slack, exactly as after a
//! crash).

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
    /// written at. Extends [`Source::len`] by `bytes.len()`.
    fn append(&mut self, bytes: &[u8]) -> Result<u64, Error>;

    /// Overwrite `[offset, offset + bytes.len())` in place. The caller
    /// guarantees the range already exists.
    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error>;

    /// Shrink the image to `len` bytes, physically shortening the file.
    fn truncate(&mut self, len: u64) -> Result<(), Error>;

    /// Flush buffered writes and force the file's *data* to durable storage.
    fn sync_data(&mut self) -> Result<(), Error>;

    /// Flush buffered writes and force the file's data **and metadata** to
    /// durable storage. Distinct from [`sync_data`](Self::sync_data) because a
    /// commit changes the file's length, which lives in that metadata.
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
        self.handle
            .seek(SeekFrom::Start(offset))
            .map_err(Error::Io)?;
        self.handle.write_all(bytes).map_err(Error::Io)?;
        let offset = offset.to_usize()?;
        self.data[offset..offset + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    fn truncate(&mut self, len: u64) -> Result<(), Error> {
        self.handle.set_len(len).map_err(Error::Io)?;
        self.data.truncate(len.to_usize()?);
        Ok(())
    }

    fn sync_data(&mut self) -> Result<(), Error> {
        self.handle.flush().map_err(Error::Io)?;
        self.handle.sync_data().map_err(Error::Io)?;
        Ok(())
    }

    fn sync_all(&mut self) -> Result<(), Error> {
        self.handle.sync_all().map_err(Error::Io)?;
        Ok(())
    }

    fn as_slice(&self) -> Option<&[u8]> {
        Some(&self.data)
    }
}
