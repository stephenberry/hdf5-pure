//! SWMR (single-writer / multiple-reader) append writer.
//!
//! **Deprecated:** superseded by the owned-handle API — open with
//! [`crate::File::open_swmr_writer`] and append through a [`crate::Dataset`]
//! handle, then [`crate::File::close`] to clear the flag; recover a stale flag
//! with [`crate::File::clear_swmr_flag`]. See the [`SwmrWriter`] type docs for the
//! migration.
//!
//! Opens an existing HDF5 file (created by this crate, the reference C library,
//! or h5py with the latest format) and appends chunks to a one-dimensional,
//! unlimited, Extensible-Array-indexed dataset *in place*: each appended chunk
//! is written at end-of-file, its address is stored into the next free element
//! slot of the chunk index, the index grows by appending new blocks only when a
//! block boundary is crossed (never relocating existing data), the dataspace
//! dimension and array header counts are patched, and the superblock end-of-file
//! is advanced. Writes are issued child-before-parent with `fsync` barriers so a
//! concurrent reader (via [`crate::File::refresh`], the C library's `H5Drefresh`,
//! or h5py's `Dataset.refresh()`) only ever observes a consistent prefix.
//!
//! The in-place chunk-index mechanics live in [`crate::chunk_index_inplace`] and
//! are shared with the general (non-SWMR) append writer
//! [`crate::AppendWriter`]; this module layers the SWMR policy on top.
//!
//! # Supported subset (v1)
//!
//! - One unlimited dimension, chunked, Extensible-Array index (the index the C
//!   library and h5py select for a single unlimited dimension under the latest
//!   format).
//! - Unfiltered datasets (no compression/shuffle/scale-offset on the appended
//!   dataset). Filtered append is rejected with a clear error. For filtered
//!   in-place appends without concurrent readers, see [`crate::AppendWriter`].
//! - Appends land on chunk boundaries: the dataset's current length and the
//!   appended length are both whole multiples of the chunk length. (A chunk
//!   length of 1 — the common streaming layout — always satisfies this.)
//! - Unbounded growth: super blocks and (past ~131060 chunks) paged data blocks
//!   are allocated incrementally as block boundaries are crossed.
//! - Files with a zero base address (no userblock).
//!
//! ## Crash recovery (the consistency flag)
//!
//! The writer marks the file with the superblock's SWMR-write *consistency flag*
//! (`0x05`) while active and clears it on [`SwmrWriter::close`] / `Drop`. This
//! flag is durable file data, *not* an OS lock: a hard crash (`SIGKILL`, power
//! loss, or a `panic = "abort"` build where `Drop` never runs) leaves it set, and
//! the reference C library then refuses to open the file until it is cleared.
//! Recover such a file with [`SwmrWriter::clear_swmr_flag`] (the `h5clear -s`
//! equivalent).
//!
//! Unlike a read-write [`crate::File::open_rw`] handle, the SWMR writer takes
//! **no OS file lock**: SWMR
//! is single-writer by contract and built for concurrent reads (the reference
//! library likewise runs SWMR with file locking disabled), so a whole-file lock
//! would block readers — fatally so on Windows, where locks are mandatory. The
//! single-writer invariant is the caller's responsibility, as it is in HDF5.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::chunk_index_inplace::{ElemRecord, InPlaceFile, Located};
use crate::error::{Error, FormatError};
use crate::file_lock::{self, FileLocking};
use crate::signature;
use crate::superblock::Superblock;

/// An append writer over an existing HDF5 file.
///
/// The writer keeps a full in-memory mirror of the file (`O(file size)` memory)
/// so it can read existing structures and recompute checksums without hitting
/// the disk. For the unbounded "streaming log" use case this grows with the
/// file; a future revision could bound it, but v1 favors simplicity.
///
/// # Deprecated
///
/// Superseded by the owned-handle API: open the file with
/// [`File::open_swmr_writer`](crate::File::open_swmr_writer) and append through a
/// [`Dataset`](crate::Dataset) handle. That path layers the same SWMR policy on
/// the shared in-place engine — no OS lock, the `0x05` SWMR-write flag raised
/// while active and cleared on [`File::close`](crate::File::close), and the same
/// unfiltered, chunk-aligned append subset — and the one open file also reads
/// back what it appends. Recover a flag left set by a crashed writer with
/// [`File::clear_swmr_flag`](crate::File::clear_swmr_flag).
///
/// ```no_run
/// use hdf5_pure::File;
///
/// let file = File::open_swmr_writer("log.h5")?;
/// let mut samples = file.dataset("samples")?;
/// samples.append(&[8i32, 9, 10, 11])?; // one whole chunk
/// file.close()?; // clears the SWMR-write flag
/// # Ok::<(), hdf5_pure::Error>(())
/// ```
///
/// The type will be removed in a later release.
#[deprecated(
    since = "0.22.0",
    note = "use File::open_swmr_writer + Dataset::append; see the SwmrWriter type docs for migration"
)]
pub struct SwmrWriter {
    /// In-memory mirror + on-disk handle, shared with the general append writer.
    file: InPlaceFile,
    located: HashMap<String, Located>,
    /// Whether the superblock's SWMR-write consistency flag is currently set
    /// (so [`Drop`] knows to clear it).
    flag_set: bool,
}

/// Superblock consistency-flag bits: bit 0 = write access, bit 2 = SWMR write
/// access. A SWMR writer sets both (`0x05`) while writing and clears them on a
/// clean close — matching the reference C library and h5py.
const SWMR_WRITE_FLAGS: u32 = 0x05;

#[allow(deprecated)] // this type's own impl legitimately names and builds the (deprecated) type
impl SwmrWriter {
    /// Open an existing HDF5 file for appending.
    ///
    /// Does **not** take an OS file lock: SWMR is single-writer *by contract* and
    /// designed for concurrent reads, so (like the reference library, which runs
    /// SWMR with locking off) a lock would defeat the multiple-reader half — and
    /// on Windows, where locks are mandatory, it would block readers outright.
    /// The writer instead marks the file with the superblock SWMR flag while
    /// active; recover a file left flagged by a crashed writer with
    /// [`clear_swmr_flag`](Self::clear_swmr_flag).
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let file = InPlaceFile::open(path, None, Error::SwmrAppendUnsupported)?;
        let mut w = Self {
            file,
            located: HashMap::new(),
            flag_set: false,
        };
        // Mark the file as having an active SWMR writer so a concurrent reader
        // may open it with `H5F_ACC_SWMR_READ` / h5py `swmr=True`. Cleared on
        // `close`/drop.
        w.set_swmr_flag(true)?;
        Ok(w)
    }

    /// Set or clear the superblock's SWMR-write consistency flags, recompute the
    /// superblock checksum, and flush.
    fn set_swmr_flag(&mut self, active: bool) -> Result<(), Error> {
        self.file
            .set_consistency_flags(if active { SWMR_WRITE_FLAGS } else { 0 })?;
        self.flag_set = active;
        Ok(())
    }

    /// Finish writing: clear the SWMR-write flag and flush, marking the file
    /// cleanly closed. Prefer this over relying on `Drop` so the (rare) flush
    /// error surfaces.
    pub fn close(mut self) -> Result<(), Error> {
        self.set_swmr_flag(false)
    }

    /// Clear a stale SWMR-write flag left in `path` by a writer that exited
    /// without a clean close (the h5clear equivalent). Safe to call on a file
    /// with the flag already clear.
    pub fn clear_swmr_flag<P: AsRef<Path>>(path: P) -> Result<(), Error> {
        clear_swmr_flag_at(path.as_ref())
    }

    /// Append `i32` values to an unlimited dataset.
    pub fn append_i32(&mut self, dataset: &str, values: &[i32]) -> Result<(), Error> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for &v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.append_raw(dataset, &bytes)
    }

    /// Append `f64` values to an unlimited dataset.
    pub fn append_f64(&mut self, dataset: &str, values: &[f64]) -> Result<(), Error> {
        let mut bytes = Vec::with_capacity(values.len() * 8);
        for &v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.append_raw(dataset, &bytes)
    }

    /// Append raw little-endian element bytes to an unlimited dataset. `bytes`
    /// must be a whole number of chunks' worth of elements, and the dataset's
    /// current length must be chunk-aligned.
    ///
    /// On success the appended chunks are durably committed and visible to a
    /// refreshing reader. On error (including an underlying I/O failure) the new
    /// length is never published, so a reader still sees the prior consistent
    /// prefix; the writer should then be dropped rather than reused, as its
    /// in-memory mirror may have advanced past what reached disk.
    pub fn append_raw(&mut self, dataset: &str, bytes: &[u8]) -> Result<(), Error> {
        self.append_phased(dataset, bytes, 4)
    }

    /// Append, running only the first `max_phase` durability phases (1-4). Used
    /// by crash-consistency tests to stop at a phase boundary; production
    /// callers always use `append_raw` (`max_phase = 4`).
    fn append_phased(&mut self, dataset: &str, bytes: &[u8], max_phase: u8) -> Result<(), Error> {
        if !self.located.contains_key(dataset) {
            let result = Located::locate(&self.file, dataset, Error::SwmrAppendUnsupported)?;
            if result.has_filters {
                // The SWMR path stores bare-address EA elements and never
                // compresses; filtered append is the general writer's job.
                return Err(Error::SwmrAppendUnsupported(
                    "filtered datasets are not supported",
                ));
            }
            self.located.insert(dataset.to_string(), result.located);
        }
        // Pull required immutable facts out before mutating self.
        let (chunk_bytes, chunk_elems, elem_bytes, current_dim, num_chunks) = {
            let loc = &self.located[dataset];
            (
                loc.chunk_bytes,
                loc.chunk_elems,
                loc.elem_bytes,
                loc.current_dim,
                loc.num_chunks,
            )
        };

        if elem_bytes == 0 || chunk_bytes == 0 {
            return Err(Error::SwmrAppendUnsupported(
                "dataset has zero-sized elements or chunks",
            ));
        }
        if bytes.len() % elem_bytes != 0 {
            return Err(Error::Format(FormatError::ChunkedReadError(
                "append byte length is not a whole number of elements".into(),
            )));
        }
        let new_elems = (bytes.len() / elem_bytes) as u64;
        if new_elems == 0 {
            return Ok(());
        }
        if current_dim % chunk_elems != 0 || new_elems % chunk_elems != 0 {
            return Err(Error::Format(FormatError::ChunkedReadError(
                "SWMR append must be chunk-aligned (current length and appended length \
                 must be multiples of the chunk length)"
                    .into(),
            )));
        }

        let n_new_chunks = bytes.len() / chunk_bytes;

        // The four phases below are ordered child-before-parent with an fsync
        // barrier after each, so a reader (and a crash) only ever observes a
        // consistent prefix:
        //
        //   1. raw chunk data + the chunk-index structures that point at it
        //      (new blocks, element slots, recomputed block checksums), but
        //      NOT the published element count yet;
        //   2. the superblock end-of-file, so the file's allocated extent
        //      covers everything written in phase 1 before any published
        //      metadata references it;
        //   3. the EA header element count, which makes the new chunks
        //      reachable through the index; and
        //   4. the dataspace dimension — the dataset's authoritative size,
        //      which a reader bounds chunk reads by. Publishing it last is the
        //      single commit point: before it a reader sees the old length,
        //      after it the new one, and never a torn view.

        // Phase 1.
        for c in 0..n_new_chunks {
            let chunk_data = &bytes[c * chunk_bytes..(c + 1) * chunk_bytes];
            let chunk_addr = self.file.append_bytes(chunk_data)?;
            let e = num_chunks + c as u64;
            self.located[dataset].ea_insert(
                &mut self.file,
                e,
                ElemRecord::addr_only(chunk_addr),
            )?;
        }
        self.file.sync()?;
        if max_phase < 2 {
            return Ok(());
        }

        // Phase 2.
        self.file.patch_superblock_eof()?;
        self.file.sync()?;
        if max_phase < 3 {
            return Ok(());
        }

        // Phase 3.
        let new_num_chunks = num_chunks + n_new_chunks as u64;
        self.located[dataset].update_ea_header(&mut self.file, new_num_chunks)?;
        self.file.sync()?;
        if max_phase < 4 {
            return Ok(());
        }

        // Phase 4.
        let new_dim = current_dim + new_elems;
        self.located[dataset].patch_dimension(&mut self.file, new_dim)?;
        self.file.sync()?;

        if let Some(loc) = self.located.get_mut(dataset) {
            loc.current_dim = new_dim;
            loc.num_chunks = new_num_chunks;
        }
        Ok(())
    }
}

/// Clear a stale SWMR-write flag left in `path` by a writer that exited without a
/// clean close — the `h5clear -s` equivalent, shared by the deprecated
/// [`SwmrWriter::clear_swmr_flag`] and the owned
/// [`File::clear_swmr_flag`](crate::File::clear_swmr_flag). Safe to call on a file
/// whose flag is already clear.
pub(crate) fn clear_swmr_flag_at(path: &Path) -> Result<(), Error> {
    let mut w = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(Error::Io)?;
    // Refuse to clear the flag out from under a live writer: an exclusive
    // lock here fails with `FileLocked` if another writer still holds the
    // file. A stale flag from a *crashed* writer has no live lock, so this
    // succeeds and the recovery proceeds.
    file_lock::acquire_exclusive(&w, FileLocking::Enabled, path)?;
    let mut data = Vec::new();
    w.read_to_end(&mut data).map_err(Error::Io)?;
    let sig = signature::find_signature(&data)?;
    let mut sb = Superblock::parse(&data, sig)?;
    if sb.version < 2 {
        // `Superblock::serialize` emits the v2/v3 layout, so rewriting a
        // v0/v1 superblock here would corrupt it (the same hazard `open`
        // guards against). This crate never SWMR-flags a v0/v1 file, so
        // there is nothing to clear; treat it as already clean rather than
        // risk a destructive rewrite.
        return Ok(());
    }
    if sb.consistency_flags == 0 {
        return Ok(());
    }
    sb.consistency_flags = 0;
    let bytes = sb.serialize();
    w.seek(SeekFrom::Start(sig as u64)).map_err(Error::Io)?;
    w.write_all(&bytes).map_err(Error::Io)?;
    w.sync_data().map_err(Error::Io)?;
    Ok(())
}

#[allow(deprecated)] // Drop impl of the deprecated type
impl Drop for SwmrWriter {
    /// Best-effort clear of the SWMR-write flag so a writer that is merely
    /// dropped (rather than `close`d) still leaves the file cleanly marked. Use
    /// [`SwmrWriter::close`] to observe flush errors.
    fn drop(&mut self) {
        if self.flag_set {
            let _ = self.set_swmr_flag(false);
        }
    }
}

#[cfg(test)]
#[allow(deprecated)] // exercises the deprecated SwmrWriter shim
mod tests {
    use super::*;
    use crate::reader::File as PureFile;
    use crate::writer::FileBuilder;
    use tempfile::tempdir;

    fn i32_bytes(range: std::ops::Range<i32>) -> Vec<u8> {
        let mut b = Vec::new();
        for v in range {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b
    }

    /// Stopping an append at any phase boundary must leave the file readable as
    /// a consistent prefix: the old length until the final (dimension) commit,
    /// the new length after it — never a torn view or out-of-bounds chunk.
    #[test]
    fn crash_consistency_consistent_prefix() {
        let dir = tempdir().unwrap();
        let base = dir.path().join("base.h5");
        let n = 50i32;
        let target = 250i32; // crosses the inline -> direct -> super-block boundary
        {
            let data: Vec<i32> = (0..n).collect();
            let mut b = FileBuilder::new();
            b.create_dataset("d")
                .with_i32_data(&data)
                .with_shape(&[n as u64])
                .with_maxshape(&[u64::MAX])
                .with_chunks(&[1]);
            b.write(&base).unwrap();
        }

        for max_phase in 1u8..=4 {
            let p = dir.path().join(format!("crash_{max_phase}.h5"));
            std::fs::copy(&base, &p).unwrap();
            {
                let mut w = SwmrWriter::open(&p).unwrap();
                w.append_phased("d", &i32_bytes(n..target), max_phase)
                    .unwrap();
                // writer dropped here, simulating a crash after `max_phase`
            }
            let expected_len = if max_phase == 4 { target } else { n };
            let f = PureFile::from_bytes(std::fs::read(&p).unwrap()).unwrap();
            let v = f.dataset("d").unwrap().read_i32().unwrap();
            assert_eq!(
                v,
                (0..expected_len).collect::<Vec<_>>(),
                "inconsistent view after crash at phase {max_phase}"
            );
        }
    }

    /// Same consistent-prefix guarantee, but for an append that crosses the
    /// paged-data-block boundary (~131060 chunks) so the most intricate in-place
    /// growth runs: allocating a paged super block, paged data blocks, and the
    /// per-page checksums + page-init bitmap, all in phase 1. Stopping at any
    /// phase boundary must still read back as the old prefix until the final
    /// dimension commit. Slow (~131k chunks), like `append_crosses_paging_boundary`.
    #[test]
    fn crash_consistency_paged_prefix() {
        let dir = tempdir().unwrap();
        let base = dir.path().join("base.h5");
        let start = 131_000i32; // just below the paging boundary
        let target = 132_000i32; // crosses it -> paged super block + data blocks
        {
            let data: Vec<i32> = (0..start).collect();
            let mut b = FileBuilder::new();
            b.create_dataset("d")
                .with_i32_data(&data)
                .with_shape(&[start as u64])
                .with_maxshape(&[u64::MAX])
                .with_chunks(&[1]);
            b.write(&base).unwrap();
        }

        for max_phase in 1u8..=4 {
            let p = dir.path().join(format!("crash_paged_{max_phase}.h5"));
            std::fs::copy(&base, &p).unwrap();
            {
                let mut w = SwmrWriter::open(&p).unwrap();
                w.append_phased("d", &i32_bytes(start..target), max_phase)
                    .unwrap();
                // writer dropped here, simulating a crash after `max_phase`
            }
            let expected_len = if max_phase == 4 { target } else { start };
            let f = PureFile::from_bytes(std::fs::read(&p).unwrap()).unwrap();
            let v = f.dataset("d").unwrap().read_i32().unwrap();
            assert_eq!(
                v,
                (0..expected_len).collect::<Vec<_>>(),
                "inconsistent paged view after crash at phase {max_phase}"
            );
        }
    }

    /// The consistent-prefix guarantee must hold for the *reference C library*,
    /// not only this crate's reader. The pure reader bounds chunk reads by
    /// `min(EA count, dimension)`, so it tolerates a phase-3 state where the EA
    /// count has advanced past the dimension; the C library instead walks
    /// strictly by the dataspace dimension and re-validates block checksums. A
    /// stale end-of-file, a half-grown index, or a mis-checksummed block at an
    /// intermediate phase could therefore satisfy the pure reader yet break C
    /// or h5py. Open the file fresh with the C library at each stopped phase and
    /// confirm it reads the old length until the phase-4 dimension commit.
    #[test]
    // Reads back with the reference HDF5 C library (`hdf5-metno`), which is a
    // 64-bit-only dev-dependency; skip on 32-bit so the lib tests run there.
    #[cfg(not(target_pointer_width = "32"))]
    fn crash_consistency_c_library_reads_prefix() {
        let dir = tempdir().unwrap();
        let base = dir.path().join("base.h5");
        let n = 50i32;
        let target = 250i32; // crosses the inline -> direct -> super-block boundary
        {
            let data: Vec<i32> = (0..n).collect();
            let mut b = FileBuilder::new();
            b.create_dataset("d")
                .with_i32_data(&data)
                .with_shape(&[n as u64])
                .with_maxshape(&[u64::MAX])
                .with_chunks(&[1]);
            b.write(&base).unwrap();
        }

        for max_phase in 1u8..=4 {
            let p = dir.path().join(format!("crash_c_{max_phase}.h5"));
            std::fs::copy(&base, &p).unwrap();
            {
                let mut w = SwmrWriter::open(&p).unwrap();
                w.append_phased("d", &i32_bytes(n..target), max_phase)
                    .unwrap();
                // writer dropped here, simulating a crash after `max_phase`
            }
            let expected_len = if max_phase == 4 { target } else { n };
            let f = hdf5::File::open(&p).unwrap();
            let v = f.dataset("d").unwrap().read_raw::<i32>().unwrap();
            assert_eq!(
                v,
                (0..expected_len).collect::<Vec<_>>(),
                "C library saw an inconsistent view after crash at phase {max_phase}"
            );
            f.close().unwrap();
        }
    }

    /// Crash recovery across the phase-3/phase-4 gap. A writer that crashes
    /// after publishing the EA element count (phase 3) but before publishing the
    /// dataspace dimension (phase 4) leaves the on-disk count ahead of the
    /// committed dimension. A fresh writer must roll forward from the committed
    /// dimension, overwriting the uncommitted slots, rather than appending past
    /// them and leaving a gap. The crashed and recovery appends deliberately
    /// write *different* values at the overlapping positions so a regression
    /// (seeding the chunk count from the stale EA header) surfaces the crashed
    /// writer's values instead of the recovery writer's.
    #[test]
    // Reads back with the reference HDF5 C library (`hdf5-metno`), which is a
    // 64-bit-only dev-dependency; skip on 32-bit so the lib tests run there.
    #[cfg(not(target_pointer_width = "32"))]
    fn recover_and_reappend_after_phase3_crash() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("phase3_recover.h5");
        let n = 50i32;
        {
            let data: Vec<i32> = (0..n).collect();
            let mut b = FileBuilder::new();
            b.create_dataset("d")
                .with_i32_data(&data)
                .with_shape(&[n as u64])
                .with_maxshape(&[u64::MAX])
                .with_chunks(&[1]);
            b.write(&path).unwrap();
        }

        // Writer 1 crashes after phase 3: the EA count advances to 250 but the
        // dimension stays 50. Its appended values (1000..1200) are distinct from
        // the eventual correct continuation so a leak is detectable.
        {
            let mut w = SwmrWriter::open(&path).unwrap();
            w.append_phased("d", &i32_bytes(1000..1200), 3).unwrap();
            // dropped without phase 4 -> dimension not published
        }
        // The committed prefix is still the original 50 elements (pure + C).
        let pf = PureFile::from_bytes(std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(
            pf.dataset("d").unwrap().read_i32().unwrap(),
            (0..n).collect::<Vec<_>>(),
            "phase-3 crash exposed uncommitted data to the pure reader"
        );
        {
            let f = hdf5::File::open(&path).unwrap();
            assert_eq!(
                f.dataset("d").unwrap().read_raw::<i32>().unwrap(),
                (0..n).collect::<Vec<_>>(),
                "phase-3 crash exposed uncommitted data to the C library"
            );
            f.close().unwrap();
        }

        // Writer 2 recovers: it must roll forward from the committed dimension
        // (50), overwriting the uncommitted slots, and append the real
        // continuation 50..150.
        {
            let mut w = SwmrWriter::open(&path).unwrap();
            w.append_i32("d", &(n..150).collect::<Vec<_>>()).unwrap();
            w.close().unwrap();
        }

        let expected: Vec<i32> = (0..150).collect();
        let pf = PureFile::from_bytes(std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(
            pf.dataset("d").unwrap().read_i32().unwrap(),
            expected,
            "recovery did not roll forward correctly (pure reader)"
        );
        let f = hdf5::File::open(&path).unwrap();
        assert_eq!(
            f.dataset("d").unwrap().read_raw::<i32>().unwrap(),
            expected,
            "recovery did not roll forward correctly (C library)"
        );
        f.close().unwrap();
    }
}
