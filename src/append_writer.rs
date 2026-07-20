//! General (non-SWMR) in-place append writer.
//!
//! **Deprecated:** superseded by the owned-handle API — open with
//! [`crate::File::open_rw`] and append through a [`crate::Dataset`] handle. See
//! the [`AppendWriter`] type docs for the migration.
//!
//! [`AppendWriter`] opens an existing HDF5 file once and appends to a
//! one-dimensional, unlimited, Extensible-Array-indexed dataset *in place*,
//! across many calls, at amortized `O(1)` index cost per append: an appended
//! chunk is written at end-of-file and its record is stored into an element slot
//! of the chunk index, growing the index by appending new blocks only when a
//! block boundary is crossed. It never re-reads the whole file or rebuilds the
//! index, so it is the throughput-oriented counterpart to
//! [`EditSession::append_dataset`](crate::EditSession::append_dataset) (which
//! rebuilds the index and re-reads the file on every commit) and the
//! filter-capable, any-length counterpart to [`SwmrWriter`](crate::SwmrWriter)
//! (which is unfiltered and chunk-aligned only).
//!
//! Unlike the SWMR writer it supports **filtered** (compressed) datasets, and
//! for **unfiltered** datasets it supports **any-length** appends (when the
//! current length is not a whole multiple of the chunk length, the trailing
//! partial chunk is read, extended, rewritten at end-of-file, and its existing
//! index element — a single chunk address — is repointed in place). A *filtered*
//! append must be **chunk-aligned** (see the crash-safety note below);
//! [`EditSession::append_dataset`](crate::EditSession::append_dataset) handles a
//! non-chunk-aligned filtered append instead.
//!
//! # Durability and crash safety
//!
//! Every append this writer performs is crash-atomic. Each append publishes
//! durably before it returns; writes are ordered child-before-parent with `fsync`
//! barriers, and the dataspace dimension is published last as the single commit
//! point, so a crash between appends leaves the file at either the previous
//! length or the new one — never a torn or lost view. New chunks are inserted as
//! index elements the reader ignores until the dimension is committed, and the
//! one case that mutates a *visible* element — repointing an unfiltered dataset's
//! trailing chunk when growing a partial tail — overwrites a single chunk address
//! (an atomic write) that points only at data whose live prefix reproduces the
//! previous view's bytes. The relocated chunk's old bytes are abandoned as dead
//! space (reclaimed by [repack](crate::repack), not reused within the
//! session in this release).
//!
//! A *filtered* dataset's index element is a multi-field record
//! (address + compressed size + filter mask); repointing it in place to grow a
//! partial tail is **not** power-loss atomic (a torn write across a disk sector
//! could leave the committed view unreadable), so a filtered append is required
//! to be chunk-aligned — the current length and the appended length must both be
//! whole multiples of the chunk length — which means a filtered append only ever
//! inserts new, not-yet-visible elements. Use
//! [`EditSession::append_dataset`](crate::EditSession::append_dataset) for a
//! non-chunk-aligned filtered append; it rebuilds the index and repoints the
//! superblock last, which is fully atomic.
//!
//! Like [`SwmrWriter`](crate::SwmrWriter), the writer keeps a full in-memory mirror of the file
//! (`O(file size)` memory). Unlike it, [`AppendWriter`] takes an **exclusive OS
//! file lock** for the session's life (it has no concurrent-reader contract to
//! uphold and should keep other writers out) and sets **no** SWMR flag.

use std::collections::HashMap;
use std::path::Path;

use crate::chunk_index_inplace::{InPlaceFile, Located, apply_ea_append, plan_ea_append};
use crate::datatype::Datatype;
use crate::edit::{AppendBuilder, datatype_is_raw_appendable, pipeline_reencodable};
use crate::element::H5Element;
use crate::error::Error;
use crate::file_lock::FileLocking;
use crate::filter_pipeline::FilterPipeline;

/// Per-dataset state located once, then maintained across appends.
struct DatasetState {
    loc: Located,
    /// The dataset's on-disk element datatype (for the append type check and the
    /// filter chunk context).
    datatype: Datatype,
    /// Spatial (rank-length) chunk dimensions in elements: `[chunk_elems]`.
    spatial: Vec<u64>,
    /// Bytes per element (datatype size).
    element_size: usize,
    /// The re-encodable filter pipeline, when the dataset is filtered.
    pipeline: Option<FilterPipeline>,
}

/// A throughput-oriented in-place append writer for a chunked, unlimited dataset.
///
/// [`AppendWriter`] opens an existing HDF5 file once and appends to a
/// one-dimensional (rank 1), unlimited, Extensible-Array-indexed dataset *in
/// place*, across many calls, at amortized `O(1)` index cost per append. It
/// never re-reads the whole file or rebuilds the chunk index the way
/// [`EditSession::append_dataset`](crate::EditSession::append_dataset) does on
/// every commit, so it is the path for a high-frequency append loop. It takes an
/// **exclusive OS file lock** for the writer's lifetime and sets **no** SWMR
/// flag; when a separate process must read a dataset while it grows, reach for
/// [`SwmrWriter`](crate::SwmrWriter) instead.
///
/// # Length rules
///
/// - **Unfiltered** datasets accept **any-length** appends: a partial trailing
///   chunk is rewritten and its index element (a single chunk address) is
///   repointed with one atomic write.
/// - **Filtered** datasets (deflate, shuffle, fletcher32, scale-offset, and ZFP
///   with the `zfp` feature) accept **whole-chunk** appends only — the current
///   length and the appended length must both be whole multiples of the chunk
///   length. A non-chunk-aligned filtered append is refused: a filtered index
///   element is a multi-field record (address, compressed size, filter mask)
///   whose in-place repoint is not power-loss atomic. Use
///   [`EditSession::append_dataset`](crate::EditSession::append_dataset), which
///   rebuilds the index and repoints the superblock last, for that case.
///
/// # Durability
///
/// Every append is crash-atomic: the dataspace dimension is published last as
/// the single commit point, so a crash between appends leaves the file at either
/// the previous length or the new one, never a torn or lost view. The result
/// reads back in the reference HDF5 C library and h5py.
///
/// Requires the `std` feature (a filesystem); it is not available on the
/// in-memory / `no_std` / WASM path.
///
/// # Errors
///
/// Appending returns [`Error::AppendUnsupported`](crate::Error::AppendUnsupported)
/// when:
///
/// - the target dataset is not chunked, not rank 1, not unlimited, or not
///   Extensible-Array-indexed;
/// - its Extensible-Array index is not yet allocated — an empty dataset the C
///   library created with no initial data. Make the first append with
///   [`EditSession::append_dataset`](crate::EditSession::append_dataset) (which
///   materializes the index), or create the dataset with initial data (this
///   crate allocates the index eagerly, so a dataset it wrote can be grown from
///   the first append);
/// - a filtered append is not chunk-aligned (see [Length rules](#length-rules));
/// - the append datatype does not match the dataset (including mixing element
///   types within one call), or [`append_raw`](Self::append_raw) targets a
///   non-little-endian, variable-length, or reference datatype.
///
/// Eligibility is validated on the first append to a dataset, not at
/// [`open`](Self::open).
///
/// # Deprecated
///
/// Superseded by the owned-handle API: open the file for writing with
/// [`File::open_rw`](crate::File::open_rw) and append through a
/// [`Dataset`](crate::Dataset) handle. That path gives the same amortized
/// `O(1)` in-place append under the same contract (filtered whole-chunk /
/// unfiltered any-length, an exclusive OS lock, no SWMR flag), and the one open
/// file also reads, edits, and reaches every dataset by name — no separate
/// writer type.
///
/// ```no_run
/// use hdf5_pure::File;
///
/// let file = File::open_rw("log.h5")?;
/// let mut samples = file.dataset("samples")?;
/// samples.append(&[8i32, 9, 10, 11])?;
/// samples.append(&[12i32, 13])?; // unfiltered: any length
/// file.close()?;
/// # Ok::<(), hdf5_pure::Error>(())
/// ```
///
/// The one behavior with no [`File::open_rw`](crate::File::open_rw) equivalent yet is
/// [`open_with_locking`](Self::open_with_locking) with
/// [`FileLocking::Disabled`](crate::FileLocking) — keep using `AppendWriter` if
/// you must open without the exclusive lock. The type will be removed in a later
/// release.
#[deprecated(
    since = "0.22.0",
    note = "use File::open_rw + Dataset::append; see the AppendWriter type docs for migration"
)]
pub struct AppendWriter {
    /// In-memory mirror + on-disk handle, shared with the SWMR writer.
    file: InPlaceFile,
    datasets: HashMap<String, DatasetState>,
}

#[allow(deprecated)] // this type's own impl legitimately touches its (deprecated) fields
impl AppendWriter {
    /// Open an existing HDF5 file for in-place appends, taking an exclusive OS
    /// file lock held for the writer's life.
    ///
    /// A target dataset's append eligibility (see [`AppendWriter`]) is validated
    /// on its first append, not here. Returns
    /// [`Error::AppendUnsupported`](crate::Error::AppendUnsupported) if the file
    /// cannot be opened, is not a latest-format HDF5 file, or the exclusive lock
    /// cannot be taken.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        Self::open_with_locking(path, FileLocking::Enabled)
    }

    /// Open with an explicit file-locking policy. Use [`FileLocking::Disabled`]
    /// only when an external mechanism already guarantees single-writer access.
    pub fn open_with_locking<P: AsRef<Path>>(path: P, locking: FileLocking) -> Result<Self, Error> {
        let file = InPlaceFile::open(path, Some(locking), Error::AppendUnsupported)?;
        Ok(Self {
            file,
            datasets: HashMap::new(),
        })
    }

    /// Append raw little-endian element bytes to `dataset`. The concatenated
    /// length must be a whole multiple of the dataset's on-disk element size, and
    /// the dataset's datatype must be little-endian (and neither variable-length
    /// nor a reference). Prefer the typed methods when the element type is known.
    ///
    /// A filtered dataset accepts only chunk-aligned appends; see
    /// [`AppendWriter`] for the full contract and the refusal cases.
    pub fn append_raw(&mut self, dataset: &str, bytes: &[u8]) -> Result<(), Error> {
        let mut b = AppendBuilder::new();
        b.append_raw(bytes);
        self.append_gathered(dataset, &b, 4)
    }

    /// Generic append of a flat slice of any supported scalar type. A filtered
    /// dataset accepts only chunk-aligned appends; see [`AppendWriter`] for the
    /// full contract and the refusal cases.
    pub fn append<T: H5Element>(&mut self, dataset: &str, data: &[T]) -> Result<(), Error> {
        let mut b = AppendBuilder::new();
        b.append(data);
        self.append_gathered(dataset, &b, 4)
    }

    /// Flush to durable storage and release the file lock. Each append is already
    /// durable, so this is a final barrier; the lock also releases when the
    /// writer is dropped.
    pub fn close(mut self) -> Result<(), Error> {
        self.file.sync()
    }

    /// Ensure `dataset` is located and cached, validating append eligibility.
    fn ensure_located(&mut self, dataset: &str) -> Result<(), Error> {
        if self.datasets.contains_key(dataset) {
            return Ok(());
        }
        let result = Located::locate(&self.file, dataset, Error::AppendUnsupported)?;
        if result.located.chunk_elems == 0 {
            return Err(Error::AppendUnsupported(
                "append requires a nonzero chunk length",
            ));
        }
        let data = self.file.data();
        let (dt_off, dt_size) = result.spans.datatype;
        let (datatype, _) = Datatype::parse(&data[dt_off..dt_off + dt_size])
            .map_err(|_| Error::AppendUnsupported("dataset datatype could not be parsed"))?;
        let pipeline = match result.spans.filter {
            Some((fb, fsize)) => {
                let parsed = FilterPipeline::parse(&data[fb..fb + fsize]).map_err(|_| {
                    Error::AppendUnsupported("dataset filter pipeline could not be parsed")
                })?;
                if !pipeline_reencodable(&parsed) {
                    return Err(Error::AppendUnsupported(
                        "dataset uses a filter this engine cannot re-encode",
                    ));
                }
                Some(parsed)
            }
            None => None,
        };
        let element_size = result.located.elem_bytes;
        let spatial = vec![result.located.chunk_elems];
        self.datasets.insert(
            dataset.to_string(),
            DatasetState {
                loc: result.located,
                datatype,
                spatial,
                element_size,
                pipeline,
            },
        );
        Ok(())
    }

    /// Apply a gathered append (typed/generic/raw bytes) to `dataset` in place,
    /// running only the first `max_phase` durability phases (1-4). Production
    /// callers always pass `max_phase = 4`; the crash-consistency tests stop at a
    /// phase boundary to simulate a crash.
    fn append_gathered(
        &mut self,
        dataset: &str,
        b: &AppendBuilder,
        max_phase: u8,
    ) -> Result<(), Error> {
        if b.dt_conflict() {
            return Err(Error::AppendUnsupported(
                "append mixes element types in one call; use one element type per append",
            ));
        }
        self.ensure_located(dataset)?;

        // Validate the appended bytes against the on-disk datatype.
        let raw = b.raw();
        {
            let st = &self.datasets[dataset];
            if raw.len() % st.element_size != 0 {
                return Err(Error::AppendUnsupported(
                    "appended byte length is not a whole number of elements",
                ));
            }
            match b.elem_dt() {
                Some(expected) if *expected != st.datatype => {
                    return Err(Error::AppendUnsupported(
                        "append datatype does not match the on-disk dataset (wrong element \
                         type or byte order)",
                    ));
                }
                Some(_) => {}
                None => {
                    if !datatype_is_raw_appendable(&st.datatype) {
                        return Err(Error::AppendUnsupported(
                            "append_raw onto this dataset's datatype (non-little-endian, \
                             variable-length, or reference) could misencode the bytes; use a \
                             typed append",
                        ));
                    }
                }
            }
        }

        let new_elems = (raw.len() / self.datasets[dataset].element_size) as u64;
        if new_elems == 0 {
            return Ok(());
        }

        // Read/plan phase (immutable borrows only, nothing published yet), then
        // the ordered, fsync-barriered write phase. Both are shared with
        // `EditSession::append_inplace` through the chunk-index engine, so the
        // crash-safety sequence lives in exactly one place. The relocated trailing
        // chunk's old bytes become dead space (reclaimed by repack, not reused
        // in-session in this release).
        let plan = {
            let st = &self.datasets[dataset];
            plan_ea_append(
                &self.file,
                &st.loc,
                &st.datatype,
                &st.spatial,
                st.element_size,
                st.pipeline.as_ref(),
                raw,
                new_elems,
            )?
        };
        let st = self
            .datasets
            .get_mut(dataset)
            .expect("dataset was located above");
        apply_ea_append(&mut self.file, &mut st.loc, &plan, max_phase)
    }

    /// Test-only phased append (stops after `max_phase` durability phases) used by
    /// the crash-consistency tests.
    #[cfg(test)]
    fn append_i32_phased(
        &mut self,
        dataset: &str,
        values: &[i32],
        max_phase: u8,
    ) -> Result<(), Error> {
        let mut b = AppendBuilder::new();
        b.append_i32(values);
        self.append_gathered(dataset, &b, max_phase)
    }
}

/// Generate the typed `append_*` methods, mirroring [`AppendBuilder`]'s
/// vocabulary: each gathers into a builder and applies the append in place.
macro_rules! append_typed {
    ($($method:ident, $ty:ty;)*) => {
        #[allow(deprecated)] // impl of the deprecated type; each method delegates to append_gathered
        impl AppendWriter {
            $(
                #[doc = concat!("Append `", stringify!($ty), "` values to `dataset`. \
                    A filtered dataset accepts only chunk-aligned appends; see \
                    [`AppendWriter`] for the full contract.")]
                pub fn $method(&mut self, dataset: &str, data: &[$ty]) -> Result<(), Error> {
                    let mut b = AppendBuilder::new();
                    b.$method(data);
                    self.append_gathered(dataset, &b, 4)
                }
            )*
        }
    };
}

append_typed! {
    append_f64, f64;
    append_f32, f32;
    append_i8, i8;
    append_i16, i16;
    append_i32, i32;
    append_i64, i64;
    append_u8, u8;
    append_u16, u16;
    append_u32, u32;
    append_u64, u64;
}

#[cfg(test)]
#[allow(deprecated)] // exercises the deprecated shim's shared crash-safety engine
mod tests {
    use super::*;
    use crate::reader::File as PureFile;
    use crate::writer::FileBuilder;
    use tempfile::tempdir;

    /// Build an unfiltered rank-1 unlimited i32 dataset `d` seeded with `0..n`
    /// and the given chunk length. (A filtered dataset cannot be grown in place
    /// from a partial tail — that is refused — so the in-place trailing-element
    /// repoint is only ever exercised on unfiltered datasets, where the element
    /// is a single address whose overwrite is atomic.)
    fn build_unfiltered(path: &std::path::Path, n: i32, chunk: u64) {
        let data: Vec<i32> = (0..n).collect();
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&data)
            .with_shape(&[n as u64])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[chunk]);
        b.write(path).unwrap();
    }

    /// Stopping a partial-tail append at any phase boundary must leave the file
    /// readable as a consistent prefix — the old length until the phase-4
    /// dimension commit, the new length after it — even though the append
    /// *repoints the visible trailing element in place*. Two starting layouts:
    /// the trailing element inline in the index block, and in a data block (slot
    /// 0, exercising the reuse-existing-block path).
    #[test]
    fn crash_consistency_partial_tail_prefix() {
        // (n, chunk, append_len): n%chunk != 0 so a partial tail is relocated.
        // chunk 4, n 6  -> trailing element index 1 (inline).
        // chunk 2, n 9  -> trailing element index 4 (data-block slot 0).
        for (n, chunk, add) in [(6i32, 4u64, 5i32), (9, 2, 6)] {
            let dir = tempdir().unwrap();
            let base = dir.path().join("base.h5");
            build_unfiltered(&base, n, chunk);

            for max_phase in 1u8..=4 {
                let p = dir.path().join(format!("crash_{n}_{chunk}_{max_phase}.h5"));
                std::fs::copy(&base, &p).unwrap();
                {
                    let mut w = AppendWriter::open(&p).unwrap();
                    w.append_i32_phased("d", &(n..n + add).collect::<Vec<_>>(), max_phase)
                        .unwrap();
                    // dropped here, simulating a crash after `max_phase`
                }
                let expected_len = if max_phase == 4 { n + add } else { n };
                let f = PureFile::from_bytes(std::fs::read(&p).unwrap()).unwrap();
                assert_eq!(
                    f.dataset("d").unwrap().read_i32().unwrap(),
                    (0..expected_len).collect::<Vec<_>>(),
                    "inconsistent view after crash at phase {max_phase} (n={n}, chunk={chunk})"
                );
            }
        }
    }

    /// The same consistent-prefix guarantee must hold for the reference C
    /// library, which walks strictly by the dataspace dimension and re-validates
    /// every touched block checksum — so a mis-checksummed index block or a
    /// visible element pointing past the recorded EOF at an intermediate phase
    /// would break C even where the pure reader tolerates it.
    #[test]
    #[cfg(not(target_pointer_width = "32"))]
    fn crash_consistency_c_reads_partial_tail_prefix() {
        let dir = tempdir().unwrap();
        let base = dir.path().join("base.h5");
        let n = 6i32;
        let add = 5i32; // 6 -> 11, chunk 4: relocate the partial tail, insert two
        build_unfiltered(&base, n, 4);

        for max_phase in 1u8..=4 {
            let p = dir.path().join(format!("crash_c_{max_phase}.h5"));
            std::fs::copy(&base, &p).unwrap();
            {
                let mut w = AppendWriter::open(&p).unwrap();
                w.append_i32_phased("d", &(n..n + add).collect::<Vec<_>>(), max_phase)
                    .unwrap();
            }
            let expected_len = if max_phase == 4 { n + add } else { n };
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
}
