//! General (non-SWMR) in-place append writer.
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

use crate::chunk_index_inplace::{ElemRecord, InPlaceFile, Located};
use crate::chunked_write::split_into_chunks;
use crate::convert::TryToUsize;
use crate::datatype::Datatype;
use crate::edit::{AppendBuilder, datatype_is_raw_appendable, pipeline_reencodable};
use crate::element::H5Element;
use crate::error::Error;
use crate::file_lock::FileLocking;
use crate::filter_pipeline::FilterPipeline;
use crate::filters::{ChunkContext, compress_chunk, decompress_chunk};

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
/// # Example
///
/// ```no_run
/// use hdf5_pure::AppendWriter;
///
/// let mut writer = AppendWriter::open("log.h5")?;
/// writer.append_i32("samples", &[8, 9, 10, 11])?;
/// writer.append_i32("samples", &[12, 13])?; // unfiltered: any length
/// writer.close()?;
/// # Ok::<(), hdf5_pure::Error>(())
/// ```
pub struct AppendWriter {
    /// In-memory mirror + on-disk handle, shared with the SWMR writer.
    file: InPlaceFile,
    datasets: HashMap<String, DatasetState>,
}

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

        // ---- Read phase: compute the new chunk blobs and the dead tail extent
        // using only immutable borrows, so nothing is published yet.
        let plan = self.plan_append(dataset, raw, new_elems)?;

        // ---- Write phase, ordered child-before-parent with fsync barriers, so a
        // crash between appends leaves either the old length or the new one:
        //
        //   1. new/relocated chunk bytes at EOF, then advance the superblock's
        //      recorded end-of-file to cover them. This must precede the index
        //      writes: the trailing partial chunk's element is *visible* at the
        //      old dimension, so once it is repointed to the relocated chunk that
        //      chunk must already lie within the recorded EOF (unlike the SWMR
        //      writer, which only ever inserts elements the reader clamps away).
        //   2. the index element writes: a fresh insert for each new chunk, or an
        //      in-place repoint of the trailing element (which only ever points at
        //      data whose live prefix reproduces the old view's bytes). This may
        //      allocate new EA blocks past EOF, so advance the EOF again.
        //   3. the EA header element count; and
        //   4. the dataspace dimension -- the single commit point.
        let mut chunk_addrs = Vec::with_capacity(plan.new_chunk_bytes.len());
        for blob in &plan.new_chunk_bytes {
            chunk_addrs.push((self.file.append_bytes(blob)?, blob.len() as u64));
        }
        self.file.sync()?;
        self.file.patch_superblock_eof()?;
        self.file.sync()?;
        if max_phase < 2 {
            return Ok(());
        }

        for (k, &(addr, stored_size)) in chunk_addrs.iter().enumerate() {
            let e = plan.n_full + k as u64;
            let rec = ElemRecord {
                addr,
                stored_size,
                filter_mask: 0,
            };
            self.datasets[dataset]
                .loc
                .ea_insert(&mut self.file, e, rec)?;
        }
        self.file.sync()?;
        if max_phase < 3 {
            return Ok(());
        }

        // Cover any EA blocks allocated during the element writes, then publish
        // the element count.
        self.file.patch_superblock_eof()?;
        self.datasets[dataset]
            .loc
            .update_ea_header(&mut self.file, plan.new_num_chunks)?;
        self.file.sync()?;
        if max_phase < 4 {
            return Ok(());
        }

        self.datasets[dataset]
            .loc
            .patch_dimension(&mut self.file, plan.new_dim)?;
        self.file.sync()?;

        // The relocated trailing chunk's old bytes are now dead (reclaimed by
        // repack, not reused in-session in this release).

        let st = self
            .datasets
            .get_mut(dataset)
            .expect("dataset was located above");
        st.loc.current_dim = plan.new_dim;
        st.loc.num_chunks = plan.new_num_chunks;
        Ok(())
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

    /// Compute, without mutating the file, the new chunk blobs to write, the
    /// first element index they occupy, and the new dimension/chunk count.
    fn plan_append(&self, dataset: &str, raw: &[u8], new_elems: u64) -> Result<AppendPlan, Error> {
        let st = &self.datasets[dataset];
        let chunk_elems = st.loc.chunk_elems;
        let element_size = st.element_size;
        let current_dim = st.loc.current_dim;
        let new_dim = current_dim
            .checked_add(new_elems)
            .ok_or(Error::AppendUnsupported(
                "append would overflow the dataset dimension",
            ))?;
        let n_full = current_dim / chunk_elems;
        let has_partial = current_dim % chunk_elems != 0;

        // Filtered appends must be chunk-aligned. Growing a *filtered* partial
        // trailing chunk would repoint that chunk's existing index element in
        // place, and a filtered element is a multi-field record
        // (address + compressed_size + filter_mask) that is visible at the old
        // dimension before the commit — so a power-loss crash tearing that record
        // across a disk sector could leave the committed view unreadable. Refuse
        // rather than weaken the all-or-nothing guarantee; the trailing element of
        // an *unfiltered* dataset is a single address whose overwrite is atomic,
        // so any-length unfiltered appends are allowed. For a filtered, non-
        // chunk-aligned append use `EditSession::append_dataset`, which rebuilds
        // the index and repoints the superblock last (fully atomic).
        if st.pipeline.is_some() && (has_partial || new_elems % chunk_elems != 0) {
            return Err(Error::AppendUnsupported(
                "a filtered dataset can only be appended in place in whole chunks (the current \
                 length and the appended length must both be multiples of the chunk length); \
                 use EditSession::append_dataset for a non-chunk-aligned filtered append",
            ));
        }

        // Build the raw tail region: the live prefix of any rewritten partial
        // chunk, then the appended bytes.
        let mut tail_raw: Vec<u8> = Vec::new();
        if has_partial {
            let rec = st
                .loc
                .read_element(&self.file, n_full)?
                .ok_or(Error::AppendUnsupported(
                    "trailing partial chunk is missing from the index",
                ))?;
            let start = usize::try_from(rec.addr)
                .map_err(|_| Error::AppendUnsupported("chunk address exceeds this platform"))?;
            let stored_len = if st.pipeline.is_some() {
                usize::try_from(rec.stored_size)
                    .map_err(|_| Error::AppendUnsupported("chunk size exceeds this platform"))?
            } else {
                chunk_elems.to_usize()? * element_size
            };
            let data = self.file.data();
            let end = start
                .checked_add(stored_len)
                .filter(|&e| e <= data.len())
                .ok_or(Error::AppendUnsupported(
                    "trailing chunk extends past end-of-file",
                ))?;
            let stored = &data[start..end];
            let full = if let Some(pl) = &st.pipeline {
                let ctx = ChunkContext::from_datatype(&st.spatial, &st.datatype);
                decompress_chunk(stored, pl, ctx, rec.filter_mask).map_err(Error::Format)?
            } else {
                stored.to_vec()
            };
            let live_elems = usize::try_from(current_dim % chunk_elems)
                .map_err(|_| Error::AppendUnsupported("chunk length exceeds this platform"))?;
            let live_bytes = live_elems * element_size;
            if full.len() < live_bytes {
                return Err(Error::AppendUnsupported(
                    "trailing chunk decoded shorter than its live element count",
                ));
            }
            tail_raw.extend_from_slice(&full[..live_bytes]);
        }
        tail_raw.extend_from_slice(raw);

        // Split the tail into full chunk buffers (edge overhang zero-filled) and
        // compress each through the pipeline when filtered.
        let tail_len_elems = new_dim - n_full * chunk_elems;
        let split = split_into_chunks(&tail_raw, &[tail_len_elems], &st.spatial, element_size);
        let new_chunk_bytes: Vec<Vec<u8>> = if let Some(pl) = &st.pipeline {
            let ctx = ChunkContext::from_datatype(&st.spatial, &st.datatype);
            let mut out = Vec::with_capacity(split.len());
            for (_, buf) in &split {
                out.push(compress_chunk(buf, pl, ctx).map_err(Error::Format)?);
            }
            out
        } else {
            split.into_iter().map(|(_, buf)| buf).collect()
        };

        let new_num_chunks = n_full + new_chunk_bytes.len() as u64;
        Ok(AppendPlan {
            new_chunk_bytes,
            n_full,
            new_dim,
            new_num_chunks,
        })
    }
}

/// The mutation-free result of planning one append.
struct AppendPlan {
    /// Per-chunk (possibly compressed) bytes to write, in element order starting
    /// at `n_full`. The first entry is the rewritten trailing chunk when the
    /// current length was not chunk-aligned.
    new_chunk_bytes: Vec<Vec<u8>>,
    /// Element index the first new chunk occupies.
    n_full: u64,
    /// New axis-0 dimension after the append (the commit value).
    new_dim: u64,
    /// New number of indexed chunks.
    new_num_chunks: u64,
}

/// Generate the typed `append_*` methods, mirroring [`AppendBuilder`]'s
/// vocabulary: each gathers into a builder and applies the append in place.
macro_rules! append_typed {
    ($($method:ident, $ty:ty;)*) => {
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
