//! Bounded-memory read-write backend (issue #147).
//!
//! [`File::open_rw_bounded`](crate::File::open_rw_bounded) opens a file for
//! reading and immediate in-place appending **without a whole-file mirror**:
//! where [`File::open_rw`](crate::File::open_rw) loads the entire file into
//! memory, this backend keeps only the superblock, an end-of-file cursor, the
//! per-dataset append geometry ([`LocatedState`]), and the configured caches.
//! Reads are served by positioned I/O like
//! [`File::open_streaming`](crate::File::open_streaming); appends run the same
//! crash-atomic Extensible-Array engine as `open_rw`
//! ([`plan_ea_append`]/[`apply_ea_append`]) over bounded [`Source`] windows,
//! and large appends are split into whole-chunk batches so peak memory stays a
//! few chunks regardless of call size.
//!
//! Peak resident memory is bounded by: the metadata being parsed + the
//! metadata-cache budget + the chunk-cache budget + one append batch
//! ([`APPEND_BATCH_BYTES`] plus the trailing chunk) — independent of both the
//! file size and the bytes appended per call.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::chunk_index_inplace::{Store, apply_ea_append, plan_ea_append};
use crate::edit::{
    AppendBuilder, LocatedState, as_inplace_error, locate_dataset_state, validate_gathered_append,
};
use crate::error::{Error, FormatError};
use crate::file_lock::{self, FileLocking};
use crate::file_space_info::FileSpaceInfo;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::signature;
use crate::source::{MetadataCacheConfig, MetadataReadCache, Source};
use crate::superblock::Superblock;

/// Byte budget for one append batch: a large append is split into whole-chunk
/// batches of at most this many raw bytes (always at least one chunk), each
/// applied as its own crash-atomic fsync-barriered sequence, so peak append
/// memory never scales with the caller's slice. A crash between batches leaves
/// a valid shorter dataset — exactly as if the caller had looped.
const APPEND_BATCH_BYTES: u64 = 1 << 20;

/// Read exactly `buf.len()` bytes at `offset` from a shared file handle,
/// bounds-checked against `len` (mirroring `ReadSeekSource`). Uses the
/// `Read`/`Seek` impls on `&std::fs::File`; callers serialize access through
/// the backend's engine lock, so the shared cursor is never raced.
fn read_at_handle(
    handle: &std::fs::File,
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

use crate::convert::TryToUsize;

/// A minimal [`Source`] over a raw handle, used during open before the
/// [`BoundedStore`] exists (signature scan, superblock, extension probe).
struct RawSource<'a> {
    handle: &'a std::fs::File,
    len: u64,
}

impl Source for RawSource<'_> {
    fn len(&self) -> u64 {
        self.len
    }
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        read_at_handle(self.handle, self.len, offset, buf)
    }
}

/// The bounded backend's [`Store`]: a read-write handle (holding the exclusive
/// OS lock), the parsed superblock, an explicit end-of-file cursor (the mirror
/// backends derive it from their `Vec` length), and an optional bounded
/// metadata cache whose entries are invalidated by overlapping writes.
pub(crate) struct BoundedStore {
    handle: std::fs::File,
    /// Logical end-of-file: the real file length at open, advanced by appends.
    len: u64,
    sb_sig_off: u64,
    superblock: Superblock,
    metadata_cache: Option<(MetadataCacheConfig, std::sync::Mutex<MetadataReadCache>)>,
}

impl BoundedStore {
    pub(crate) fn superblock(&self) -> &Superblock {
        &self.superblock
    }

    fn write_at_raw(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
        // Writes never extend the file here (appends go through
        // `append_bytes`), so an out-of-range patch is an engine invariant
        // violation surfaced as a clean error rather than silent growth.
        let end = offset
            .checked_add(bytes.len() as u64)
            .filter(|&e| e <= self.len)
            .ok_or(Error::Format(FormatError::UnexpectedEof {
                expected: offset.to_usize().unwrap_or(usize::MAX),
                available: self.len.to_usize().unwrap_or(usize::MAX),
            }))?;
        debug_assert!(end <= self.len);
        if let Some((_, cache)) = &self.metadata_cache {
            cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .invalidate_overlapping(offset, bytes.len());
        }
        self.handle
            .seek(SeekFrom::Start(offset))
            .map_err(Error::Io)?;
        self.handle.write_all(bytes).map_err(Error::Io)?;
        Ok(())
    }
}

impl Source for BoundedStore {
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

impl Store for BoundedStore {
    fn offset_size(&self) -> u8 {
        self.superblock.offset_size
    }
    fn length_size(&self) -> u8 {
        self.superblock.length_size
    }
    fn append_bytes(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        let addr = self.len;
        self.handle.seek(SeekFrom::Start(addr)).map_err(Error::Io)?;
        self.handle.write_all(bytes).map_err(Error::Io)?;
        self.len += bytes.len() as u64;
        Ok(addr)
    }
    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
        self.write_at_raw(offset, bytes)
    }
    fn patch_superblock_eof(&mut self) -> Result<(), Error> {
        self.superblock.eof_address = self.len;
        let bytes = self.superblock.serialize();
        self.write_at_raw(self.sb_sig_off, &bytes)
    }
    fn sync(&mut self) -> Result<(), Error> {
        self.handle.flush().map_err(Error::Io)?;
        self.handle.sync_data().map_err(Error::Io)?;
        Ok(())
    }
}

/// The engine behind [`Backend::Bounded`](crate::reader): the store plus the
/// object-header-address-keyed append geometry cache (the same shape as
/// `WriteEngine::located`; two hard links to one dataset share one entry).
/// Reads and writes are serialized by the backend's `Mutex`, exactly like the
/// mirror backend.
pub(crate) struct BoundedEngine {
    store: BoundedStore,
    located: HashMap<u64, LocatedState>,
}

impl BoundedEngine {
    /// Open `path` read-write with bounded memory: exclusive OS lock, bounded
    /// superblock discovery, and the same eligibility rules the immediate
    /// in-place append enforces — refused up front (at open) because this
    /// backend has no staged fallback: a latest-format (v2/v3) superblock,
    /// 8-byte offsets and lengths, no userblock, and no persisted free-space
    /// managers.
    pub(crate) fn open(path: &Path, metadata_cache: MetadataCacheConfig) -> Result<Self, Error> {
        let handle = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(Error::Io)?;
        file_lock::acquire_exclusive(&handle, FileLocking::Enabled, path)?;
        let len = handle.metadata().map_err(Error::Io)?.len();

        let raw = RawSource {
            handle: &handle,
            len,
        };
        let sb_sig_off = signature::find_signature_in(&raw)?;
        let superblock = Superblock::parse_from_source(&raw, sb_sig_off)?;
        if superblock.version < 2 {
            return Err(Error::EditUnsupported(
                "bounded read-write access requires a latest-format file (v2/v3 superblock); \
                 use File::open_rw",
            ));
        }
        if superblock.offset_size != 8 || superblock.length_size != 8 {
            return Err(Error::EditUnsupported(
                "bounded read-write access requires 8-byte offsets and lengths",
            ));
        }
        if superblock.base_address != 0 || sb_sig_off != 0 {
            return Err(Error::EditUnsupported(
                "bounded read-write access does not support a file with a userblock \
                 (non-zero base address); use File::open_rw",
            ));
        }
        if persisted_free_space_armed(&raw, &superblock) {
            // The mirror backend's staged commit rewrites persisted free-space
            // managers consistently; this backend has no staged path, so an EOF
            // append would leave the on-disk managers stale. Refuse at open
            // (conservatively: even when the manager blocks themselves would
            // fail to load) rather than corrupt them.
            return Err(Error::EditUnsupported(
                "bounded read-write access does not support a file that persists its \
                 free space (H5Pset_file_space_strategy persist=true); use File::open_rw",
            ));
        }

        Ok(Self {
            store: BoundedStore {
                handle,
                len,
                sb_sig_off,
                superblock,
                metadata_cache: metadata_cache.is_enabled().then(|| {
                    (
                        metadata_cache,
                        std::sync::Mutex::new(MetadataReadCache::new()),
                    )
                }),
            },
            located: HashMap::new(),
        })
    }

    pub(crate) fn store(&self) -> &BoundedStore {
        &self.store
    }

    /// Flush buffered writes durably (each append is already durable; this is a
    /// final barrier for [`File::close`](crate::File::close)).
    pub(crate) fn sync(&mut self) -> Result<(), Error> {
        Store::sync(&mut self.store)
    }

    /// Immediate in-place append of a gathered builder to the dataset whose
    /// object header sits at `oh_addr` — the bounded counterpart of
    /// `WriteEngine::append_inplace_gathered`, sharing its locate, validation,
    /// and plan/apply engine, with the whole-file guards (userblock, pre-v2,
    /// persisted free space) already enforced at open.
    ///
    /// A large append is split into whole-chunk batches of at most
    /// [`APPEND_BATCH_BYTES`] raw bytes, each its own crash-atomic apply, so
    /// peak memory is independent of the call size. `max_phase` (production: 4)
    /// stops the *first* batch at a durability phase boundary for the
    /// crash-consistency tests.
    pub(crate) fn append_gathered(
        &mut self,
        oh_addr: u64,
        b: &AppendBuilder,
        max_phase: u8,
    ) -> Result<(), Error> {
        if b.dt_conflict() {
            return Err(Error::AppendInPlaceUnsupported(
                "append mixes element types in one call; use one element type per append",
            ));
        }
        let Self { store, located } = self;
        let st = match located.entry(oh_addr) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(locate_dataset_state(&*store, oh_addr)?)
            }
        };
        let new_elems = validate_gathered_append(st, b)?;
        if new_elems == 0 {
            return Ok(());
        }
        let raw = b.raw();

        let chunk_elems = st.loc.chunk_elems.max(1);
        let elem_bytes = st.element_size as u64;
        let batch_chunks = (APPEND_BATCH_BYTES / (st.loc.chunk_bytes.max(1) as u64)).max(1);
        let full_batch_elems = batch_chunks * chunk_elems;

        let mut done = 0u64;
        while done < new_elems {
            // Fill the trailing partial chunk first (so later batches start
            // chunk-aligned and never rewrite it again), then whole-chunk
            // batches. Filtered datasets are chunk-aligned by contract, so
            // every batch stays chunk-aligned there too.
            let to_boundary = (chunk_elems - st.loc.current_dim % chunk_elems) % chunk_elems;
            let take = (new_elems - done).min(to_boundary + full_batch_elems);
            let start = (done * elem_bytes).to_usize()?;
            let end = ((done + take) * elem_bytes).to_usize()?;
            let batch = &raw[start..end];

            let plan = plan_ea_append(
                &*store,
                &st.loc,
                &st.datatype,
                &st.spatial,
                st.element_size,
                st.pipeline.as_ref(),
                batch,
                take,
            )
            .map_err(as_inplace_error)?;
            apply_ea_append(store, &mut st.loc, &plan, max_phase).map_err(as_inplace_error)?;
            if max_phase < 4 {
                // Crash-consistency hook: simulate a crash inside the first
                // batch's durability sequence.
                return Ok(());
            }
            done += take;
        }
        Ok(())
    }
}

/// Whether the file records persisted free-space managers (the
/// `H5Pset_file_space_strategy(..., persist = true)` case): a parseable File
/// Space Info message in the superblock extension with its persist flag set.
/// Best-effort on the *extension* itself (a missing or malformed extension
/// reads as non-persisting, matching `WriteEngine`'s loader).
fn persisted_free_space_armed(raw: &RawSource<'_>, superblock: &Superblock) -> bool {
    let Some(rel) = superblock.superblock_extension_address else {
        return false;
    };
    if rel == u64::MAX {
        return false;
    }
    // base_address == 0 is validated before this runs, so `rel` is absolute.
    let Ok(header) = ObjectHeader::parse_from_source(
        raw,
        rel,
        superblock.offset_size,
        superblock.length_size,
        0,
    ) else {
        return false;
    };
    let Some(msg) = header
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::FileSpaceInfo)
    else {
        return false;
    };
    matches!(
        FileSpaceInfo::parse(&msg.data, superblock.offset_size, superblock.length_size),
        Ok(info) if info.persist
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::group_v2;
    use crate::writer::FileBuilder;
    use tempfile::tempdir;

    /// Build a rank-1 unlimited chunked i32 dataset `d` seeded with `0..n`.
    fn build(path: &Path, n: i32, chunk: u64) {
        let data: Vec<i32> = (0..n).collect();
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&data)
            .with_shape(&[n as u64])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[chunk]);
        b.write(path).unwrap();
    }

    fn dataset_addr(engine: &BoundedEngine) -> u64 {
        group_v2::resolve_path_any_from_source(engine.store(), &engine.store().superblock, "d")
            .unwrap()
    }

    /// Crash consistency, mirroring the `WriteEngine` and `AppendWriter`
    /// harnesses: stop the append after only the first `max_phase` durability
    /// phases (simulating a crash at that boundary) and assert the reopened
    /// file reads either the old length (phases 1-3) or the new one (phase 4),
    /// never a torn view. Layouts cover a partial trailing chunk (relocated
    /// tail) and a chunk-aligned start.
    #[test]
    fn append_crash_consistency_partial_tail_prefix() {
        let dir = tempdir().unwrap();
        for (case, (n, chunk, add)) in [(0usize, (6i32, 4u64, 5i32)), (1, (8, 2, 6))] {
            let base = dir.path().join(std::format!("base_{case}.h5"));
            build(&base, n, chunk);
            for max_phase in 1u8..=4 {
                let p = dir.path().join(std::format!("crash_{case}_{max_phase}.h5"));
                std::fs::copy(&base, &p).unwrap();
                {
                    let mut engine =
                        BoundedEngine::open(&p, MetadataCacheConfig::disabled()).unwrap();
                    let addr = dataset_addr(&engine);
                    let mut b = AppendBuilder::new();
                    b.append_i32(&(n..n + add).collect::<Vec<_>>());
                    engine.append_gathered(addr, &b, max_phase).unwrap();
                    // Dropping the engine simulates the crash: no further
                    // phases, no close barrier.
                }
                let expected_len = if max_phase == 4 { n + add } else { n };
                let got = crate::File::open(&p)
                    .unwrap()
                    .dataset("d")
                    .unwrap()
                    .read_i32()
                    .unwrap();
                assert_eq!(
                    got,
                    (0..expected_len).collect::<Vec<_>>(),
                    "case {case} phase {max_phase}"
                );
            }
        }
    }

    /// The batching loop only honors `max_phase < 4` on its first batch, and a
    /// full multi-batch append leaves every batch fully committed: after a
    /// large append the file reads the complete sequence.
    #[test]
    fn multi_batch_append_commits_every_batch() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("multibatch.h5");
        build(&p, 5, 512);
        let total = 700_000i32;
        {
            let mut engine = BoundedEngine::open(&p, MetadataCacheConfig::disabled()).unwrap();
            let addr = dataset_addr(&engine);
            let mut b = AppendBuilder::new();
            b.append_i32(&(5..total).collect::<Vec<_>>());
            engine.append_gathered(addr, &b, 4).unwrap();
        }
        let got = crate::File::open(&p)
            .unwrap()
            .dataset("d")
            .unwrap()
            .read_i32()
            .unwrap();
        assert_eq!(got.len(), total as usize);
        assert!(got.iter().enumerate().all(|(i, &v)| v == i as i32));
    }
}
