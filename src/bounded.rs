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
    AppendBuilder, LocatedState, as_inplace_error, build_v2_object_header, locate_dataset_state,
    read_single_chunk_ext_region, rewrite_extension_region_bytes, validate_gathered_append,
};
use crate::error::{Error, FormatError};
use crate::file_lock::{self, FileLocking};
use crate::file_space_info::{FileSpaceInfo, FileSpaceStrategy};
use crate::free_space::FreeList;
use crate::free_space_manager::{
    FreeSection, SECT_CLASS_SIMPLE, fshd_len, read_persisted_sections_source, serialize_file_fsm,
};
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

/// One dataset's append geometry, handed to the public append path so it can
/// slice a large call into aligned batches without materializing the whole
/// call's bytes first.
pub(crate) struct AppendGeometry {
    /// Elements per chunk along axis 0 (>= 1).
    pub(crate) chunk_elems: u64,
    /// Bytes per on-disk element.
    pub(crate) element_size: usize,
    /// Current length along the unlimited dimension.
    pub(crate) current_dim: u64,
    /// Whether a filter pipeline applies (whole-chunk appends only).
    pub(crate) filtered: bool,
    /// Whole-chunk elements in one full batch (>= one chunk's worth).
    pub(crate) full_batch_elems: u64,
}

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

    /// Append a fresh superblock extension + free-space-manager blocks describing
    /// `sections` at end-of-file, then repoint the superblock (root unchanged, new
    /// EOF and extension address, consistency flags cleared) as the crash-atomic
    /// linearization point. This is the bounded, `Store`-seam mirror of
    /// [`WriteEngine::commit_persisting`](crate::edit): the new blocks sit strictly
    /// past everything live and are unreferenced until the superblock repoint, so a
    /// crash before it leaves the prior file wholly intact. Returns the extents of
    /// the freshly written blocks (the next finalize's `old_blocks`).
    fn write_persist_tail(
        &mut self,
        old_ext_region: &[u8],
        strategy: FileSpaceStrategy,
        threshold: u64,
        page_size: u64,
        sections: &[FreeSection],
    ) -> Result<Vec<(u64, u64)>, Error> {
        let os = self.superblock.offset_size;
        let new_root = self.superblock.root_group_address;
        let ext_addr = self.len;

        // The persist File Space Info message is fixed-size, so the rewritten
        // extension's length is independent of the addresses it will carry: size
        // it with a placeholder to place the FSM blocks that follow it.
        let placeholder =
            FileSpaceInfo::persistent_single_manager(strategy, threshold, page_size, 0, 0);
        let ext_len = build_v2_object_header(&rewrite_extension_region_bytes(
            old_ext_region,
            &placeholder,
        )?)
        .len() as u64;
        let fshd_addr = ext_addr + ext_len;

        let (ext_oh, fsm_blocks, final_eof) = if sections.is_empty() {
            let info = FileSpaceInfo::persistent_empty(strategy, threshold, page_size);
            let ext_oh =
                build_v2_object_header(&rewrite_extension_region_bytes(old_ext_region, &info)?);
            let final_eof = ext_addr + ext_oh.len() as u64;
            (ext_oh, None, final_eof)
        } else {
            let fsse_addr = fshd_addr + fshd_len(os);
            // `eoa_pre_fsm` points at the FSHD (the extension below it persists);
            // see the mirror path for the shrink-and-rebuild convention.
            let eoa_pre_fsm = fshd_addr;
            let info = FileSpaceInfo::persistent_single_manager(
                strategy,
                threshold,
                page_size,
                fshd_addr,
                eoa_pre_fsm,
            );
            let ext_oh =
                build_v2_object_header(&rewrite_extension_region_bytes(old_ext_region, &info)?);
            debug_assert_eq!(
                ext_oh.len() as u64,
                ext_len,
                "extension length must be stable across the placeholder and real messages"
            );
            let (fshd, fsse) =
                serialize_file_fsm(sections, fshd_addr, fsse_addr, os, SECT_CLASS_SIMPLE);
            let final_eof = fsse_addr + fsse.len() as u64;
            (ext_oh, Some((fshd, fsse)), final_eof)
        };

        // Append the extension, then the FSM blocks, at end-of-file. They are
        // unreferenced until the superblock repoint, so a crash here is harmless.
        let written_ext = self.append_bytes(&ext_oh)?;
        debug_assert_eq!(written_ext, ext_addr);
        let mut new_old_blocks = vec![(ext_addr, ext_oh.len() as u64)];
        if let Some((fshd, fsse)) = fsm_blocks {
            let wf = self.append_bytes(&fshd)?;
            debug_assert_eq!(wf, fshd_addr);
            new_old_blocks.push((fshd_addr, fshd.len() as u64));
            let ws = self.append_bytes(&fsse)?;
            new_old_blocks.push((ws, fsse.len() as u64));
        }

        // Barrier, then repoint the superblock (the linearization point), then sync.
        Store::sync(self)?;
        self.repoint_persist_superblock(new_root, final_eof, ext_addr)?;
        Store::sync(self)?;
        Ok(new_old_blocks)
    }

    /// Repoint the superblock at `new_root`/`new_eof`/`new_ext_addr` and clear the
    /// consistency flags, rewriting it in place. Called last by
    /// [`write_persist_tail`](Self::write_persist_tail).
    fn repoint_persist_superblock(
        &mut self,
        new_root: u64,
        new_eof: u64,
        new_ext_addr: u64,
    ) -> Result<(), Error> {
        self.superblock.root_group_address = new_root;
        self.superblock.eof_address = new_eof;
        self.superblock.superblock_extension_address = Some(new_ext_addr);
        self.superblock.consistency_flags = 0;
        // `self.len` already equals `new_eof` (we appended up to it); keep the
        // store's logical length and the superblock in step.
        self.len = new_eof;
        let bytes = self.superblock.serialize();
        self.write_at_raw(self.sb_sig_off, &bytes)
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

/// State for a bounded session on a file that persists its free space (issue
/// #173). Seeded from the on-disk managers at open, carried across appends, and
/// written back into canonical (manager-at-tail) shape at
/// [`finalize_persist`](BoundedEngine::finalize_persist). The bounded backend
/// only ever appends at end-of-file (it never reuses a freed hole), so the
/// seeded holes stay valid untouched throughout the session; finalize only has
/// to re-home the managers past the appended data.
struct BoundedPersistState {
    strategy: FileSpaceStrategy,
    threshold: u64,
    page_size: u64,
    /// The live free list: the on-disk holes seeded at open (never handed out —
    /// the bounded engine appends only), plus, at finalize, the superseded old
    /// manager/extension blocks.
    free: FreeList,
    /// `(addr, len)` of the superblock-extension header and every `FSHD`/`FSSE`
    /// block the *current* on-disk file uses; freed and rewritten by the next
    /// finalize.
    old_blocks: Vec<(u64, u64)>,
    /// Absolute address of the current superblock-extension object header.
    old_ext_addr: u64,
}

/// The engine behind [`Backend::Bounded`](crate::reader): the store plus the
/// object-header-address-keyed append geometry cache (the same shape as
/// `WriteEngine::located`; two hard links to one dataset share one entry).
/// Reads and writes are serialized by the backend's `Mutex`, exactly like the
/// mirror backend.
pub(crate) struct BoundedEngine {
    store: BoundedStore,
    located: HashMap<u64, LocatedState>,
    /// Free-space persistence state when the file was created with
    /// `H5Pset_file_space_strategy(persist = true)`; `None` for a non-persisting
    /// file (the common case).
    persist: Option<BoundedPersistState>,
    /// End-of-file at open. The on-disk free-space managers become stale only
    /// once an append grows the file past them, so `store.len() == original_len`
    /// at [`finalize_persist`](BoundedEngine::finalize_persist) means nothing was
    /// appended above the managers and the rewrite can be skipped (no no-op
    /// growth on an unchanged file).
    original_len: u64,
}

impl BoundedEngine {
    /// Open `path` read-write with bounded memory: exclusive OS lock, bounded
    /// superblock discovery, and the eligibility rules refused up front (at open)
    /// because this backend has no staged fallback: a latest-format (v2/v3)
    /// superblock, 8-byte offsets and lengths, and no userblock. A file that
    /// persists its free space is supported when non-paged (its managers are
    /// seeded here and rewritten at [`finalize_persist`](Self::finalize_persist));
    /// the paged strategy is refused (issue #173).
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
        // A file that persists its free space is now supported (issue #173): seed
        // the free list from its on-disk managers and rewrite them at close. The
        // paged strategy is still refused here (it needs page-aligned allocation,
        // issue #173 Phase 2); `load_bounded_persist` returns that error.
        let persist = load_bounded_persist(&raw, &superblock)?;

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
            persist,
            original_len: len,
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

    /// Rewrite the on-disk free-space managers into canonical (manager-at-tail)
    /// shape for a file that persists its free space (issue #173). A no-op for a
    /// non-persisting file, and skipped when nothing was appended past the
    /// managers (`store.len()` unchanged), so an unchanged file never grows.
    /// Called by [`File::close`](crate::File::close) and, best-effort, by
    /// `FileInner::drop` for a handle dropped without `close`: the bounded backend
    /// appends at end-of-file throughout a session, so once it grows, the old
    /// managers end up mid-file with data after them; this re-homes a fresh
    /// extension + `FSHD`/`FSSE` pair past the appended data and repoints the
    /// superblock last, leaving exactly the shape the reference C library writes
    /// and reads back (verified in the crosscheck).
    ///
    /// If a session that grew the file ends without a `close` or `drop` running
    /// (a true crash — `SIGKILL`, power loss), the managers are left mid-file with
    /// the extension's `eoa_pre_fsm` below the appended data. Every append was
    /// durable and crash-atomic, so no data is lost, and both this crate and the
    /// reference C library reopen the file and read it correctly; but the on-disk
    /// managers are non-canonical until a clean rewrite (this crate re-seeds them
    /// from the still-valid message on the next open; a C-library reopen re-settles
    /// them). Prefer an explicit `close` for the canonical result.
    pub(crate) fn finalize_persist(&mut self) -> Result<(), Error> {
        // Nothing appended past the managers -> they are still canonical; skip the
        // rewrite so an unchanged (or partial-chunk-only) session never grows the
        // file.
        if self.persist.is_none() || self.store.len() == self.original_len {
            return Ok(());
        }
        let Some(ps) = self.persist.take() else {
            return Ok(());
        };
        let BoundedPersistState {
            strategy,
            threshold,
            page_size,
            mut free,
            old_blocks,
            old_ext_addr,
        } = ps;

        // The old extension + manager blocks are superseded by the rewrite, so
        // they join the free list (their space becomes reclaimable).
        for &(a, l) in &old_blocks {
            free.free(a, l);
        }
        let sections: Vec<FreeSection> = free
            .sections()
            .into_iter()
            .map(|(addr, size)| FreeSection { addr, size })
            .collect();

        // Read the current extension's message region (single chunk), then write
        // the canonical tail past the appended data and repoint the superblock.
        let (region, _ext_len) = read_single_chunk_ext_region(&self.store, old_ext_addr)?;
        let new_old_blocks = self
            .store
            .write_persist_tail(&region, strategy, threshold, page_size, &sections)?;

        // Re-arm from the just-written shape so a subsequent finalize (or a
        // future append+finalize) stays consistent. `free` is the post list; the
        // new extension address is the one the repoint recorded.
        self.persist = Some(BoundedPersistState {
            strategy,
            threshold,
            page_size,
            free,
            old_blocks: new_old_blocks,
            old_ext_addr: self
                .store
                .superblock
                .superblock_extension_address
                .unwrap_or(old_ext_addr),
        });
        Ok(())
    }

    /// Locate (or fetch the cached) append geometry for the dataset at
    /// `oh_addr`, so the public append path can slice a large call into
    /// aligned batches *before* building each batch's byte buffer — keeping
    /// peak memory at one batch rather than the whole call.
    pub(crate) fn append_geometry(&mut self, oh_addr: u64) -> Result<AppendGeometry, Error> {
        let Self { store, located, .. } = self;
        let st = match located.entry(oh_addr) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(locate_dataset_state(&*store, oh_addr)?)
            }
        };
        let chunk_elems = st.loc.chunk_elems.max(1);
        let batch_chunks = (APPEND_BATCH_BYTES / (st.loc.chunk_bytes.max(1) as u64)).max(1);
        Ok(AppendGeometry {
            chunk_elems,
            element_size: st.element_size,
            current_dim: st.loc.current_dim,
            filtered: st.pipeline.is_some(),
            full_batch_elems: batch_chunks * chunk_elems,
        })
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
        let Self { store, located, .. } = self;
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
        // Refuse a non-chunk-aligned filtered append before ANY batch applies,
        // so the refusal is as atomic as the mirror path's. Without this,
        // `plan_ea_append` would reject it only when the final (unaligned)
        // batch is reached — after earlier batches had durably committed.
        if st.pipeline.is_some()
            && (st.loc.current_dim % chunk_elems != 0 || new_elems % chunk_elems != 0)
        {
            return Err(Error::AppendInPlaceUnsupported(
                "a filtered dataset can only be appended in place in whole chunks (the current \
                 length and the appended length must both be multiples of the chunk length); \
                 use Dataset::append_staged for a non-chunk-aligned filtered append",
            ));
        }
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

/// Parse the File Space Info message from the file's superblock extension, if
/// present and readable. Best-effort on the *extension* itself (a missing or
/// malformed extension reads as `None`, matching `WriteEngine`'s loader).
fn ext_file_space_info(raw: &RawSource<'_>, superblock: &Superblock) -> Option<FileSpaceInfo> {
    let rel = superblock.superblock_extension_address?;
    if rel == u64::MAX {
        return None;
    }
    // base_address == 0 is validated before this runs, so `rel` is absolute.
    let header = ObjectHeader::parse_from_source(
        raw,
        rel,
        superblock.offset_size,
        superblock.length_size,
        0,
    )
    .ok()?;
    let msg = header
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::FileSpaceInfo)?;
    FileSpaceInfo::parse(&msg.data, superblock.offset_size, superblock.length_size).ok()
}

/// Seed a [`BoundedPersistState`] for a file that persists its free space (issue
/// #173), or `Ok(None)` when the file does not persist. Mirrors
/// `WriteEngine::load_persisted_free_space` over a random-access [`Source`]: it
/// reads only the small manager/extension blocks, so it stays bounded-memory.
/// The paged strategy is refused (it needs page-aligned allocation, Phase 2).
/// The caller has already validated `base_address == 0` and a v2/v3 superblock.
fn load_bounded_persist(
    raw: &RawSource<'_>,
    superblock: &Superblock,
) -> Result<Option<BoundedPersistState>, Error> {
    let Some(info) = ext_file_space_info(raw, superblock) else {
        return Ok(None);
    };
    // Refuse the paged strategy regardless of the persist flag: the writer does
    // not yet do page-aligned allocation, so appending at a raw end-of-file would
    // break the paged invariants (issue #173 Phase 2). Checked before the persist
    // early-return so a paged, non-persisting file is refused too, matching the
    // documented behavior.
    if info.strategy == FileSpaceStrategy::Page {
        return Err(Error::EditUnsupported(
            "bounded read-write does not yet support the paged file-space strategy \
             (H5F_FSPACE_STRATEGY_PAGE); use File::open_rw",
        ));
    }
    if !info.persist {
        return Ok(None);
    }
    let os = superblock.offset_size;
    let ext_addr = superblock
        .superblock_extension_address
        .expect("ext_file_space_info returned Some, so the extension address is set");

    // Seed the free list from the on-disk managers. A malformed or truncated
    // manager is tolerated exactly as the mirror loader tolerates it (open, arm
    // persistence, seed nothing) rather than failing the open — the file stays
    // openable, and the finalize simply rewrites whatever we could recover.
    let (mut sections, manager_blocks) =
        read_persisted_sections_source(raw, &info.manager_addrs, 0, os)
            .unwrap_or_else(|_| (Vec::new(), Vec::new()));
    let mut free = FreeList::new();
    let file_len = raw.len();
    sections.sort_by_key(|s| s.addr);
    let mut prev_end = 0u64;
    for s in sections {
        let Some(end) = s.addr.checked_add(s.size) else {
            continue;
        };
        if s.size == 0 || end > file_len || s.addr < prev_end {
            continue;
        }
        prev_end = end;
        free.free(s.addr, s.size);
    }

    // Record the blocks the live file uses (extension header + each FSHD/FSSE) so
    // the next finalize frees them when it writes replacements.
    let mut old_blocks = manager_blocks;
    let (_region, ext_len) = read_single_chunk_ext_region(raw, ext_addr)?;
    old_blocks.push((ext_addr, ext_len));

    Ok(Some(BoundedPersistState {
        strategy: info.strategy,
        threshold: info.threshold,
        page_size: info.page_size,
        free,
        old_blocks,
        old_ext_addr: ext_addr,
    }))
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

    /// A persisting file appended through the bounded engine and dropped WITHOUT
    /// `finalize_persist` (the true-crash case — `BoundedEngine` itself has no
    /// `Drop`; only `FileInner::drop` finalizes) still reads back every durable
    /// append. Dropping the engine releases the exclusive lock, so the reopen is
    /// portable (no leaked lock). The finalize-at-close path is covered by the
    /// `tests/bounded_append.rs` integration tests.
    #[test]
    fn persist_append_without_finalize_is_readable() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("persist_crash.h5");
        let mut b = FileBuilder::new();
        b.with_file_space_strategy(crate::FileSpaceStrategy::FsmAggr, true, 1);
        b.create_dataset("d")
            .with_i32_data(&(0..6).collect::<Vec<i32>>())
            .with_shape(&[6])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[4]);
        b.write(&p).unwrap();
        {
            let mut engine = BoundedEngine::open(&p, MetadataCacheConfig::disabled()).unwrap();
            assert!(engine.persist.is_some(), "persist state is armed at open");
            let addr = dataset_addr(&engine);
            let mut ab = AppendBuilder::new();
            ab.append_i32(&(6..20).collect::<Vec<_>>());
            engine.append_gathered(addr, &ab, 4).unwrap();
            // Drop the engine without calling finalize_persist: models a true
            // crash and releases the OS lock.
        }
        let got = crate::File::open(&p)
            .unwrap()
            .dataset("d")
            .unwrap()
            .read_i32()
            .unwrap();
        assert_eq!(got, (0..20).collect::<Vec<_>>());
    }
}
