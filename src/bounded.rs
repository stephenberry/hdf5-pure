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
use crate::file_space_info::{FileSpaceInfo, FileSpaceStrategy, NUM_FILE_FSM_MANAGERS};
use crate::free_space::FreeList;
use crate::free_space_manager::{
    FreeSection, SECT_CLASS_LARGE, SECT_CLASS_SIMPLE, SECT_CLASS_SMALL, fshd_len, fsse_len,
    read_persisted_sections_source, serialize_file_fsm,
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
    /// Paged-append state for a genuine paged file (`H5F_FSPACE_STRATEGY_PAGE`,
    /// issue #173 Phase 2). `None` for the common non-paged file, where
    /// [`append_raw`](Store::append_raw) / [`append_meta`](Store::append_meta) are
    /// plain end-of-file appends. When `Some`, they keep pages homogeneous.
    paged: Option<PagedAppend>,
}

/// Page type of a paged allocation. A paged file never mixes metadata and raw
/// data within one page, so appends of the two types are kept in separate pages.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PageType {
    /// File metadata (extensible-array blocks, headers, free-space blocks).
    Meta,
    /// Raw dataset data (chunk contents).
    Raw,
}

/// Paged-append bookkeeping on a paged [`BoundedStore`] (issue #173 Phase 2). The
/// bounded backend only ever appends at end-of-file (it never reuses a hole), so
/// keeping a paged file's pages homogeneous reduces to: whenever an append's page
/// type differs from the tail page's, pad the tail page to a page boundary first.
/// The padded tails are recorded per page type so [`finalize_persist`] can hand
/// them to the SUPER (metadata) / DRAW (raw) managers.
struct PagedAppend {
    page_size: u64,
    /// Page type of the current tail page. `None` until the first typed append of
    /// the session; the file is page-aligned at open, so the first append never
    /// needs to pad regardless of this.
    last: Option<PageType>,
    /// Free tails left by padding a metadata page before a raw append.
    meta_pad: Vec<(u64, u64)>,
    /// Free tails left by padding a raw page before a metadata append.
    raw_pad: Vec<(u64, u64)>,
}

impl BoundedStore {
    pub(crate) fn superblock(&self) -> &Superblock {
        &self.superblock
    }

    /// Append `bytes` at end-of-file as page type `ty`. For a non-paged store this
    /// is a plain [`append_bytes`](Store::append_bytes). For a paged store it keeps
    /// pages homogeneous: if the tail page holds the *other* page type and is only
    /// partially filled, it is first padded to a page boundary (the padding being
    /// recorded as free space of the outgoing type), so `bytes` start a fresh page.
    fn append_typed(&mut self, bytes: &[u8], ty: PageType) -> Result<u64, Error> {
        // Decide the pad without holding a borrow of `self.paged` across the append.
        // `prev` is the outgoing page type to record the padding under, or `None`
        // for a crash-recovery pad whose tail-page type is unknown.
        let pad = match &self.paged {
            Some(pg) if self.len % pg.page_size != 0 => {
                let pad_len = pg.page_size - self.len % pg.page_size;
                match pg.last {
                    // Normal case: the tail page holds a known type; pad only on a
                    // type switch, recording the tail as free of the outgoing type.
                    Some(prev) if prev != ty => Some((Some(prev), pad_len)),
                    Some(_) => None, // same type: keep packing the tail page
                    // Crash recovery: a previous bounded session grew this paged
                    // file and was killed before finalize page-aligned the tail, so
                    // the file opened non-page-aligned with no known tail type. Pad
                    // it up (extending whatever the tail page holds, so the page
                    // stays homogeneous) and leave the padding untracked, since
                    // recording it under the wrong page type could let the reader
                    // reuse it and mix the page.
                    None => Some((None, pad_len)),
                }
            }
            _ => None,
        };
        if let Some((prev, pad_len)) = pad {
            let pad_at = self.len;
            self.append_at_eof(&vec![0u8; pad_len.to_usize()?])?;
            if let Some(pg) = self.paged.as_mut() {
                match prev {
                    Some(PageType::Meta) => pg.meta_pad.push((pad_at, pad_len)),
                    Some(PageType::Raw) => pg.raw_pad.push((pad_at, pad_len)),
                    None => {} // crash-recovery pad: untracked (tail type unknown)
                }
            }
        }
        let addr = self.append_at_eof(bytes)?;
        if let Some(pg) = self.paged.as_mut() {
            pg.last = Some(ty);
        }
        Ok(addr)
    }

    /// Pad the file to a page boundary if it is a paged store whose tail page is
    /// partially filled, recording the padding as free space of the tail page's
    /// type. Returns the file's (now page-aligned, for a paged store) length.
    /// Called once by [`finalize_persist`] before it lays the rewritten managers
    /// into a fresh metadata page. A no-op for a non-paged store.
    fn pad_to_page(&mut self) -> Result<u64, Error> {
        let pad = match &self.paged {
            Some(pg) if self.len % pg.page_size != 0 => {
                Some((pg.last, pg.page_size - self.len % pg.page_size))
            }
            _ => None,
        };
        if let Some((last, pad_len)) = pad {
            let pad_at = self.len;
            self.append_at_eof(&vec![0u8; pad_len.to_usize()?])?;
            if let Some(pg) = self.paged.as_mut() {
                match last {
                    // A partially-filled tail page at finalize is a raw page (the
                    // last thing an append writes is raw chunk data); default an
                    // unknown tail (no typed append this session) to raw too.
                    Some(PageType::Meta) => pg.meta_pad.push((pad_at, pad_len)),
                    _ => pg.raw_pad.push((pad_at, pad_len)),
                }
            }
        }
        Ok(self.len)
    }

    /// The unconditional end-of-file append (no page bookkeeping); the primitive
    /// [`append_typed`](Self::append_typed) and the `Store` impl build on.
    fn append_at_eof(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        let addr = self.len;
        self.handle.seek(SeekFrom::Start(addr)).map_err(Error::Io)?;
        self.handle.write_all(bytes).map_err(Error::Io)?;
        self.len += bytes.len() as u64;
        Ok(addr)
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

    /// The paged (`H5F_FSPACE_STRATEGY_PAGE`) counterpart of
    /// [`write_persist_tail`](Self::write_persist_tail) (issue #173 Phase 2).
    /// Instead of one generic manager it lays *per-page-type* managers into a fresh
    /// metadata page at end-of-file — SUPER (slot 0) for the metadata free tails,
    /// DRAW (slot 2) for small-raw tails, and the generic-large manager (slot 6) for
    /// any large-raw fragments — with a page-aligned `eoa_pre_fsm` at the (padded)
    /// file end, matching the paged file the writer creates from scratch. The caller
    /// has page-aligned the file first, so the extension and manager blocks begin on
    /// a page boundary and stay in metadata pages. Crash-atomic exactly as the
    /// non-paged path: the new blocks are unreferenced until the superblock repoint.
    #[allow(clippy::too_many_arguments)]
    fn write_persist_tail_paged(
        &mut self,
        old_ext_region: &[u8],
        strategy: FileSpaceStrategy,
        threshold: u64,
        page_size: u64,
        meta: &[FreeSection],
        raw_small: &[FreeSection],
        raw_large: &[FreeSection],
    ) -> Result<Vec<(u64, u64)>, Error> {
        let os = self.superblock.offset_size;
        let new_root = self.superblock.root_group_address;
        let ext_addr = self.len;
        debug_assert_eq!(
            ext_addr % page_size,
            0,
            "extension begins on a page boundary"
        );

        // The 12-slot persist message is fixed-size, so a placeholder sizes the
        // rewritten extension before its manager addresses are known.
        let placeholder = FileSpaceInfo::persistent_managers(
            strategy,
            threshold,
            page_size,
            [u64::MAX; NUM_FILE_FSM_MANAGERS],
            0,
        );
        let ext_len = build_v2_object_header(&rewrite_extension_region_bytes(
            old_ext_region,
            &placeholder,
        )?)
        .len() as u64;

        // Split every section at page boundaries, then class each fragment by size:
        // an intra-page (< page) fragment stays in its SMALL-class per-type manager
        // (SUPER=0 for metadata, DRAW=2 for small raw), while a whole free page
        // (>= page, which only arises from freeing a page's worth of metadata below
        // a page tail) goes to the single generic-large manager (slot 6), together
        // with any pre-existing large-raw fragments. This keeps a SMALL section from
        // ever spanning a page or reaching page_size, matching the reference library.
        let mut slot0 = Vec::new();
        let mut slot2 = Vec::new();
        let mut slot6 = Vec::new();
        for s in split_at_pages(meta, page_size) {
            if s.size < page_size {
                slot0.push(s)
            } else {
                slot6.push(s)
            }
        }
        for s in split_at_pages(raw_small, page_size) {
            if s.size < page_size {
                slot2.push(s)
            } else {
                slot6.push(s)
            }
        }
        for s in split_at_pages(raw_large, page_size) {
            slot6.push(s);
        }
        slot6.sort_by_key(|s| s.addr);
        let managers: [(usize, u8, &[FreeSection]); 3] = [
            (0, SECT_CLASS_SMALL, &slot0),
            (2, SECT_CLASS_SMALL, &slot2),
            (6, SECT_CLASS_LARGE, &slot6),
        ];

        if managers.iter().all(|(_, _, s)| s.is_empty()) {
            // No free space to track: an empty persist message, page-aligned.
            let info = FileSpaceInfo::persistent_empty(strategy, threshold, page_size);
            let ext_oh =
                build_v2_object_header(&rewrite_extension_region_bytes(old_ext_region, &info)?);
            let written_ext = self.append_bytes(&ext_oh)?;
            debug_assert_eq!(written_ext, ext_addr);
            let final_eof = align_up(ext_addr + ext_oh.len() as u64, page_size);
            self.pad_zeros_to(final_eof)?;
            Store::sync(self)?;
            self.repoint_persist_superblock(new_root, final_eof, ext_addr)?;
            Store::sync(self)?;
            return Ok(vec![(ext_addr, ext_oh.len() as u64)]);
        }

        // Closed-form layout: the extension, then each active manager's FSHD/FSSE.
        // FSSE byte length depends only on section count (fixed field widths), so a
        // single forward pass fixes every address with no fixpoint iteration.
        let mut slots = [u64::MAX; NUM_FILE_FSM_MANAGERS];
        let mut blocks: Vec<(usize, u64, u64, u8, &[FreeSection])> = Vec::new(); // slot, fshd, fsse, class, sections
        let mut cursor = ext_addr + ext_len;
        for &(slot, class, sections) in &managers {
            if sections.is_empty() {
                continue;
            }
            let fshd_addr = cursor;
            let fsse_addr = fshd_addr + fshd_len(os);
            let section_sizes: Vec<u64> = sections.iter().map(|s| s.size).collect();
            cursor = fsse_addr + fsse_len(&section_sizes, os);
            slots[slot] = fshd_addr;
            blocks.push((slot, fshd_addr, fsse_addr, class, sections));
        }
        let end_of_managers = cursor;
        let final_eof = align_up(end_of_managers, page_size);
        // Paged convention (matching the from-scratch writer): the managers are
        // ordinary metadata below a page-aligned end-of-allocation.
        let eoa_pre_fsm = final_eof;

        let info =
            FileSpaceInfo::persistent_managers(strategy, threshold, page_size, slots, eoa_pre_fsm);
        let ext_oh =
            build_v2_object_header(&rewrite_extension_region_bytes(old_ext_region, &info)?);
        debug_assert_eq!(
            ext_oh.len() as u64,
            ext_len,
            "extension length must be stable across the placeholder and real messages"
        );

        // Append the extension, then every manager block, at (page-aligned) EOF.
        // They are unreferenced until the repoint, so a crash here is harmless.
        let written_ext = self.append_bytes(&ext_oh)?;
        debug_assert_eq!(written_ext, ext_addr);
        let mut new_old_blocks = vec![(ext_addr, ext_oh.len() as u64)];
        for &(_slot, fshd_addr, fsse_addr, class, sections) in &blocks {
            let (fshd, fsse) = serialize_file_fsm(sections, fshd_addr, fsse_addr, os, class);
            let wf = self.append_bytes(&fshd)?;
            debug_assert_eq!(wf, fshd_addr);
            new_old_blocks.push((fshd_addr, fshd.len() as u64));
            let ws = self.append_bytes(&fsse)?;
            debug_assert_eq!(ws, fsse_addr);
            new_old_blocks.push((ws, fsse.len() as u64));
        }
        debug_assert_eq!(self.len, end_of_managers);
        // Pad the final metadata page to its boundary. This trailing tail is left
        // untracked (a valid free-space under-report), keeping the manager layout
        // closed-form rather than self-referential.
        self.pad_zeros_to(final_eof)?;

        // Barrier, then repoint the superblock (the linearization point), then sync.
        Store::sync(self)?;
        self.repoint_persist_superblock(new_root, final_eof, ext_addr)?;
        Store::sync(self)?;
        Ok(new_old_blocks)
    }

    /// Extend the file with zeros up to `target` (>= current length). Used by the
    /// paged finalize to pad to a page boundary.
    fn pad_zeros_to(&mut self, target: u64) -> Result<(), Error> {
        if target > self.len {
            let pad = (target - self.len).to_usize()?;
            self.append_at_eof(&vec![0u8; pad])?;
        }
        debug_assert_eq!(self.len, target);
        Ok(())
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
        self.append_at_eof(bytes)
    }
    fn append_raw(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        self.append_typed(bytes, PageType::Raw)
    }
    fn append_meta(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        self.append_typed(bytes, PageType::Meta)
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
    /// The live free space seeded from the on-disk managers (never handed out —
    /// the bounded engine appends only), plus, at finalize, the superseded old
    /// manager/extension blocks. A non-paged file tracks one generic manager; a
    /// paged file tracks per-page-type managers.
    free: PersistFree,
    /// `(addr, len)` of the superblock-extension header and every `FSHD`/`FSSE`
    /// block the *current* on-disk file uses; freed and rewritten by the next
    /// finalize.
    old_blocks: Vec<(u64, u64)>,
    /// Absolute address of the current superblock-extension object header.
    old_ext_addr: u64,
}

/// The free space a persisting bounded session tracks, in the shape its managers
/// take at finalize. A non-paged file has a single generic free-space manager; a
/// paged file (issue #173 Phase 2) has one manager per page type.
enum PersistFree {
    /// Non-paged: one generic (simple-class) free-space manager.
    Flat(FreeList),
    /// Paged: the SUPER (metadata), DRAW (small raw), and generic-large (raw)
    /// managers, kept separate because a paged manager holds one page type only.
    Paged {
        meta: FreeList,
        raw_small: FreeList,
        raw_large: FreeList,
    },
}

impl PagedAppend {
    fn new(page_size: u64) -> Self {
        Self {
            page_size,
            last: None,
            meta_pad: Vec::new(),
            raw_pad: Vec::new(),
        }
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
        // A file that persists its free space is supported (issue #173): seed the
        // free list from its on-disk managers and rewrite them at close. A genuine
        // paged file (persist=true) is supported in Phase 2 by segregating appends
        // into metadata vs raw pages; a paged non-persisting file is refused (it
        // has no managers to describe the segregated free space).
        let persist = load_bounded_persist(&raw, &superblock)?;
        // Arm the paged-append machinery for a paged file so raw and metadata
        // appends land in separate pages. The file is page-aligned at open (a
        // genuine paged file has a page-aligned end-of-allocation), so the first
        // typed append never needs to pad.
        // Arm paged appends whenever the file is paged. A cleanly written or closed
        // paged file is page-aligned at open, but a bounded session that grew a
        // paged file and was killed before finalize leaves the physical end
        // non-page-aligned; that file still opens and reads correctly, and the
        // first typed append re-aligns the tail page (see `append_typed`), so no
        // page-alignment is assumed here.
        let paged = match &persist {
            Some(ps) if ps.strategy == FileSpaceStrategy::Page => {
                Some(PagedAppend::new(ps.page_size))
            }
            _ => None,
        };

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
                paged,
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
            free,
            old_blocks,
            old_ext_addr,
        } = ps;

        // Read the current extension's message region (single chunk), then write
        // the canonical tail past the appended data and repoint the superblock.
        let (region, _ext_len) = read_single_chunk_ext_region(&self.store, old_ext_addr)?;

        let (new_old_blocks, new_free) = match free {
            PersistFree::Flat(mut free) => {
                // The old extension + manager blocks are superseded by the rewrite,
                // so they join the free list (their space becomes reclaimable).
                for &(a, l) in &old_blocks {
                    free.free(a, l);
                }
                let sections = free_sections(&free);
                let nob = self
                    .store
                    .write_persist_tail(&region, strategy, threshold, page_size, &sections)?;
                (nob, PersistFree::Flat(free))
            }
            PersistFree::Paged {
                mut meta,
                mut raw_small,
                raw_large,
            } => {
                // Page-align the file first, then fold the session's page-padding
                // tails into their managers (metadata pad -> SUPER, raw pad -> DRAW)
                // and free the superseded old extension + manager blocks (metadata)
                // into SUPER. Do this before laying the new managers so the free
                // lists describe every reclaimable hole below the managers.
                self.store.pad_to_page()?;
                if let Some(pg) = self.store.paged.as_mut() {
                    for &(a, l) in &pg.meta_pad {
                        meta.free(a, l);
                    }
                    for &(a, l) in &pg.raw_pad {
                        raw_small.free(a, l);
                    }
                    pg.meta_pad.clear();
                    pg.raw_pad.clear();
                }
                for &(a, l) in &old_blocks {
                    meta.free(a, l);
                }
                let nob = self.store.write_persist_tail_paged(
                    &region,
                    strategy,
                    threshold,
                    page_size,
                    &free_sections(&meta),
                    &free_sections(&raw_small),
                    &free_sections(&raw_large),
                )?;
                (
                    nob,
                    PersistFree::Paged {
                        meta,
                        raw_small,
                        raw_large,
                    },
                )
            }
        };

        // Re-arm from the just-written shape so a subsequent finalize (or a
        // future append+finalize) stays consistent. `new_free` is the post list;
        // the new extension address is the one the repoint recorded.
        self.persist = Some(BoundedPersistState {
            strategy,
            threshold,
            page_size,
            free: new_free,
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
/// A genuine paged file (persist=true) is supported by seeding its per-page-type
/// managers; a paged *non-persisting* file is refused (it has no managers to
/// describe the segregated free space). The caller has already validated
/// `base_address == 0` and a v2/v3 superblock.
fn load_bounded_persist(
    raw: &RawSource<'_>,
    superblock: &Superblock,
) -> Result<Option<BoundedPersistState>, Error> {
    let Some(info) = ext_file_space_info(raw, superblock) else {
        return Ok(None);
    };
    let paged = info.strategy == FileSpaceStrategy::Page;
    if paged && !info.persist {
        // A paged file without persisted managers has no on-disk record of which
        // pages are metadata vs raw, so bounded appends cannot keep the free space
        // segregated. Refuse rather than corrupt the paging.
        return Err(Error::EditUnsupported(
            "bounded read-write of a paged file (H5F_FSPACE_STRATEGY_PAGE) requires \
             persisted free space (persist=true); use File::open_rw",
        ));
    }
    if !info.persist {
        return Ok(None);
    }
    let os = superblock.offset_size;
    let ext_addr = superblock
        .superblock_extension_address
        .expect("ext_file_space_info returned Some, so the extension address is set");
    let file_len = raw.len();

    // Record the blocks the live file uses (each FSHD/FSSE + the extension header)
    // so the next finalize frees them when it writes replacements. A malformed or
    // truncated manager is tolerated exactly as the mirror loader tolerates it
    // (seed nothing) rather than failing the open.
    let mut old_blocks = Vec::new();
    let free = if paged {
        // Seed the per-page-type managers by their on-disk slot: SUPER (slot 0) is
        // metadata, DRAW (slot 2) is small raw, and the generic-large manager (slot
        // 6) holds large-raw fragments. Genuine paged files use only these slots.
        let mut tagged: Vec<(FreeSection, u8)> = Vec::new(); // 0 = meta, 1 = raw_small, 2 = raw_large
        for (k, &m) in info.manager_addrs.iter().enumerate() {
            if m == u64::MAX {
                continue;
            }
            let (secs, blocks) = read_persisted_sections_source(raw, &[m], 0, os)
                .unwrap_or_else(|_| (Vec::new(), Vec::new()));
            old_blocks.extend(blocks);
            let which = match k {
                2 => 1u8,
                6 => 2u8,
                _ => 0u8, // slot 0 and any other (unexpected) slot -> metadata
            };
            for s in secs {
                tagged.push((s, which));
            }
        }
        // Validate globally (sorted, non-overlapping, within the file), then route
        // each surviving section into its page-type list.
        tagged.sort_by_key(|(s, _)| s.addr);
        let mut meta = FreeList::new();
        let mut raw_small = FreeList::new();
        let mut raw_large = FreeList::new();
        let mut prev_end = 0u64;
        for (s, which) in tagged {
            let Some(end) = s.addr.checked_add(s.size) else {
                continue;
            };
            if s.size == 0 || end > file_len || s.addr < prev_end {
                continue;
            }
            prev_end = end;
            match which {
                1 => raw_small.free(s.addr, s.size),
                2 => raw_large.free(s.addr, s.size),
                _ => meta.free(s.addr, s.size),
            }
        }
        PersistFree::Paged {
            meta,
            raw_small,
            raw_large,
        }
    } else {
        let (mut sections, manager_blocks) =
            read_persisted_sections_source(raw, &info.manager_addrs, 0, os)
                .unwrap_or_else(|_| (Vec::new(), Vec::new()));
        old_blocks.extend(manager_blocks);
        let mut free = FreeList::new();
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
        PersistFree::Flat(free)
    };

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

/// The free `(addr, size)` runs a [`FreeList`] holds, as [`FreeSection`]s in the
/// shape the manager serializer expects.
fn free_sections(free: &FreeList) -> Vec<FreeSection> {
    free.sections()
        .into_iter()
        .map(|(addr, size)| FreeSection { addr, size })
        .collect()
}

/// Round `value` up to the next multiple of `page` (`page` is a power of two >=
/// 512, validated at file creation).
fn align_up(value: u64, page: u64) -> u64 {
    value.div_ceil(page) * page
}

/// Split each free section at page boundaries so no section spans a page. The
/// bounded finalize coalesces a page-tail free section with freed manager blocks
/// below it, which can produce a run that crosses a page boundary or reaches
/// `page`; splitting lets each intra-page fragment stay in its SMALL-class manager
/// while a whole free page is routed to the generic-large manager, matching the
/// reference library's small-vs-large section classes.
fn split_at_pages(sections: &[FreeSection], page: u64) -> Vec<FreeSection> {
    let mut out = Vec::new();
    for s in sections {
        let end = s.addr.saturating_add(s.size);
        let mut start = s.addr;
        while start < end {
            let boundary = (start / page + 1) * page;
            let piece_end = end.min(boundary);
            out.push(FreeSection {
                addr: start,
                size: piece_end - start,
            });
            start = piece_end;
        }
    }
    out
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

    /// A bounded session grows a PAGED persisting file and is killed before
    /// finalize (models a crash), leaving the file non-page-aligned. Reopening it
    /// must not panic, and the next append must re-align the crashed tail page
    /// before writing raw data (so no page mixes metadata and raw); a clean close
    /// then re-page-aligns the file and every row reads back.
    #[test]
    fn paged_reopen_after_crash_realigns_and_stays_readable() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("paged_crash.h5");
        let mut b = FileBuilder::new();
        b.with_file_space_strategy(crate::FileSpaceStrategy::Page, true, 0)
            .with_file_space_page_size(4096);
        b.create_dataset("d")
            .with_i32_data(&(0..64).collect::<Vec<i32>>())
            .with_shape(&[64])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[64]);
        b.write(&p).unwrap();

        // Grow enough to force extensible-array index growth, so the last write of
        // the session is metadata and the tail page is a partial metadata page.
        {
            let mut engine = BoundedEngine::open(&p, MetadataCacheConfig::disabled()).unwrap();
            let addr = dataset_addr(&engine);
            let mut ab = AppendBuilder::new();
            ab.append_i32(&(64..2000).collect::<Vec<_>>());
            engine.append_gathered(addr, &ab, 4).unwrap();
            // Drop without finalize: models a crash and releases the OS lock.
        }
        assert_ne!(
            std::fs::metadata(&p).unwrap().len() % 4096,
            0,
            "a crashed (un-finalized) paged session leaves the file non-page-aligned"
        );

        // Reopen must not panic on the non-aligned file; the next append re-aligns
        // the crashed tail page, and finalize re-page-aligns the whole file.
        {
            let mut engine = BoundedEngine::open(&p, MetadataCacheConfig::disabled()).unwrap();
            let addr = dataset_addr(&engine);
            let mut ab = AppendBuilder::new();
            ab.append_i32(&(2000..2500).collect::<Vec<_>>());
            engine.append_gathered(addr, &ab, 4).unwrap();
            engine.finalize_persist().unwrap();
            engine.sync().unwrap();
        }
        assert_eq!(
            std::fs::metadata(&p).unwrap().len() % 4096,
            0,
            "reopen + append + finalize re-aligns the paged file"
        );
        let got = crate::File::open(&p)
            .unwrap()
            .dataset("d")
            .unwrap()
            .read_i32()
            .unwrap();
        assert_eq!(got, (0..2500).collect::<Vec<_>>());
    }

    #[test]
    fn split_at_pages_splits_on_boundaries_preserving_total() {
        let page = 4096;
        // A run that crosses one boundary splits into a sub-page head and a whole
        // free page (the accounting fix routes the >= page piece to slot 6).
        assert_eq!(
            split_at_pages(
                &[FreeSection {
                    addr: 3740,
                    size: 4452
                }],
                page
            ),
            vec![
                FreeSection {
                    addr: 3740,
                    size: 356
                }, // [3740, 4096)
                FreeSection {
                    addr: 4096,
                    size: 4096
                }, // [4096, 8192)
            ]
        );
        // A sub-page section is returned unchanged (the common case is a no-op).
        assert_eq!(
            split_at_pages(
                &[FreeSection {
                    addr: 100,
                    size: 200
                }],
                page
            ),
            vec![FreeSection {
                addr: 100,
                size: 200
            }]
        );
        // A page-aligned multi-page run splits into whole pages.
        let whole = split_at_pages(
            &[FreeSection {
                addr: 0,
                size: 3 * 4096,
            }],
            page,
        );
        assert_eq!(whole.len(), 3);
        assert!(whole.iter().all(|s| s.size == 4096));
        // Splitting never loses or overlaps space: pieces are contiguous and sum
        // to the original size.
        let pieces = split_at_pages(
            &[FreeSection {
                addr: 5000,
                size: 10000,
            }],
            page,
        );
        assert_eq!(pieces.iter().map(|s| s.size).sum::<u64>(), 10000);
        let mut prev = pieces[0].addr;
        for s in &pieces {
            assert_eq!(s.addr, prev);
            prev = s.addr + s.size;
        }
        assert_eq!(prev, 15000);
    }
}
