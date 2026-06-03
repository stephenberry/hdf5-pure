//! SWMR (single-writer / multiple-reader) append writer.
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
//! # Supported subset (v1)
//!
//! - One unlimited dimension, chunked, Extensible-Array index (the index the C
//!   library and h5py select for a single unlimited dimension under the latest
//!   format).
//! - Unfiltered datasets (no compression/shuffle/scale-offset on the appended
//!   dataset). Filtered append is rejected with a clear error.
//! - Appends land on chunk boundaries: the dataset's current length and the
//!   appended length are both whole multiples of the chunk length. (A chunk
//!   length of 1 — the common streaming layout — always satisfies this.)
//! - Unbounded growth: super blocks and (past ~131060 chunks) paged data blocks
//!   are allocated incrementally as block boundaries are crossed.
//! - Files with a zero base address (no userblock).

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};

use crate::checksum::jenkins_lookup3;
use crate::chunked_write::ea_compute_stats;
use crate::data_layout::DataLayout;
use crate::dataspace::Dataspace;
use crate::error::{Error, FormatError};
use crate::extensible_array::{EaGeometry, ExtensibleArrayHeader};
use crate::group_v2;
use crate::message_type::MessageType;
use crate::signature;
use crate::superblock::Superblock;

/// The undefined-address sentinel for a given offset size.
fn undef_addr(offset_size: u8) -> u64 {
    match offset_size {
        4 => 0xFFFF_FFFF,
        _ => u64::MAX,
    }
}

fn is_undef(addr: u64, offset_size: u8) -> bool {
    addr == undef_addr(offset_size)
}

/// Metadata located once per dataset, then maintained across appends.
struct Located {
    /// File offset of the dataspace message's first current-dimension value.
    dim0_off: usize,
    /// Current length along the unlimited (axis-0) dimension.
    current_dim: u64,
    /// Object-header chunk that contains the dataspace message: byte range whose
    /// Jenkins checksum must be recomputed after patching the dimension. The
    /// checksum itself occupies `chunk_msg_end .. chunk_msg_end + 4`.
    ohdr_chunk_start: usize,
    ohdr_chunk_msg_end: usize,

    /// Elements per chunk along axis 0 (the only varying axis for rank 1).
    chunk_elems: u64,
    /// Bytes per dataset element (datatype size).
    elem_bytes: usize,
    /// Bytes per chunk.
    chunk_bytes: usize,

    /// Extensible Array header address and derived geometry.
    ea_addr: usize,
    geom: EaGeometry,
    idx_blk_elmts: u64,
    /// Size of one stored EA element in bytes (offset size for unfiltered).
    ea_elem_size: usize,
    page_nelmts: u64,
    /// Block-offset field width inside EA blocks (= ceil(max_nelmts_bits / 8)).
    blk_off_size: usize,
    /// Address of the EA index block (`EAIB`).
    index_block_addr: usize,
    /// Current number of chunks indexed (EA element count).
    num_chunks: u64,
}

/// An append writer over an existing HDF5 file.
///
/// The writer keeps a full in-memory mirror of the file (`O(file size)` memory)
/// so it can read existing structures and recompute checksums without hitting
/// the disk. For the unbounded "streaming log" use case this grows with the
/// file; a future revision could bound it, but v1 favors simplicity.
pub struct SwmrWriter {
    handle: std::fs::File,
    /// Full in-memory mirror of the file; mutated in lock-step with on-disk
    /// writes so reads of existing structures never hit the disk.
    data: Vec<u8>,
    offset_size: u8,
    length_size: u8,
    /// Offset of the superblock signature within the file.
    sb_sig_off: usize,
    superblock: Superblock,
    located: HashMap<String, Located>,
    /// Whether the superblock's SWMR-write consistency flag is currently set
    /// (so [`Drop`] knows to clear it).
    flag_set: bool,
}

/// Superblock consistency-flag bits: bit 0 = write access, bit 2 = SWMR write
/// access. A SWMR writer sets both (`0x05`) while writing and clears them on a
/// clean close — matching the reference C library and h5py.
const SWMR_WRITE_FLAGS: u32 = 0x05;

impl SwmrWriter {
    /// Open an existing HDF5 file for appending.
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        let mut handle = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path.as_ref())
            .map_err(Error::Io)?;
        let mut data = Vec::new();
        handle.read_to_end(&mut data).map_err(Error::Io)?;

        let sb_sig_off = signature::find_signature(&data)?;
        let superblock = Superblock::parse(&data, sb_sig_off)?;
        if superblock.version < 2 {
            // `Superblock::serialize` always emits the v2/v3 layout, so writing
            // the SWMR flag (or the EOF) onto a v0/v1 superblock would clobber
            // it. EA-indexed datasets always imply the latest format, so a
            // legitimate append target is always v2/v3; reject the rest before
            // any mutating write rather than corrupt the file.
            return Err(Error::SwmrAppendUnsupported(
                "SWMR append requires a latest-format file (v2/v3 superblock)",
            ));
        }
        if superblock.base_address != 0 {
            // Userblock files store addresses relative to the base; unsupported
            // by the append writer for now.
            return Err(Error::SwmrAppendUnsupported(
                "files with a userblock (non-zero base address) are not supported",
            ));
        }
        let offset_size = superblock.offset_size;
        let length_size = superblock.length_size;
        let mut w = Self {
            handle,
            data,
            offset_size,
            length_size,
            sb_sig_off,
            superblock,
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
        self.superblock.consistency_flags = if active { SWMR_WRITE_FLAGS } else { 0 };
        let bytes = self.superblock.serialize();
        self.write_at(self.sb_sig_off, &bytes)?;
        self.sync()?;
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
    pub fn clear_swmr_flag<P: AsRef<std::path::Path>>(path: P) -> Result<(), Error> {
        let mut w = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path.as_ref())
            .map_err(Error::Io)?;
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
            let loc = self.locate(dataset)?;
            self.located.insert(dataset.to_string(), loc);
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
            let chunk_addr = self.append_bytes(chunk_data)?;
            let e = num_chunks + c as u64;
            self.ea_insert(dataset, e, chunk_addr)?;
        }
        self.sync()?;
        if max_phase < 2 {
            return Ok(());
        }

        // Phase 2.
        self.patch_superblock_eof()?;
        self.sync()?;
        if max_phase < 3 {
            return Ok(());
        }

        // Phase 3.
        let new_num_chunks = num_chunks + n_new_chunks as u64;
        self.update_ea_header(dataset, new_num_chunks)?;
        self.sync()?;
        if max_phase < 4 {
            return Ok(());
        }

        // Phase 4.
        let new_dim = current_dim + new_elems;
        self.patch_dimension(dataset, new_dim)?;
        self.sync()?;

        if let Some(loc) = self.located.get_mut(dataset) {
            loc.current_dim = new_dim;
            loc.num_chunks = new_num_chunks;
        }
        Ok(())
    }

    // ----- location ------------------------------------------------------

    fn locate(&self, dataset: &str) -> Result<Located, Error> {
        let oh_addr = group_v2::resolve_path_any(&self.data, &self.superblock, dataset)?;
        let os = self.offset_size;
        let ls = self.length_size;

        // Walk the v2 object header, recording the dataspace and layout messages
        // (with file offsets) and the checksum region of their containing chunk.
        let walk = walk_v2_object_header(&self.data, oh_addr as usize, os, ls)?;

        let dataspace_msg = walk
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::Dataspace)
            .ok_or(Error::MissingMessage(MessageType::Dataspace))?;
        let layout_msg = walk
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::DataLayout)
            .ok_or(Error::MissingMessage(MessageType::DataLayout))?;

        // Reject filtered datasets (the EA element encoding differs and the
        // appended chunk would need compressing).
        if walk
            .messages
            .iter()
            .any(|m| m.msg_type == MessageType::FilterPipeline)
        {
            return Err(Error::SwmrAppendUnsupported(
                "filtered datasets are not supported",
            ));
        }

        // Parse the dataspace: must be rank 1 with one unlimited dimension.
        let ds_bytes =
            &self.data[dataspace_msg.data_off..dataspace_msg.data_off + dataspace_msg.size];
        let dataspace = Dataspace::parse(ds_bytes, ls)?;
        if dataspace.rank != 1 {
            return Err(Error::SwmrAppendUnsupported(
                "only rank-1 datasets are supported",
            ));
        }
        match &dataspace.max_dimensions {
            Some(maxs) if maxs.first() == Some(&u64::MAX) => {}
            _ => {
                return Err(Error::SwmrAppendUnsupported(
                    "dataset has no unlimited (maxshape) dimension",
                ));
            }
        }
        let current_dim = dataspace.dimensions[0];
        // v2 dataspace header is 4 bytes (version, rank, flags, type); dim 0 follows.
        let dim0_off = dataspace_msg.data_off + 4;

        // Parse the layout: must be a v4 chunked Extensible Array (index type 4).
        let layout_bytes = &self.data[layout_msg.data_off..layout_msg.data_off + layout_msg.size];
        let layout = DataLayout::parse(layout_bytes, os, ls)?;
        let (ea_addr, chunk_dims) = match layout {
            DataLayout::Chunked {
                chunk_index_type: Some(4),
                btree_address: Some(addr),
                chunk_dimensions,
                ..
            } => (addr as usize, chunk_dimensions),
            _ => {
                return Err(Error::SwmrAppendUnsupported(
                    "only Extensible-Array-indexed chunked datasets are supported",
                ));
            }
        };
        // chunk_dimensions for a v4 layout includes the element-size pseudo
        // dimension as its last entry; the leading entry is the axis-0 chunk size.
        if chunk_dims.len() != 2 {
            return Err(Error::SwmrAppendUnsupported(
                "unexpected chunk dimensionality (expected a rank-1 chunked layout)",
            ));
        }
        let chunk_elems = chunk_dims[0] as u64;
        let elem_bytes = chunk_dims[1] as usize;
        let chunk_bytes = chunk_elems as usize * elem_bytes;

        // Parse the EA header for its creation parameters and current count.
        let ea_header = ExtensibleArrayHeader::parse(&self.data, ea_addr, os, ls)?;
        if ea_header.client_id != 0 {
            return Err(Error::SwmrAppendUnsupported(
                "filtered datasets are not supported",
            ));
        }
        let geom = EaGeometry::from_header(&ea_header);
        let page_nelmts = 1u64 << ea_header.max_dblk_nelmts_bits;
        let blk_off_size = (ea_header.max_nelmts_bits as usize).div_ceil(8);
        let index_block_addr = ea_header.index_block_address as usize;
        // The dataspace dimension is the single commit point (append phase 4);
        // the EA element count is published one phase earlier (phase 3). If a
        // prior writer crashed between the two, the on-disk EA count is ahead of
        // the committed dimension. Seed the writer's chunk count from the
        // *committed* dimension so the next append rolls forward from the last
        // commit -- overwriting any slots a crashed writer wrote but never
        // committed -- instead of appending past them and leaving a gap. In a
        // cleanly-closed file the two already agree, so this is a no-op there.
        // (`chunk_elems == 0` is a malformed layout the append path rejects up
        // front; fall back to the stored count to avoid dividing by zero.)
        let num_chunks = if chunk_elems == 0 {
            ea_header.num_elements
        } else {
            current_dim.div_ceil(chunk_elems)
        };

        Ok(Located {
            dim0_off,
            current_dim,
            ohdr_chunk_start: dataspace_msg.chunk_start,
            ohdr_chunk_msg_end: dataspace_msg.chunk_msg_end,
            chunk_elems,
            elem_bytes,
            chunk_bytes,
            ea_addr,
            geom,
            idx_blk_elmts: ea_header.idx_blk_elmts as u64,
            ea_elem_size: ea_header.element_size as usize,
            page_nelmts,
            blk_off_size,
            index_block_addr,
            num_chunks,
        })
    }

    // ----- extensible array in-place growth ------------------------------

    /// Store `chunk_addr` into element slot `e` of the chunk index, allocating
    /// new data blocks / super blocks at EOF as block boundaries are crossed.
    /// Handles both non-paged and paged data blocks.
    fn ea_insert(&mut self, dataset: &str, e: u64, chunk_addr: u64) -> Result<(), Error> {
        let (os, elem_size, idx, blk_off, page_nelmts, index_block_addr, ea_addr) = {
            let loc = &self.located[dataset];
            (
                self.offset_size,
                loc.ea_elem_size,
                loc.idx_blk_elmts,
                loc.blk_off_size,
                loc.page_nelmts,
                loc.index_block_addr,
                loc.ea_addr,
            )
        };

        // Inline element slots live directly in the index block.
        if e < idx {
            let ib_prefix = 4 + 1 + 1 + os as usize; // sig + ver + client + hdr_addr
            let slot_off = index_block_addr + ib_prefix + e as usize * elem_size;
            self.write_addr_at(slot_off, chunk_addr)?;
            self.rechecksum_index_block(dataset)?;
            return Ok(());
        }

        // Otherwise the element lives in a data block. Find which one and how it
        // is reached (a direct pointer in the index block, or via a super block).
        let region = locate_data_block(&self.located[dataset].geom, idx, e);
        if region.ndblks == 0 {
            // The geometry failed to cover `e` (see `locate_data_block`). Refuse
            // rather than write garbage with the `u64::MAX` block-size sentinel.
            return Err(Error::SwmrAppendUnsupported(
                "chunk index geometry does not cover the appended element",
            ));
        }
        let dblk_nelmts = region.dblk_nelmts;
        let is_paged = dblk_nelmts > page_nelmts;
        let slot = (e - region.db_start) as usize;
        let block_offset_rel = region.db_start - idx;
        let ndblks = region.ndblks;

        // Paged data blocks only ever appear inside a super block. Ensure the
        // owning super block exists and capture its address.
        let sblk_addr = match region.parent {
            Parent::Super { sblk_j, .. } => Some(self.ensure_super_block(
                dataset,
                sblk_j,
                region.sb_block_offset,
                ndblks,
                dblk_nelmts,
            )?),
            Parent::IndexDirect { .. } => None,
        };

        // Resolve the data block's address, allocating it on the first element.
        let dblk_addr = if slot == 0 {
            let new_addr = if is_paged {
                self.alloc_undef_paged_data_block(
                    ea_addr,
                    dblk_nelmts,
                    page_nelmts,
                    block_offset_rel,
                    blk_off,
                    elem_size,
                )?
            } else {
                self.alloc_undef_data_block(ea_addr, dblk_nelmts, block_offset_rel, blk_off)?
            };
            match region.parent {
                Parent::IndexDirect { ordinal } => {
                    let ib_prefix = 4 + 1 + 1 + os as usize;
                    let slot_off = index_block_addr
                        + ib_prefix
                        + idx as usize * elem_size
                        + ordinal * os as usize;
                    self.write_addr_at(slot_off, new_addr)?;
                    self.rechecksum_index_block(dataset)?;
                }
                Parent::Super { dblk_local, .. } => {
                    let sblk_addr = sblk_addr.unwrap();
                    let slot_off = self.sb_dblk_slot_off(
                        sblk_addr,
                        dblk_local,
                        ndblks,
                        dblk_nelmts,
                        page_nelmts,
                        blk_off,
                    );
                    self.write_addr_at(slot_off, new_addr)?;
                    self.rechecksum_super_block(
                        sblk_addr,
                        ndblks,
                        dblk_nelmts,
                        page_nelmts,
                        blk_off,
                    )?;
                }
            }
            new_addr
        } else {
            match region.parent {
                Parent::IndexDirect { ordinal } => {
                    let ib_prefix = 4 + 1 + 1 + os as usize;
                    let slot_off = index_block_addr
                        + ib_prefix
                        + idx as usize * elem_size
                        + ordinal * os as usize;
                    self.read_addr_at(slot_off)
                }
                Parent::Super { dblk_local, .. } => {
                    let sblk_addr = sblk_addr.unwrap();
                    let slot_off = self.sb_dblk_slot_off(
                        sblk_addr,
                        dblk_local,
                        ndblks,
                        dblk_nelmts,
                        page_nelmts,
                        blk_off,
                    );
                    self.read_addr_at(slot_off)
                }
            }
        };

        if !is_paged {
            // Write the element into the data block and re-checksum the block.
            let db_prefix = 4 + 1 + 1 + os as usize + blk_off;
            let elem_off = dblk_addr as usize + db_prefix + slot * elem_size;
            self.write_addr_at(elem_off, chunk_addr)?;
            let cks_off = dblk_addr as usize + db_prefix + dblk_nelmts as usize * elem_size;
            self.rechecksum_range(dblk_addr as usize, cks_off)?;
        } else {
            // Write the element into the right page, re-checksum that page, and
            // (when the page is first touched) mark it initialized in the super
            // block's page-init bitmap.
            let page_nelmts = page_nelmts as usize;
            let header_size = 4 + 1 + 1 + os as usize + blk_off + 4; // includes header checksum
            let page = slot / page_nelmts;
            let slot_in_page = slot % page_nelmts;
            let page_bytes = page_nelmts * elem_size + 4;
            let page_off = dblk_addr as usize + header_size + page * page_bytes;
            self.write_addr_at(page_off + slot_in_page * elem_size, chunk_addr)?;
            let page_cks_off = page_off + page_nelmts * elem_size;
            self.rechecksum_range(page_off, page_cks_off)?;

            if slot_in_page == 0 {
                let sblk_addr = sblk_addr.unwrap();
                let npages = (dblk_nelmts / page_nelmts as u64) as usize;
                if let Parent::Super { dblk_local, .. } = region.parent {
                    let global_page = dblk_local * npages + page;
                    self.set_sb_page_bit(sblk_addr, blk_off, global_page)?;
                    self.rechecksum_super_block(
                        sblk_addr,
                        ndblks,
                        dblk_nelmts,
                        page_nelmts as u64,
                        blk_off,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Byte size of a super block's page-init bitmap (0 when its data blocks are
    /// not paged): `ndblks * ceil(npages / 8)`.
    fn sb_bitmap_size(&self, ndblks: u64, dblk_nelmts: u64, page_nelmts: u64) -> usize {
        if dblk_nelmts > page_nelmts {
            let npages = (dblk_nelmts / page_nelmts) as usize;
            ndblks as usize * npages.div_ceil(8)
        } else {
            0
        }
    }

    /// File offset of the `dblk_local`-th data-block-address slot inside a super
    /// block, accounting for the page-init bitmap when the block is paged.
    fn sb_dblk_slot_off(
        &self,
        sblk_addr: u64,
        dblk_local: usize,
        ndblks: u64,
        dblk_nelmts: u64,
        page_nelmts: u64,
        blk_off: usize,
    ) -> usize {
        let os = self.offset_size as usize;
        let prefix = 4 + 1 + 1 + os + blk_off;
        let bitmap = self.sb_bitmap_size(ndblks, dblk_nelmts, page_nelmts);
        sblk_addr as usize + prefix + bitmap + dblk_local * os
    }

    /// Set page `global_page`'s bit in a super block's page-init bitmap
    /// (MSB-first), in memory and on disk.
    fn set_sb_page_bit(
        &mut self,
        sblk_addr: u64,
        blk_off: usize,
        global_page: usize,
    ) -> Result<(), Error> {
        let os = self.offset_size as usize;
        let bitmap_start = sblk_addr as usize + 4 + 1 + 1 + os + blk_off;
        let byte = bitmap_start + global_page / 8;
        let mask = 0x80u8 >> (global_page % 8);
        let v = self.data[byte] | mask;
        self.write_at(byte, &[v])
    }

    /// Address of an already-allocated super block (`sblk_j`-th super-block
    /// pointer in the index block).
    fn super_block_addr(&self, dataset: &str, sblk_j: usize) -> u64 {
        let loc = &self.located[dataset];
        let os = self.offset_size as usize;
        let ib_prefix = 4 + 1 + 1 + os;
        let ndblk_addrs = loc.geom.direct_dblk_nelmts.len();
        let slot_off = loc.index_block_addr
            + ib_prefix
            + loc.idx_blk_elmts as usize * loc.ea_elem_size
            + ndblk_addrs * os
            + sblk_j * os;
        self.read_addr_at(slot_off)
    }

    /// Return the address of super block `sblk_j`, allocating an empty one (all
    /// data-block pointers undefined, plus a zeroed page-init bitmap when its
    /// data blocks are paged) at EOF if it does not exist yet.
    fn ensure_super_block(
        &mut self,
        dataset: &str,
        sblk_j: usize,
        sb_block_offset: u64,
        ndblks: u64,
        dblk_nelmts: u64,
    ) -> Result<u64, Error> {
        let existing = self.super_block_addr(dataset, sblk_j);
        if !is_undef(existing, self.offset_size) {
            return Ok(existing);
        }
        let (
            blk_off,
            os,
            ib_prefix,
            ndblk_addrs,
            idx_elems,
            elem_size,
            ib_addr,
            ea_addr,
            page_nelmts,
        ) = {
            let loc = &self.located[dataset];
            (
                loc.blk_off_size,
                self.offset_size as usize,
                4 + 1 + 1 + self.offset_size as usize,
                loc.geom.direct_dblk_nelmts.len(),
                loc.idx_blk_elmts as usize,
                loc.ea_elem_size,
                loc.index_block_addr,
                loc.ea_addr as u64,
                loc.page_nelmts,
            )
        };

        let bitmap = vec![0u8; self.sb_bitmap_size(ndblks, dblk_nelmts, page_nelmts)];
        let undef = vec![undef_addr(self.offset_size); ndblks as usize];
        let aesb = crate::chunked_write::build_aesb(
            ea_addr,
            sb_block_offset,
            &bitmap,
            &undef,
            self.offset_size,
            blk_off,
            0,
        );
        let new_addr = self.append_bytes(&aesb)?;

        // Link the super block from the index block and re-checksum it.
        let slot_off = ib_addr + ib_prefix + idx_elems * elem_size + ndblk_addrs * os + sblk_j * os;
        self.write_addr_at(slot_off, new_addr)?;
        self.rechecksum_index_block(dataset)?;
        Ok(new_addr)
    }

    /// Allocate a fresh non-paged data block (`EADB`) at EOF with every element
    /// slot undefined, returning its address. The element will be written into
    /// slot 0 by the caller. For unfiltered datasets each element is one
    /// offset-sized address.
    fn alloc_undef_data_block(
        &mut self,
        ea_addr: usize,
        dblk_nelmts: u64,
        block_offset_rel: u64,
        blk_off: usize,
    ) -> Result<u64, Error> {
        let os = self.offset_size;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"EADB");
        buf.push(0); // version
        buf.push(0); // client id (unfiltered)
        crate::chunked_write::write_ea_addr(&mut buf, ea_addr as u64, os);
        buf.extend_from_slice(&block_offset_rel.to_le_bytes()[..blk_off]);
        let undef = undef_addr(os);
        for _ in 0..dblk_nelmts {
            crate::chunked_write::write_ea_addr(&mut buf, undef, os);
        }
        let cks = jenkins_lookup3(&buf);
        buf.extend_from_slice(&cks.to_le_bytes());
        self.append_bytes(&buf)
    }

    /// Allocate a fresh *paged* data block (`EADB`) at EOF: a header carrying its
    /// own checksum, followed by `dblk_nelmts / page_nelmts` fully-undefined
    /// pages (each `page_nelmts` undefined elements + a checksum). The owning
    /// super block's page-init bitmap stays all-zero until pages are written.
    fn alloc_undef_paged_data_block(
        &mut self,
        ea_addr: usize,
        dblk_nelmts: u64,
        page_nelmts: u64,
        block_offset_rel: u64,
        blk_off: usize,
        elem_size: usize,
    ) -> Result<u64, Error> {
        let os = self.offset_size;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"EADB");
        buf.push(0); // version
        buf.push(0); // client id (unfiltered)
        crate::chunked_write::write_ea_addr(&mut buf, ea_addr as u64, os);
        buf.extend_from_slice(&block_offset_rel.to_le_bytes()[..blk_off]);
        let header_cks = jenkins_lookup3(&buf);
        buf.extend_from_slice(&header_cks.to_le_bytes());

        let undef = undef_addr(os);
        let npages = (dblk_nelmts / page_nelmts) as usize;
        for _ in 0..npages {
            let mut page = Vec::with_capacity(page_nelmts as usize * elem_size + 4);
            for _ in 0..page_nelmts {
                crate::chunked_write::write_ea_addr(&mut page, undef, os);
            }
            let page_cks = jenkins_lookup3(&page);
            page.extend_from_slice(&page_cks.to_le_bytes());
            buf.extend_from_slice(&page);
        }
        self.append_bytes(&buf)
    }

    /// Recompute the index block checksum from the located dataset metadata.
    fn rechecksum_index_block(&mut self, dataset: &str) -> Result<(), Error> {
        let loc = &self.located[dataset];
        let os = self.offset_size as usize;
        let ib_prefix = 4 + 1 + 1 + os;
        let ndblk_addrs = loc.geom.direct_dblk_nelmts.len();
        let nsblk_addrs = loc.geom.nsblk_addrs;
        let cks_off = loc.index_block_addr
            + ib_prefix
            + loc.idx_blk_elmts as usize * loc.ea_elem_size
            + ndblk_addrs * os
            + nsblk_addrs * os;
        self.rechecksum_range(loc.index_block_addr, cks_off)
    }

    fn rechecksum_super_block(
        &mut self,
        sblk_addr: u64,
        ndblks: u64,
        dblk_nelmts: u64,
        page_nelmts: u64,
        blk_off: usize,
    ) -> Result<(), Error> {
        let os = self.offset_size as usize;
        let prefix = 4 + 1 + 1 + os + blk_off;
        let bitmap = self.sb_bitmap_size(ndblks, dblk_nelmts, page_nelmts);
        let cks_off = sblk_addr as usize + prefix + bitmap + ndblks as usize * os;
        self.rechecksum_range(sblk_addr as usize, cks_off)
    }

    /// Recompute the Jenkins checksum over `[start, cks_off)` and write it at
    /// `cks_off`.
    fn rechecksum_range(&mut self, start: usize, cks_off: usize) -> Result<(), Error> {
        let cks = jenkins_lookup3(&self.data[start..cks_off]);
        self.write_at(cks_off, &cks.to_le_bytes())
    }

    /// Patch the six EA header statistics and recompute the header checksum.
    fn update_ea_header(&mut self, dataset: &str, num_chunks: u64) -> Result<(), Error> {
        let (ea_addr, stats, ls) = {
            let loc = &self.located[dataset];
            let stats = ea_compute_stats(
                &loc.geom,
                loc.idx_blk_elmts,
                loc.ea_elem_size,
                loc.page_nelmts,
                self.offset_size,
                loc.blk_off_size,
                num_chunks,
            );
            (loc.ea_addr, stats, self.length_size)
        };
        let write_stat = |this: &mut Self, k: usize, v: u64| -> Result<(), Error> {
            let off = ea_addr + 12 + k * ls as usize;
            this.write_length_at(off, v)
        };
        write_stat(self, 0, stats.nsuper_blks)?;
        write_stat(self, 1, stats.super_blk_size)?;
        write_stat(self, 2, stats.ndata_blks)?;
        write_stat(self, 3, stats.data_blk_size)?;
        write_stat(self, 4, stats.max_idx_set)?;
        write_stat(self, 5, stats.nelmts)?;
        // EAHD checksum covers signature .. just before the trailing checksum.
        let aehd_size = ExtensibleArrayHeader::serialized_size(self.offset_size, self.length_size);
        let cks_off = ea_addr + aehd_size - 4;
        self.rechecksum_range(ea_addr, cks_off)
    }

    fn patch_dimension(&mut self, dataset: &str, new_dim: u64) -> Result<(), Error> {
        let (dim0_off, chunk_start, chunk_msg_end) = {
            let loc = &self.located[dataset];
            (loc.dim0_off, loc.ohdr_chunk_start, loc.ohdr_chunk_msg_end)
        };
        self.write_length_at(dim0_off, new_dim)?;
        // Recompute the containing object-header chunk's checksum.
        self.rechecksum_range(chunk_start, chunk_msg_end)
    }

    fn patch_superblock_eof(&mut self) -> Result<(), Error> {
        let eof = self.data.len() as u64;
        self.superblock.eof_address = eof;
        let bytes = self.superblock.serialize();
        self.write_at(self.sb_sig_off, &bytes)
    }

    // ----- raw byte / address IO ----------------------------------------

    /// Append `bytes` at end-of-file (in memory and on disk), returning the
    /// address (offset) at which they were written.
    ///
    /// The disk write happens now, not at the next `sync`, so a later in-place
    /// patch of this region lands on top of bytes that already exist on disk. A
    /// failed write is propagated rather than swallowed: the append aborts
    /// before any later phase publishes the new length, so a partial write is
    /// never reported as success.
    fn append_bytes(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        let addr = self.data.len() as u64;
        self.data.extend_from_slice(bytes);
        self.handle.seek(SeekFrom::Start(addr)).map_err(Error::Io)?;
        self.handle.write_all(bytes).map_err(Error::Io)?;
        Ok(addr)
    }

    fn write_at(&mut self, offset: usize, bytes: &[u8]) -> Result<(), Error> {
        self.data[offset..offset + bytes.len()].copy_from_slice(bytes);
        self.handle
            .seek(SeekFrom::Start(offset as u64))
            .map_err(Error::Io)?;
        self.handle.write_all(bytes).map_err(Error::Io)?;
        Ok(())
    }

    fn write_addr_at(&mut self, offset: usize, addr: u64) -> Result<(), Error> {
        match self.offset_size {
            4 => self.write_at(offset, &(addr as u32).to_le_bytes()),
            _ => self.write_at(offset, &addr.to_le_bytes()),
        }
    }

    fn write_length_at(&mut self, offset: usize, val: u64) -> Result<(), Error> {
        match self.length_size {
            4 => self.write_at(offset, &(val as u32).to_le_bytes()),
            _ => self.write_at(offset, &val.to_le_bytes()),
        }
    }

    fn read_addr_at(&self, offset: usize) -> u64 {
        match self.offset_size {
            4 => u32::from_le_bytes(self.data[offset..offset + 4].try_into().unwrap()) as u64,
            _ => u64::from_le_bytes(self.data[offset..offset + 8].try_into().unwrap()),
        }
    }

    /// Flush buffered writes to durable storage.
    fn sync(&mut self) -> Result<(), Error> {
        self.handle.flush().map_err(Error::Io)?;
        self.handle.sync_data().map_err(Error::Io)?;
        Ok(())
    }
}

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

// ---------------------------------------------------------------------------
// Extensible-array element location (where element `e` lives in the structure)
// ---------------------------------------------------------------------------

enum Parent {
    /// Reached by a direct data-block pointer in the index block.
    IndexDirect { ordinal: usize },
    /// Reached via a super block (`sblk_j`-th super-block pointer), as the
    /// `dblk_local`-th data block within it.
    Super { sblk_j: usize, dblk_local: usize },
}

struct DataBlockLoc {
    db_start: u64,
    dblk_nelmts: u64,
    ndblks: u64,
    sb_block_offset: u64,
    parent: Parent,
}

/// Locate the data block containing element `e` (which is `>= idx_blk_elmts`).
fn locate_data_block(geom: &EaGeometry, idx_blk_elmts: u64, e: u64) -> DataBlockLoc {
    // Direct data blocks first.
    let mut elem = idx_blk_elmts;
    for (ordinal, &dn) in geom.direct_dblk_nelmts.iter().enumerate() {
        if e < elem + dn {
            return DataBlockLoc {
                db_start: elem,
                dblk_nelmts: dn,
                ndblks: 1,
                sb_block_offset: 0,
                parent: Parent::IndexDirect { ordinal },
            };
        }
        elem += dn;
    }
    // Then super blocks.
    for j in 0..geom.nsblk_addrs {
        let (ndblks, dn) = geom.sblks[geom.first_indirect_sblk + j];
        let span = ndblks * dn;
        if e < elem + span {
            let sb_block_offset = elem - idx_blk_elmts;
            let within = e - elem;
            let dblk_local = (within / dn) as usize;
            let db_start = elem + dblk_local as u64 * dn;
            return DataBlockLoc {
                db_start,
                dblk_nelmts: dn,
                ndblks,
                sb_block_offset,
                parent: Parent::Super {
                    sblk_j: j,
                    dblk_local,
                },
            };
        }
        elem += span;
    }
    // Should be unreachable for valid `e`; return a sentinel that callers reject.
    DataBlockLoc {
        db_start: e,
        dblk_nelmts: u64::MAX,
        ndblks: 0,
        sb_block_offset: 0,
        parent: Parent::IndexDirect { ordinal: 0 },
    }
}

// ---------------------------------------------------------------------------
// Object-header walk that records message file offsets + chunk checksum regions
// ---------------------------------------------------------------------------

struct WalkedMessage {
    msg_type: MessageType,
    data_off: usize,
    size: usize,
    /// Containing chunk's checksum coverage: `[chunk_start, chunk_msg_end)`, with
    /// the 4-byte checksum stored at `chunk_msg_end`.
    chunk_start: usize,
    chunk_msg_end: usize,
}

struct Walk {
    messages: Vec<WalkedMessage>,
}

/// Walk a version-2 object header (chunk 0 plus any continuation chunks),
/// recording each message's data offset and its containing chunk's checksum
/// region. Only the message types the append writer needs are retained.
fn walk_v2_object_header(
    data: &[u8],
    offset: usize,
    offset_size: u8,
    length_size: u8,
) -> Result<Walk, Error> {
    if offset + 6 > data.len() || &data[offset..offset + 4] != b"OHDR" {
        return Err(Error::Format(FormatError::InvalidObjectHeaderSignature));
    }
    let flags = data[offset + 5];
    let mut pos = offset + 6;
    if flags & 0x20 != 0 {
        pos += 16; // timestamps
    }
    if flags & 0x10 != 0 {
        pos += 4; // attr storage phase-change
    }
    let chunk_size_width = 1usize << (flags & 0x03);
    let chunk0_size = read_uint(data, pos, chunk_size_width)? as usize;
    pos += chunk_size_width;
    let chunk0_start = offset;
    let chunk0_msg_start = pos;
    let chunk0_msg_end = chunk0_msg_start + chunk0_size;

    let has_creation_order = flags & 0x04 != 0;
    let mut messages = Vec::new();
    let mut continuations: Vec<(usize, usize)> = Vec::new();

    walk_messages(
        data,
        chunk0_msg_start,
        chunk0_msg_end,
        chunk0_start,
        chunk0_msg_end,
        has_creation_order,
        offset_size,
        length_size,
        &mut messages,
        &mut continuations,
    )?;

    let mut guard = 256;
    while let Some((cont_off, cont_len)) = continuations.pop() {
        guard -= 1;
        if guard == 0 {
            return Err(Error::Format(FormatError::NestingDepthExceeded));
        }
        if cont_off + 4 > data.len() || &data[cont_off..cont_off + 4] != b"OCHK" {
            return Err(Error::Format(FormatError::InvalidObjectHeaderSignature));
        }
        let msg_start = cont_off + 4;
        let msg_end = cont_off + cont_len - 4; // checksum is the last 4 bytes
        walk_messages(
            data,
            msg_start,
            msg_end,
            cont_off,
            msg_end,
            has_creation_order,
            offset_size,
            length_size,
            &mut messages,
            &mut continuations,
        )?;
    }

    Ok(Walk { messages })
}

#[allow(clippy::too_many_arguments)]
fn walk_messages(
    data: &[u8],
    start: usize,
    end: usize,
    chunk_start: usize,
    chunk_msg_end: usize,
    has_creation_order: bool,
    offset_size: u8,
    length_size: u8,
    messages: &mut Vec<WalkedMessage>,
    continuations: &mut Vec<(usize, usize)>,
) -> Result<(), Error> {
    let msg_header_size = if has_creation_order { 6 } else { 4 };
    let mut pos = start;
    while pos + msg_header_size <= end {
        let msg_type_raw = data[pos] as u16;
        let msg_data_size = u16::from_le_bytes([data[pos + 1], data[pos + 2]]) as usize;
        pos += msg_header_size;
        if pos + msg_data_size > end {
            break; // padding
        }
        let msg_type = MessageType::from_u16(msg_type_raw);
        if msg_type == MessageType::ObjectHeaderContinuation {
            let cont_off = read_uint(data, pos, offset_size as usize)? as usize;
            let cont_len =
                read_uint(data, pos + offset_size as usize, length_size as usize)? as usize;
            continuations.push((cont_off, cont_len));
        } else {
            messages.push(WalkedMessage {
                msg_type,
                data_off: pos,
                size: msg_data_size,
                chunk_start,
                chunk_msg_end,
            });
        }
        pos += msg_data_size;
    }
    Ok(())
}

fn read_uint(data: &[u8], pos: usize, size: usize) -> Result<u64, Error> {
    if pos + size > data.len() {
        return Err(Error::Format(FormatError::UnexpectedEof {
            expected: pos + size,
            available: data.len(),
        }));
    }
    let mut v = 0u64;
    for i in 0..size {
        v |= (data[pos + i] as u64) << (8 * i);
    }
    Ok(v)
}

#[cfg(test)]
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
