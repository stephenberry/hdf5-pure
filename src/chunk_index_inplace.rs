//! In-place Extensible-Array chunk-index growth, shared by the SWMR append
//! writer ([`crate::swmr_writer`]) and the general append writer
//! ([`crate::append_writer`]).
//!
//! Both writers grow a one-dimensional, unlimited, Extensible-Array-indexed
//! dataset *in place*: an appended chunk is written at end-of-file, its record is
//! stored into an element slot of the chunk index, the index grows by appending
//! new blocks only when a block boundary is crossed (never relocating existing
//! data), and the dataspace dimension and array-header counts are patched. Writes
//! land child-before-parent with `fsync` barriers so a crash (and, for SWMR, a
//! concurrent reader) only ever observes a consistent prefix, with the dataspace
//! dimension published last as the single commit point.
//!
//! This module owns the byte-level mechanics that do not depend on *why* the
//! append is happening: the in-memory-mirror file cursor ([`InPlaceFile`]), the
//! per-dataset geometry cache ([`Located`]), and the element-slot / block /
//! super-block writes that maintain the Extensible Array. It is element-width
//! agnostic: the same code path stores a bare address for an unfiltered array
//! (client id 0) or the full `address + compressed_size + filter_mask` record for
//! a filtered array (client id 1), selected by the array header's client id. The
//! callers layer their own policy on top: SWMR sets the superblock consistency
//! flag and refuses filters; the general writer takes an exclusive lock, accepts
//! filters, and relocates a partial trailing chunk.

use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::checksum::jenkins_lookup3;
use crate::chunked_write::{ea_compute_stats, split_into_chunks, write_ea_addr};
use crate::convert::TryToUsize;
use crate::data_layout::DataLayout;
use crate::dataspace::Dataspace;
use crate::datatype::Datatype;
use crate::error::{Error, FormatError};
use crate::extensible_array::{EaGeometry, ExtensibleArrayHeader};
use crate::file_lock::{self, FileLocking};
use crate::filter_pipeline::FilterPipeline;
use crate::filters::{ChunkContext, compress_chunk, decompress_chunk};
use crate::message_type::MessageType;
use crate::signature;
use crate::source::{BytesSource, Source};
use crate::superblock::Superblock;

/// The undefined-address sentinel for a given offset size.
pub(crate) fn undef_addr(offset_size: u8) -> u64 {
    match offset_size {
        4 => 0xFFFF_FFFF,
        _ => u64::MAX,
    }
}

pub(crate) fn is_undef(addr: u64, offset_size: u8) -> bool {
    addr == undef_addr(offset_size)
}

/// Push one undefined Extensible-Array element to `buf`: an offset-sized
/// all-`0xFF` address, followed (for a filtered array whose element is wider than
/// one address) by zeroed compressed-size and filter-mask fields. Mirrors
/// `chunked_write::write_undefined_element` so a freshly-allocated block matches
/// what the bulk writer and reader expect.
fn push_undef_element(buf: &mut Vec<u8>, offset_size: u8, ea_elem_size: usize) {
    write_ea_addr(buf, undef_addr(offset_size), offset_size);
    for _ in offset_size as usize..ea_elem_size {
        buf.push(0);
    }
}

/// One stored Extensible-Array element: the chunk address, plus (for filtered
/// arrays) the stored/compressed chunk size and this chunk's filter mask. The
/// size and mask are ignored for unfiltered arrays (client id 0).
#[derive(Clone, Copy, Debug)]
pub(crate) struct ElemRecord {
    pub addr: u64,
    pub stored_size: u64,
    pub filter_mask: u32,
}

impl ElemRecord {
    /// A bare-address record (the size/mask are unused for unfiltered arrays).
    pub(crate) fn addr_only(addr: u64) -> Self {
        Self {
            addr,
            stored_size: 0,
            filter_mask: 0,
        }
    }
}

/// A file opened for in-place appends: a read/write OS handle plus a full
/// in-memory mirror of the file (`O(file size)` memory) kept in lock-step with
/// on-disk writes, so existing structures are read and checksums recomputed
/// without hitting the disk.
pub(crate) struct InPlaceFile {
    handle: std::fs::File,
    /// Full in-memory mirror; every write updates this and the disk together.
    data: Vec<u8>,
    pub offset_size: u8,
    pub length_size: u8,
    /// Offset of the superblock signature within the file.
    sb_sig_off: usize,
    pub superblock: Superblock,
}

impl InPlaceFile {
    /// Open an existing latest-format HDF5 file for in-place appends, reading it
    /// into memory. When `lock` is `Some`, an exclusive OS byte-range lock is
    /// taken and held by the retained handle for the session's life (the general
    /// append writer); when `None`, no lock is taken (the SWMR writer, which is
    /// single-writer by contract and must not block concurrent readers).
    ///
    /// Rejects a pre-v2 superblock (this crate only serializes the v2/v3 layout,
    /// so patching a v0/v1 superblock in place would clobber it) and a non-zero
    /// base address (userblock files store addresses relative to the base, which
    /// the in-place slot math does not apply).
    pub(crate) fn open<P: AsRef<Path>>(
        path: P,
        lock: Option<FileLocking>,
        unsupported: fn(&'static str) -> Error,
    ) -> Result<Self, Error> {
        let path = path.as_ref();
        let mut handle = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(Error::Io)?;
        if let Some(locking) = lock {
            file_lock::acquire_exclusive(&handle, locking, path)?;
        }
        let mut data = Vec::new();
        handle.read_to_end(&mut data).map_err(Error::Io)?;

        let sb_sig_off = signature::find_signature(&data)?;
        let superblock = Superblock::parse(&data, sb_sig_off)?;
        if superblock.version < 2 {
            return Err(unsupported(
                "in-place append requires a latest-format file (v2/v3 superblock)",
            ));
        }
        if superblock.base_address != 0 {
            return Err(unsupported(
                "files with a userblock (non-zero base address) are not supported",
            ));
        }
        let offset_size = superblock.offset_size;
        let length_size = superblock.length_size;
        Ok(Self {
            handle,
            data,
            offset_size,
            length_size,
            sb_sig_off,
            superblock,
        })
    }

    pub(crate) fn data(&self) -> &[u8] {
        &self.data
    }

    /// Append `bytes` at end-of-file (in memory and on disk), returning the
    /// address at which they were written. The disk write happens now, not at the
    /// next `sync`, so a later in-place patch of this region lands on top of bytes
    /// that already exist on disk.
    pub(crate) fn append_bytes(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        let addr = self.data.len() as u64;
        self.data.extend_from_slice(bytes);
        self.handle.seek(SeekFrom::Start(addr)).map_err(Error::Io)?;
        self.handle.write_all(bytes).map_err(Error::Io)?;
        Ok(addr)
    }

    pub(crate) fn write_at(&mut self, offset: usize, bytes: &[u8]) -> Result<(), Error> {
        self.data[offset..offset + bytes.len()].copy_from_slice(bytes);
        self.handle
            .seek(SeekFrom::Start(offset as u64))
            .map_err(Error::Io)?;
        self.handle.write_all(bytes).map_err(Error::Io)?;
        Ok(())
    }

    /// Advance the superblock's recorded end-of-file to the current mirror length
    /// and rewrite the superblock.
    pub(crate) fn patch_superblock_eof(&mut self) -> Result<(), Error> {
        let eof = self.data.len() as u64;
        self.superblock.eof_address = eof;
        let bytes = self.superblock.serialize();
        self.write_at(self.sb_sig_off, &bytes)
    }

    /// Set the superblock consistency flags, rewrite the superblock, and flush.
    /// Used by the SWMR writer to raise/clear the SWMR-write flag.
    pub(crate) fn set_consistency_flags(&mut self, flags: u32) -> Result<(), Error> {
        self.superblock.consistency_flags = flags;
        let bytes = self.superblock.serialize();
        self.write_at(self.sb_sig_off, &bytes)?;
        self.sync()
    }

    /// Flush buffered writes to durable storage.
    pub(crate) fn sync(&mut self) -> Result<(), Error> {
        self.handle.flush().map_err(Error::Io)?;
        self.handle.sync_data().map_err(Error::Io)?;
        Ok(())
    }
}

/// Writable byte-level I/O the Extensible-Array growth engine ([`Located`])
/// depends on, extending the read-only [`Source`] seam with in-place mutation.
/// Its owners today are [`InPlaceFile`] (the append/SWMR writers' own mirror +
/// handle) and [`EditSession`](crate::EditSession)'s borrowed mirror; issue #147's
/// bounded backend adds a store with no mirror at all, which is why every engine
/// *read* goes through [`Source`] (bounded, random-access) rather than a
/// whole-file `&[u8]`. Genericizing the engine over this trait lets a long-lived
/// `EditSession` drive an O(1) in-place append against its *own* single mirror and
/// exclusive lock rather than constructing a second `InPlaceFile` (which would
/// take a second exclusive lock and keep a divergent mirror). Each owner keeps its
/// own crash-safety discipline for the primitives — `InPlaceFile` mirrors before
/// disk, the edit mirror writes disk before mirror — while sharing the checksummed
/// slot/block/super-block mechanics through the derived operations below.
pub(crate) trait Store: Source {
    /// This file's address (offset) field width in bytes.
    fn offset_size(&self) -> u8;
    /// This file's length field width in bytes.
    fn length_size(&self) -> u8;

    /// Append `bytes` at end-of-file, returning their start address. The bytes are
    /// made durable-visible immediately (not buffered until the next `sync`), so a
    /// later in-place patch of the region lands on bytes that already exist.
    fn append_bytes(&mut self, bytes: &[u8]) -> Result<u64, Error>;
    /// Overwrite `[offset, offset + bytes.len())` in place.
    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error>;
    /// Advance the superblock's recorded end-of-file to the store's current
    /// logical length and rewrite the superblock.
    fn patch_superblock_eof(&mut self) -> Result<(), Error>;
    /// Flush buffered writes to durable storage (an `fsync` barrier).
    fn sync(&mut self) -> Result<(), Error>;

    /// Write an offset-sized address at `offset`.
    fn write_addr_at(&mut self, offset: u64, addr: u64) -> Result<(), Error> {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the 4-byte arm is taken only when this file's offset_size is 4 bytes"
        )]
        match self.offset_size() {
            4 => self.write_at(offset, &(addr as u32).to_le_bytes()),
            _ => self.write_at(offset, &addr.to_le_bytes()),
        }
    }

    /// Write a length-sized value at `offset`.
    fn write_length_at(&mut self, offset: u64, val: u64) -> Result<(), Error> {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the 4-byte arm is taken only when this file's length_size is 4 bytes"
        )]
        match self.length_size() {
            4 => self.write_at(offset, &(val as u32).to_le_bytes()),
            _ => self.write_at(offset, &val.to_le_bytes()),
        }
    }

    /// Read an offset-sized address at `offset`.
    fn read_addr_at(&self, offset: u64) -> Result<u64, Error> {
        let mut buf = [0u8; 8];
        let width = if self.offset_size() == 4 { 4 } else { 8 };
        self.read_at(offset, &mut buf[..width])?;
        Ok(u64::from_le_bytes(buf))
    }

    /// Recompute the Jenkins checksum over `[start, cks_off)` and store it at
    /// `cks_off`.
    fn rechecksum_range(&mut self, start: u64, cks_off: u64) -> Result<(), Error> {
        let bytes = self.read_exact_at(start, (cks_off - start).to_usize()?)?;
        let cks = jenkins_lookup3(&bytes);
        self.write_at(cks_off, &cks.to_le_bytes())
    }
}

impl Source for InPlaceFile {
    fn len(&self) -> u64 {
        self.data.len() as u64
    }
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), crate::error::FormatError> {
        BytesSource::new(&self.data).read_at(offset, buf)
    }
}

impl Store for InPlaceFile {
    fn offset_size(&self) -> u8 {
        self.offset_size
    }
    fn length_size(&self) -> u8 {
        self.length_size
    }
    fn append_bytes(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        InPlaceFile::append_bytes(self, bytes)
    }
    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
        InPlaceFile::write_at(self, offset.to_usize()?, bytes)
    }
    fn patch_superblock_eof(&mut self) -> Result<(), Error> {
        InPlaceFile::patch_superblock_eof(self)
    }
    fn sync(&mut self) -> Result<(), Error> {
        InPlaceFile::sync(self)
    }
}

/// Absolute file offsets of the object-header messages a caller may need to
/// parse after locating a dataset.
pub(crate) struct MessageSpans {
    /// `(data_off, size)` of the Datatype message.
    pub datatype: (u64, usize),
    /// `(data_off, size)` of the Filter Pipeline message, when present.
    pub filter: Option<(u64, usize)>,
}

/// Result of locating a dataset: its maintained geometry plus the message spans
/// and filter status the caller needs to decide policy.
pub(crate) struct LocateResult {
    pub located: Located,
    pub spans: MessageSpans,
    pub has_filters: bool,
}

/// Metadata located once per dataset, then maintained across appends.
pub(crate) struct Located {
    /// File offset of the dataspace message's first current-dimension value.
    pub dim0_off: u64,
    /// Current length along the unlimited (axis-0) dimension.
    pub current_dim: u64,
    /// Object-header chunk that contains the dataspace message: the byte range
    /// whose Jenkins checksum must be recomputed after patching the dimension.
    /// The checksum itself occupies `chunk_msg_end .. chunk_msg_end + 4`.
    pub ohdr_chunk_start: u64,
    pub ohdr_chunk_msg_end: u64,

    /// Elements per chunk along axis 0 (the only varying axis for rank 1).
    pub chunk_elems: u64,
    /// Bytes per dataset element (datatype size).
    pub elem_bytes: usize,
    /// Bytes per chunk (uncompressed).
    pub chunk_bytes: usize,

    /// Extensible Array client id: 0 = unfiltered (bare-address element), 1 =
    /// filtered (`address + compressed_size + filter_mask` element).
    pub client_id: u8,
    /// Extensible Array header address and derived geometry.
    pub ea_addr: u64,
    pub geom: EaGeometry,
    pub idx_blk_elmts: u64,
    /// Size of one stored EA element in bytes (offset size for unfiltered; wider
    /// for filtered).
    pub ea_elem_size: usize,
    pub page_nelmts: u64,
    /// Block-offset field width inside EA blocks (= ceil(max_nelmts_bits / 8)).
    pub blk_off_size: usize,
    /// Address of the EA index block (`EAIB`).
    pub index_block_addr: u64,
    /// Current number of chunks indexed (EA element count).
    pub num_chunks: u64,
}

impl Located {
    /// Locate the dataset whose object-header address is `oh_addr`, deriving its
    /// dataspace/layout/Extensible-Array geometry. Filter-neutral: it records
    /// whether the dataset is filtered and returns the datatype/filter message
    /// spans, leaving the accept/reject policy to the caller. `unsupported`
    /// builds the caller's "unsupported target" error so each writer reports
    /// through its own variant. Path resolution is the caller's job: an owned
    /// handle already carries its address, and the path-addressed writers
    /// resolve against their own mirrors.
    ///
    /// Every read is a bounded [`Source`] window (the object-header chunks, one
    /// message span, the EA header), so this works over a store with no
    /// whole-file mirror.
    pub(crate) fn locate_at<F: Store>(
        file: &F,
        oh_addr: u64,
        unsupported: fn(&'static str) -> Error,
    ) -> Result<LocateResult, Error> {
        let os = file.offset_size();
        let ls = file.length_size();

        let walk = walk_v2_object_header(file, oh_addr, os, ls)?;

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
        let datatype_msg = walk
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::Datatype)
            .ok_or(Error::MissingMessage(MessageType::Datatype))?;
        let filter_msg = walk
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::FilterPipeline);

        // Parse the dataspace: must be rank 1 with one unlimited dimension.
        let ds_bytes = file.read_metadata_at(dataspace_msg.data_off, dataspace_msg.size)?;
        let dataspace = Dataspace::parse(&ds_bytes, ls)?;
        if dataspace.rank != 1 {
            return Err(unsupported("only rank-1 datasets are supported"));
        }
        match &dataspace.max_dimensions {
            Some(maxs) if maxs.first() == Some(&u64::MAX) => {}
            _ => {
                return Err(unsupported("dataset has no unlimited (maxshape) dimension"));
            }
        }
        let current_dim = dataspace.dimensions[0];
        // v2 dataspace header is 4 bytes (version, rank, flags, type); dim 0 follows.
        let dim0_off = dataspace_msg.data_off + 4;

        // Parse the layout: must be a v4 chunked Extensible Array (index type 4).
        let layout_bytes = file.read_metadata_at(layout_msg.data_off, layout_msg.size)?;
        let layout = DataLayout::parse(&layout_bytes, os, ls)?;
        let (ea_addr, chunk_dims) = match layout {
            DataLayout::Chunked {
                chunk_index_type: Some(4),
                btree_address: Some(addr),
                chunk_dimensions,
                ..
            } => (addr, chunk_dimensions),
            DataLayout::Chunked {
                chunk_index_type: Some(4),
                btree_address: None,
                ..
            } => {
                // An Extensible Array whose header/index block is not allocated
                // yet: an empty dataset the C library created without writing any
                // chunk (this crate's writer allocates it eagerly). In-place
                // growth needs an existing index; make the first append through
                // `Dataset::append_staged`, which materializes the index, or
                // create the dataset with initial data.
                return Err(unsupported(
                    "the dataset's extensible-array index is not allocated yet (an empty \
                     dataset with no chunks); write initial data at creation or make the \
                     first append with Dataset::append_staged",
                ));
            }
            _ => {
                return Err(unsupported(
                    "only Extensible-Array-indexed chunked datasets are supported",
                ));
            }
        };
        // chunk_dimensions for a v4 layout includes the element-size pseudo
        // dimension as its last entry; the leading entry is the axis-0 chunk size.
        if chunk_dims.len() != 2 {
            return Err(unsupported(
                "unexpected chunk dimensionality (expected a rank-1 chunked layout)",
            ));
        }
        let chunk_elems = chunk_dims[0] as u64;
        let elem_bytes = chunk_dims[1] as usize;
        if elem_bytes == 0 {
            // A zero element-size pseudo-dimension is a malformed layout; refuse
            // rather than divide by it when validating an append length.
            return Err(unsupported("dataset has a zero-sized element"));
        }
        let chunk_bytes = chunk_elems.to_usize()? * elem_bytes;

        let ea_header = ExtensibleArrayHeader::parse_from_source(file, ea_addr, os, ls)?;
        let has_filters = filter_msg.is_some();
        if (ea_header.client_id == 1) != has_filters {
            return Err(unsupported(
                "dataset filter metadata is inconsistent (chunk-index client id \
                 disagrees with the filter pipeline)",
            ));
        }
        // A filtered element is `address + compressed_size + filter_mask`, so its
        // stored width must leave room for at least a one-byte size field. Reject
        // a corrupt header whose element_size is too small before the width
        // arithmetic (`ea_elem_size - offset_size - 4`) would underflow.
        if ea_header.client_id == 1 && (ea_header.element_size as usize) < os as usize + 5 {
            return Err(unsupported(
                "malformed filtered extensible-array element width",
            ));
        }
        let geom = EaGeometry::from_header(&ea_header);
        let page_nelmts = 1u64 << ea_header.max_dblk_nelmts_bits;
        let blk_off_size = (ea_header.max_nelmts_bits as usize).div_ceil(8);
        let index_block_addr = ea_header.index_block_address;
        // The dataspace dimension is the single commit point; the EA element
        // count is published one step earlier. If a prior writer crashed between
        // the two, the on-disk EA count is ahead of the committed dimension. Seed
        // the chunk count from the *committed* dimension so the next append rolls
        // forward from the last commit -- overwriting any slots a crashed writer
        // wrote but never committed -- instead of appending past them.
        let num_chunks = if chunk_elems == 0 {
            ea_header.num_elements
        } else {
            current_dim.div_ceil(chunk_elems)
        };

        Ok(LocateResult {
            located: Located {
                dim0_off,
                current_dim,
                ohdr_chunk_start: dataspace_msg.chunk_start,
                ohdr_chunk_msg_end: dataspace_msg.chunk_msg_end,
                chunk_elems,
                elem_bytes,
                chunk_bytes,
                client_id: ea_header.client_id,
                ea_addr,
                geom,
                idx_blk_elmts: ea_header.idx_blk_elmts as u64,
                ea_elem_size: ea_header.element_size as usize,
                page_nelmts,
                blk_off_size,
                index_block_addr,
                num_chunks,
            },
            spans: MessageSpans {
                datatype: (datatype_msg.data_off, datatype_msg.size),
                filter: filter_msg.map(|m| (m.data_off, m.size)),
            },
            has_filters,
        })
    }

    /// Write element record `rec` at byte offset `off` (a slot known to exist).
    /// For an unfiltered array only the address is written; for a filtered array
    /// the full `address + compressed_size + filter_mask` record is written,
    /// refusing a stored size that does not fit the array's fixed element width.
    fn write_element_at<F: Store>(
        &self,
        file: &mut F,
        off: u64,
        rec: ElemRecord,
    ) -> Result<(), Error> {
        if self.client_id == 0 {
            return file.write_addr_at(off, rec.addr);
        }
        let os = file.offset_size() as usize;
        let csz = self.ea_elem_size - os - 4;
        if csz < 8 && rec.stored_size >= (1u64 << (8 * csz)) {
            return Err(Error::AppendUnsupported(
                "recompressed chunk size exceeds the dataset's extensible-array element width",
            ));
        }
        file.write_addr_at(off, rec.addr)?;
        file.write_at(off + os as u64, &rec.stored_size.to_le_bytes()[..csz])?;
        file.write_at(off + (os + csz) as u64, &rec.filter_mask.to_le_bytes())
    }

    /// Read the element record stored at byte offset `off`.
    fn read_element_at<F: Store>(&self, file: &F, off: u64) -> Result<ElemRecord, Error> {
        let os = file.offset_size() as usize;
        if self.client_id == 0 {
            return Ok(ElemRecord::addr_only(file.read_addr_at(off)?));
        }
        // One bounded read of the whole fixed-width element record.
        let elem = file.read_exact_at(off, self.ea_elem_size)?;
        let addr_w = if file.offset_size() == 4 { 4 } else { 8 };
        let mut a = [0u8; 8];
        a[..addr_w].copy_from_slice(&elem[..addr_w]);
        let addr = u64::from_le_bytes(a);
        let csz = self.ea_elem_size - os - 4;
        let mut stored_size = 0u64;
        for i in 0..csz {
            stored_size |= (elem[os + i] as u64) << (8 * i);
        }
        let fm = os + csz;
        let filter_mask = u32::from_le_bytes([elem[fm], elem[fm + 1], elem[fm + 2], elem[fm + 3]]);
        Ok(ElemRecord {
            addr,
            stored_size,
            filter_mask,
        })
    }

    /// Byte offset of element `e`'s record, or `None` when the containing block
    /// (or super block) is not yet allocated, i.e. the element does not exist on
    /// disk yet.
    fn elem_slot_off<F: Store>(&self, file: &F, e: u64) -> Result<Option<u64>, Error> {
        let os = file.offset_size() as usize;
        let elem_size = self.ea_elem_size as u64;
        let idx = self.idx_blk_elmts;
        let blk_off = self.blk_off_size;
        let page_nelmts = self.page_nelmts;

        if e < idx {
            let ib_prefix = (4 + 1 + 1 + os) as u64;
            return Ok(Some(self.index_block_addr + ib_prefix + e * elem_size));
        }

        let region = locate_data_block(&self.geom, idx, e);
        if region.ndblks == 0 {
            return Err(Error::AppendUnsupported(
                "chunk index geometry does not cover the appended element",
            ));
        }
        let is_paged = region.dblk_nelmts > page_nelmts;
        let slot = e - region.db_start;

        // Resolve the owning super block (if any) and the data-block pointer,
        // returning None if either is unallocated.
        let sblk_addr = match region.parent {
            Parent::Super { sblk_j, .. } => {
                let a = self.super_block_addr(file, sblk_j)?;
                if is_undef(a, file.offset_size()) {
                    return Ok(None);
                }
                Some(a)
            }
            Parent::IndexDirect { .. } => None,
        };
        let dblk_ptr_off = self.dblk_ptr_off(sblk_addr, &region, os, blk_off)?;
        let dblk_addr = file.read_addr_at(dblk_ptr_off)?;
        if is_undef(dblk_addr, file.offset_size()) {
            return Ok(None);
        }

        let off = if !is_paged {
            let db_prefix = (4 + 1 + 1 + os + blk_off) as u64;
            dblk_addr + db_prefix + slot * elem_size
        } else {
            let header_size = (4 + 1 + 1 + os + blk_off + 4) as u64;
            let page = slot / page_nelmts;
            let slot_in_page = slot % page_nelmts;
            let page_bytes = page_nelmts * elem_size + 4;
            let page_off = dblk_addr + header_size + page * page_bytes;
            page_off + slot_in_page * elem_size
        };
        Ok(Some(off))
    }

    /// File offset of the data-block-pointer slot for `region` (either a direct
    /// pointer in the index block or a pointer inside the resolved super block).
    fn dblk_ptr_off(
        &self,
        sblk_addr: Option<u64>,
        region: &DataBlockLoc,
        os: usize,
        blk_off: usize,
    ) -> Result<u64, Error> {
        match region.parent {
            Parent::IndexDirect { ordinal } => {
                let ib_prefix = (4 + 1 + 1 + os) as u64;
                Ok(self.index_block_addr
                    + ib_prefix
                    + self.idx_blk_elmts * self.ea_elem_size as u64
                    + (ordinal * os) as u64)
            }
            Parent::Super { dblk_local, .. } => {
                let sblk_addr = sblk_addr.expect("super-block address resolved for a Super parent");
                sb_dblk_slot_off(
                    os,
                    sblk_addr,
                    dblk_local,
                    region.ndblks,
                    region.dblk_nelmts,
                    self.page_nelmts,
                    blk_off,
                )
            }
        }
    }

    /// Read element `e`, or `None` if its slot is not allocated / is undefined.
    pub(crate) fn read_element<F: Store>(
        &self,
        file: &F,
        e: u64,
    ) -> Result<Option<ElemRecord>, Error> {
        match self.elem_slot_off(file, e)? {
            None => Ok(None),
            Some(off) => {
                let rec = self.read_element_at(file, off)?;
                if is_undef(rec.addr, file.offset_size()) {
                    Ok(None)
                } else {
                    Ok(Some(rec))
                }
            }
        }
    }

    /// Store `rec` into element slot `e` of the chunk index, allocating new data
    /// blocks / super blocks at EOF as block boundaries are crossed, and
    /// re-checksumming the touched block. Works for both a fresh insert (the
    /// block is allocated on first touch) and an in-place update of an existing
    /// element (the block already exists, so it is reused rather than
    /// re-allocated). Handles non-paged and paged data blocks.
    pub(crate) fn ea_insert<F: Store>(
        &self,
        file: &mut F,
        e: u64,
        rec: ElemRecord,
    ) -> Result<(), Error> {
        let os = file.offset_size() as usize;
        let elem_size = self.ea_elem_size as u64;
        let idx = self.idx_blk_elmts;
        let blk_off = self.blk_off_size;

        // Inline element slots live directly in the index block.
        if e < idx {
            let ib_prefix = (4 + 1 + 1 + os) as u64;
            let slot_off = self.index_block_addr + ib_prefix + e * elem_size;
            self.write_element_at(file, slot_off, rec)?;
            self.rechecksum_index_block(file)?;
            return Ok(());
        }

        let region = locate_data_block(&self.geom, idx, e);
        if region.ndblks == 0 {
            return Err(Error::AppendUnsupported(
                "chunk index geometry does not cover the appended element",
            ));
        }
        let dblk_nelmts = region.dblk_nelmts;
        let is_paged = dblk_nelmts > self.page_nelmts;
        let slot = e - region.db_start;
        let block_offset_rel = region.db_start - idx;
        let ndblks = region.ndblks;

        // Ensure the owning super block exists (idempotent) for a Super parent.
        let sblk_addr = match region.parent {
            Parent::Super { sblk_j, .. } => Some(self.ensure_super_block(
                file,
                sblk_j,
                region.sb_block_offset,
                ndblks,
                dblk_nelmts,
            )?),
            Parent::IndexDirect { .. } => None,
        };

        // Resolve the data-block address, allocating a fresh block when the
        // parent pointer is undefined (first touch of the block). Keying on the
        // pointer being undefined -- rather than on `slot == 0` -- makes an
        // in-place element update reuse the existing block instead of leaking it.
        let dblk_ptr_off = self.dblk_ptr_off(sblk_addr, &region, os, blk_off)?;
        let existing = file.read_addr_at(dblk_ptr_off)?;
        let dblk_addr = if is_undef(existing, file.offset_size()) {
            let new_addr = if is_paged {
                self.alloc_undef_paged_data_block(file, dblk_nelmts, block_offset_rel)?
            } else {
                self.alloc_undef_data_block(file, dblk_nelmts, block_offset_rel)?
            };
            file.write_addr_at(dblk_ptr_off, new_addr)?;
            match region.parent {
                Parent::IndexDirect { .. } => self.rechecksum_index_block(file)?,
                Parent::Super { .. } => self.rechecksum_super_block(
                    file,
                    sblk_addr.unwrap(),
                    ndblks,
                    dblk_nelmts,
                    self.page_nelmts,
                    blk_off,
                )?,
            }
            new_addr
        } else {
            existing
        };

        if !is_paged {
            let db_prefix = (4 + 1 + 1 + os + blk_off) as u64;
            let elem_off = dblk_addr + db_prefix + slot * elem_size;
            self.write_element_at(file, elem_off, rec)?;
            let cks_off = dblk_addr + db_prefix + dblk_nelmts * elem_size;
            file.rechecksum_range(dblk_addr, cks_off)?;
        } else {
            let page_nelmts = self.page_nelmts;
            let header_size = (4 + 1 + 1 + os + blk_off + 4) as u64;
            let page = slot / page_nelmts;
            let slot_in_page = slot % page_nelmts;
            let page_bytes = page_nelmts * elem_size + 4;
            let page_off = dblk_addr + header_size + page * page_bytes;
            self.write_element_at(file, page_off + slot_in_page * elem_size, rec)?;
            let page_cks_off = page_off + page_nelmts * elem_size;
            file.rechecksum_range(page_off, page_cks_off)?;

            if slot_in_page == 0 {
                let sblk_addr = sblk_addr.unwrap();
                let npages = dblk_nelmts / self.page_nelmts;
                if let Parent::Super { dblk_local, .. } = region.parent {
                    let global_page = dblk_local as u64 * npages + page;
                    self.set_sb_page_bit(file, sblk_addr, blk_off, global_page)?;
                    self.rechecksum_super_block(
                        file,
                        sblk_addr,
                        ndblks,
                        dblk_nelmts,
                        self.page_nelmts,
                        blk_off,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Address of an already-allocated super block (`sblk_j`-th super-block
    /// pointer in the index block); the undefined sentinel when not yet allocated.
    fn super_block_addr<F: Store>(&self, file: &F, sblk_j: usize) -> Result<u64, Error> {
        let os = file.offset_size() as usize;
        let ib_prefix = (4 + 1 + 1 + os) as u64;
        let ndblk_addrs = self.geom.direct_dblk_nelmts.len();
        let slot_off = self.index_block_addr
            + ib_prefix
            + self.idx_blk_elmts * self.ea_elem_size as u64
            + ((ndblk_addrs + sblk_j) * os) as u64;
        file.read_addr_at(slot_off)
    }

    /// Return the address of super block `sblk_j`, allocating an empty one (all
    /// data-block pointers undefined, plus a zeroed page-init bitmap when its data
    /// blocks are paged) at EOF if it does not exist yet.
    fn ensure_super_block<F: Store>(
        &self,
        file: &mut F,
        sblk_j: usize,
        sb_block_offset: u64,
        ndblks: u64,
        dblk_nelmts: u64,
    ) -> Result<u64, Error> {
        let existing = self.super_block_addr(file, sblk_j)?;
        if !is_undef(existing, file.offset_size()) {
            return Ok(existing);
        }
        let os = file.offset_size() as usize;
        let ib_prefix = (4 + 1 + 1 + os) as u64;
        let ndblk_addrs = self.geom.direct_dblk_nelmts.len();

        let bitmap = vec![0u8; sb_bitmap_size(ndblks, dblk_nelmts, self.page_nelmts)?];
        let undef = vec![undef_addr(file.offset_size()); ndblks.to_usize()?];
        let aesb = crate::chunked_write::build_aesb(
            self.ea_addr,
            sb_block_offset,
            &bitmap,
            &undef,
            file.offset_size(),
            self.blk_off_size,
            self.client_id,
        );
        let new_addr = file.append_bytes(&aesb)?;

        let slot_off = self.index_block_addr
            + ib_prefix
            + self.idx_blk_elmts * self.ea_elem_size as u64
            + ((ndblk_addrs + sblk_j) * os) as u64;
        file.write_addr_at(slot_off, new_addr)?;
        self.rechecksum_index_block(file)?;
        Ok(new_addr)
    }

    /// Allocate a fresh non-paged data block (`EADB`) at EOF with every element
    /// slot undefined, returning its address.
    fn alloc_undef_data_block<F: Store>(
        &self,
        file: &mut F,
        dblk_nelmts: u64,
        block_offset_rel: u64,
    ) -> Result<u64, Error> {
        let os = file.offset_size();
        let mut buf = Vec::new();
        buf.extend_from_slice(b"EADB");
        buf.push(0); // version
        buf.push(self.client_id);
        write_ea_addr(&mut buf, self.ea_addr, os);
        buf.extend_from_slice(&block_offset_rel.to_le_bytes()[..self.blk_off_size]);
        for _ in 0..dblk_nelmts {
            push_undef_element(&mut buf, os, self.ea_elem_size);
        }
        let cks = jenkins_lookup3(&buf);
        buf.extend_from_slice(&cks.to_le_bytes());
        file.append_bytes(&buf)
    }

    /// Allocate a fresh *paged* data block (`EADB`) at EOF: a header carrying its
    /// own checksum, followed by `dblk_nelmts / page_nelmts` fully-undefined pages
    /// (each `page_nelmts` undefined elements + a checksum).
    fn alloc_undef_paged_data_block<F: Store>(
        &self,
        file: &mut F,
        dblk_nelmts: u64,
        block_offset_rel: u64,
    ) -> Result<u64, Error> {
        let os = file.offset_size();
        let page_nelmts = self.page_nelmts;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"EADB");
        buf.push(0); // version
        buf.push(self.client_id);
        write_ea_addr(&mut buf, self.ea_addr, os);
        buf.extend_from_slice(&block_offset_rel.to_le_bytes()[..self.blk_off_size]);
        let header_cks = jenkins_lookup3(&buf);
        buf.extend_from_slice(&header_cks.to_le_bytes());

        let npages = (dblk_nelmts / page_nelmts).to_usize()?;
        for _ in 0..npages {
            let mut page = Vec::with_capacity(page_nelmts.to_usize()? * self.ea_elem_size + 4);
            for _ in 0..page_nelmts {
                push_undef_element(&mut page, os, self.ea_elem_size);
            }
            let page_cks = jenkins_lookup3(&page);
            page.extend_from_slice(&page_cks.to_le_bytes());
            buf.extend_from_slice(&page);
        }
        file.append_bytes(&buf)
    }

    /// Set page `global_page`'s bit in a super block's page-init bitmap
    /// (MSB-first).
    fn set_sb_page_bit<F: Store>(
        &self,
        file: &mut F,
        sblk_addr: u64,
        blk_off: usize,
        global_page: u64,
    ) -> Result<(), Error> {
        let os = file.offset_size() as usize;
        let bitmap_start = sblk_addr + (4 + 1 + 1 + os + blk_off) as u64;
        let byte = bitmap_start + global_page / 8;
        let mask = 0x80u8 >> (global_page % 8);
        let mut v = [0u8; 1];
        file.read_at(byte, &mut v)?;
        file.write_at(byte, &[v[0] | mask])
    }

    /// Recompute the index block checksum from the located dataset metadata.
    fn rechecksum_index_block<F: Store>(&self, file: &mut F) -> Result<(), Error> {
        let os = file.offset_size() as usize;
        let ib_prefix = (4 + 1 + 1 + os) as u64;
        let ndblk_addrs = self.geom.direct_dblk_nelmts.len();
        let nsblk_addrs = self.geom.nsblk_addrs;
        let cks_off = self.index_block_addr
            + ib_prefix
            + self.idx_blk_elmts * self.ea_elem_size as u64
            + ((ndblk_addrs + nsblk_addrs) * os) as u64;
        file.rechecksum_range(self.index_block_addr, cks_off)
    }

    fn rechecksum_super_block<F: Store>(
        &self,
        file: &mut F,
        sblk_addr: u64,
        ndblks: u64,
        dblk_nelmts: u64,
        page_nelmts: u64,
        blk_off: usize,
    ) -> Result<(), Error> {
        let os = file.offset_size() as usize;
        let prefix = (4 + 1 + 1 + os + blk_off) as u64;
        let bitmap = sb_bitmap_size(ndblks, dblk_nelmts, page_nelmts)? as u64;
        let cks_off = sblk_addr + prefix + bitmap + ndblks * os as u64;
        file.rechecksum_range(sblk_addr, cks_off)
    }

    /// Patch the six EA header statistics and recompute the header checksum.
    pub(crate) fn update_ea_header<F: Store>(
        &self,
        file: &mut F,
        num_chunks: u64,
    ) -> Result<(), Error> {
        let stats = ea_compute_stats(
            &self.geom,
            self.idx_blk_elmts,
            self.ea_elem_size,
            self.page_nelmts,
            file.offset_size(),
            self.blk_off_size,
            num_chunks,
        );
        let ls = file.length_size() as u64;
        let ea_addr = self.ea_addr;
        file.write_length_at(ea_addr + 12, stats.nsuper_blks)?;
        file.write_length_at(ea_addr + 12 + ls, stats.super_blk_size)?;
        file.write_length_at(ea_addr + 12 + 2 * ls, stats.ndata_blks)?;
        file.write_length_at(ea_addr + 12 + 3 * ls, stats.data_blk_size)?;
        file.write_length_at(ea_addr + 12 + 4 * ls, stats.max_idx_set)?;
        file.write_length_at(ea_addr + 12 + 5 * ls, stats.nelmts)?;
        let aehd_size =
            ExtensibleArrayHeader::serialized_size(file.offset_size(), file.length_size()) as u64;
        let cks_off = ea_addr + aehd_size - 4;
        file.rechecksum_range(ea_addr, cks_off)
    }

    /// Publish `new_dim` as the dataspace axis-0 dimension (the commit point) and
    /// recompute the containing object-header chunk's checksum.
    pub(crate) fn patch_dimension<F: Store>(
        &self,
        file: &mut F,
        new_dim: u64,
    ) -> Result<(), Error> {
        file.write_length_at(self.dim0_off, new_dim)?;
        file.rechecksum_range(self.ohdr_chunk_start, self.ohdr_chunk_msg_end)
    }
}

/// The mutation-free result of planning one in-place Extensible-Array append: the
/// per-chunk (possibly compressed) blobs to write at end-of-file plus the
/// bookkeeping the write phase publishes.
pub(crate) struct AppendPlan {
    /// Per-chunk (possibly compressed) bytes to write, in element order starting
    /// at `n_full`. The first entry is the rewritten trailing chunk when the
    /// current length was not chunk-aligned.
    pub new_chunk_bytes: Vec<Vec<u8>>,
    /// Element index the first new chunk occupies.
    pub n_full: u64,
    /// New axis-0 dimension after the append (the commit value).
    pub new_dim: u64,
    /// New number of indexed chunks.
    pub new_num_chunks: u64,
}

/// Compute, without mutating the file, the new chunk blobs to write for an append
/// of `new_elems` elements (`raw` little-endian bytes) to the Extensible-Array
/// dataset described by `loc`, together with the first element index they occupy
/// and the new dimension / chunk count. Shared by the general append writer and
/// `EditSession`'s in-place append so the read/plan logic lives in one place.
///
/// A *filtered* dataset can only be appended in whole chunks: a non-chunk-aligned
/// filtered append is refused here rather than repointing a multi-field trailing
/// element whose in-place overwrite is not power-loss atomic; use
/// [`EditSession::append_dataset`](crate::EditSession::append_dataset) for that.
pub(crate) fn plan_ea_append<F: Store>(
    file: &F,
    loc: &Located,
    datatype: &Datatype,
    spatial: &[u64],
    element_size: usize,
    pipeline: Option<&FilterPipeline>,
    raw: &[u8],
    new_elems: u64,
) -> Result<AppendPlan, Error> {
    let chunk_elems = loc.chunk_elems;
    let current_dim = loc.current_dim;
    let new_dim = current_dim
        .checked_add(new_elems)
        .ok_or(Error::AppendUnsupported(
            "append would overflow the dataset dimension",
        ))?;
    let n_full = current_dim / chunk_elems;
    let has_partial = current_dim % chunk_elems != 0;

    // Filtered appends must be chunk-aligned. Growing a *filtered* partial trailing
    // chunk would repoint that chunk's existing index element in place, and a
    // filtered element is a multi-field record (address + compressed_size +
    // filter_mask) that is visible at the old dimension before the commit — so a
    // power-loss crash tearing that record across a disk sector could leave the
    // committed view unreadable. The trailing element of an *unfiltered* dataset is
    // a single address whose overwrite is atomic, so any-length unfiltered appends
    // are allowed.
    if pipeline.is_some() && (has_partial || new_elems % chunk_elems != 0) {
        return Err(Error::AppendUnsupported(
            "a filtered dataset can only be appended in place in whole chunks (the current \
             length and the appended length must both be multiples of the chunk length); \
             use Dataset::append_staged for a non-chunk-aligned filtered append",
        ));
    }

    // Build the raw tail region: the live prefix of any rewritten partial chunk,
    // then the appended bytes.
    let mut tail_raw: Vec<u8> = Vec::new();
    if has_partial {
        let rec = loc
            .read_element(file, n_full)?
            .ok_or(Error::AppendUnsupported(
                "trailing partial chunk is missing from the index",
            ))?;
        let stored_len = if pipeline.is_some() {
            usize::try_from(rec.stored_size)
                .map_err(|_| Error::AppendUnsupported("chunk size exceeds this platform"))?
        } else {
            chunk_elems.to_usize()? * element_size
        };
        rec.addr
            .checked_add(stored_len as u64)
            .filter(|&e| e <= file.len())
            .ok_or(Error::AppendUnsupported(
                "trailing chunk extends past end-of-file",
            ))?;
        // One bounded read of the single trailing chunk.
        let stored = file.read_exact_at(rec.addr, stored_len)?;
        let full = if let Some(pl) = pipeline {
            let ctx = ChunkContext::from_datatype(spatial, datatype);
            decompress_chunk(&stored, pl, ctx, rec.filter_mask).map_err(Error::Format)?
        } else {
            stored
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
    let split = split_into_chunks(&tail_raw, &[tail_len_elems], spatial, element_size);
    let new_chunk_bytes: Vec<Vec<u8>> = if let Some(pl) = pipeline {
        let ctx = ChunkContext::from_datatype(spatial, datatype);
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

/// Apply a planned append to `file` in place, ordered child-before-parent with
/// `fsync` barriers so a crash between calls leaves either the old length or the
/// new one, never a torn or lost view. `max_phase` runs only the first N of the
/// four durability phases (production callers pass 4; the crash-consistency tests
/// stop at a boundary to simulate a crash). On a full (phase-4) apply, `loc`'s
/// cached `current_dim` / `num_chunks` are advanced to match. Shared by the
/// general append writer and `EditSession`'s in-place append so this ordered
/// write sequence — the crash-safety heart of the engine — lives in exactly one
/// place and is never copy-pasted.
pub(crate) fn apply_ea_append<F: Store>(
    file: &mut F,
    loc: &mut Located,
    plan: &AppendPlan,
    max_phase: u8,
) -> Result<(), Error> {
    // Phase 1: new/relocated chunk bytes at EOF, then advance the superblock's
    // recorded end-of-file to cover them. This must precede the index writes: the
    // trailing partial chunk's element is *visible* at the old dimension, so once
    // it is repointed to the relocated chunk that chunk must already lie within the
    // recorded EOF.
    let mut chunk_addrs = Vec::with_capacity(plan.new_chunk_bytes.len());
    for blob in &plan.new_chunk_bytes {
        chunk_addrs.push((file.append_bytes(blob)?, blob.len() as u64));
    }
    file.sync()?;
    file.patch_superblock_eof()?;
    file.sync()?;
    if max_phase < 2 {
        return Ok(());
    }

    // Phase 2: the index element writes — a fresh insert for each new chunk, or an
    // in-place repoint of the trailing element (which only ever points at data
    // whose live prefix reproduces the old view's bytes). This may allocate new EA
    // blocks past EOF, covered by the phase-3 EOF patch.
    for (k, &(addr, stored_size)) in chunk_addrs.iter().enumerate() {
        let e = plan.n_full + k as u64;
        let rec = ElemRecord {
            addr,
            stored_size,
            filter_mask: 0,
        };
        loc.ea_insert(file, e, rec)?;
    }
    file.sync()?;
    if max_phase < 3 {
        return Ok(());
    }

    // Phase 3: cover any EA blocks allocated during the element writes, then
    // publish the EA header element count.
    file.patch_superblock_eof()?;
    loc.update_ea_header(file, plan.new_num_chunks)?;
    file.sync()?;
    if max_phase < 4 {
        return Ok(());
    }

    // Phase 4: publish the dataspace dimension — the single commit point.
    loc.patch_dimension(file, plan.new_dim)?;
    file.sync()?;

    loc.current_dim = plan.new_dim;
    loc.num_chunks = plan.new_num_chunks;
    Ok(())
}

/// Byte size of a super block's page-init bitmap (0 when its data blocks are not
/// paged): `ndblks * ceil(npages / 8)`.
fn sb_bitmap_size(ndblks: u64, dblk_nelmts: u64, page_nelmts: u64) -> Result<usize, Error> {
    if dblk_nelmts > page_nelmts {
        let npages = (dblk_nelmts / page_nelmts).to_usize()?;
        Ok(ndblks.to_usize()? * npages.div_ceil(8))
    } else {
        Ok(0)
    }
}

/// File offset of the `dblk_local`-th data-block-address slot inside a super
/// block, accounting for the page-init bitmap when the block is paged.
#[allow(clippy::too_many_arguments)]
fn sb_dblk_slot_off(
    os: usize,
    sblk_addr: u64,
    dblk_local: usize,
    ndblks: u64,
    dblk_nelmts: u64,
    page_nelmts: u64,
    blk_off: usize,
) -> Result<u64, Error> {
    let prefix = (4 + 1 + 1 + os + blk_off) as u64;
    let bitmap = sb_bitmap_size(ndblks, dblk_nelmts, page_nelmts)? as u64;
    Ok(sblk_addr + prefix + bitmap + (dblk_local * os) as u64)
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
    for j in 0..geom.nsblk_addrs {
        let (ndblks, dn) = geom.sblks[geom.first_indirect_sblk + j];
        let span = ndblks * dn;
        if e < elem + span {
            let sb_block_offset = elem - idx_blk_elmts;
            let within = e - elem;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "within/dn is a data-block index bounded by ndblks (small)"
            )]
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
    /// Absolute file offset of the message body.
    data_off: u64,
    size: usize,
    /// Containing chunk's checksum coverage: `[chunk_start, chunk_msg_end)`, with
    /// the 4-byte checksum stored at `chunk_msg_end`.
    chunk_start: u64,
    chunk_msg_end: u64,
}

struct Walk {
    messages: Vec<WalkedMessage>,
}

/// Walk a version-2 object header (chunk 0 plus any continuation chunks),
/// recording each message's absolute data offset and its containing chunk's
/// checksum region. Reads one bounded window per header chunk from `source`, so
/// the walk works over a store with no whole-file mirror.
fn walk_v2_object_header<S: Source + ?Sized>(
    source: &S,
    offset: u64,
    offset_size: u8,
    length_size: u8,
) -> Result<Walk, Error> {
    let head = match source.read_metadata_at(offset, 6) {
        Ok(head) => head,
        // A header running past end-of-file reads as a missing signature, like
        // the whole-buffer walk before it; other backend errors pass through.
        Err(FormatError::UnexpectedEof { .. }) => {
            return Err(Error::Format(FormatError::InvalidObjectHeaderSignature));
        }
        Err(e) => return Err(Error::Format(e)),
    };
    if &head[..4] != b"OHDR" {
        return Err(Error::Format(FormatError::InvalidObjectHeaderSignature));
    }
    let flags = head[5];
    let mut pos = offset + 6;
    if flags & 0x20 != 0 {
        pos += 16; // timestamps
    }
    if flags & 0x10 != 0 {
        pos += 4; // attr storage phase-change
    }
    let chunk_size_width = 1usize << (flags & 0x03);
    let size_buf = source.read_metadata_at(pos, chunk_size_width)?;
    let chunk0_size = read_uint(&size_buf, 0, chunk_size_width)?.to_usize()?;
    pos += chunk_size_width as u64;
    let chunk0_start = offset;
    let chunk0_msg_start = pos;
    let chunk0_msg_end = chunk0_msg_start + chunk0_size as u64;

    let has_creation_order = flags & 0x04 != 0;
    let mut messages = Vec::new();
    let mut continuations: Vec<(u64, usize)> = Vec::new();

    let chunk0 = source.read_metadata_at(chunk0_msg_start, chunk0_size)?;
    walk_messages(
        &chunk0,
        chunk0_msg_start,
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
        // A continuation chunk is `OCHK` + messages + a trailing 4-byte
        // checksum, so anything shorter than 8 bytes is malformed.
        if cont_len < 8 {
            return Err(Error::Format(FormatError::InvalidObjectHeaderSignature));
        }
        let chunk = source.read_metadata_at(cont_off, cont_len)?;
        if &chunk[..4] != b"OCHK" {
            return Err(Error::Format(FormatError::InvalidObjectHeaderSignature));
        }
        let msg_start = cont_off + 4;
        let msg_end = cont_off + (cont_len - 4) as u64; // checksum is the last 4 bytes
        walk_messages(
            &chunk[4..cont_len - 4],
            msg_start,
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

/// Scan one object-header chunk's message region. `chunk` holds exactly the
/// message bytes and `base` is the absolute file offset of `chunk[0]`, so
/// recorded message offsets are absolute.
#[allow(clippy::too_many_arguments)]
fn walk_messages(
    chunk: &[u8],
    base: u64,
    chunk_start: u64,
    chunk_msg_end: u64,
    has_creation_order: bool,
    offset_size: u8,
    length_size: u8,
    messages: &mut Vec<WalkedMessage>,
    continuations: &mut Vec<(u64, usize)>,
) -> Result<(), Error> {
    let msg_header_size = if has_creation_order { 6 } else { 4 };
    let end = chunk.len();
    let mut pos = 0usize;
    while pos + msg_header_size <= end {
        let msg_type_raw = chunk[pos] as u16;
        let msg_data_size = u16::from_le_bytes([chunk[pos + 1], chunk[pos + 2]]) as usize;
        pos += msg_header_size;
        if pos + msg_data_size > end {
            break; // padding
        }
        let msg_type = MessageType::from_u16(msg_type_raw);
        if msg_type == MessageType::ObjectHeaderContinuation {
            let cont_off = read_uint(chunk, pos, offset_size as usize)?;
            let cont_len =
                read_uint(chunk, pos + offset_size as usize, length_size as usize)?.to_usize()?;
            continuations.push((cont_off, cont_len));
        } else {
            messages.push(WalkedMessage {
                msg_type,
                data_off: base + pos as u64,
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
    use crate::group_v2;
    use crate::writer::FileBuilder;
    use std::cell::{Cell, RefCell};

    /// An in-test [`Store`] over a `Vec<u8>` that records every read window, so
    /// the tests below can assert the engine's bounded-memory contract: the
    /// append path never reads more than a bounded window at once, no matter how
    /// large the file is. This is the seam contract issue #147's mirror-less
    /// bounded backend relies on.
    struct WindowProbeStore {
        data: Vec<u8>,
        superblock: Superblock,
        sb_sig_off: usize,
        max_read: Cell<usize>,
        total_read: Cell<usize>,
        reads: RefCell<Vec<(u64, usize)>>,
    }

    impl WindowProbeStore {
        fn open(data: Vec<u8>) -> Self {
            let sb_sig_off = signature::find_signature(&data).unwrap();
            let superblock = Superblock::parse(&data, sb_sig_off).unwrap();
            Self {
                data,
                superblock,
                sb_sig_off,
                max_read: Cell::new(0),
                total_read: Cell::new(0),
                reads: RefCell::new(Vec::new()),
            }
        }

        fn reset_counters(&self) {
            self.max_read.set(0);
            self.total_read.set(0);
            self.reads.borrow_mut().clear();
        }
    }

    impl Source for WindowProbeStore {
        fn len(&self) -> u64 {
            self.data.len() as u64
        }
        fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
            self.max_read.set(self.max_read.get().max(buf.len()));
            self.total_read.set(self.total_read.get() + buf.len());
            self.reads.borrow_mut().push((offset, buf.len()));
            BytesSource::new(&self.data).read_at(offset, buf)
        }
    }

    impl Store for WindowProbeStore {
        fn offset_size(&self) -> u8 {
            self.superblock.offset_size
        }
        fn length_size(&self) -> u8 {
            self.superblock.length_size
        }
        fn append_bytes(&mut self, bytes: &[u8]) -> Result<u64, Error> {
            let addr = self.data.len() as u64;
            self.data.extend_from_slice(bytes);
            Ok(addr)
        }
        fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
            let offset = offset.to_usize()?;
            self.data[offset..offset + bytes.len()].copy_from_slice(bytes);
            Ok(())
        }
        fn patch_superblock_eof(&mut self) -> Result<(), Error> {
            self.superblock.eof_address = self.data.len() as u64;
            let bytes = self.superblock.serialize();
            let off = self.sb_sig_off;
            self.data[off..off + bytes.len()].copy_from_slice(&bytes);
            Ok(())
        }
        fn sync(&mut self) -> Result<(), Error> {
            Ok(())
        }
    }

    /// Build an in-memory latest-format file with one unlimited chunked i32
    /// dataset seeded with `0..n`, returning its bytes.
    fn build_unlimited(n: i32, chunk: u64) -> Vec<u8> {
        let data: Vec<i32> = (0..n).collect();
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&data)
            .with_shape(&[n as u64])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[chunk]);
        b.finish().unwrap()
    }

    /// Drive the full located-append path (`locate_at` + `plan_ea_append` +
    /// `apply_ea_append`) through the probe store and return the located state.
    fn locate(store: &WindowProbeStore) -> (Located, crate::datatype::Datatype) {
        let oh_addr = group_v2::resolve_path_any(&store.data, &store.superblock, "d").unwrap();
        let result = Located::locate_at(store, oh_addr, Error::AppendUnsupported).unwrap();
        let (dt_off, dt_size) = result.spans.datatype;
        let dt_bytes = store.read_metadata_at(dt_off, dt_size).unwrap();
        let (datatype, _) = crate::datatype::Datatype::parse(&dt_bytes).unwrap();
        (result.located, datatype)
    }

    fn append_i32s(
        store: &mut WindowProbeStore,
        loc: &mut Located,
        datatype: &crate::datatype::Datatype,
        values: std::ops::Range<i32>,
    ) {
        let raw: Vec<u8> = values.clone().flat_map(|v| v.to_le_bytes()).collect();
        let new_elems = (values.end - values.start) as u64;
        let spatial = vec![loc.chunk_elems];
        let plan = plan_ea_append(
            store,
            loc,
            datatype,
            &spatial,
            loc.elem_bytes,
            None,
            &raw,
            new_elems,
        )
        .unwrap();
        apply_ea_append(store, loc, &plan, 4).unwrap();
    }

    /// The engine's bounded-read contract: appends against a file far larger
    /// than any metadata structure never read more than a bounded window at
    /// once, and never the whole file.
    #[test]
    fn append_reads_stay_bounded_windows() {
        // ~400 KiB of data: far larger than any single bounded window below.
        let n = 100_000i32;
        let mut store = WindowProbeStore::open(build_unlimited(n, 256));
        let file_len = store.data.len();
        assert!(
            file_len > 300_000,
            "test file unexpectedly small: {file_len}"
        );

        let (mut loc, datatype) = locate(&store);
        // Setup (locate + datatype) already obeys the window bound.
        const WINDOW: usize = 16 * 1024;
        assert!(
            store.max_read.get() <= WINDOW,
            "locate read a {}-byte window (> {WINDOW})",
            store.max_read.get()
        );

        // A small append against the large file: every read (trailing chunk,
        // element slots, checksum regions) stays within the window bound, and
        // the total read volume is a small constant, not O(file size).
        store.reset_counters();
        append_i32s(&mut store, &mut loc, &datatype, n..n + 10);
        assert!(
            store.max_read.get() <= WINDOW,
            "append read a {}-byte window (> {WINDOW})",
            store.max_read.get()
        );
        assert!(
            store.total_read.get() <= 64 * 1024,
            "append read {} bytes total (> 64 KiB) on a {file_len}-byte file",
            store.total_read.get()
        );
        assert!(
            store
                .reads
                .borrow()
                .iter()
                .all(|&(_, len)| len < file_len / 2),
            "an append read scaled with file size"
        );

        // Growth loop: the per-append read volume stays flat as the file grows.
        let mut worst = 0usize;
        let mut next = n + 10;
        for _ in 0..20 {
            store.reset_counters();
            append_i32s(&mut store, &mut loc, &datatype, next..next + 300);
            worst = worst.max(store.total_read.get());
            next += 300;
        }
        assert!(
            worst <= 64 * 1024,
            "per-append read volume grew to {worst} bytes"
        );

        // The grown file reads back correctly through the public reader.
        let file = crate::File::from_bytes(store.data).unwrap();
        let ds = file.dataset("d").unwrap();
        let got = ds.read_i32().unwrap();
        let expected: Vec<i32> = (0..next).collect();
        assert_eq!(got.len(), expected.len());
        assert_eq!(got, expected);
    }

    /// Unaligned appends rewrite the trailing partial chunk: still bounded — the
    /// one chunk is the largest data read the plan performs.
    #[test]
    fn partial_tail_append_reads_one_chunk_window() {
        let n = 50_000i32;
        let chunk = 256u64;
        let mut store = WindowProbeStore::open(build_unlimited(n + 7, chunk));
        let (mut loc, datatype) = locate(&store);

        store.reset_counters();
        append_i32s(&mut store, &mut loc, &datatype, n + 7..n + 7 + 13);
        let chunk_bytes = (chunk as usize) * 4;
        assert!(
            store.max_read.get() <= chunk_bytes.max(8 * 1024),
            "partial-tail append read a {}-byte window",
            store.max_read.get()
        );

        let file = crate::File::from_bytes(store.data).unwrap();
        let got = file.dataset("d").unwrap().read_i32().unwrap();
        let expected: Vec<i32> = (0..n + 7 + 13).collect();
        assert_eq!(got, expected);
    }
}
