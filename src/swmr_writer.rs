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
//! - Up to the paged-data-block boundary (131056 chunks on the unlimited axis).
//!   Larger appends return [`Error::SwmrUnsupported`] rather than risk an
//!   incorrect paged write.
//! - Files with a zero base address (no userblock).

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};

use crate::chunked_write::ea_compute_stats;
use crate::checksum::jenkins_lookup3;
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
}

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
        if superblock.base_address != 0 {
            // Userblock files store addresses relative to the base; unsupported
            // by the append writer for now.
            return Err(Error::SwmrUnsupported);
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
            located: HashMap::new(),
        })
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
    pub fn append_raw(&mut self, dataset: &str, bytes: &[u8]) -> Result<(), Error> {
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
            return Err(Error::SwmrUnsupported);
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

        // Phase 1: write chunk data + grow/patch the chunk index, all durable,
        // before publishing the new counts.
        for c in 0..n_new_chunks {
            let chunk_data = &bytes[c * chunk_bytes..(c + 1) * chunk_bytes];
            let chunk_addr = self.append_bytes(chunk_data);
            let e = num_chunks + c as u64;
            self.ea_insert(dataset, e, chunk_addr)?;
        }
        self.sync()?;

        // Phase 2: publish the new element count in the EA header (this makes the
        // appended chunks visible to a reader traversing the index).
        let new_num_chunks = num_chunks + n_new_chunks as u64;
        self.update_ea_header(dataset, new_num_chunks)?;
        self.sync()?;

        // Phase 3: publish the new dataspace dimension (the dataset's new shape).
        let new_dim = current_dim + new_elems;
        self.patch_dimension(dataset, new_dim)?;
        self.sync()?;

        // Phase 4: advance the superblock end-of-file.
        self.patch_superblock_eof()?;
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
            return Err(Error::SwmrUnsupported);
        }

        // Parse the dataspace: must be rank 1 with one unlimited dimension.
        let ds_bytes = &self.data[dataspace_msg.data_off..dataspace_msg.data_off + dataspace_msg.size];
        let dataspace = Dataspace::parse(ds_bytes, ls)?;
        if dataspace.rank != 1 {
            return Err(Error::SwmrUnsupported);
        }
        match &dataspace.max_dimensions {
            Some(maxs) if maxs.first() == Some(&u64::MAX) => {}
            _ => return Err(Error::SwmrUnsupported),
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
            _ => return Err(Error::SwmrUnsupported),
        };
        // chunk_dimensions for a v4 layout includes the element-size pseudo
        // dimension as its last entry; the leading entry is the axis-0 chunk size.
        if chunk_dims.len() != 2 {
            return Err(Error::SwmrUnsupported);
        }
        let chunk_elems = chunk_dims[0] as u64;
        let elem_bytes = chunk_dims[1] as usize;
        let chunk_bytes = chunk_elems as usize * elem_bytes;

        // Parse the EA header for its creation parameters and current count.
        let ea_header = ExtensibleArrayHeader::parse(&self.data, ea_addr, os, ls)?;
        if ea_header.client_id != 0 {
            return Err(Error::SwmrUnsupported); // filtered
        }
        let geom = EaGeometry::from_header(&ea_header);
        let page_nelmts = 1u64 << ea_header.max_dblk_nelmts_bits;
        let blk_off_size = (ea_header.max_nelmts_bits as usize).div_ceil(8);
        let index_block_addr = ea_header.index_block_address as usize;
        let num_chunks = ea_header.num_elements;

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
    fn ea_insert(&mut self, dataset: &str, e: u64, chunk_addr: u64) -> Result<(), Error> {
        let loc = &self.located[dataset];
        let os = self.offset_size;
        let elem_size = loc.ea_elem_size;
        let idx = loc.idx_blk_elmts;
        let blk_off = loc.blk_off_size;
        let page_nelmts = loc.page_nelmts;
        let index_block_addr = loc.index_block_addr;
        let ea_addr = loc.ea_addr;

        // Inline element slots live directly in the index block.
        if e < idx {
            let ib_prefix = 4 + 1 + 1 + os as usize; // sig + ver + client + hdr_addr
            let slot_off = index_block_addr + ib_prefix + e as usize * elem_size;
            self.write_addr_at(slot_off, chunk_addr);
            self.rechecksum_index_block(dataset);
            return Ok(());
        }

        // Otherwise the element lives in a data block. Find which one and how it
        // is reached (a direct pointer in the index block, or via a super block).
        let region = locate_data_block(&loc.geom, idx, e);
        let dblk_nelmts = region.dblk_nelmts;
        if dblk_nelmts > page_nelmts {
            // Paged data blocks (very large datasets) not yet supported on the
            // write path.
            return Err(Error::SwmrUnsupported);
        }
        let slot = (e - region.db_start) as usize;
        let block_offset_rel = region.db_start - idx;

        // Resolve the data block's address, allocating it on the first element.
        let dblk_addr = if slot == 0 {
            // Allocate a fresh, fully-undefined data block at EOF.
            let new_addr =
                self.alloc_undef_data_block(ea_addr, dblk_nelmts, block_offset_rel, blk_off);
            // Link it from its parent (index block direct slot, or a super block,
            // allocating the super block first if needed).
            match region.parent {
                Parent::IndexDirect { ordinal } => {
                    let ib_prefix = 4 + 1 + 1 + os as usize;
                    let slot_off = index_block_addr
                        + ib_prefix
                        + idx as usize * elem_size
                        + ordinal * os as usize;
                    self.write_addr_at(slot_off, new_addr);
                    self.rechecksum_index_block(dataset);
                }
                Parent::Super { sblk_j, dblk_local } => {
                    let sblk_addr = self.ensure_super_block(dataset, sblk_j, region.sb_block_offset)?;
                    let sb_prefix = 4 + 1 + 1 + os as usize + blk_off;
                    let slot_off = sblk_addr as usize + sb_prefix + dblk_local * os as usize;
                    self.write_addr_at(slot_off, new_addr);
                    self.rechecksum_super_block(sblk_addr, region.ndblks, blk_off);
                }
            }
            new_addr
        } else {
            // The data block already exists; read its address from the parent.
            match region.parent {
                Parent::IndexDirect { ordinal } => {
                    let ib_prefix = 4 + 1 + 1 + os as usize;
                    let slot_off = index_block_addr
                        + ib_prefix
                        + idx as usize * elem_size
                        + ordinal * os as usize;
                    self.read_addr_at(slot_off)
                }
                Parent::Super { sblk_j, dblk_local } => {
                    let sblk_addr = self.super_block_addr(dataset, sblk_j);
                    let sb_prefix = 4 + 1 + 1 + os as usize + blk_off;
                    self.read_addr_at(sblk_addr as usize + sb_prefix + dblk_local * os as usize)
                }
            }
        };

        // Write the element into the data block and re-checksum the block.
        let db_prefix = 4 + 1 + 1 + os as usize + blk_off;
        let elem_off = dblk_addr as usize + db_prefix + slot * elem_size;
        self.write_addr_at(elem_off, chunk_addr);
        let cks_off = dblk_addr as usize + db_prefix + dblk_nelmts as usize * elem_size;
        self.rechecksum_range(dblk_addr as usize, cks_off);
        Ok(())
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

    /// Return the address of super block `sblk_j`, allocating an empty one
    /// (all data-block pointers undefined) at EOF if it does not exist yet.
    fn ensure_super_block(
        &mut self,
        dataset: &str,
        sblk_j: usize,
        sb_block_offset: u64,
    ) -> Result<u64, Error> {
        let existing = self.super_block_addr(dataset, sblk_j);
        if !is_undef(existing, self.offset_size) {
            return Ok(existing);
        }
        let (ndblks, dblk_nelmts, blk_off, os, ib_prefix, ndblk_addrs, idx_elems, elem_size, ib_addr) = {
            let loc = &self.located[dataset];
            let (ndblks, dn) = loc.geom.sblks[loc.geom.first_indirect_sblk + sblk_j];
            (
                ndblks,
                dn,
                loc.blk_off_size,
                self.offset_size as usize,
                4 + 1 + 1 + self.offset_size as usize,
                loc.geom.direct_dblk_nelmts.len(),
                loc.idx_blk_elmts as usize,
                loc.ea_elem_size,
                loc.index_block_addr,
            )
        };
        if dblk_nelmts > self.located[dataset].page_nelmts {
            return Err(Error::SwmrUnsupported); // paged super block
        }
        let ea_addr = self.located[dataset].ea_addr as u64;

        // Build an empty super block (all data-block pointers undefined).
        let undef = vec![undef_addr(self.offset_size); ndblks as usize];
        let aesb = crate::chunked_write::build_aesb(
            ea_addr,
            sb_block_offset,
            &[],
            &undef,
            self.offset_size,
            blk_off,
            0,
        );
        let new_addr = self.append_bytes(&aesb);

        // Link the super block from the index block and re-checksum it.
        let slot_off =
            ib_addr + ib_prefix + idx_elems * elem_size + ndblk_addrs * os + sblk_j * os;
        self.write_addr_at(slot_off, new_addr);
        self.rechecksum_index_block(dataset);
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
    ) -> u64 {
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

    /// Recompute the index block checksum from the located dataset metadata.
    fn rechecksum_index_block(&mut self, dataset: &str) {
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
        self.rechecksum_range(loc.index_block_addr, cks_off);
    }

    fn rechecksum_super_block(&mut self, sblk_addr: u64, ndblks: u64, blk_off: usize) {
        let os = self.offset_size as usize;
        let sb_prefix = 4 + 1 + 1 + os + blk_off;
        let cks_off = sblk_addr as usize + sb_prefix + ndblks as usize * os;
        self.rechecksum_range(sblk_addr as usize, cks_off);
    }

    /// Recompute the Jenkins checksum over `[start, cks_off)` and write it at
    /// `cks_off`.
    fn rechecksum_range(&mut self, start: usize, cks_off: usize) {
        let cks = jenkins_lookup3(&self.data[start..cks_off]);
        self.write_at(cks_off, &cks.to_le_bytes());
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
        let write_stat = |this: &mut Self, k: usize, v: u64| {
            let off = ea_addr + 12 + k * ls as usize;
            this.write_length_at(off, v);
        };
        write_stat(self, 0, stats.nsuper_blks);
        write_stat(self, 1, stats.super_blk_size);
        write_stat(self, 2, stats.ndata_blks);
        write_stat(self, 3, stats.data_blk_size);
        write_stat(self, 4, stats.max_idx_set);
        write_stat(self, 5, stats.nelmts);
        // EAHD checksum covers signature .. just before the trailing checksum.
        let aehd_size = ExtensibleArrayHeader::serialized_size(self.offset_size, self.length_size);
        let cks_off = ea_addr + aehd_size - 4;
        self.rechecksum_range(ea_addr, cks_off);
        Ok(())
    }

    fn patch_dimension(&mut self, dataset: &str, new_dim: u64) -> Result<(), Error> {
        let (dim0_off, chunk_start, chunk_msg_end) = {
            let loc = &self.located[dataset];
            (loc.dim0_off, loc.ohdr_chunk_start, loc.ohdr_chunk_msg_end)
        };
        self.write_length_at(dim0_off, new_dim);
        // Recompute the containing object-header chunk's checksum.
        self.rechecksum_range(chunk_start, chunk_msg_end);
        Ok(())
    }

    fn patch_superblock_eof(&mut self) -> Result<(), Error> {
        let eof = self.data.len() as u64;
        self.superblock.eof_address = eof;
        let bytes = self.superblock.serialize();
        self.write_at(self.sb_sig_off, &bytes);
        Ok(())
    }

    // ----- raw byte / address IO ----------------------------------------

    /// Append `bytes` at end-of-file (in memory and on disk), returning the
    /// address (offset) at which they were written.
    fn append_bytes(&mut self, bytes: &[u8]) -> u64 {
        let addr = self.data.len() as u64;
        self.data.extend_from_slice(bytes);
        // Defer the disk write to the next `sync`/`write_at`? No — write now so a
        // later in-place patch of this region also reaches disk.
        let _ = self.handle.seek(SeekFrom::Start(addr));
        let _ = self.handle.write_all(bytes);
        addr
    }

    fn write_at(&mut self, offset: usize, bytes: &[u8]) {
        self.data[offset..offset + bytes.len()].copy_from_slice(bytes);
        let _ = self.handle.seek(SeekFrom::Start(offset as u64));
        let _ = self.handle.write_all(bytes);
    }

    fn write_addr_at(&mut self, offset: usize, addr: u64) {
        match self.offset_size {
            4 => self.write_at(offset, &(addr as u32).to_le_bytes()),
            _ => self.write_at(offset, &addr.to_le_bytes()),
        }
    }

    fn write_length_at(&mut self, offset: usize, val: u64) {
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
                parent: Parent::Super { sblk_j: j, dblk_local },
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
        let msg_data_size =
            u16::from_le_bytes([data[pos + 1], data[pos + 2]]) as usize;
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
