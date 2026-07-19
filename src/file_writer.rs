//! HDF5 file creation (write pipeline).
//!
//! Produces valid HDF5 files with v3 superblock, v2 object headers,
//! link messages, contiguous datasets, inline and dense attributes.

#[cfg(not(feature = "std"))]
use alloc::{string::String, string::ToString, vec, vec::Vec};

#[cfg(not(feature = "std"))]
use alloc::format;

#[cfg(not(feature = "std"))]
use alloc::collections::BTreeMap as HashMap;
#[cfg(feature = "std")]
use std::collections::HashMap;

use crate::attribute::AttributeMessage;
use crate::chunked_write::{
    ByteSink, ChunkOptions, CompressedChunkSet, VerbatimLayout, VerbatimPlan, assemble_chunked_at,
    compress_chunks, emit_chunked_data_verbatim, plan_chunked_data_verbatim,
};
use crate::convert::TryToUsize;
use crate::dataspace::{Dataspace, DataspaceType};
use crate::error::FormatError;
use crate::file_space_info::{
    DEFAULT_PAGE_SIZE, DEFAULT_THRESHOLD, FileSpaceInfo, FileSpaceStrategy,
};
use crate::libver::LibVer;
use crate::link_message::{LinkMessage, LinkTarget};
use crate::message_type::MessageType;
use crate::object_header_writer::ObjectHeaderWriter;
use crate::superblock::Superblock;
use crate::type_builders::{
    DatasetBuilder, FinishedGroup, GroupBuilder, VlStringStaging, build_attr_message,
    build_global_heap_collection, patch_vl_refs, patch_vl_refs_masked,
};

// `AttrValue` lives in `type_builders`; `types` and `mat` reference it through
// this module's path, so keep it re-exported here.
pub use crate::type_builders::AttrValue;

use crate::datatype::{CharacterSet, Datatype};

pub(crate) const OFFSET_SIZE: u8 = 8;
pub(crate) const LENGTH_SIZE: u8 = 8;
const SUPERBLOCK_SIZE: usize = 48;

/// Threshold for switching from compact (inline) to dense attribute storage.
const DENSE_ATTR_THRESHOLD: usize = 8;

// ---- OH builders ----

pub(crate) fn build_chunked_dataset_oh(
    dt: &Datatype,
    ds: &Dataspace,
    layout_message: &[u8],
    pipeline_message: Option<&[u8]>,
    attrs: &[AttributeMessage],
    dense_blob: Option<&DenseAttrBlob>,
    fill: Option<&[u8]>,
) -> Vec<u8> {
    let mut w = ObjectHeaderWriter::new();
    w.add_message_with_flags(MessageType::Datatype, dt.serialize(), 0x01);
    w.add_message(MessageType::Dataspace, ds.serialize(LENGTH_SIZE));
    w.add_message_with_flags(
        MessageType::FillValue,
        crate::fill_value::fill_value_message_v3(fill),
        0x01,
    );
    w.add_message(MessageType::DataLayout, layout_message.to_vec());
    if let Some(pm) = pipeline_message {
        w.add_message(MessageType::FilterPipeline, pm.to_vec());
    }
    if let Some(blob) = dense_blob {
        w.add_message(MessageType::AttributeInfo, blob.attr_info_message.clone());
    } else {
        for attr in attrs {
            w.add_message(MessageType::Attribute, attr.serialize(LENGTH_SIZE));
        }
    }
    w.serialize()
}

pub(crate) fn build_dataset_oh(
    dt: &Datatype,
    ds: &Dataspace,
    data_addr: u64,
    data_size: u64,
    attrs: &[AttributeMessage],
    dense_blob: Option<&DenseAttrBlob>,
    fill: Option<&[u8]>,
) -> Vec<u8> {
    let mut w = ObjectHeaderWriter::new();
    w.add_message_with_flags(MessageType::Datatype, dt.serialize(), 0x01);
    w.add_message(MessageType::Dataspace, ds.serialize(LENGTH_SIZE));
    w.add_message_with_flags(
        MessageType::FillValue,
        crate::fill_value::fill_value_message_v3(fill),
        0x01,
    );
    let mut dl = Vec::new();
    dl.push(4); // version
    dl.push(1); // class = contiguous
    dl.extend_from_slice(&data_addr.to_le_bytes());
    dl.extend_from_slice(&data_size.to_le_bytes());
    w.add_message(MessageType::DataLayout, dl);
    if let Some(blob) = dense_blob {
        w.add_message(MessageType::AttributeInfo, blob.attr_info_message.clone());
    } else {
        for attr in attrs {
            w.add_message(MessageType::Attribute, attr.serialize(LENGTH_SIZE));
        }
    }
    w.serialize()
}

pub(crate) fn build_group_oh(
    links: &[LinkMessage],
    attrs: &[AttributeMessage],
    dense_blob: Option<&DenseAttrBlob>,
) -> Vec<u8> {
    let mut w = ObjectHeaderWriter::new();
    let mut li = Vec::new();
    li.push(0); // version
    li.push(0); // flags
    li.extend_from_slice(&u64::MAX.to_le_bytes()); // fractal heap addr = UNDEF
    li.extend_from_slice(&u64::MAX.to_le_bytes()); // btree name index addr = UNDEF
    w.add_message(MessageType::LinkInfo, li);
    // A new-style group (one with a Link Info message) must also carry a Group
    // Info message, or the HDF5 C library refuses to insert links into it:
    // `H5G_obj_insert` reads the Group Info message unconditionally and fails
    // with "message type not found", so the file is readable but not writable by
    // the C library. The minimal body (version 0, no optional fields) leaves the
    // C library to use its defaults (max compact = 8, min dense = 6).
    w.add_message(MessageType::GroupInfo, vec![0, 0]);
    for link in links {
        w.add_message(MessageType::Link, link.serialize(OFFSET_SIZE));
    }
    if let Some(blob) = dense_blob {
        w.add_message(MessageType::AttributeInfo, blob.attr_info_message.clone());
    } else {
        for attr in attrs {
            w.add_message(MessageType::Attribute, attr.serialize(LENGTH_SIZE));
        }
    }
    w.serialize()
}

pub(crate) fn make_link(name: &str, addr: u64) -> LinkMessage {
    LinkMessage {
        name: name.to_string(),
        link_target: LinkTarget::Hard {
            object_header_address: addr,
        },
        creation_order: None,
        charset: CharacterSet::Ascii,
    }
}

// ---- Dense attribute blob ----

/// Pre-built dense attribute storage (fractal heap + B-tree v2 + attribute info message).
pub(crate) struct DenseAttrBlob {
    /// Serialized AttributeInfo message data (to embed in the object header).
    pub(crate) attr_info_message: Vec<u8>,
    /// The combined fractal heap header + direct block + B-tree v2 bytes.
    pub(crate) blob: Vec<u8>,
}

/// The largest direct block (and therefore the largest total managed data) the
/// single-block [`build_dense_attrs`] emitter can represent. A heap whose data
/// would need a larger root block would require indirect blocks, which the
/// emitter does not build.
pub(crate) const DENSE_ATTR_MAX_DIRECT_BLOCK: u64 = 65536;

/// Whether [`build_dense_attrs`] can faithfully represent `attrs` in its
/// single-direct-block, single-leaf-B-tree layout. Returns `false` when the
/// serialized attribute set would overflow the one direct block the emitter
/// allocates (it does not build indirect blocks) or exceed the record count a
/// single B-tree v2 leaf can index. Callers that cannot fall back to a larger
/// layout must refuse rather than mis-encode (see [`build_dense_attrs`]).
pub(crate) fn dense_attrs_fit(attrs: &[AttributeMessage]) -> bool {
    let os = OFFSET_SIZE as usize;
    let max_heap_size: u16 = 40;
    let block_offset_bytes = (max_heap_size as usize).div_ceil(8); // 5
    // Direct block layout mirrors `build_dense_attrs`: sig(4) + ver(1) +
    // heap_addr(os) + block_offset(bo_bytes) + checksum(4) + data.
    let dblock_header_size = 4 + 1 + os + block_offset_bytes + 4;
    let total_data_size: usize = attrs
        .iter()
        .map(|a| a.serialize_v3(LENGTH_SIZE).len())
        .sum();
    let dblock_content_size = (dblock_header_size + total_data_size) as u64;
    if dblock_content_size > DENSE_ATTR_MAX_DIRECT_BLOCK {
        return false;
    }
    // The single B-tree v2 leaf writes its record count into a 2-byte field.
    attrs.len() <= u16::MAX as usize
}

/// Build dense attribute storage for a set of attributes.
///
/// The caller must have checked [`dense_attrs_fit`] first: this emitter builds a
/// single direct block and a single-leaf B-tree, so an attribute set larger than
/// that can hold would be mis-encoded.
pub(crate) fn build_dense_attrs(attrs: &[AttributeMessage], base_address: u64) -> DenseAttrBlob {
    // Dense attrs use v3 attribute messages (adds character set encoding byte).
    let serialized: Vec<Vec<u8>> = attrs.iter().map(|a| a.serialize_v3(LENGTH_SIZE)).collect();

    let name_hashes: Vec<u32> = attrs
        .iter()
        .map(|a| crate::checksum::jenkins_lookup3(a.name.as_bytes()))
        .collect();

    let os = OFFSET_SIZE as usize;
    let ls = LENGTH_SIZE as usize;
    let max_heap_size: u16 = 40;
    let block_offset_bytes = (max_heap_size as usize).div_ceil(8); // 5
    let heap_id_length: u16 = 8;
    let max_direct_block_size: u64 = 65536;

    // Direct block layout: sig(4) + ver(1) + heap_addr(os) + block_offset(bo_bytes)
    //   + checksum(4) [when flags bit 1 set] + data...
    let dblock_header_size = 4 + 1 + os + block_offset_bytes + 4; // +4 for checksum
    let total_data_size: usize = serialized.iter().map(|s| s.len()).sum();
    let dblock_content_size = dblock_header_size + total_data_size;
    let starting_block_size = dblock_content_size.next_power_of_two().max(512) as u64;

    // Fractal heap header size
    let frhp_size = 4
        + 1
        + 2
        + 2
        + 1
        + 4
        + ls
        + os
        + ls
        + os
        + ls
        + ls
        + ls
        + ls
        + ls
        + ls
        + ls
        + ls
        + 2
        + ls
        + ls
        + 2
        + 2
        + os
        + 2
        + 4;

    let frhp_addr = base_address;
    let dblock_addr = frhp_addr + frhp_size as u64;
    let btree_addr = dblock_addr + starting_block_size;

    #[expect(
        clippy::cast_possible_truncation,
        reason = "starting_block_size is a fractal-heap direct-block size (KiB-scale heap \
                  geometry), so it fits usize on every supported target"
    )]
    let data_space = starting_block_size as usize - dblock_header_size;
    let free_space = data_space - total_data_size;

    // Build fractal heap header
    let mut frhp = Vec::with_capacity(frhp_size);
    frhp.extend_from_slice(b"FRHP");
    frhp.push(0); // version
    frhp.extend_from_slice(&heap_id_length.to_le_bytes());
    frhp.extend_from_slice(&0u16.to_le_bytes()); // io_filter_encoded_length
    frhp.push(0x02); // flags: bit 1 = checksum direct blocks
    #[expect(
        clippy::cast_possible_truncation,
        reason = "max_direct_block_size and the header size are KiB-scale heap geometry that \
                  fit the 4-byte max-managed-object-size field"
    )]
    let max_managed = max_direct_block_size as u32 - dblock_header_size as u32;
    frhp.extend_from_slice(&max_managed.to_le_bytes());
    write_length(&mut frhp, 0, LENGTH_SIZE); // next_huge_object_id
    write_undef_offset(&mut frhp, OFFSET_SIZE); // btree_huge_objects_address
    write_length(&mut frhp, free_space as u64, LENGTH_SIZE); // free_space_managed_blocks
    write_undef_offset(&mut frhp, OFFSET_SIZE); // free_space_mgr_addr
    write_length(&mut frhp, starting_block_size, LENGTH_SIZE); // managed_space_in_heap
    write_length(&mut frhp, starting_block_size, LENGTH_SIZE); // allocated_managed_space
    write_length(&mut frhp, 0, LENGTH_SIZE); // dblock_alloc_iter
    write_length(&mut frhp, attrs.len() as u64, LENGTH_SIZE); // managed_objects_count
    write_length(&mut frhp, 0, LENGTH_SIZE); // huge_objects_size
    write_length(&mut frhp, 0, LENGTH_SIZE); // huge_objects_count
    write_length(&mut frhp, 0, LENGTH_SIZE); // tiny_objects_size
    write_length(&mut frhp, 0, LENGTH_SIZE); // tiny_objects_count
    frhp.extend_from_slice(&4u16.to_le_bytes()); // table_width
    write_length(&mut frhp, starting_block_size, LENGTH_SIZE);
    write_length(&mut frhp, max_direct_block_size, LENGTH_SIZE); // max_direct_block_size
    frhp.extend_from_slice(&max_heap_size.to_le_bytes());
    let sri: u16 = 1;
    frhp.extend_from_slice(&sri.to_le_bytes()); // start_root_rows
    write_offset(&mut frhp, dblock_addr, OFFSET_SIZE);
    frhp.extend_from_slice(&0u16.to_le_bytes()); // root is direct block
    let frhp_checksum = crate::checksum::jenkins_lookup3(&frhp);
    frhp.extend_from_slice(&frhp_checksum.to_le_bytes());
    debug_assert_eq!(frhp.len(), frhp_size);

    // Build direct block: header (with checksum) + data + padding
    #[expect(
        clippy::cast_possible_truncation,
        reason = "starting_block_size is a KiB-scale heap direct-block size that fits usize"
    )]
    let mut dblock = Vec::with_capacity(starting_block_size as usize);
    dblock.extend_from_slice(b"FHDB");
    dblock.push(0); // version
    write_offset(&mut dblock, frhp_addr, OFFSET_SIZE);
    dblock.extend_from_slice(&vec![0u8; block_offset_bytes]); // block_offset = 0 for root
    let cksum_pos = dblock.len();
    dblock.extend_from_slice(&[0u8; 4]); // checksum placeholder
    debug_assert_eq!(dblock.len(), dblock_header_size);

    // Data area starts after header
    let mut attr_offsets: Vec<(u64, u64)> = Vec::with_capacity(attrs.len());
    for s in &serialized {
        let offset_in_heap = dblock.len() as u64;
        attr_offsets.push((offset_in_heap, s.len() as u64));
        dblock.extend_from_slice(s);
    }

    // Pad to full block size
    #[expect(
        clippy::cast_possible_truncation,
        reason = "starting_block_size is a KiB-scale heap direct-block size that fits usize"
    )]
    dblock.resize(starting_block_size as usize, 0);

    // Checksum: computed over entire block with checksum field zeroed
    let dblock_checksum = crate::checksum::jenkins_lookup3(&dblock);
    dblock[cksum_pos..cksum_pos + 4].copy_from_slice(&dblock_checksum.to_le_bytes());
    debug_assert_eq!(dblock.len() as u64, starting_block_size);

    // Build heap IDs
    let heap_ids: Vec<Vec<u8>> = attr_offsets
        .iter()
        .map(|(off, len)| encode_managed_id(*off, *len, max_heap_size, heap_id_length))
        .collect();

    // Build B-tree v2 type 8 records (17 bytes each)
    let record_size: u16 = heap_id_length + 1 + 4 + 4;
    let mut records: Vec<(u32, u32, Vec<u8>)> = Vec::with_capacity(attrs.len());
    #[expect(
        clippy::cast_possible_truncation,
        reason = "i is an attribute index bounded by the attribute count, far below u32::MAX"
    )]
    for (i, heap_id) in heap_ids.iter().enumerate() {
        let mut rec = Vec::with_capacity(record_size as usize);
        rec.extend_from_slice(heap_id);
        rec.push(0); // msg_flags
        rec.extend_from_slice(&(i as u32).to_le_bytes()); // creation_order
        rec.extend_from_slice(&name_hashes[i].to_le_bytes()); // hash
        records.push((name_hashes[i], i as u32, rec));
    }
    records.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    let bthd_size = 4 + 1 + 1 + 4 + 2 + 2 + 1 + 1 + os + 2 + ls + 4;
    let num_records = attrs.len();
    let btlf_size = 4 + 1 + 1 + (num_records * record_size as usize) + 4;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "b-tree leaf node size is a small power of two (>= 512) written into the \
                  4-byte node-size field"
    )]
    let node_size = btlf_size.next_power_of_two().max(512) as u32;

    let bthd_addr = btree_addr;
    let btlf_addr = bthd_addr + bthd_size as u64;

    let mut bthd = Vec::with_capacity(bthd_size);
    bthd.extend_from_slice(b"BTHD");
    bthd.push(0); // version
    bthd.push(8); // type = attribute name index
    bthd.extend_from_slice(&node_size.to_le_bytes());
    bthd.extend_from_slice(&record_size.to_le_bytes());
    bthd.extend_from_slice(&0u16.to_le_bytes()); // depth = 0
    bthd.push(100); // split_percent
    bthd.push(40); // merge_percent
    write_offset(&mut bthd, btlf_addr, OFFSET_SIZE);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "record count is written into the 2-byte number-of-records field"
    )]
    bthd.extend_from_slice(&(num_records as u16).to_le_bytes());
    write_length(&mut bthd, num_records as u64, LENGTH_SIZE);
    let bthd_checksum = crate::checksum::jenkins_lookup3(&bthd);
    bthd.extend_from_slice(&bthd_checksum.to_le_bytes());
    debug_assert_eq!(bthd.len(), bthd_size);

    let mut btlf = Vec::with_capacity(node_size as usize);
    btlf.extend_from_slice(b"BTLF");
    btlf.push(0); // version
    btlf.push(8); // type
    for (_, _, rec) in &records {
        btlf.extend_from_slice(rec);
    }
    // Checksum goes immediately after records (NOT at end of node).
    // HDF5 C library computes checksum over sig+ver+type+records only.
    let btlf_checksum = crate::checksum::jenkins_lookup3(&btlf);
    btlf.extend_from_slice(&btlf_checksum.to_le_bytes());
    // Pad to node_size
    btlf.resize(node_size as usize, 0);

    let mut blob = Vec::with_capacity(frhp.len() + dblock.len() + bthd.len() + btlf.len());
    blob.extend_from_slice(&frhp);
    blob.extend_from_slice(&dblock);
    blob.extend_from_slice(&bthd);
    blob.extend_from_slice(&btlf);

    let attr_info = serialize_attribute_info(frhp_addr, bthd_addr);

    DenseAttrBlob {
        attr_info_message: attr_info,
        blob,
    }
}

fn encode_managed_id(offset: u64, length: u64, max_heap_size: u16, id_length: u16) -> Vec<u8> {
    let mut id = vec![0u8; id_length as usize];
    id[0] = 0x00; // type = 0 (managed)
    let combined = offset | (length << max_heap_size);
    let payload_len = (id_length as usize) - 1;
    for i in 0..payload_len.min(8) {
        id[1 + i] = ((combined >> (i * 8)) & 0xFF) as u8;
    }
    id
}

fn serialize_attribute_info(fh_addr: u64, btree_name_addr: u64) -> Vec<u8> {
    let mut data = Vec::new();
    data.push(0); // version
    data.push(0x00); // flags
    data.extend_from_slice(&fh_addr.to_le_bytes());
    data.extend_from_slice(&btree_name_addr.to_le_bytes());
    data
}

fn write_offset(buf: &mut Vec<u8>, val: u64, offset_size: u8) {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "each arm narrows to offset_size, the on-disk address width chosen for this file"
    )]
    match offset_size {
        2 => buf.extend_from_slice(&(val as u16).to_le_bytes()),
        4 => buf.extend_from_slice(&(val as u32).to_le_bytes()),
        8 => buf.extend_from_slice(&val.to_le_bytes()),
        _ => {}
    }
}

fn write_length(buf: &mut Vec<u8>, val: u64, length_size: u8) {
    write_offset(buf, val, length_size);
}

fn write_undef_offset(buf: &mut Vec<u8>, offset_size: u8) {
    for _ in 0..offset_size {
        buf.push(0xFF);
    }
}

// ---- FileWriter ----

/// The main file creation API.
pub struct FileWriter {
    root_datasets: Vec<DatasetBuilder>,
    root_attrs: Vec<(String, AttrValue)>,
    groups: Vec<FinishedGroup>,
    userblock_size: u64,
    /// Requested library-version bounds (low, high), validated in `finish`.
    /// `None` means no constraint (any output the writer produces is accepted).
    libver_bounds: Option<(LibVer, LibVer)>,
    /// File-space strategy `(strategy, persist, threshold)` from
    /// `with_file_space_strategy`. `None` leaves the file-space defaults.
    file_space_strategy: Option<(FileSpaceStrategy, bool, u64)>,
    /// File-space page size from `with_file_space_page_size`.
    file_space_page_size: Option<u64>,
}

impl Default for FileWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl FileWriter {
    pub fn new() -> Self {
        Self {
            root_datasets: Vec::new(),
            root_attrs: Vec::new(),
            groups: Vec::new(),
            userblock_size: 0,
            libver_bounds: None,
            file_space_strategy: None,
            file_space_page_size: None,
        }
    }

    /// Constrain the on-disk format version of the file, mirroring HDF5's
    /// `H5Pset_libver_bounds`. The produced file must fall within `[low, high]`;
    /// otherwise [`finish`](Self::finish) fails with
    /// [`FormatError::LibverBoundsUnsatisfiable`].
    ///
    /// This crate's writer emits exactly one format — the version 3 superblock
    /// introduced in HDF5 1.10 ([`LibVer::WRITER_OUTPUT`]) — so this is an
    /// assertion guard rather than a format selector: it lets a caller demand
    /// compatibility (and get a loud error if it cannot be met) instead of
    /// discovering an incompatible file downstream. Leaving this unset places no
    /// constraint. Bounds that straddle 1.10 (e.g. `Earliest..=Latest`) are
    /// accepted; an upper bound older than 1.10, or a lower bound newer than it,
    /// is rejected.
    pub fn with_libver_bounds(&mut self, low: LibVer, high: LibVer) -> &mut Self {
        self.libver_bounds = Some((low, high));
        self
    }

    /// Validate the requested [`libver_bounds`](Self::libver_bounds) against the
    /// format this writer actually produces.
    fn check_libver_bounds(&self) -> Result<(), FormatError> {
        if let Some((low, high)) = self.libver_bounds {
            let produced = LibVer::WRITER_OUTPUT;
            if produced < low || produced > high {
                return Err(FormatError::LibverBoundsUnsatisfiable {
                    writes: produced.name(),
                    requested_low: low.name(),
                    requested_high: high.name(),
                });
            }
        }
        Ok(())
    }

    /// Set the userblock size in bytes. Must be a power of two >= 512 or 0 (no userblock).
    /// The userblock region will be filled with zeros; the caller can write into
    /// the returned bytes at `[0..userblock_size]`.
    pub fn with_userblock(&mut self, size: u64) -> &mut Self {
        self.userblock_size = size;
        self
    }

    /// Set the file-space management strategy, mirroring
    /// `H5Pset_file_space_strategy`. The choice is recorded in the file's
    /// superblock extension so other tools (and a later reopen) see it.
    ///
    /// `persist` requests that freed space be tracked on disk across closes. A
    /// freshly built file has no free space to track, so this records the persist
    /// intent (matching what the C library writes for a brand-new persisted
    /// file); a later [`EditSession`](crate::EditSession) that frees space writes
    /// the on-disk free-space-manager blocks. `threshold` is the smallest
    /// free-space section size the managers track.
    pub fn with_file_space_strategy(
        &mut self,
        strategy: FileSpaceStrategy,
        persist: bool,
        threshold: u64,
    ) -> &mut Self {
        self.file_space_strategy = Some((strategy, persist, threshold));
        self
    }

    /// Set the file-space page size, mirroring `H5Pset_file_space_page_size`.
    /// Recorded in the superblock extension; meaningful for the paged strategy.
    pub fn with_file_space_page_size(&mut self, page_size: u64) -> &mut Self {
        self.file_space_page_size = Some(page_size);
        self
    }

    /// Reject file-space settings this writer cannot reproduce yet.
    /// The File Space Info message to write, if any file-space option was set.
    ///
    /// A freshly built file has no free space, so `persist = true` emits the
    /// persisting-but-empty form (persist flag set, all managers undefined, no
    /// FSM blocks); a later [`EditSession`](crate::EditSession) that frees space
    /// fills in the on-disk managers. `persist = false` emits the non-persistent
    /// form.
    fn file_space_info(&self) -> Option<FileSpaceInfo> {
        if self.file_space_strategy.is_none() && self.file_space_page_size.is_none() {
            return None;
        }
        let (strategy, persist, threshold) = self.file_space_strategy.unwrap_or((
            FileSpaceStrategy::FsmAggr,
            false,
            DEFAULT_THRESHOLD,
        ));
        let page_size = self.file_space_page_size.unwrap_or(DEFAULT_PAGE_SIZE);
        Some(if persist {
            FileSpaceInfo::persistent_empty(strategy, threshold, page_size)
        } else {
            FileSpaceInfo::non_persistent(strategy, threshold, page_size)
        })
    }

    /// The superblock-extension object header bytes carrying the File Space Info
    /// message, if file-space was configured.
    fn file_space_extension_oh(&self) -> Option<Vec<u8>> {
        self.file_space_info().map(|info| {
            let mut oh = ObjectHeaderWriter::new();
            // Message flags 0x14 match what the reference C library writes for
            // this message (do-not-share + mark-if-unknown); no must-understand
            // bit, so older readers still open the file.
            oh.add_message_with_flags(MessageType::FileSpaceInfo, info.serialize(), 0x14);
            oh.serialize()
        })
    }

    pub fn create_group(&mut self, name: &str) -> GroupBuilder {
        GroupBuilder::new(name)
    }

    pub fn add_group(&mut self, group: FinishedGroup) {
        self.groups.push(group);
    }

    pub fn create_dataset(&mut self, name: &str) -> &mut DatasetBuilder {
        self.root_datasets.push(DatasetBuilder::new(name));
        self.root_datasets.last_mut().unwrap()
    }

    pub fn set_root_attr(&mut self, name: &str, value: AttrValue) {
        self.root_attrs.push((name.to_string(), value));
    }

    pub fn finish(self) -> Result<Vec<u8>, FormatError> {
        let mut buf = Vec::new();
        self.finish_to_sink(&mut buf)?;
        Ok(buf)
    }

    /// Assemble the file and write it to `sink` in ascending-address order.
    /// Backs both the buffered [`finish`](Self::finish) (a `Vec<u8>` sink) and
    /// the streaming `FileBuilder::finish_to` (an `io::Write` sink), so the two
    /// produce byte-identical files. A streamed dataset's chunk bytes are pulled
    /// from its provider one chunk at a time here, never all held at once.
    pub(crate) fn finish_to_sink<S: ByteSink>(self, sink: &mut S) -> Result<(), FormatError> {
        self.check_libver_bounds()?;
        // The superblock-extension header (carrying a File Space Info message)
        // is independent of the file layout, so build it up front and place it
        // after all other content below.
        let ext_oh = self.file_space_extension_oh();
        struct DsFlat {
            name: String,
            dt: Datatype,
            ds: Dataspace,
            raw: Vec<u8>,
            attrs: Vec<AttributeMessage>,
            chunk_options: ChunkOptions,
            maxshape: Option<Vec<u64>>,
            /// Repack's verbatim chunk payload, when this dataset's chunks are
            /// copied compressed-as-is rather than encoded from `raw`.
            raw_chunks: Option<crate::type_builders::RawChunkPayload>,
            reference_targets: Option<Vec<crate::type_builders::ObjectRefTarget>>,
            /// Staged global heap collection + patch mask for a VL-string
            /// dataset, whose element references in `raw` need their heap
            /// addresses patched once the post-data cursor is known.
            vl_string_staging: Option<VlStringStaging>,
            /// A user-defined fill value, encoded in the dataset's datatype, or
            /// `None` for the library default. Validated against the datatype
            /// element size in `flatten_dataset`.
            fill: Option<Vec<u8>>,
        }

        /// One dataset's data region for the assembly pass: either materialized
        /// in memory, or a plan whose chunk bytes are streamed from a provider.
        enum DsData {
            InMemory(Vec<u8>),
            /// A verbatim chunked dataset streamed one chunk at a time; the
            /// provider lives in the matching `DsFlat.raw_chunks` (`Lazy`).
            Streamed(VerbatimPlan),
        }
        impl DsData {
            fn len(&self) -> u64 {
                match self {
                    DsData::InMemory(v) => v.len() as u64,
                    DsData::Streamed(plan) => plan.total_len,
                }
            }
        }

        /// A built chunked dataset's layout/pipeline messages plus its data
        /// region (materialized for the encode and eager-verbatim paths, planned
        /// for the streamed verbatim path).
        struct ChunkedBuilt {
            layout_message: Vec<u8>,
            pipeline_message: Option<Vec<u8>>,
            data: DsData,
        }

        /// Build the chunked data + layout/pipeline messages for one chunked
        /// dataset at `base_address`, dispatching to the verbatim path when the
        /// dataset carries a raw-chunk payload, else the normal encode path. The
        /// single dispatch point keeps the dummy-sizing and real-address passes
        /// from diverging. The layout is computed from chunk *sizes* alone, so
        /// it is identical whether the chunks are in memory or streamed.
        fn build_chunked(
            d: &DsFlat,
            base_address: u64,
            chunk_set: Option<&CompressedChunkSet>,
        ) -> Result<ChunkedBuilt, FormatError> {
            if let Some(rc) = &d.raw_chunks {
                // Verbatim chunks are always streamed: the layout is planned from
                // chunk sizes alone, and the bytes are pulled from the provider in
                // the assembly loop (buffered `finish` and streaming `finish_to`
                // share that one emitter, so their output is byte-identical).
                let VerbatimLayout {
                    plan,
                    layout_message,
                    pipeline_message,
                } = plan_chunked_data_verbatim(
                    &rc.meta,
                    &rc.chunk_dims,
                    rc.element_size,
                    rc.raw_size,
                    rc.pipeline_message.as_deref(),
                    base_address,
                    d.maxshape.as_deref(),
                )?;
                Ok(ChunkedBuilt {
                    layout_message,
                    pipeline_message,
                    data: DsData::Streamed(plan),
                })
            } else {
                // Encode path: the chunks were compressed once up front; just lay
                // the cached set out at this address (no recompression).
                let set = chunk_set
                    .expect("an encode-path chunked dataset must have a precomputed chunk set");
                let result = assemble_chunked_at(set, base_address)?;
                Ok(ChunkedBuilt {
                    layout_message: result.layout_message,
                    pipeline_message: result.pipeline_message,
                    data: DsData::InMemory(result.data_bytes),
                })
            }
        }
        struct GrpFlat {
            name: String,
            attrs: Vec<AttributeMessage>,
            ds_indices: Vec<usize>,
            sub_group_indices: Vec<usize>,
        }

        let mut all_ds: Vec<DsFlat> = Vec::new();
        let mut groups: Vec<GrpFlat> = Vec::new();
        let mut root_ds_indices: Vec<usize> = Vec::new();
        let mut root_group_indices: Vec<usize> = Vec::new();

        fn flatten_dataset(
            db: DatasetBuilder,
            all_ds: &mut Vec<DsFlat>,
            ds_vl: &mut Vec<Vec<VlPatch>>,
        ) -> Result<usize, FormatError> {
            let dt = db.datatype.ok_or(FormatError::DatasetMissingData)?;
            let shape = db.shape.ok_or(FormatError::DatasetMissingShape)?;
            // A verbatim-chunk dataset (repack) owns no flat `raw` element bytes;
            // its storage is the pre-compressed chunks in `raw_chunks`. Skip the
            // flat-data requirement and the shape/data-length check for it.
            let raw_chunks = db.raw_chunks;
            // Allow empty data for zero-element datasets (e.g. shape [0, 0]).
            let is_empty = shape.contains(&0);
            let raw = if is_empty || raw_chunks.is_some() {
                db.data.unwrap_or_default()
            } else {
                db.data.ok_or(FormatError::DatasetMissingData)?
            };
            // Guard against a shape that disagrees with the supplied data. The
            // reader enforces the same `num_elements * element_size` invariant
            // (see `data_read::read_raw_data_full`), so without this check a
            // mismatch (e.g. data for 3 elements with shape `[2, 2]`) would
            // produce a file that fails to read back. `saturating_mul` keeps an
            // absurd shape from overflowing into a false match.
            let elem_size = dt.type_size() as u64;
            if !is_empty && raw_chunks.is_none() && elem_size > 0 {
                // Multiply with checked arithmetic, saturating on overflow: an
                // absurd shape whose element count exceeds `u64` must not panic a
                // debug build in `Iterator::product` (nor silently wrap a release
                // build into a false match). A saturated `u64::MAX` can never
                // equal a real `data.len()`, so it is correctly reported as a
                // mismatch.
                let num_elements = shape
                    .iter()
                    .copied()
                    .try_fold(1u64, |acc, d| acc.checked_mul(d))
                    .unwrap_or(u64::MAX);
                let expected = num_elements.saturating_mul(elem_size);
                if raw.len() as u64 != expected {
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "byte counts reported in a shape-mismatch error; display-only"
                    )]
                    return Err(FormatError::ShapeDataMismatch {
                        expected: expected as usize,
                        actual: raw.len(),
                        element_size: elem_size as usize,
                    });
                }
            }
            // Validate the chunk geometry up front for a chunked / filtered /
            // extensible dataset, so a malformed request (chunk dimensions of the
            // wrong rank, a zero chunk dimension, a bad maximum shape, or
            // chunking a scalar) is refused here instead of panicking in the
            // chunk splitter or producing an unreadable dataset.
            if db.chunk_options.is_chunked() || db.maxshape.is_some() {
                db.chunk_options
                    .validate_geometry(&shape, db.maxshape.as_deref())
                    .map_err(FormatError::InvalidChunkGeometry)?;
            }
            // Variable-length string element references live in the global heap,
            // whose addresses are only known after all dataset data is laid
            // out. For chunked/filtered/resizable storage the references sit
            // inside compressed chunks written before those addresses exist, so
            // the heap addresses cannot be patched in. Refuse rather than emit a
            // dataset with dangling VL references.
            if db.vl_string_staging.is_some()
                && (db.chunk_options.is_chunked() || db.maxshape.is_some())
            {
                return Err(FormatError::ChunkedVlenStringUnsupported);
            }
            let max_dimensions = db.maxshape.clone();
            let dspace = Dataspace {
                space_type: if shape.is_empty() {
                    DataspaceType::Scalar
                } else {
                    DataspaceType::Simple
                },
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "dataspace rank fits the 1-byte dimensionality field (HDF5 caps \
                              rank at 32)"
                )]
                rank: shape.len() as u8,
                dimensions: shape,
                max_dimensions,
            };
            let patches = collect_vl_patches(&db.attrs);
            let mut attrs = Vec::new();
            for (n, v) in &db.attrs {
                attrs.push(build_attr_message(n, v));
            }
            #[cfg(feature = "provenance")]
            if let Some(ref prov) = db.provenance {
                let p = crate::provenance::Provenance {
                    creator: prov.creator.clone(),
                    timestamp: prov.timestamp.clone(),
                    source: prov.source.clone(),
                };
                attrs.extend(p.build_attrs(&raw));
            }
            // A user-defined fill value is one element wide, so its byte length
            // must equal the datatype's element size.
            if let Some(fill) = &db.fill {
                let expected = elem_size.to_usize()?;
                if fill.len() != expected {
                    return Err(FormatError::FillValueSizeMismatch {
                        expected,
                        actual: fill.len(),
                    });
                }
            }
            let idx = all_ds.len();
            all_ds.push(DsFlat {
                name: db.name,
                dt,
                ds: dspace,
                raw,
                attrs,
                chunk_options: db.chunk_options,
                maxshape: db.maxshape,
                raw_chunks,
                reference_targets: db.reference_targets,
                vl_string_staging: db.vl_string_staging,
                fill: db.fill,
            });
            ds_vl.push(patches);
            Ok(idx)
        }

        fn flatten_group(
            g: FinishedGroup,
            all_ds: &mut Vec<DsFlat>,
            groups: &mut Vec<GrpFlat>,
            grp_vl: &mut Vec<Vec<VlPatch>>,
            ds_vl: &mut Vec<Vec<VlPatch>>,
        ) -> Result<usize, FormatError> {
            let patches = collect_vl_patches(&g.attrs);
            let mut gattrs = Vec::new();
            for (n, v) in &g.attrs {
                gattrs.push(build_attr_message(n, v));
            }
            let mut ds_idx = Vec::new();
            for db in g.datasets {
                ds_idx.push(flatten_dataset(db, all_ds, ds_vl)?);
            }
            let mut sub_grp_idx = Vec::new();
            for sg in g.sub_groups {
                sub_grp_idx.push(flatten_group(sg, all_ds, groups, grp_vl, ds_vl)?);
            }
            let gi = groups.len();
            groups.push(GrpFlat {
                name: g.name,
                attrs: gattrs,
                ds_indices: ds_idx,
                sub_group_indices: sub_grp_idx,
            });
            grp_vl.push(patches);
            Ok(gi)
        }

        let mut grp_vl: Vec<Vec<VlPatch>> = Vec::new();
        let mut ds_vl: Vec<Vec<VlPatch>> = Vec::new();

        for db in self.root_datasets {
            root_ds_indices.push(flatten_dataset(db, &mut all_ds, &mut ds_vl)?);
        }

        for g in self.groups.into_iter() {
            root_group_indices.push(flatten_group(
                g,
                &mut all_ds,
                &mut groups,
                &mut grp_vl,
                &mut ds_vl,
            )?);
        }

        // Build global heap collections for VarLenAsciiArray attributes.
        // Track which attribute messages need VL patching, across root, groups, and datasets.
        struct VlPatch {
            collection_bytes: Vec<u8>,
            attr_index: usize, // index into the relevant attrs Vec
        }

        fn collect_vl_patches(attrs_raw: &[(String, AttrValue)]) -> Vec<VlPatch> {
            let mut patches = Vec::new();
            for (i, (_n, v)) in attrs_raw.iter().enumerate() {
                if let AttrValue::VarLenAsciiArray(strings) = v {
                    let str_refs: Vec<&str> = strings.iter().map(|s| s.as_str()).collect();
                    patches.push(VlPatch {
                        collection_bytes: build_global_heap_collection(&str_refs),
                        attr_index: i,
                    });
                }
            }
            patches
        }

        let vl_root = collect_vl_patches(&self.root_attrs);

        let mut root_attrs: Vec<AttributeMessage> = Vec::new();
        for (n, v) in &self.root_attrs {
            root_attrs.push(build_attr_message(n, v));
        }

        let is_chunked: Vec<bool> = all_ds
            .iter()
            .map(|d| d.chunk_options.is_chunked() || d.maxshape.is_some() || d.raw_chunks.is_some())
            .collect();

        // Compress each encode-path chunked dataset exactly once, up front. The
        // object-header sizing pass and the data-emit pass both need the chunk
        // layout, but only the embedded addresses differ between them — the
        // (expensive) compression does not. Caching the `CompressedChunkSet` here
        // lets both passes call the cheap `assemble_chunked_at` instead of
        // recompressing the whole dataset twice. Verbatim datasets carry their
        // bytes pre-compressed and are planned (not recompressed), so they get no
        // entry.
        let chunk_sets: Vec<Option<CompressedChunkSet>> = all_ds
            .iter()
            .enumerate()
            .map(|(i, d)| {
                if is_chunked[i] && d.raw_chunks.is_none() {
                    let chunk_dims = d.chunk_options.resolve_chunk_dims(&d.ds.dimensions);
                    let ctx = crate::filters::ChunkContext::from_datatype(&chunk_dims, &d.dt);
                    Ok(Some(compress_chunks(
                        &d.raw,
                        &d.ds.dimensions,
                        ctx,
                        &d.chunk_options,
                        d.maxshape.as_deref(),
                    )?))
                } else {
                    Ok(None)
                }
            })
            .collect::<Result<_, FormatError>>()?;

        let root_dense = root_attrs.len() > DENSE_ATTR_THRESHOLD;
        let group_dense: Vec<bool> = groups
            .iter()
            .map(|g| g.attrs.len() > DENSE_ATTR_THRESHOLD)
            .collect();
        let ds_dense: Vec<bool> = all_ds
            .iter()
            .map(|d| d.attrs.len() > DENSE_ATTR_THRESHOLD)
            .collect();

        // Pass 1: compute OH sizes with dummy addresses
        let group_oh_sizes: Vec<usize> = groups
            .iter()
            .enumerate()
            .map(|(gi, g)| {
                let mut dummy_links: Vec<LinkMessage> = g
                    .ds_indices
                    .iter()
                    .map(|&i| make_link(&all_ds[i].name, 0))
                    .collect();
                for &sgi in &g.sub_group_indices {
                    dummy_links.push(make_link(&groups[sgi].name, 0));
                }
                if group_dense[gi] {
                    let dummy_blob = build_dense_attrs(&g.attrs, 0);
                    build_group_oh(&dummy_links, &g.attrs, Some(&dummy_blob)).len()
                } else {
                    build_group_oh(&dummy_links, &g.attrs, None).len()
                }
            })
            .collect();

        let root_dummy_links: Vec<LinkMessage> = {
            let mut links = Vec::new();
            for &i in &root_ds_indices {
                links.push(make_link(&all_ds[i].name, 0));
            }
            for &gi in &root_group_indices {
                links.push(make_link(&groups[gi].name, 0));
            }
            links
        };
        let root_oh_size = if root_dense {
            let dummy_blob = build_dense_attrs(&root_attrs, 0);
            build_group_oh(&root_dummy_links, &root_attrs, Some(&dummy_blob)).len()
        } else {
            build_group_oh(&root_dummy_links, &root_attrs, None).len()
        };

        // Pass 1: compute dataset object-header sizes from a dummy layout. No
        // data bytes are materialized here — the object-header size depends only
        // on the layout/pipeline messages, and a chunk index's byte size is a
        // function of chunk count/size, not of the (dummy) base address. For a
        // streamed (lazy) dataset this touches no chunk bytes at all.
        let mut actual_ds_oh_sizes: Vec<usize> = Vec::with_capacity(all_ds.len());
        let mut dummy_cursor = 0u64;
        for (i, d) in all_ds.iter().enumerate() {
            let dense_blob = if ds_dense[i] {
                Some(build_dense_attrs(&d.attrs, 0))
            } else {
                None
            };
            let oh = if is_chunked[i] {
                let built = build_chunked(d, dummy_cursor, chunk_sets[i].as_ref())?;
                dummy_cursor += built.data.len();
                build_chunked_dataset_oh(
                    &d.dt,
                    &d.ds,
                    &built.layout_message,
                    built.pipeline_message.as_deref(),
                    &d.attrs,
                    dense_blob.as_ref(),
                    d.fill.as_deref(),
                )
            } else {
                build_dataset_oh(
                    &d.dt,
                    &d.ds,
                    0,
                    d.raw.len() as u64,
                    &d.attrs,
                    dense_blob.as_ref(),
                    d.fill.as_deref(),
                )
            };
            actual_ds_oh_sizes.push(oh.len());
        }

        // Pass 2: compute real addresses.
        // All addresses stored in the file are relative to base_address.
        // base_address = userblock_size. cursor2 tracks relative positions.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "userblock_size is a small power-of-two header size used as an in-memory \
                      buffer offset; it fits usize on every supported target"
        )]
        let ub = self.userblock_size as usize;
        let root_group_addr = SUPERBLOCK_SIZE as u64;
        let mut cursor2 = SUPERBLOCK_SIZE + root_oh_size;

        let root_dense_blob = if root_dense {
            let blob = build_dense_attrs(&root_attrs, cursor2 as u64);
            cursor2 += blob.blob.len();
            Some(blob)
        } else {
            None
        };

        let mut group_dense_blobs: Vec<Option<DenseAttrBlob>> = Vec::new();
        let group_addrs2: Vec<u64> = group_oh_sizes
            .iter()
            .enumerate()
            .map(|(gi, &sz)| {
                let addr = cursor2 as u64;
                cursor2 += sz;
                if group_dense[gi] {
                    let blob = build_dense_attrs(&groups[gi].attrs, cursor2 as u64);
                    cursor2 += blob.blob.len();
                    group_dense_blobs.push(Some(blob));
                } else {
                    group_dense_blobs.push(None);
                }
                addr
            })
            .collect();

        let mut ds_dense_blobs: Vec<Option<DenseAttrBlob>> = Vec::new();
        let ds_oh_addrs2: Vec<u64> = actual_ds_oh_sizes
            .iter()
            .enumerate()
            .map(|(i, &sz)| {
                let addr = cursor2 as u64;
                cursor2 += sz;
                if ds_dense[i] {
                    let blob = build_dense_attrs(&all_ds[i].attrs, cursor2 as u64);
                    cursor2 += blob.blob.len();
                    ds_dense_blobs.push(Some(blob));
                } else {
                    ds_dense_blobs.push(None);
                }
                addr
            })
            .collect();

        // Resolve path-based references now that all addresses are known.
        // Build a map of (group_name, child_name) -> address for resolution.
        {
            // Build a path->address map for all datasets and groups.
            // Root-level datasets: path = dataset_name
            // Group-level datasets: path = group_name/dataset_name (recursive)
            // Groups: path = group_name (recursive)
            let mut path_map = HashMap::<String, u64>::new();
            // The root group is referenceable under the empty path (repack maps a
            // reference to the source root group to "").
            path_map.insert(String::new(), root_group_addr);
            for &i in &root_ds_indices {
                path_map.insert(all_ds[i].name.clone(), ds_oh_addrs2[i]);
            }
            for &gi in &root_group_indices {
                fn register_group(
                    prefix: &str,
                    gi: usize,
                    groups: &[GrpFlat],
                    ds_addrs: &[u64],
                    grp_addrs: &[u64],
                    all_ds: &[DsFlat],
                    map: &mut HashMap<String, u64>,
                ) {
                    map.insert(prefix.to_string(), grp_addrs[gi]);
                    for &di in &groups[gi].ds_indices {
                        map.insert(format!("{}/{}", prefix, all_ds[di].name), ds_addrs[di]);
                    }
                    for &sgi in &groups[gi].sub_group_indices {
                        register_group(
                            &format!("{}/{}", prefix, groups[sgi].name),
                            sgi,
                            groups,
                            ds_addrs,
                            grp_addrs,
                            all_ds,
                            map,
                        );
                    }
                }
                register_group(
                    &groups[gi].name,
                    gi,
                    &groups,
                    &ds_oh_addrs2,
                    &group_addrs2,
                    &all_ds,
                    &mut path_map,
                );
            }

            // Patch reference datasets: a path target resolves to its object's
            // destination address (an unknown path falls back to the undefined
            // address); a raw target is written verbatim (null / undefined).
            for d in all_ds.iter_mut() {
                if let Some(ref targets) = d.reference_targets {
                    let mut patched = Vec::with_capacity(targets.len() * 8);
                    for target in targets {
                        let addr = match target {
                            crate::type_builders::ObjectRefTarget::Path(path) => {
                                path_map.get(path).copied().unwrap_or(u64::MAX)
                            }
                            crate::type_builders::ObjectRefTarget::Raw(addr) => *addr,
                        };
                        patched.extend_from_slice(&addr.to_le_bytes());
                    }
                    d.raw = patched;
                }
            }
        }

        // Compute data layout (addresses + chunked data blobs) separately from OHs
        // so we can patch VL attrs before building OHs.
        struct DsLayout {
            data: DsData,
            data_addr: u64,
            chunked_msgs: Option<(Vec<u8>, Option<Vec<u8>>)>,
        }
        let mut ds_layouts: Vec<DsLayout> = Vec::new();
        for (i, d) in all_ds.iter_mut().enumerate() {
            if is_chunked[i] {
                let base_address = cursor2 as u64;
                let built = build_chunked(d, base_address, chunk_sets[i].as_ref())?;
                cursor2 += built.data.len().to_usize()?;
                ds_layouts.push(DsLayout {
                    data: built.data,
                    data_addr: base_address,
                    chunked_msgs: Some((built.layout_message, built.pipeline_message)),
                });
            } else {
                // `d.raw` is not read again for a contiguous/compact dataset, so
                // move its element buffer into the layout rather than cloning it.
                let data = core::mem::take(&mut d.raw);
                let addr = if data.is_empty() {
                    u64::MAX
                } else {
                    let a = cursor2 as u64;
                    cursor2 += data.len();
                    a
                };
                ds_layouts.push(DsLayout {
                    data: DsData::InMemory(data),
                    data_addr: addr,
                    chunked_msgs: None,
                });
            }
        }

        // Patch VL references (attribute and dataset-element) with the GCOL
        // addresses, which sit after all dataset data. Attribute collections are
        // emitted first (root, groups, datasets), then dataset-element
        // collections, and the cursor walk below assigns addresses in that same
        // order so it matches the emission order at the end of the buffer.
        let has_vl = !vl_root.is_empty()
            || grp_vl.iter().any(|v| !v.is_empty())
            || ds_vl.iter().any(|v| !v.is_empty())
            || all_ds.iter().any(|d| d.vl_string_staging.is_some());

        let mut gcol_total_size = 0usize;
        if has_vl {
            let mut gcol_cursor = cursor2 as u64;
            for patch in &vl_root {
                patch_vl_refs(&mut root_attrs[patch.attr_index].raw_data, gcol_cursor);
                gcol_cursor += patch.collection_bytes.len() as u64;
            }
            for (gi, patches) in grp_vl.iter().enumerate() {
                for patch in patches {
                    patch_vl_refs(
                        &mut groups[gi].attrs[patch.attr_index].raw_data,
                        gcol_cursor,
                    );
                    gcol_cursor += patch.collection_bytes.len() as u64;
                }
            }
            for (di, patches) in ds_vl.iter().enumerate() {
                for patch in patches {
                    patch_vl_refs(
                        &mut all_ds[di].attrs[patch.attr_index].raw_data,
                        gcol_cursor,
                    );
                    gcol_cursor += patch.collection_bytes.len() as u64;
                }
            }
            // Dataset-element VL references. The references live in the
            // contiguous/compact element bytes (`ds_layouts[i].data`, cloned
            // from `d.raw`); chunked/filtered/resizable VL datasets were refused
            // in `flatten_dataset`, so every staged dataset is here non-chunked.
            for (i, d) in all_ds.iter().enumerate() {
                if let Some(staging) = &d.vl_string_staging {
                    // A staged VL-string dataset is always non-chunked (chunked
                    // VL is refused in `flatten_dataset`), so its element bytes
                    // are in memory and patchable in place. A streamed (lazy)
                    // dataset never carries VL staging, so this is unreachable for
                    // it — assert that rather than risk silently corrupting one.
                    let DsData::InMemory(ref mut bytes) = ds_layouts[i].data else {
                        unreachable!(
                            "VL-string staging is refused on chunked datasets, so a staged \
                             VL dataset's data is always in memory"
                        );
                    };
                    patch_vl_refs_masked(bytes, &staging.patch_mask, gcol_cursor);
                    gcol_cursor += staging.collection_bytes.len() as u64;
                }
            }
            #[expect(
                clippy::cast_possible_truncation,
                reason = "global-heap total size is an in-memory output span bounded by \
                          addressable memory on the target"
            )]
            {
                gcol_total_size = (gcol_cursor - cursor2 as u64) as usize;
            }
        }

        // Build dataset OHs now that attrs are patched. Only the header bytes
        // are kept here; each dataset's data is emitted directly from
        // `ds_layouts` in the assembly loop (a streamed dataset has no data
        // bytes to keep at all).
        let mut ds_oh_bytes2: Vec<Vec<u8>> = Vec::with_capacity(all_ds.len());
        for (i, d) in all_ds.iter().enumerate() {
            let layout = &ds_layouts[i];
            let oh = if let Some((ref lm, ref pm)) = layout.chunked_msgs {
                build_chunked_dataset_oh(
                    &d.dt,
                    &d.ds,
                    lm,
                    pm.as_deref(),
                    &d.attrs,
                    ds_dense_blobs[i].as_ref(),
                    d.fill.as_deref(),
                )
            } else {
                build_dataset_oh(
                    &d.dt,
                    &d.ds,
                    layout.data_addr,
                    layout.data.len(),
                    &d.attrs,
                    ds_dense_blobs[i].as_ref(),
                    d.fill.as_deref(),
                )
            };
            ds_oh_bytes2.push(oh);
        }

        let actual_ds_oh_sizes2: Vec<usize> = ds_oh_bytes2.iter().map(|b| b.len()).collect();
        debug_assert_eq!(actual_ds_oh_sizes, actual_ds_oh_sizes2);

        // The superblock extension, if any, is appended after the GCOLs. Its
        // address is base-relative (like every other stored address); the reader
        // adds the base address. eof grows by the extension's size.
        let ext_addr = ext_oh.as_ref().map(|_| (cursor2 + gcol_total_size) as u64);
        let ext_len = ext_oh.as_ref().map_or(0, |b| b.len());

        // eof_address is absolute file size (includes userblock + GCOLs + ext)
        let eof_addr2 = (ub + cursor2 + gcol_total_size + ext_len) as u64;

        // Let a buffered (Vec) sink preallocate the whole file up front, as the
        // writer did before streaming; a streaming sink ignores this.
        sink.reserve(eof_addr2.to_usize()?);

        // Userblock: prepend zeros
        if ub > 0 {
            sink.put_zeros(ub)?;
        }

        let sb = Superblock {
            version: 3,
            offset_size: OFFSET_SIZE,
            length_size: LENGTH_SIZE,
            base_address: ub as u64,
            eof_address: eof_addr2,
            root_group_address: root_group_addr,
            group_leaf_node_k: None,
            group_internal_node_k: None,
            indexed_storage_internal_node_k: None,
            free_space_address: None,
            driver_info_address: None,
            consistency_flags: 0,
            superblock_extension_address: Some(ext_addr.unwrap_or(u64::MAX)),
            checksum: None,
        };
        sink.put(&sb.serialize())?;

        // Root group OH
        let root_links: Vec<LinkMessage> = {
            let mut v = Vec::new();
            for &i in &root_ds_indices {
                v.push(make_link(&all_ds[i].name, ds_oh_addrs2[i]));
            }
            for &gi in &root_group_indices {
                v.push(make_link(&groups[gi].name, group_addrs2[gi]));
            }
            v
        };
        sink.put(&build_group_oh(
            &root_links,
            &root_attrs,
            root_dense_blob.as_ref(),
        ))?;
        if let Some(ref blob) = root_dense_blob {
            sink.put(&blob.blob)?;
        }

        // Group OHs + dense blobs
        for (gi, g) in groups.iter().enumerate() {
            let mut links: Vec<LinkMessage> = g
                .ds_indices
                .iter()
                .map(|&i| make_link(&all_ds[i].name, ds_oh_addrs2[i]))
                .collect();
            for &sgi in &g.sub_group_indices {
                links.push(make_link(&groups[sgi].name, group_addrs2[sgi]));
            }
            sink.put(&build_group_oh(
                &links,
                &g.attrs,
                group_dense_blobs[gi].as_ref(),
            ))?;
            if let Some(ref blob) = group_dense_blobs[gi] {
                sink.put(&blob.blob)?;
            }
        }

        // Dataset OHs + dense blobs
        for (i, oh) in ds_oh_bytes2.iter().enumerate() {
            sink.put(oh)?;
            if let Some(ref dense) = ds_dense_blobs[i] {
                sink.put(&dense.blob)?;
            }
        }

        // Data. Contiguous/compact and eager chunked datasets emit their
        // in-memory bytes; a streamed (lazy) chunked dataset pulls each chunk
        // from its provider one at a time, so its bytes never all reside here.
        for (i, layout) in ds_layouts.iter().enumerate() {
            match &layout.data {
                DsData::InMemory(bytes) => sink.put(bytes)?,
                DsData::Streamed(plan) => {
                    let provider = match all_ds[i].raw_chunks.as_ref() {
                        Some(rc) => rc.provider.0.as_ref(),
                        None => unreachable!("a streamed data region implies a raw-chunk payload"),
                    };
                    emit_chunked_data_verbatim(sink, plan, provider)?;
                }
            }
        }

        // Global heap collections
        for patch in &vl_root {
            sink.put(&patch.collection_bytes)?;
        }
        for patches in &grp_vl {
            for patch in patches {
                sink.put(&patch.collection_bytes)?;
            }
        }
        for patches in &ds_vl {
            for patch in patches {
                sink.put(&patch.collection_bytes)?;
            }
        }
        // Dataset-element VL string collections, in the same order their
        // addresses were assigned above.
        for d in &all_ds {
            if let Some(staging) = &d.vl_string_staging {
                sink.put(&staging.collection_bytes)?;
            }
        }

        // Superblock extension (File Space Info), at the address recorded above.
        if let Some(bytes) = &ext_oh {
            debug_assert_eq!(
                sink.position(),
                ub as u64 + ext_addr.unwrap(),
                "extension header must land at its recorded base-relative address"
            );
            sink.put(bytes)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::group_v2::resolve_path_any;
    use crate::link_info::LinkInfoMessage;
    use crate::object_header::ObjectHeader;
    use crate::signature;

    fn parse_file(bytes: &[u8]) -> (Superblock, ObjectHeader) {
        let sig = signature::find_signature(bytes).unwrap();
        let sb = Superblock::parse(bytes, sig).unwrap();
        let oh = ObjectHeader::parse(
            bytes,
            sb.root_group_address as usize,
            sb.offset_size,
            sb.length_size,
        )
        .unwrap();
        (sb, oh)
    }

    fn read_dataset_f64(bytes: &[u8], path: &str) -> Vec<f64> {
        let sig = signature::find_signature(bytes).unwrap();
        let sb = Superblock::parse(bytes, sig).unwrap();
        let addr = resolve_path_any(bytes, &sb, path).unwrap();
        let hdr =
            ObjectHeader::parse(bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
        let dt_data = &hdr
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::Datatype)
            .unwrap()
            .data;
        let ds_data = &hdr
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::Dataspace)
            .unwrap()
            .data;
        let dl_data = &hdr
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::DataLayout)
            .unwrap()
            .data;
        let (dt, _) = Datatype::parse(dt_data).unwrap();
        let ds = Dataspace::parse(ds_data, sb.length_size).unwrap();
        let dl =
            crate::data_layout::DataLayout::parse(dl_data, sb.offset_size, sb.length_size).unwrap();
        let raw = crate::data_read::read_raw_data(bytes, &dl, &ds, &dt).unwrap();
        crate::data_read::read_as_f64(&raw, &dt).unwrap()
    }

    #[test]
    fn empty_file_root_group_only() {
        let fw = FileWriter::new();
        let bytes = fw.finish().unwrap();
        let (sb, oh) = parse_file(&bytes);
        assert_eq!(sb.version, 3);
        assert_eq!(oh.version, 2);
    }

    #[test]
    fn file_with_f64_dataset() {
        let mut fw = FileWriter::new();
        fw.create_dataset("data").with_f64_data(&[1.0, 2.0, 3.0]);
        let bytes = fw.finish().unwrap();
        assert_eq!(read_dataset_f64(&bytes, "data"), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn file_with_dataset_attrs() {
        let mut fw = FileWriter::new();
        fw.create_dataset("data")
            .with_f64_data(&[1.0, 2.0])
            .set_attr("scale", AttrValue::F64(0.5));
        let bytes = fw.finish().unwrap();
        assert_eq!(read_dataset_f64(&bytes, "data"), vec![1.0, 2.0]);
        let sig = signature::find_signature(&bytes).unwrap();
        let sb = Superblock::parse(&bytes, sig).unwrap();
        let addr = resolve_path_any(&bytes, &sb, "data").unwrap();
        let hdr =
            ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
        let attrs = crate::attribute::extract_attributes(&hdr, sb.length_size).unwrap();
        assert_eq!(attrs.len(), 1);
        assert_eq!(attrs[0].name, "scale");
    }

    #[test]
    fn file_with_group_and_dataset() {
        let mut fw = FileWriter::new();
        let mut gb = fw.create_group("grp");
        gb.create_dataset("vals").with_f64_data(&[10.0, 20.0]);
        fw.add_group(gb.finish());
        let bytes = fw.finish().unwrap();
        assert_eq!(read_dataset_f64(&bytes, "grp/vals"), vec![10.0, 20.0]);
    }

    // hdf5-pure has no group creation property list: every object header it
    // writes is fixed to one shape, equivalent to the C library's
    // `obj_track_times = false` (see issue #131) — never toggleable, so these
    // lock in the "no timestamps" half of that fixed shape for both the root
    // group and an ordinary sub-group.
    #[test]
    fn root_group_carries_no_timestamps() {
        let fw = FileWriter::new();
        let bytes = fw.finish().unwrap();
        let (_, oh) = parse_file(&bytes);
        assert_eq!(oh.flags & 0x20, 0, "times-stored flag must be clear");
        assert!(oh.modification_time.is_none());
        assert!(oh.access_time.is_none());
        assert!(oh.change_time.is_none());
        assert!(oh.birth_time.is_none());
    }

    #[test]
    fn sub_group_carries_no_timestamps() {
        let mut fw = FileWriter::new();
        let mut gb = fw.create_group("grp");
        gb.create_dataset("vals").with_f64_data(&[1.0]);
        fw.add_group(gb.finish());
        let bytes = fw.finish().unwrap();
        let sig = signature::find_signature(&bytes).unwrap();
        let sb = Superblock::parse(&bytes, sig).unwrap();
        let addr = resolve_path_any(&bytes, &sb, "grp").unwrap();
        let hdr =
            ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
        assert_eq!(hdr.flags & 0x20, 0, "times-stored flag must be clear");
        assert!(hdr.modification_time.is_none());
    }

    // The other half of the fixed shape: every group is "new style" (a Link
    // Info + Group Info message pair) with links stored inline, regardless of
    // child count — hdf5-pure never converts a group to dense (fractal-heap)
    // link storage on write (see issue #131 and the tracked gap in #102).
    #[test]
    fn group_links_stay_compact_regardless_of_child_count() {
        let mut fw = FileWriter::new();
        let mut gb = fw.create_group("grp");
        for i in 0..20 {
            gb.create_dataset(&format!("d{i}"))
                .with_f64_data(&[i as f64]);
        }
        fw.add_group(gb.finish());
        let bytes = fw.finish().unwrap();
        let sig = signature::find_signature(&bytes).unwrap();
        let sb = Superblock::parse(&bytes, sig).unwrap();
        let addr = resolve_path_any(&bytes, &sb, "grp").unwrap();
        let hdr =
            ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();

        let link_info_msg = hdr
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::LinkInfo)
            .unwrap();
        let link_info = LinkInfoMessage::parse(&link_info_msg.data, sb.offset_size).unwrap();
        assert!(
            link_info.fractal_heap_address.is_none(),
            "no dense link storage is ever used"
        );

        let group_info_msg = hdr
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::GroupInfo)
            .unwrap();
        assert_eq!(group_info_msg.data, vec![0, 0]);

        let link_count = hdr
            .messages
            .iter()
            .filter(|m| m.msg_type == MessageType::Link)
            .count();
        assert_eq!(link_count, 20);
    }

    #[test]
    fn file_with_root_attr() {
        let mut fw = FileWriter::new();
        fw.set_root_attr("version", AttrValue::I64(42));
        let bytes = fw.finish().unwrap();
        let (sb, oh) = parse_file(&bytes);
        let attrs = crate::attribute::extract_attributes(&oh, sb.length_size).unwrap();
        assert_eq!(attrs[0].name, "version");
    }

    #[test]
    fn dense_attrs_self_roundtrip() {
        let mut fw = FileWriter::new();
        let ds = fw.create_dataset("data");
        ds.with_f64_data(&[1.0, 2.0, 3.0]);
        for i in 0..20 {
            ds.set_attr(&format!("attr_{i:03}"), AttrValue::F64(i as f64 * 1.5));
        }
        let bytes = fw.finish().unwrap();
        let sig = signature::find_signature(&bytes).unwrap();
        let sb = Superblock::parse(&bytes, sig).unwrap();
        let addr = resolve_path_any(&bytes, &sb, "data").unwrap();
        let hdr =
            ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
        let attrs =
            crate::attribute::extract_attributes_full(&bytes, &hdr, sb.offset_size, sb.length_size)
                .unwrap();
        assert_eq!(attrs.len(), 20);
        for i in 0..20 {
            let attr = attrs
                .iter()
                .find(|a| a.name == format!("attr_{i:03}"))
                .unwrap();
            let v = attr.read_as_f64().unwrap();
            assert!((v[0] - i as f64 * 1.5).abs() < 1e-10);
        }
        assert_eq!(read_dataset_f64(&bytes, "data"), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn dense_attrs_root_group_self_roundtrip() {
        let mut fw = FileWriter::new();
        fw.create_dataset("dummy").with_f64_data(&[0.0]);
        for i in 0..15 {
            fw.set_root_attr(&format!("root_{i:02}"), AttrValue::F64(i as f64 * 2.0));
        }
        let bytes = fw.finish().unwrap();
        let sig = signature::find_signature(&bytes).unwrap();
        let sb = Superblock::parse(&bytes, sig).unwrap();
        let oh = ObjectHeader::parse(
            &bytes,
            sb.root_group_address as usize,
            sb.offset_size,
            sb.length_size,
        )
        .unwrap();
        let attrs =
            crate::attribute::extract_attributes_full(&bytes, &oh, sb.offset_size, sb.length_size)
                .unwrap();
        assert_eq!(attrs.len(), 15);
    }

    #[test]
    fn inline_attrs_below_threshold() {
        let mut fw = FileWriter::new();
        let ds = fw.create_dataset("data");
        ds.with_f64_data(&[1.0]);
        for i in 0..5 {
            ds.set_attr(&format!("a{i}"), AttrValue::F64(i as f64));
        }
        let bytes = fw.finish().unwrap();
        let sig = signature::find_signature(&bytes).unwrap();
        let sb = Superblock::parse(&bytes, sig).unwrap();
        let addr = resolve_path_any(&bytes, &sb, "data").unwrap();
        let hdr =
            ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
        assert!(
            !hdr.messages
                .iter()
                .any(|m| m.msg_type == MessageType::AttributeInfo)
        );
        let attrs = crate::attribute::extract_attributes(&hdr, sb.length_size).unwrap();
        assert_eq!(attrs.len(), 5);
    }

    #[test]
    fn encode_decode_managed_id_roundtrip() {
        let id = encode_managed_id(100, 42, 40, 8);
        let fh = crate::fractal_heap::FractalHeapHeader {
            heap_id_length: 8,
            io_filter_encoded_length: 0,
            max_managed_object_size: 1024,
            btree_huge_objects_address: u64::MAX,
            table_width: 4,
            starting_block_size: 4096,
            max_direct_block_size: 65536,
            max_heap_size: 40,
            start_root_rows: 1,
            root_block_address: 0,
            current_rows_in_root_indirect_block: 0,
            managed_objects_count: 0,
        };
        let (off, len) = fh.decode_managed_id(&id).unwrap();
        assert_eq!(off, 100);
        assert_eq!(len, 42);
    }

    #[test]
    fn dense_attrs_fit_bounds_the_single_direct_block() {
        // A modest set fits the single direct block the emitter produces.
        let small: Vec<AttributeMessage> = (0..30)
            .map(|i| build_attr_message(&format!("a{i}"), &AttrValue::I64(i)))
            .collect();
        assert!(dense_attrs_fit(&small));

        // A set whose serialized attributes overflow the one direct block (here
        // via very long names) would need fractal-heap indirect blocks the
        // emitter does not build, so it must report as not-fitting.
        let big: Vec<AttributeMessage> = (0..40)
            .map(|i| {
                let name = format!("{}{i}", "n".repeat(3000));
                build_attr_message(&name, &AttrValue::I64(i))
            })
            .collect();
        let total: usize = big.iter().map(|a| a.serialize_v3(LENGTH_SIZE).len()).sum();
        assert!(
            total as u64 > DENSE_ATTR_MAX_DIRECT_BLOCK,
            "test set should exceed one direct block (got {total} bytes)",
        );
        assert!(!dense_attrs_fit(&big));
    }

    /// Read a dataset's VL-string byte objects from a freshly-written file.
    fn read_vl_bytes(bytes: Vec<u8>, path: &str) -> Vec<crate::vl_data::VlByteObject> {
        let file = crate::reader::File::from_bytes(bytes).unwrap();
        file.dataset(path)
            .unwrap()
            .read_vlen_string_bytes(crate::vl_data::VlenStringReadOptions::default())
            .unwrap()
    }

    #[test]
    fn vlen_string_dataset_roundtrips_values() {
        let mut fw = FileWriter::new();
        fw.create_dataset("labels")
            .with_vlen_strings(&["alpha", "beta", "gamma"]);
        let bytes = fw.finish().unwrap();
        let objs = read_vl_bytes(bytes, "labels");
        let got: Vec<_> = objs
            .iter()
            .map(|o| match o {
                crate::vl_data::VlByteObject::Bytes(b) => String::from_utf8(b.clone()).unwrap(),
                crate::vl_data::VlByteObject::Null => "<null>".to_string(),
            })
            .collect();
        assert_eq!(got, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn vlen_string_dataset_preserves_null_vs_empty() {
        use crate::type_builders::VlStringElement;
        use crate::vl_data::VlByteObject;

        let dt = crate::type_builders::make_vlen_string_type(CharacterSet::Utf8);
        let elements = vec![
            VlStringElement::Bytes(b"hi".to_vec()),
            VlStringElement::Null,
            VlStringElement::Bytes(Vec::new()), // empty string, not null
            VlStringElement::Bytes(b"end".to_vec()),
        ];
        let mut fw = FileWriter::new();
        fw.create_dataset("mixed")
            .with_vlen_string_elements(dt, &elements)
            .unwrap();
        let bytes = fw.finish().unwrap();
        let objs = read_vl_bytes(bytes, "mixed");
        assert_eq!(
            objs,
            vec![
                VlByteObject::Bytes(b"hi".to_vec()),
                VlByteObject::Null,
                VlByteObject::Bytes(Vec::new()),
                VlByteObject::Bytes(b"end".to_vec()),
            ]
        );
    }

    #[test]
    fn vlen_string_dataset_preserves_embedded_nul() {
        use crate::type_builders::VlStringElement;
        use crate::vl_data::VlByteObject;

        let dt = crate::type_builders::make_vlen_string_type(CharacterSet::Ascii);
        let payload = b"a\0b\0c".to_vec();
        let elements = vec![VlStringElement::Bytes(payload.clone())];
        let mut fw = FileWriter::new();
        fw.create_dataset("nul")
            .with_vlen_string_elements(dt, &elements)
            .unwrap();
        let bytes = fw.finish().unwrap();
        let objs = read_vl_bytes(bytes, "nul");
        assert_eq!(objs, vec![VlByteObject::Bytes(payload)]);
    }

    #[test]
    fn vlen_string_dataset_preserves_non_utf8_bytes() {
        // The byte-exact write/read path must round-trip a payload that is not
        // valid UTF-8 (the headline faithfulness claim for issue #83). A
        // String-based path would corrupt this via lossy decoding; the
        // VlStringElement::Bytes / read_vlen_string_bytes path must not.
        use crate::type_builders::VlStringElement;
        use crate::vl_data::VlByteObject;

        let dt = crate::type_builders::make_vlen_string_type(CharacterSet::Ascii);
        let payload = vec![0xffu8, 0xfe, 0x80, 0x00, 0x41];
        let elements = vec![VlStringElement::Bytes(payload.clone())];
        let mut fw = FileWriter::new();
        fw.create_dataset("raw")
            .with_vlen_string_elements(dt, &elements)
            .unwrap();
        let bytes = fw.finish().unwrap();
        let objs = read_vl_bytes(bytes, "raw");
        assert_eq!(objs, vec![VlByteObject::Bytes(payload)]);
    }

    #[test]
    fn vlen_string_dataset_2d_shape_roundtrips() {
        let mut fw = FileWriter::new();
        fw.create_dataset("grid")
            .with_vlen_strings(&["a", "bb", "ccc", "dddd"])
            .with_shape(&[2, 2]);
        let bytes = fw.finish().unwrap();
        let file = crate::reader::File::from_bytes(bytes).unwrap();
        let ds = file.dataset("grid").unwrap();
        assert_eq!(ds.shape().unwrap(), vec![2, 2]);
        assert_eq!(
            ds.read_vlen_strings(crate::vl_data::VlenStringReadOptions::default())
                .unwrap(),
            vec!["a", "bb", "ccc", "dddd"]
        );
    }

    #[test]
    fn vlen_string_dataset_all_null_no_heap() {
        use crate::type_builders::VlStringElement;
        use crate::vl_data::VlByteObject;

        let dt = crate::type_builders::make_vlen_string_type(CharacterSet::Utf8);
        let elements = vec![VlStringElement::Null, VlStringElement::Null];
        let mut fw = FileWriter::new();
        fw.create_dataset("nulls")
            .with_vlen_string_elements(dt, &elements)
            .unwrap();
        let bytes = fw.finish().unwrap();
        let objs = read_vl_bytes(bytes, "nulls");
        assert_eq!(objs, vec![VlByteObject::Null, VlByteObject::Null]);
    }

    #[test]
    fn chunked_vlen_string_dataset_refused() {
        let mut fw = FileWriter::new();
        fw.create_dataset("chunked")
            .with_vlen_strings(&["a", "b", "c", "d"])
            .with_chunks(&[2]);
        let err = fw.finish().unwrap_err();
        assert!(
            matches!(err, FormatError::ChunkedVlenStringUnsupported),
            "expected ChunkedVlenStringUnsupported, got {err:?}"
        );
    }

    #[test]
    fn vlen_sequence_dataset_roundtrips_i32() {
        // Non-string VL (`H5T_VLEN { i32 }`): the per-element reference stores an
        // element *count*, while the heap object holds count*4 bytes. The
        // writer/reader pair must agree on that, including an empty sequence.
        use crate::type_builders::VlStringElement;
        use crate::vl_data::{VlByteObject, VlenStringReadOptions};

        let dt = Datatype::VariableLength {
            is_string: false,
            padding: None,
            charset: None,
            base_type: Box::new(crate::type_builders::make_i32_type()),
        };
        let seqs: Vec<Vec<i32>> = vec![vec![1, 2, 3], vec![], vec![-7, 42]];
        let elements: Vec<VlStringElement> = seqs
            .iter()
            .map(|s| VlStringElement::Bytes(s.iter().flat_map(|v| v.to_le_bytes()).collect()))
            .collect();
        let mut fw = FileWriter::new();
        fw.create_dataset("seq")
            .with_vlen_sequence_elements(dt, &elements)
            .unwrap();
        let bytes = fw.finish().unwrap();

        let file = crate::reader::File::from_bytes(bytes).unwrap();
        let ds = file.dataset("seq").unwrap();
        assert!(
            matches!(
                ds.datatype().unwrap(),
                Datatype::VariableLength {
                    is_string: false,
                    ..
                }
            ),
            "datatype must stay a non-string variable-length sequence"
        );
        let (objs, elem_size) = ds
            .read_vlen_sequence_bytes(VlenStringReadOptions::default())
            .unwrap();
        assert_eq!(elem_size, 4);
        let got: Vec<Vec<i32>> = objs
            .iter()
            .map(|o| match o {
                VlByteObject::Null => Vec::new(),
                VlByteObject::Bytes(b) => b
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            })
            .collect();
        assert_eq!(got, seqs);
    }

    #[test]
    fn vlen_sequence_rejects_string_datatype() {
        // The sequence builder must refuse a string-shaped VL datatype, which
        // belongs to the VL-string path.
        use crate::type_builders::VlStringElement;
        let dt = crate::type_builders::make_vlen_string_type(CharacterSet::Utf8);
        let mut fw = FileWriter::new();
        let res = fw
            .create_dataset("x")
            .with_vlen_sequence_elements(dt, &[VlStringElement::Bytes(b"hi".to_vec())]);
        assert!(matches!(res, Err(FormatError::TypeMismatch { .. })));
    }
}
