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
use crate::error::{FormatError, OBJECT_HEADER_MESSAGE_MAX};
use crate::file_space_info::{
    DEFAULT_PAGE_SIZE, DEFAULT_THRESHOLD, FileSpaceInfo, FileSpaceStrategy, NUM_FILE_FSM_MANAGERS,
};
use crate::free_space_manager::{
    FreeSection, SECT_CLASS_LARGE, SECT_CLASS_SMALL, fshd_len, fsse_len, serialize_file_fsm,
};
use crate::libver::LibVer;
use crate::link_message::{LinkMessage, LinkTarget};
use crate::message_type::MessageType;
use crate::object_header_writer::ObjectHeaderWriter;
use crate::superblock::Superblock;
use crate::type_builders::{
    DatasetBuilder, FinishedGroup, GroupBuilder, VlStringStaging, build_attr_message,
    build_global_heap_collections, patch_vl_refs, patch_vl_refs_masked,
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

/// Round `value` up to the next multiple of `page` (a power of two). Used by the
/// paged file-space writer to page-align region starts and the end-of-allocation.
fn align_up(value: u64, page: u64) -> u64 {
    value.div_ceil(page) * page
}

// ---- OH builders ----

pub(crate) fn build_chunked_dataset_oh(
    dt: &Datatype,
    ds: &Dataspace,
    layout_message: &[u8],
    pipeline_message: Option<&[u8]>,
    attrs: &[AttributeMessage],
    dense_blob: Option<&DenseAttrBlob>,
    fill: Option<&[u8]>,
) -> Result<Vec<u8>, FormatError> {
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
) -> Result<Vec<u8>, FormatError> {
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
) -> Result<Vec<u8>, FormatError> {
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

/// Bits of heap offset the dense attribute heap declares (its "Maximum Heap
/// Size"), and the byte width that implies for a block offset.
const DENSE_ATTR_MAX_HEAP_SIZE_BITS: u16 = 40;
const DENSE_ATTR_BLOCK_OFFSET_BYTES: usize = (DENSE_ATTR_MAX_HEAP_SIZE_BITS as usize).div_ceil(8);

/// Direct-block header bytes ahead of the data area, mirroring what
/// [`build_dense_attrs`] emits: signature(4) + version(1) + heap address +
/// block offset + checksum(4).
const DENSE_ATTR_DBLOCK_HEADER: usize =
    4 + 1 + OFFSET_SIZE as usize + DENSE_ATTR_BLOCK_OFFSET_BYTES + 4;

/// The maximum direct block size the heap declares when its own root block is
/// no larger, matching what the reference C library writes for an attribute
/// heap. A heap whose root block is bigger declares that larger size instead,
/// so the header never claims a maximum its own block exceeds.
///
/// A byte size rather than an on-disk address, so it is a `usize`: it is compared
/// and subtracted against in-memory buffer sizes, and only widened to the 8-byte
/// on-disk length field at the point it is written.
const DENSE_ATTR_DEFAULT_MAX_DIRECT_BLOCK: usize = 65536;

/// The largest attribute [`build_dense_attrs`] can store as a managed object.
///
/// The externally imposed limit is the heap ID: the 8-byte managed IDs this
/// emitter writes spend one byte on flags and [`DENSE_ATTR_BLOCK_OFFSET_BYTES`]
/// on the offset, leaving 2 bytes for the object's length, so no length above
/// 65,535 is representable at all. This constant is the slightly tighter value
/// the heap header also declares as its maximum managed object size, which the
/// reference C library then enforces on read: it rejects the oversized object as
/// one that should have been standalone, and an assertion-enabled build goes on
/// to abort while releasing the half-built attribute table.
///
/// An object past this belongs in fractal-heap *huge* storage, which this
/// emitter does not write, so it must be refused rather than stored as managed
/// (see [`dense_attrs_check`]).
pub(crate) const DENSE_ATTR_MAX_MANAGED_OBJECT: usize =
    DENSE_ATTR_DEFAULT_MAX_DIRECT_BLOCK - DENSE_ATTR_DBLOCK_HEADER;

/// One B-tree v2 record as [`build_dense_attrs`] writes it: heap ID(8) +
/// message flags(1) + creation order(4) + name hash(4).
const DENSE_ATTR_BTREE_RECORD: usize = 8 + 1 + 4 + 4;

/// A B-tree v2 leaf node's fixed bytes around its records: signature(4) +
/// version(1) + type(1) + checksum(4). The reference C library subtracts the
/// same 10 when deriving a node's record capacity.
const DENSE_ATTR_BTLF_OVERHEAD: usize = 4 + 1 + 1 + 4;

/// The leaf node size [`build_dense_attrs`] declares for `count` records.
///
/// Shared with [`dense_attrs_check`] so the bound is computed from the node size
/// actually written. The power-of-two rounding is what makes the bound
/// non-obvious — see [`DENSE_ATTR_MAX_COUNT`].
fn dense_attr_leaf_node_size(count: usize) -> usize {
    (DENSE_ATTR_BTLF_OVERHEAD + count * DENSE_ATTR_BTREE_RECORD)
        .next_power_of_two()
        .max(512)
}

/// The largest leaf node whose implied record capacity the reference C library
/// can still describe in the 2 bytes it allots: `H5B2__hdr_init` derives
/// `max_nrec_size` from the node's *capacity*, and asserts it fits 2 bytes.
const DENSE_ATTR_MAX_LEAF_NODE: usize = {
    let ceiling = DENSE_ATTR_BTLF_OVERHEAD + (u16::MAX as usize) * DENSE_ATTR_BTREE_RECORD;
    // Round *down* to a power of two: the emitter only ever declares one of those.
    1usize << (usize::BITS - 1 - ceiling.leading_zeros())
};

/// The most attributes dense storage can index — 61,680, not the 65,535 the
/// leaf's 2-byte record-count field would suggest.
///
/// The binding constraint is one step removed from that field. The reference C
/// library derives the byte width it needs for a record count from the leaf's
/// *capacity*, and capacity follows the node size this emitter declares — which
/// is rounded up to a power of two. Once that rounded node passes
/// [`DENSE_ATTR_MAX_LEAF_NODE`] the implied capacity needs 3 bytes and an
/// assertion-enabled build aborts in `H5B2__hdr_init`, even though the count
/// itself still fits the 2-byte field. Deriving the limit from
/// [`dense_attr_leaf_node_size`] keeps it correct if the record size or the
/// rounding ever changes.
pub(crate) const DENSE_ATTR_MAX_COUNT: usize =
    (DENSE_ATTR_MAX_LEAF_NODE - DENSE_ATTR_BTLF_OVERHEAD) / DENSE_ATTR_BTREE_RECORD;

/// The largest direct block the reference C library will construct
/// (`H5HF_MAX_DIRECT_SIZE_LIMIT`). It reads the heap's block sizes through
/// 32-bit helpers that assert on a power of two, so a larger block is a heap it
/// would mis-read rather than reject.
const DENSE_ATTR_MAX_DIRECT_BLOCK_LIMIT: u64 = 2 * 1024 * 1024 * 1024;

/// The root direct block size [`build_dense_attrs`] will emit for `attrs`, and
/// the maximum direct block size the heap header should declare alongside it.
///
/// Single source of truth for that geometry so [`dense_attrs_check`] validates
/// exactly what [`build_dense_attrs`] emits, and so the declared maximum cannot
/// drift below the block actually written.
fn dense_attr_block_geometry(serialized_total: usize) -> (u64, u64) {
    // Rounded up in `u64`, not `usize`: on a 32-bit host the power-of-two
    // rounding of a large heap would otherwise overflow (and panic) before
    // `dense_attrs_check` got the chance to refuse it.
    let content = DENSE_ATTR_DBLOCK_HEADER as u64 + serialized_total as u64;
    let starting_block_size = content.next_power_of_two().max(512);
    let max_direct_block_size = starting_block_size.max(DENSE_ATTR_DEFAULT_MAX_DIRECT_BLOCK as u64);
    (starting_block_size, max_direct_block_size)
}

/// Whether [`build_dense_attrs`] can faithfully represent `attrs` in its
/// single-direct-block, single-leaf-B-tree layout.
///
/// The bounds are the ones the emitter actually has to honour: each attribute
/// must fit [`DENSE_ATTR_MAX_MANAGED_OBJECT`], the set must fit
/// [`DENSE_ATTR_MAX_COUNT`], and the root direct block must stay inside
/// [`DENSE_ATTR_MAX_DIRECT_BLOCK_LIMIT`]. Notably the *total* is not bounded at
/// 64 KiB — the emitter sizes its root direct block to the content, and
/// multi-megabyte heaps of individually small attributes are written and read
/// back correctly, so refusing them would reject files that encode fine.
///
/// Callers that cannot fall back to a larger layout must refuse rather than
/// mis-encode (see [`build_dense_attrs`]).
pub(crate) fn dense_attrs_check(attrs: &[AttributeMessage]) -> Result<(), FormatError> {
    // Counted first so the running total below cannot overflow a 32-bit `usize`
    // before an absurd set is refused.
    if attrs.len() > DENSE_ATTR_MAX_COUNT {
        return Err(FormatError::TooManyDenseAttributes {
            count: attrs.len(),
            limit: DENSE_ATTR_MAX_COUNT,
        });
    }
    let mut total = 0usize;
    for a in attrs {
        let size = a.serialize_v3(LENGTH_SIZE).len();
        if size > DENSE_ATTR_MAX_MANAGED_OBJECT {
            return Err(FormatError::DenseAttributeTooLarge {
                name: a.name.clone(),
                size,
                limit: DENSE_ATTR_MAX_MANAGED_OBJECT,
            });
        }
        total += size;
    }
    dense_attrs_check_geometry(total)
}

/// Bound the heap geometry that `total` bytes of serialized attributes imply.
///
/// Split out from [`dense_attrs_check`] so the block-size limit can be tested
/// without materializing gigabytes of attributes to reach it.
fn dense_attrs_check_geometry(total: usize) -> Result<(), FormatError> {
    let (_, max_direct_block_size) = dense_attr_block_geometry(total);
    if max_direct_block_size > DENSE_ATTR_MAX_DIRECT_BLOCK_LIMIT {
        return Err(FormatError::DenseAttributeHeapTooLarge {
            block_size: max_direct_block_size,
            limit: DENSE_ATTR_MAX_DIRECT_BLOCK_LIMIT,
        });
    }
    Ok(())
}

/// Build dense attribute storage for a set of attributes.
///
/// The caller must have checked [`dense_attrs_check`] first: this emitter builds
/// a single direct block and a single-leaf B-tree, and stores every attribute as
/// a managed object, so an attribute set outside those bounds would be
/// mis-encoded.
pub(crate) fn build_dense_attrs(attrs: &[AttributeMessage], base_address: u64) -> DenseAttrBlob {
    // Dense attrs use v3 attribute messages (adds character set encoding byte).
    let serialized: Vec<Vec<u8>> = attrs.iter().map(|a| a.serialize_v3(LENGTH_SIZE)).collect();

    let name_hashes: Vec<u32> = attrs
        .iter()
        .map(|a| crate::checksum::jenkins_lookup3(a.name.as_bytes()))
        .collect();

    let os = OFFSET_SIZE as usize;
    let ls = LENGTH_SIZE as usize;
    let max_heap_size: u16 = DENSE_ATTR_MAX_HEAP_SIZE_BITS;
    let block_offset_bytes = DENSE_ATTR_BLOCK_OFFSET_BYTES; // 5
    let heap_id_length: u16 = 8;

    // Direct block layout: sig(4) + ver(1) + heap_addr(os) + block_offset(bo_bytes)
    //   + checksum(4) [when flags bit 1 set] + data...
    let dblock_header_size = DENSE_ATTR_DBLOCK_HEADER;
    let total_data_size: usize = serialized.iter().map(|s| s.len()).sum();
    // Both sizes come from the shared geometry, so the maximum this header
    // declares always covers the block it goes on to emit.
    let (starting_block_size, max_direct_block_size) = dense_attr_block_geometry(total_data_size);

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
        reason = "dense_attrs_check, which every caller must run first, bounds this \
                  direct-block size at 2 GiB, so it fits usize on every supported target"
    )]
    let data_space = starting_block_size as usize - dblock_header_size;
    let free_space = data_space - total_data_size;

    // The reference C library does not read a heap ID's length field at a fixed
    // width: it derives that width from the heap's declared maximum managed
    // object size, then decodes `1 + offset_bytes + length_bytes` from the ID. If
    // that total ever exceeded `heap_id_length` it would read past the ID stored
    // in each B-tree record. Keeping the declared maximum pinned to
    // DENSE_ATTR_MAX_MANAGED_OBJECT is what holds the two in agreement, so assert
    // it here rather than leaving it to the constant's doc comment.
    debug_assert_eq!(
        1 + block_offset_bytes + encoded_size_width(DENSE_ATTR_MAX_MANAGED_OBJECT as u64),
        heap_id_length as usize,
        "managed heap ID width must match what the declared maximum managed object size implies"
    );

    // Build fractal heap header
    let mut frhp = Vec::with_capacity(frhp_size);
    frhp.extend_from_slice(b"FRHP");
    frhp.push(0); // version
    frhp.extend_from_slice(&heap_id_length.to_le_bytes());
    frhp.extend_from_slice(&0u16.to_le_bytes()); // io_filter_encoded_length
    frhp.push(0x02); // flags: bit 1 = checksum direct blocks
    // Deliberately a constant rather than a function of `max_direct_block_size`:
    // this is the per-object cap the 2-byte length field of an 8-byte managed
    // heap ID can encode, so it must not grow with the block. `dense_attrs_check`
    // bounds every attribute by the same constant.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "DENSE_ATTR_MAX_MANAGED_OBJECT is 65,514, well inside the 4-byte \
                  max-managed-object-size field"
    )]
    let max_managed = DENSE_ATTR_MAX_MANAGED_OBJECT as u32;
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
    debug_assert_eq!(record_size as usize, DENSE_ATTR_BTREE_RECORD);
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
    let btlf_size = DENSE_ATTR_BTLF_OVERHEAD + num_records * record_size as usize;
    // Shared with `dense_attrs_check`, which bounds the record count by the
    // largest node size the reference C library can describe.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "dense_attrs_check bounds the record count so this node size stays at or \
                  below DENSE_ATTR_MAX_LEAF_NODE (2^20), well inside the 4-byte field"
    )]
    let node_size = dense_attr_leaf_node_size(num_records) as u32;
    debug_assert!(node_size as usize >= btlf_size);

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

/// Bytes the reference C library uses to encode a limit of `value`
/// (`H5VM_limit_enc_size`): the width of the smallest field that can hold it.
fn encoded_size_width(value: u64) -> usize {
    (64 - value.leading_zeros() as usize).div_ceil(8).max(1)
}

fn encode_managed_id(offset: u64, length: u64, max_heap_size: u16, id_length: u16) -> Vec<u8> {
    // `length << max_heap_size` must not overflow, and the offset must not run
    // into the length's bits. Both hold for every set `dense_attrs_check` admits;
    // asserted so a change to either constant cannot silently break the packing.
    debug_assert!(length <= DENSE_ATTR_MAX_MANAGED_OBJECT as u64);
    debug_assert_eq!(
        offset >> max_heap_size,
        0,
        "heap offset overflows its field"
    );
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
    fn file_space_extension_oh(&self) -> Result<Option<Vec<u8>>, FormatError> {
        self.file_space_info()
            .map(|info| {
                let mut oh = ObjectHeaderWriter::new();
                // Message flags 0x14 match what the reference C library writes for
                // this message (do-not-share + mark-if-unknown); no must-understand
                // bit, so older readers still open the file.
                oh.add_message_with_flags(MessageType::FileSpaceInfo, info.serialize(), 0x14);
                oh.serialize()
            })
            .transpose()
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

        // Genuine paged allocation: page-align every allocation and, when
        // persisting, emit per-page-type free-space managers. Gated entirely on
        // the Page strategy so every other strategy keeps its exact byte layout.
        let (paged, persist_paged, page_size, fs_threshold) = match self.file_space_strategy {
            Some((FileSpaceStrategy::Page, persist, threshold)) => {
                let ps = self.file_space_page_size.unwrap_or(DEFAULT_PAGE_SIZE);
                if ps < 512 || !ps.is_power_of_two() {
                    return Err(FormatError::InvalidFileSpacePageSize(ps));
                }
                // File-space pages are measured from the file base; the layout
                // below is base-relative, so base-relative boundaries coincide
                // with absolute ones only when the userblock is a whole number of
                // pages (zero trivially qualifies).
                if self.userblock_size % ps != 0 {
                    return Err(FormatError::InvalidFileSpacePageSize(ps));
                }
                (true, persist, ps, threshold)
            }
            _ => (false, false, 0, DEFAULT_THRESHOLD),
        };

        // The superblock-extension header (carrying a File Space Info message)
        // is independent of the file layout, so build it up front and place it
        // after all other content below.
        let ext_oh = self.file_space_extension_oh()?;
        // A persisting *non-paged* file's placeholder File Space Info message (built
        // just above) records `eoa_pre_fsm` = UNDEF, because a fresh file has no
        // free-space-manager blocks. libhdf5 requires `fs_persist => eoa_fsm_fsalloc
        // != UNDEF`, and an assertion-enabled build aborts on the sentinel
        // (H5Fsuper.c), so the non-paged tail below rewrites the message with a real
        // end-of-allocation once the layout is known (issue #178). Capture the
        // parameters here, while `self` is intact. (The paged path has its own
        // manager-aware rewrite; a `Page` file never reaches the non-paged tail.)
        let nonpaged_persist: Option<(FileSpaceStrategy, u64, u64)> = match self.file_space_strategy
        {
            Some((strategy, true, threshold)) if strategy != FileSpaceStrategy::Page => Some((
                strategy,
                threshold,
                self.file_space_page_size.unwrap_or(DEFAULT_PAGE_SIZE),
            )),
            _ => None,
        };
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
            /// Staged global heap collections + patch mask for a VL-string
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

        /// Emit one dataset's data region: in-memory bytes directly, or a
        /// streamed verbatim plan pulled from its raw-chunk provider one chunk at
        /// a time (so a streamed dataset's bytes never all reside in memory).
        fn emit_ds_data<Sk: ByteSink>(
            sink: &mut Sk,
            data: &DsData,
            raw_chunks: Option<&crate::type_builders::RawChunkPayload>,
        ) -> Result<(), FormatError> {
            match data {
                DsData::InMemory(bytes) => sink.put(bytes),
                DsData::Streamed(plan) => {
                    let provider = raw_chunks
                        .expect("a streamed data region implies a raw-chunk payload")
                        .provider
                        .0
                        .as_ref();
                    emit_chunked_data_verbatim(sink, plan, provider)
                }
            }
        }

        /// Emit one variable-length dataset's or attribute's global heap
        /// collections back to back, in the order `place_collections` assigned
        /// their addresses.
        fn emit_collections<Sk: ByteSink>(
            sink: &mut Sk,
            collections: &[Vec<u8>],
        ) -> Result<(), FormatError> {
            for collection in collections {
                sink.put(collection)?;
            }
            Ok(())
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
            // Variable-length string element references live in the global heap.
            // For chunked/filtered/resizable storage the references sit inside
            // chunks that are split (and possibly compressed) before the rest of
            // the file is laid out, so their heap addresses cannot be patched in
            // afterwards. Such a dataset instead has its collections placed
            // *ahead* of everything else, at a fixed address known before any
            // chunk is encoded — see the early-placement block in
            // `finish_to_sink` that fills `early_gcol`. Nothing is refused here.
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
            /// The attribute's collections, in the order their objects appear.
            collections: Vec<Vec<u8>>,
            attr_index: usize, // index into the relevant attrs Vec
        }

        /// Assign consecutive addresses to `collections` starting at `*cursor`,
        /// advancing it past them, and return those addresses in order. The
        /// GCOL emission loops below walk the collections in this same order,
        /// so the addresses patched into the references are the ones the
        /// collections land at.
        fn place_collections(collections: &[Vec<u8>], cursor: &mut u64) -> Vec<u64> {
            collections
                .iter()
                .map(|c| {
                    let addr = *cursor;
                    *cursor += c.len() as u64;
                    addr
                })
                .collect()
        }

        fn collect_vl_patches(attrs_raw: &[(String, AttrValue)]) -> Vec<VlPatch> {
            let mut patches = Vec::new();
            for (i, (_n, v)) in attrs_raw.iter().enumerate() {
                if let AttrValue::VarLenAsciiArray(strings) = v {
                    let str_refs: Vec<&str> = strings.iter().map(|s| s.as_str()).collect();
                    patches.push(VlPatch {
                        collections: build_global_heap_collections(&str_refs),
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

        let root_dense = root_attrs.len() > DENSE_ATTR_THRESHOLD;
        let group_dense: Vec<bool> = groups
            .iter()
            .map(|g| g.attrs.len() > DENSE_ATTR_THRESHOLD)
            .collect();
        let ds_dense: Vec<bool> = all_ds
            .iter()
            .map(|d| d.attrs.len() > DENSE_ATTR_THRESHOLD)
            .collect();

        // A compact attribute is stored as an object-header message, whose size
        // field is 2 bytes wide. An oversized one would be written with a
        // truncated length, which silently loses the attribute or leaves the
        // reader parsing the next message body as a message header, so refuse it
        // here — while the attribute's name is still at hand — before any header
        // is built. `ObjectHeaderWriter::serialize` is the unnamed backstop for
        // every other message this writer emits.
        //
        // This runs before the compression pass below so a file that will be
        // refused does not first pay to compress every chunked dataset in it.
        //
        // Dense attributes are stored in a fractal heap rather than the header, so
        // this limit does not apply to them and the check skips them; the dense
        // check just below bounds those instead.
        fn check_compact_attrs(attrs: &[AttributeMessage]) -> Result<(), FormatError> {
            for a in attrs {
                let size = a.serialize(LENGTH_SIZE).len();
                if size > OBJECT_HEADER_MESSAGE_MAX {
                    return Err(FormatError::AttributeMessageTooLarge {
                        name: a.name.clone(),
                        size,
                    });
                }
            }
            Ok(())
        }
        // The dense emitter has bounds of its own — a per-attribute size and a
        // B-tree record count — which it documents its callers must check. An
        // attribute set past them was previously written anyway, producing a heap
        // that reads back empty here and aborts an assertion-enabled reference C
        // library (issue #191). Between this and the compact check above, every
        // attribute is bounded on whichever path it takes.
        if root_dense {
            dense_attrs_check(&root_attrs)?;
        } else {
            check_compact_attrs(&root_attrs)?;
        }
        for (gi, g) in groups.iter().enumerate() {
            if group_dense[gi] {
                dense_attrs_check(&g.attrs)?;
            } else {
                check_compact_attrs(&g.attrs)?;
            }
        }
        for (i, d) in all_ds.iter().enumerate() {
            if ds_dense[i] {
                dense_attrs_check(&d.attrs)?;
            } else {
                check_compact_attrs(&d.attrs)?;
            }
        }

        let is_chunked: Vec<bool> = all_ds
            .iter()
            .map(|d| d.chunk_options.is_chunked() || d.maxshape.is_some() || d.raw_chunks.is_some())
            .collect();

        // A chunked/filtered/resizable variable-length dataset's element
        // references are split into chunks (and compressed) below, before the
        // file layout that would normally fix its global-heap addresses. Placing
        // those collections *first* — immediately after the superblock, at an
        // address that depends on nothing but the userblock — makes the addresses
        // known up front, so the references can be patched into `raw` before a
        // single chunk is encoded. Everything else shifts down by the collections'
        // total size, which is known here because staging serialized them
        // already.
        //
        // Only a chunked dataset takes this path. A contiguous one keeps the
        // established late placement (its element bytes stay patchable in place
        // until emission), so a file without a chunked VL dataset is laid out
        // byte-for-byte as before.
        let mut early_gcol: Vec<usize> = Vec::new();
        let early_gcol_size = {
            let mut cursor = SUPERBLOCK_SIZE as u64;
            for i in 0..all_ds.len() {
                // `raw_chunks` is excluded for the same reason the `chunk_sets`
                // loop below excludes it: a verbatim chunk payload is emitted
                // as-is and its `raw` is empty, so there is nothing to patch.
                if !is_chunked[i] || all_ds[i].raw_chunks.is_some() {
                    continue;
                }
                let Some(staging) = all_ds[i].vl_string_staging.take() else {
                    continue;
                };
                let addrs = place_collections(&staging.collections, &mut cursor);
                patch_vl_refs_masked(&mut all_ds[i].raw, &staging.patch_mask, &addrs);
                all_ds[i].vl_string_staging = Some(staging);
                early_gcol.push(i);
            }
            (cursor - SUPERBLOCK_SIZE as u64).to_usize()?
        };

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
                let oh = if group_dense[gi] {
                    let dummy_blob = build_dense_attrs(&g.attrs, 0);
                    build_group_oh(&dummy_links, &g.attrs, Some(&dummy_blob))?
                } else {
                    build_group_oh(&dummy_links, &g.attrs, None)?
                };
                Ok(oh.len())
            })
            .collect::<Result<_, FormatError>>()?;

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
            build_group_oh(&root_dummy_links, &root_attrs, Some(&dummy_blob))?.len()
        } else {
            build_group_oh(&root_dummy_links, &root_attrs, None)?.len()
        };

        // Pass 1: compute dataset object-header sizes from a dummy layout. No
        // data bytes are materialized here — the object-header size depends only
        // on the layout/pipeline messages, and a chunk index's byte size is a
        // function of chunk count/size, not of the (dummy) base address. For a
        // streamed (lazy) dataset this touches no chunk bytes at all.
        let mut actual_ds_oh_sizes: Vec<usize> = Vec::with_capacity(all_ds.len());
        // Each dataset's data-region byte length, captured here (where chunked
        // data is already built once for OH sizing) so the paged layout can
        // classify small vs large allocations and size its free-space managers
        // without a second build. A chunked data length is base-address
        // independent, so the dummy-base build gives the true length.
        let mut ds_data_lens: Vec<u64> = Vec::with_capacity(all_ds.len());
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
                ds_data_lens.push(built.data.len());
                build_chunked_dataset_oh(
                    &d.dt,
                    &d.ds,
                    &built.layout_message,
                    built.pipeline_message.as_deref(),
                    &d.attrs,
                    dense_blob.as_ref(),
                    d.fill.as_deref(),
                )?
            } else {
                ds_data_lens.push(d.raw.len() as u64);
                build_dataset_oh(
                    &d.dt,
                    &d.ds,
                    0,
                    d.raw.len() as u64,
                    &d.attrs,
                    dense_blob.as_ref(),
                    d.fill.as_deref(),
                )?
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
        // Early-placed VL collections (if any) occupy the span immediately after
        // the superblock, so the root object header — and everything after it —
        // starts past them. `early_gcol_size` is 0 for a file without a chunked
        // VL dataset, leaving every other file's addresses unchanged.
        let root_group_addr = (SUPERBLOCK_SIZE + early_gcol_size) as u64;
        let mut cursor2 = SUPERBLOCK_SIZE + early_gcol_size + root_oh_size;

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

        // ---- Paged file-space layout + emission ----
        // Lay the metadata (object headers, dense blobs, global heaps, the
        // superblock extension, and the free-space-manager blocks) into a
        // page-0+ metadata region, then start the raw data on a fresh page
        // boundary. Small (< page) raw data packs into its own page run; each
        // large (>= page) block gets its own page-aligned run. Every region's
        // page tail is tracked in a per-page-type free-space manager (SUPER for
        // metadata, DRAW for small raw, generic-large for large fragments) when
        // persisting. Emission is address-driven: gaps are zero-filled so the
        // physical file reaches the page-aligned end-of-allocation.
        if paged {
            let os = OFFSET_SIZE;
            let base = ub as u64;
            let mut meta = cursor2 as u64; // metadata cursor, base-relative

            // (a) Global-heap collections live in the metadata region. Assign
            // their addresses and patch attribute VL references now; dataset
            // element references are patched after their data is built below.
            let gcol_start = meta;
            let mut gcol_cursor = meta;
            let mut elem_gcol: Vec<(usize, Vec<u64>)> = Vec::new();
            {
                for patch in &vl_root {
                    let addrs = place_collections(&patch.collections, &mut gcol_cursor);
                    patch_vl_refs(&mut root_attrs[patch.attr_index].raw_data, &addrs);
                }
                for (gi, patches) in grp_vl.iter().enumerate() {
                    for patch in patches {
                        let addrs = place_collections(&patch.collections, &mut gcol_cursor);
                        patch_vl_refs(&mut groups[gi].attrs[patch.attr_index].raw_data, &addrs);
                    }
                }
                for (di, patches) in ds_vl.iter().enumerate() {
                    for patch in patches {
                        let addrs = place_collections(&patch.collections, &mut gcol_cursor);
                        patch_vl_refs(&mut all_ds[di].attrs[patch.attr_index].raw_data, &addrs);
                    }
                }
                for (i, d) in all_ds.iter().enumerate() {
                    // A chunked VL dataset's collections were placed and patched
                    // before its chunks were encoded; only the contiguous ones
                    // are still awaiting an address here.
                    if early_gcol.contains(&i) {
                        continue;
                    }
                    if let Some(staging) = &d.vl_string_staging {
                        elem_gcol
                            .push((i, place_collections(&staging.collections, &mut gcol_cursor)));
                    }
                }
            }
            let gcol_total_size = gcol_cursor - gcol_start;
            meta += gcol_total_size;

            // (b) Superblock extension (File Space Info) in the metadata region.
            let ext_addr = meta;
            let ext_len = ext_oh.as_ref().map_or(0, |b| b.len()) as u64;
            meta += ext_len;
            let meta_content_end = meta;

            // (c) Classify each dataset's data region by length. Contiguous empty
            // data is unallocated (undefined address); a chunked region always
            // has index bytes, so it is never "empty".
            let mut empty_indices: Vec<usize> = Vec::new();
            let mut small_indices: Vec<usize> = Vec::new();
            let mut large_indices: Vec<usize> = Vec::new();
            for i in 0..all_ds.len() {
                let len = ds_data_lens[i];
                if !is_chunked[i] && len == 0 {
                    empty_indices.push(i);
                } else if len < page_size {
                    small_indices.push(i);
                } else {
                    large_indices.push(i);
                }
            }
            let small_raw_total: u64 = small_indices.iter().map(|&i| ds_data_lens[i]).sum();
            let large_frag_sizes: Vec<u64> = large_indices
                .iter()
                .map(|&i| align_up(ds_data_lens[i], page_size) - ds_data_lens[i])
                .filter(|&f| f > 0)
                .collect();

            // (d) Which per-page-type managers are active, and place their
            // FSHD/FSSE blocks in the metadata region (persisting only). Block
            // lengths depend only on section counts (fixed field widths), so the
            // page tails are computed in a single forward pass with no iteration.
            let draw_active =
                small_raw_total > 0 && align_up(small_raw_total, page_size) != small_raw_total;
            let large_active = !large_frag_sizes.is_empty();
            let mut slots = [u64::MAX; NUM_FILE_FSM_MANAGERS];
            let mut super_fsm: Option<(u64, u64)> = None;
            let mut draw_fsm: Option<(u64, u64)> = None;
            let mut large_fsm: Option<(u64, u64)> = None;
            let super_block_len = fshd_len(os) + fsse_len(&[0], os);
            let draw_block_len = if draw_active {
                fshd_len(os) + fsse_len(&[0], os)
            } else {
                0
            };
            let large_block_len = if large_active {
                fshd_len(os) + fsse_len(&large_frag_sizes, os)
            } else {
                0
            };
            // SUPER tracks the metadata page tail. Placing its own block shifts
            // that tail, so only keep SUPER when a tail actually remains; in the
            // (astronomically rare) exact-fill case, drop it and leave the tiny
            // tail untracked. This decision is O(1), not a fixpoint.
            let super_active = if persist_paged {
                let with = meta_content_end + super_block_len + draw_block_len + large_block_len;
                align_up(with, page_size) > with
            } else {
                false
            };
            if persist_paged {
                if super_active {
                    let fshd_addr = meta;
                    meta += fshd_len(os);
                    let fsse_addr = meta;
                    meta += fsse_len(&[0], os);
                    slots[0] = fshd_addr;
                    super_fsm = Some((fshd_addr, fsse_addr));
                }
                if draw_active {
                    let fshd_addr = meta;
                    meta += fshd_len(os);
                    let fsse_addr = meta;
                    meta += fsse_len(&[0], os);
                    slots[2] = fshd_addr;
                    draw_fsm = Some((fshd_addr, fsse_addr));
                }
                if large_active {
                    let fshd_addr = meta;
                    meta += fshd_len(os);
                    let fsse_addr = meta;
                    meta += fsse_len(&large_frag_sizes, os);
                    slots[6] = fshd_addr;
                    large_fsm = Some((fshd_addr, fsse_addr));
                }
            }
            let meta_end = meta;

            // (e) The raw-data region starts on a fresh page boundary. The
            // metadata page tail is the SUPER section (when active).
            let raw_start = align_up(meta_end, page_size);
            let super_section = super_fsm.map(|_| FreeSection {
                addr: meta_end,
                size: raw_start - meta_end,
            });

            // (f) Build the raw data. Small blocks pack; the region is padded to
            // a page boundary (DRAW tail). Each large block starts a fresh page
            // run and its sub-page remainder becomes a generic-large section.
            let mut layouts: Vec<Option<DsLayout>> = (0..all_ds.len()).map(|_| None).collect();
            for &i in &empty_indices {
                let raw = core::mem::take(&mut all_ds[i].raw);
                layouts[i] = Some(DsLayout {
                    data: DsData::InMemory(raw),
                    data_addr: u64::MAX,
                    chunked_msgs: None,
                });
            }
            let mut c = raw_start;
            for &i in &small_indices {
                let base_addr = c;
                let layout = if is_chunked[i] {
                    let built = build_chunked(&all_ds[i], base_addr, chunk_sets[i].as_ref())?;
                    // The small/large classification and the free-space-manager
                    // sizing used the sizing-pass length (`ds_data_lens[i]`); the
                    // real build must match it, or the reserved manager space and
                    // the emitted layout would diverge (chunk-index byte length is
                    // base-address independent, so this always holds).
                    debug_assert_eq!(built.data.len(), ds_data_lens[i]);
                    c += built.data.len();
                    DsLayout {
                        data: built.data,
                        data_addr: base_addr,
                        chunked_msgs: Some((built.layout_message, built.pipeline_message)),
                    }
                } else {
                    let raw = core::mem::take(&mut all_ds[i].raw);
                    c += raw.len() as u64;
                    DsLayout {
                        data: DsData::InMemory(raw),
                        data_addr: base_addr,
                        chunked_msgs: None,
                    }
                };
                layouts[i] = Some(layout);
            }
            let small_raw_end = c;
            let draw_section = if draw_active {
                let padded = align_up(small_raw_end, page_size);
                c = padded;
                Some(FreeSection {
                    addr: small_raw_end,
                    size: padded - small_raw_end,
                })
            } else {
                if small_raw_total > 0 {
                    c = align_up(small_raw_end, page_size);
                }
                None
            };
            let mut large_sections: Vec<FreeSection> = Vec::new();
            for &i in &large_indices {
                c = align_up(c, page_size);
                let data_addr = c;
                let built_len;
                let layout = if is_chunked[i] {
                    let built = build_chunked(&all_ds[i], data_addr, chunk_sets[i].as_ref())?;
                    // See the small-run note: the real build length must equal the
                    // sizing-pass length the large classification/fragment used.
                    debug_assert_eq!(built.data.len(), ds_data_lens[i]);
                    built_len = built.data.len();
                    DsLayout {
                        data: built.data,
                        data_addr,
                        chunked_msgs: Some((built.layout_message, built.pipeline_message)),
                    }
                } else {
                    let raw = core::mem::take(&mut all_ds[i].raw);
                    built_len = raw.len() as u64;
                    DsLayout {
                        data: DsData::InMemory(raw),
                        data_addr,
                        chunked_msgs: None,
                    }
                };
                layouts[i] = Some(layout);
                let data_end = data_addr + built_len;
                let frag = align_up(data_end, page_size) - data_end;
                if frag > 0 {
                    large_sections.push(FreeSection {
                        addr: data_end,
                        size: frag,
                    });
                }
                c = align_up(data_end, page_size);
            }
            let eoa_rel = c; // already page-aligned
            let eof_addr2 = base + eoa_rel;
            let eoa_pre_fsm = eoa_rel;

            // (g) Now that the element bytes exist, patch dataset-element VL refs.
            for (i, gaddrs) in &elem_gcol {
                let staging = all_ds[*i]
                    .vl_string_staging
                    .as_ref()
                    .expect("elem_gcol only holds datasets with VL staging");
                let Some(DsLayout {
                    data: DsData::InMemory(bytes),
                    ..
                }) = layouts[*i].as_mut()
                else {
                    unreachable!(
                        "a staged VL-string dataset is non-chunked, so its data is in memory"
                    )
                };
                patch_vl_refs_masked(bytes, &staging.patch_mask, gaddrs);
            }

            let ds_layouts: Vec<DsLayout> = layouts
                .into_iter()
                .map(|o| o.expect("every dataset placed"))
                .collect();

            // (h) Build dataset OHs from the final data addresses.
            let mut ds_oh_bytes: Vec<Vec<u8>> = Vec::with_capacity(all_ds.len());
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
                    )?
                } else {
                    build_dataset_oh(
                        &d.dt,
                        &d.ds,
                        layout.data_addr,
                        layout.data.len(),
                        &d.attrs,
                        ds_dense_blobs[i].as_ref(),
                        d.fill.as_deref(),
                    )?
                };
                ds_oh_bytes.push(oh);
            }
            debug_assert_eq!(
                ds_oh_bytes.iter().map(|b| b.len()).collect::<Vec<_>>(),
                actual_ds_oh_sizes
            );

            // (i) Rebuild the real superblock-extension header. For persisting
            // files this replaces the placeholder (empty-manager) message with
            // the per-page-type manager addresses; its length is unchanged.
            let real_ext_oh = if persist_paged {
                let info = FileSpaceInfo::persistent_managers(
                    FileSpaceStrategy::Page,
                    fs_threshold,
                    page_size,
                    slots,
                    eoa_pre_fsm,
                );
                let mut oh = ObjectHeaderWriter::new();
                oh.add_message_with_flags(MessageType::FileSpaceInfo, info.serialize(), 0x14);
                oh.serialize()?
            } else {
                ext_oh
                    .clone()
                    .expect("a paged file always emits a File Space Info message")
            };
            debug_assert_eq!(real_ext_oh.len() as u64, ext_len);

            // (j) Serialize the free-space-manager blocks.
            let super_blocks = super_fsm.map(|(fshd_addr, fsse_addr)| {
                serialize_file_fsm(
                    &[super_section.expect("SUPER active implies a section")],
                    fshd_addr,
                    fsse_addr,
                    os,
                    SECT_CLASS_SMALL,
                )
            });
            let draw_blocks = draw_fsm.map(|(fshd_addr, fsse_addr)| {
                serialize_file_fsm(
                    &[draw_section.expect("DRAW active implies a section")],
                    fshd_addr,
                    fsse_addr,
                    os,
                    SECT_CLASS_SMALL,
                )
            });
            let large_blocks = large_fsm.map(|(fshd_addr, fsse_addr)| {
                serialize_file_fsm(&large_sections, fshd_addr, fsse_addr, os, SECT_CLASS_LARGE)
            });

            // (k) Emit, address-ascending, zero-filling every alignment gap.
            sink.reserve(eof_addr2.to_usize()?);
            if ub > 0 {
                sink.put_zeros(ub)?;
            }
            let sb = Superblock {
                version: 3,
                offset_size: OFFSET_SIZE,
                length_size: LENGTH_SIZE,
                base_address: base,
                eof_address: eof_addr2,
                root_group_address: root_group_addr,
                group_leaf_node_k: None,
                group_internal_node_k: None,
                indexed_storage_internal_node_k: None,
                free_space_address: None,
                driver_info_address: None,
                consistency_flags: 0,
                superblock_extension_address: Some(ext_addr),
                checksum: None,
            };
            sink.put(&sb.serialize())?;

            // Early-placed VL collections, at the addresses patched into the
            // chunked datasets' references before their chunks were encoded.
            for &i in &early_gcol {
                let staging = all_ds[i]
                    .vl_string_staging
                    .as_ref()
                    .expect("early_gcol only holds datasets with VL staging");
                emit_collections(sink, &staging.collections)?;
            }

            // Root group OH + dense blob.
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
            )?)?;
            if let Some(ref blob) = root_dense_blob {
                sink.put(&blob.blob)?;
            }
            // Group OHs + dense blobs.
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
                )?)?;
                if let Some(ref blob) = group_dense_blobs[gi] {
                    sink.put(&blob.blob)?;
                }
            }
            // Dataset OHs + dense blobs.
            for (i, oh) in ds_oh_bytes.iter().enumerate() {
                sink.put(oh)?;
                if let Some(ref dense) = ds_dense_blobs[i] {
                    sink.put(&dense.blob)?;
                }
            }
            // Global heap collections.
            for patch in &vl_root {
                emit_collections(sink, &patch.collections)?;
            }
            for patches in &grp_vl {
                for patch in patches {
                    emit_collections(sink, &patch.collections)?;
                }
            }
            for patches in &ds_vl {
                for patch in patches {
                    emit_collections(sink, &patch.collections)?;
                }
            }
            for (i, d) in all_ds.iter().enumerate() {
                if early_gcol.contains(&i) {
                    continue; // emitted right after the superblock
                }
                if let Some(staging) = &d.vl_string_staging {
                    emit_collections(sink, &staging.collections)?;
                }
            }
            debug_assert_eq!(sink.position(), base + ext_addr);
            sink.put(&real_ext_oh)?;
            // Free-space-manager blocks (SUPER, DRAW, generic-large), ascending.
            for blocks in [&super_blocks, &draw_blocks, &large_blocks]
                .into_iter()
                .flatten()
            {
                sink.put(&blocks.0)?;
                sink.put(&blocks.1)?;
            }
            debug_assert_eq!(sink.position(), base + meta_end);

            // Raw data region: metadata page tail, then small data, DRAW tail,
            // large runs (with their fragments), padded to the page-aligned EOA.
            sink.put_zeros((raw_start - meta_end).to_usize()?)?;
            for &i in &small_indices {
                debug_assert_eq!(sink.position(), base + ds_layouts[i].data_addr);
                emit_ds_data(sink, &ds_layouts[i].data, all_ds[i].raw_chunks.as_ref())?;
            }
            if small_raw_total > 0 {
                sink.put_zeros((align_up(small_raw_end, page_size) - small_raw_end).to_usize()?)?;
            }
            for &i in &large_indices {
                let data_addr = ds_layouts[i].data_addr;
                let gap = (base + data_addr) - sink.position();
                sink.put_zeros(gap.to_usize()?)?;
                emit_ds_data(sink, &ds_layouts[i].data, all_ds[i].raw_chunks.as_ref())?;
                let end_rel = sink.position() - base;
                sink.put_zeros((align_up(end_rel, page_size) - end_rel).to_usize()?)?;
            }
            let final_pad = eof_addr2 - sink.position();
            sink.put_zeros(final_pad.to_usize()?)?;
            debug_assert_eq!(sink.position(), eof_addr2);
            return Ok(());
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
                let addrs = place_collections(&patch.collections, &mut gcol_cursor);
                patch_vl_refs(&mut root_attrs[patch.attr_index].raw_data, &addrs);
            }
            for (gi, patches) in grp_vl.iter().enumerate() {
                for patch in patches {
                    let addrs = place_collections(&patch.collections, &mut gcol_cursor);
                    patch_vl_refs(&mut groups[gi].attrs[patch.attr_index].raw_data, &addrs);
                }
            }
            for (di, patches) in ds_vl.iter().enumerate() {
                for patch in patches {
                    let addrs = place_collections(&patch.collections, &mut gcol_cursor);
                    patch_vl_refs(&mut all_ds[di].attrs[patch.attr_index].raw_data, &addrs);
                }
            }
            // Dataset-element VL references. The references live in the
            // contiguous/compact element bytes (`ds_layouts[i].data`, cloned
            // from `d.raw`). A chunked dataset's references are *not* here: they
            // sit inside chunks that were encoded earlier, so the loop below must
            // skip it. Dropping that skip would patch heap addresses into the
            // encoded (possibly compressed) chunk blob and silently corrupt it.
            for (i, d) in all_ds.iter().enumerate() {
                // A chunked VL dataset was placed and patched before its chunks
                // were encoded; its references are already final.
                if early_gcol.contains(&i) {
                    continue;
                }
                if let Some(staging) = &d.vl_string_staging {
                    // Every dataset still awaiting an address here is contiguous
                    // or compact, so its element bytes are in memory and
                    // patchable in place. A streamed (lazy) dataset never carries
                    // VL staging, so this is unreachable for it — assert that
                    // rather than risk silently corrupting one.
                    let DsData::InMemory(ref mut bytes) = ds_layouts[i].data else {
                        unreachable!(
                            "a chunked VL-string dataset is patched before encoding, so a \
                             dataset patched here always has its data in memory"
                        );
                    };
                    let addrs = place_collections(&staging.collections, &mut gcol_cursor);
                    patch_vl_refs_masked(bytes, &staging.patch_mask, &addrs);
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
                )?
            } else {
                build_dataset_oh(
                    &d.dt,
                    &d.ds,
                    layout.data_addr,
                    layout.data.len(),
                    &d.attrs,
                    ds_dense_blobs[i].as_ref(),
                    d.fill.as_deref(),
                )?
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

        // Early-placed VL collections, at the addresses patched into the chunked
        // datasets' references before their chunks were encoded. This must walk
        // `early_gcol` in the order the placement loop built it, or the patched
        // addresses name the wrong collections.
        for &i in &early_gcol {
            let staging = all_ds[i]
                .vl_string_staging
                .as_ref()
                .expect("early_gcol only holds datasets with VL staging");
            emit_collections(sink, &staging.collections)?;
        }
        debug_assert_eq!(
            sink.position(),
            ub as u64 + root_group_addr,
            "early VL collections must occupy exactly the space reserved for them"
        );

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
        )?)?;
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
            )?)?;
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
            emit_ds_data(sink, &layout.data, all_ds[i].raw_chunks.as_ref())?;
        }

        // Global heap collections
        for patch in &vl_root {
            emit_collections(sink, &patch.collections)?;
        }
        for patches in &grp_vl {
            for patch in patches {
                emit_collections(sink, &patch.collections)?;
            }
        }
        for patches in &ds_vl {
            for patch in patches {
                emit_collections(sink, &patch.collections)?;
            }
        }
        // Dataset-element VL string collections, in the same order their
        // addresses were assigned above.
        for (i, d) in all_ds.iter().enumerate() {
            if early_gcol.contains(&i) {
                continue; // emitted right after the superblock
            }
            if let Some(staging) = &d.vl_string_staging {
                emit_collections(sink, &staging.collections)?;
            }
        }

        // Superblock extension (File Space Info), at the address recorded above.
        // For a persisting non-paged file, rebuild the message with a real
        // end-of-allocation — the end of all content, since no FSM blocks follow —
        // in place of the UNDEF placeholder (see the capture above; issue #178). The
        // message length is unchanged (only the `eoa_pre_fsm` field differs), so the
        // reserved layout still holds.
        let real_ext_oh = match (&ext_oh, nonpaged_persist) {
            (Some(_), Some((strategy, threshold, np_page_size))) => {
                let mut info = FileSpaceInfo::persistent_empty(strategy, threshold, np_page_size);
                info.eoa_pre_fsm = eof_addr2 - ub as u64;
                let mut oh = ObjectHeaderWriter::new();
                oh.add_message_with_flags(MessageType::FileSpaceInfo, info.serialize(), 0x14);
                Some(oh.serialize()?)
            }
            (other, _) => other.clone(),
        };
        debug_assert_eq!(
            real_ext_oh.as_ref().map_or(0, |b| b.len()),
            ext_len,
            "rebuilt extension header length must match the reserved length"
        );
        if let Some(bytes) = &real_ext_oh {
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

    /// An `AsciiString` attribute named `name` whose serialized (v3) size is
    /// exactly `size` bytes, so a bound can be tested on the value it bounds.
    fn dense_attr_of_size(name: &str, size: usize) -> AttributeMessage {
        let probe = build_attr_message(name, &AttrValue::AsciiString(String::new()));
        let overhead = probe.serialize_v3(LENGTH_SIZE).len();
        let attr = build_attr_message(name, &AttrValue::AsciiString("y".repeat(size - overhead)));
        assert_eq!(attr.serialize_v3(LENGTH_SIZE).len(), size);
        attr
    }

    #[test]
    fn dense_attrs_check_bounds_each_attribute_not_the_total() {
        // A multi-megabyte set of individually small attributes is exactly what
        // the emitter handles: it sizes its root direct block to the content, and
        // both this crate and the reference C library read such a heap back. The
        // old bound rejected these.
        let many: Vec<AttributeMessage> = (0..40)
            .map(|i| dense_attr_of_size(&format!("a{i}"), 60_000))
            .collect();
        let total: usize = many.iter().map(|a| a.serialize_v3(LENGTH_SIZE).len()).sum();
        assert!(total > 2_000_000, "expected a multi-megabyte set");
        assert_eq!(dense_attrs_check(&many), Ok(()));

        // One attribute at the managed-object limit is still fine.
        let at_limit = vec![dense_attr_of_size("edge", DENSE_ATTR_MAX_MANAGED_OBJECT)];
        assert_eq!(dense_attrs_check(&at_limit), Ok(()));

        // One byte past it needs fractal-heap huge storage, which the emitter
        // does not write.
        let past = vec![dense_attr_of_size(
            "edge",
            DENSE_ATTR_MAX_MANAGED_OBJECT + 1,
        )];
        assert_eq!(
            dense_attrs_check(&past),
            Err(FormatError::DenseAttributeTooLarge {
                name: "edge".to_string(),
                size: DENSE_ATTR_MAX_MANAGED_OBJECT + 1,
                limit: DENSE_ATTR_MAX_MANAGED_OBJECT,
            })
        );
    }

    /// The attribute-count limit exists because the reference C library derives a
    /// record-count width from the leaf's *capacity*, not from the count. Pin
    /// that derivation directly: at the limit the capacity still fits 2 bytes,
    /// and one attribute more pushes the rounded node size up a power of two and
    /// takes it to 3 — which is the abort this bound prevents.
    #[test]
    fn the_attribute_count_limit_is_where_the_capacity_width_grows() {
        let width_at = |count: usize| {
            let capacity = (dense_attr_leaf_node_size(count) - DENSE_ATTR_BTLF_OVERHEAD)
                / DENSE_ATTR_BTREE_RECORD;
            encoded_size_width(capacity as u64)
        };
        assert_eq!(width_at(DENSE_ATTR_MAX_COUNT), 2);
        assert_eq!(width_at(DENSE_ATTR_MAX_COUNT + 1), 3);
        // Guards the derivation against a silent change in the constants.
        assert_eq!(DENSE_ATTR_MAX_COUNT, 61_680);
    }

    /// Reaching the direct-block limit takes gigabytes of attributes, so the
    /// geometry check is exercised directly rather than by building them.
    #[test]
    fn dense_heap_past_the_direct_block_limit_is_refused() {
        assert_eq!(dense_attrs_check_geometry(1_000_000), Ok(()));

        let over = DENSE_ATTR_MAX_DIRECT_BLOCK_LIMIT as usize;
        match dense_attrs_check_geometry(over) {
            Err(FormatError::DenseAttributeHeapTooLarge { block_size, limit }) => {
                assert!(block_size > limit);
                assert_eq!(limit, DENSE_ATTR_MAX_DIRECT_BLOCK_LIMIT);
            }
            other => panic!("expected DenseAttributeHeapTooLarge, got {other:?}"),
        }
    }

    /// The header must never declare a maximum direct block size its own root
    /// block exceeds — the inconsistency the old fixed 65,536 produced for any
    /// heap larger than that.
    #[test]
    fn declared_max_direct_block_covers_the_emitted_block() {
        for total in [0usize, 100, 60_000, 65_600, 2_000_000] {
            let (starting, max_direct) = dense_attr_block_geometry(total);
            assert!(
                max_direct >= starting,
                "total {total}: declared max {max_direct} < emitted block {starting}"
            );
        }
        // Small heaps keep the value the reference C library writes, so existing
        // output is unchanged.
        assert_eq!(
            dense_attr_block_geometry(100).1,
            DENSE_ATTR_DEFAULT_MAX_DIRECT_BLOCK as u64
        );
    }

    /// The number of `i64` elements in the largest attribute whose serialized
    /// message still fits the object header's 2-byte message-size field, derived
    /// from a measured probe rather than hard-coded so it tracks the encoder.
    fn largest_fitting_i64_attr_elements() -> usize {
        let one = build_attr_message("boundary", &AttrValue::I64Array(vec![0i64; 1]));
        let overhead = one.serialize(LENGTH_SIZE).len() - 8;
        (OBJECT_HEADER_MESSAGE_MAX - overhead) / 8
    }

    #[test]
    fn compact_attr_at_the_message_size_limit_is_written() {
        let n = largest_fitting_i64_attr_elements();
        let attr = build_attr_message("boundary", &AttrValue::I64Array(vec![7i64; n]));
        // Pin the probe to the boundary itself: one more element must not fit,
        // or this test would still pass while exercising a tiny attribute.
        let size = attr.serialize(LENGTH_SIZE).len();
        assert!(size <= OBJECT_HEADER_MESSAGE_MAX);
        assert!(
            size + 8 > OBJECT_HEADER_MESSAGE_MAX,
            "probe is not at the limit (got {size})"
        );

        let mut fw = FileWriter::new();
        fw.set_root_attr("boundary", AttrValue::I64Array(vec![7i64; n]));
        fw.create_dataset("d").with_f64_data(&[1.0]);
        let bytes = fw.finish().unwrap();

        let file = crate::reader::File::from_bytes(bytes).unwrap();
        let attrs = file.root().attrs().unwrap();
        assert_eq!(attrs.len(), 1);
    }

    #[test]
    fn compact_attr_past_the_message_size_limit_is_refused() {
        let n = largest_fitting_i64_attr_elements() + 1;
        let size = build_attr_message("boundary", &AttrValue::I64Array(vec![0i64; n]))
            .serialize(LENGTH_SIZE)
            .len();
        assert!(size > OBJECT_HEADER_MESSAGE_MAX);

        let mut fw = FileWriter::new();
        fw.set_root_attr("boundary", AttrValue::I64Array(vec![7i64; n]));
        fw.create_dataset("d").with_f64_data(&[1.0]);
        assert_eq!(
            fw.finish(),
            Err(FormatError::AttributeMessageTooLarge {
                name: "boundary".to_string(),
                size,
            })
        );
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
    fn vlen_string_dataset_with_nulls_spans_multiple_heap_collections() {
        // A null element takes no heap object, so the collection an element's
        // reference resolves to follows its *object* position, not its element
        // position: interleaving nulls shifts the split point away from element
        // 65,535. Elements around both are checked, plus the nulls themselves.
        use crate::type_builders::VlStringElement;
        use crate::vl_data::VlByteObject;

        let count = 100_000;
        let elements: Vec<VlStringElement> = (0..count)
            .map(|i| {
                if i % 3 == 0 {
                    VlStringElement::Null
                } else {
                    VlStringElement::Bytes(format!("s{i}").into_bytes())
                }
            })
            .collect();
        let dt = crate::type_builders::make_vlen_string_type(CharacterSet::Utf8);
        let mut fw = FileWriter::new();
        fw.create_dataset("mixed")
            .with_vlen_string_elements(dt, &elements)
            .unwrap();
        let bytes = fw.finish().unwrap();
        let objs = read_vl_bytes(bytes, "mixed");

        assert_eq!(objs.len(), count);
        // Two thirds of the elements carry objects, so the 65,536th object —
        // the first of the second collection — is element 98,303, not 65,535.
        for i in [0, 1, 65_535, 98_302, 98_303, 98_304, count - 1] {
            let expected = if i % 3 == 0 {
                VlByteObject::Null
            } else {
                VlByteObject::Bytes(format!("s{i}").into_bytes())
            };
            assert_eq!(objs[i], expected, "element {i} did not round-trip");
        }
    }

    /// A chunked VL-string dataset has its heap collections placed ahead of the
    /// object headers, so the references inside its chunks carry real addresses
    /// (issue #109). Reading the elements back is the check that the addresses
    /// patched before chunk encoding are the ones the collections landed at.
    #[test]
    fn chunked_vlen_string_dataset_roundtrips() {
        let mut fw = FileWriter::new();
        fw.create_dataset("chunked")
            .with_vlen_strings(&["a", "bb", "ccc", "dddd"])
            .with_chunks(&[2]);
        let bytes = fw.finish().unwrap();
        let f = crate::reader::File::from_bytes(bytes).unwrap();
        let ds = f.dataset("chunked").unwrap();
        assert_eq!(ds.read_string().unwrap(), ["a", "bb", "ccc", "dddd"]);
    }

    /// The compressed case: the filter runs over element bytes whose heap
    /// addresses are already final, which is the whole reason the collections are
    /// placed first. A stale placeholder address would survive compression and
    /// read back as a dangling reference.
    #[test]
    #[cfg(feature = "deflate")]
    fn filtered_chunked_vlen_string_dataset_roundtrips() {
        let mut fw = FileWriter::new();
        fw.create_dataset("filtered")
            .with_vlen_strings(&["alpha", "beta", "gamma", "delta"])
            .with_chunks(&[2])
            .with_deflate(6);
        let bytes = fw.finish().unwrap();
        let f = crate::reader::File::from_bytes(bytes).unwrap();
        let ds = f.dataset("filtered").unwrap();
        assert_eq!(
            ds.read_string().unwrap(),
            ["alpha", "beta", "gamma", "delta"]
        );
    }

    /// A resizable (unlimited) VL-string dataset takes the same path — `maxshape`
    /// alone makes a dataset chunked.
    #[test]
    fn resizable_vlen_string_dataset_roundtrips() {
        let mut fw = FileWriter::new();
        fw.create_dataset("growable")
            .with_vlen_strings(&["one", "two", "three"])
            .with_shape(&[3])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[2]);
        let bytes = fw.finish().unwrap();
        let f = crate::reader::File::from_bytes(bytes).unwrap();
        let ds = f.dataset("growable").unwrap();
        assert_eq!(ds.read_string().unwrap(), ["one", "two", "three"]);
    }

    /// Null elements keep an undefined address through the early patch, exactly as
    /// they do on the contiguous path: the mask selects which references move.
    #[test]
    fn chunked_vlen_string_dataset_preserves_nulls() {
        use crate::type_builders::VlStringElement;
        use crate::vl_data::VlByteObject;

        let dt = crate::type_builders::make_vlen_string_type(CharacterSet::Utf8);
        let elements = vec![
            VlStringElement::Bytes(b"set".to_vec()),
            VlStringElement::Null,
            VlStringElement::Bytes(Vec::new()), // empty string, not null
            VlStringElement::Bytes(b"tail".to_vec()),
        ];
        let mut fw = FileWriter::new();
        fw.create_dataset("mixed")
            .with_vlen_string_elements(dt, &elements)
            .unwrap()
            .with_chunks(&[2]);
        let bytes = fw.finish().unwrap();
        assert_eq!(
            read_vl_bytes(bytes, "mixed"),
            vec![
                VlByteObject::Bytes(b"set".to_vec()),
                VlByteObject::Null,
                VlByteObject::Bytes(Vec::new()),
                VlByteObject::Bytes(b"tail".to_vec()),
            ]
        );
    }

    /// A file with only *contiguous* VL datasets must keep the established layout
    /// (collections after the data, nothing placed early), so adding the chunked
    /// capability cannot perturb the bytes of files that do not use it.
    #[test]
    fn contiguous_vlen_layout_is_unchanged_by_the_early_path() {
        let build = || {
            let mut fw = FileWriter::new();
            fw.create_dataset("plain")
                .with_vlen_strings(&["a", "bb", "ccc"]);
            fw.finish().unwrap()
        };
        // The root group object header sits immediately after the superblock when
        // nothing is placed early.
        let bytes = build();
        assert_eq!(
            &bytes[SUPERBLOCK_SIZE..SUPERBLOCK_SIZE + 4],
            b"OHDR",
            "a contiguous-only VL file must still open with the root OH at SUPERBLOCK_SIZE"
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
