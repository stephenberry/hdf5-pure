//! HDF5 file creation (write pipeline).
//!
//! Produces valid HDF5 files with v3 superblock, v2 object headers,
//! link messages, contiguous datasets, inline and dense attributes.

#[cfg(not(feature = "std"))]
use alloc::{string::String, string::ToString, vec, vec::Vec};

#[cfg(not(feature = "std"))]
use alloc::format;

#[cfg(feature = "std")]
use std::collections::HashMap;
#[cfg(not(feature = "std"))]
use alloc::collections::BTreeMap as HashMap;

use crate::attribute::AttributeMessage;
use crate::chunked_write::{ChunkOptions, build_chunked_data_at_ext};
use crate::filters::zfp_element_type_from_datatype;
use crate::dataspace::{Dataspace, DataspaceType};
use crate::error::FormatError;
use crate::link_message::{LinkMessage, LinkTarget};
use crate::message_type::MessageType;
use crate::metadata_index::{DatasetMetadata, MetadataBlock, MetadataIndex};
use crate::object_header_writer::ObjectHeaderWriter;
use crate::superblock::Superblock;
use crate::type_builders::{
    build_attr_message, build_global_heap_collection, patch_vl_refs,
    DatasetBuilder, FinishedGroup, GroupBuilder,
};

// Re-export public types that moved to type_builders for API compatibility.
pub use crate::type_builders::{AttrValue, CompoundTypeBuilder, EnumTypeBuilder};
#[cfg(feature = "provenance")]
pub use crate::type_builders::ProvenanceConfig;

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
) -> Vec<u8> {
    let mut w = ObjectHeaderWriter::new();
    w.add_message_with_flags(MessageType::Datatype, dt.serialize(), 0x01);
    w.add_message(MessageType::Dataspace, ds.serialize(LENGTH_SIZE));
    w.add_message_with_flags(MessageType::FillValue, vec![3, 0x0a], 0x01);
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
) -> Vec<u8> {
    let mut w = ObjectHeaderWriter::new();
    w.add_message_with_flags(MessageType::Datatype, dt.serialize(), 0x01);
    w.add_message(MessageType::Dataspace, ds.serialize(LENGTH_SIZE));
    w.add_message_with_flags(MessageType::FillValue, vec![3, 0x0a], 0x01);
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

/// Build dense attribute storage for a set of attributes.
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
    let frhp_size = 4 + 1 + 2 + 2 + 1 + 4
        + ls + os + ls + os + ls + ls + ls + ls + ls + ls + ls + ls
        + 2 + ls + ls + 2 + 2 + os + 2 + 4;

    let frhp_addr = base_address;
    let dblock_addr = frhp_addr + frhp_size as u64;
    let btree_addr = dblock_addr + starting_block_size;

    let data_space = starting_block_size as usize - dblock_header_size;
    let free_space = data_space - total_data_size;

    // Build fractal heap header
    let mut frhp = Vec::with_capacity(frhp_size);
    frhp.extend_from_slice(b"FRHP");
    frhp.push(0); // version
    frhp.extend_from_slice(&heap_id_length.to_le_bytes());
    frhp.extend_from_slice(&0u16.to_le_bytes()); // io_filter_encoded_length
    frhp.push(0x02); // flags: bit 1 = checksum direct blocks
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
    frhp.extend_from_slice(&sri.to_le_bytes()); // starting_row_of_indirect_blocks
    write_offset(&mut frhp, dblock_addr, OFFSET_SIZE);
    frhp.extend_from_slice(&0u16.to_le_bytes()); // root is direct block
    let frhp_checksum = crate::checksum::jenkins_lookup3(&frhp);
    frhp.extend_from_slice(&frhp_checksum.to_le_bytes());
    debug_assert_eq!(frhp.len(), frhp_size);

    // Build direct block: header (with checksum) + data + padding
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
    dblock.resize(starting_block_size as usize, 0);

    // Checksum: computed over entire block with checksum field zeroed
    let dblock_checksum = crate::checksum::jenkins_lookup3(&dblock);
    dblock[cksum_pos..cksum_pos + 4].copy_from_slice(&dblock_checksum.to_le_bytes());
    debug_assert_eq!(dblock.len(), starting_block_size as usize);

    // Build heap IDs
    let heap_ids: Vec<Vec<u8>> = attr_offsets
        .iter()
        .map(|(off, len)| encode_managed_id(*off, *len, max_heap_size, heap_id_length))
        .collect();

    // Build B-tree v2 type 8 records (17 bytes each)
    let record_size: u16 = heap_id_length + 1 + 4 + 4;
    let mut records: Vec<(u32, u32, Vec<u8>)> = Vec::with_capacity(attrs.len());
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

/// An opaque handle representing a dataset or group whose address will be
/// resolved during file serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObjectHandle(usize);

/// The main file creation API.
pub struct FileWriter {
    root_datasets: Vec<DatasetBuilder>,
    root_attrs: Vec<(String, AttrValue)>,
    groups: Vec<FinishedGroup>,
    userblock_size: u64,
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
        }
    }

    /// Set the userblock size in bytes. Must be a power of two >= 512 or 0 (no userblock).
    /// The userblock region will be filled with zeros; the caller can write into
    /// the returned bytes at `[0..userblock_size]`.
    pub fn with_userblock(&mut self, size: u64) -> &mut Self {
        self.userblock_size = size;
        self
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
        struct DsFlat {
            name: String,
            dt: Datatype,
            ds: Dataspace,
            raw: Vec<u8>,
            attrs: Vec<AttributeMessage>,
            chunk_options: ChunkOptions,
            maxshape: Option<Vec<u64>>,
            reference_targets: Option<Vec<String>>,
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
            // Allow empty data for zero-element datasets (e.g. shape [0, 0]).
            let is_empty = shape.iter().any(|&d| d == 0);
            let raw = if is_empty {
                db.data.unwrap_or_default()
            } else {
                db.data.ok_or(FormatError::DatasetMissingData)?
            };
            let max_dimensions = db.maxshape.clone();
            let dspace = Dataspace {
                space_type: if shape.is_empty() { DataspaceType::Scalar } else { DataspaceType::Simple },
                rank: shape.len() as u8, dimensions: shape, max_dimensions,
            };
            let patches = collect_vl_patches(&db.attrs);
            let mut attrs = Vec::new();
            for (n, v) in &db.attrs { attrs.push(build_attr_message(n, v)); }
            #[cfg(feature = "provenance")]
            if let Some(ref prov) = db.provenance {
                let p = crate::provenance::Provenance {
                    creator: prov.creator.clone(),
                    timestamp: prov.timestamp.clone(),
                    source: prov.source.clone(),
                };
                attrs.extend(p.build_attrs(&raw));
            }
            let idx = all_ds.len();
            all_ds.push(DsFlat { name: db.name, dt, ds: dspace, raw, attrs, chunk_options: db.chunk_options, maxshape: db.maxshape, reference_targets: db.reference_targets });
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
            for (n, v) in &g.attrs { gattrs.push(build_attr_message(n, v)); }
            let mut ds_idx = Vec::new();
            for db in g.datasets {
                ds_idx.push(flatten_dataset(db, all_ds, ds_vl)?);
            }
            let mut sub_grp_idx = Vec::new();
            for sg in g.sub_groups {
                sub_grp_idx.push(flatten_group(sg, all_ds, groups, grp_vl, ds_vl)?);
            }
            let gi = groups.len();
            groups.push(GrpFlat { name: g.name, attrs: gattrs, ds_indices: ds_idx, sub_group_indices: sub_grp_idx });
            grp_vl.push(patches);
            Ok(gi)
        }

        let mut grp_vl: Vec<Vec<VlPatch>> = Vec::new();
        let mut ds_vl: Vec<Vec<VlPatch>> = Vec::new();

        for db in self.root_datasets {
            root_ds_indices.push(flatten_dataset(db, &mut all_ds, &mut ds_vl)?);
        }

        for g in self.groups.into_iter() {
            root_group_indices.push(flatten_group(g, &mut all_ds, &mut groups, &mut grp_vl, &mut ds_vl)?);
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

        let is_chunked: Vec<bool> = all_ds.iter().map(|d| d.chunk_options.is_chunked() || d.maxshape.is_some()).collect();
        let root_dense = root_attrs.len() > DENSE_ATTR_THRESHOLD;
        let group_dense: Vec<bool> = groups.iter().map(|g| g.attrs.len() > DENSE_ATTR_THRESHOLD).collect();
        let ds_dense: Vec<bool> = all_ds.iter().map(|d| d.attrs.len() > DENSE_ATTR_THRESHOLD).collect();

        // Pass 1: compute OH sizes with dummy addresses
        let group_oh_sizes: Vec<usize> = groups.iter().enumerate().map(|(gi, g)| {
            let mut dummy_links: Vec<LinkMessage> = g.ds_indices.iter().map(|&i| make_link(&all_ds[i].name, 0)).collect();
            for &sgi in &g.sub_group_indices { dummy_links.push(make_link(&groups[sgi].name, 0)); }
            if group_dense[gi] {
                let dummy_blob = build_dense_attrs(&g.attrs, 0);
                build_group_oh(&dummy_links, &g.attrs, Some(&dummy_blob)).len()
            } else {
                build_group_oh(&dummy_links, &g.attrs, None).len()
            }
        }).collect();

        let root_dummy_links: Vec<LinkMessage> = {
            let mut links = Vec::new();
            for &i in &root_ds_indices { links.push(make_link(&all_ds[i].name, 0)); }
            for &gi in &root_group_indices { links.push(make_link(&groups[gi].name, 0)); }
            links
        };
        let root_oh_size = if root_dense {
            let dummy_blob = build_dense_attrs(&root_attrs, 0);
            build_group_oh(&root_dummy_links, &root_attrs, Some(&dummy_blob)).len()
        } else {
            build_group_oh(&root_dummy_links, &root_attrs, None).len()
        };

        struct DataBlob { data: Vec<u8>, oh_bytes: Vec<u8> }

        let mut dummy_blobs: Vec<DataBlob> = Vec::new();
        let mut dummy_cursor = 0u64;
        for (i, d) in all_ds.iter().enumerate() {
            if is_chunked[i] {
                let chunk_dims = d.chunk_options.resolve_chunk_dims(&d.ds.dimensions);
                let elem_size = d.dt.type_size() as usize;
                let zfp_elem_ty = zfp_element_type_from_datatype(&d.dt);
                let result = build_chunked_data_at_ext(&d.raw, &d.ds.dimensions, &chunk_dims, elem_size, &d.chunk_options, dummy_cursor, d.maxshape.as_deref(), zfp_elem_ty)?;
                dummy_cursor += result.data_bytes.len() as u64;
                let dense_blob = if ds_dense[i] { Some(build_dense_attrs(&d.attrs, 0)) } else { None };
                let oh = build_chunked_dataset_oh(&d.dt, &d.ds, &result.layout_message, result.pipeline_message.as_deref(), &d.attrs, dense_blob.as_ref());
                dummy_blobs.push(DataBlob { data: result.data_bytes, oh_bytes: oh });
            } else {
                let dense_blob = if ds_dense[i] { Some(build_dense_attrs(&d.attrs, 0)) } else { None };
                let oh = build_dataset_oh(&d.dt, &d.ds, 0, d.raw.len() as u64, &d.attrs, dense_blob.as_ref());
                dummy_blobs.push(DataBlob { data: d.raw.clone(), oh_bytes: oh });
            }
        }

        let actual_ds_oh_sizes: Vec<usize> = dummy_blobs.iter().map(|b| b.oh_bytes.len()).collect();

        // Pass 2: compute real addresses.
        // All addresses stored in the file are relative to base_address.
        // base_address = userblock_size. cursor2 tracks relative positions.
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
        let group_addrs2: Vec<u64> = group_oh_sizes.iter().enumerate().map(|(gi, &sz)| {
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
        }).collect();

        let mut ds_dense_blobs: Vec<Option<DenseAttrBlob>> = Vec::new();
        let ds_oh_addrs2: Vec<u64> = actual_ds_oh_sizes.iter().enumerate().map(|(i, &sz)| {
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
        }).collect();

        // Resolve path-based references now that all addresses are known.
        // Build a map of (group_name, child_name) -> address for resolution.
        {
            // Build a path->address map for all datasets and groups.
            // Root-level datasets: path = dataset_name
            // Group-level datasets: path = group_name/dataset_name (recursive)
            // Groups: path = group_name (recursive)
            let mut path_map = HashMap::<String, u64>::new();
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
                            sgi, groups, ds_addrs, grp_addrs, all_ds, map,
                        );
                    }
                }
                register_group(&groups[gi].name, gi, &groups, &ds_oh_addrs2, &group_addrs2, &all_ds, &mut path_map);
            }

            // Patch reference datasets
            for d in all_ds.iter_mut() {
                if let Some(ref targets) = d.reference_targets {
                    let mut patched = Vec::with_capacity(targets.len() * 8);
                    for path in targets {
                        let addr = path_map.get(path).copied().unwrap_or(u64::MAX);
                        patched.extend_from_slice(&addr.to_le_bytes());
                    }
                    d.raw = patched;
                }
            }
        }

        // Compute data layout (addresses + chunked data blobs) separately from OHs
        // so we can patch VL attrs before building OHs.
        struct DsLayout {
            data: Vec<u8>,
            data_addr: u64,
            chunked_msgs: Option<(Vec<u8>, Option<Vec<u8>>)>,
        }
        let mut ds_layouts: Vec<DsLayout> = Vec::new();
        for (i, d) in all_ds.iter().enumerate() {
            if is_chunked[i] {
                let chunk_dims = d.chunk_options.resolve_chunk_dims(&d.ds.dimensions);
                let elem_size = d.dt.type_size() as usize;
                let base_address = cursor2 as u64;
                let zfp_elem_ty = zfp_element_type_from_datatype(&d.dt);
                let result = build_chunked_data_at_ext(&d.raw, &d.ds.dimensions, &chunk_dims, elem_size, &d.chunk_options, base_address, d.maxshape.as_deref(), zfp_elem_ty)?;
                cursor2 += result.data_bytes.len();
                ds_layouts.push(DsLayout {
                    data: result.data_bytes, data_addr: base_address,
                    chunked_msgs: Some((result.layout_message, result.pipeline_message)),
                });
            } else {
                let data = d.raw.clone();
                let addr = if data.is_empty() { u64::MAX } else {
                    let a = cursor2 as u64;
                    cursor2 += data.len();
                    a
                };
                ds_layouts.push(DsLayout { data, data_addr: addr, chunked_msgs: None });
            }
        }

        // Patch VL attrs with pre-computed GCOL addresses (GCOLs go after all data).
        let has_vl = !vl_root.is_empty()
            || grp_vl.iter().any(|v| !v.is_empty())
            || ds_vl.iter().any(|v| !v.is_empty());

        let mut gcol_total_size = 0usize;
        if has_vl {
            let mut gcol_cursor = cursor2 as u64;
            for patch in &vl_root {
                patch_vl_refs(&mut root_attrs[patch.attr_index].raw_data, gcol_cursor);
                gcol_cursor += patch.collection_bytes.len() as u64;
            }
            for (gi, patches) in grp_vl.iter().enumerate() {
                for patch in patches {
                    patch_vl_refs(&mut groups[gi].attrs[patch.attr_index].raw_data, gcol_cursor);
                    gcol_cursor += patch.collection_bytes.len() as u64;
                }
            }
            for (di, patches) in ds_vl.iter().enumerate() {
                for patch in patches {
                    patch_vl_refs(&mut all_ds[di].attrs[patch.attr_index].raw_data, gcol_cursor);
                    gcol_cursor += patch.collection_bytes.len() as u64;
                }
            }
            gcol_total_size = (gcol_cursor - cursor2 as u64) as usize;
        }

        // Build dataset OHs now that attrs are patched.
        let mut ds_blobs2: Vec<DataBlob> = Vec::new();
        for (i, d) in all_ds.iter().enumerate() {
            let layout = &ds_layouts[i];
            let oh = if let Some((ref lm, ref pm)) = layout.chunked_msgs {
                build_chunked_dataset_oh(&d.dt, &d.ds, lm, pm.as_deref(), &d.attrs, ds_dense_blobs[i].as_ref())
            } else {
                build_dataset_oh(&d.dt, &d.ds, layout.data_addr, layout.data.len() as u64, &d.attrs, ds_dense_blobs[i].as_ref())
            };
            ds_blobs2.push(DataBlob { data: layout.data.clone(), oh_bytes: oh });
        }

        let actual_ds_oh_sizes2: Vec<usize> = ds_blobs2.iter().map(|b| b.oh_bytes.len()).collect();
        debug_assert_eq!(actual_ds_oh_sizes, actual_ds_oh_sizes2);

        // eof_address is absolute file size (includes userblock + GCOLs)
        let eof_addr2 = (ub + cursor2 + gcol_total_size) as u64;
        let mut buf = Vec::with_capacity(eof_addr2 as usize);

        // Userblock: prepend zeros
        if ub > 0 {
            buf.resize(ub, 0);
        }

        let sb = Superblock {
            version: 3, offset_size: OFFSET_SIZE, length_size: LENGTH_SIZE,
            base_address: ub as u64, eof_address: eof_addr2, root_group_address: root_group_addr,
            group_leaf_node_k: None, group_internal_node_k: None, indexed_storage_internal_node_k: None,
            free_space_address: None, driver_info_address: None,
            consistency_flags: 0, superblock_extension_address: Some(u64::MAX), checksum: None,
        };
        buf.extend_from_slice(&sb.serialize());

        // Root group OH
        let root_links: Vec<LinkMessage> = {
            let mut v = Vec::new();
            for &i in &root_ds_indices { v.push(make_link(&all_ds[i].name, ds_oh_addrs2[i])); }
            for &gi in &root_group_indices { v.push(make_link(&groups[gi].name, group_addrs2[gi])); }
            v
        };
        buf.extend_from_slice(&build_group_oh(&root_links, &root_attrs, root_dense_blob.as_ref()));
        if let Some(ref blob) = root_dense_blob { buf.extend_from_slice(&blob.blob); }

        // Group OHs + dense blobs
        for (gi, g) in groups.iter().enumerate() {
            let mut links: Vec<LinkMessage> = g.ds_indices.iter().map(|&i| make_link(&all_ds[i].name, ds_oh_addrs2[i])).collect();
            for &sgi in &g.sub_group_indices { links.push(make_link(&groups[sgi].name, group_addrs2[sgi])); }
            buf.extend_from_slice(&build_group_oh(&links, &g.attrs, group_dense_blobs[gi].as_ref()));
            if let Some(ref blob) = group_dense_blobs[gi] { buf.extend_from_slice(&blob.blob); }
        }

        // Dataset OHs + dense blobs
        for (i, blob) in ds_blobs2.iter().enumerate() {
            buf.extend_from_slice(&blob.oh_bytes);
            if let Some(ref dense) = ds_dense_blobs[i] { buf.extend_from_slice(&dense.blob); }
        }

        // Data
        for blob in &ds_blobs2 { buf.extend_from_slice(&blob.data); }

        // Global heap collections
        for patch in &vl_root { buf.extend_from_slice(&patch.collection_bytes); }
        for patches in &grp_vl { for patch in patches { buf.extend_from_slice(&patch.collection_bytes); } }
        for patches in &ds_vl { for patch in patches { buf.extend_from_slice(&patch.collection_bytes); } }

        Ok(buf)
    }
}

// ---- Independent parallel dataset creation ----

/// Builder that creates datasets without locking the file header.
///
/// Each `IndependentDatasetBuilder` accumulates its own [`MetadataBlock`]
/// independently. On [`IndependentDatasetBuilder::finish`], the block is
/// returned for later merging.
///
/// Thread-safety: each thread should own its own builder instance.
pub struct IndependentDatasetBuilder {
    block: MetadataBlock,
}

impl IndependentDatasetBuilder {
    /// Create a new independent builder with the given creator id.
    pub fn new(creator_id: u32) -> Self {
        Self {
            block: MetadataBlock::new(creator_id),
        }
    }

    /// Add a dataset specification to this builder.
    pub fn add_dataset(&mut self, meta: DatasetMetadata) {
        self.block.add_dataset(meta);
    }

    /// Consume the builder and return the metadata block.
    pub fn finish(self) -> MetadataBlock {
        self.block
    }
}

/// Finalize multiple independently-created metadata blocks into a complete HDF5 file.
///
/// This implements the write-ahead approach: each block's data is laid out
/// sequentially, then the index table (root group with links) is written last
/// to point at all the dataset object headers.
pub fn finalize_parallel(blocks: Vec<MetadataBlock>) -> Result<Vec<u8>, FormatError> {
    let index = MetadataIndex::merge_blocks(&blocks)?;
    finalize_from_index(index)
}

/// Build a complete HDF5 file from a merged MetadataIndex.
fn finalize_from_index(index: MetadataIndex) -> Result<Vec<u8>, FormatError> {
    // Convert DatasetMetadata into the internal DsFlat representation and
    // delegate to the same two-pass algorithm used by FileWriter.
    let mut fw = FileWriter::new();
    for ds_meta in &index.datasets {
        let db = fw.create_dataset(&ds_meta.name);
        // Set the datatype and raw data directly via internal fields
        db.datatype = Some(ds_meta.datatype.clone());
        db.shape = Some(ds_meta.dataspace.dimensions.clone());
        db.maxshape = ds_meta.maxshape.clone();
        db.data = Some(ds_meta.raw_data.clone());
        db.chunk_options = ds_meta.chunk_options.clone();
        for (name, val) in &ds_meta.attrs {
            db.set_attr(name, val.clone());
        }
    }
    fw.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::group_v2::resolve_path_any;
    use crate::object_header::ObjectHeader;
    use crate::signature;

    fn parse_file(bytes: &[u8]) -> (Superblock, ObjectHeader) {
        let sig = signature::find_signature(bytes).unwrap();
        let sb = Superblock::parse(bytes, sig).unwrap();
        let oh = ObjectHeader::parse(bytes, sb.root_group_address as usize, sb.offset_size, sb.length_size).unwrap();
        (sb, oh)
    }

    fn read_dataset_f64(bytes: &[u8], path: &str) -> Vec<f64> {
        let sig = signature::find_signature(bytes).unwrap();
        let sb = Superblock::parse(bytes, sig).unwrap();
        let addr = resolve_path_any(bytes, &sb, path).unwrap();
        let hdr = ObjectHeader::parse(bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
        let dt_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::Datatype).unwrap().data;
        let ds_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::Dataspace).unwrap().data;
        let dl_data = &hdr.messages.iter().find(|m| m.msg_type == MessageType::DataLayout).unwrap().data;
        let (dt, _) = Datatype::parse(dt_data).unwrap();
        let ds = Dataspace::parse(ds_data, sb.length_size).unwrap();
        let dl = crate::data_layout::DataLayout::parse(dl_data, sb.offset_size, sb.length_size).unwrap();
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
        fw.create_dataset("data").with_f64_data(&[1.0, 2.0]).set_attr("scale", AttrValue::F64(0.5));
        let bytes = fw.finish().unwrap();
        assert_eq!(read_dataset_f64(&bytes, "data"), vec![1.0, 2.0]);
        let sig = signature::find_signature(&bytes).unwrap();
        let sb = Superblock::parse(&bytes, sig).unwrap();
        let addr = resolve_path_any(&bytes, &sb, "data").unwrap();
        let hdr = ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
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
        let hdr = ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
        let attrs = crate::attribute::extract_attributes_full(&bytes, &hdr, sb.offset_size, sb.length_size).unwrap();
        assert_eq!(attrs.len(), 20);
        for i in 0..20 {
            let attr = attrs.iter().find(|a| a.name == format!("attr_{i:03}")).unwrap();
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
        let oh = ObjectHeader::parse(&bytes, sb.root_group_address as usize, sb.offset_size, sb.length_size).unwrap();
        let attrs = crate::attribute::extract_attributes_full(&bytes, &oh, sb.offset_size, sb.length_size).unwrap();
        assert_eq!(attrs.len(), 15);
    }

    #[test]
    fn inline_attrs_below_threshold() {
        let mut fw = FileWriter::new();
        let ds = fw.create_dataset("data");
        ds.with_f64_data(&[1.0]);
        for i in 0..5 { ds.set_attr(&format!("a{i}"), AttrValue::F64(i as f64)); }
        let bytes = fw.finish().unwrap();
        let sig = signature::find_signature(&bytes).unwrap();
        let sb = Superblock::parse(&bytes, sig).unwrap();
        let addr = resolve_path_any(&bytes, &sb, "data").unwrap();
        let hdr = ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
        assert!(!hdr.messages.iter().any(|m| m.msg_type == MessageType::AttributeInfo));
        let attrs = crate::attribute::extract_attributes(&hdr, sb.length_size).unwrap();
        assert_eq!(attrs.len(), 5);
    }

    #[test]
    fn encode_decode_managed_id_roundtrip() {
        let id = encode_managed_id(100, 42, 40, 8);
        let fh = crate::fractal_heap::FractalHeapHeader {
            heap_id_length: 8, io_filter_encoded_length: 0,
            max_managed_object_size: 1024, table_width: 4,
            starting_block_size: 4096, max_direct_block_size: 65536,
            max_heap_size: 40, starting_row_of_indirect_blocks: 1,
            root_block_address: 0, current_rows_in_root_indirect_block: 0,
            managed_objects_count: 0,
        };
        let (off, len) = fh.decode_managed_id(&id).unwrap();
        assert_eq!(off, 100);
        assert_eq!(len, 42);
    }

    #[test]
    fn finalize_parallel_basic() {
        use crate::metadata_index::{MetadataBlock, build_dataset_metadata};
        use crate::chunked_write::ChunkOptions;
        use crate::type_builders::make_f64_type;

        let mut b0 = MetadataBlock::new(0);
        let data_a: Vec<u8> = [1.0f64, 2.0, 3.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        b0.add_dataset(build_dataset_metadata(
            "alpha", make_f64_type(), vec![3], data_a,
            ChunkOptions::default(), None, vec![],
        ));

        let mut b1 = MetadataBlock::new(1);
        let data_b: Vec<u8> = [10.0f64, 20.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        b1.add_dataset(build_dataset_metadata(
            "beta", make_f64_type(), vec![2], data_b,
            ChunkOptions::default(), None, vec![],
        ));

        let bytes = finalize_parallel(vec![b0, b1]).unwrap();
        assert_eq!(read_dataset_f64(&bytes, "alpha"), vec![1.0, 2.0, 3.0]);
        assert_eq!(read_dataset_f64(&bytes, "beta"), vec![10.0, 20.0]);
    }

    #[test]
    fn finalize_parallel_duplicate_error() {
        use crate::metadata_index::{MetadataBlock, build_dataset_metadata};
        use crate::chunked_write::ChunkOptions;
        use crate::type_builders::make_f64_type;

        let mut b0 = MetadataBlock::new(0);
        b0.add_dataset(build_dataset_metadata(
            "dup", make_f64_type(), vec![1], vec![0u8; 8],
            ChunkOptions::default(), None, vec![],
        ));
        let mut b1 = MetadataBlock::new(1);
        b1.add_dataset(build_dataset_metadata(
            "dup", make_f64_type(), vec![1], vec![0u8; 8],
            ChunkOptions::default(), None, vec![],
        ));
        let err = finalize_parallel(vec![b0, b1]).unwrap_err();
        assert!(matches!(err, FormatError::DuplicateDatasetName(_)));
    }
}
