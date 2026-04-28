//! V1 group traversal: resolve group children and navigate paths.

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

use crate::btree_v1::collect_symbol_table_nodes;
use crate::error::FormatError;
use crate::local_heap::LocalHeap;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::symbol_table::{SymbolTableMessage, SymbolTableNode};

/// A resolved group entry (child name + object header address).
#[derive(Debug, Clone)]
pub struct GroupEntry {
    /// Name of the child object.
    pub name: String,
    /// Address of the child's object header.
    pub object_header_address: u64,
    /// Cache type from the symbol table entry.
    pub cache_type: u32,
}

/// Given a SymbolTableMessage, resolve all group children.
///
/// `base_address` is the superblock base address (typically the userblock size).
/// All addresses stored in v1 structures are relative to this value.
pub fn resolve_v1_group_entries(
    file_data: &[u8],
    sym_table_msg: &SymbolTableMessage,
    offset_size: u8,
    length_size: u8,
    base_address: u64,
) -> Result<Vec<GroupEntry>, FormatError> {
    // Parse local heap (address is relative to base_address)
    let mut heap = LocalHeap::parse(
        file_data,
        (sym_table_msg.local_heap_address + base_address) as usize,
        offset_size,
        length_size,
    )?;
    // The data segment address stored in the heap is also relative to base_address
    heap.data_segment_address += base_address;

    // Collect all SNOD addresses from B-tree (btree_address is relative to base_address)
    let snod_addrs = collect_symbol_table_nodes(
        file_data,
        sym_table_msg.btree_address,
        offset_size,
        length_size,
        base_address,
    )?;

    let mut entries = Vec::new();
    for snod_addr in snod_addrs {
        // SNOD addresses from B-tree children are also relative to base_address
        let snod =
            SymbolTableNode::parse(file_data, (snod_addr + base_address) as usize, offset_size)?;
        for entry in &snod.entries {
            let name = heap.read_string(file_data, entry.link_name_offset)?;
            entries.push(GroupEntry {
                name,
                object_header_address: entry.object_header_address,
                cache_type: entry.cache_type,
            });
        }
    }

    Ok(entries)
}

/// Extract the SymbolTableMessage from an object header's messages.
fn find_symbol_table_message(
    obj_header: &ObjectHeader,
    offset_size: u8,
) -> Result<SymbolTableMessage, FormatError> {
    for msg in &obj_header.messages {
        if msg.msg_type == MessageType::SymbolTable {
            return SymbolTableMessage::parse(&msg.data, offset_size);
        }
    }
    Err(FormatError::PathNotFound(String::from(
        "no symbol table message found in object header",
    )))
}

/// Navigate a path like "group1/subgroup/dataset" from a root group.
/// Returns the object header address of the target.
///
/// Note: this function does not apply base_address offsets. It is intended for
/// use with files that have base_address=0. Use `resolve_path_any` from group_v2
/// for general-purpose path resolution.
pub fn resolve_path(
    file_data: &[u8],
    root_sym_table: &SymbolTableMessage,
    path: &str,
    offset_size: u8,
    length_size: u8,
) -> Result<u64, FormatError> {
    let components: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    if components.is_empty() {
        return Err(FormatError::PathNotFound(String::from(path)));
    }

    let mut current_sym_table = root_sym_table.clone();

    for (i, component) in components.iter().enumerate() {
        let entries =
            resolve_v1_group_entries(file_data, &current_sym_table, offset_size, length_size, 0)?;

        let found = entries.iter().find(|e| e.name == *component);
        match found {
            Some(entry) => {
                if i == components.len() - 1 {
                    // Last component — return its address
                    return Ok(entry.object_header_address);
                }
                // Not last — must be a group, parse its object header to get symbol table
                let obj_header = ObjectHeader::parse(
                    file_data,
                    entry.object_header_address as usize,
                    offset_size,
                    length_size,
                )?;
                current_sym_table = find_symbol_table_message(&obj_header, offset_size)?;
            }
            None => {
                return Err(FormatError::PathNotFound(String::from(*component)));
            }
        }
    }

    Err(FormatError::PathNotFound(String::from(path)))
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Build a minimal synthetic file with a group containing named children.
    /// Returns (file_data, SymbolTableMessage).
    fn build_synthetic_group(
        children: &[(&str, u64, u32)], // (name, obj_header_addr, cache_type)
        offset_size: u8,
        length_size: u8,
    ) -> (Vec<u8>, SymbolTableMessage) {
        let os = offset_size as usize;
        let ls = length_size as usize;

        // Build local heap data segment (names)
        let mut heap_data = Vec::new();
        let mut name_offsets = Vec::new();
        for (name, _, _) in children {
            name_offsets.push(heap_data.len() as u64);
            heap_data.extend_from_slice(name.as_bytes());
            heap_data.push(0);
        }
        let heap_data_size = heap_data.len();

        // Layout:
        // 0: local heap header
        // heap_header_end: heap data segment
        // after heap data: SNOD
        // after SNOD: B-tree leaf

        let heap_offset = 0usize;
        let heap_header_size = 8 + ls * 2 + os;
        let heap_data_offset = heap_header_size;
        let snod_offset = heap_data_offset + heap_data_size;
        // Pad to nice offset
        let snod_offset = (snod_offset + 7) & !7;

        let entry_size = os + os + 4 + 4 + 16;
        let snod_size = 8 + children.len() * entry_size;
        let btree_offset = snod_offset + snod_size;
        let btree_offset = (btree_offset + 7) & !7;

        // B-tree: entries_used = 1 child (the SNOD), keys = [0, last_name_end]
        let last_key = if children.is_empty() {
            0u64
        } else {
            heap_data_size as u64
        };
        let btree_header_size = 8 + os * 2; // sig + type + level + entries + siblings
        let btree_keys_children = os + os + os; // key[0] + child[0] + key[1]
        let total_size = btree_offset + btree_header_size + btree_keys_children + 64;

        let mut file = vec![0u8; total_size];

        // Write heap header
        {
            let mut pos = heap_offset;
            file[pos..pos + 4].copy_from_slice(b"HEAP");
            pos += 4;
            file[pos] = 0; // version
            pos += 4; // version(1) + reserved(3)
            // data_segment_size
            match length_size {
                4 => file[pos..pos + 4].copy_from_slice(&(heap_data_size as u32).to_le_bytes()),
                8 => file[pos..pos + 8].copy_from_slice(&(heap_data_size as u64).to_le_bytes()),
                _ => {}
            }
            pos += ls;
            // free_list_head_offset
            match length_size {
                4 => file[pos..pos + 4].copy_from_slice(&0xFFFFFFFFu32.to_le_bytes()),
                8 => file[pos..pos + 8].copy_from_slice(&0xFFFFFFFFFFFFFFFFu64.to_le_bytes()),
                _ => {}
            }
            pos += ls;
            // data_segment_address
            match offset_size {
                4 => file[pos..pos + 4].copy_from_slice(&(heap_data_offset as u32).to_le_bytes()),
                8 => file[pos..pos + 8].copy_from_slice(&(heap_data_offset as u64).to_le_bytes()),
                _ => {}
            }
        }

        // Write heap data segment
        file[heap_data_offset..heap_data_offset + heap_data_size].copy_from_slice(&heap_data);

        // Write SNOD
        {
            let mut pos = snod_offset;
            file[pos..pos + 4].copy_from_slice(b"SNOD");
            pos += 4;
            file[pos] = 1; // version
            pos += 1;
            pos += 1; // reserved
            file[pos..pos + 2].copy_from_slice(&(children.len() as u16).to_le_bytes());
            pos += 2;
            for (idx, &(_, obj_addr, cache_type)) in children.iter().enumerate() {
                // link_name_offset
                match offset_size {
                    4 => file[pos..pos + 4]
                        .copy_from_slice(&(name_offsets[idx] as u32).to_le_bytes()),
                    8 => file[pos..pos + 8].copy_from_slice(&name_offsets[idx].to_le_bytes()),
                    _ => {}
                }
                pos += os;
                // object_header_address
                match offset_size {
                    4 => file[pos..pos + 4].copy_from_slice(&(obj_addr as u32).to_le_bytes()),
                    8 => file[pos..pos + 8].copy_from_slice(&obj_addr.to_le_bytes()),
                    _ => {}
                }
                pos += os;
                file[pos..pos + 4].copy_from_slice(&cache_type.to_le_bytes());
                pos += 4;
                pos += 4; // reserved
                pos += 16; // scratch pad (zeros)
            }
        }

        // Write B-tree (leaf, level 0, 1 entry pointing to SNOD)
        {
            let mut pos = btree_offset;
            file[pos..pos + 4].copy_from_slice(b"TREE");
            pos += 4;
            file[pos] = 0; // type=group
            pos += 1;
            file[pos] = 0; // level=leaf
            pos += 1;
            file[pos..pos + 2].copy_from_slice(&1u16.to_le_bytes()); // entries_used=1
            pos += 2;
            // siblings = undefined
            for _ in 0..2 {
                match offset_size {
                    4 => file[pos..pos + 4].copy_from_slice(&0xFFFFFFFFu32.to_le_bytes()),
                    8 => file[pos..pos + 8].copy_from_slice(&0xFFFFFFFFFFFFFFFFu64.to_le_bytes()),
                    _ => {}
                }
                pos += os;
            }
            // key[0]
            match offset_size {
                4 => file[pos..pos + 4].copy_from_slice(&0u32.to_le_bytes()),
                8 => file[pos..pos + 8].copy_from_slice(&0u64.to_le_bytes()),
                _ => {}
            }
            pos += os;
            // child[0] = snod_offset
            match offset_size {
                4 => file[pos..pos + 4].copy_from_slice(&(snod_offset as u32).to_le_bytes()),
                8 => file[pos..pos + 8].copy_from_slice(&(snod_offset as u64).to_le_bytes()),
                _ => {}
            }
            pos += os;
            // key[1]
            match offset_size {
                4 => file[pos..pos + 4].copy_from_slice(&(last_key as u32).to_le_bytes()),
                8 => file[pos..pos + 8].copy_from_slice(&last_key.to_le_bytes()),
                _ => {}
            }
        }

        let msg = SymbolTableMessage {
            btree_address: btree_offset as u64,
            local_heap_address: heap_offset as u64,
        };

        (file, msg)
    }

    #[test]
    fn resolve_entries_two_children() {
        let (file, msg) = build_synthetic_group(&[("alpha", 0x1000, 0), ("beta", 0x2000, 0)], 8, 8);
        let entries = resolve_v1_group_entries(&file, &msg, 8, 8, 0).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "alpha");
        assert_eq!(entries[0].object_header_address, 0x1000);
        assert_eq!(entries[1].name, "beta");
        assert_eq!(entries[1].object_header_address, 0x2000);
    }

    #[test]
    fn resolve_path_single_level() {
        let (file, msg) =
            build_synthetic_group(&[("child1", 0x3000, 0), ("child2", 0x4000, 0)], 8, 8);
        let addr = resolve_path(&file, &msg, "child1", 8, 8).unwrap();
        assert_eq!(addr, 0x3000);
    }

    #[test]
    fn resolve_path_not_found() {
        let (file, msg) = build_synthetic_group(&[("x", 0x100, 0)], 8, 8);
        let err = resolve_path(&file, &msg, "nonexistent", 8, 8).unwrap_err();
        assert!(matches!(err, FormatError::PathNotFound(_)));
    }

    // Helper to extract dataset components from an object header
    fn extract_dataset(
        _file_data: &[u8],
        hdr: &crate::object_header::ObjectHeader,
        offset_size: u8,
        length_size: u8,
    ) -> (
        crate::datatype::Datatype,
        crate::dataspace::Dataspace,
        crate::data_layout::DataLayout,
    ) {
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
        let (dt, _) = crate::datatype::Datatype::parse(dt_data).unwrap();
        let ds = crate::dataspace::Dataspace::parse(ds_data, length_size).unwrap();
        let dl = crate::data_layout::DataLayout::parse(dl_data, offset_size, length_size).unwrap();
        (dt, ds, dl)
    }

    fn get_root_sym_table(
        file_data: &[u8],
        sb: &crate::superblock::Superblock,
    ) -> SymbolTableMessage {
        let root_header = ObjectHeader::parse(
            file_data,
            sb.root_group_address as usize,
            sb.offset_size,
            sb.length_size,
        )
        .unwrap();
        let sym_msg = root_header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::SymbolTable)
            .unwrap();
        SymbolTableMessage::parse(&sym_msg.data, sb.offset_size).unwrap()
    }

    // Integration tests with real HDF5 files

    #[test]
    fn integration_simple_dataset_full_traversal() {
        let file_data: &[u8] = include_bytes!("../tests/fixtures/simple_dataset.h5");
        let sig_offset = crate::signature::find_signature(file_data).unwrap();
        let sb = crate::superblock::Superblock::parse(file_data, sig_offset).unwrap();
        let root_sym = get_root_sym_table(file_data, &sb);

        let entries =
            resolve_v1_group_entries(file_data, &root_sym, sb.offset_size, sb.length_size, 0)
                .unwrap();
        let data_entry = entries
            .iter()
            .find(|e| e.name == "data")
            .expect("should have 'data'");

        let hdr = ObjectHeader::parse(
            file_data,
            data_entry.object_header_address as usize,
            sb.offset_size,
            sb.length_size,
        )
        .unwrap();
        let (dt, ds, dl) = extract_dataset(file_data, &hdr, sb.offset_size, sb.length_size);
        let raw = crate::data_read::read_raw_data(file_data, &dl, &ds, &dt).unwrap();
        let values = crate::data_read::read_as_f64(&raw, &dt).unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn integration_two_groups_group1_values() {
        let file_data: &[u8] = include_bytes!("../tests/fixtures/two_groups.h5");
        let sig_offset = crate::signature::find_signature(file_data).unwrap();
        let sb = crate::superblock::Superblock::parse(file_data, sig_offset).unwrap();
        let root_sym = get_root_sym_table(file_data, &sb);

        let addr = resolve_path(
            file_data,
            &root_sym,
            "group1/values",
            sb.offset_size,
            sb.length_size,
        )
        .unwrap();
        let hdr =
            ObjectHeader::parse(file_data, addr as usize, sb.offset_size, sb.length_size).unwrap();
        let (dt, ds, dl) = extract_dataset(file_data, &hdr, sb.offset_size, sb.length_size);
        let raw = crate::data_read::read_raw_data(file_data, &dl, &ds, &dt).unwrap();
        let values = crate::data_read::read_as_i32(&raw, &dt).unwrap();
        assert_eq!(values, vec![10, 20, 30]);
    }

    #[test]
    fn integration_two_groups_group2_temps() {
        let file_data: &[u8] = include_bytes!("../tests/fixtures/two_groups.h5");
        let sig_offset = crate::signature::find_signature(file_data).unwrap();
        let sb = crate::superblock::Superblock::parse(file_data, sig_offset).unwrap();
        let root_sym = get_root_sym_table(file_data, &sb);

        let addr = resolve_path(
            file_data,
            &root_sym,
            "group2/temps",
            sb.offset_size,
            sb.length_size,
        )
        .unwrap();
        let hdr =
            ObjectHeader::parse(file_data, addr as usize, sb.offset_size, sb.length_size).unwrap();
        let (dt, ds, dl) = extract_dataset(file_data, &hdr, sb.offset_size, sb.length_size);
        let raw = crate::data_read::read_raw_data(file_data, &dl, &ds, &dt).unwrap();
        let values = crate::data_read::read_as_f32(&raw, &dt).unwrap();
        assert!((values[0] - 98.6).abs() < 0.01);
        assert!((values[1] - 37.0).abs() < 0.01);
    }

    #[test]
    fn integration_nested_groups() {
        let file_data: &[u8] = include_bytes!("../tests/fixtures/nested_groups.h5");
        let sig_offset = crate::signature::find_signature(file_data).unwrap();
        let sb = crate::superblock::Superblock::parse(file_data, sig_offset).unwrap();
        let root_sym = get_root_sym_table(file_data, &sb);

        let addr = resolve_path(
            file_data,
            &root_sym,
            "a/b/c/deep",
            sb.offset_size,
            sb.length_size,
        )
        .unwrap();
        let hdr =
            ObjectHeader::parse(file_data, addr as usize, sb.offset_size, sb.length_size).unwrap();
        let (dt, ds, dl) = extract_dataset(file_data, &hdr, sb.offset_size, sb.length_size);
        let raw = crate::data_read::read_raw_data(file_data, &dl, &ds, &dt).unwrap();
        let values = crate::data_read::read_as_f64(&raw, &dt).unwrap();
        assert_eq!(values, vec![42.0]);
    }
}
