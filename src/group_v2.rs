//! V2 group traversal: resolve group children and navigate paths.
//!
//! Handles both compact storage (Link messages in object header) and
//! dense storage (fractal heap + B-tree v2).

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

use crate::btree_v2::{collect_btree_v2_records, BTreeV2Header};
use crate::error::FormatError;
use crate::fractal_heap::FractalHeapHeader;
use crate::group_v1::{self, GroupEntry};
use crate::link_info::LinkInfoMessage;
use crate::link_message::{LinkMessage, LinkTarget};
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::superblock::Superblock;
use crate::symbol_table::SymbolTableMessage;

/// Resolve v2 group entries from an object header.
///
/// Handles both compact (Link messages) and dense (fractal heap + B-tree v2) storage.
pub fn resolve_v2_group_entries(
    file_data: &[u8],
    object_header: &ObjectHeader,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<GroupEntry>, FormatError> {
    // Look for Link Info message to determine storage type
    let link_info = find_link_info(object_header, offset_size)?;

    if let Some(fh_addr) = link_info.fractal_heap_address {
        // Dense storage
        resolve_dense_entries(file_data, &link_info, fh_addr, offset_size, length_size)
    } else {
        // Compact storage: links are stored directly as Link messages
        resolve_compact_entries(object_header, offset_size)
    }
}

/// Extract link entries from Link messages directly in the object header (compact storage).
fn resolve_compact_entries(
    object_header: &ObjectHeader,
    offset_size: u8,
) -> Result<Vec<GroupEntry>, FormatError> {
    let mut entries = Vec::new();
    for msg in &object_header.messages {
        if msg.msg_type == MessageType::Link {
            let link = LinkMessage::parse(&msg.data, offset_size)?;
            if let LinkTarget::Hard {
                object_header_address,
            } = link.link_target
            {
                entries.push(GroupEntry {
                    name: link.name,
                    object_header_address,
                    cache_type: 0,
                });
            }
            // Skip soft and external links for path resolution
        }
    }
    Ok(entries)
}

/// Resolve entries from dense storage (fractal heap + B-tree v2).
fn resolve_dense_entries(
    file_data: &[u8],
    link_info: &LinkInfoMessage,
    fh_addr: u64,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<GroupEntry>, FormatError> {
    // Parse fractal heap
    let fh = FractalHeapHeader::parse(file_data, fh_addr as usize, offset_size, length_size)?;

    // Parse B-tree v2 for name index
    let btree_addr = link_info
        .btree_name_index_address
        .ok_or_else(|| FormatError::PathNotFound(String::from("no B-tree v2 name index")))?;
    let btree_hdr =
        BTreeV2Header::parse(file_data, btree_addr as usize, offset_size, length_size)?;
    let records = collect_btree_v2_records(file_data, &btree_hdr, offset_size, length_size)?;

    let mut entries = Vec::new();
    for record in &records {
        // For type 5 (name index): hash(4) + heap_id(heap_id_length)
        // For type 6 (creation order): creation_order(8) + heap_id(heap_id_length)
        let id_offset = if btree_hdr.tree_type == 5 {
            4 // skip hash
        } else {
            8 // skip creation_order
        };

        if record.data.len() < id_offset + fh.heap_id_length as usize {
            continue;
        }
        let id_bytes = &record.data[id_offset..id_offset + fh.heap_id_length as usize];

        // Read managed object from fractal heap
        let link_data = fh.read_managed_object(file_data, id_bytes, offset_size)?;

        // Parse as Link message
        let link = LinkMessage::parse(&link_data, offset_size)?;
        if let LinkTarget::Hard {
            object_header_address,
        } = link.link_target
        {
            entries.push(GroupEntry {
                name: link.name,
                object_header_address,
                cache_type: 0,
            });
        }
    }

    Ok(entries)
}

/// Find and parse the Link Info message from an object header.
fn find_link_info(
    object_header: &ObjectHeader,
    offset_size: u8,
) -> Result<LinkInfoMessage, FormatError> {
    for msg in &object_header.messages {
        if msg.msg_type == MessageType::LinkInfo {
            return LinkInfoMessage::parse(&msg.data, offset_size);
        }
    }
    // No Link Info message — might have direct Link messages
    // Return a "compact" link info with no fractal heap
    Ok(LinkInfoMessage {
        max_creation_order: None,
        fractal_heap_address: None,
        btree_name_index_address: None,
        btree_creation_order_address: None,
    })
}

/// Detect whether an object header represents a v1 group, v2 group, or neither.
fn is_v2_group(object_header: &ObjectHeader) -> bool {
    object_header
        .messages
        .iter()
        .any(|m| m.msg_type == MessageType::LinkInfo || m.msg_type == MessageType::Link)
}

fn is_v1_group(object_header: &ObjectHeader) -> bool {
    object_header
        .messages
        .iter()
        .any(|m| m.msg_type == MessageType::SymbolTable)
}

/// Unified path resolution that works for both v1 and v2 groups.
///
/// Detects group version from object header messages and dispatches accordingly.
pub fn resolve_path_any(
    file_data: &[u8],
    superblock: &Superblock,
    path: &str,
) -> Result<u64, FormatError> {
    let components: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    if components.is_empty() {
        return Ok(superblock.root_group_address);
    }

    let os = superblock.offset_size;
    let ls = superblock.length_size;
    let base = superblock.base_address;

    let root_header =
        ObjectHeader::parse(file_data, superblock.root_group_address as usize, os, ls)?;

    let mut current_addr = superblock.root_group_address;
    let mut current_header = root_header;

    for (i, component) in components.iter().enumerate() {
        let entries = resolve_group_entries(file_data, &current_header, os, ls)?;

        let found = entries.iter().find(|e| e.name == *component);
        match found {
            Some(entry) => {
                // Link addresses are relative to base_address; convert to absolute.
                let abs_addr = entry.object_header_address + base;
                if i == components.len() - 1 {
                    return Ok(abs_addr);
                }
                current_addr = abs_addr;
                current_header =
                    ObjectHeader::parse(file_data, current_addr as usize, os, ls)?;
            }
            None => {
                return Err(FormatError::PathNotFound(String::from(*component)));
            }
        }
    }

    Ok(current_addr)
}

/// Resolve group entries from an object header, auto-detecting v1 vs v2.
pub fn resolve_group_entries(
    file_data: &[u8],
    object_header: &ObjectHeader,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<GroupEntry>, FormatError> {
    if is_v1_group(object_header) {
        // v1: find SymbolTableMessage and use existing v1 code
        let sym_msg = object_header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::SymbolTable)
            .ok_or_else(|| {
                FormatError::PathNotFound(String::from("no symbol table message"))
            })?;
        let stm = SymbolTableMessage::parse(&sym_msg.data, offset_size)?;
        group_v1::resolve_v1_group_entries(file_data, &stm, offset_size, length_size)
    } else if is_v2_group(object_header) {
        resolve_v2_group_entries(file_data, object_header, offset_size, length_size)
    } else {
        Err(FormatError::PathNotFound(String::from(
            "object header is not a group",
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_layout::DataLayout;
    use crate::data_read;
    use crate::dataspace::Dataspace;
    use crate::datatype::Datatype;
    use crate::signature;

    fn extract_dataset(
        _file_data: &[u8],
        hdr: &ObjectHeader,
        offset_size: u8,
        length_size: u8,
    ) -> (Datatype, Dataspace, DataLayout) {
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
        let ds = Dataspace::parse(ds_data, length_size).unwrap();
        let dl = DataLayout::parse(dl_data, offset_size, length_size).unwrap();
        (dt, ds, dl)
    }

    #[test]
    fn compact_storage_link_messages() {
        // Build a v2 object header with Link messages (compact storage)
        // We'll test with the actual v2_groups.h5 file since building synthetic v2 headers
        // with proper checksums is complex.

        // Instead, test the resolve_compact_entries path with a simple object header
        let link_data = {
            // Build a Link message: hard link, name="test", addr=0x1000
            let mut d = Vec::new();
            d.push(1); // version
            d.push(0x00); // flags: no creation order, no link type (=hard), no charset, name_size=1byte
            d.push(4); // name length = 4
            d.extend_from_slice(b"test");
            d.extend_from_slice(&0x1000u64.to_le_bytes()); // address
            d
        };

        let oh = ObjectHeader {
            version: 2,
            messages: vec![
                crate::object_header::HeaderMessage {
                    msg_type: MessageType::LinkInfo,
                    size: 18,
                    flags: 0,
                    creation_order: None,
                    data: {
                        let mut d = Vec::new();
                        d.push(0); // version
                        d.push(0); // flags
                        d.extend_from_slice(&0xFFFF_FFFF_FFFF_FFFFu64.to_le_bytes()); // fh undef
                        d.extend_from_slice(&0xFFFF_FFFF_FFFF_FFFFu64.to_le_bytes()); // btree undef
                        d
                    },
                },
                crate::object_header::HeaderMessage {
                    msg_type: MessageType::Link,
                    size: link_data.len(),
                    flags: 0,
                    creation_order: None,
                    data: link_data,
                },
            ],
            reference_count: None,
            flags: 0,
            access_time: None,
            modification_time: None,
            change_time: None,
            birth_time: None,
        };

        let entries = resolve_v2_group_entries(&[], &oh, 8, 8).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "test");
        assert_eq!(entries[0].object_header_address, 0x1000);
    }

    #[test]
    fn integration_v2_groups_temperature() {
        let file_data: &[u8] = include_bytes!("../tests/fixtures/v2_groups.h5");
        let sig_offset = signature::find_signature(file_data).unwrap();
        let sb = Superblock::parse(file_data, sig_offset).unwrap();
        assert!(sb.version >= 2); // v2/v3 superblock

        let addr = resolve_path_any(file_data, &sb, "sensors/temperature").unwrap();
        let hdr = ObjectHeader::parse(file_data, addr as usize, sb.offset_size, sb.length_size)
            .unwrap();
        let (dt, ds, dl) = extract_dataset(file_data, &hdr, sb.offset_size, sb.length_size);
        let raw = data_read::read_raw_data(file_data, &dl, &ds, &dt).unwrap();
        let values = data_read::read_as_f64(&raw, &dt).unwrap();
        assert_eq!(values, vec![22.5, 23.1, 21.8]);
    }

    #[test]
    fn integration_v2_groups_humidity() {
        let file_data: &[u8] = include_bytes!("../tests/fixtures/v2_groups.h5");
        let sig_offset = signature::find_signature(file_data).unwrap();
        let sb = Superblock::parse(file_data, sig_offset).unwrap();

        let addr = resolve_path_any(file_data, &sb, "sensors/humidity").unwrap();
        let hdr = ObjectHeader::parse(file_data, addr as usize, sb.offset_size, sb.length_size)
            .unwrap();
        let (dt, ds, dl) = extract_dataset(file_data, &hdr, sb.offset_size, sb.length_size);
        let raw = data_read::read_raw_data(file_data, &dl, &ds, &dt).unwrap();
        let values = data_read::read_as_i32(&raw, &dt).unwrap();
        assert_eq!(values, vec![45, 50, 55]);
    }

    #[test]
    fn integration_v2_many_links() {
        let file_data: &[u8] = include_bytes!("../tests/fixtures/v2_many_links.h5");
        let sig_offset = signature::find_signature(file_data).unwrap();
        let sb = Superblock::parse(file_data, sig_offset).unwrap();

        let addr = resolve_path_any(file_data, &sb, "dataset_015").unwrap();
        let hdr = ObjectHeader::parse(file_data, addr as usize, sb.offset_size, sb.length_size)
            .unwrap();
        let (dt, ds, dl) = extract_dataset(file_data, &hdr, sb.offset_size, sb.length_size);
        let raw = data_read::read_raw_data(file_data, &dl, &ds, &dt).unwrap();
        let values = data_read::read_as_f64(&raw, &dt).unwrap();
        assert_eq!(values, vec![15.0]);
    }

    #[test]
    fn integration_resolve_path_any_v1() {
        // Test that resolve_path_any also works for v1 files
        let file_data: &[u8] = include_bytes!("../tests/fixtures/two_groups.h5");
        let sig_offset = signature::find_signature(file_data).unwrap();
        let sb = Superblock::parse(file_data, sig_offset).unwrap();

        let addr = resolve_path_any(file_data, &sb, "group1/values").unwrap();
        let hdr = ObjectHeader::parse(file_data, addr as usize, sb.offset_size, sb.length_size)
            .unwrap();
        let (dt, ds, dl) = extract_dataset(file_data, &hdr, sb.offset_size, sb.length_size);
        let raw = data_read::read_raw_data(file_data, &dl, &ds, &dt).unwrap();
        let values = data_read::read_as_i32(&raw, &dt).unwrap();
        assert_eq!(values, vec![10, 20, 30]);
    }

    #[test]
    fn integration_resolve_path_any_v2() {
        let file_data: &[u8] = include_bytes!("../tests/fixtures/v2_groups.h5");
        let sig_offset = signature::find_signature(file_data).unwrap();
        let sb = Superblock::parse(file_data, sig_offset).unwrap();

        let addr = resolve_path_any(file_data, &sb, "sensors/temperature").unwrap();
        let hdr = ObjectHeader::parse(file_data, addr as usize, sb.offset_size, sb.length_size)
            .unwrap();
        let (dt, ds, dl) = extract_dataset(file_data, &hdr, sb.offset_size, sb.length_size);
        let raw = data_read::read_raw_data(file_data, &dl, &ds, &dt).unwrap();
        let values = data_read::read_as_f64(&raw, &dt).unwrap();
        assert_eq!(values, vec![22.5, 23.1, 21.8]);
    }

    #[test]
    fn path_not_found_v2() {
        let file_data: &[u8] = include_bytes!("../tests/fixtures/v2_groups.h5");
        let sig_offset = signature::find_signature(file_data).unwrap();
        let sb = Superblock::parse(file_data, sig_offset).unwrap();

        let err = resolve_path_any(file_data, &sb, "nonexistent").unwrap_err();
        assert!(matches!(err, FormatError::PathNotFound(_)));
    }
}
