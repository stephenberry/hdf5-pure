//! HDF5 Shared Object Header Message resolution.
//!
//! When a header message has its "shared" flag (bit 1 of msg_flags) set,
//! the message data is not the actual message content but a reference
//! to a shared copy stored elsewhere.
//!
//! Shared message reference types:
//! - Type 0: shared in the same object header (not typically used)
//! - Type 1: shared in another object header (version 1-2)
//! - Type 2: shared in the SOHM table (via fractal heap, version 3)
//! - Type 3: shared in another object header (version 3)
//!
//! SOHM table structures:
//! - SharedMessageTable message (0x000F) in superblock extension: version + table_addr + nindexes
//! - SMTB structure at table_addr: per-index metadata (type flags, addresses, etc.)
//! - SMLI list structure: simple list of shared message entries
//! - B-tree v2 type 7: indexed shared message entries

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::btree_v2::{BTreeV2Header, collect_btree_v2_records};
use crate::error::FormatError;
use crate::fractal_heap::FractalHeapHeader;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;

/// Fractal heap ID length for SOHM entries (fixed at 8 bytes).
const FHEAP_ID_LEN: usize = 8;

/// A resolved shared message reference.
#[derive(Debug, Clone)]
pub struct SharedMessageRef {
    /// The type of shared message reference.
    pub ref_type: u8,
    /// Version of the shared message encoding.
    pub version: u8,
    /// Address of the object header containing the shared message (type 1, 3).
    pub object_header_address: Option<u64>,
    /// Fractal heap ID for type 2 (SOHM) references.
    pub heap_id: Option<[u8; FHEAP_ID_LEN]>,
}

/// Parsed Shared Message Table message (type 0x000F from superblock extension).
#[derive(Debug, Clone)]
pub struct SohmTableMessage {
    /// Version of the shared message table message.
    pub version: u8,
    /// Address of the SOHM table (SMTB structure).
    pub table_address: u64,
    /// Number of shared message indexes.
    pub nindexes: u8,
}

/// A single SOHM index entry from the SMTB table.
#[derive(Debug, Clone)]
pub struct SohmIndex {
    /// Index type: 0 = list, 1 = B-tree.
    pub index_type: u8,
    /// Bitmask of message types stored in this index.
    pub mesg_types: u16,
    /// Minimum message size to share.
    pub min_mesg_size: u32,
    /// Maximum messages before converting list to B-tree.
    pub list_max: u16,
    /// Minimum messages before converting B-tree back to list.
    pub btree_min: u16,
    /// Number of messages currently in this index.
    pub num_messages: u16,
    /// Address of the list (SMLI) or B-tree v2 header (BTHD).
    pub index_addr: u64,
    /// Address of the fractal heap for this index.
    pub heap_addr: u64,
}

/// Parsed SOHM table (SMTB structure).
#[derive(Debug, Clone)]
pub struct SohmTable {
    /// The indexes in this table.
    pub indexes: Vec<SohmIndex>,
}

/// A single entry in a SOHM list or B-tree.
#[derive(Debug, Clone)]
pub struct SohmEntry {
    /// Location: 0 = in fractal heap, 1 = in object header.
    pub location: u8,
    /// Hash of the message.
    pub hash: u32,
    /// Fractal heap ID (when location = 0).
    pub heap_id: Option<[u8; FHEAP_ID_LEN]>,
    /// Reference count (when location = 0).
    pub ref_count: Option<u32>,
    /// Message index within OH (when location = 1).
    pub mesg_index: Option<u16>,
    /// Object header address (when location = 1).
    pub oh_addr: Option<u64>,
}

fn read_offset(data: &[u8], pos: usize, size: u8) -> Result<u64, FormatError> {
    let s = size as usize;
    if pos + s > data.len() {
        return Err(FormatError::UnexpectedEof {
            expected: pos + s,
            available: data.len(),
        });
    }
    Ok(match size {
        2 => u16::from_le_bytes([data[pos], data[pos + 1]]) as u64,
        4 => u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as u64,
        8 => u64::from_le_bytes([
            data[pos], data[pos + 1], data[pos + 2], data[pos + 3],
            data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7],
        ]),
        _ => return Err(FormatError::InvalidOffsetSize(size)),
    })
}

fn ensure_len(data: &[u8], pos: usize, needed: usize) -> Result<(), FormatError> {
    match pos.checked_add(needed) {
        Some(end) if end <= data.len() => Ok(()),
        _ => Err(FormatError::UnexpectedEof {
            expected: pos.saturating_add(needed),
            available: data.len(),
        }),
    }
}

/// Check whether a header message has its shared flag set.
pub fn is_shared(msg_flags: u8) -> bool {
    msg_flags & 0x02 != 0
}

/// Parse a shared message reference from the message data.
///
/// When the shared flag is set on a message, the data contains a reference
/// instead of the actual message content.
pub fn parse_shared_ref(
    data: &[u8],
    offset_size: u8,
) -> Result<SharedMessageRef, FormatError> {
    ensure_len(data, 0, 2)?;
    let version = data[0];
    let ref_type = data[1];

    match version {
        1 | 2 => {
            // v1/v2: reserved(6) + address(offset_size)
            let pos = 2 + 6; // skip reserved bytes
            ensure_len(data, pos, offset_size as usize)?;
            let addr = read_offset(data, pos, offset_size)?;
            Ok(SharedMessageRef {
                ref_type,
                version,
                object_header_address: Some(addr),
                heap_id: None,
            })
        }
        3 => {
            match ref_type {
                1 | 3 => {
                    // type 1/3: message in another object header
                    // v3 layout: version(1) + type(1) + address(offset_size)
                    ensure_len(data, 2, offset_size as usize)?;
                    let addr = read_offset(data, 2, offset_size)?;
                    Ok(SharedMessageRef {
                        ref_type,
                        version,
                        object_header_address: Some(addr),
                        heap_id: None,
                    })
                }
                2 => {
                    // type 2: SOHM table (fractal heap ID)
                    ensure_len(data, 2, FHEAP_ID_LEN)?;
                    let mut id = [0u8; FHEAP_ID_LEN];
                    id.copy_from_slice(&data[2..2 + FHEAP_ID_LEN]);
                    Ok(SharedMessageRef {
                        ref_type,
                        version,
                        object_header_address: None,
                        heap_id: Some(id),
                    })
                }
                _ => Err(FormatError::InvalidSharedMessageVersion(ref_type)),
            }
        }
        _ => Err(FormatError::InvalidSharedMessageVersion(version)),
    }
}

// ---- SOHM Table Message (0x000F) parsing ----

/// Parse a Shared Message Table message (type 0x000F) from the superblock extension.
///
/// Format: version(1) + table_address(offset_size) + nindexes(1)
pub fn parse_sohm_table_message(
    data: &[u8],
    offset_size: u8,
) -> Result<SohmTableMessage, FormatError> {
    ensure_len(data, 0, 1)?;
    let version = data[0];
    if version != 0 {
        return Err(FormatError::InvalidSohmTableVersion(version));
    }
    let pos = 1;
    ensure_len(data, pos, offset_size as usize + 1)?;
    let table_address = read_offset(data, pos, offset_size)?;
    let nindexes = data[pos + offset_size as usize];
    Ok(SohmTableMessage {
        version,
        table_address,
        nindexes,
    })
}

// ---- SMTB table parsing ----

/// Parse the SOHM table structure (signature "SMTB") from the file.
///
/// Each index entry: index_type(1) + mesg_types(2) + min_mesg_size(4) +
///   list_max(2) + btree_min(2) + num_messages(2) + index_addr(offset_size) +
///   heap_addr(offset_size)
pub fn parse_sohm_table(
    file_data: &[u8],
    table_addr: usize,
    nindexes: u8,
    offset_size: u8,
) -> Result<SohmTable, FormatError> {
    ensure_len(file_data, table_addr, 4)?;
    if &file_data[table_addr..table_addr + 4] != b"SMTB" {
        return Err(FormatError::InvalidSohmTableSignature);
    }
    let mut pos = table_addr + 4;
    let os = offset_size as usize;
    let entry_size = 1 + 2 + 4 + 2 + 2 + 2 + os + os; // 13 + 2*offset_size

    let mut indexes = Vec::with_capacity(nindexes as usize);
    for _ in 0..nindexes {
        ensure_len(file_data, pos, entry_size)?;
        let index_type = file_data[pos];
        pos += 1;
        let mesg_types = u16::from_le_bytes([file_data[pos], file_data[pos + 1]]);
        pos += 2;
        let min_mesg_size = u32::from_le_bytes([
            file_data[pos], file_data[pos + 1], file_data[pos + 2], file_data[pos + 3],
        ]);
        pos += 4;
        let list_max = u16::from_le_bytes([file_data[pos], file_data[pos + 1]]);
        pos += 2;
        let btree_min = u16::from_le_bytes([file_data[pos], file_data[pos + 1]]);
        pos += 2;
        let num_messages = u16::from_le_bytes([file_data[pos], file_data[pos + 1]]);
        pos += 2;
        let index_addr = read_offset(file_data, pos, offset_size)?;
        pos += os;
        let heap_addr = read_offset(file_data, pos, offset_size)?;
        pos += os;

        indexes.push(SohmIndex {
            index_type,
            mesg_types,
            min_mesg_size,
            list_max,
            btree_min,
            num_messages,
            index_addr,
            heap_addr,
        });
    }
    // 4-byte checksum follows (skip for now)
    Ok(SohmTable { indexes })
}

// ---- SMLI list parsing ----

/// Compute the size of a single SOHM entry in a list or B-tree record.
///
/// Entry: location(1) + hash(4) + max(oh_entry, heap_entry)
/// OH entry: mesg_index(2) + oh_addr(offset_size)
/// Heap entry: heap_id(8) + ref_count(4) = 12
fn sohm_entry_size(offset_size: u8) -> usize {
    let oh_size = 2 + offset_size as usize;
    let heap_size = FHEAP_ID_LEN + 4;
    1 + 4 + oh_size.max(heap_size)
}

/// Parse a single SOHM entry from raw bytes.
fn parse_sohm_entry(data: &[u8], offset_size: u8) -> Result<SohmEntry, FormatError> {
    let entry_sz = sohm_entry_size(offset_size);
    ensure_len(data, 0, entry_sz)?;
    let location = data[0];
    let hash = u32::from_le_bytes([data[1], data[2], data[3], data[4]]);
    let pos = 5;

    if location == 0 {
        // In fractal heap
        ensure_len(data, pos, FHEAP_ID_LEN + 4)?;
        let mut heap_id = [0u8; FHEAP_ID_LEN];
        heap_id.copy_from_slice(&data[pos..pos + FHEAP_ID_LEN]);
        let ref_count = u32::from_le_bytes([
            data[pos + FHEAP_ID_LEN],
            data[pos + FHEAP_ID_LEN + 1],
            data[pos + FHEAP_ID_LEN + 2],
            data[pos + FHEAP_ID_LEN + 3],
        ]);
        Ok(SohmEntry {
            location,
            hash,
            heap_id: Some(heap_id),
            ref_count: Some(ref_count),
            mesg_index: None,
            oh_addr: None,
        })
    } else {
        // In object header
        ensure_len(data, pos, 2 + offset_size as usize)?;
        let mesg_index = u16::from_le_bytes([data[pos], data[pos + 1]]);
        let oh_addr = read_offset(data, pos + 2, offset_size)?;
        Ok(SohmEntry {
            location,
            hash,
            heap_id: None,
            ref_count: None,
            mesg_index: Some(mesg_index),
            oh_addr: Some(oh_addr),
        })
    }
}

/// Parse a SOHM list (signature "SMLI") and return all entries.
pub fn parse_sohm_list(
    file_data: &[u8],
    list_addr: usize,
    num_messages: u16,
    offset_size: u8,
) -> Result<Vec<SohmEntry>, FormatError> {
    ensure_len(file_data, list_addr, 4)?;
    if &file_data[list_addr..list_addr + 4] != b"SMLI" {
        return Err(FormatError::InvalidSohmListSignature);
    }
    let entry_sz = sohm_entry_size(offset_size);
    let mut pos = list_addr + 4;
    let mut entries = Vec::with_capacity(num_messages as usize);
    for _ in 0..num_messages {
        ensure_len(file_data, pos, entry_sz)?;
        let entry = parse_sohm_entry(&file_data[pos..], offset_size)?;
        entries.push(entry);
        pos += entry_sz;
    }
    Ok(entries)
}

/// Parse SOHM entries from a B-tree v2 type 7 index.
pub fn parse_sohm_btree_entries(
    file_data: &[u8],
    btree_addr: usize,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<SohmEntry>, FormatError> {
    let header = BTreeV2Header::parse(file_data, btree_addr, offset_size, length_size)?;
    let records = collect_btree_v2_records(file_data, &header, offset_size, length_size)?;
    let mut entries = Vec::with_capacity(records.len());
    for rec in &records {
        let entry = parse_sohm_entry(&rec.data, offset_size)?;
        entries.push(entry);
    }
    Ok(entries)
}

// ---- SOHM resolution ----

/// Find the SOHM index that handles the given message type.
fn find_index_for_msg_type(table: &SohmTable, msg_type: MessageType) -> Option<&SohmIndex> {
    let type_bit = 1u16 << msg_type.to_u16();
    table.indexes.iter().find(|idx| idx.mesg_types & type_bit != 0)
}

fn is_undefined(val: u64, offset_size: u8) -> bool {
    match offset_size {
        2 => val == 0xFFFF,
        4 => val == 0xFFFF_FFFF,
        8 => val == 0xFFFF_FFFF_FFFF_FFFF,
        _ => false,
    }
}

/// Resolve a type 2 (SOHM) shared message reference.
///
/// Uses the heap ID from the shared ref to read the message data from
/// the fractal heap associated with the matching SOHM index.
pub fn resolve_sohm_message(
    file_data: &[u8],
    heap_id: &[u8; FHEAP_ID_LEN],
    sohm_table: &SohmTable,
    target_msg_type: MessageType,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<u8>, FormatError> {
    let index = find_index_for_msg_type(sohm_table, target_msg_type).ok_or(
        FormatError::InvalidSharedMessageVersion(2),
    )?;

    if is_undefined(index.heap_addr, offset_size) {
        return Err(FormatError::InvalidSharedMessageVersion(2));
    }

    let fh_header = FractalHeapHeader::parse(
        file_data, index.heap_addr as usize, offset_size, length_size,
    )?;
    fh_header.read_managed_object(file_data, heap_id, offset_size)
}

/// Resolve a shared message to its actual message data.
///
/// For type 1/3 (shared in another object header), reads the target object header
/// and finds the message of the specified type.
/// For type 2 (SOHM), uses the fractal heap from the SOHM table.
pub fn resolve_shared_message(
    file_data: &[u8],
    shared_ref: &SharedMessageRef,
    target_msg_type: MessageType,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<u8>, FormatError> {
    resolve_shared_message_with_sohm(
        file_data, shared_ref, target_msg_type, offset_size, length_size, None,
    )
}

/// Resolve a shared message, optionally using a SOHM table for type 2 refs.
pub fn resolve_shared_message_with_sohm(
    file_data: &[u8],
    shared_ref: &SharedMessageRef,
    target_msg_type: MessageType,
    offset_size: u8,
    length_size: u8,
    sohm_table: Option<&SohmTable>,
) -> Result<Vec<u8>, FormatError> {
    match shared_ref.ref_type {
        1 | 3 => {
            let addr = shared_ref.object_header_address.ok_or(
                FormatError::UnexpectedEof {
                    expected: 1,
                    available: 0,
                }
            )?;
            let target_header =
                ObjectHeader::parse(file_data, addr as usize, offset_size, length_size)?;
            for msg in &target_header.messages {
                if msg.msg_type == target_msg_type && !is_shared(msg.flags) {
                    return Ok(msg.data.clone());
                }
            }
            // The message at that OH address is the message itself
            // In many cases with type 1, the entire OH at that address IS the shared message
            // Try returning the first message of any type that isn't Nil
            for msg in &target_header.messages {
                if msg.msg_type == target_msg_type {
                    return Ok(msg.data.clone());
                }
            }
            // Fall back to first non-nil message
            for msg in &target_header.messages {
                if msg.msg_type != MessageType::Nil {
                    return Ok(msg.data.clone());
                }
            }
            Err(FormatError::UnexpectedEof {
                expected: 1,
                available: 0,
            })
        }
        2 => {
            let heap_id = shared_ref.heap_id.as_ref().ok_or(
                FormatError::InvalidSharedMessageVersion(2),
            )?;
            let table = sohm_table.ok_or(
                FormatError::InvalidSharedMessageVersion(2),
            )?;
            resolve_sohm_message(
                file_data, heap_id, table, target_msg_type, offset_size, length_size,
            )
        }
        _ => {
            Err(FormatError::InvalidSharedMessageVersion(shared_ref.ref_type))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_shared_flag() {
        assert!(!is_shared(0x00));
        assert!(!is_shared(0x01));
        assert!(is_shared(0x02));
        assert!(is_shared(0x03));
        assert!(is_shared(0x06));
    }

    #[test]
    fn parse_v3_type1_ref() {
        let mut data = Vec::new();
        data.push(3); // version
        data.push(1); // type 1 = shared in another OH
        data.extend_from_slice(&0x1234u64.to_le_bytes()); // address

        let shared = parse_shared_ref(&data, 8).unwrap();
        assert_eq!(shared.version, 3);
        assert_eq!(shared.ref_type, 1);
        assert_eq!(shared.object_header_address, Some(0x1234));
        assert!(shared.heap_id.is_none());
    }

    #[test]
    fn parse_v3_type3_ref() {
        let mut data = Vec::new();
        data.push(3); // version
        data.push(3); // type 3 = shared in another OH (v3 encoding)
        data.extend_from_slice(&0xABCDu64.to_le_bytes());

        let shared = parse_shared_ref(&data, 8).unwrap();
        assert_eq!(shared.version, 3);
        assert_eq!(shared.ref_type, 3);
        assert_eq!(shared.object_header_address, Some(0xABCD));
    }

    #[test]
    fn parse_v1_ref() {
        let mut data = Vec::new();
        data.push(1); // version
        data.push(0); // type
        data.extend_from_slice(&[0u8; 6]); // reserved
        data.extend_from_slice(&0x5678u64.to_le_bytes());

        let shared = parse_shared_ref(&data, 8).unwrap();
        assert_eq!(shared.version, 1);
        assert_eq!(shared.object_header_address, Some(0x5678));
    }

    #[test]
    fn parse_v2_ref() {
        let mut data = Vec::new();
        data.push(2); // version
        data.push(0); // type
        data.extend_from_slice(&[0u8; 6]); // reserved
        data.extend_from_slice(&0x9000u32.to_le_bytes());

        let shared = parse_shared_ref(&data, 4).unwrap();
        assert_eq!(shared.version, 2);
        assert_eq!(shared.object_header_address, Some(0x9000));
    }

    #[test]
    fn parse_v3_type2_sohm() {
        let mut data = Vec::new();
        data.push(3); // version
        data.push(2); // type 2 = SOHM heap
        data.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD, 0x11, 0x22, 0x33, 0x44]);

        let shared = parse_shared_ref(&data, 8).unwrap();
        assert_eq!(shared.version, 3);
        assert_eq!(shared.ref_type, 2);
        assert_eq!(shared.object_header_address, None);
        assert_eq!(
            shared.heap_id,
            Some([0xAA, 0xBB, 0xCC, 0xDD, 0x11, 0x22, 0x33, 0x44])
        );
    }

    #[test]
    fn parse_v3_type2_too_short() {
        let mut data = Vec::new();
        data.push(3); // version
        data.push(2); // type 2 = SOHM heap
        data.extend_from_slice(&[0xAA, 0xBB]); // only 2 bytes, need 8

        let err = parse_shared_ref(&data, 8).unwrap_err();
        assert!(matches!(err, FormatError::UnexpectedEof { .. }));
    }

    #[test]
    fn invalid_version() {
        let data = vec![99, 0];
        let err = parse_shared_ref(&data, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidSharedMessageVersion(99));
    }

    #[test]
    fn truncated_data() {
        let data = vec![3u8]; // too short
        let err = parse_shared_ref(&data, 8).unwrap_err();
        assert!(matches!(err, FormatError::UnexpectedEof { .. }));
    }

    #[test]
    fn parse_four_byte_offsets() {
        let mut data = Vec::new();
        data.push(3); // version
        data.push(1); // type 1
        data.extend_from_slice(&0x1000u32.to_le_bytes());

        let shared = parse_shared_ref(&data, 4).unwrap();
        assert_eq!(shared.object_header_address, Some(0x1000));
    }

    // ---- SOHM table message tests ----

    #[test]
    fn parse_sohm_table_message_8byte() {
        let mut data = Vec::new();
        data.push(0); // version
        data.extend_from_slice(&0x2000u64.to_le_bytes()); // table address
        data.push(3); // nindexes

        let msg = parse_sohm_table_message(&data, 8).unwrap();
        assert_eq!(msg.version, 0);
        assert_eq!(msg.table_address, 0x2000);
        assert_eq!(msg.nindexes, 3);
    }

    #[test]
    fn parse_sohm_table_message_4byte() {
        let mut data = Vec::new();
        data.push(0); // version
        data.extend_from_slice(&0x1000u32.to_le_bytes()); // table address
        data.push(1); // nindexes

        let msg = parse_sohm_table_message(&data, 4).unwrap();
        assert_eq!(msg.table_address, 0x1000);
        assert_eq!(msg.nindexes, 1);
    }

    #[test]
    fn parse_sohm_table_message_bad_version() {
        let data = vec![1]; // version 1 is invalid
        let err = parse_sohm_table_message(&data, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidSohmTableVersion(1));
    }

    // ---- SMTB table tests ----

    fn build_smtb(indexes: &[SohmIndex], offset_size: u8) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"SMTB");
        for idx in indexes {
            buf.push(idx.index_type);
            buf.extend_from_slice(&idx.mesg_types.to_le_bytes());
            buf.extend_from_slice(&idx.min_mesg_size.to_le_bytes());
            buf.extend_from_slice(&idx.list_max.to_le_bytes());
            buf.extend_from_slice(&idx.btree_min.to_le_bytes());
            buf.extend_from_slice(&idx.num_messages.to_le_bytes());
            match offset_size {
                4 => {
                    buf.extend_from_slice(&(idx.index_addr as u32).to_le_bytes());
                    buf.extend_from_slice(&(idx.heap_addr as u32).to_le_bytes());
                }
                8 => {
                    buf.extend_from_slice(&idx.index_addr.to_le_bytes());
                    buf.extend_from_slice(&idx.heap_addr.to_le_bytes());
                }
                _ => {}
            }
        }
        // Checksum placeholder
        buf.extend_from_slice(&[0u8; 4]);
        buf
    }

    #[test]
    fn parse_smtb_one_index() {
        let indexes = vec![SohmIndex {
            index_type: 0,
            mesg_types: 0x0008, // Datatype
            min_mesg_size: 50,
            list_max: 50,
            btree_min: 40,
            num_messages: 2,
            index_addr: 0x3000,
            heap_addr: 0x4000,
        }];
        let data = build_smtb(&indexes, 8);
        let table = parse_sohm_table(&data, 0, 1, 8).unwrap();
        assert_eq!(table.indexes.len(), 1);
        assert_eq!(table.indexes[0].index_type, 0);
        assert_eq!(table.indexes[0].mesg_types, 0x0008);
        assert_eq!(table.indexes[0].min_mesg_size, 50);
        assert_eq!(table.indexes[0].num_messages, 2);
        assert_eq!(table.indexes[0].index_addr, 0x3000);
        assert_eq!(table.indexes[0].heap_addr, 0x4000);
    }

    #[test]
    fn parse_smtb_two_indexes_4byte() {
        let indexes = vec![
            SohmIndex {
                index_type: 0, mesg_types: 0x0008, min_mesg_size: 50,
                list_max: 50, btree_min: 40, num_messages: 1,
                index_addr: 0x1000, heap_addr: 0x2000,
            },
            SohmIndex {
                index_type: 1, mesg_types: 0x0002, min_mesg_size: 100,
                list_max: 25, btree_min: 15, num_messages: 5,
                index_addr: 0x5000, heap_addr: 0x6000,
            },
        ];
        let data = build_smtb(&indexes, 4);
        let table = parse_sohm_table(&data, 0, 2, 4).unwrap();
        assert_eq!(table.indexes.len(), 2);
        assert_eq!(table.indexes[1].index_type, 1);
        assert_eq!(table.indexes[1].mesg_types, 0x0002);
        assert_eq!(table.indexes[1].num_messages, 5);
        assert_eq!(table.indexes[1].index_addr, 0x5000);
    }

    #[test]
    fn parse_smtb_bad_signature() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"XXXX");
        let err = parse_sohm_table(&data, 0, 1, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidSohmTableSignature);
    }

    // ---- SOHM entry tests ----

    #[test]
    fn parse_heap_entry() {
        let mut data = Vec::new();
        data.push(0); // location = heap
        data.extend_from_slice(&0x12345678u32.to_le_bytes()); // hash
        data.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]); // heap_id
        data.extend_from_slice(&3u32.to_le_bytes()); // ref_count
        // Pad to entry size (entry_size for 8-byte offsets = 1+4+max(10,12)=17)
        // OH size with 8-byte offsets = 2+8=10, heap size = 12, max=12
        // Total entry: 1+4+12=17
        // We wrote 1+4+8+4=17 bytes — no padding needed
        let entry = parse_sohm_entry(&data, 8).unwrap();
        assert_eq!(entry.location, 0);
        assert_eq!(entry.hash, 0x12345678);
        assert_eq!(entry.heap_id, Some([1, 2, 3, 4, 5, 6, 7, 8]));
        assert_eq!(entry.ref_count, Some(3));
        assert!(entry.oh_addr.is_none());
    }

    #[test]
    fn parse_oh_entry() {
        let mut data = Vec::new();
        data.push(1); // location = OH
        data.extend_from_slice(&0xAABBCCDDu32.to_le_bytes()); // hash
        data.extend_from_slice(&5u16.to_le_bytes()); // mesg_index
        data.extend_from_slice(&0x7000u64.to_le_bytes()); // oh_addr
        // OH entry: 2+8=10 bytes, heap entry: 12 bytes, so max=12, need 2 bytes padding
        data.extend_from_slice(&[0u8; 2]);

        let entry = parse_sohm_entry(&data, 8).unwrap();
        assert_eq!(entry.location, 1);
        assert_eq!(entry.hash, 0xAABBCCDD);
        assert_eq!(entry.mesg_index, Some(5));
        assert_eq!(entry.oh_addr, Some(0x7000));
        assert!(entry.heap_id.is_none());
    }

    // ---- SMLI list tests ----

    fn build_smli(entries: &[SohmEntry], offset_size: u8) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"SMLI");
        let entry_sz = sohm_entry_size(offset_size);
        for entry in entries {
            let start = buf.len();
            buf.push(entry.location);
            buf.extend_from_slice(&entry.hash.to_le_bytes());
            if entry.location == 0 {
                buf.extend_from_slice(entry.heap_id.as_ref().unwrap());
                buf.extend_from_slice(&entry.ref_count.unwrap().to_le_bytes());
            } else {
                buf.extend_from_slice(&entry.mesg_index.unwrap().to_le_bytes());
                match offset_size {
                    4 => buf.extend_from_slice(&(entry.oh_addr.unwrap() as u32).to_le_bytes()),
                    8 => buf.extend_from_slice(&entry.oh_addr.unwrap().to_le_bytes()),
                    _ => {}
                }
            }
            // Pad to entry_sz
            let written = buf.len() - start;
            if written < entry_sz {
                buf.resize(buf.len() + entry_sz - written, 0);
            }
        }
        buf.extend_from_slice(&[0u8; 4]); // checksum
        buf
    }

    #[test]
    fn parse_smli_two_entries() {
        let entries = vec![
            SohmEntry {
                location: 0, hash: 0x1111,
                heap_id: Some([10, 20, 30, 40, 50, 60, 70, 80]),
                ref_count: Some(1), mesg_index: None, oh_addr: None,
            },
            SohmEntry {
                location: 0, hash: 0x2222,
                heap_id: Some([11, 21, 31, 41, 51, 61, 71, 81]),
                ref_count: Some(2), mesg_index: None, oh_addr: None,
            },
        ];
        let data = build_smli(&entries, 8);
        let parsed = parse_sohm_list(&data, 0, 2, 8).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].hash, 0x1111);
        assert_eq!(parsed[0].heap_id, Some([10, 20, 30, 40, 50, 60, 70, 80]));
        assert_eq!(parsed[1].hash, 0x2222);
        assert_eq!(parsed[1].ref_count, Some(2));
    }

    #[test]
    fn parse_smli_bad_signature() {
        let data = vec![b'X', b'X', b'X', b'X'];
        let err = parse_sohm_list(&data, 0, 0, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidSohmListSignature);
    }

    // ---- Index lookup tests ----

    #[test]
    fn find_index_for_datatype() {
        let table = SohmTable {
            indexes: vec![
                SohmIndex {
                    index_type: 0,
                    mesg_types: 0x0008, // bit 3 = Datatype (0x0003)
                    min_mesg_size: 50, list_max: 50, btree_min: 40,
                    num_messages: 1, index_addr: 0x1000, heap_addr: 0x2000,
                },
            ],
        };
        let idx = find_index_for_msg_type(&table, MessageType::Datatype);
        assert!(idx.is_some());
        assert_eq!(idx.unwrap().heap_addr, 0x2000);
    }

    #[test]
    fn find_index_no_match() {
        let table = SohmTable {
            indexes: vec![
                SohmIndex {
                    index_type: 0,
                    mesg_types: 0x0002, // bit 1 = Dataspace
                    min_mesg_size: 50, list_max: 50, btree_min: 40,
                    num_messages: 1, index_addr: 0x1000, heap_addr: 0x2000,
                },
            ],
        };
        let idx = find_index_for_msg_type(&table, MessageType::Datatype);
        assert!(idx.is_none());
    }

    #[test]
    fn entry_size_calculations() {
        // With 8-byte offsets: OH=2+8=10, heap=12, entry=1+4+12=17
        assert_eq!(sohm_entry_size(8), 17);
        // With 4-byte offsets: OH=2+4=6, heap=12, entry=1+4+12=17
        assert_eq!(sohm_entry_size(4), 17);
        // With 2-byte offsets: OH=2+2=4, heap=12, entry=1+4+12=17
        assert_eq!(sohm_entry_size(2), 17);
    }
}
