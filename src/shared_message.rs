//! HDF5 Shared Object Header Message resolution.
//!
//! When a header message has its "shared" flag (bit 1 of msg_flags) set,
//! the message data is not the actual message content but a reference
//! to a shared copy stored elsewhere.
//!
//! Shared message reference types:
//! - Type 0: shared in the same object header (not typically used)
//! - Type 1: shared in another object header (version 1-2)
//! - Type 2: shared in the SOHM table (via fractal heap, version 3) — parsed
//!   into a reference but not resolved (the SOHM table reader is not implemented)
//! - Type 3: shared in another object header (version 3)

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::convert::TryToUsize;
use crate::error::FormatError;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::source::Source;

/// Fractal heap ID length for SOHM entries (fixed at 8 bytes).
const FHEAP_ID_LEN: usize = 8;

/// A resolved shared message reference.
///
/// `version` and `heap_id` are decoded for on-disk-format completeness; the
/// resolver only follows type 1/3 (object-header) references, so they are not
/// currently read.
#[allow(dead_code)]
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

fn read_offset(data: &[u8], pos: usize, size: u8) -> Result<u64, FormatError> {
    let s = size as usize;
    if s > data.len() || pos > data.len() - s {
        return Err(FormatError::UnexpectedEof {
            expected: pos.saturating_add(s),
            available: data.len(),
        });
    }
    Ok(match size {
        2 => u16::from_le_bytes([data[pos], data[pos + 1]]) as u64,
        4 => u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as u64,
        8 => u64::from_le_bytes([
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3],
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
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
pub fn parse_shared_ref(data: &[u8], offset_size: u8) -> Result<SharedMessageRef, FormatError> {
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

/// Resolve a shared message to its actual message data.
///
/// For type 1/3 (shared in another object header), reads the target object
/// header and finds the message of the specified type. Type 2 (SOHM table)
/// references are not supported.
pub fn resolve_shared_message(
    file_data: &[u8],
    shared_ref: &SharedMessageRef,
    target_msg_type: MessageType,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<u8>, FormatError> {
    match shared_ref.ref_type {
        1 | 3 => {
            let addr = shared_object_header_address(shared_ref)?;
            let target_header =
                ObjectHeader::parse(file_data, addr.to_usize()?, offset_size, length_size)?;
            select_shared_message(&target_header, target_msg_type)
        }
        _ => Err(FormatError::InvalidSharedMessageVersion(
            shared_ref.ref_type,
        )),
    }
}

/// Streaming counterpart of [`resolve_shared_message`]: reads the target object
/// header from a [`Source`] on demand instead of indexing a whole-file slice.
pub fn resolve_shared_message_from_source<S: Source + ?Sized>(
    source: &S,
    shared_ref: &SharedMessageRef,
    target_msg_type: MessageType,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<u8>, FormatError> {
    match shared_ref.ref_type {
        1 | 3 => {
            let addr = shared_object_header_address(shared_ref)?;
            // base_address 0 matches the buffered path's `ObjectHeader::parse`,
            // which treats the shared-message address as absolute.
            let target_header =
                ObjectHeader::parse_from_source(source, addr, offset_size, length_size, 0)?;
            select_shared_message(&target_header, target_msg_type)
        }
        _ => Err(FormatError::InvalidSharedMessageVersion(
            shared_ref.ref_type,
        )),
    }
}

/// Address of the object header holding a type 1/3 shared message.
fn shared_object_header_address(shared_ref: &SharedMessageRef) -> Result<u64, FormatError> {
    shared_ref
        .object_header_address
        .ok_or(FormatError::UnexpectedEof {
            expected: 1,
            available: 0,
        })
}

/// Pick the message of `target_msg_type` out of a resolved target object header.
///
/// Prefers a non-shared message of the requested type; falls back to a shared
/// one of that type, then to the first non-Nil message (with type 1 references
/// the whole target object header often *is* the shared message).
fn select_shared_message(
    target_header: &ObjectHeader,
    target_msg_type: MessageType,
) -> Result<Vec<u8>, FormatError> {
    for msg in &target_header.messages {
        if msg.msg_type == target_msg_type && !is_shared(msg.flags) {
            return Ok(msg.data.clone());
        }
    }
    for msg in &target_header.messages {
        if msg.msg_type == target_msg_type {
            return Ok(msg.data.clone());
        }
    }
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
}
