//! HDF5 Attribute Info message parsing (message type 0x0015).
//!
//! The Attribute Info message describes dense attribute storage: a fractal heap
//! and B-tree v2 indexes for attribute lookup by name or creation order.

use crate::error::FormatError;

/// Parsed Attribute Info message from an object header.
#[derive(Debug, Clone, PartialEq)]
pub struct AttributeInfoMessage {
    /// Maximum creation order index (if creation-order tracking is enabled).
    pub max_creation_index: Option<u16>,
    /// Address of the fractal heap storing attribute messages.
    pub fractal_heap_address: Option<u64>,
    /// Address of B-tree v2 (type 8) for name-ordered attribute index.
    pub btree_name_index_address: Option<u64>,
    /// Address of B-tree v2 (type 9) for creation-order attribute index.
    pub btree_creation_order_address: Option<u64>,
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

fn is_undefined(val: u64, offset_size: u8) -> bool {
    match offset_size {
        2 => val == 0xFFFF,
        4 => val == 0xFFFF_FFFF,
        8 => val == 0xFFFF_FFFF_FFFF_FFFF,
        _ => false,
    }
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

impl AttributeInfoMessage {
    /// Parse an Attribute Info message from raw message data.
    ///
    /// Layout: version(1) + flags(1) + [max_creation_index(2)] +
    ///         fractal_heap_address(os) + btree_name_index(os) +
    ///         [btree_creation_order(os)]
    pub fn parse(data: &[u8], offset_size: u8) -> Result<AttributeInfoMessage, FormatError> {
        ensure_len(data, 0, 2)?;

        let version = data[0];
        if version != 0 {
            return Err(FormatError::InvalidAttributeInfoVersion(version));
        }

        let flags = data[1];
        let has_max_creation_index = flags & 0x01 != 0;
        let has_creation_order_index = flags & 0x02 != 0;

        let mut pos = 2;

        let max_creation_index = if has_max_creation_index {
            ensure_len(data, pos, 2)?;
            let v = u16::from_le_bytes([data[pos], data[pos + 1]]);
            pos += 2;
            Some(v)
        } else {
            None
        };

        let fh_addr = read_offset(data, pos, offset_size)?;
        pos += offset_size as usize;
        let fractal_heap_address = if is_undefined(fh_addr, offset_size) {
            None
        } else {
            Some(fh_addr)
        };

        let btree_addr = read_offset(data, pos, offset_size)?;
        pos += offset_size as usize;
        let btree_name_index_address = if is_undefined(btree_addr, offset_size) {
            None
        } else {
            Some(btree_addr)
        };

        let btree_creation_order_address = if has_creation_order_index {
            let addr = read_offset(data, pos, offset_size)?;
            if is_undefined(addr, offset_size) {
                None
            } else {
                Some(addr)
            }
        } else {
            None
        };

        Ok(AttributeInfoMessage {
            max_creation_index,
            fractal_heap_address,
            btree_name_index_address,
            btree_creation_order_address,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_compact_storage() {
        // version=0, flags=0, fractal_heap=undef, btree=undef
        let mut data = vec![0u8; 2 + 8 + 8];
        data[0] = 0; // version
        data[1] = 0; // flags
        data[2..10].copy_from_slice(&0xFFFF_FFFF_FFFF_FFFFu64.to_le_bytes());
        data[10..18].copy_from_slice(&0xFFFF_FFFF_FFFF_FFFFu64.to_le_bytes());

        let msg = AttributeInfoMessage::parse(&data, 8).unwrap();
        assert_eq!(msg.fractal_heap_address, None);
        assert_eq!(msg.btree_name_index_address, None);
        assert_eq!(msg.max_creation_index, None);
        assert_eq!(msg.btree_creation_order_address, None);
    }

    #[test]
    fn parse_dense_storage() {
        let mut data = Vec::new();
        data.push(0); // version
        data.push(0x00); // flags: no creation order
        data.extend_from_slice(&0x1000u64.to_le_bytes()); // fractal heap
        data.extend_from_slice(&0x2000u64.to_le_bytes()); // btree name

        let msg = AttributeInfoMessage::parse(&data, 8).unwrap();
        assert_eq!(msg.fractal_heap_address, Some(0x1000));
        assert_eq!(msg.btree_name_index_address, Some(0x2000));
        assert_eq!(msg.max_creation_index, None);
        assert_eq!(msg.btree_creation_order_address, None);
    }

    #[test]
    fn parse_dense_with_creation_order() {
        let mut data = Vec::new();
        data.push(0); // version
        data.push(0x03); // flags: max_creation_index + creation_order_index
        data.extend_from_slice(&42u16.to_le_bytes()); // max_creation_index
        data.extend_from_slice(&0x1000u64.to_le_bytes()); // fractal heap
        data.extend_from_slice(&0x2000u64.to_le_bytes()); // btree name
        data.extend_from_slice(&0x3000u64.to_le_bytes()); // btree creation order

        let msg = AttributeInfoMessage::parse(&data, 8).unwrap();
        assert_eq!(msg.max_creation_index, Some(42));
        assert_eq!(msg.fractal_heap_address, Some(0x1000));
        assert_eq!(msg.btree_name_index_address, Some(0x2000));
        assert_eq!(msg.btree_creation_order_address, Some(0x3000));
    }

    #[test]
    fn parse_four_byte_offsets() {
        let mut data = Vec::new();
        data.push(0); // version
        data.push(0x00); // flags
        data.extend_from_slice(&0x100u32.to_le_bytes()); // fractal heap
        data.extend_from_slice(&0x200u32.to_le_bytes()); // btree name

        let msg = AttributeInfoMessage::parse(&data, 4).unwrap();
        assert_eq!(msg.fractal_heap_address, Some(0x100));
        assert_eq!(msg.btree_name_index_address, Some(0x200));
    }

    #[test]
    fn invalid_version() {
        let data = vec![1, 0, 0, 0];
        let err = AttributeInfoMessage::parse(&data, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidAttributeInfoVersion(1));
    }

    #[test]
    fn truncated_data() {
        let data = vec![0u8]; // too short
        let err = AttributeInfoMessage::parse(&data, 8).unwrap_err();
        assert!(matches!(err, FormatError::UnexpectedEof { .. }));
    }

    #[test]
    fn undefined_addresses_four_byte() {
        let mut data = Vec::new();
        data.push(0); // version
        data.push(0x00); // flags
        data.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // fractal heap undef
        data.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // btree undef

        let msg = AttributeInfoMessage::parse(&data, 4).unwrap();
        assert_eq!(msg.fractal_heap_address, None);
        assert_eq!(msg.btree_name_index_address, None);
    }
}
