//! HDF5 Group Info message parsing (message type 0x000A).

use crate::error::FormatError;

/// Parsed Group Info message.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupInfoMessage {
    /// Maximum number of links in compact storage before switching to dense.
    pub max_compact: Option<u16>,
    /// Minimum number of links in dense storage before switching back to compact.
    pub min_dense: Option<u16>,
    /// Estimated number of entries (hint).
    pub estimated_num_entries: Option<u16>,
    /// Estimated average link name length (hint).
    pub estimated_name_length: Option<u16>,
}

impl GroupInfoMessage {
    /// Parse a Group Info message from raw message data.
    pub fn parse(data: &[u8]) -> Result<GroupInfoMessage, FormatError> {
        if data.len() < 2 {
            return Err(FormatError::UnexpectedEof {
                expected: 2,
                available: data.len(),
            });
        }

        let version = data[0];
        if version != 0 {
            return Err(FormatError::InvalidGroupInfoVersion(version));
        }

        let flags = data[1];
        let has_link_phase = flags & 0x01 != 0;
        let has_estimated = flags & 0x02 != 0;

        let mut pos = 2;

        let (max_compact, min_dense) = if has_link_phase {
            if pos + 4 > data.len() {
                return Err(FormatError::UnexpectedEof {
                    expected: pos + 4,
                    available: data.len(),
                });
            }
            let mc = u16::from_le_bytes([data[pos], data[pos + 1]]);
            let md = u16::from_le_bytes([data[pos + 2], data[pos + 3]]);
            pos += 4;
            (Some(mc), Some(md))
        } else {
            (None, None)
        };

        let (estimated_num_entries, estimated_name_length) = if has_estimated {
            if pos + 4 > data.len() {
                return Err(FormatError::UnexpectedEof {
                    expected: pos + 4,
                    available: data.len(),
                });
            }
            let ne = u16::from_le_bytes([data[pos], data[pos + 1]]);
            let nl = u16::from_le_bytes([data[pos + 2], data[pos + 3]]);
            (Some(ne), Some(nl))
        } else {
            (None, None)
        };

        Ok(GroupInfoMessage {
            max_compact,
            min_dense,
            estimated_num_entries,
            estimated_name_length,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_link_phase_change() {
        let data = vec![0, 0x01, 8, 0, 6, 0]; // version=0, flags=1, max_compact=8, min_dense=6
        let msg = GroupInfoMessage::parse(&data).unwrap();
        assert_eq!(msg.max_compact, Some(8));
        assert_eq!(msg.min_dense, Some(6));
        assert_eq!(msg.estimated_num_entries, None);
    }

    #[test]
    fn with_estimated_entry_info() {
        let data = vec![0, 0x02, 4, 0, 16, 0]; // flags=2
        let msg = GroupInfoMessage::parse(&data).unwrap();
        assert_eq!(msg.max_compact, None);
        assert_eq!(msg.estimated_num_entries, Some(4));
        assert_eq!(msg.estimated_name_length, Some(16));
    }

    #[test]
    fn both_flags_set() {
        let data = vec![0, 0x03, 8, 0, 6, 0, 10, 0, 32, 0];
        let msg = GroupInfoMessage::parse(&data).unwrap();
        assert_eq!(msg.max_compact, Some(8));
        assert_eq!(msg.min_dense, Some(6));
        assert_eq!(msg.estimated_num_entries, Some(10));
        assert_eq!(msg.estimated_name_length, Some(32));
    }

    #[test]
    fn no_flags() {
        let data = vec![0, 0x00];
        let msg = GroupInfoMessage::parse(&data).unwrap();
        assert_eq!(msg.max_compact, None);
        assert_eq!(msg.min_dense, None);
        assert_eq!(msg.estimated_num_entries, None);
        assert_eq!(msg.estimated_name_length, None);
    }

    #[test]
    fn invalid_version() {
        let data = vec![1, 0];
        let err = GroupInfoMessage::parse(&data).unwrap_err();
        assert_eq!(err, FormatError::InvalidGroupInfoVersion(1));
    }
}
