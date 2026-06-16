//! Object header writer for v2 format.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::checksum::jenkins_lookup3;
use crate::message_type::MessageType;

/// Writer for v2 object headers with proper checksums.
pub struct ObjectHeaderWriter {
    messages: Vec<(MessageType, Vec<u8>, u8)>, // (type, data, msg_flags)
}

impl ObjectHeaderWriter {
    /// Create a new empty object header writer.
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    /// Add a message to the header with default flags (0).
    pub fn add_message(&mut self, msg_type: MessageType, data: Vec<u8>) {
        self.messages.push((msg_type, data, 0));
    }

    /// Add a message with specific flags.
    pub fn add_message_with_flags(&mut self, msg_type: MessageType, data: Vec<u8>, flags: u8) {
        self.messages.push((msg_type, data, flags));
    }

    /// Serialize the complete v2 object header (OHDR + messages + checksum).
    pub fn serialize(&self) -> Vec<u8> {
        // Calculate total message bytes: each message has type(1) + size(2) + flags(1) + data
        let msg_bytes_total: usize = self
            .messages
            .iter()
            .map(|(_, data, _)| 4 + data.len())
            .sum();

        // Determine chunk size field width based on msg_bytes_total
        let (flags, chunk_size_width) = if msg_bytes_total <= 255 {
            (0x00u8, 1usize)
        } else if msg_bytes_total <= 65535 {
            (0x01u8, 2)
        } else {
            (0x02u8, 4)
        };

        let mut buf = Vec::new();

        // OHDR signature
        buf.extend_from_slice(b"OHDR");
        // version
        buf.push(2);
        // flags
        buf.push(flags);
        // chunk0 size
        #[expect(
            clippy::cast_possible_truncation,
            reason = "chunk_size_width is the byte width chosen to hold msg_bytes_total, so each arm casts to a width that fits by construction"
        )]
        match chunk_size_width {
            1 => buf.push(msg_bytes_total as u8),
            2 => buf.extend_from_slice(&(msg_bytes_total as u16).to_le_bytes()),
            4 => buf.extend_from_slice(&(msg_bytes_total as u32).to_le_bytes()),
            _ => {}
        }

        // Messages
        for (msg_type, data, msg_flags) in &self.messages {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "message type id is a small enum value written into the 1-byte message-type field of the v2 object header"
            )]
            buf.push(msg_type.to_u16() as u8); // type (1 byte in v2)
            #[expect(
                clippy::cast_possible_truncation,
                reason = "message data length is written into the 2-byte message-size field of the v2 object header"
            )]
            buf.extend_from_slice(&(data.len() as u16).to_le_bytes()); // size (2 bytes)
            buf.push(*msg_flags); // flags
            buf.extend_from_slice(data);
        }

        // Checksum
        let checksum = jenkins_lookup3(&buf);
        buf.extend_from_slice(&checksum.to_le_bytes());

        buf
    }
}

impl Default for ObjectHeaderWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object_header::ObjectHeader;

    #[test]
    fn empty_header_roundtrip() {
        let writer = ObjectHeaderWriter::new();
        let bytes = writer.serialize();
        let hdr = ObjectHeader::parse(&bytes, 0, 8, 8).unwrap();
        assert_eq!(hdr.version, 2);
        assert_eq!(hdr.messages.len(), 0);
    }

    #[test]
    fn two_messages_roundtrip() {
        let mut writer = ObjectHeaderWriter::new();
        writer.add_message(MessageType::Dataspace, vec![1, 2, 3, 4]);
        writer.add_message(MessageType::Datatype, vec![5, 6]);
        let bytes = writer.serialize();
        let hdr = ObjectHeader::parse(&bytes, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 2);
        assert_eq!(hdr.messages[0].msg_type, MessageType::Dataspace);
        assert_eq!(hdr.messages[0].data, vec![1, 2, 3, 4]);
        assert_eq!(hdr.messages[1].msg_type, MessageType::Datatype);
        assert_eq!(hdr.messages[1].data, vec![5, 6]);
    }

    #[test]
    fn large_header_uses_2byte_chunk_size() {
        let mut writer = ObjectHeaderWriter::new();
        // Add a message with >255 bytes of payload
        writer.add_message(MessageType::Datatype, vec![0xAA; 300]);
        let bytes = writer.serialize();
        let hdr = ObjectHeader::parse(&bytes, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 1);
        assert_eq!(hdr.messages[0].data.len(), 300);
    }
}
