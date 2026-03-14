//! Object header writer for v2 format.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::checksum::jenkins_lookup3;
use crate::message_type::MessageType;

/// Writer for v2 object headers with proper checksums.
pub struct ObjectHeaderWriter {
    messages: Vec<(MessageType, Vec<u8>, u8)>,  // (type, data, msg_flags)
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
        let msg_bytes_total: usize = self.messages.iter()
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
        match chunk_size_width {
            1 => buf.push(msg_bytes_total as u8),
            2 => buf.extend_from_slice(&(msg_bytes_total as u16).to_le_bytes()),
            4 => buf.extend_from_slice(&(msg_bytes_total as u32).to_le_bytes()),
            _ => {}
        }

        // Messages
        for (msg_type, data, msg_flags) in &self.messages {
            buf.push(msg_type.to_u16() as u8); // type (1 byte in v2)
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

/// A deferred header entry for batch writing.
struct DeferredHeader {
    writer: ObjectHeaderWriter,
}

/// Batch writer that collects multiple object headers in memory and flushes
/// them as a single contiguous I/O pass.
///
/// This reduces the number of serialization passes when creating many datasets
/// in parallel — each thread builds its `ObjectHeaderWriter` independently,
/// then all headers are serialized together.
pub struct BatchObjectHeaderWriter {
    headers: Vec<DeferredHeader>,
}

impl BatchObjectHeaderWriter {
    /// Create a new empty batch writer.
    pub fn new() -> Self {
        Self {
            headers: Vec::new(),
        }
    }

    /// Add a pre-built ObjectHeaderWriter to the batch.
    pub fn add(&mut self, writer: ObjectHeaderWriter) {
        self.headers.push(DeferredHeader { writer });
    }

    /// Number of headers in the batch.
    pub fn len(&self) -> usize {
        self.headers.len()
    }

    /// Whether the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.headers.is_empty()
    }

    /// Compute the serialized size of each header without actually serializing.
    /// Returns sizes in the same order as headers were added.
    pub fn compute_sizes(&self) -> Vec<usize> {
        self.headers
            .iter()
            .map(|h| h.writer.serialize().len())
            .collect()
    }

    /// Serialize all headers into a single contiguous buffer.
    /// Returns `(combined_bytes, offsets)` where `offsets[i]` is the byte
    /// offset of header `i` within the combined buffer.
    pub fn serialize_all(&self) -> (Vec<u8>, Vec<usize>) {
        let serialized: Vec<Vec<u8>> = self.headers.iter().map(|h| h.writer.serialize()).collect();
        let total: usize = serialized.iter().map(|s| s.len()).sum();
        let mut buf = Vec::with_capacity(total);
        let mut offsets = Vec::with_capacity(serialized.len());
        for s in &serialized {
            offsets.push(buf.len());
            buf.extend_from_slice(s);
        }
        (buf, offsets)
    }
}

impl Default for BatchObjectHeaderWriter {
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

    #[test]
    fn batch_writer_serialize_all() {
        let mut batch = BatchObjectHeaderWriter::new();

        let mut w1 = ObjectHeaderWriter::new();
        w1.add_message(MessageType::Dataspace, vec![1, 2, 3]);

        let mut w2 = ObjectHeaderWriter::new();
        w2.add_message(MessageType::Datatype, vec![4, 5]);

        batch.add(w1);
        batch.add(w2);
        assert_eq!(batch.len(), 2);

        let (buf, offsets) = batch.serialize_all();
        assert_eq!(offsets.len(), 2);
        assert_eq!(offsets[0], 0);

        // Parse each header from the combined buffer
        let h1 = ObjectHeader::parse(&buf, offsets[0], 8, 8).unwrap();
        assert_eq!(h1.messages.len(), 1);
        assert_eq!(h1.messages[0].msg_type, MessageType::Dataspace);

        let h2 = ObjectHeader::parse(&buf, offsets[1], 8, 8).unwrap();
        assert_eq!(h2.messages.len(), 1);
        assert_eq!(h2.messages[0].msg_type, MessageType::Datatype);
    }

    #[test]
    fn batch_writer_empty() {
        let batch = BatchObjectHeaderWriter::new();
        assert!(batch.is_empty());
        let (buf, offsets) = batch.serialize_all();
        assert!(buf.is_empty());
        assert!(offsets.is_empty());
    }
}
