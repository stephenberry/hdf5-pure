//! HDF5 Object Header parsing (v1 and v2).

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use byteorder::{ByteOrder, LittleEndian};

use crate::error::FormatError;
use crate::message_type::MessageType;

/// OHDR signature for v2 object headers.
const OHDR_SIGNATURE: [u8; 4] = [b'O', b'H', b'D', b'R'];

/// OCHK signature for v2 continuation chunks.
const OCHK_SIGNATURE: [u8; 4] = [b'O', b'C', b'H', b'K'];

/// A single parsed header message.
#[derive(Debug, Clone)]
pub struct HeaderMessage {
    /// The message type.
    pub msg_type: MessageType,
    /// Size of the message data in bytes.
    pub size: usize,
    /// Message flags byte.
    pub flags: u8,
    /// Creation order (v2 only, when tracking is enabled).
    pub creation_order: Option<u16>,
    /// Raw message data bytes.
    pub data: Vec<u8>,
}

/// Parsed HDF5 object header.
#[derive(Debug, Clone)]
pub struct ObjectHeader {
    /// Header version (1 or 2).
    pub version: u8,
    /// All non-NIL messages collected from all chunks.
    pub messages: Vec<HeaderMessage>,
    /// Object reference count (v1 only).
    pub reference_count: Option<u32>,
    /// Object header flags (v2 only; 0 for v1).
    pub flags: u8,
    /// Access time (v2, when flags bit 2 set).
    pub access_time: Option<u32>,
    /// Modification time (v2, when flags bit 2 set).
    pub modification_time: Option<u32>,
    /// Change time (v2, when flags bit 2 set).
    pub change_time: Option<u32>,
    /// Birth time (v2, when flags bit 2 set).
    pub birth_time: Option<u32>,
}

fn ensure_len(data: &[u8], offset: usize, needed: usize) -> Result<(), FormatError> {
    match offset.checked_add(needed) {
        Some(end) if end <= data.len() => Ok(()),
        _ => Err(FormatError::UnexpectedEof {
            expected: offset.saturating_add(needed),
            available: data.len(),
        }),
    }
}

fn read_offset(data: &[u8], pos: usize, size: u8) -> Result<u64, FormatError> {
    let s = size as usize;
    ensure_len(data, pos, s)?;
    let slice = &data[pos..pos + s];
    Ok(match size {
        2 => LittleEndian::read_u16(slice) as u64,
        4 => LittleEndian::read_u32(slice) as u64,
        8 => LittleEndian::read_u64(slice),
        1 => slice[0] as u64,
        _ => {
            return Err(FormatError::InvalidOffsetSize(size));
        }
    })
}

impl ObjectHeader {
    /// Parse an object header at the given offset in the data buffer.
    ///
    /// `offset_size` and `length_size` come from the superblock.
    pub fn parse(
        data: &[u8],
        offset: usize,
        offset_size: u8,
        length_size: u8,
    ) -> Result<ObjectHeader, FormatError> {
        ensure_len(data, offset, 4)?;
        if data[offset..offset + 4] == OHDR_SIGNATURE {
            Self::parse_v2(data, offset, offset_size, length_size)
        } else {
            Self::parse_v1(data, offset, offset_size, length_size)
        }
    }

    fn parse_v1(
        data: &[u8],
        offset: usize,
        offset_size: u8,
        length_size: u8,
    ) -> Result<ObjectHeader, FormatError> {
        // version(1) + reserved(1) + num_messages(2) + ref_count(4) + header_size(4) = 12
        // then pad to 8-byte alignment from start of header
        ensure_len(data, offset, 12)?;

        let version = data[offset];
        if version != 1 {
            return Err(FormatError::InvalidObjectHeaderVersion(version));
        }

        let num_messages = LittleEndian::read_u16(&data[offset + 2..offset + 4]);
        let reference_count = LittleEndian::read_u32(&data[offset + 4..offset + 8]);
        let header_data_size = LittleEndian::read_u32(&data[offset + 8..offset + 12]) as usize;

        // Pad to 8-byte alignment: header prefix is 12 bytes, pad to 16
        let padding = 4; // pad 12-byte prefix to 16-byte alignment
        let msg_start = offset.checked_add(12 + padding).ok_or(FormatError::UnexpectedEof {
            expected: usize::MAX,
            available: data.len(),
        })?;

        ensure_len(data, msg_start, header_data_size)?;

        let mut messages = Vec::new();
        let mut pos = msg_start;
        let msg_end = msg_start.checked_add(header_data_size).ok_or(FormatError::UnexpectedEof {
            expected: usize::MAX,
            available: data.len(),
        })?;

        for _ in 0..num_messages {
            if pos + 8 > msg_end {
                break;
            }
            let msg_type_raw = LittleEndian::read_u16(&data[pos..pos + 2]);
            let msg_data_size = LittleEndian::read_u16(&data[pos + 2..pos + 4]) as usize;
            let msg_flags = data[pos + 4];
            // reserved(3) at pos+5..pos+8
            pos += 8;

            ensure_len(data, pos, msg_data_size)?;
            let msg_type = MessageType::from_u16(msg_type_raw);

            // Check if unknown + must-understand (bit 3 of msg_flags)
            if let MessageType::Unknown(id) = msg_type {
                if msg_flags & 0x08 != 0 {
                    return Err(FormatError::UnsupportedMessage(id));
                }
            }

            if msg_type != MessageType::Nil {
                messages.push(HeaderMessage {
                    msg_type,
                    size: msg_data_size,
                    flags: msg_flags,
                    creation_order: None,
                    data: data[pos..pos + msg_data_size].to_vec(),
                });
            }

            pos += msg_data_size;

            // Follow continuations
            if msg_type == MessageType::ObjectHeaderContinuation {
                let cont_msg_data = &messages.last()
                    .ok_or(FormatError::InvalidObjectHeaderSignature)?.data;
                if cont_msg_data.len() >= (offset_size as usize + length_size as usize) {
                    let cont_offset =
                        read_offset(cont_msg_data, 0, offset_size)? as usize;
                    let cont_length =
                        read_offset(cont_msg_data, offset_size as usize, length_size)?
                            as usize;
                    // Parse continuation block (v1: just raw messages, no signature)
                    let cont_msgs = Self::parse_v1_continuation(
                        data,
                        cont_offset,
                        cont_length,
                        offset_size,
                        length_size,
                        32, // max continuation depth
                    )?;
                    messages.extend(cont_msgs);
                }
            }
        }

        Ok(ObjectHeader {
            version: 1,
            messages,
            reference_count: Some(reference_count),
            flags: 0,
            access_time: None,
            modification_time: None,
            change_time: None,
            birth_time: None,
        })
    }

    fn parse_v1_continuation(
        data: &[u8],
        offset: usize,
        length: usize,
        offset_size: u8,
        length_size: u8,
        depth_remaining: u16,
    ) -> Result<Vec<HeaderMessage>, FormatError> {
        if depth_remaining == 0 {
            return Err(FormatError::NestingDepthExceeded);
        }
        ensure_len(data, offset, length)?;
        let mut messages = Vec::new();
        let mut pos = offset;
        let end = offset.saturating_add(length);

        while pos + 8 <= end {
            let msg_type_raw = LittleEndian::read_u16(&data[pos..pos + 2]);
            let msg_data_size = LittleEndian::read_u16(&data[pos + 2..pos + 4]) as usize;
            let msg_flags = data[pos + 4];
            pos += 8;

            if pos + msg_data_size > end {
                break;
            }

            let msg_type = MessageType::from_u16(msg_type_raw);

            if let MessageType::Unknown(id) = msg_type {
                if msg_flags & 0x08 != 0 {
                    return Err(FormatError::UnsupportedMessage(id));
                }
            }

            if msg_type != MessageType::Nil {
                messages.push(HeaderMessage {
                    msg_type,
                    size: msg_data_size,
                    flags: msg_flags,
                    creation_order: None,
                    data: data[pos..pos + msg_data_size].to_vec(),
                });
            }

            pos += msg_data_size;

            // Recursive continuations
            if msg_type == MessageType::ObjectHeaderContinuation {
                let cont_msg_data = &messages.last()
                    .ok_or(FormatError::InvalidObjectHeaderSignature)?.data;
                if cont_msg_data.len() >= (offset_size as usize + length_size as usize) {
                    let cont_offset =
                        read_offset(cont_msg_data, 0, offset_size)? as usize;
                    let cont_length =
                        read_offset(cont_msg_data, offset_size as usize, length_size)?
                            as usize;
                    let cont_msgs = Self::parse_v1_continuation(
                        data, cont_offset, cont_length, offset_size, length_size,
                        depth_remaining - 1,
                    )?;
                    messages.extend(cont_msgs);
                }
            }
        }

        Ok(messages)
    }

    fn parse_v2(
        data: &[u8],
        offset: usize,
        offset_size: u8,
        length_size: u8,
    ) -> Result<ObjectHeader, FormatError> {
        // signature(4) + version(1) + flags(1) = 6
        ensure_len(data, offset, 6)?;

        let version = data[offset + 4];
        if version != 2 {
            return Err(FormatError::InvalidObjectHeaderVersion(version));
        }
        let flags = data[offset + 5];

        let mut pos = offset + 6;

        // Optional timestamps (flags bit 5)
        let (access_time, modification_time, change_time, birth_time) = if flags & 0x20 != 0 {
            ensure_len(data, pos, 16)?;
            let at = LittleEndian::read_u32(&data[pos..pos + 4]);
            let mt = LittleEndian::read_u32(&data[pos + 4..pos + 8]);
            let ct = LittleEndian::read_u32(&data[pos + 8..pos + 12]);
            let bt = LittleEndian::read_u32(&data[pos + 12..pos + 16]);
            pos += 16;
            (Some(at), Some(mt), Some(ct), Some(bt))
        } else {
            (None, None, None, None)
        };

        // Optional attribute storage thresholds (flags bit 4)
        if flags & 0x10 != 0 {
            ensure_len(data, pos, 4)?;
            // max_compact_attrs(2) + min_dense_attrs(2) — read but don't store for now
            pos += 4;
        }

        // chunk0 size: width depends on flags bits 0-1
        let chunk_size_width = match flags & 0x03 {
            0 => 1u8,
            1 => 2,
            2 => 4,
            3 => 8,
            _ => unreachable!(),
        };
        ensure_len(data, pos, chunk_size_width as usize)?;
        let chunk0_size = read_offset(data, pos, chunk_size_width)? as usize;
        pos += chunk_size_width as usize;

        let chunk0_msg_start = pos;
        let chunk0_msg_end = pos.checked_add(chunk0_size).ok_or(FormatError::UnexpectedEof {
            expected: usize::MAX,
            available: data.len(),
        })?;

        // Validate checksum: from OHDR signature through all messages (before checksum)
        ensure_len(data, chunk0_msg_end, 4)?;
        #[cfg(feature = "checksum")]
        {
            let stored = LittleEndian::read_u32(&data[chunk0_msg_end..chunk0_msg_end + 4]);
            let computed = crate::checksum::jenkins_lookup3(&data[offset..chunk0_msg_end]);
            if computed != stored {
                return Err(FormatError::ChecksumMismatch {
                    expected: stored,
                    computed,
                });
            }
        }

        // Bit 2: attribute creation order tracked → messages include creation order field
        let has_creation_order = flags & 0x04 != 0;

        // Parse messages from chunk0
        let mut messages = Vec::new();
        let mut continuations = Vec::new();
        Self::parse_v2_messages(
            data,
            chunk0_msg_start,
            chunk0_msg_end,
            has_creation_order,
            offset_size,
            length_size,
            &mut messages,
            &mut continuations,
        )?;

        // Follow continuations (limit to prevent cycles in malformed data)
        let mut cont_remaining = 256u16;
        while let Some((cont_offset, cont_length)) = continuations.pop() {
            if cont_remaining == 0 {
                return Err(FormatError::NestingDepthExceeded);
            }
            cont_remaining -= 1;
            Self::parse_v2_continuation(
                data,
                cont_offset,
                cont_length,
                has_creation_order,
                offset_size,
                length_size,
                &mut messages,
                &mut continuations,
            )?;
        }

        Ok(ObjectHeader {
            version: 2,
            messages,
            reference_count: None,
            flags,
            access_time,
            modification_time,
            change_time,
            birth_time,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn parse_v2_messages(
        data: &[u8],
        start: usize,
        end: usize,
        has_creation_order: bool,
        offset_size: u8,
        length_size: u8,
        messages: &mut Vec<HeaderMessage>,
        continuations: &mut Vec<(usize, usize)>,
    ) -> Result<(), FormatError> {
        let msg_header_size = if has_creation_order { 6 } else { 4 };
        let mut pos = start;

        while pos + msg_header_size <= end {
            let msg_type_raw = data[pos] as u16;
            let msg_data_size = LittleEndian::read_u16(&data[pos + 1..pos + 3]) as usize;
            let msg_flags = data[pos + 3];
            let creation_order = if has_creation_order {
                Some(LittleEndian::read_u16(&data[pos + 4..pos + 6]))
            } else {
                None
            };
            pos += msg_header_size;

            if pos + msg_data_size > end {
                // Could be padding at end of chunk
                break;
            }

            let msg_type = MessageType::from_u16(msg_type_raw);

            if let MessageType::Unknown(id) = msg_type {
                if msg_flags & 0x08 != 0 {
                    return Err(FormatError::UnsupportedMessage(id));
                }
            }

            let msg_data = data[pos..pos + msg_data_size].to_vec();

            if msg_type == MessageType::ObjectHeaderContinuation {
                // Parse continuation offset/length from message data
                if msg_data.len() >= (offset_size as usize + length_size as usize) {
                    let cont_off = read_offset(&msg_data, 0, offset_size)? as usize;
                    let cont_len =
                        read_offset(&msg_data, offset_size as usize, length_size)? as usize;
                    continuations.push((cont_off, cont_len));
                }
            } else if msg_type != MessageType::Nil {
                messages.push(HeaderMessage {
                    msg_type,
                    size: msg_data_size,
                    flags: msg_flags,
                    creation_order,
                    data: msg_data,
                });
            }

            pos += msg_data_size;
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn parse_v2_continuation(
        data: &[u8],
        offset: usize,
        length: usize,
        has_creation_order: bool,
        offset_size: u8,
        length_size: u8,
        messages: &mut Vec<HeaderMessage>,
        continuations: &mut Vec<(usize, usize)>,
    ) -> Result<(), FormatError> {
        // OCHK signature(4) + messages + checksum(4)
        ensure_len(data, offset, length)?;
        if length < 8 {
            return Err(FormatError::UnexpectedEof {
                expected: 8,
                available: length,
            });
        }

        ensure_len(data, offset, 4)?;
        if data[offset..offset + 4] != OCHK_SIGNATURE {
            return Err(FormatError::InvalidObjectHeaderSignature);
        }

        let msg_start = offset + 4;
        let checksum_pos = offset + length - 4;

        #[cfg(feature = "checksum")]
        {
            let stored = LittleEndian::read_u32(&data[checksum_pos..checksum_pos + 4]);
            let computed = crate::checksum::jenkins_lookup3(&data[offset..checksum_pos]);
            if computed != stored {
                return Err(FormatError::ChecksumMismatch {
                    expected: stored,
                    computed,
                });
            }
        }

        Self::parse_v2_messages(
            data,
            msg_start,
            checksum_pos,
            has_creation_order,
            offset_size,
            length_size,
            messages,
            continuations,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a v1 object header with given messages
    fn build_v1_header(
        messages: &[(u16, &[u8], u8)], // (type, data, flags)
        offset_size: u8,
        length_size: u8,
    ) -> Vec<u8> {
        let _ = (offset_size, length_size);
        // Calculate total header message data size
        let mut msg_bytes = Vec::new();
        for (mtype, mdata, mflags) in messages {
            msg_bytes.extend_from_slice(&mtype.to_le_bytes()); // type(2)
            msg_bytes.extend_from_slice(&(mdata.len() as u16).to_le_bytes()); // size(2)
            msg_bytes.push(*mflags); // flags(1)
            msg_bytes.extend_from_slice(&[0u8; 3]); // reserved(3)
            msg_bytes.extend_from_slice(mdata); // data
        }

        let mut buf = Vec::new();
        buf.push(1); // version
        buf.push(0); // reserved
        buf.extend_from_slice(&(messages.len() as u16).to_le_bytes()); // num_messages
        buf.extend_from_slice(&1u32.to_le_bytes()); // reference_count
        buf.extend_from_slice(&(msg_bytes.len() as u32).to_le_bytes()); // header_data_size
        // Pad to 8-byte alignment (12 bytes so far, pad 4)
        buf.extend_from_slice(&[0u8; 4]);
        buf.extend_from_slice(&msg_bytes);
        buf
    }

    // Helper: build a v2 object header chunk0 with given messages
    fn build_v2_header(
        flags: u8,
        messages: &[(u8, &[u8], u8)], // (type, data, msg_flags)
        timestamps: Option<(u32, u32, u32, u32)>,
    ) -> Vec<u8> {
        let has_creation_order = flags & 0x04 != 0;
        let has_timestamps = flags & 0x20 != 0;
        let mut buf = Vec::new();
        buf.extend_from_slice(&OHDR_SIGNATURE); // 4
        buf.push(2); // version
        buf.push(flags);

        if has_timestamps {
            if let Some((at, mt, ct, bt)) = timestamps {
                buf.extend_from_slice(&at.to_le_bytes());
                buf.extend_from_slice(&mt.to_le_bytes());
                buf.extend_from_slice(&ct.to_le_bytes());
                buf.extend_from_slice(&bt.to_le_bytes());
            }
        }

        if flags & 0x10 != 0 {
            buf.extend_from_slice(&8u16.to_le_bytes()); // max_compact
            buf.extend_from_slice(&6u16.to_le_bytes()); // min_dense
        }

        // Build message bytes to get chunk size
        let mut msg_bytes = Vec::new();
        for (mtype, mdata, mflags) in messages {
            msg_bytes.push(*mtype); // type(1)
            msg_bytes.extend_from_slice(&(mdata.len() as u16).to_le_bytes()); // size(2)
            msg_bytes.push(*mflags); // flags(1)
            if has_creation_order {
                msg_bytes.extend_from_slice(&0u16.to_le_bytes()); // creation_order(2)
            }
            msg_bytes.extend_from_slice(mdata);
        }

        let chunk_size = msg_bytes.len();
        // Write chunk size based on flags bits 0-1
        match flags & 0x03 {
            0 => buf.push(chunk_size as u8),
            1 => buf.extend_from_slice(&(chunk_size as u16).to_le_bytes()),
            2 => buf.extend_from_slice(&(chunk_size as u32).to_le_bytes()),
            3 => buf.extend_from_slice(&(chunk_size as u64).to_le_bytes()),
            _ => unreachable!(),
        }

        buf.extend_from_slice(&msg_bytes);

        // Checksum (CRC32C of everything from OHDR to here)
        let checksum = crate::checksum::jenkins_lookup3(&buf);
        buf.extend_from_slice(&checksum.to_le_bytes());
        buf
    }

    #[test]
    fn parse_v1_zero_messages() {
        let data = build_v1_header(&[], 8, 8);
        let hdr = ObjectHeader::parse(&data, 0, 8, 8).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.messages.len(), 0);
        assert_eq!(hdr.reference_count, Some(1));
        assert_eq!(hdr.flags, 0);
    }

    #[test]
    fn parse_v1_two_messages() {
        let messages = [
            (0x0001u16, &[1u8, 2, 3, 4][..], 0u8), // Dataspace
            (0x0008, &[5u8, 6][..], 0),              // DataLayout
        ];
        let data = build_v1_header(&messages, 8, 8);
        let hdr = ObjectHeader::parse(&data, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 2);
        assert_eq!(hdr.messages[0].msg_type, MessageType::Dataspace);
        assert_eq!(hdr.messages[0].data, vec![1, 2, 3, 4]);
        assert_eq!(hdr.messages[1].msg_type, MessageType::DataLayout);
        assert_eq!(hdr.messages[1].data, vec![5, 6]);
    }

    #[test]
    fn parse_v1_unknown_message_ok() {
        let messages = [(0x00FFu16, &[0xAA, 0xBB][..], 0u8)];
        let data = build_v1_header(&messages, 8, 8);
        let hdr = ObjectHeader::parse(&data, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 1);
        assert_eq!(hdr.messages[0].msg_type, MessageType::Unknown(0x00FF));
    }

    #[test]
    fn parse_v1_unknown_must_understand_errors() {
        // Bit 3 of msg_flags = must understand
        let messages = [(0x00FFu16, &[0xAA][..], 0x08u8)];
        let data = build_v1_header(&messages, 8, 8);
        let err = ObjectHeader::parse(&data, 0, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::UnsupportedMessage(0x00FF));
    }

    #[test]
    fn parse_v2_no_timestamps_one_message() {
        let data = build_v2_header(0x00, &[(0x01, &[10, 20], 0)], None);
        let hdr = ObjectHeader::parse(&data, 0, 8, 8).unwrap();
        assert_eq!(hdr.version, 2);
        assert_eq!(hdr.flags, 0);
        assert_eq!(hdr.messages.len(), 1);
        assert_eq!(hdr.messages[0].msg_type, MessageType::Dataspace);
        assert_eq!(hdr.messages[0].data, vec![10, 20]);
        assert!(hdr.access_time.is_none());
    }

    #[test]
    fn parse_v2_with_timestamps() {
        let data = build_v2_header(
            0x20,
            &[(0x01, &[1], 0)],
            Some((100, 200, 300, 400)),
        );
        let hdr = ObjectHeader::parse(&data, 0, 8, 8).unwrap();
        assert_eq!(hdr.access_time, Some(100));
        assert_eq!(hdr.modification_time, Some(200));
        assert_eq!(hdr.change_time, Some(300));
        assert_eq!(hdr.birth_time, Some(400));
        assert_eq!(hdr.messages.len(), 1);
        // flags bit 5 = timestamps, but bit 2 not set → no creation order in messages
        assert!(hdr.messages[0].creation_order.is_none());
    }

    #[test]
    fn parse_v2_creation_order() {
        // flags bit 2 enables attribute/message creation order tracking
        // flags bit 5 enables timestamps
        // Use 0x24 = bit 2 + bit 5
        let data = build_v2_header(
            0x24,
            &[(0x03, &[9], 0), (0x05, &[8], 0)],
            Some((0, 0, 0, 0)),
        );
        let hdr = ObjectHeader::parse(&data, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 2);
        assert!(hdr.messages[0].creation_order.is_some());
        assert!(hdr.messages[1].creation_order.is_some());
        assert_eq!(hdr.access_time, Some(0));
    }

    #[test]
    fn parse_v2_checksum_valid() {
        let data = build_v2_header(0x00, &[(0x01, &[1, 2, 3], 0)], None);
        // Should succeed — checksum is valid
        let hdr = ObjectHeader::parse(&data, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 1);
    }

    #[test]
    fn parse_v2_checksum_invalid() {
        let mut data = build_v2_header(0x00, &[(0x01, &[1, 2, 3], 0)], None);
        // Corrupt checksum
        let len = data.len();
        data[len - 1] ^= 0xFF;
        let err = ObjectHeader::parse(&data, 0, 8, 8).unwrap_err();
        assert!(matches!(err, FormatError::ChecksumMismatch { .. }));
    }

    #[test]
    fn parse_v2_nil_padding_skipped() {
        let data = build_v2_header(
            0x00,
            &[
                (0x00, &[0, 0, 0, 0], 0), // NIL
                (0x01, &[42], 0),           // Dataspace
            ],
            None,
        );
        let hdr = ObjectHeader::parse(&data, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 1);
        assert_eq!(hdr.messages[0].msg_type, MessageType::Dataspace);
    }

    #[test]
    fn parse_v2_chunk_size_1byte() {
        // flags bits 0-1 = 0 → 1-byte chunk size
        let data = build_v2_header(0x00, &[(0x01, &[1], 0)], None);
        let hdr = ObjectHeader::parse(&data, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 1);
    }

    #[test]
    fn parse_v2_chunk_size_2byte() {
        let data = build_v2_header(0x01, &[(0x01, &[1], 0)], None);
        let hdr = ObjectHeader::parse(&data, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 1);
    }

    #[test]
    fn parse_v2_chunk_size_4byte() {
        let data = build_v2_header(0x02, &[(0x01, &[1], 0)], None);
        let hdr = ObjectHeader::parse(&data, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 1);
    }

    #[test]
    fn parse_v2_continuation() {
        // Build a continuation chunk (OCHK) at a known offset
        let ochk_offset = 256usize;
        let ochk_msg_type = 0x03u8; // Datatype
        let ochk_msg_data = [0xDE, 0xAD];

        // Build the OCHK chunk
        let mut ochk_buf = Vec::new();
        ochk_buf.extend_from_slice(&OCHK_SIGNATURE);
        ochk_buf.push(ochk_msg_type);
        ochk_buf.extend_from_slice(&(ochk_msg_data.len() as u16).to_le_bytes());
        ochk_buf.push(0); // msg flags
        ochk_buf.extend_from_slice(&ochk_msg_data);
        let checksum = crate::checksum::jenkins_lookup3(&ochk_buf);
        ochk_buf.extend_from_slice(&checksum.to_le_bytes());

        let ochk_length = ochk_buf.len();

        // Build continuation message data: offset(8 LE) + length(8 LE)
        let mut cont_data = Vec::new();
        cont_data.extend_from_slice(&(ochk_offset as u64).to_le_bytes());
        cont_data.extend_from_slice(&(ochk_length as u64).to_le_bytes());

        // Build main header with continuation message + a regular message
        let header = build_v2_header(
            0x00,
            &[
                (0x01, &[42], 0),       // Dataspace
                (0x10, &cont_data, 0),   // Continuation
            ],
            None,
        );

        // Assemble full "file"
        let total_size = ochk_offset + ochk_buf.len();
        let mut file_data = vec![0u8; total_size];
        file_data[..header.len()].copy_from_slice(&header);
        file_data[ochk_offset..ochk_offset + ochk_buf.len()].copy_from_slice(&ochk_buf);

        let hdr = ObjectHeader::parse(&file_data, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 2);
        assert_eq!(hdr.messages[0].msg_type, MessageType::Dataspace);
        assert_eq!(hdr.messages[1].msg_type, MessageType::Datatype);
        assert_eq!(hdr.messages[1].data, vec![0xDE, 0xAD]);
    }

    #[test]
    fn truncated_v1_header() {
        let data = vec![1u8, 0]; // version 1, but too short
        let err = ObjectHeader::parse(&data, 0, 8, 8).unwrap_err();
        assert!(matches!(err, FormatError::UnexpectedEof { .. }));
    }

    #[test]
    fn truncated_v2_header() {
        let data = [b'O', b'H', b'D', b'R', 2]; // signature + version, but no flags
        let err = ObjectHeader::parse(&data, 0, 8, 8).unwrap_err();
        assert!(matches!(err, FormatError::UnexpectedEof { .. }));
    }
}
