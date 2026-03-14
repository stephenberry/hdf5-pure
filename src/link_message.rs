//! HDF5 Link message parsing (message type 0x0006).

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

use crate::datatype::CharacterSet;
use crate::error::FormatError;

/// The type of a link in an HDF5 v2 group.
#[derive(Debug, Clone, PartialEq)]
pub enum LinkTarget {
    /// Hard link pointing to an object header address.
    Hard { object_header_address: u64 },
    /// Soft (symbolic) link with a target path string.
    Soft { target_path: String },
    /// External link pointing to a file and object path within it.
    External {
        filename: String,
        object_path: String,
    },
}

/// A parsed HDF5 Link message (type 0x0006).
#[derive(Debug, Clone, PartialEq)]
pub struct LinkMessage {
    /// Name of this link.
    pub name: String,
    /// What this link points to.
    pub link_target: LinkTarget,
    /// Creation order, if tracked.
    pub creation_order: Option<u64>,
    /// Character set of the link name.
    pub charset: CharacterSet,
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
        1 => data[pos] as u64,
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
        _ => {
            return Err(FormatError::InvalidOffsetSize(size));
        }
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

impl LinkMessage {
    /// Serialize link message to HDF5 message bytes.
    pub fn serialize(&self, offset_size: u8) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(1); // version

        let name_bytes = self.name.as_bytes();
        let name_len = name_bytes.len();
        let name_size_width: u8 = if name_len <= 0xFF { 1 }
            else if name_len <= 0xFFFF { 2 }
            else { 4 };

        let is_hard = matches!(self.link_target, LinkTarget::Hard { .. });
        let has_link_type = !is_hard;
        let has_creation_order = self.creation_order.is_some();
        let has_charset = self.charset != CharacterSet::Ascii;

        let mut flags: u8 = 0;
        // Bits 0-1: size of name length field
        let size_bits = match name_size_width { 1 => 0u8, 2 => 1, 4 => 2, _ => 3 };
        flags |= size_bits;
        // Bit 2: creation order present
        if has_creation_order { flags |= 0x04; }
        // Bit 3: link type present
        if has_link_type { flags |= 0x08; }
        // Bit 4: charset present
        if has_charset { flags |= 0x10; }
        buf.push(flags);

        if has_link_type {
            match &self.link_target {
                LinkTarget::Soft { .. } => buf.push(1),
                LinkTarget::External { .. } => buf.push(64),
                _ => {}
            }
        }

        if let Some(co) = self.creation_order {
            buf.extend_from_slice(&co.to_le_bytes());
        }

        if has_charset {
            buf.push(match self.charset {
                CharacterSet::Ascii => 0,
                CharacterSet::Utf8 => 1,
            });
        }

        match name_size_width {
            1 => buf.push(name_len as u8),
            2 => buf.extend_from_slice(&(name_len as u16).to_le_bytes()),
            4 => buf.extend_from_slice(&(name_len as u32).to_le_bytes()),
            _ => buf.extend_from_slice(&(name_len as u64).to_le_bytes()),
        }
        buf.extend_from_slice(name_bytes);

        match &self.link_target {
            LinkTarget::Hard { object_header_address } => {
                match offset_size {
                    2 => buf.extend_from_slice(&(*object_header_address as u16).to_le_bytes()),
                    4 => buf.extend_from_slice(&(*object_header_address as u32).to_le_bytes()),
                    8 => buf.extend_from_slice(&object_header_address.to_le_bytes()),
                    _ => {}
                }
            }
            LinkTarget::Soft { target_path } => {
                let path_bytes = target_path.as_bytes();
                buf.extend_from_slice(&(path_bytes.len() as u16).to_le_bytes());
                buf.extend_from_slice(path_bytes);
            }
            LinkTarget::External { filename, object_path } => {
                let mut ext_data = Vec::new();
                ext_data.push(0); // flags
                ext_data.extend_from_slice(filename.as_bytes());
                ext_data.push(0);
                ext_data.extend_from_slice(object_path.as_bytes());
                ext_data.push(0);
                buf.extend_from_slice(&(ext_data.len() as u16).to_le_bytes());
                buf.extend_from_slice(&ext_data);
            }
        }

        buf
    }

    /// Parse a Link message from raw message data.
    ///
    /// `offset_size` is needed for hard link target addresses.
    pub fn parse(data: &[u8], offset_size: u8) -> Result<LinkMessage, FormatError> {
        ensure_len(data, 0, 2)?;

        let version = data[0];
        if version != 1 {
            return Err(FormatError::InvalidLinkVersion(version));
        }

        let flags = data[1];
        // Bits 0-1: size of the name length field (1/2/4/8 bytes)
        let name_size_field_width = match flags & 0x03 {
            0 => 1u8,
            1 => 2,
            2 => 4,
            3 => 8,
            _ => unreachable!(),
        };
        // Bit 2: creation order field present
        let has_creation_order = flags & 0x04 != 0;
        // Bit 3: link type field present
        let has_link_type = flags & 0x08 != 0;
        // Bit 4: link name character set field present
        let has_charset = flags & 0x10 != 0;

        let mut pos = 2;

        // Link type
        let link_type_code = if has_link_type {
            ensure_len(data, pos, 1)?;
            let v = data[pos];
            pos += 1;
            v
        } else {
            0 // hard link
        };

        // Creation order
        let creation_order = if has_creation_order {
            ensure_len(data, pos, 8)?;
            let co = u64::from_le_bytes([
                data[pos],
                data[pos + 1],
                data[pos + 2],
                data[pos + 3],
                data[pos + 4],
                data[pos + 5],
                data[pos + 6],
                data[pos + 7],
            ]);
            pos += 8;
            Some(co)
        } else {
            None
        };

        // Character set
        let charset = if has_charset {
            ensure_len(data, pos, 1)?;
            let cs = data[pos];
            pos += 1;
            match cs {
                0 => CharacterSet::Ascii,
                1 => CharacterSet::Utf8,
                _ => return Err(FormatError::InvalidCharacterSet(cs)),
            }
        } else {
            CharacterSet::Ascii
        };

        // Link name length
        let name_len = read_offset(data, pos, name_size_field_width)? as usize;
        pos += name_size_field_width as usize;

        // Link name
        ensure_len(data, pos, name_len)?;
        let name = String::from_utf8_lossy(&data[pos..pos + name_len]).into_owned();
        pos += name_len;

        // Link target data
        let link_target = match link_type_code {
            0 => {
                // Hard link
                let addr = read_offset(data, pos, offset_size)?;
                LinkTarget::Hard {
                    object_header_address: addr,
                }
            }
            1 => {
                // Soft link
                ensure_len(data, pos, 2)?;
                let soft_len =
                    u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
                pos += 2;
                ensure_len(data, pos, soft_len)?;
                let target_path =
                    String::from_utf8_lossy(&data[pos..pos + soft_len]).into_owned();
                LinkTarget::Soft { target_path }
            }
            64 => {
                // External link
                ensure_len(data, pos, 2)?;
                let ext_len =
                    u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
                pos += 2;
                ensure_len(data, pos, ext_len)?;
                let ext_data = &data[pos..pos + ext_len];
                // External link value: flags(1) + null-terminated filename + null-terminated obj path
                // Skip the flags byte
                let start = if !ext_data.is_empty() { 1 } else { 0 };
                let rest = &ext_data[start..];
                let null1 = rest.iter().position(|&b| b == 0).unwrap_or(rest.len());
                let filename =
                    String::from_utf8_lossy(&rest[..null1]).into_owned();
                let after_null1 = if null1 + 1 < rest.len() {
                    null1 + 1
                } else {
                    rest.len()
                };
                let rest2 = &rest[after_null1..];
                let null2 = rest2.iter().position(|&b| b == 0).unwrap_or(rest2.len());
                let object_path =
                    String::from_utf8_lossy(&rest2[..null2]).into_owned();
                LinkTarget::External {
                    filename,
                    object_path,
                }
            }
            other => return Err(FormatError::InvalidLinkType(other)),
        };

        Ok(LinkMessage {
            name,
            link_target,
            creation_order,
            charset,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a hard link message with given parameters.
    fn build_hard_link(
        name: &str,
        addr: u64,
        offset_size: u8,
        creation_order: Option<u64>,
        charset: Option<u8>,
        name_size_width: u8, // 1, 2, 4
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(1); // version

        let mut flags: u8 = 0;
        // Bits 0-1: name length field size
        let size_bits = match name_size_width {
            1 => 0u8,
            2 => 1,
            4 => 2,
            8 => 3,
            _ => 0,
        };
        flags |= size_bits;
        // Bit 2: creation order present
        if creation_order.is_some() {
            flags |= 0x04;
        }
        // hard link: don't set bit 3 (link type field not present)
        // Bit 4: charset present
        if charset.is_some() {
            flags |= 0x10;
        }
        buf.push(flags);

        // no link_type field for hard links (bit 1 not set)

        if let Some(co) = creation_order {
            buf.extend_from_slice(&co.to_le_bytes());
        }

        if let Some(cs) = charset {
            buf.push(cs);
        }

        // name length
        let name_len = name.len();
        match name_size_width {
            1 => buf.push(name_len as u8),
            2 => buf.extend_from_slice(&(name_len as u16).to_le_bytes()),
            4 => buf.extend_from_slice(&(name_len as u32).to_le_bytes()),
            8 => buf.extend_from_slice(&(name_len as u64).to_le_bytes()),
            _ => {}
        }

        buf.extend_from_slice(name.as_bytes());

        // hard link data: address
        match offset_size {
            4 => buf.extend_from_slice(&(addr as u32).to_le_bytes()),
            8 => buf.extend_from_slice(&addr.to_le_bytes()),
            _ => {}
        }

        buf
    }

    #[test]
    fn hard_link_ascii_no_creation_order() {
        let data = build_hard_link("mydata", 0x1000, 8, None, None, 1);
        let msg = LinkMessage::parse(&data, 8).unwrap();
        assert_eq!(msg.name, "mydata");
        assert_eq!(
            msg.link_target,
            LinkTarget::Hard {
                object_header_address: 0x1000
            }
        );
        assert_eq!(msg.creation_order, None);
        assert_eq!(msg.charset, CharacterSet::Ascii);
    }

    #[test]
    fn hard_link_utf8_with_creation_order() {
        let data = build_hard_link("données", 0x2000, 8, Some(42), Some(1), 1);
        let msg = LinkMessage::parse(&data, 8).unwrap();
        assert_eq!(msg.name, "données");
        assert_eq!(
            msg.link_target,
            LinkTarget::Hard {
                object_header_address: 0x2000
            }
        );
        assert_eq!(msg.creation_order, Some(42));
        assert_eq!(msg.charset, CharacterSet::Utf8);
    }

    #[test]
    fn soft_link() {
        let target = "/group1/dataset";
        let mut data = Vec::new();
        data.push(1); // version
        data.push(0x08); // flags: bit 3 = link type present, name size = 1 byte (bits 0-1 = 0)
        data.push(1); // link type = soft
        data.push(4); // name length = 4
        data.extend_from_slice(b"link");
        data.extend_from_slice(&(target.len() as u16).to_le_bytes());
        data.extend_from_slice(target.as_bytes());

        let msg = LinkMessage::parse(&data, 8).unwrap();
        assert_eq!(msg.name, "link");
        assert_eq!(
            msg.link_target,
            LinkTarget::Soft {
                target_path: target.to_string()
            }
        );
    }

    #[test]
    fn name_length_2bytes() {
        let data = build_hard_link("test", 0x500, 8, None, None, 2);
        let msg = LinkMessage::parse(&data, 8).unwrap();
        assert_eq!(msg.name, "test");
    }

    #[test]
    fn name_length_4bytes() {
        let data = build_hard_link("abcd", 0x600, 8, None, None, 4);
        let msg = LinkMessage::parse(&data, 8).unwrap();
        assert_eq!(msg.name, "abcd");
    }

    #[test]
    fn invalid_version() {
        let data = vec![2, 0, 0, 0]; // version 2
        let err = LinkMessage::parse(&data, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidLinkVersion(2));
    }

    #[test]
    fn invalid_link_type() {
        let mut data = Vec::new();
        data.push(1); // version
        data.push(0x08); // flags: bit 3 = link type present
        data.push(99); // invalid link type
        data.push(1); // name length = 1
        data.push(b'x');
        let err = LinkMessage::parse(&data, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidLinkType(99));
    }
}
