//! HDF5 Superblock parsing for versions 0, 1, 2, and 3.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use byteorder::{ByteOrder, LittleEndian};

use crate::error::FormatError;
use crate::signature::HDF5_SIGNATURE;

/// Parsed HDF5 superblock (all versions).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Superblock {
    /// Superblock version (0–3).
    pub version: u8,
    /// Size of offsets in bytes (2, 4, or 8).
    pub offset_size: u8,
    /// Size of lengths in bytes (2, 4, or 8).
    pub length_size: u8,
    /// File base address.
    pub base_address: u64,
    /// End-of-file address.
    pub eof_address: u64,
    /// Root group object header address (v2/v3) or from symbol table entry (v0/v1).
    pub root_group_address: u64,
    /// Group leaf node K (v0/v1 only).
    pub group_leaf_node_k: Option<u16>,
    /// Group internal node K (v0/v1 only).
    pub group_internal_node_k: Option<u16>,
    /// Indexed storage internal node K (v1 only).
    pub indexed_storage_internal_node_k: Option<u16>,
    /// Free space address (v0/v1 only).
    pub free_space_address: Option<u64>,
    /// Driver info block address (v0/v1 only).
    pub driver_info_address: Option<u64>,
    /// File consistency flags.
    pub consistency_flags: u32,
    /// Superblock extension address (v2/v3 only).
    pub superblock_extension_address: Option<u64>,
    /// CRC32C checksum (v2/v3 only).
    pub checksum: Option<u32>,
}

/// Read an unsigned integer of `size` bytes (LE) from `data` at `pos`.
fn read_offset(data: &[u8], pos: usize, size: u8) -> Result<u64, FormatError> {
    let s = size as usize;
    if pos + s > data.len() {
        return Err(FormatError::UnexpectedEof {
            expected: pos + s,
            available: data.len(),
        });
    }
    let slice = &data[pos..pos + s];
    Ok(match size {
        2 => LittleEndian::read_u16(slice) as u64,
        4 => LittleEndian::read_u32(slice) as u64,
        8 => LittleEndian::read_u64(slice),
        _ => unreachable!(), // validated before calling
    })
}

fn validate_sizes(offset_size: u8, length_size: u8) -> Result<(), FormatError> {
    if !matches!(offset_size, 2 | 4 | 8) {
        return Err(FormatError::InvalidOffsetSize(offset_size));
    }
    if !matches!(length_size, 2 | 4 | 8) {
        return Err(FormatError::InvalidLengthSize(length_size));
    }
    Ok(())
}

fn ensure_len(data: &[u8], needed: usize) -> Result<(), FormatError> {
    if data.len() < needed {
        Err(FormatError::UnexpectedEof {
            expected: needed,
            available: data.len(),
        })
    } else {
        Ok(())
    }
}

impl Superblock {
    /// Serialize this superblock to bytes.
    ///
    /// Always writes v2/v3 format. Computes and appends Jenkins lookup3 checksum.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(48);
        buf.extend_from_slice(&HDF5_SIGNATURE);
        buf.push(self.version);
        buf.push(self.offset_size);
        buf.push(self.length_size);
        buf.push(self.consistency_flags as u8);
        // base_address
        Self::write_offset(&mut buf, self.base_address, self.offset_size);
        // superblock extension address
        let ext_addr = self.superblock_extension_address.unwrap_or(u64::MAX);
        Self::write_offset(&mut buf, ext_addr, self.offset_size);
        // eof_address
        Self::write_offset(&mut buf, self.eof_address, self.offset_size);
        // root_group_address
        Self::write_offset(&mut buf, self.root_group_address, self.offset_size);
        // checksum
        let checksum = crate::checksum::jenkins_lookup3(&buf);
        buf.extend_from_slice(&checksum.to_le_bytes());
        buf
    }

    fn write_offset(buf: &mut Vec<u8>, val: u64, size: u8) {
        match size {
            2 => buf.extend_from_slice(&(val as u16).to_le_bytes()),
            4 => buf.extend_from_slice(&(val as u32).to_le_bytes()),
            8 => buf.extend_from_slice(&val.to_le_bytes()),
            _ => {}
        }
    }

    /// Parse a superblock from `data` starting at `signature_offset`.
    ///
    /// The signature must be present at the given offset.
    pub fn parse(data: &[u8], signature_offset: usize) -> Result<Superblock, FormatError> {
        let d = &data[signature_offset..];
        ensure_len(d, 9)?; // signature(8) + version(1)

        // Verify signature
        if d[..8] != HDF5_SIGNATURE {
            return Err(FormatError::SignatureNotFound);
        }

        let version = d[8];
        match version {
            0 => Self::parse_v0(d),
            1 => Self::parse_v1(d),
            2 | 3 => Self::parse_v2v3(d, version),
            v => Err(FormatError::UnsupportedVersion(v)),
        }
    }

    fn parse_v0(d: &[u8]) -> Result<Superblock, FormatError> {
        // sig(8) + version(1) + free_space_ver(1) + root_grp_ver(1) + reserved(1)
        // + shared_hdr_ver(1) + offset_size(1) + length_size(1) + reserved(1)
        // + group_leaf_k(2) + group_internal_k(2) + consistency_flags(4)
        // = 24 bytes before variable-sized fields
        ensure_len(d, 24)?;

        let offset_size = d[13];
        let length_size = d[14];
        validate_sizes(offset_size, length_size)?;

        let group_leaf_node_k = LittleEndian::read_u16(&d[16..18]);
        let group_internal_node_k = LittleEndian::read_u16(&d[18..20]);
        let consistency_flags = LittleEndian::read_u32(&d[20..24]);

        let os = offset_size as usize;
        // 4 addresses + root symbol table entry
        let var_start = 24;
        let sym_entry_size = os + os + 4 + 4 + 16; // link_name_off, obj_hdr_addr, cache_type, reserved, scratch
        let total = var_start + 4 * os + sym_entry_size;
        ensure_len(d, total)?;

        let mut pos = var_start;
        let base_address = read_offset(d, pos, offset_size)?;
        pos += os;
        let free_space_address = read_offset(d, pos, offset_size)?;
        pos += os;
        let eof_address = read_offset(d, pos, offset_size)?;
        pos += os;
        let driver_info_address = read_offset(d, pos, offset_size)?;
        pos += os;

        // Root symbol table entry
        let _link_name_offset = read_offset(d, pos, offset_size)?;
        pos += os;
        let object_header_addr = read_offset(d, pos, offset_size)?;

        Ok(Superblock {
            version: 0,
            offset_size,
            length_size,
            base_address,
            eof_address,
            root_group_address: object_header_addr,
            group_leaf_node_k: Some(group_leaf_node_k),
            group_internal_node_k: Some(group_internal_node_k),
            indexed_storage_internal_node_k: None,
            free_space_address: Some(free_space_address),
            driver_info_address: Some(driver_info_address),
            consistency_flags,
            superblock_extension_address: None,
            checksum: None,
        })
    }

    fn parse_v1(d: &[u8]) -> Result<Superblock, FormatError> {
        // Same as v0 but adds indexed_storage_internal_node_k(2) + reserved(2) after group_internal_k
        // sig(8) + version(1) + free_space_ver(1) + root_grp_ver(1) + reserved(1)
        // + shared_hdr_ver(1) + offset_size(1) + length_size(1) + reserved(1)
        // + group_leaf_k(2) + group_internal_k(2) + indexed_storage_k(2) + reserved(2)
        // + consistency_flags(4) = 28
        ensure_len(d, 28)?;

        let offset_size = d[13];
        let length_size = d[14];
        validate_sizes(offset_size, length_size)?;

        let group_leaf_node_k = LittleEndian::read_u16(&d[16..18]);
        let group_internal_node_k = LittleEndian::read_u16(&d[18..20]);
        let indexed_storage_internal_node_k = LittleEndian::read_u16(&d[20..22]);
        // d[22..24] reserved
        let consistency_flags = LittleEndian::read_u32(&d[24..28]);

        let os = offset_size as usize;
        let var_start = 28;
        let sym_entry_size = os + os + 4 + 4 + 16;
        let total = var_start + 4 * os + sym_entry_size;
        ensure_len(d, total)?;

        let mut pos = var_start;
        let base_address = read_offset(d, pos, offset_size)?;
        pos += os;
        let free_space_address = read_offset(d, pos, offset_size)?;
        pos += os;
        let eof_address = read_offset(d, pos, offset_size)?;
        pos += os;
        let driver_info_address = read_offset(d, pos, offset_size)?;
        pos += os;

        // Root symbol table entry
        let _link_name_offset = read_offset(d, pos, offset_size)?;
        pos += os;
        let object_header_addr = read_offset(d, pos, offset_size)?;

        Ok(Superblock {
            version: 1,
            offset_size,
            length_size,
            base_address,
            eof_address,
            root_group_address: object_header_addr,
            group_leaf_node_k: Some(group_leaf_node_k),
            group_internal_node_k: Some(group_internal_node_k),
            indexed_storage_internal_node_k: Some(indexed_storage_internal_node_k),
            free_space_address: Some(free_space_address),
            driver_info_address: Some(driver_info_address),
            consistency_flags,
            superblock_extension_address: None,
            checksum: None,
        })
    }

    fn parse_v2v3(d: &[u8], version: u8) -> Result<Superblock, FormatError> {
        // sig(8) + version(1) + offset_size(1) + length_size(1) + consistency_flags(1) = 12
        ensure_len(d, 12)?;

        let offset_size = d[9];
        let length_size = d[10];
        validate_sizes(offset_size, length_size)?;
        let consistency_flags = d[11] as u32;

        let os = offset_size as usize;
        // 4 addresses + checksum(4)
        let total = 12 + 4 * os + 4;
        ensure_len(d, total)?;

        let mut pos = 12;
        let base_address = read_offset(d, pos, offset_size)?;
        pos += os;
        let superblock_extension_address = read_offset(d, pos, offset_size)?;
        pos += os;
        let eof_address = read_offset(d, pos, offset_size)?;
        pos += os;
        let root_group_address = read_offset(d, pos, offset_size)?;
        pos += os;

        let stored_checksum = LittleEndian::read_u32(&d[pos..pos + 4]);

        // Validate checksum if feature enabled
        #[cfg(feature = "checksum")]
        {
            let computed = crate::checksum::jenkins_lookup3(&d[..pos]);
            if computed != stored_checksum {
                return Err(FormatError::ChecksumMismatch {
                    expected: stored_checksum,
                    computed,
                });
            }
        }

        Ok(Superblock {
            version,
            offset_size,
            length_size,
            base_address,
            eof_address,
            root_group_address,
            group_leaf_node_k: None,
            group_internal_node_k: None,
            indexed_storage_internal_node_k: None,
            free_space_address: None,
            driver_info_address: None,
            consistency_flags,
            superblock_extension_address: Some(superblock_extension_address),
            checksum: Some(stored_checksum),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to build a v0 superblock byte buffer with 8-byte offsets.
    fn build_v0_bytes(offset_size: u8) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&HDF5_SIGNATURE); // 0..8
        buf.push(0); // version = 0
        buf.push(0); // free_space_version
        buf.push(0); // root_group_version
        buf.push(0); // reserved
        buf.push(0); // shared_header_version
        buf.push(offset_size); // offset_size
        buf.push(offset_size); // length_size (same for simplicity)
        buf.push(0); // reserved
        buf.extend_from_slice(&4u16.to_le_bytes()); // group_leaf_node_k
        buf.extend_from_slice(&16u16.to_le_bytes()); // group_internal_node_k
        buf.extend_from_slice(&0u32.to_le_bytes()); // consistency_flags
        // base_address
        write_offset(&mut buf, 0, offset_size);
        // free_space_address
        write_offset(&mut buf, 0xFFFFFFFFFFFFFFFF, offset_size);
        // eof_address
        write_offset(&mut buf, 4096, offset_size);
        // driver_info_address
        write_offset(&mut buf, 0xFFFFFFFFFFFFFFFF, offset_size);
        // Root symbol table entry
        write_offset(&mut buf, 0, offset_size); // link_name_offset
        write_offset(&mut buf, 96, offset_size); // object_header_addr (root group)
        buf.extend_from_slice(&0u32.to_le_bytes()); // cache_type
        buf.extend_from_slice(&0u32.to_le_bytes()); // reserved
        buf.extend_from_slice(&[0u8; 16]); // scratch pad
        buf
    }

    fn write_offset(buf: &mut Vec<u8>, val: u64, size: u8) {
        match size {
            2 => buf.extend_from_slice(&(val as u16).to_le_bytes()),
            4 => buf.extend_from_slice(&(val as u32).to_le_bytes()),
            8 => buf.extend_from_slice(&val.to_le_bytes()),
            _ => panic!("bad test offset size"),
        }
    }

    fn build_v1_bytes(offset_size: u8) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&HDF5_SIGNATURE);
        buf.push(1); // version
        buf.push(0); // free_space_version
        buf.push(0); // root_group_version
        buf.push(0); // reserved
        buf.push(0); // shared_header_version
        buf.push(offset_size);
        buf.push(offset_size);
        buf.push(0); // reserved
        buf.extend_from_slice(&4u16.to_le_bytes()); // group_leaf_node_k
        buf.extend_from_slice(&16u16.to_le_bytes()); // group_internal_node_k
        buf.extend_from_slice(&32u16.to_le_bytes()); // indexed_storage_internal_node_k
        buf.extend_from_slice(&0u16.to_le_bytes()); // reserved
        buf.extend_from_slice(&0u32.to_le_bytes()); // consistency_flags
        write_offset(&mut buf, 0, offset_size); // base
        write_offset(&mut buf, 0xFFFFFFFFFFFFFFFF, offset_size); // free space
        write_offset(&mut buf, 8192, offset_size); // eof
        write_offset(&mut buf, 0xFFFFFFFFFFFFFFFF, offset_size); // driver info
        // Root symbol table entry
        write_offset(&mut buf, 0, offset_size);
        write_offset(&mut buf, 200, offset_size); // root group addr
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 16]);
        buf
    }

    fn build_v2_bytes(offset_size: u8, version: u8) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&HDF5_SIGNATURE);
        buf.push(version);
        buf.push(offset_size);
        buf.push(offset_size); // length_size
        buf.push(0); // consistency_flags
        write_offset(&mut buf, 0, offset_size); // base_address
        write_offset(&mut buf, 0xFFFFFFFFFFFFFFFF, offset_size); // superblock ext
        write_offset(&mut buf, 2048, offset_size); // eof
        write_offset(&mut buf, 48, offset_size); // root group obj hdr

        // Compute CRC32C of everything so far
        let checksum = crate::checksum::jenkins_lookup3(&buf);
        buf.extend_from_slice(&checksum.to_le_bytes());
        buf
    }

    #[test]
    fn parse_v0_8byte_offsets() {
        let data = build_v0_bytes(8);
        let sb = Superblock::parse(&data, 0).unwrap();
        assert_eq!(sb.version, 0);
        assert_eq!(sb.offset_size, 8);
        assert_eq!(sb.base_address, 0);
        assert_eq!(sb.eof_address, 4096);
        assert_eq!(sb.root_group_address, 96);
        assert_eq!(sb.group_leaf_node_k, Some(4));
        assert_eq!(sb.group_internal_node_k, Some(16));
        assert_eq!(sb.indexed_storage_internal_node_k, None);
        assert_eq!(sb.free_space_address, Some(0xFFFFFFFFFFFFFFFF));
        assert_eq!(sb.driver_info_address, Some(0xFFFFFFFFFFFFFFFF));
        assert_eq!(sb.checksum, None);
    }

    #[test]
    fn parse_v0_4byte_offsets() {
        let data = build_v0_bytes(4);
        let sb = Superblock::parse(&data, 0).unwrap();
        assert_eq!(sb.version, 0);
        assert_eq!(sb.offset_size, 4);
        assert_eq!(sb.eof_address, 4096);
        assert_eq!(sb.root_group_address, 96);
    }

    #[test]
    fn parse_v1_8byte_offsets() {
        let data = build_v1_bytes(8);
        let sb = Superblock::parse(&data, 0).unwrap();
        assert_eq!(sb.version, 1);
        assert_eq!(sb.offset_size, 8);
        assert_eq!(sb.eof_address, 8192);
        assert_eq!(sb.root_group_address, 200);
        assert_eq!(sb.indexed_storage_internal_node_k, Some(32));
        assert_eq!(sb.group_leaf_node_k, Some(4));
    }

    #[test]
    fn parse_v1_4byte_offsets() {
        let data = build_v1_bytes(4);
        let sb = Superblock::parse(&data, 0).unwrap();
        assert_eq!(sb.version, 1);
        assert_eq!(sb.offset_size, 4);
    }

    #[test]
    fn parse_v2_8byte_offsets() {
        let data = build_v2_bytes(8, 2);
        let sb = Superblock::parse(&data, 0).unwrap();
        assert_eq!(sb.version, 2);
        assert_eq!(sb.offset_size, 8);
        assert_eq!(sb.eof_address, 2048);
        assert_eq!(sb.root_group_address, 48);
        assert!(sb.checksum.is_some());
        assert_eq!(sb.group_leaf_node_k, None);
    }

    #[test]
    fn parse_v2_4byte_offsets() {
        let data = build_v2_bytes(4, 2);
        let sb = Superblock::parse(&data, 0).unwrap();
        assert_eq!(sb.version, 2);
        assert_eq!(sb.offset_size, 4);
    }

    #[test]
    fn parse_v3() {
        let data = build_v2_bytes(8, 3);
        let sb = Superblock::parse(&data, 0).unwrap();
        assert_eq!(sb.version, 3);
    }

    #[test]
    fn checksum_mismatch_v2() {
        let mut data = build_v2_bytes(8, 2);
        // Corrupt the checksum
        let len = data.len();
        data[len - 1] ^= 0xFF;
        let err = Superblock::parse(&data, 0).unwrap_err();
        matches!(err, FormatError::ChecksumMismatch { .. });
    }

    #[test]
    fn unsupported_version() {
        let mut data = vec![0u8; 64];
        data[..8].copy_from_slice(&HDF5_SIGNATURE);
        data[8] = 99;
        assert_eq!(
            Superblock::parse(&data, 0),
            Err(FormatError::UnsupportedVersion(99))
        );
    }

    #[test]
    fn truncated_data() {
        let data = HDF5_SIGNATURE.to_vec(); // Just the signature, no version
        // Only 8 bytes, need at least 9
        assert!(matches!(
            Superblock::parse(&data, 0),
            Err(FormatError::UnexpectedEof { .. })
        ));
    }

    #[test]
    fn truncated_v0() {
        let mut data = vec![0u8; 20]; // Too short for v0
        data[..8].copy_from_slice(&HDF5_SIGNATURE);
        data[8] = 0; // version 0
        data[13] = 8; // offset_size
        data[14] = 8; // length_size
        assert!(matches!(
            Superblock::parse(&data, 0),
            Err(FormatError::UnexpectedEof { .. })
        ));
    }

    #[test]
    fn invalid_offset_size() {
        let mut data = vec![0u8; 64];
        data[..8].copy_from_slice(&HDF5_SIGNATURE);
        data[8] = 0; // version 0
        data[13] = 3; // invalid offset_size
        data[14] = 8;
        assert_eq!(
            Superblock::parse(&data, 0),
            Err(FormatError::InvalidOffsetSize(3))
        );
    }

    #[test]
    fn invalid_length_size() {
        let mut data = vec![0u8; 64];
        data[..8].copy_from_slice(&HDF5_SIGNATURE);
        data[8] = 0;
        data[13] = 8;
        data[14] = 5; // invalid length_size
        assert_eq!(
            Superblock::parse(&data, 0),
            Err(FormatError::InvalidLengthSize(5))
        );
    }

    #[test]
    fn parse_at_nonzero_offset() {
        let mut data = vec![0u8; 1024];
        let v0 = build_v0_bytes(8);
        data[512..512 + v0.len()].copy_from_slice(&v0);
        let sb = Superblock::parse(&data, 512).unwrap();
        assert_eq!(sb.version, 0);
        assert_eq!(sb.root_group_address, 96);
    }

    #[test]
    fn v2_2byte_offsets() {
        let data = build_v2_bytes(2, 2);
        let sb = Superblock::parse(&data, 0).unwrap();
        assert_eq!(sb.offset_size, 2);
        assert_eq!(sb.eof_address, 2048);
    }
}
