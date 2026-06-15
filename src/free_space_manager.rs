//! HDF5 persistent free-space manager blocks: the Free-space Manager Header
//! (`"FSHD"`) and the Free-space Section Info (`"FSSE"`).
//!
//! When a file persists its free space (`H5Pset_file_space_strategy` with
//! `persist = true`), each free-space manager's tracked sections are written to
//! disk so a later reopen — by this crate or the reference C library — recovers
//! them. The [File Space Info message](crate::file_space_info) in the superblock
//! extension points at the manager headers.
//!
//! This module reads those blocks. Layout (little-endian; `O` = offset size,
//! `L` = length size, both 8 for standard files), verified byte-for-byte against
//! HDF5 1.14.6:
//!
//! ```text
//! FSHD (header):
//!   "FSHD"                4
//!   version              1   = 0
//!   client id            1   = 1 (file free space)
//!   total space tracked  L
//!   total section count  L
//!   serialized count     L
//!   ghost count          L
//!   section class count  2
//!   shrink percent       2
//!   expand percent       2
//!   address space bits   2   = 63 (for 8-byte offsets)
//!   max section size     L   = 2^63 - 1
//!   FSSE address         O
//!   FSSE size used       L
//!   FSSE size allocated  L
//!   checksum             4
//!
//! FSSE (section list):
//!   "FSSE"               4
//!   version              1   = 0
//!   FSHD back-pointer    O
//!   per size-group (ascending size):
//!     section count      count_width
//!     section size       size_width
//!     per section: offset (offset_width) + class id (1, = 0 simple)
//!   checksum             4
//! ```
//!
//! The three serialized widths derive from the header: `offset_width =
//! ceil(addr_space_bits / 8)`, `size_width = enc_size(max_section_size)`, and
//! `count_width = enc_size(total_section_count)`, where `enc_size(v)` is the
//! fewest bytes that hold `v`.

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::convert::TryToUsize;
use crate::error::FormatError;

const FSHD_SIGNATURE: &[u8; 4] = b"FSHD";
const FSSE_SIGNATURE: &[u8; 4] = b"FSSE";

/// One free region: a file offset and its byte length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FreeSection {
    pub addr: u64,
    pub size: u64,
}

/// A parsed Free-space Manager Header (`FSHD`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FsmHeader {
    pub total_space: u64,
    pub total_sections: u64,
    pub addr_space_bits: u16,
    pub max_section_size: u64,
    pub fsse_addr: u64,
    pub fsse_used: u64,
}

/// The fewest bytes needed to hold `value` (HDF5's `H5VM_limit_enc_size`), at
/// least 1.
fn enc_size(value: u64) -> usize {
    let bits = 64 - value.leading_zeros() as usize;
    bits.div_ceil(8).max(1)
}

fn offset_width(addr_space_bits: u16) -> usize {
    (addr_space_bits as usize).div_ceil(8)
}

fn read_uint_le(bytes: &[u8]) -> u64 {
    let mut v = 0u64;
    for (i, &b) in bytes.iter().enumerate() {
        v |= (b as u64) << (8 * i);
    }
    v
}

impl FsmHeader {
    /// Parse an `FSHD` at the start of `data`.
    pub(crate) fn parse(data: &[u8], offset_size: u8) -> Result<FsmHeader, FormatError> {
        let os = offset_size as usize;
        // sig(4) ver(1) client(1) + 4*L + classes/shrink/expand/abits (2 each) +
        // max(L) + fsse_addr(O) + used(L) + alloc(L) + checksum(4)
        let need = 4 + 1 + 1 + 4 * 8 + 2 * 4 + 8 + os + 8 + 8 + 4;
        if data.len() < need {
            return Err(FormatError::UnexpectedEof {
                expected: need,
                available: data.len(),
            });
        }
        if &data[0..4] != FSHD_SIGNATURE {
            return Err(FormatError::InvalidFreeSpaceManager);
        }
        // sig(4) + version(1) + client(1)
        let mut pos = 6;
        let total_space = read_uint_le(&data[pos..pos + 8]);
        pos += 8;
        let total_sections = read_uint_le(&data[pos..pos + 8]);
        pos += 8;
        // skip serialized + ghost section counts
        pos += 16;
        // skip section-class count + shrink% + expand%
        pos += 6;
        let addr_space_bits = u16::from_le_bytes([data[pos], data[pos + 1]]);
        pos += 2;
        let max_section_size = read_uint_le(&data[pos..pos + 8]);
        pos += 8;
        let fsse_addr = read_uint_le(&data[pos..pos + os]);
        pos += os;
        let fsse_used = read_uint_le(&data[pos..pos + 8]);
        Ok(FsmHeader {
            total_space,
            total_sections,
            addr_space_bits,
            max_section_size,
            fsse_addr,
            fsse_used,
        })
    }
}

/// Parse the `FSSE` section list `data` (the whole block, including checksum) for
/// the manager described by `header`, returning its free sections.
pub(crate) fn parse_fsse(
    data: &[u8],
    header: &FsmHeader,
    offset_size: u8,
) -> Result<Vec<FreeSection>, FormatError> {
    let os = offset_size as usize;
    let header_len = 4 + 1 + os; // "FSSE" + version + back-pointer
    if data.len() < header_len + 4 {
        return Err(FormatError::UnexpectedEof {
            expected: header_len + 4,
            available: data.len(),
        });
    }
    if &data[0..4] != FSSE_SIGNATURE {
        return Err(FormatError::InvalidFreeSpaceManager);
    }
    let off_w = offset_width(header.addr_space_bits);
    let size_w = enc_size(header.max_section_size);
    let count_w = enc_size(header.total_sections);
    if off_w == 0 || size_w == 0 || count_w == 0 {
        return Err(FormatError::InvalidFreeSpaceManager);
    }

    let mut pos = header_len;
    let payload_end = data.len() - 4; // exclude checksum
    let total = header.total_sections.to_usize()?;
    let mut sections = Vec::with_capacity(total);
    while sections.len() < total {
        if pos + count_w + size_w > payload_end {
            return Err(FormatError::InvalidFreeSpaceManager);
        }
        let count = read_uint_le(&data[pos..pos + count_w]);
        pos += count_w;
        let size = read_uint_le(&data[pos..pos + size_w]);
        pos += size_w;
        for _ in 0..count {
            if pos + off_w + 1 > payload_end || sections.len() >= total {
                return Err(FormatError::InvalidFreeSpaceManager);
            }
            let addr = read_uint_le(&data[pos..pos + off_w]);
            pos += off_w;
            // class id byte (0 = simple; class data is empty)
            pos += 1;
            sections.push(FreeSection { addr, size });
        }
    }
    Ok(sections)
}

/// Read every persisted free section from the managers named in `manager_addrs`,
/// fetching the `FSHD`/`FSSE` blocks from `data`. `base` is added to every stored
/// address (the file's base address, normally 0). Undefined (`u64::MAX`) manager
/// slots are skipped. Used on reopen to restore a free list.
pub(crate) fn read_persisted_sections(
    data: &[u8],
    manager_addrs: &[u64],
    base: u64,
    offset_size: u8,
) -> Result<Vec<FreeSection>, FormatError> {
    let bad = || FormatError::InvalidFreeSpaceManager;
    let mut sections = Vec::new();
    for &addr in manager_addrs {
        if addr == u64::MAX {
            continue;
        }
        let a = base.checked_add(addr).ok_or_else(bad)?.to_usize()?;
        let header = FsmHeader::parse(data.get(a..).ok_or_else(bad)?, offset_size)?;
        if header.fsse_addr == u64::MAX {
            continue;
        }
        let fa = base
            .checked_add(header.fsse_addr)
            .ok_or_else(bad)?
            .to_usize()?;
        let end = fa
            .checked_add(header.fsse_used.to_usize()?)
            .ok_or_else(bad)?;
        let block = data.get(fa..end).ok_or_else(bad)?;
        sections.extend(parse_fsse(block, &header, offset_size)?);
    }
    Ok(sections)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(hex: &str) -> Vec<u8> {
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect()
    }

    // Fixtures captured from HDF5 1.14.6 (tmp/probe_fsm.py).

    #[test]
    fn parses_c_library_single_section() {
        // manager @619: one 1600-byte free section at offset 2848, FSSE @701.
        let fshd = bytes(
            "46534844000140060000000000000100000000000000010000000000000000000000000000000300500078003f00ffffffffffffff7fbd0200000000000023000000000000002300000000000000ea133710",
        );
        let header = FsmHeader::parse(&fshd, 8).unwrap();
        assert_eq!(header.total_space, 1600);
        assert_eq!(header.total_sections, 1);
        assert_eq!(header.addr_space_bits, 63);
        assert_eq!(header.fsse_addr, 701);
        assert_eq!(header.fsse_used, 35);

        let fsse = bytes("46535345006b02000000000000014006000000000000200b000000000000005c797631");
        let sections = parse_fsse(&fsse, &header, 8).unwrap();
        assert_eq!(
            sections,
            vec![FreeSection {
                addr: 2848,
                size: 1600
            }]
        );
    }

    #[test]
    fn parses_c_library_two_sections() {
        // manager @736: 16 bytes @871 and 893 bytes @1155, FSSE @818.
        let fshd = bytes(
            "4653484400018d030000000000000200000000000000020000000000000000000000000000000300500078003f00ffffffffffffff7f320300000000000035000000000000003500000000000000d681e354",
        );
        let header = FsmHeader::parse(&fshd, 8).unwrap();
        assert_eq!(header.total_space, 909);
        assert_eq!(header.total_sections, 2);
        assert_eq!(header.fsse_addr, 818);
        assert_eq!(header.fsse_used, 53);

        let fsse = bytes(
            "4653534500e002000000000000011000000000000000670300000000000000017d03000000000000830400000000000000910245b2",
        );
        let mut sections = parse_fsse(&fsse, &header, 8).unwrap();
        sections.sort_by_key(|s| s.addr);
        assert_eq!(
            sections,
            vec![
                FreeSection {
                    addr: 871,
                    size: 16
                },
                FreeSection {
                    addr: 1155,
                    size: 893
                },
            ]
        );
        // The section sizes sum to the header's tracked total.
        let total: u64 = sections.iter().map(|s| s.size).sum();
        assert_eq!(header.total_space, total);
    }

    #[test]
    fn read_persisted_sections_follows_managers() {
        // Place the single-section FSHD@619 + FSSE@701 fixtures in a buffer at
        // their real offsets and read them through the manager-address indirection.
        let fshd = bytes(
            "46534844000140060000000000000100000000000000010000000000000000000000000000000300500078003f00ffffffffffffff7fbd0200000000000023000000000000002300000000000000ea133710",
        );
        let fsse = bytes("46535345006b02000000000000014006000000000000200b000000000000005c797631");
        let mut buf = vec![0u8; 701 + fsse.len()];
        buf[619..619 + fshd.len()].copy_from_slice(&fshd);
        buf[701..701 + fsse.len()].copy_from_slice(&fsse);

        let got = read_persisted_sections(&buf, &[619, u64::MAX, u64::MAX], 0, 8).unwrap();
        assert_eq!(
            got,
            vec![FreeSection {
                addr: 2848,
                size: 1600
            }]
        );
        // No defined managers -> no sections.
        assert!(
            read_persisted_sections(&buf, &[u64::MAX], 0, 8)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn enc_size_matches_reference() {
        assert_eq!(enc_size(0), 1);
        assert_eq!(enc_size(255), 1);
        assert_eq!(enc_size(256), 2);
        assert_eq!(enc_size((1 << 63) - 1), 8);
    }

    #[test]
    fn rejects_bad_signature() {
        let mut fshd = bytes(
            "46534844000140060000000000000100000000000000010000000000000000000000000000000300500078003f00ffffffffffffff7fbd0200000000000023000000000000002300000000000000ea133710",
        );
        fshd[0] = b'X';
        assert!(matches!(
            FsmHeader::parse(&fshd, 8),
            Err(FormatError::InvalidFreeSpaceManager)
        ));
    }
}
