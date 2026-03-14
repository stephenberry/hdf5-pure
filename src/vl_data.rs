//! Variable-length data reading (VL strings & VL sequences).
//!
//! VL data elements in HDF5 store their values in the global heap.
//! The raw data for each element contains a global heap ID:
//! `sequence_length(4 LE) + collection_address(offset_size LE) + object_index(4 LE)`.

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

use crate::error::FormatError;
use crate::global_heap::GlobalHeapCollection;

/// A parsed variable-length element reference (global heap ID).
#[derive(Debug, Clone)]
pub struct VlElement {
    /// Length of the VL data.
    pub length: u32,
    /// Address of the global heap collection containing the data.
    pub collection_address: u64,
    /// Index of the object within the collection.
    pub object_index: u32,
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

fn read_offset(data: &[u8], pos: usize, offset_size: u8) -> Result<u64, FormatError> {
    let s = offset_size as usize;
    ensure_len(data, pos, s)?;
    let slice = &data[pos..pos + s];
    Ok(match offset_size {
        2 => u16::from_le_bytes([slice[0], slice[1]]) as u64,
        4 => u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]) as u64,
        8 => u64::from_le_bytes([
            slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
        ]),
        _ => return Err(FormatError::InvalidOffsetSize(offset_size)),
    })
}

/// Parse VL global heap references from raw attribute/dataset data.
pub fn parse_vl_references(
    raw_data: &[u8],
    num_elements: u64,
    offset_size: u8,
) -> Result<Vec<VlElement>, FormatError> {
    let elem_size = 4 + offset_size as usize + 4; // length + address + index
    let total = num_elements as usize * elem_size;
    if raw_data.len() < total {
        return Err(FormatError::UnexpectedEof {
            expected: total,
            available: raw_data.len(),
        });
    }

    let mut elements = Vec::with_capacity(num_elements as usize);
    let mut pos = 0;

    for _ in 0..num_elements {
        let length = u32::from_le_bytes([
            raw_data[pos],
            raw_data[pos + 1],
            raw_data[pos + 2],
            raw_data[pos + 3],
        ]);
        pos += 4;

        let collection_address = read_offset(raw_data, pos, offset_size)?;
        pos += offset_size as usize;

        let object_index = u32::from_le_bytes([
            raw_data[pos],
            raw_data[pos + 1],
            raw_data[pos + 2],
            raw_data[pos + 3],
        ]);
        pos += 4;

        elements.push(VlElement {
            length,
            collection_address,
            object_index,
        });
    }

    Ok(elements)
}

/// Check if an address represents an undefined/null address.
fn is_undefined_address(addr: u64, offset_size: u8) -> bool {
    match offset_size {
        2 => addr == 0xFFFF,
        4 => addr == 0xFFFF_FFFF,
        8 => addr == 0xFFFF_FFFF_FFFF_FFFF,
        _ => false,
    }
}

/// Resolve VL strings from raw data by looking up each element in the global heap.
pub fn read_vl_strings(
    file_data: &[u8],
    raw_data: &[u8],
    num_elements: u64,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<String>, FormatError> {
    let refs = parse_vl_references(raw_data, num_elements, offset_size)?;
    let mut result = Vec::with_capacity(refs.len());

    for vl in &refs {
        if vl.length == 0 && is_undefined_address(vl.collection_address, offset_size) {
            result.push(String::new());
            continue;
        }
        if vl.length == 0 && vl.collection_address == 0 {
            result.push(String::new());
            continue;
        }

        let coll = GlobalHeapCollection::parse(
            file_data,
            vl.collection_address as usize,
            length_size,
        )?;
        let obj = coll
            .get_object(vl.object_index as u16)
            .ok_or(FormatError::GlobalHeapObjectNotFound {
                collection_address: vl.collection_address,
                index: vl.object_index as u16,
            })?;

        // The object data is the raw string bytes
        let len = (vl.length as usize).min(obj.data.len());
        let s = String::from_utf8_lossy(&obj.data[..len]).into_owned();
        result.push(s);
    }

    Ok(result)
}

/// Resolve VL byte sequences from raw data.
pub fn read_vl_bytes(
    file_data: &[u8],
    raw_data: &[u8],
    num_elements: u64,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<Vec<u8>>, FormatError> {
    let refs = parse_vl_references(raw_data, num_elements, offset_size)?;
    let mut result = Vec::with_capacity(refs.len());

    for vl in &refs {
        if vl.length == 0 && (is_undefined_address(vl.collection_address, offset_size) || vl.collection_address == 0) {
            result.push(Vec::new());
            continue;
        }

        let coll = GlobalHeapCollection::parse(
            file_data,
            vl.collection_address as usize,
            length_size,
        )?;
        let obj = coll
            .get_object(vl.object_index as u16)
            .ok_or(FormatError::GlobalHeapObjectNotFound {
                collection_address: vl.collection_address,
                index: vl.object_index as u16,
            })?;

        let len = (vl.length as usize).min(obj.data.len());
        result.push(obj.data[..len].to_vec());
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a global heap collection at given offset in a file buffer.
    fn build_gcol_at(
        file_data: &mut Vec<u8>,
        offset: usize,
        objects: &[(u16, &[u8])], // (index, data)
    ) {
        let length_size = 8usize;

        // Ensure file_data is large enough
        let header_size = 8 + length_size;
        let mut obj_total = 0usize;
        for (_, data) in objects {
            let padded = (data.len() + 7) & !7;
            obj_total += 8 + length_size + padded;
        }
        obj_total += 2; // free space marker
        let collection_size = header_size + obj_total;
        let needed = offset + collection_size;
        if file_data.len() < needed {
            file_data.resize(needed, 0);
        }

        let mut pos = offset;
        // Signature
        file_data[pos..pos + 4].copy_from_slice(b"GCOL");
        file_data[pos + 4] = 1; // version
        // reserved(3) already 0
        pos += 8;
        file_data[pos..pos + 8].copy_from_slice(&(collection_size as u64).to_le_bytes());
        pos += 8;

        for (index, data) in objects {
            file_data[pos..pos + 2].copy_from_slice(&index.to_le_bytes());
            file_data[pos + 2..pos + 4].copy_from_slice(&1u16.to_le_bytes()); // ref_count
            // reserved(4) already 0
            pos += 8;
            file_data[pos..pos + 8].copy_from_slice(&(data.len() as u64).to_le_bytes());
            pos += 8;
            file_data[pos..pos + data.len()].copy_from_slice(data);
            let padded = (data.len() + 7) & !7;
            pos += padded;
        }
        // free space marker
        file_data[pos..pos + 2].copy_from_slice(&0u16.to_le_bytes());
    }

    /// Build VL reference raw data for given strings at a collection address.
    fn build_vl_refs(
        strings: &[&str],
        collection_address: u64,
        start_index: u16,
        offset_size: u8,
    ) -> Vec<u8> {
        let mut raw = Vec::new();
        for (i, s) in strings.iter().enumerate() {
            raw.extend_from_slice(&(s.len() as u32).to_le_bytes());
            match offset_size {
                4 => raw.extend_from_slice(&(collection_address as u32).to_le_bytes()),
                8 => raw.extend_from_slice(&collection_address.to_le_bytes()),
                _ => panic!("unsupported"),
            }
            raw.extend_from_slice(&(start_index as u32 + i as u32).to_le_bytes());
        }
        raw
    }

    #[test]
    fn parse_vl_references_two_elements() {
        let raw = build_vl_refs(&["hello", "world"], 0x1000, 1, 8);
        let refs = parse_vl_references(&raw, 2, 8).unwrap();
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].length, 5);
        assert_eq!(refs[0].collection_address, 0x1000);
        assert_eq!(refs[0].object_index, 1);
        assert_eq!(refs[1].length, 5);
        assert_eq!(refs[1].object_index, 2);
    }

    #[test]
    fn read_vl_strings_from_heap() {
        let gcol_offset = 256usize;
        let mut file_data = vec![0u8; 512];
        build_gcol_at(&mut file_data, gcol_offset, &[
            (1, b"Alice"),
            (2, b"Bob"),
        ]);

        let raw = build_vl_refs(&["Alice", "Bob"], gcol_offset as u64, 1, 8);
        let strings = read_vl_strings(&file_data, &raw, 2, 8, 8).unwrap();
        assert_eq!(strings, vec!["Alice", "Bob"]);
    }

    #[test]
    fn null_vl_element_empty_string() {
        // length=0, address=undefined
        let mut raw = Vec::new();
        raw.extend_from_slice(&0u32.to_le_bytes()); // length=0
        raw.extend_from_slice(&u64::MAX.to_le_bytes()); // undefined address
        raw.extend_from_slice(&0u32.to_le_bytes()); // index

        let file_data = vec![0u8; 16];
        let strings = read_vl_strings(&file_data, &raw, 1, 8, 8).unwrap();
        assert_eq!(strings, vec![""]);
    }

    #[test]
    fn null_vl_element_zero_address() {
        let mut raw = Vec::new();
        raw.extend_from_slice(&0u32.to_le_bytes());
        raw.extend_from_slice(&0u64.to_le_bytes());
        raw.extend_from_slice(&0u32.to_le_bytes());

        let file_data = vec![0u8; 16];
        let strings = read_vl_strings(&file_data, &raw, 1, 8, 8).unwrap();
        assert_eq!(strings, vec![""]);
    }

    #[test]
    fn read_vl_bytes_from_heap() {
        let gcol_offset = 128usize;
        let mut file_data = vec![0u8; 512];
        build_gcol_at(&mut file_data, gcol_offset, &[
            (1, &[0xDE, 0xAD]),
            (2, &[0xBE, 0xEF, 0xCA]),
        ]);

        let _raw = build_vl_refs(&["ab", "abc"], gcol_offset as u64, 1, 8);
        // Fix lengths to match actual byte lengths
        let mut raw_fixed = Vec::new();
        raw_fixed.extend_from_slice(&2u32.to_le_bytes());
        raw_fixed.extend_from_slice(&(gcol_offset as u64).to_le_bytes());
        raw_fixed.extend_from_slice(&1u32.to_le_bytes());
        raw_fixed.extend_from_slice(&3u32.to_le_bytes());
        raw_fixed.extend_from_slice(&(gcol_offset as u64).to_le_bytes());
        raw_fixed.extend_from_slice(&2u32.to_le_bytes());

        let bytes = read_vl_bytes(&file_data, &raw_fixed, 2, 8, 8).unwrap();
        assert_eq!(bytes, vec![vec![0xDE, 0xAD], vec![0xBE, 0xEF, 0xCA]]);
    }

    #[test]
    fn parse_vl_references_truncated_error() {
        let raw = vec![0u8; 10]; // too short for 1 element with offset_size=8
        let err = parse_vl_references(&raw, 1, 8).unwrap_err();
        assert!(matches!(err, FormatError::UnexpectedEof { .. }));
    }
}
