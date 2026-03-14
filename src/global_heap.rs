//! HDF5 Global Heap collection parsing.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::error::FormatError;

/// Magic signature for global heap collections.
const GCOL_SIGNATURE: [u8; 4] = [b'G', b'C', b'O', b'L'];

/// A parsed global heap collection.
#[derive(Debug, Clone)]
pub struct GlobalHeapCollection {
    /// Total size of this collection including header.
    pub collection_size: u64,
    /// Objects within this collection.
    pub objects: Vec<GlobalHeapObject>,
}

/// A single object within a global heap collection.
#[derive(Debug, Clone)]
pub struct GlobalHeapObject {
    /// Object index (1-based; 0 is the free space marker).
    pub index: u16,
    /// Reference count.
    pub reference_count: u16,
    /// Object data.
    pub data: Vec<u8>,
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

fn read_length(data: &[u8], offset: usize, length_size: u8) -> Result<u64, FormatError> {
    let s = length_size as usize;
    ensure_len(data, offset, s)?;
    let slice = &data[offset..offset + s];
    Ok(match length_size {
        2 => u16::from_le_bytes([slice[0], slice[1]]) as u64,
        4 => u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]) as u64,
        8 => u64::from_le_bytes([
            slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
        ]),
        _ => return Err(FormatError::InvalidLengthSize(length_size)),
    })
}

/// Round up to next multiple of 8.
fn pad8(x: usize) -> usize {
    (x + 7) & !7
}

impl GlobalHeapCollection {
    /// Parse a global heap collection at the given offset in the file data.
    pub fn parse(
        file_data: &[u8],
        offset: usize,
        length_size: u8,
    ) -> Result<GlobalHeapCollection, FormatError> {
        // signature(4) + version(1) + reserved(3) + collection_size(length_size)
        let header_size = 8 + length_size as usize;
        ensure_len(file_data, offset, header_size)?;

        if file_data[offset..offset + 4] != GCOL_SIGNATURE {
            return Err(FormatError::InvalidGlobalHeapSignature);
        }

        let version = file_data[offset + 4];
        if version != 1 {
            return Err(FormatError::InvalidGlobalHeapVersion(version));
        }

        let collection_size = read_length(file_data, offset + 8, length_size)?;
        let collection_end = offset + collection_size as usize;

        let mut pos = offset + header_size;
        let mut objects = Vec::new();

        // Parse objects until we hit index 0 (free space) or run out of space
        while pos + 2 <= collection_end {
            ensure_len(file_data, pos, 2)?;
            let object_index = u16::from_le_bytes([file_data[pos], file_data[pos + 1]]);

            if object_index == 0 {
                // Free space marker — done
                break;
            }

            // object_index(2) + reference_count(2) + reserved(4) + object_size(length_size)
            let obj_header_size = 8 + length_size as usize;
            ensure_len(file_data, pos, obj_header_size)?;

            let reference_count =
                u16::from_le_bytes([file_data[pos + 2], file_data[pos + 3]]);
            let object_size = read_length(file_data, pos + 8, length_size)? as usize;

            pos += obj_header_size;
            ensure_len(file_data, pos, object_size)?;
            let data = file_data[pos..pos + object_size].to_vec();

            objects.push(GlobalHeapObject {
                index: object_index,
                reference_count,
                data,
            });

            // Advance past data + padding to 8-byte boundary
            pos += pad8(object_size);
        }

        Ok(GlobalHeapCollection {
            collection_size,
            objects,
        })
    }

    /// Get an object by its index.
    pub fn get_object(&self, index: u16) -> Option<&GlobalHeapObject> {
        self.objects.iter().find(|o| o.index == index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a global heap collection with given objects.
    fn build_collection(
        objects: &[(u16, u16, &[u8])], // (index, ref_count, data)
        length_size: u8,
    ) -> Vec<u8> {
        let ls = length_size as usize;

        // Calculate total size
        let header_size = 8 + ls;
        let mut obj_size_total = 0usize;
        for (_, _, data) in objects {
            let obj_header = 8 + ls;
            obj_size_total += obj_header + pad8(data.len());
        }
        // Free space marker (2 bytes for index 0)
        obj_size_total += 2;
        let collection_size = header_size + obj_size_total;

        let mut buf = Vec::new();
        buf.extend_from_slice(&GCOL_SIGNATURE);
        buf.push(1); // version
        buf.extend_from_slice(&[0u8; 3]); // reserved

        // collection_size
        match length_size {
            4 => buf.extend_from_slice(&(collection_size as u32).to_le_bytes()),
            8 => buf.extend_from_slice(&(collection_size as u64).to_le_bytes()),
            _ => panic!("unsupported length_size"),
        }

        // Objects
        for (index, ref_count, data) in objects {
            buf.extend_from_slice(&index.to_le_bytes());
            buf.extend_from_slice(&ref_count.to_le_bytes());
            buf.extend_from_slice(&[0u8; 4]); // reserved
            match length_size {
                4 => buf.extend_from_slice(&(data.len() as u32).to_le_bytes()),
                8 => buf.extend_from_slice(&(data.len() as u64).to_le_bytes()),
                _ => panic!("unsupported"),
            }
            buf.extend_from_slice(data);
            // Pad to 8 bytes
            let padded = pad8(data.len());
            for _ in data.len()..padded {
                buf.push(0);
            }
        }

        // Free space marker
        buf.extend_from_slice(&0u16.to_le_bytes());

        buf
    }

    #[test]
    fn parse_collection_two_objects() {
        let data = build_collection(
            &[
                (1, 1, b"hello"),
                (2, 1, b"world!!!"),
            ],
            8,
        );
        let coll = GlobalHeapCollection::parse(&data, 0, 8).unwrap();
        assert_eq!(coll.objects.len(), 2);
        assert_eq!(coll.objects[0].index, 1);
        assert_eq!(coll.objects[0].data, b"hello");
        assert_eq!(coll.objects[1].index, 2);
        assert_eq!(coll.objects[1].data, b"world!!!");
    }

    #[test]
    fn get_object_by_index() {
        let data = build_collection(
            &[
                (1, 1, b"aaa"),
                (3, 2, b"bbb"),
            ],
            8,
        );
        let coll = GlobalHeapCollection::parse(&data, 0, 8).unwrap();
        let obj = coll.get_object(3).unwrap();
        assert_eq!(obj.data, b"bbb");
        assert_eq!(obj.reference_count, 2);
        assert!(coll.get_object(99).is_none());
    }

    #[test]
    fn free_space_terminates_parsing() {
        // Build collection with free space marker immediately
        let mut data = Vec::new();
        data.extend_from_slice(&GCOL_SIGNATURE);
        data.push(1);
        data.extend_from_slice(&[0u8; 3]);
        let size = 8u64 + 8 + 2; // header + length_size + free space marker
        data.extend_from_slice(&size.to_le_bytes());
        data.extend_from_slice(&0u16.to_le_bytes()); // free space

        let coll = GlobalHeapCollection::parse(&data, 0, 8).unwrap();
        assert_eq!(coll.objects.len(), 0);
    }

    #[test]
    fn invalid_signature_error() {
        let mut data = build_collection(&[(1, 1, b"x")], 8);
        data[0] = b'X'; // corrupt
        let err = GlobalHeapCollection::parse(&data, 0, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidGlobalHeapSignature);
    }

    #[test]
    fn invalid_version_error() {
        let mut data = build_collection(&[(1, 1, b"x")], 8);
        data[4] = 2; // wrong version
        let err = GlobalHeapCollection::parse(&data, 0, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidGlobalHeapVersion(2));
    }

    #[test]
    fn parse_with_4byte_length() {
        let data = build_collection(&[(1, 1, b"test")], 4);
        let coll = GlobalHeapCollection::parse(&data, 0, 4).unwrap();
        assert_eq!(coll.objects.len(), 1);
        assert_eq!(coll.objects[0].data, b"test");
    }
}
