//! HDF5 Global Heap collection parsing.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::convert::TryToUsize;
use crate::error::FormatError;
use crate::source::FileSource;

/// Magic signature for global heap collections.
const GCOL_SIGNATURE: [u8; 4] = [b'G', b'C', b'O', b'L'];

/// Metadata index for a global heap collection.
///
/// This stores object locations rather than copying every object payload. VL
/// readers can parse a shared collection once and fetch only the referenced
/// object bytes.
#[derive(Debug, Clone)]
pub struct GlobalHeapIndex {
    /// Object locations within this collection.
    pub objects: Vec<GlobalHeapObjectInfo>,
}

/// Location and size of one object in a global heap collection.
#[derive(Debug, Clone)]
pub struct GlobalHeapObjectInfo {
    /// Object index (1-based; 0 is the free space marker).
    pub index: u16,
    /// Absolute address of the object payload.
    pub data_address: u64,
    /// Object payload size in bytes.
    pub size: u64,
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
#[cfg(test)]
fn pad8(x: usize) -> usize {
    (x + 7) & !7
}

fn pad8_u64(x: u64) -> Result<u64, FormatError> {
    x.checked_add(7)
        .map(|value| value & !7)
        .ok_or(FormatError::OffsetOverflow {
            offset: x,
            length: 7,
        })
}

impl GlobalHeapIndex {
    /// Parse collection metadata from a random-access source without copying
    /// object payloads.
    pub fn parse<S: FileSource + ?Sized>(
        source: &S,
        offset: u64,
        length_size: u8,
    ) -> Result<Self, FormatError> {
        let header_size = 8 + length_size as usize;
        let header = source.read_exact_at(offset, header_size)?;

        if header[..4] != GCOL_SIGNATURE {
            return Err(FormatError::InvalidGlobalHeapSignature);
        }
        let version = header[4];
        if version != 1 {
            return Err(FormatError::InvalidGlobalHeapVersion(version));
        }

        let collection_size = read_length(&header, 8, length_size)?;
        if collection_size < header_size as u64 {
            return Err(FormatError::VlDataError(
                "global heap collection is smaller than its header".into(),
            ));
        }
        let collection_end =
            offset
                .checked_add(collection_size)
                .ok_or(FormatError::OffsetOverflow {
                    offset,
                    length: collection_size,
                })?;
        if collection_end > source.len() {
            return Err(FormatError::UnexpectedEof {
                expected: collection_end.to_usize().unwrap_or(usize::MAX),
                available: source.len().to_usize().unwrap_or(usize::MAX),
            });
        }

        let object_header_size = 8 + length_size as usize;
        let mut pos =
            offset
                .checked_add(header_size as u64)
                .ok_or(FormatError::OffsetOverflow {
                    offset,
                    length: header_size as u64,
                })?;
        let mut objects = Vec::new();

        while pos
            .checked_add(2)
            .is_some_and(|index_end| index_end <= collection_end)
        {
            let index_bytes = source.read_exact_at(pos, 2)?;
            let object_index = u16::from_le_bytes([index_bytes[0], index_bytes[1]]);
            if object_index == 0 {
                break;
            }

            let object_header_end =
                pos.checked_add(object_header_size as u64)
                    .ok_or(FormatError::OffsetOverflow {
                        offset: pos,
                        length: object_header_size as u64,
                    })?;
            if object_header_end > collection_end {
                return Err(FormatError::UnexpectedEof {
                    expected: object_header_end.to_usize().unwrap_or(usize::MAX),
                    available: collection_end.to_usize().unwrap_or(usize::MAX),
                });
            }
            let object_header = source.read_exact_at(pos, object_header_size)?;
            let object_size = read_length(&object_header, 8, length_size)?;
            let data_address = object_header_end;
            let data_end =
                data_address
                    .checked_add(object_size)
                    .ok_or(FormatError::OffsetOverflow {
                        offset: data_address,
                        length: object_size,
                    })?;
            if data_end > collection_end {
                return Err(FormatError::UnexpectedEof {
                    expected: data_end.to_usize().unwrap_or(usize::MAX),
                    available: collection_end.to_usize().unwrap_or(usize::MAX),
                });
            }

            objects.push(GlobalHeapObjectInfo {
                index: object_index,
                data_address,
                size: object_size,
            });

            let padded_size = pad8_u64(object_size)?;
            pos = data_address
                .checked_add(padded_size)
                .ok_or(FormatError::OffsetOverflow {
                    offset: data_address,
                    length: padded_size,
                })?;
        }

        Ok(Self { objects })
    }

    /// Get object metadata by its collection-local index.
    pub fn get_object(&self, index: u16) -> Option<&GlobalHeapObjectInfo> {
        self.objects.iter().find(|object| object.index == index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::BytesSource;

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
        let data = build_collection(&[(1, 1, b"hello"), (2, 1, b"world!!!")], 8);
        let source = BytesSource::new(&data);
        let coll = GlobalHeapIndex::parse(&source, 0, 8).unwrap();
        assert_eq!(coll.objects.len(), 2);
        assert_eq!(coll.objects[0].index, 1);
        assert_eq!(
            source
                .read_exact_at(coll.objects[0].data_address, coll.objects[0].size as usize)
                .unwrap(),
            b"hello"
        );
        assert_eq!(coll.objects[1].index, 2);
        assert_eq!(
            source
                .read_exact_at(coll.objects[1].data_address, coll.objects[1].size as usize)
                .unwrap(),
            b"world!!!"
        );
    }

    #[test]
    fn get_object_by_index() {
        let data = build_collection(&[(1, 1, b"aaa"), (3, 2, b"bbb")], 8);
        let source = BytesSource::new(&data);
        let coll = GlobalHeapIndex::parse(&source, 0, 8).unwrap();
        let obj = coll.get_object(3).unwrap();
        assert_eq!(
            source
                .read_exact_at(obj.data_address, obj.size as usize)
                .unwrap(),
            b"bbb"
        );
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

        let coll = GlobalHeapIndex::parse(&BytesSource::new(&data), 0, 8).unwrap();
        assert_eq!(coll.objects.len(), 0);
    }

    #[test]
    fn invalid_signature_error() {
        let mut data = build_collection(&[(1, 1, b"x")], 8);
        data[0] = b'X'; // corrupt
        let err = GlobalHeapIndex::parse(&BytesSource::new(&data), 0, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidGlobalHeapSignature);
    }

    #[test]
    fn invalid_version_error() {
        let mut data = build_collection(&[(1, 1, b"x")], 8);
        data[4] = 2; // wrong version
        let err = GlobalHeapIndex::parse(&BytesSource::new(&data), 0, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidGlobalHeapVersion(2));
    }

    #[test]
    fn object_header_cannot_cross_collection_boundary() {
        let mut data = build_collection(&[(1, 1, b"x")], 8);
        let truncated_collection_size = 8u64 + 8 + 2;
        data[8..16].copy_from_slice(&truncated_collection_size.to_le_bytes());

        let err = GlobalHeapIndex::parse(&BytesSource::new(&data), 0, 8).unwrap_err();
        assert!(matches!(err, FormatError::UnexpectedEof { .. }));
    }

    #[test]
    fn parse_with_4byte_length() {
        let data = build_collection(&[(1, 1, b"test")], 4);
        let source = BytesSource::new(&data);
        let coll = GlobalHeapIndex::parse(&source, 0, 4).unwrap();
        assert_eq!(coll.objects.len(), 1);
        let object = &coll.objects[0];
        assert_eq!(
            source
                .read_exact_at(object.data_address, object.size as usize)
                .unwrap(),
            b"test"
        );
    }
}
