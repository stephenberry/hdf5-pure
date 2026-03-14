//! HDF5 Local Heap parsing.

#[cfg(not(feature = "std"))]
use alloc::string::String;

use crate::error::FormatError;

/// Parsed HDF5 Local Heap header.
#[derive(Debug, Clone)]
pub struct LocalHeap {
    /// Size of the data segment in bytes.
    pub data_segment_size: u64,
    /// Offset of the free list head within the data segment.
    pub free_list_head_offset: u64,
    /// File address of the data segment.
    pub data_segment_address: u64,
}

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
        2 => u16::from_le_bytes([slice[0], slice[1]]) as u64,
        4 => u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]) as u64,
        8 => u64::from_le_bytes([
            slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
        ]),
        _ => return Err(FormatError::InvalidOffsetSize(size)),
    })
}

impl LocalHeap {
    /// Parse a local heap header at the given offset in the file data.
    pub fn parse(
        file_data: &[u8],
        offset: usize,
        offset_size: u8,
        length_size: u8,
    ) -> Result<LocalHeap, FormatError> {
        // signature(4) + version(1) + reserved(3) = 8, then length_size*2 + offset_size
        let ls = length_size as usize;
        let os = offset_size as usize;
        let total = 8 + ls * 2 + os;
        if offset + total > file_data.len() {
            return Err(FormatError::UnexpectedEof {
                expected: offset + total,
                available: file_data.len(),
            });
        }

        if &file_data[offset..offset + 4] != b"HEAP" {
            return Err(FormatError::InvalidLocalHeapSignature);
        }

        let version = file_data[offset + 4];
        if version != 0 {
            return Err(FormatError::InvalidLocalHeapVersion(version));
        }

        let mut pos = offset + 8;
        let data_segment_size = read_offset(file_data, pos, length_size)?;
        pos += ls;
        let free_list_head_offset = read_offset(file_data, pos, length_size)?;
        pos += ls;
        let data_segment_address = read_offset(file_data, pos, offset_size)?;

        Ok(LocalHeap {
            data_segment_size,
            free_list_head_offset,
            data_segment_address,
        })
    }

    /// Read a null-terminated string from the heap's data segment at the given byte offset.
    pub fn read_string(&self, file_data: &[u8], string_offset: u64) -> Result<String, FormatError> {
        let seg_addr = self.data_segment_address as usize;
        let str_start = seg_addr + string_offset as usize;
        let seg_end = seg_addr + self.data_segment_size as usize;

        if str_start >= file_data.len() || str_start >= seg_end {
            return Err(FormatError::UnexpectedEof {
                expected: str_start + 1,
                available: file_data.len(),
            });
        }

        // Find null terminator
        let search_end = seg_end.min(file_data.len());
        let mut end = str_start;
        while end < search_end && file_data[end] != 0 {
            end += 1;
        }

        if end >= search_end {
            return Err(FormatError::UnexpectedEof {
                expected: end + 1,
                available: search_end,
            });
        }

        let s = core::str::from_utf8(&file_data[str_start..end])
            .map_err(|_| FormatError::InvalidLocalHeapSignature)?;
        Ok(String::from(s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_heap_file(
        heap_offset: usize,
        data_seg_offset: usize,
        strings: &[&str],
        offset_size: u8,
        length_size: u8,
    ) -> Vec<u8> {
        // Build data segment
        let mut data_seg = Vec::new();
        for s in strings {
            data_seg.extend_from_slice(s.as_bytes());
            data_seg.push(0); // null terminator
        }
        let data_seg_size = data_seg.len();

        let total_size = data_seg_offset + data_seg_size + 64;
        let mut file = vec![0u8; total_size];

        // Write heap header at heap_offset
        let mut pos = heap_offset;
        file[pos..pos + 4].copy_from_slice(b"HEAP");
        pos += 4;
        file[pos] = 0; // version
        pos += 1;
        // reserved 3
        pos += 3;
        // data_segment_size
        write_val(&mut file, pos, data_seg_size as u64, length_size);
        pos += length_size as usize;
        // free_list_head_offset
        write_val(&mut file, pos, 0xFFFFFFFF, length_size);
        pos += length_size as usize;
        // data_segment_address
        write_val(&mut file, pos, data_seg_offset as u64, offset_size);

        // Write data segment
        file[data_seg_offset..data_seg_offset + data_seg_size].copy_from_slice(&data_seg);

        file
    }

    fn write_val(buf: &mut [u8], pos: usize, val: u64, size: u8) {
        match size {
            4 => buf[pos..pos + 4].copy_from_slice(&(val as u32).to_le_bytes()),
            8 => buf[pos..pos + 8].copy_from_slice(&val.to_le_bytes()),
            _ => panic!("test"),
        }
    }

    #[test]
    fn parse_heap_header() {
        let file = build_heap_file(0, 100, &["hello", "world"], 8, 8);
        let heap = LocalHeap::parse(&file, 0, 8, 8).unwrap();
        assert_eq!(heap.data_segment_address, 100);
        assert_eq!(heap.data_segment_size, 12); // "hello\0world\0"
    }

    #[test]
    fn read_string_at_offset_0() {
        let file = build_heap_file(0, 100, &["hello", "world"], 8, 8);
        let heap = LocalHeap::parse(&file, 0, 8, 8).unwrap();
        let s = heap.read_string(&file, 0).unwrap();
        assert_eq!(s, "hello");
    }

    #[test]
    fn read_string_at_offset_6() {
        let file = build_heap_file(0, 100, &["hello", "world"], 8, 8);
        let heap = LocalHeap::parse(&file, 0, 8, 8).unwrap();
        let s = heap.read_string(&file, 6).unwrap();
        assert_eq!(s, "world");
    }

    #[test]
    fn invalid_signature() {
        let mut file = build_heap_file(0, 100, &["x"], 8, 8);
        file[0] = b'X';
        let err = LocalHeap::parse(&file, 0, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidLocalHeapSignature);
    }

    #[test]
    fn read_string_past_segment() {
        let file = build_heap_file(0, 100, &["hi"], 8, 8);
        let heap = LocalHeap::parse(&file, 0, 8, 8).unwrap();
        let err = heap.read_string(&file, 100).unwrap_err();
        assert!(matches!(err, FormatError::UnexpectedEof { .. }));
    }

    #[test]
    fn parse_heap_4byte_offsets() {
        let file = build_heap_file(0, 80, &["test"], 4, 4);
        let heap = LocalHeap::parse(&file, 0, 4, 4).unwrap();
        assert_eq!(heap.data_segment_address, 80);
        let s = heap.read_string(&file, 0).unwrap();
        assert_eq!(s, "test");
    }

    #[test]
    fn invalid_version() {
        let mut file = build_heap_file(0, 100, &["x"], 8, 8);
        file[4] = 1; // bad version
        let err = LocalHeap::parse(&file, 0, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidLocalHeapVersion(1));
    }
}
