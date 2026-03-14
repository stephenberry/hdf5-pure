//! HDF5 Fixed Array index parsing for chunked datasets (v4 index type 3).

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};

use crate::chunked_read::ChunkInfo;
use crate::error::FormatError;

/// Parsed Fixed Array header (FAHD).
#[derive(Debug, Clone)]
pub struct FixedArrayHeader {
    /// Client ID: 0 = non-filtered chunks, 1 = filtered chunks.
    pub client_id: u8,
    /// Size of each array element in bytes.
    pub element_size: u8,
    /// Log2 of max number of elements in a data block page.
    pub max_nelmts_bits: u8,
    /// Total number of elements (chunks) in the array.
    pub num_elements: u64,
    /// Address of the data block.
    pub data_block_address: u64,
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

fn read_length(data: &[u8], pos: usize, size: u8) -> Result<u64, FormatError> {
    read_offset(data, pos, size)
}

fn is_undefined(data: &[u8], pos: usize, size: u8) -> bool {
    let s = size as usize;
    if pos + s > data.len() {
        return false;
    }
    data[pos..pos + s].iter().all(|&b| b == 0xFF)
}

impl FixedArrayHeader {
    /// Parse a Fixed Array header from file data at the given offset.
    pub fn parse(
        file_data: &[u8],
        offset: usize,
        offset_size: u8,
        length_size: u8,
    ) -> Result<Self, FormatError> {
        // FAHD signature(4) + version(1) + client_id(1) + element_size(1) +
        // max_nelmts_bits(1) + num_elements(length_size) + data_block_addr(offset_size) + checksum(4)
        let min_size = 4 + 1 + 1 + 1 + 1 + length_size as usize + offset_size as usize + 4;
        if offset + min_size > file_data.len() {
            return Err(FormatError::UnexpectedEof {
                expected: offset + min_size,
                available: file_data.len(),
            });
        }

        let d = &file_data[offset..];
        if &d[0..4] != b"FAHD" {
            return Err(FormatError::ChunkedReadError(
                "invalid Fixed Array header signature".into(),
            ));
        }

        let version = d[4];
        if version != 0 {
            return Err(FormatError::ChunkedReadError(
                format!("unsupported Fixed Array header version: {version}"),
            ));
        }

        let client_id = d[5];
        let element_size = d[6];
        let max_nelmts_bits = d[7];

        let mut pos = 8;
        let num_elements = read_length(d, pos, length_size)?;
        pos += length_size as usize;
        let data_block_address = read_offset(d, pos, offset_size)?;

        Ok(FixedArrayHeader {
            client_id,
            element_size,
            max_nelmts_bits,
            num_elements,
            data_block_address,
        })
    }
}

/// Read chunk records from a Fixed Array data block.
///
/// Returns a `Vec<ChunkInfo>` with one entry per allocated chunk.
/// `chunk_dimensions` should be the spatial chunk dims only (not including the element-size dim).
/// `element_size` is the datatype size in bytes.
#[allow(clippy::too_many_arguments)]
pub fn read_fixed_array_chunks(
    file_data: &[u8],
    header: &FixedArrayHeader,
    dataset_dims: &[u64],
    chunk_dimensions: &[u32],
    element_size: u32,
    offset_size: u8,
    _length_size: u8,
) -> Result<Vec<ChunkInfo>, FormatError> {
    let db_offset = header.data_block_address as usize;
    let rank = chunk_dimensions.len();

    // Parse data block header: FADB(4) + version(1) + client_id(1) + header_address(offset_size)
    let db_header_size = 4 + 1 + 1 + offset_size as usize;
    if db_offset + db_header_size > file_data.len() {
        return Err(FormatError::UnexpectedEof {
            expected: db_offset + db_header_size,
            available: file_data.len(),
        });
    }

    let d = &file_data[db_offset..];
    if &d[0..4] != b"FADB" {
        return Err(FormatError::ChunkedReadError(
            "invalid Fixed Array data block signature".into(),
        ));
    }

    // Skip version(1) + client_id(1) + header_address(offset_size)
    let mut pos = db_header_size;

    // Check if paged
    let page_size = 1u64 << header.max_nelmts_bits;
    let is_paged = header.num_elements > page_size;

    if is_paged {
        // For paged data blocks, we need to handle page bitmap + pages
        // For now, implement non-paged path (covers most real-world cases)
        return Err(FormatError::ChunkedReadError(
            "paged Fixed Array data blocks not yet supported".into(),
        ));
    }

    // Non-paged: elements stored directly
    let num_elements = header.num_elements as usize;
    let os = offset_size as usize;

    // Compute chunk offsets based on index
    // Chunks are stored in row-major order within the dataset space
    let mut num_chunks_per_dim = Vec::with_capacity(rank);
    for d_idx in 0..rank {
        let ds_dim = dataset_dims[d_idx];
        let ch_dim = chunk_dimensions[d_idx] as u64;
        num_chunks_per_dim.push(ds_dim.div_ceil(ch_dim));
    }

    let chunk_byte_size: u64 = chunk_dimensions.iter().map(|&d| d as u64).product::<u64>()
        * element_size as u64;

    let mut chunks = Vec::new();

    for i in 0..num_elements {
        let elem_data = &file_data[db_offset + pos..];
        if header.client_id == 0 {
            // Non-filtered: just address
            if pos + os > file_data.len() - db_offset {
                return Err(FormatError::UnexpectedEof {
                    expected: db_offset + pos + os,
                    available: file_data.len(),
                });
            }
            let address = read_offset(elem_data, 0, offset_size)?;
            pos += os;

            if is_undefined(file_data, db_offset + pos - os, offset_size) {
                continue; // unallocated chunk
            }

            let offsets = index_to_chunk_offsets(i, &num_chunks_per_dim, chunk_dimensions);
            chunks.push(ChunkInfo {
                chunk_size: chunk_byte_size as u32,
                filter_mask: 0,
                offsets,
                address,
            });
        } else {
            // Filtered: address(offset_size) + chunk_size(variable) + filter_mask(4)
            let chunk_size_bytes = header.element_size as usize - os - 4;
            let elem_total = os + chunk_size_bytes + 4;
            if pos + elem_total > file_data.len() - db_offset {
                return Err(FormatError::UnexpectedEof {
                    expected: db_offset + pos + elem_total,
                    available: file_data.len(),
                });
            }

            let address = read_offset(elem_data, 0, offset_size)?;

            // Read chunk_size (variable length, little-endian)
            let chunk_size = read_variable_length(&elem_data[os..], chunk_size_bytes)?;

            let fm_off = os + chunk_size_bytes;
            let filter_mask = u32::from_le_bytes([
                elem_data[fm_off],
                elem_data[fm_off + 1],
                elem_data[fm_off + 2],
                elem_data[fm_off + 3],
            ]);
            pos += elem_total;

            if is_undefined(file_data, db_offset + pos - elem_total, offset_size) {
                continue; // unallocated chunk
            }

            let offsets = index_to_chunk_offsets(i, &num_chunks_per_dim, chunk_dimensions);
            chunks.push(ChunkInfo {
                chunk_size: chunk_size as u32,
                filter_mask,
                offsets,
                address,
            });
        }
    }

    Ok(chunks)
}

/// Convert a linear chunk index to N-dimensional chunk offsets in dataset space.
fn index_to_chunk_offsets(
    index: usize,
    num_chunks_per_dim: &[u64],
    chunk_dimensions: &[u32],
) -> Vec<u64> {
    let rank = num_chunks_per_dim.len();
    let mut offsets = vec![0u64; rank];
    let mut remaining = index as u64;
    for d in (0..rank).rev() {
        let nchunks = num_chunks_per_dim[d];
        let chunk_idx = remaining % nchunks;
        remaining /= nchunks;
        offsets[d] = chunk_idx * chunk_dimensions[d] as u64;
    }
    offsets
}

/// Read a variable-length little-endian unsigned integer.
fn read_variable_length(data: &[u8], size: usize) -> Result<u64, FormatError> {
    if size > 8 || data.len() < size {
        return Err(FormatError::ChunkedReadError(
            "invalid variable-length size".into(),
        ));
    }
    let mut val = 0u64;
    for (i, &byte) in data.iter().enumerate().take(size) {
        val |= (byte as u64) << (i * 8);
    }
    Ok(val)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_to_offsets_1d() {
        let num_chunks = vec![5u64];
        let chunk_dims = vec![20u32];
        assert_eq!(index_to_chunk_offsets(0, &num_chunks, &chunk_dims), vec![0]);
        assert_eq!(index_to_chunk_offsets(1, &num_chunks, &chunk_dims), vec![20]);
        assert_eq!(index_to_chunk_offsets(4, &num_chunks, &chunk_dims), vec![80]);
    }

    #[test]
    fn index_to_offsets_2d() {
        // 10x6 dataset with 4x3 chunks => ceil(10/4)=3, ceil(6/3)=2 => 6 chunks
        let num_chunks = vec![3u64, 2];
        let chunk_dims = vec![4u32, 3];
        assert_eq!(index_to_chunk_offsets(0, &num_chunks, &chunk_dims), vec![0, 0]);
        assert_eq!(index_to_chunk_offsets(1, &num_chunks, &chunk_dims), vec![0, 3]);
        assert_eq!(index_to_chunk_offsets(2, &num_chunks, &chunk_dims), vec![4, 0]);
        assert_eq!(index_to_chunk_offsets(3, &num_chunks, &chunk_dims), vec![4, 3]);
        assert_eq!(index_to_chunk_offsets(5, &num_chunks, &chunk_dims), vec![8, 3]);
    }

    #[test]
    fn read_variable_length_values() {
        assert_eq!(read_variable_length(&[0x78, 0x56], 2).unwrap(), 0x5678);
        assert_eq!(read_variable_length(&[0x01, 0x02, 0x03, 0x04], 4).unwrap(), 0x04030201);
        assert_eq!(read_variable_length(&[0xFF], 1).unwrap(), 0xFF);
    }

    #[test]
    fn parse_fixed_array_header_valid() {
        let mut buf = vec![0u8; 256];
        // FAHD signature
        buf[0..4].copy_from_slice(b"FAHD");
        buf[4] = 0; // version
        buf[5] = 1; // client_id = filtered
        buf[6] = 16; // element_size
        buf[7] = 10; // max_nelmts_bits (page_size = 1024)
        // num_elements (length_size=8)
        buf[8..16].copy_from_slice(&5u64.to_le_bytes());
        // data_block_address (offset_size=8)
        buf[16..24].copy_from_slice(&0x1000u64.to_le_bytes());
        // checksum (4 bytes, we don't validate in parse)

        let header = FixedArrayHeader::parse(&buf, 0, 8, 8).unwrap();
        assert_eq!(header.client_id, 1);
        assert_eq!(header.element_size, 16);
        assert_eq!(header.max_nelmts_bits, 10);
        assert_eq!(header.num_elements, 5);
        assert_eq!(header.data_block_address, 0x1000);
    }

    #[test]
    fn parse_fixed_array_header_invalid_signature() {
        let mut buf = vec![0u8; 256];
        buf[0..4].copy_from_slice(b"XXXX");
        let result = FixedArrayHeader::parse(&buf, 0, 8, 8);
        assert!(result.is_err());
    }

    #[test]
    fn parse_fixed_array_header_invalid_version() {
        let mut buf = vec![0u8; 256];
        buf[0..4].copy_from_slice(b"FAHD");
        buf[4] = 1; // unsupported version
        let result = FixedArrayHeader::parse(&buf, 0, 8, 8);
        assert!(result.is_err());
    }

    /// Build a synthetic Fixed Array (non-filtered) and verify reading.
    #[test]
    fn read_non_filtered_chunks() {
        let offset_size: u8 = 8;
        let length_size: u8 = 8;
        let os = offset_size as usize;
        let num_chunks = 5u64;

        let mut file_data = vec![0u8; 0x3000];

        // Build FAHD at offset 0x100
        let fahd_offset = 0x100usize;
        let db_offset = 0x200usize;
        file_data[fahd_offset..fahd_offset + 4].copy_from_slice(b"FAHD");
        file_data[fahd_offset + 4] = 0; // version
        file_data[fahd_offset + 5] = 0; // client_id = non-filtered
        file_data[fahd_offset + 6] = os as u8; // element_size = just address
        file_data[fahd_offset + 7] = 10; // max_nelmts_bits
        file_data[fahd_offset + 8..fahd_offset + 16].copy_from_slice(&num_chunks.to_le_bytes());
        file_data[fahd_offset + 16..fahd_offset + 24]
            .copy_from_slice(&(db_offset as u64).to_le_bytes());

        // Build FADB at db_offset
        file_data[db_offset..db_offset + 4].copy_from_slice(b"FADB");
        file_data[db_offset + 4] = 0; // version
        file_data[db_offset + 5] = 0; // client_id
        file_data[db_offset + 6..db_offset + 14]
            .copy_from_slice(&(fahd_offset as u64).to_le_bytes()); // header_address

        // Elements: 5 addresses
        let elem_start = db_offset + 6 + os;
        let base_addr = 0x1000u64;
        let chunk_byte_size = 20 * 8; // 20 elements × 8 bytes
        for i in 0..5 {
            let addr = base_addr + i as u64 * chunk_byte_size as u64;
            let pos = elem_start + i * os;
            file_data[pos..pos + os].copy_from_slice(&addr.to_le_bytes());
        }

        let header = FixedArrayHeader::parse(&file_data, fahd_offset, offset_size, length_size)
            .unwrap();
        let ds_dims = vec![100u64];
        let chunk_dims = vec![20u32];
        let chunks = read_fixed_array_chunks(
            &file_data, &header, &ds_dims, &chunk_dims, 8, offset_size, length_size,
        ).unwrap();

        assert_eq!(chunks.len(), 5);
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.address, base_addr + i as u64 * chunk_byte_size as u64);
            assert_eq!(c.offsets, vec![i as u64 * 20]);
            assert_eq!(c.filter_mask, 0);
            assert_eq!(c.chunk_size, chunk_byte_size as u32);
        }
    }

    /// Build a synthetic Fixed Array (filtered) and verify reading.
    #[test]
    fn read_filtered_chunks() {
        let offset_size: u8 = 8;
        let length_size: u8 = 8;
        let os = offset_size as usize;
        let num_chunks = 3u64;
        // element_size for filtered: offset_size + chunk_size_bytes + 4(filter_mask)
        // chunk_size_bytes: let's use 4 bytes
        let chunk_size_bytes = 4usize;
        let elem_size = os + chunk_size_bytes + 4;

        let mut file_data = vec![0u8; 0x3000];

        let fahd_offset = 0x100usize;
        let db_offset = 0x200usize;
        file_data[fahd_offset..fahd_offset + 4].copy_from_slice(b"FAHD");
        file_data[fahd_offset + 4] = 0;
        file_data[fahd_offset + 5] = 1; // client_id = filtered
        file_data[fahd_offset + 6] = elem_size as u8;
        file_data[fahd_offset + 7] = 10;
        file_data[fahd_offset + 8..fahd_offset + 16].copy_from_slice(&num_chunks.to_le_bytes());
        file_data[fahd_offset + 16..fahd_offset + 24]
            .copy_from_slice(&(db_offset as u64).to_le_bytes());

        file_data[db_offset..db_offset + 4].copy_from_slice(b"FADB");
        file_data[db_offset + 4] = 0;
        file_data[db_offset + 5] = 1;
        file_data[db_offset + 6..db_offset + 14]
            .copy_from_slice(&(fahd_offset as u64).to_le_bytes());

        let elem_start = db_offset + 6 + os;
        let test_chunks = [
            (0x1000u64, 120u32, 0u32),
            (0x2000u64, 115u32, 0u32),
            (0x3000u64, 100u32, 0u32),
        ];

        for (i, &(addr, csize, fmask)) in test_chunks.iter().enumerate() {
            let pos = elem_start + i * elem_size;
            file_data[pos..pos + os].copy_from_slice(&addr.to_le_bytes());
            // chunk_size as 4 bytes LE
            file_data[pos + os..pos + os + 4].copy_from_slice(&csize.to_le_bytes());
            file_data[pos + os + 4..pos + os + 8].copy_from_slice(&fmask.to_le_bytes());
        }

        let header = FixedArrayHeader::parse(&file_data, fahd_offset, offset_size, length_size)
            .unwrap();
        let ds_dims = vec![60u64];
        let chunk_dims = vec![20u32];
        let chunks = read_fixed_array_chunks(
            &file_data, &header, &ds_dims, &chunk_dims, 8, offset_size, length_size,
        ).unwrap();

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].address, 0x1000);
        assert_eq!(chunks[0].chunk_size, 120);
        assert_eq!(chunks[0].filter_mask, 0);
        assert_eq!(chunks[0].offsets, vec![0]);
        assert_eq!(chunks[1].address, 0x2000);
        assert_eq!(chunks[1].chunk_size, 115);
        assert_eq!(chunks[2].address, 0x3000);
        assert_eq!(chunks[2].chunk_size, 100);
    }
}
