//! HDF5 Extensible Array index parsing for chunked datasets (v4 index type 4).
//!
//! Extensible Arrays are used for datasets with exactly one unlimited dimension.
//! Structures: AEHD (header), AEIB (index block), AEDB (data block), AESB (super block).

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};

use crate::chunked_read::ChunkInfo;
use crate::error::FormatError;

/// Parsed Extensible Array header (AEHD).
#[derive(Debug, Clone)]
pub struct ExtensibleArrayHeader {
    /// Client ID: 0 = non-filtered chunks, 1 = filtered chunks.
    pub client_id: u8,
    /// Size of each array element in bytes.
    pub element_size: u8,
    /// Max number of elements bits (log2 of the max number of data block elements per page).
    pub max_nelmts_bits: u8,
    /// Number of elements in the index block.
    pub idx_blk_elmts: u8,
    /// Minimum number of data block elements.
    pub min_dblk_nelmts: u8,
    /// Minimum number of elements in a super block.
    pub super_blk_min_nelmts: u8,
    /// Max number of data block elements bits.
    pub max_dblk_nelmts_bits: u8,
    /// Total number of elements stored.
    pub num_elements: u64,
    /// Address of the index block.
    pub index_block_address: u64,
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

fn is_undefined_addr(addr: u64, offset_size: u8) -> bool {
    match offset_size {
        2 => addr == 0xFFFF,
        4 => addr == 0xFFFF_FFFF,
        8 => addr == 0xFFFF_FFFF_FFFF_FFFF,
        _ => false,
    }
}

fn is_undefined(data: &[u8], pos: usize, size: u8) -> bool {
    let s = size as usize;
    if pos + s > data.len() {
        return false;
    }
    data[pos..pos + s].iter().all(|&b| b == 0xFF)
}

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

impl ExtensibleArrayHeader {
    /// Parse an Extensible Array header from file data at the given offset.
    pub fn parse(
        file_data: &[u8],
        offset: usize,
        offset_size: u8,
        length_size: u8,
    ) -> Result<Self, FormatError> {
        // EAHD: signature(4) + version(1) + client_id(1) + element_size(1) +
        //   max_nelmts_bits(1) + idx_blk_elmts(1) + min_dblk_nelmts(1) +
        //   super_blk_min_nelmts(1) + max_dblk_nelmts_bits(1) +
        //   6 stats fields (each length_size) + index_block_address(offset_size) + checksum(4)
        let min_size = 4 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1
            + 6 * length_size as usize + offset_size as usize + 4;
        if offset + min_size > file_data.len() {
            return Err(FormatError::UnexpectedEof {
                expected: offset + min_size,
                available: file_data.len(),
            });
        }

        let d = &file_data[offset..];
        if &d[0..4] != b"EAHD" {
            return Err(FormatError::ChunkedReadError(
                "invalid Extensible Array header signature".into(),
            ));
        }

        let version = d[4];
        if version != 0 {
            return Err(FormatError::ChunkedReadError(
                format!("unsupported Extensible Array header version: {version}"),
            ));
        }

        let client_id = d[5];
        let element_size = d[6];
        let max_nelmts_bits = d[7];
        let idx_blk_elmts = d[8];
        let min_dblk_nelmts = d[9];
        let super_blk_min_nelmts = d[10];
        let max_dblk_nelmts_bits = d[11];

        let mut pos = 12;
        // 6 stats fields: [0] unknown, [1] unknown, [2] nsuper_blks_created,
        // [3] super_blk_size, [4] nelmts, [5] max_idx_set
        // We only need nelmts (field[4]) and skip the rest.
        let ls = length_size as usize;
        pos += 4 * ls; // skip first 4 stats fields
        let num_elements = read_offset(d, pos, length_size)?;
        pos += ls; // skip nelmts
        pos += ls; // skip max_idx_set (6th stats field)
        let index_block_address = read_offset(d, pos, offset_size)?;

        Ok(ExtensibleArrayHeader {
            client_id,
            element_size,
            max_nelmts_bits,
            idx_blk_elmts,
            min_dblk_nelmts,
            super_blk_min_nelmts,
            max_dblk_nelmts_bits,
            num_elements,
            index_block_address,
        })
    }

    /// Compute the size of this header in bytes (for write support).
    pub fn serialized_size(offset_size: u8, length_size: u8) -> usize {
        4 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1
            + 6 * length_size as usize + offset_size as usize + 4
    }
}

/// Read a single element from the extensible array element data.
/// Returns (chunk_info, bytes_consumed) or None if unallocated.
#[allow(clippy::too_many_arguments)]
fn read_element(
    data: &[u8],
    pos: usize,
    client_id: u8,
    element_size: u8,
    offset_size: u8,
    chunk_byte_size: u64,
    linear_index: usize,
    num_chunks_per_dim: &[u64],
    chunk_dimensions: &[u32],
) -> Result<(Option<ChunkInfo>, usize), FormatError> {
    let os = offset_size as usize;

    if client_id == 0 {
        // Non-filtered: just address
        if pos + os > data.len() {
            return Err(FormatError::UnexpectedEof {
                expected: pos + os,
                available: data.len(),
            });
        }
        if is_undefined(data, pos, offset_size) {
            return Ok((None, os));
        }
        let address = read_offset(data, pos, offset_size)?;
        let offsets = index_to_chunk_offsets(linear_index, num_chunks_per_dim, chunk_dimensions);
        Ok((
            Some(ChunkInfo {
                chunk_size: chunk_byte_size as u32,
                filter_mask: 0,
                offsets,
                address,
            }),
            os,
        ))
    } else {
        // Filtered: address + compressed_size + filter_mask
        let chunk_size_bytes = element_size as usize - os - 4;
        let elem_total = os + chunk_size_bytes + 4;
        if pos + elem_total > data.len() {
            return Err(FormatError::UnexpectedEof {
                expected: pos + elem_total,
                available: data.len(),
            });
        }
        if is_undefined(data, pos, offset_size) {
            return Ok((None, elem_total));
        }
        let address = read_offset(data, pos, offset_size)?;
        let chunk_size = read_variable_length(&data[pos + os..], chunk_size_bytes)?;
        let fm_off = pos + os + chunk_size_bytes;
        let filter_mask = u32::from_le_bytes([
            data[fm_off],
            data[fm_off + 1],
            data[fm_off + 2],
            data[fm_off + 3],
        ]);
        let offsets = index_to_chunk_offsets(linear_index, num_chunks_per_dim, chunk_dimensions);
        Ok((
            Some(ChunkInfo {
                chunk_size: chunk_size as u32,
                filter_mask,
                offsets,
                address,
            }),
            elem_total,
        ))
    }
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

/// Collect elements from a data block at the given offset.
#[allow(clippy::too_many_arguments)]
fn read_data_block_elements(
    file_data: &[u8],
    db_offset: usize,
    nelmts: usize,
    header: &ExtensibleArrayHeader,
    offset_size: u8,
    chunk_byte_size: u64,
    start_index: usize,
    num_chunks_per_dim: &[u64],
    chunk_dimensions: &[u32],
) -> Result<Vec<ChunkInfo>, FormatError> {
    // AEDB: signature(4) + version(1) + client_id(1) + header_address(offset_size)
    let db_header_size = 4 + 1 + 1 + offset_size as usize;
    if db_offset + db_header_size > file_data.len() {
        return Err(FormatError::UnexpectedEof {
            expected: db_offset + db_header_size,
            available: file_data.len(),
        });
    }

    let d = &file_data[db_offset..];
    if &d[0..4] != b"EADB" {
        return Err(FormatError::ChunkedReadError(
            "invalid Extensible Array data block signature".into(),
        ));
    }
    // Skip version(1) + client_id(1) + header_address(offset_size) + block_offset
    // Block offset is encoded in ceil(max_nelmts_bits/8) bytes
    let blk_off_size = (header.max_nelmts_bits as usize).div_ceil(8);
    let mut pos = db_offset + db_header_size + blk_off_size;

    // Check if paged
    let page_nelmts = 1usize << header.max_nelmts_bits;
    let is_paged = nelmts > page_nelmts;

    let mut chunks = Vec::new();

    if !is_paged {
        for i in 0..nelmts {
            let (info, consumed) = read_element(
                file_data,
                pos,
                header.client_id,
                header.element_size,
                offset_size,
                chunk_byte_size,
                start_index + i,
                num_chunks_per_dim,
                chunk_dimensions,
            )?;
            if let Some(ci) = info {
                chunks.push(ci);
            }
            pos += consumed;
        }
    } else {
        // Paged: elements are split into pages of page_nelmts.
        // After the data block header comes a page bitmap, then each page
        // has page_nelmts elements followed by a 4-byte checksum.
        let npages = nelmts.div_ceil(page_nelmts);
        // Page bitmap: ceil(npages / 8) bytes
        let bitmap_size = npages.div_ceil(8);
        // Read bitmap
        if pos + bitmap_size > file_data.len() {
            return Err(FormatError::UnexpectedEof {
                expected: pos + bitmap_size,
                available: file_data.len(),
            });
        }
        let bitmap = &file_data[pos..pos + bitmap_size];
        pos += bitmap_size;

        let elem_bytes = if header.client_id == 0 {
            offset_size as usize
        } else {
            header.element_size as usize
        };

        let mut global_idx = start_index;
        for page_idx in 0..npages {
            let byte_idx = page_idx / 8;
            let bit_idx = page_idx % 8;
            let page_has_data = (bitmap[byte_idx] >> bit_idx) & 1 != 0;

            let elems_this_page = if page_idx == npages - 1 {
                let remainder = nelmts % page_nelmts;
                if remainder == 0 { page_nelmts } else { remainder }
            } else {
                page_nelmts
            };

            if page_has_data {
                for i in 0..elems_this_page {
                    let (info, consumed) = read_element(
                        file_data,
                        pos,
                        header.client_id,
                        header.element_size,
                        offset_size,
                        chunk_byte_size,
                        global_idx + i,
                        num_chunks_per_dim,
                        chunk_dimensions,
                    )?;
                    if let Some(ci) = info {
                        chunks.push(ci);
                    }
                    pos += consumed;
                }
                // Skip page checksum (4 bytes)
                pos += 4;
            } else {
                // Empty page: skip all elements + checksum
                pos += elems_this_page * elem_bytes + 4;
            }
            global_idx += elems_this_page;
        }
    }

    Ok(chunks)
}

/// Read chunk records from an Extensible Array.
///
/// Traverses AEHD -> AEIB -> AEDB/AESB to collect all allocated chunks.
#[allow(clippy::too_many_arguments)]
pub fn read_extensible_array_chunks(
    file_data: &[u8],
    header: &ExtensibleArrayHeader,
    dataset_dims: &[u64],
    chunk_dimensions: &[u32],
    element_size: u32,
    offset_size: u8,
    _length_size: u8,
) -> Result<Vec<ChunkInfo>, FormatError> {
    let rank = chunk_dimensions.len();
    let os = offset_size as usize;

    let mut num_chunks_per_dim = Vec::with_capacity(rank);
    for d in 0..rank {
        let ds_dim = dataset_dims[d];
        let ch_dim = chunk_dimensions[d] as u64;
        num_chunks_per_dim.push(ds_dim.div_ceil(ch_dim));
    }

    let chunk_byte_size: u64 = chunk_dimensions.iter().map(|&d| d as u64).product::<u64>()
        * element_size as u64;

    // Parse index block (AEIB)
    let ib_offset = header.index_block_address as usize;
    let ib_header_size = 4 + 1 + 1 + offset_size as usize; // sig + ver + client + hdr_addr
    if ib_offset + ib_header_size > file_data.len() {
        return Err(FormatError::UnexpectedEof {
            expected: ib_offset + ib_header_size,
            available: file_data.len(),
        });
    }

    let ib = &file_data[ib_offset..];
    if &ib[0..4] != b"EAIB" {
        return Err(FormatError::ChunkedReadError(
            "invalid Extensible Array index block signature".into(),
        ));
    }
    // Skip version(1) + client_id(1) + header_address(offset_size)
    let mut pos = ib_offset + ib_header_size;

    let mut chunks = Vec::new();
    let mut global_index = 0usize;
    let total_elements = header.num_elements as usize;

    // 1. Read inline elements in index block
    let n_inline = header.idx_blk_elmts as usize;
    for i in 0..n_inline {
        if global_index + i >= total_elements {
            break;
        }
        let (info, consumed) = read_element(
            file_data,
            pos,
            header.client_id,
            header.element_size,
            offset_size,
            chunk_byte_size,
            global_index + i,
            &num_chunks_per_dim,
            chunk_dimensions,
        )?;
        if let Some(ci) = info {
            chunks.push(ci);
        }
        pos += consumed;
    }
    global_index += n_inline.min(total_elements);

    // If all elements were inline, we're done
    if global_index >= total_elements {
        return Ok(chunks);
    }

    // Compute data block and super block counts
    let min_dblk = header.min_dblk_nelmts as usize;
    let sblk_min = header.super_blk_min_nelmts as usize;

    // The first sblk_min super block levels have their data blocks listed directly
    // in the index block. Compute their sizes.
    let mut n_direct_dblks = 0usize;
    let mut dblk_sizes: Vec<usize> = Vec::new();
    {
        let mut nelmts = min_dblk;
        for sb_level in 0..sblk_min {
            let ndblks = 1usize << sb_level;
            for _ in 0..ndblks {
                dblk_sizes.push(nelmts);
                n_direct_dblks += 1;
            }
            if sb_level > 0 {
                nelmts *= 2;
            }
        }
    }

    // Read direct data block addresses from index block
    let mut dblk_addrs: Vec<u64> = Vec::with_capacity(n_direct_dblks);
    for _ in 0..n_direct_dblks {
        if pos + os > file_data.len() {
            break;
        }
        let addr = read_offset(file_data, pos, offset_size)?;
        dblk_addrs.push(addr);
        pos += os;
    }

    // Read elements from direct data blocks
    for (i, &addr) in dblk_addrs.iter().enumerate() {
        if i >= dblk_sizes.len() {
            break;
        }
        let nelmts = dblk_sizes[i];
        if is_undefined_addr(addr, offset_size) {
            global_index += nelmts;
            continue;
        }
        let block_chunks = read_data_block_elements(
            file_data,
            addr as usize,
            nelmts,
            header,
            offset_size,
            chunk_byte_size,
            global_index,
            &num_chunks_per_dim,
            chunk_dimensions,
        )?;
        chunks.extend(block_chunks);
        global_index += nelmts;
    }

    // Remaining elements are in super blocks
    let total_in_ib_and_direct: usize = n_inline + dblk_sizes.iter().sum::<usize>();
    if total_elements <= total_in_ib_and_direct {
        return Ok(chunks);
    }
    let remaining_elements = total_elements - total_in_ib_and_direct;

    // Compute super block layout
    let mut sb_addrs: Vec<u64> = Vec::new();
    let mut sb_infos: Vec<(usize, usize)> = Vec::new();
    {
        let mut covered = 0usize;
        let mut sb_level = sblk_min;
        let mut nelmts_per_dblk = min_dblk;
        for lev in 0..sblk_min {
            if lev > 0 {
                nelmts_per_dblk *= 2;
            }
        }

        while covered < remaining_elements {
            let ndblks = 1usize << sb_level;
            nelmts_per_dblk *= 2;
            let total_in_sb = ndblks * nelmts_per_dblk;
            sb_infos.push((ndblks, nelmts_per_dblk));
            covered += total_in_sb;
            sb_level += 1;
        }
    }

    // Read super block addresses from index block
    for _ in 0..sb_infos.len() {
        if pos + os > file_data.len() {
            break;
        }
        let addr = read_offset(file_data, pos, offset_size)?;
        sb_addrs.push(addr);
        pos += os;
    }

    // Process each super block
    for (sb_idx, &sb_addr) in sb_addrs.iter().enumerate() {
        let (ndblks, nelmts_per_dblk) = sb_infos[sb_idx];
        if is_undefined_addr(sb_addr, offset_size) {
            global_index += ndblks * nelmts_per_dblk;
            continue;
        }
        let sb_chunks = read_super_block(
            file_data,
            sb_addr as usize,
            ndblks,
            nelmts_per_dblk,
            header,
            offset_size,
            chunk_byte_size,
            global_index,
            &num_chunks_per_dim,
            chunk_dimensions,
        )?;
        chunks.extend(sb_chunks);
        global_index += ndblks * nelmts_per_dblk;
    }

    Ok(chunks)
}

/// Read a super block (AESB) and its data blocks.
#[allow(clippy::too_many_arguments)]
fn read_super_block(
    file_data: &[u8],
    sb_offset: usize,
    ndblks: usize,
    nelmts_per_dblk: usize,
    header: &ExtensibleArrayHeader,
    offset_size: u8,
    chunk_byte_size: u64,
    start_index: usize,
    num_chunks_per_dim: &[u64],
    chunk_dimensions: &[u32],
) -> Result<Vec<ChunkInfo>, FormatError> {
    let os = offset_size as usize;

    // AESB: signature(4) + version(1) + client_id(1) + header_address(offset_size)
    let sb_header_size = 4 + 1 + 1 + os;
    if sb_offset + sb_header_size > file_data.len() {
        return Err(FormatError::UnexpectedEof {
            expected: sb_offset + sb_header_size,
            available: file_data.len(),
        });
    }

    if &file_data[sb_offset..sb_offset + 4] != b"EASB" {
        return Err(FormatError::ChunkedReadError(
            "invalid Extensible Array super block signature".into(),
        ));
    }

    let mut pos = sb_offset + sb_header_size;

    // Read data block addresses
    let mut dblk_addrs: Vec<u64> = Vec::with_capacity(ndblks);
    for _ in 0..ndblks {
        if pos + os > file_data.len() {
            return Err(FormatError::UnexpectedEof {
                expected: pos + os,
                available: file_data.len(),
            });
        }
        let addr = read_offset(file_data, pos, offset_size)?;
        dblk_addrs.push(addr);
        pos += os;
    }

    let mut chunks = Vec::new();
    let mut global_idx = start_index;

    for &addr in &dblk_addrs {
        if is_undefined_addr(addr, offset_size) {
            global_idx += nelmts_per_dblk;
            continue;
        }
        let block_chunks = read_data_block_elements(
            file_data,
            addr as usize,
            nelmts_per_dblk,
            header,
            offset_size,
            chunk_byte_size,
            global_idx,
            num_chunks_per_dim,
            chunk_dimensions,
        )?;
        chunks.extend(block_chunks);
        global_idx += nelmts_per_dblk;
    }

    Ok(chunks)
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
        let num_chunks = vec![3u64, 2];
        let chunk_dims = vec![4u32, 3];
        assert_eq!(index_to_chunk_offsets(0, &num_chunks, &chunk_dims), vec![0, 0]);
        assert_eq!(index_to_chunk_offsets(1, &num_chunks, &chunk_dims), vec![0, 3]);
        assert_eq!(index_to_chunk_offsets(2, &num_chunks, &chunk_dims), vec![4, 0]);
    }

    #[test]
    fn parse_header_valid() {
        let os: u8 = 8;
        let ls: u8 = 8;
        let mut buf = vec![0u8; 256];
        buf[0..4].copy_from_slice(b"EAHD");
        buf[4] = 0; // version
        buf[5] = 0; // client_id = non-filtered
        buf[6] = 8; // element_size
        buf[7] = 10; // max_nelmts_bits
        buf[8] = 2; // idx_blk_elmts
        buf[9] = 4; // min_dblk_nelmts
        buf[10] = 2; // super_blk_min_nelmts
        buf[11] = 8; // max_dblk_nelmts_bits
        // 6 stats fields (each 8 bytes)
        buf[12..20].copy_from_slice(&0u64.to_le_bytes()); // stat[0]
        buf[20..28].copy_from_slice(&0u64.to_le_bytes()); // stat[1]
        buf[28..36].copy_from_slice(&0u64.to_le_bytes()); // stat[2]
        buf[36..44].copy_from_slice(&0u64.to_le_bytes()); // stat[3]
        buf[44..52].copy_from_slice(&5u64.to_le_bytes()); // stat[4] = num_elements
        buf[52..60].copy_from_slice(&0u64.to_le_bytes()); // stat[5]
        buf[60..68].copy_from_slice(&0x1000u64.to_le_bytes()); // index_block_address

        let hdr = ExtensibleArrayHeader::parse(&buf, 0, os, ls).unwrap();
        assert_eq!(hdr.client_id, 0);
        assert_eq!(hdr.element_size, 8);
        assert_eq!(hdr.idx_blk_elmts, 2);
        assert_eq!(hdr.min_dblk_nelmts, 4);
        assert_eq!(hdr.num_elements, 5);
        assert_eq!(hdr.index_block_address, 0x1000);
    }

    #[test]
    fn parse_header_invalid_signature() {
        let mut buf = vec![0u8; 256];
        buf[0..4].copy_from_slice(b"XXXX");
        let result = ExtensibleArrayHeader::parse(&buf, 0, 8, 8);
        assert!(result.is_err());
    }

    #[test]
    fn parse_header_invalid_version() {
        let mut buf = vec![0u8; 256];
        buf[0..4].copy_from_slice(b"EAHD");
        buf[4] = 1;
        let result = ExtensibleArrayHeader::parse(&buf, 0, 8, 8);
        assert!(result.is_err());
    }

    /// Build a synthetic Extensible Array with only inline elements (simplest case).
    /// All chunks fit in the index block.
    #[test]
    fn read_inline_only() {
        let os: u8 = 8;
        let ls: u8 = 8;
        let osv = os as usize;
        let num_chunks = 2usize;
        let chunk_byte_size = 20u64 * 8; // 20 elements × 8 bytes

        let mut file_data = vec![0u8; 0x3000];

        // AEHD at offset 0x100
        let aehd_offset = 0x100usize;
        let aeib_offset = 0x200usize;

        // Build AEHD
        file_data[aehd_offset..aehd_offset + 4].copy_from_slice(b"EAHD");
        file_data[aehd_offset + 4] = 0; // version
        file_data[aehd_offset + 5] = 0; // client_id = non-filtered
        file_data[aehd_offset + 6] = osv as u8; // element_size
        file_data[aehd_offset + 7] = 10; // max_nelmts_bits
        file_data[aehd_offset + 8] = num_chunks as u8; // idx_blk_elmts (all inline)
        file_data[aehd_offset + 9] = 4; // min_dblk_nelmts
        file_data[aehd_offset + 10] = 2; // super_blk_min_nelmts
        file_data[aehd_offset + 11] = 8; // max_dblk_nelmts_bits
        // 6 stats fields (each 8 bytes), nelmts at stat[4]
        file_data[aehd_offset + 44..aehd_offset + 52]
            .copy_from_slice(&(num_chunks as u64).to_le_bytes());
        file_data[aehd_offset + 60..aehd_offset + 68]
            .copy_from_slice(&(aeib_offset as u64).to_le_bytes());
        // checksum (4 bytes at +68) — not validated

        // Build AEIB at aeib_offset
        file_data[aeib_offset..aeib_offset + 4].copy_from_slice(b"EAIB");
        file_data[aeib_offset + 4] = 0; // version
        file_data[aeib_offset + 5] = 0; // client_id
        file_data[aeib_offset + 6..aeib_offset + 14]
            .copy_from_slice(&(aehd_offset as u64).to_le_bytes());

        // Inline elements
        let elem_start = aeib_offset + 6 + osv;
        let base_addr = 0x1000u64;
        for i in 0..num_chunks {
            let addr = base_addr + i as u64 * chunk_byte_size;
            let p = elem_start + i * osv;
            file_data[p..p + osv].copy_from_slice(&addr.to_le_bytes());
        }

        let header = ExtensibleArrayHeader::parse(&file_data, aehd_offset, os, ls).unwrap();
        let ds_dims = vec![40u64]; // 2 chunks × 20 elements
        let chunk_dims = vec![20u32];
        let chunks = read_extensible_array_chunks(
            &file_data,
            &header,
            &ds_dims,
            &chunk_dims,
            8,
            os,
            ls,
        )
        .unwrap();

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].address, base_addr);
        assert_eq!(chunks[0].offsets, vec![0]);
        assert_eq!(chunks[0].chunk_size, chunk_byte_size as u32);
        assert_eq!(chunks[1].address, base_addr + chunk_byte_size);
        assert_eq!(chunks[1].offsets, vec![20]);
    }

    /// Build a synthetic EA with inline elements + one direct data block.
    #[test]
    fn read_inline_plus_data_blocks() {
        let os: u8 = 8;
        let ls: u8 = 8;
        let osv = os as usize;
        let chunk_byte_size = 10u64 * 8; // 10 elements × 8 bytes
        let idx_blk_elmts = 2u8;
        let min_dblk_nelmts = 2u8;
        let sblk_min = 2u8;
        let total_chunks = 4usize; // 2 inline + 2 in data block (1 dblk from sb_level 0)

        let mut file_data = vec![0u8; 0x5000];
        let aehd_offset = 0x100usize;
        let aeib_offset = 0x200usize;
        let aedb_offset = 0x300usize;

        // EAHD
        file_data[aehd_offset..aehd_offset + 4].copy_from_slice(b"EAHD");
        file_data[aehd_offset + 4] = 0;
        file_data[aehd_offset + 5] = 0; // client_id
        file_data[aehd_offset + 6] = osv as u8; // element_size
        file_data[aehd_offset + 7] = 10;
        file_data[aehd_offset + 8] = idx_blk_elmts;
        file_data[aehd_offset + 9] = min_dblk_nelmts;
        file_data[aehd_offset + 10] = sblk_min;
        file_data[aehd_offset + 11] = 8;
        // 6 stats fields (each 8 bytes), nelmts at stat[4] (offset 12 + 4*8 = 44)
        file_data[aehd_offset + 44..aehd_offset + 52]
            .copy_from_slice(&(total_chunks as u64).to_le_bytes());
        // idx_blk_addr at offset 12 + 6*8 = 60
        file_data[aehd_offset + 60..aehd_offset + 68]
            .copy_from_slice(&(aeib_offset as u64).to_le_bytes());

        // AEIB
        file_data[aeib_offset..aeib_offset + 4].copy_from_slice(b"EAIB");
        file_data[aeib_offset + 4] = 0;
        file_data[aeib_offset + 5] = 0;
        file_data[aeib_offset + 6..aeib_offset + 14]
            .copy_from_slice(&(aehd_offset as u64).to_le_bytes());

        let mut pos = aeib_offset + 6 + osv;

        // Inline elements (2 chunks)
        let base_addr = 0x1000u64;
        for i in 0..idx_blk_elmts as usize {
            let addr = base_addr + i as u64 * chunk_byte_size;
            file_data[pos..pos + osv].copy_from_slice(&addr.to_le_bytes());
            pos += osv;
        }

        // Direct data block addresses: first sb_level=0 has 1 dblk, sb_level=1 has 1 dblk
        // Total direct dblks for sblk_min=2: 2^0 + 2^1 = 1 + 2 = 3 (oops)
        // Actually: sblk_min levels. level 0: 2^0=1 dblk, level 1: 2^1=2 dblks => 3 dblks
        // But we only have 2 remaining elements.
        // dblk sizes: level 0: 1 dblk of min_dblk=2; level 1: 2 dblks of 2 each (nelmts doubles at level > 0)
        // Wait, re-reading the code: at level 0, nelmts=min_dblk=2, 1 dblk.
        // At level 1, 1 dblk, nelmts still 2 (doubles only at level > 0... but the code says
        // `if sb_level > 0 { nelmts *= 2 }` after pushing). Let me re-check.
        // After push at level 0: nelmts=2. Then if 0>0 false, no double. Push 1 dblk of 2.
        // Level 1: ndblks=2. Push 2 dblks of 2. Then 1>0 true, nelmts=4.
        // Total: 3 dblks with sizes [2, 2, 2]. Total = 6.
        // We only need 2 more elements. So only the first dblk has data.
        let n_direct_dblks = 3;
        file_data[pos..pos + osv].copy_from_slice(&(aedb_offset as u64).to_le_bytes());
        pos += osv;
        // 2 more dblk addresses - undefined
        for _ in 1..n_direct_dblks {
            file_data[pos..pos + osv].copy_from_slice(&u64::MAX.to_le_bytes());
            pos += osv;
        }

        // EADB at aedb_offset (min_dblk_nelmts elements)
        file_data[aedb_offset..aedb_offset + 4].copy_from_slice(b"EADB");
        file_data[aedb_offset + 4] = 0;
        file_data[aedb_offset + 5] = 0;
        file_data[aedb_offset + 6..aedb_offset + 14]
            .copy_from_slice(&(aehd_offset as u64).to_le_bytes());
        // block_offset: ceil(max_nelmts_bits/8) = ceil(10/8) = 2 bytes
        // block_offset = 0 for first data block
        let blk_off_size = (10usize).div_ceil(8); // max_nelmts_bits=10
        let mut dbpos = aedb_offset + 6 + osv + blk_off_size;
        for i in 0..min_dblk_nelmts as usize {
            let addr = base_addr + (idx_blk_elmts as u64 + i as u64) * chunk_byte_size;
            file_data[dbpos..dbpos + osv].copy_from_slice(&addr.to_le_bytes());
            dbpos += osv;
        }

        let header = ExtensibleArrayHeader::parse(&file_data, aehd_offset, os, ls).unwrap();
        let ds_dims = vec![40u64];
        let chunk_dims = vec![10u32];
        let chunks = read_extensible_array_chunks(
            &file_data, &header, &ds_dims, &chunk_dims, 8, os, ls,
        )
        .unwrap();

        assert_eq!(chunks.len(), 4);
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.address, base_addr + i as u64 * chunk_byte_size);
            assert_eq!(c.offsets, vec![i as u64 * 10]);
        }
    }

    /// Test serialized_size computation.
    #[test]
    fn header_serialized_size() {
        // 12 fixed + 6*8 stats + 8 addr + 4 checksum = 72
        assert_eq!(ExtensibleArrayHeader::serialized_size(8, 8), 72);
        // 12 fixed + 6*4 stats + 4 addr + 4 checksum = 44
        assert_eq!(ExtensibleArrayHeader::serialized_size(4, 4), 44);
    }

    /// Verify read_element for unallocated slots.
    #[test]
    fn read_element_unallocated() {
        let data = vec![0xFFu8; 16];
        let num_chunks = vec![5u64];
        let chunk_dims = vec![10u32];
        let (info, consumed) = read_element(
            &data, 0, 0, 8, 8, 80, 0, &num_chunks, &chunk_dims,
        )
        .unwrap();
        assert!(info.is_none());
        assert_eq!(consumed, 8);
    }

    /// Verify filtered element reading.
    #[test]
    fn read_element_filtered() {
        let os: u8 = 8;
        let chunk_size_bytes = 4usize;
        let elem_size = os as usize + chunk_size_bytes + 4;
        let mut data = vec![0u8; elem_size + 16];
        // Address
        data[0..8].copy_from_slice(&0x2000u64.to_le_bytes());
        // Compressed size (4 bytes LE)
        data[8..12].copy_from_slice(&120u32.to_le_bytes());
        // Filter mask
        data[12..16].copy_from_slice(&0u32.to_le_bytes());

        let num_chunks = vec![5u64];
        let chunk_dims = vec![10u32];
        let (info, consumed) = read_element(
            &data, 0, 1, elem_size as u8, os, 80, 2, &num_chunks, &chunk_dims,
        )
        .unwrap();
        let ci = info.unwrap();
        assert_eq!(ci.address, 0x2000);
        assert_eq!(ci.chunk_size, 120);
        assert_eq!(ci.filter_mask, 0);
        assert_eq!(ci.offsets, vec![20]);
        assert_eq!(consumed, elem_size);
    }
}
