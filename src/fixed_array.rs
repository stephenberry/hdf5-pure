//! HDF5 Fixed Array index parsing for chunked datasets (v4 index type 3).

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};

use crate::chunked_read::ChunkInfo;
use crate::convert::{TryToUsize, u32_from};
use crate::error::FormatError;
use crate::source::FileSource;

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
            return Err(FormatError::ChunkedReadError(format!(
                "unsupported Fixed Array header version: {version}"
            )));
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

    /// Parse a Fixed Array header from a [`FileSource`] (bounded window).
    pub fn parse_from_source<S: FileSource + ?Sized>(
        source: &S,
        address: u64,
        offset_size: u8,
        length_size: u8,
    ) -> Result<Self, FormatError> {
        let min_size = 4 + 1 + 1 + 1 + 1 + length_size as usize + offset_size as usize + 4;
        let buf = source.read_metadata_at(address, min_size)?;
        Self::parse(&buf, 0, offset_size, length_size)
    }
}

/// On-disk byte spans `(addr, len)` of a Fixed Array chunk index's own
/// structure — its header (`FAHD`) and its single data block (`FADB`, paged or
/// not) — for reclaiming a deleted chunked dataset. The chunk *data* blocks the
/// array points at are enumerated separately (via [`read_fixed_array_chunks`]),
/// so they are not included here.
///
/// `fa_base` is the Fixed Array header address taken from the data-layout
/// message. The returned spans are exact (the writer allocates each block at the
/// size computed here); the caller validates them against the file bounds.
pub(crate) fn fixed_array_index_spans(
    file_data: &[u8],
    fa_base: u64,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<(u64, u64)>, FormatError> {
    let header = FixedArrayHeader::parse(file_data, fa_base.to_usize()?, offset_size, length_size)?;
    let os = offset_size as usize;

    // FAHD: sig(4)+ver(1)+client(1)+elem_size(1)+max_bits(1)+num_elements(ls)+dblk_addr(os)+checksum(4).
    let fahd_size = (4 + 1 + 1 + 1 + 1 + length_size as usize + os + 4) as u64;
    let mut spans = vec![(fa_base, fahd_size)];

    // An empty array has no data block (the address is the undefined sentinel);
    // there is nothing more to reclaim.
    if is_undefined_addr(header.data_block_address, offset_size) {
        return Ok(spans);
    }

    // FADB element stride: just the chunk address when unfiltered, the full
    // filtered element record otherwise.
    let elem_size = if header.client_id == 0 {
        os
    } else {
        header.element_size as usize
    };
    if header.max_nelmts_bits >= 64 {
        return Err(FormatError::ChunkedReadError(
            "Fixed Array page exponent out of range".into(),
        ));
    }
    let page_size = 1usize << header.max_nelmts_bits;
    let num_elements = header.num_elements.to_usize()?;
    let db_prefix = 4 + 1 + 1 + os; // FADB sig(4)+ver(1)+client(1)+header_addr(os)

    let fadb_size: u64 = if num_elements <= page_size {
        // Non-paged: prefix + element records + checksum.
        (db_prefix
            + num_elements
                .checked_mul(elem_size)
                .ok_or(FormatError::OffsetOverflow {
                    offset: num_elements as u64,
                    length: elem_size as u64,
                })?
            + 4) as u64
    } else {
        // Paged: prefix + page-init bitmap + prefix checksum, then the element
        // records (each written exactly once) split into `npages` pages, each
        // followed by its own 4-byte checksum. Only the last page is partial;
        // the writer does not pad it to full stride (see `build_fixed_array_at`),
        // so the data block is `num_elements * elem_size + npages * 4` element
        // and checksum bytes after the prefix.
        let npages = num_elements.div_ceil(page_size);
        let bitmap_size = npages.div_ceil(8);
        let elements_bytes =
            num_elements
                .checked_mul(elem_size)
                .ok_or(FormatError::OffsetOverflow {
                    offset: num_elements as u64,
                    length: elem_size as u64,
                })?;
        (db_prefix + bitmap_size + 4 + elements_bytes + npages * 4) as u64
    };

    spans.push((header.data_block_address, fadb_size));
    Ok(spans)
}

/// True when an `offset_size`-wide address is the all-`0xFF` "undefined" value.
fn is_undefined_addr(addr: u64, offset_size: u8) -> bool {
    match offset_size {
        2 => addr == 0xFFFF,
        4 => addr == 0xFFFF_FFFF,
        8 => addr == 0xFFFF_FFFF_FFFF_FFFF,
        _ => false,
    }
}

/// Decode one Fixed/Extensible-Array element record from `block` at `elem_pos`.
///
/// Returns `None` for the all-`0xFF` sentinel (an unallocated chunk). `block`
/// may be the whole-file buffer (buffered path, `elem_pos` absolute) or a
/// data-block region read from a source (streaming path, `elem_pos` relative).
#[allow(clippy::too_many_arguments)]
fn parse_fa_element(
    block: &[u8],
    elem_pos: usize,
    index: usize,
    client_id: u8,
    chunk_byte_size: u64,
    elem_size: usize,
    chunk_size_bytes: usize,
    offset_size: u8,
    num_chunks_per_dim: &[u64],
    chunk_dimensions: &[u32],
) -> Result<Option<ChunkInfo>, FormatError> {
    if elem_pos + elem_size > block.len() {
        return Err(FormatError::UnexpectedEof {
            expected: elem_pos + elem_size,
            available: block.len(),
        });
    }
    if is_undefined(block, elem_pos, offset_size) {
        return Ok(None);
    }
    let address = read_offset(block, elem_pos, offset_size)?;
    let offsets = index_to_chunk_offsets(index, num_chunks_per_dim, chunk_dimensions);
    if client_id == 0 {
        Ok(Some(ChunkInfo {
            chunk_size: u32_from(chunk_byte_size)?,
            filter_mask: 0,
            offsets,
            address,
        }))
    } else {
        let os = offset_size as usize;
        let chunk_size = read_variable_length(&block[elem_pos + os..], chunk_size_bytes)?;
        let fm_off = elem_pos + os + chunk_size_bytes;
        let filter_mask = u32::from_le_bytes([
            block[fm_off],
            block[fm_off + 1],
            block[fm_off + 2],
            block[fm_off + 3],
        ]);
        Ok(Some(ChunkInfo {
            chunk_size: u32_from(chunk_size)?,
            filter_mask,
            offsets,
            address,
        }))
    }
}

/// Read chunk records from a Fixed Array data block.
///
/// Returns a `Vec<ChunkInfo>` with one entry per allocated chunk.
/// `chunk_dimensions` should be the spatial chunk dims only (not including the element-size dim).
/// `element_size` is the datatype size in bytes.
///
/// Handles both the non-paged layout (elements stored directly after the data
/// block prefix) and the paged layout used when the chunk count exceeds the
/// page size (`2^max_nelmts_bits`). In the paged layout the data block prefix
/// is followed by a page-initialization bitmap and a checksum, after which the
/// elements live in fixed-stride pages, each terminated by its own checksum.
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
    let db_offset = header.data_block_address.to_usize()?;
    let rank = chunk_dimensions.len();
    let os = offset_size as usize;

    // Parse data block prefix: FADB(4) + version(1) + client_id(1) + header_address(offset_size)
    let db_header_size = 4 + 1 + 1 + os;
    if db_offset + db_header_size > file_data.len() {
        return Err(FormatError::UnexpectedEof {
            expected: db_offset + db_header_size,
            available: file_data.len(),
        });
    }

    if &file_data[db_offset..db_offset + 4] != b"FADB" {
        return Err(FormatError::ChunkedReadError(
            "invalid Fixed Array data block signature".into(),
        ));
    }

    // Per-element encoding width. Non-filtered: just the chunk address.
    // Filtered: address + variable-width chunk size + 4-byte filter mask.
    let chunk_size_bytes = if header.client_id == 0 {
        0
    } else {
        (header.element_size as usize)
            .checked_sub(os + 4)
            .ok_or_else(|| {
                FormatError::ChunkedReadError("Fixed Array element size too small".into())
            })?
    };
    let elem_size = if header.client_id == 0 {
        os
    } else {
        header.element_size as usize
    };

    // Number of chunks along each dimension (row-major ordering in dataset space).
    let mut num_chunks_per_dim = Vec::with_capacity(rank);
    for d_idx in 0..rank {
        let ds_dim = dataset_dims[d_idx];
        let ch_dim = chunk_dimensions[d_idx] as u64;
        num_chunks_per_dim.push(ds_dim.div_ceil(ch_dim));
    }

    let chunk_byte_size: u64 =
        chunk_dimensions.iter().map(|&d| d as u64).product::<u64>() * element_size as u64;

    let num_elements = header.num_elements.to_usize()?;
    let page_size = (1u64 << header.max_nelmts_bits).to_usize()?;
    let is_paged = num_elements > page_size;

    let mut chunks = Vec::new();

    if !is_paged {
        // Elements stored directly after the data block prefix.
        let mut pos = db_offset + db_header_size;
        for index in 0..num_elements {
            if let Some(info) = parse_fa_element(
                file_data,
                pos,
                index,
                header.client_id,
                chunk_byte_size,
                elem_size,
                chunk_size_bytes,
                offset_size,
                &num_chunks_per_dim,
                chunk_dimensions,
            )? {
                chunks.push(info);
            }
            pos += elem_size;
        }
        return Ok(chunks);
    }

    // Paged: prefix is followed by a page-init bitmap (one bit per page,
    // most-significant-bit first) and a 4-byte checksum. Pages then follow at a
    // fixed stride of `page_size` elements plus a 4-byte checksum each; the
    // whole block is allocated contiguously, so the last (partial) page still
    // begins at its full-stride offset.
    let npages = num_elements.div_ceil(page_size);
    let bitmap_size = npages.div_ceil(8);
    let bitmap_pos = db_offset + db_header_size;
    if bitmap_pos + bitmap_size + 4 > file_data.len() {
        return Err(FormatError::UnexpectedEof {
            expected: bitmap_pos + bitmap_size + 4,
            available: file_data.len(),
        });
    }
    let bitmap = &file_data[bitmap_pos..bitmap_pos + bitmap_size];
    let pages_start = bitmap_pos + bitmap_size + 4;
    let page_stride = page_size
        .checked_mul(elem_size)
        .and_then(|bytes| bytes.checked_add(4))
        .ok_or(FormatError::OffsetOverflow {
            offset: page_size as u64,
            length: elem_size as u64,
        })?;

    for page in 0..npages {
        let nelem_in_page = core::cmp::min(page_size, num_elements - page * page_size);
        // A cleared bit means the page was never initialized: every chunk it
        // would hold is unallocated, so skip it without reading.
        let initialized = (bitmap[page / 8] >> (7 - (page % 8))) & 1 == 1;
        if !initialized {
            continue;
        }
        let page_offset = page
            .checked_mul(page_stride)
            .ok_or(FormatError::OffsetOverflow {
                offset: page as u64,
                length: page_stride as u64,
            })?;
        let page_start = pages_start + page_offset;
        for j in 0..nelem_in_page {
            let index = page * page_size + j;
            let elem_pos = page_start + j * elem_size;
            if let Some(info) = parse_fa_element(
                file_data,
                elem_pos,
                index,
                header.client_id,
                chunk_byte_size,
                elem_size,
                chunk_size_bytes,
                offset_size,
                &num_chunks_per_dim,
                chunk_dimensions,
            )? {
                chunks.push(info);
            }
        }
    }

    Ok(chunks)
}

/// Read chunk records from a Fixed Array via a [`FileSource`].
///
/// Streaming counterpart of [`read_fixed_array_chunks`]: it reads the data-block
/// prefix, then the element array (non-paged) or each initialized page
/// (paged) as bounded windows via `read_at`, decoding the same records.
#[allow(clippy::too_many_arguments)]
pub fn read_fixed_array_chunks_from_source<S: FileSource + ?Sized>(
    source: &S,
    header: &FixedArrayHeader,
    dataset_dims: &[u64],
    chunk_dimensions: &[u32],
    element_size: u32,
    offset_size: u8,
    _length_size: u8,
) -> Result<Vec<ChunkInfo>, FormatError> {
    let db_address = header.data_block_address;
    let rank = chunk_dimensions.len();
    let os = offset_size as usize;
    let db_header_size = 4 + 1 + 1 + os;

    // Data block prefix: FADB(4) + version(1) + client_id(1) + header_address.
    let prefix = source.read_metadata_at(db_address, db_header_size)?;
    if &prefix[0..4] != b"FADB" {
        return Err(FormatError::ChunkedReadError(
            "invalid Fixed Array data block signature".into(),
        ));
    }

    let chunk_size_bytes = if header.client_id == 0 {
        0
    } else {
        (header.element_size as usize)
            .checked_sub(os + 4)
            .ok_or_else(|| {
                FormatError::ChunkedReadError("Fixed Array element size too small".into())
            })?
    };
    let elem_size = if header.client_id == 0 {
        os
    } else {
        header.element_size as usize
    };

    let mut num_chunks_per_dim = Vec::with_capacity(rank);
    for d_idx in 0..rank {
        let ch_dim = chunk_dimensions[d_idx] as u64;
        num_chunks_per_dim.push(dataset_dims[d_idx].div_ceil(ch_dim));
    }
    let chunk_byte_size: u64 =
        chunk_dimensions.iter().map(|&d| d as u64).product::<u64>() * element_size as u64;

    let num_elements = header.num_elements.to_usize()?;
    let page_size = (1u64 << header.max_nelmts_bits).to_usize()?;
    let is_paged = num_elements > page_size;

    let mut chunks = Vec::new();

    if !is_paged {
        // All elements live directly after the prefix; read them in one window.
        let region = source.read_metadata_at(
            db_address + db_header_size as u64,
            num_elements
                .checked_mul(elem_size)
                .ok_or(FormatError::OffsetOverflow {
                    offset: num_elements as u64,
                    length: elem_size as u64,
                })?,
        )?;
        for index in 0..num_elements {
            if let Some(info) = parse_fa_element(
                &region,
                index * elem_size,
                index,
                header.client_id,
                chunk_byte_size,
                elem_size,
                chunk_size_bytes,
                offset_size,
                &num_chunks_per_dim,
                chunk_dimensions,
            )? {
                chunks.push(info);
            }
        }
        return Ok(chunks);
    }

    // Paged: read the page-init bitmap, then each initialized page's elements.
    let npages = num_elements.div_ceil(page_size);
    let bitmap_size = npages.div_ceil(8);
    let bitmap_addr = db_address + db_header_size as u64;
    let bitmap = source.read_metadata_at(bitmap_addr, bitmap_size)?;
    let pages_start_addr = bitmap_addr + bitmap_size as u64 + 4;
    let page_stride = page_size
        .checked_mul(elem_size)
        .and_then(|bytes| bytes.checked_add(4))
        .ok_or(FormatError::OffsetOverflow {
            offset: page_size as u64,
            length: elem_size as u64,
        })?;

    for page in 0..npages {
        let nelem_in_page = core::cmp::min(page_size, num_elements - page * page_size);
        let initialized = (bitmap[page / 8] >> (7 - (page % 8))) & 1 == 1;
        if !initialized {
            continue;
        }
        let page_offset = page
            .checked_mul(page_stride)
            .ok_or(FormatError::OffsetOverflow {
                offset: page as u64,
                length: page_stride as u64,
            })?;
        let page_addr = pages_start_addr + page_offset as u64;
        let region = source.read_metadata_at(
            page_addr,
            nelem_in_page
                .checked_mul(elem_size)
                .ok_or(FormatError::OffsetOverflow {
                    offset: nelem_in_page as u64,
                    length: elem_size as u64,
                })?,
        )?;
        for j in 0..nelem_in_page {
            if let Some(info) = parse_fa_element(
                &region,
                j * elem_size,
                page * page_size + j,
                header.client_id,
                chunk_byte_size,
                elem_size,
                chunk_size_bytes,
                offset_size,
                &num_chunks_per_dim,
                chunk_dimensions,
            )? {
                chunks.push(info);
            }
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
        assert_eq!(
            index_to_chunk_offsets(1, &num_chunks, &chunk_dims),
            vec![20]
        );
        assert_eq!(
            index_to_chunk_offsets(4, &num_chunks, &chunk_dims),
            vec![80]
        );
    }

    #[test]
    fn index_to_offsets_2d() {
        // 10x6 dataset with 4x3 chunks => ceil(10/4)=3, ceil(6/3)=2 => 6 chunks
        let num_chunks = vec![3u64, 2];
        let chunk_dims = vec![4u32, 3];
        assert_eq!(
            index_to_chunk_offsets(0, &num_chunks, &chunk_dims),
            vec![0, 0]
        );
        assert_eq!(
            index_to_chunk_offsets(1, &num_chunks, &chunk_dims),
            vec![0, 3]
        );
        assert_eq!(
            index_to_chunk_offsets(2, &num_chunks, &chunk_dims),
            vec![4, 0]
        );
        assert_eq!(
            index_to_chunk_offsets(3, &num_chunks, &chunk_dims),
            vec![4, 3]
        );
        assert_eq!(
            index_to_chunk_offsets(5, &num_chunks, &chunk_dims),
            vec![8, 3]
        );
    }

    #[test]
    fn read_variable_length_values() {
        assert_eq!(read_variable_length(&[0x78, 0x56], 2).unwrap(), 0x5678);
        assert_eq!(
            read_variable_length(&[0x01, 0x02, 0x03, 0x04], 4).unwrap(),
            0x04030201
        );
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

        let header =
            FixedArrayHeader::parse(&file_data, fahd_offset, offset_size, length_size).unwrap();
        let ds_dims = vec![100u64];
        let chunk_dims = vec![20u32];
        let chunks = read_fixed_array_chunks(
            &file_data,
            &header,
            &ds_dims,
            &chunk_dims,
            8,
            offset_size,
            length_size,
        )
        .unwrap();

        assert_eq!(chunks.len(), 5);
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.address, base_addr + i as u64 * chunk_byte_size as u64);
            assert_eq!(c.offsets, vec![i as u64 * 20]);
            assert_eq!(c.filter_mask, 0);
            assert_eq!(c.chunk_size, chunk_byte_size as u32);
        }

        #[cfg(feature = "std")]
        assert_fa_streams_match(&file_data, fahd_offset, &ds_dims, &chunk_dims, 8, 8, 8);
    }

    /// Assert the streaming Fixed-Array reader matches the buffered one over both
    /// an in-memory and a `Read+Seek` source.
    #[cfg(feature = "std")]
    fn assert_fa_streams_match(
        file_data: &[u8],
        header_offset: usize,
        ds_dims: &[u64],
        chunk_dims: &[u32],
        element_size: u32,
        offset_size: u8,
        length_size: u8,
    ) {
        use crate::source::{BytesSource, ReadSeekSource};
        let h =
            FixedArrayHeader::parse(file_data, header_offset, offset_size, length_size).unwrap();
        let buffered = read_fixed_array_chunks(
            file_data,
            &h,
            ds_dims,
            chunk_dims,
            element_size,
            offset_size,
            length_size,
        )
        .unwrap();

        let mem = BytesSource::new(file_data);
        let hm = FixedArrayHeader::parse_from_source(
            &mem,
            header_offset as u64,
            offset_size,
            length_size,
        )
        .unwrap();
        let from_mem = read_fixed_array_chunks_from_source(
            &mem,
            &hm,
            ds_dims,
            chunk_dims,
            element_size,
            offset_size,
            length_size,
        )
        .unwrap();

        let seek = ReadSeekSource::new(std::io::Cursor::new(file_data.to_vec())).unwrap();
        let hs = FixedArrayHeader::parse_from_source(
            &seek,
            header_offset as u64,
            offset_size,
            length_size,
        )
        .unwrap();
        let from_seek = read_fixed_array_chunks_from_source(
            &seek,
            &hs,
            ds_dims,
            chunk_dims,
            element_size,
            offset_size,
            length_size,
        )
        .unwrap();

        assert_eq!(buffered, from_mem, "BytesSource mismatch");
        assert_eq!(buffered, from_seek, "ReadSeekSource mismatch");
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

        let header =
            FixedArrayHeader::parse(&file_data, fahd_offset, offset_size, length_size).unwrap();
        let ds_dims = vec![60u64];
        let chunk_dims = vec![20u32];
        let chunks = read_fixed_array_chunks(
            &file_data,
            &header,
            &ds_dims,
            &chunk_dims,
            8,
            offset_size,
            length_size,
        )
        .unwrap();

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].address, 0x1000);
        assert_eq!(chunks[0].chunk_size, 120);
        assert_eq!(chunks[0].filter_mask, 0);
        assert_eq!(chunks[0].offsets, vec![0]);
        assert_eq!(chunks[1].address, 0x2000);
        assert_eq!(chunks[1].chunk_size, 115);
        assert_eq!(chunks[2].address, 0x3000);
        assert_eq!(chunks[2].chunk_size, 100);

        #[cfg(feature = "std")]
        assert_fa_streams_match(&file_data, fahd_offset, &ds_dims, &chunk_dims, 8, 8, 8);
    }

    /// Build a synthetic paged (non-filtered) Fixed Array and verify both that
    /// initialized pages are read and that pages with a cleared bitmap bit are
    /// skipped as entirely unallocated.
    ///
    /// `bitmap` is the single page-init byte (MSB-first: page 0 = 0x80).
    /// Returns the chunks decoded from a 3-element array split across two
    /// pages of size 2 (`max_nelmts_bits = 1`).
    fn read_paged(bitmap: u8) -> Vec<ChunkInfo> {
        let offset_size: u8 = 8;
        let length_size: u8 = 8;
        let os = offset_size as usize;
        let num_chunks = 3u64; // > page_size(2) => paged, npages = 2

        let mut file_data = vec![0u8; 0x400];

        let fahd_offset = 0x100usize;
        let db_offset = 0x200usize;
        file_data[fahd_offset..fahd_offset + 4].copy_from_slice(b"FAHD");
        file_data[fahd_offset + 4] = 0; // version
        file_data[fahd_offset + 5] = 0; // client_id = non-filtered
        file_data[fahd_offset + 6] = os as u8; // element_size
        file_data[fahd_offset + 7] = 1; // max_nelmts_bits => page_size = 2
        file_data[fahd_offset + 8..fahd_offset + 16].copy_from_slice(&num_chunks.to_le_bytes());
        file_data[fahd_offset + 16..fahd_offset + 24]
            .copy_from_slice(&(db_offset as u64).to_le_bytes());

        // FADB prefix: sig + version + client_id + header_addr + bitmap + checksum
        file_data[db_offset..db_offset + 4].copy_from_slice(b"FADB");
        file_data[db_offset + 4] = 0; // version
        file_data[db_offset + 5] = 0; // client_id
        file_data[db_offset + 6..db_offset + 14]
            .copy_from_slice(&(fahd_offset as u64).to_le_bytes());
        file_data[db_offset + 14] = bitmap; // page-init bitmap (1 byte for 2 pages)
        // checksum at db_offset+15..+19 left zero (reader does not validate)

        // Pages: stride = page_size(2)*elem_size(8) + 4 checksum = 20 bytes.
        let pages_start = db_offset + 14 + 1 + 4;
        let stride = 2 * os + 4;
        let addrs = [0x1000u64, 0x2000, 0x3000];
        for (i, &addr) in addrs.iter().enumerate() {
            let page = i / 2;
            let j = i % 2;
            let pos = pages_start + page * stride + j * os;
            file_data[pos..pos + os].copy_from_slice(&addr.to_le_bytes());
        }

        let header =
            FixedArrayHeader::parse(&file_data, fahd_offset, offset_size, length_size).unwrap();
        assert_eq!(header.num_elements, 3);
        let buffered =
            read_fixed_array_chunks(&file_data, &header, &[3], &[1], 8, offset_size, length_size)
                .unwrap();
        // The streaming reader must produce the same chunks for the paged layout.
        #[cfg(feature = "std")]
        assert_fa_streams_match(
            &file_data,
            fahd_offset,
            &[3],
            &[1],
            8,
            offset_size,
            length_size,
        );
        buffered
    }

    #[test]
    fn read_paged_all_pages_initialized() {
        // bitmap 0xC0 => both pages initialized (page 0 and page 1).
        let chunks = read_paged(0b1100_0000);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].address, 0x1000);
        assert_eq!(chunks[0].offsets, vec![0]);
        assert_eq!(chunks[1].address, 0x2000);
        assert_eq!(chunks[1].offsets, vec![1]);
        assert_eq!(chunks[2].address, 0x3000);
        assert_eq!(chunks[2].offsets, vec![2]);
    }

    #[test]
    fn read_paged_skips_uninitialized_page() {
        // bitmap 0x80 => only page 0 initialized; page 1's chunk is unallocated.
        let chunks = read_paged(0b1000_0000);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].address, 0x1000);
        assert_eq!(chunks[1].address, 0x2000);
    }

    /// `fixed_array_index_spans` must tile exactly the contiguous FAHD + FADB
    /// blob the writer produces, in both the non-paged and paged regimes and for
    /// filtered and unfiltered element records. This pins the reclaim sizing to
    /// `build_fixed_array_at` so the two cannot drift.
    #[cfg(feature = "std")]
    #[test]
    fn index_spans_tile_fixed_array_blob() {
        use crate::chunked_write::{WrittenChunk, build_fixed_array_at};

        let os: u8 = 8;
        let ls: u8 = 8;
        let base = 0x800u64;
        for has_filters in [false, true] {
            // 1024 = page_size, so 1025+ exercises the paged FADB layout.
            for &n in &[1u64, 5, 1024, 1025, 3000] {
                let chunks: Vec<WrittenChunk> = (0..n)
                    .map(|i| WrittenChunk {
                        address: 0x100000 + i * 8,
                        compressed_size: if has_filters { 8 + (i % 7) } else { 8 },
                        raw_size: 8,
                        filter_mask: 0,
                    })
                    .collect();
                let fa = build_fixed_array_at(&chunks, os, ls, has_filters, base);
                let mut file = vec![0u8; base as usize + fa.len()];
                file[base as usize..].copy_from_slice(&fa);

                let spans = fixed_array_index_spans(&file, base, os, ls).unwrap();
                assert_eq!(
                    spans.len(),
                    2,
                    "FA index = FAHD + FADB (filters={has_filters}, n={n})"
                );

                let mut sorted = spans.clone();
                sorted.sort_by_key(|&(a, _)| a);
                // FAHD at the base, FADB immediately after, together covering the
                // whole contiguous blob with no gap or overlap.
                assert_eq!(sorted[0].0, base);
                assert_eq!(
                    sorted[0].0 + sorted[0].1,
                    sorted[1].0,
                    "FAHD must abut FADB (filters={has_filters}, n={n})"
                );
                assert_eq!(
                    sorted[1].0 + sorted[1].1,
                    base + fa.len() as u64,
                    "FA index spans must tile the blob (filters={has_filters}, n={n})"
                );
            }
        }
    }
}
