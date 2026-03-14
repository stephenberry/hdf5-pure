//! HDF5 Fractal Heap parsing for v2 group link storage.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(feature = "checksum")]
use byteorder::{ByteOrder, LittleEndian};

use crate::error::FormatError;

/// Parsed fractal heap header (signature "FRHP").
#[derive(Debug, Clone)]
pub struct FractalHeapHeader {
    /// Length of heap IDs in bytes (typically 7).
    pub heap_id_length: u16,
    /// I/O filter encoded length (0 = no filters).
    pub io_filter_encoded_length: u16,
    /// Maximum size of a managed object.
    pub max_managed_object_size: u32,
    /// Width of the doubling table.
    pub table_width: u16,
    /// Starting block size in the doubling table.
    pub starting_block_size: u64,
    /// Maximum direct block size.
    pub max_direct_block_size: u64,
    /// Maximum heap size in bits (determines offset bit width in heap IDs).
    pub max_heap_size: u16,
    /// Starting row of indirect blocks in the doubling table.
    pub starting_row_of_indirect_blocks: u16,
    /// Address of the root block.
    pub root_block_address: u64,
    /// Number of rows in root indirect block (0 = root is direct block).
    pub current_rows_in_root_indirect_block: u16,
    /// Total number of managed objects.
    pub managed_objects_count: u64,
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
        2 => u16::from_le_bytes([data[pos], data[pos + 1]]) as u64,
        4 => u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as u64,
        8 => u64::from_le_bytes([
            data[pos], data[pos + 1], data[pos + 2], data[pos + 3],
            data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7],
        ]),
        _ => return Err(FormatError::InvalidOffsetSize(size)),
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

fn is_undefined(val: u64, offset_size: u8) -> bool {
    match offset_size {
        2 => val == 0xFFFF,
        4 => val == 0xFFFF_FFFF,
        8 => val == 0xFFFF_FFFF_FFFF_FFFF,
        _ => false,
    }
}

impl FractalHeapHeader {
    /// Parse a fractal heap header at the given offset.
    pub fn parse(
        file_data: &[u8],
        offset: usize,
        offset_size: u8,
        length_size: u8,
    ) -> Result<FractalHeapHeader, FormatError> {
        ensure_len(file_data, offset, 5)?;
        if &file_data[offset..offset + 4] != b"FRHP" {
            return Err(FormatError::InvalidFractalHeapSignature);
        }

        let version = file_data[offset + 4];
        if version != 0 {
            return Err(FormatError::InvalidFractalHeapVersion(version));
        }

        let os = offset_size as usize;
        let ls = length_size as usize;

        let mut pos = offset + 5;
        ensure_len(file_data, pos, 2)?;
        let heap_id_length = u16::from_le_bytes([file_data[pos], file_data[pos + 1]]);
        pos += 2;

        ensure_len(file_data, pos, 2)?;
        let io_filter_encoded_length = u16::from_le_bytes([file_data[pos], file_data[pos + 1]]);
        pos += 2;

        ensure_len(file_data, pos, 1)?;
        let _flags = file_data[pos];
        pos += 1;

        ensure_len(file_data, pos, 4)?;
        let max_managed_object_size = u32::from_le_bytes([
            file_data[pos], file_data[pos + 1], file_data[pos + 2], file_data[pos + 3],
        ]);
        pos += 4;

        // Skip several fixed fields: next_huge_object_id(ls), btree_huge_objects_address(os),
        // free_space_managed_blocks(ls), managed_block_free_space_manager_address(os),
        // managed_space_in_heap(ls), allocated_managed_space_in_heap(ls),
        // direct_block_allocation_iterator_offset(ls)
        let skip_size = 5 * ls + 2 * os;
        ensure_len(file_data, pos, skip_size)?;
        pos += skip_size;

        // managed_objects_count (length_size)
        let managed_objects_count = read_offset(file_data, pos, length_size)?;
        pos += ls;

        // huge_objects_size (length_size)
        pos += ls;
        // huge_objects_count (length_size)
        pos += ls;
        // tiny_objects_size (length_size)
        pos += ls;
        // tiny_objects_count (length_size)
        pos += ls;

        // table_width (2)
        ensure_len(file_data, pos, 2)?;
        let table_width = u16::from_le_bytes([file_data[pos], file_data[pos + 1]]);
        pos += 2;

        // starting_block_size (length_size)
        let starting_block_size = read_offset(file_data, pos, length_size)?;
        pos += ls;

        // max_direct_block_size (length_size)
        let max_direct_block_size = read_offset(file_data, pos, length_size)?;
        pos += ls;

        // max_heap_size (2)
        ensure_len(file_data, pos, 2)?;
        let max_heap_size = u16::from_le_bytes([file_data[pos], file_data[pos + 1]]);
        pos += 2;

        // starting_row_of_indirect_blocks (2)
        ensure_len(file_data, pos, 2)?;
        let starting_row_of_indirect_blocks =
            u16::from_le_bytes([file_data[pos], file_data[pos + 1]]);
        pos += 2;

        // root_block_address (offset_size)
        let root_block_address = read_offset(file_data, pos, offset_size)?;
        pos += os;

        // current_rows_in_root_indirect_block (2)
        ensure_len(file_data, pos, 2)?;
        let current_rows_in_root_indirect_block =
            u16::from_le_bytes([file_data[pos], file_data[pos + 1]]);
        #[allow(unused_variables, unused_mut, unused_assignments)]
        let mut pos = pos + 2;

        // Skip IO filter encoded info if present
        if io_filter_encoded_length > 0 {
            // root_block_filter_info_size (length_size) + filter_mask (4)
            #[allow(unused_assignments)]
            {
                pos += ls + 4;
            }
        }

        // Validate header checksum
        #[cfg(feature = "checksum")]
        {
            ensure_len(file_data, pos, 4)?;
            let stored = LittleEndian::read_u32(&file_data[pos..pos + 4]);
            let computed = crate::checksum::jenkins_lookup3(&file_data[offset..pos]);
            if computed != stored {
                return Err(FormatError::ChecksumMismatch {
                    expected: stored,
                    computed,
                });
            }
        }

        Ok(FractalHeapHeader {
            heap_id_length,
            io_filter_encoded_length,
            max_managed_object_size,
            table_width,
            starting_block_size,
            max_direct_block_size,
            max_heap_size,
            starting_row_of_indirect_blocks,
            root_block_address,
            current_rows_in_root_indirect_block,
            managed_objects_count,
        })
    }

    /// Decode a managed heap ID into (offset_in_heap, object_length).
    ///
    /// The heap ID layout for managed objects (type 0):
    /// - Byte 0: bits 6-7 = type (0), bits 4-5 = version (0), bits 0-3 = reserved
    /// - Bytes 1+: offset (max_heap_size bits, LE) then length (remaining bits, LE)
    pub fn decode_managed_id(&self, id_bytes: &[u8]) -> Result<(u64, u64), FormatError> {
        if id_bytes.is_empty() {
            return Err(FormatError::UnexpectedEof {
                expected: 1,
                available: 0,
            });
        }

        let id_type = (id_bytes[0] >> 6) & 0x03;
        if id_type != 0 {
            return Err(FormatError::InvalidHeapIdType(id_type));
        }

        // Bytes 1+ contain offset and length packed in little-endian order.
        // offset uses max_heap_size bits, length uses the remaining bits.
        let payload = &id_bytes[1..];
        let mut combined: u64 = 0;
        for (i, &b) in payload.iter().enumerate() {
            if i >= 8 {
                break;
            }
            combined |= (b as u64) << (i * 8);
        }

        let offset_bits = self.max_heap_size as u32;
        let offset_mask = if offset_bits >= 64 {
            u64::MAX
        } else {
            (1u64 << offset_bits) - 1
        };
        let heap_offset = combined & offset_mask;

        let total_payload_bits = (payload.len() as u32) * 8;
        let length_bits = total_payload_bits.saturating_sub(offset_bits);
        let length_val = if length_bits == 0 {
            0
        } else {
            let length_mask = if length_bits >= 64 {
                u64::MAX
            } else {
                (1u64 << length_bits) - 1
            };
            (combined >> offset_bits) & length_mask
        };

        Ok((heap_offset, length_val))
    }

    /// Read a managed object from the heap given its raw heap ID bytes.
    pub fn read_managed_object(
        &self,
        file_data: &[u8],
        id_bytes: &[u8],
        offset_size: u8,
    ) -> Result<Vec<u8>, FormatError> {
        let (heap_offset, obj_len) = self.decode_managed_id(id_bytes)?;

        if is_undefined(self.root_block_address, offset_size) {
            return Err(FormatError::UnexpectedEof {
                expected: 1,
                available: 0,
            });
        }

        if self.current_rows_in_root_indirect_block == 0 {
            // Root is a direct block
            self.read_from_direct_block(
                file_data,
                self.root_block_address as usize,
                self.starting_block_size,
                0, // block offset in heap = 0 for root
                heap_offset,
                obj_len as usize,
                offset_size,
            )
        } else {
            // Root is an indirect block — limit recursion to 64 levels
            self.read_from_indirect_block(
                file_data,
                self.root_block_address as usize,
                self.current_rows_in_root_indirect_block,
                0, // block offset
                heap_offset,
                obj_len as usize,
                offset_size,
                64, // max recursion depth
            )
        }
    }

    /// Read an object from a direct block.
    ///
    /// The heap offset is relative to the start of the block (including its header),
    /// so we just add it to the block address minus the block's heap offset.
    #[allow(clippy::too_many_arguments)]
    fn read_from_direct_block(
        &self,
        file_data: &[u8],
        block_addr: usize,
        _block_size: u64,
        block_heap_offset: u64,
        target_offset: u64,
        length: usize,
        _offset_size: u8,
    ) -> Result<Vec<u8>, FormatError> {
        let local_offset = (target_offset - block_heap_offset) as usize;
        let pos = block_addr + local_offset;
        ensure_len(file_data, pos, length)?;
        Ok(file_data[pos..pos + length].to_vec())
    }

    /// Read an object by traversing an indirect block to find the right direct block.
    #[allow(clippy::too_many_arguments)]
    fn read_from_indirect_block(
        &self,
        file_data: &[u8],
        iblock_addr: usize,
        nrows: u16,
        iblock_heap_offset: u64,
        target_offset: u64,
        length: usize,
        offset_size: u8,
        depth_remaining: u16,
    ) -> Result<Vec<u8>, FormatError> {
        if depth_remaining == 0 {
            return Err(FormatError::ChunkedReadError(
                "fractal heap: maximum recursion depth exceeded".into(),
            ));
        }
        // Parse indirect block header
        ensure_len(file_data, iblock_addr, 4)?;
        if &file_data[iblock_addr..iblock_addr + 4] != b"FHIB" {
            return Err(FormatError::InvalidFractalHeapSignature);
        }

        let block_offset_bytes = (self.max_heap_size as usize).div_ceil(8);
        let iblock_header = 5 + offset_size as usize + block_offset_bytes;
        let mut pos = iblock_addr + iblock_header;

        // Compute block sizes for each row using the doubling table
        let tw = self.table_width as u64;
        
        let nrows_usize = nrows as usize;

        // Build table of (block_size, heap_offset) for each child entry
        let mut current_heap_offset = iblock_heap_offset;

        // Count direct block entries vs indirect block entries
        let start_indirect = self.starting_row_of_indirect_blocks as usize;

        // Read child addresses for direct block rows
        let max_direct_rows = nrows_usize.min(start_indirect);

        for row in 0..max_direct_rows {
            let block_size = self.block_size_for_row(row);

            for _col in 0..tw {
                let child_addr = read_offset(file_data, pos, offset_size)?;
                pos += offset_size as usize;

                if self.io_filter_encoded_length > 0 {
                    // filtered_size(length_size) + filter_mask(4)
                    // Skip for now - we don't handle filtered direct blocks in fractal heaps
                    pos += 4; // filter_mask - simplified
                }

                if !is_undefined(child_addr, offset_size) {
                    let block_end = current_heap_offset + block_size;
                    if target_offset >= current_heap_offset && target_offset < block_end {
                        return self.read_from_direct_block(
                            file_data,
                            child_addr as usize,
                            block_size,
                            current_heap_offset,
                            target_offset,
                            length,
                            offset_size,
                        );
                    }
                }
                current_heap_offset += block_size;
            }
        }

        // If we have indirect block rows
        for row in start_indirect..nrows_usize {
            let block_size = self.block_size_for_row(row);
            let child_nrows = row - start_indirect + 1;

            for _col in 0..tw {
                let child_addr = read_offset(file_data, pos, offset_size)?;
                pos += offset_size as usize;

                if !is_undefined(child_addr, offset_size) {
                    // Calculate total heap space covered by this indirect block child
                    let total_child_space = self.indirect_block_heap_size(child_nrows);
                    let block_end = current_heap_offset + total_child_space;
                    if target_offset >= current_heap_offset && target_offset < block_end {
                        return self.read_from_indirect_block(
                            file_data,
                            child_addr as usize,
                            child_nrows as u16,
                            current_heap_offset,
                            target_offset,
                            length,
                            offset_size,
                            depth_remaining - 1,
                        );
                    }
                    current_heap_offset += total_child_space;
                } else {
                    let total_child_space = self.indirect_block_heap_size(child_nrows);
                    current_heap_offset += total_child_space;
                }
            }
            let _ = block_size;
        }

        Err(FormatError::UnexpectedEof {
            expected: target_offset as usize + length,
            available: file_data.len(),
        })
    }

    /// Get block size for a given row in the doubling table.
    fn block_size_for_row(&self, row: usize) -> u64 {
        let sbs = self.starting_block_size;
        if row <= 1 {
            sbs
        } else {
            sbs * (1u64 << (row - 1))
        }
    }

    /// Total heap space covered by an indirect block with the given number of rows.
    fn indirect_block_heap_size(&self, nrows: usize) -> u64 {
        let tw = self.table_width as u64;
        let mut total = 0u64;
        for row in 0..nrows {
            total += self.block_size_for_row(row) * tw;
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal fractal heap with a single direct block at the root.
    /// Returns (file_data, FractalHeapHeader) where file_data contains
    /// the heap header at offset 0 and a direct block with known data.
    fn build_simple_heap(offset_size: u8, length_size: u8) -> (Vec<u8>, usize) {
        let os = offset_size as usize;
        let ls = length_size as usize;
        let max_heap_size: u16 = 16; // bits
        let block_offset_bytes = (max_heap_size as usize).div_ceil(8); // 2

        // Direct block at a known offset
        let dblock_offset = 256usize;
        let block_size: u64 = 128;

        // Build fractal heap header at offset 0
        let mut buf = vec![0u8; 1024];
        let mut pos = 0;
        buf[pos..pos + 4].copy_from_slice(b"FRHP");
        pos += 4;
        buf[pos] = 0; // version
        pos += 1;
        // heap_id_length = 7
        buf[pos..pos + 2].copy_from_slice(&7u16.to_le_bytes());
        pos += 2;
        // io_filter_encoded_length = 0
        buf[pos..pos + 2].copy_from_slice(&0u16.to_le_bytes());
        pos += 2;
        // flags = 0
        buf[pos] = 0;
        pos += 1;
        // max_managed_object_size
        buf[pos..pos + 4].copy_from_slice(&64u32.to_le_bytes());
        pos += 4;
        // next_huge_object_id (length_size)
        pos += ls;
        // btree_huge_objects_address (offset_size) - undefined
        for i in 0..os { buf[pos + i] = 0xFF; }
        pos += os;
        // free_space_managed_blocks (length_size)
        pos += ls;
        // managed_block_free_space_manager_address (offset_size) - undefined
        for i in 0..os { buf[pos + i] = 0xFF; }
        pos += os;
        // managed_space_in_heap (length_size)
        pos += ls;
        // allocated_managed_space_in_heap (length_size)
        pos += ls;
        // direct_block_allocation_iterator_offset (length_size)
        pos += ls;
        // managed_objects_count (length_size) = 1
        buf[pos] = 1;
        pos += ls;
        // huge_objects_size (length_size)
        pos += ls;
        // huge_objects_count (length_size)
        pos += ls;
        // tiny_objects_size (length_size)
        pos += ls;
        // tiny_objects_count (length_size)
        pos += ls;
        // table_width = 4
        buf[pos..pos + 2].copy_from_slice(&4u16.to_le_bytes());
        pos += 2;
        // starting_block_size (length_size)
        match length_size {
            4 => buf[pos..pos + 4].copy_from_slice(&(block_size as u32).to_le_bytes()),
            8 => buf[pos..pos + 8].copy_from_slice(&block_size.to_le_bytes()),
            _ => {}
        }
        pos += ls;
        // max_direct_block_size (length_size) = 1024
        match length_size {
            4 => buf[pos..pos + 4].copy_from_slice(&1024u32.to_le_bytes()),
            8 => buf[pos..pos + 8].copy_from_slice(&1024u64.to_le_bytes()),
            _ => {}
        }
        pos += ls;
        // max_heap_size (2) = 16
        buf[pos..pos + 2].copy_from_slice(&max_heap_size.to_le_bytes());
        pos += 2;
        // starting_row_of_indirect_blocks (2) = 2
        buf[pos..pos + 2].copy_from_slice(&2u16.to_le_bytes());
        pos += 2;
        // root_block_address (offset_size) = dblock_offset
        match offset_size {
            4 => buf[pos..pos + 4].copy_from_slice(&(dblock_offset as u32).to_le_bytes()),
            8 => buf[pos..pos + 8].copy_from_slice(&(dblock_offset as u64).to_le_bytes()),
            _ => {}
        }
        pos += os;
        // current_rows_in_root_indirect_block (2) = 0 (root is direct)
        buf[pos..pos + 2].copy_from_slice(&0u16.to_le_bytes());
        pos += 2;
        // checksum
        let checksum = crate::checksum::jenkins_lookup3(&buf[0..pos]);
        buf[pos..pos + 4].copy_from_slice(&checksum.to_le_bytes());
        pos += 4;
        let header_end = pos;

        // Build direct block at dblock_offset
        pos = dblock_offset;
        buf[pos..pos + 4].copy_from_slice(b"FHDB");
        pos += 4;
        buf[pos] = 0; // version
        pos += 1;
        // heap_header_address (offset_size) = 0
        pos += os;
        // block_offset (block_offset_bytes) = 0
        pos += block_offset_bytes;
        // Data starts here - write known pattern
        let data_start = pos;
        // Write "Hello, World!" at offset 0 in the data area
        let test_data = b"Hello, World!";
        buf[data_start..data_start + test_data.len()].copy_from_slice(test_data);

        (buf, header_end)
    }

    #[test]
    fn parse_header() {
        let (file_data, _) = build_simple_heap(8, 8);
        let hdr = FractalHeapHeader::parse(&file_data, 0, 8, 8).unwrap();
        assert_eq!(hdr.heap_id_length, 7);
        assert_eq!(hdr.io_filter_encoded_length, 0);
        assert_eq!(hdr.max_managed_object_size, 64);
        assert_eq!(hdr.table_width, 4);
        assert_eq!(hdr.starting_block_size, 128);
        assert_eq!(hdr.max_heap_size, 16);
        assert_eq!(hdr.current_rows_in_root_indirect_block, 0);
        assert_eq!(hdr.managed_objects_count, 1);
    }

    #[test]
    fn decode_managed_id() {
        let (file_data, _) = build_simple_heap(8, 8);
        let hdr = FractalHeapHeader::parse(&file_data, 0, 8, 8).unwrap();

        // Build a managed heap ID:
        // byte 0: type=0 (bits 6-7 = 00), version=0 (bits 4-5), reserved (bits 0-3)
        // bytes 1-6: offset (max_heap_size=16 bits) then length (remaining bits)
        // For offset=0, length=13:
        // payload = offset | (length << 16) = 0 | (13 << 16) = 0x000D0000
        let offset: u64 = 0;
        let length: u64 = 13;
        let payload = offset | (length << hdr.max_heap_size);
        let mut id = vec![0u8; 7];
        id[0] = 0x00; // type=0
        for i in 0..6 {
            id[1 + i] = ((payload >> (i * 8)) & 0xFF) as u8;
        }

        let (off, len) = hdr.decode_managed_id(&id).unwrap();
        assert_eq!(off, 0);
        assert_eq!(len, 13);
    }

    #[test]
    fn read_managed_object_from_direct_block() {
        let (file_data, _) = build_simple_heap(8, 8);
        let hdr = FractalHeapHeader::parse(&file_data, 0, 8, 8).unwrap();

        // Build heap ID for the test data written in build_simple_heap.
        // The test data "Hello, World!" is at the data area of the direct block.
        // The direct block header is 5 + 8 + 2 = 15 bytes (for max_heap_size=16, ceil(16/8)=2).
        // Wait, max_heap_size=16, ceil(16/8)=2. Header = sig(4)+ver(1)+addr(8)+bo(2) = 15.
        // The data was placed at data_start = block_addr + 15.
        // Since offset is from block start, the object is at offset 15 within the block.
        let dblock_header_size = 5 + 8 + ((hdr.max_heap_size as usize + 7) / 8); // 15
        let offset: u64 = dblock_header_size as u64;
        let length: u64 = 13;
        let payload = offset | (length << hdr.max_heap_size);
        let mut id = vec![0u8; 7];
        id[0] = 0x00;
        for i in 0..6 {
            id[1 + i] = ((payload >> (i * 8)) & 0xFF) as u8;
        }

        let obj = hdr.read_managed_object(&file_data, &id, 8).unwrap();
        assert_eq!(&obj, b"Hello, World!");
    }

    #[test]
    fn invalid_signature() {
        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(b"XXXX");
        let err = FractalHeapHeader::parse(&data, 0, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidFractalHeapSignature);
    }

    #[test]
    fn invalid_version() {
        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(b"FRHP");
        data[4] = 1; // bad version
        let err = FractalHeapHeader::parse(&data, 0, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidFractalHeapVersion(1));
    }

    #[test]
    fn invalid_heap_id_type() {
        let (file_data, _) = build_simple_heap(8, 8);
        let hdr = FractalHeapHeader::parse(&file_data, 0, 8, 8).unwrap();
        // Type = 1 (tiny) in bits 6-7
        let id = vec![0x40u8, 0, 0, 0, 0, 0, 0]; // bit 6 set = type 1
        let err = hdr.decode_managed_id(&id).unwrap_err();
        assert_eq!(err, FormatError::InvalidHeapIdType(1));
    }
}
