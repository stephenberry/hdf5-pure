//! HDF5 Fractal Heap parsing for v2 group link storage.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(feature = "checksum")]
use byteorder::{ByteOrder, LittleEndian};

use crate::btree_v2::{
    BTreeV2Header, BTreeV2Record, collect_btree_v2_records, collect_btree_v2_records_from_source,
};
use crate::convert::TryToUsize;
use crate::error::FormatError;
use crate::source::FileSource;

/// The kind of object a fractal-heap heap ID refers to, encoded in bits 4-5 of
/// the heap ID's first byte (bits 6-7 are the format version, which must be 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HeapIdType {
    /// Stored in the heap's managed direct/indirect blocks.
    Managed,
    /// Too large to manage; stored directly in the file and indexed by the
    /// heap's huge-objects v2 B-tree (or, for wide heap IDs, inline).
    Huge,
    /// Small enough to be stored directly inside the heap ID.
    Tiny,
}

/// A child entry of a fractal-heap indirect block: either a direct block (a leaf
/// holding object bytes) or a nested indirect block. Returned by
/// [`FractalHeapHeader::find_child_for_offset`] so the buffered and streaming
/// readers share the indirect-block navigation logic.
enum HeapChild {
    Direct {
        addr: u64,
        block_size: u64,
        heap_offset: u64,
    },
    Indirect {
        addr: u64,
        nrows: u16,
        heap_offset: u64,
    },
}

/// Parsed fractal heap header (signature "FRHP").
///
/// Several header fields are decoded for on-disk-format completeness but are
/// not consulted by the current heap reader; kept to document the format.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct FractalHeapHeader {
    /// Length of heap IDs in bytes (typically 7).
    pub heap_id_length: u16,
    /// I/O filter encoded length (0 = no filters).
    pub io_filter_encoded_length: u16,
    /// Maximum size of a managed object. Objects larger than this are stored
    /// as "huge" objects outside the managed direct/indirect blocks.
    pub max_managed_object_size: u32,
    /// Address of the v2 B-tree that indexes "huge" objects (objects too large
    /// to be managed). Undefined (all-ones) when the heap has no huge objects.
    pub btree_huge_objects_address: u64,
    /// Width of the doubling table.
    pub table_width: u16,
    /// Starting block size in the doubling table.
    pub starting_block_size: u64,
    /// Maximum direct block size.
    pub max_direct_block_size: u64,
    /// Maximum heap size in bits (determines offset bit width in heap IDs).
    pub max_heap_size: u16,
    /// Starting number of rows in the root indirect block (HDF5's
    /// `start_root_rows`): an allocation hint for how many rows the root
    /// indirect block begins with, **not** the direct/indirect row boundary.
    /// The boundary is computed by [`FractalHeapHeader::max_direct_rows`]. This
    /// field is decoded for format completeness but not consulted by the reader.
    pub start_root_rows: u16,
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
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3],
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
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

/// Floor of the base-2 logarithm of `v` (0 for `v == 0`). The doubling-table
/// block sizes are powers of two, so this is their exact log2; it mirrors
/// HDF5's `H5VM_log2_gen`.
fn log2_floor(v: u64) -> u32 {
    if v == 0 { 0 } else { 63 - v.leading_zeros() }
}

/// Read a little-endian integer from up to the first 8 bytes of `payload`. Used
/// for the variable-width "huge object ID" packed into a heap ID.
fn read_var_le(payload: &[u8]) -> u64 {
    let mut value = 0u64;
    for (i, &b) in payload.iter().take(8).enumerate() {
        value |= (b as u64) << (i * 8);
    }
    value
}

/// Copy `len` bytes at `addr` out of an in-memory file image, bounds-checked.
fn slice_object(file_data: &[u8], addr: u64, len: usize) -> Result<Vec<u8>, FormatError> {
    let start = addr.to_usize()?;
    let end = start.checked_add(len).ok_or(FormatError::OffsetOverflow {
        offset: addr,
        length: len as u64,
    })?;
    if end > file_data.len() {
        return Err(FormatError::UnexpectedEof {
            expected: end,
            available: file_data.len(),
        });
    }
    Ok(file_data[start..end].to_vec())
}

/// Read `len` bytes at `addr` from a streaming source.
fn read_object_at_source<S: FileSource + ?Sized>(
    source: &S,
    addr: u64,
    len: usize,
) -> Result<Vec<u8>, FormatError> {
    source.read_metadata_at(addr, len)
}

/// Find the (address, length) of a huge object in the heap's huge-objects v2
/// B-tree records. Each record (type 1, indirectly accessed, non-filtered) is
/// `address(offset_size) + length(length_size) + id(length_size)`, sorted by id.
fn find_huge_record(
    records: &[BTreeV2Record],
    huge_id: u64,
    offset_size: u8,
    length_size: u8,
) -> Result<(u64, u64), FormatError> {
    let os = offset_size as usize;
    let ls = length_size as usize;
    for record in records {
        let data = &record.data;
        if data.len() < os + 2 * ls {
            continue;
        }
        let id = read_offset(data, os + ls, length_size)?;
        if id == huge_id {
            let addr = read_offset(data, 0, offset_size)?;
            let len = read_offset(data, os, length_size)?;
            return Ok((addr, len));
        }
    }
    Err(FormatError::HugeObjectNotFound(huge_id))
}

/// Decode a "tiny" object whose bytes are stored directly inside the heap ID.
/// HDF5 uses a short form (length in the low nibble of byte 0) and, when the ID
/// is wide enough that the data would not otherwise fit, an extended form
/// (12-bit length across bytes 0-1). Per `H5HFtiny.c` the short form is kept
/// while `heap_id_length - 1 <= 16` (i.e. `heap_id_length <= 17`); the extended
/// form begins at `heap_id_length == 18`. The format version (bits 6-7) must be 0.
fn read_tiny_object(heap_id_length: u16, id_bytes: &[u8]) -> Result<Vec<u8>, FormatError> {
    const TINY_LEN_SHORT: u16 = 16;
    let extended = heap_id_length.saturating_sub(1) > TINY_LEN_SHORT;
    let (len, data_start) = if extended {
        if id_bytes.len() < 2 {
            return Err(FormatError::UnexpectedEof {
                expected: 2,
                available: id_bytes.len(),
            });
        }
        let len = ((((id_bytes[0] & 0x0F) as usize) << 8) | id_bytes[1] as usize) + 1;
        (len, 2)
    } else {
        let len = (id_bytes[0] & 0x0F) as usize + 1;
        (len, 1)
    };
    let end = data_start + len;
    if end > id_bytes.len() {
        return Err(FormatError::UnexpectedEof {
            expected: end,
            available: id_bytes.len(),
        });
    }
    Ok(id_bytes[data_start..end].to_vec())
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
            file_data[pos],
            file_data[pos + 1],
            file_data[pos + 2],
            file_data[pos + 3],
        ]);
        pos += 4;

        // next_huge_object_id (length_size) — skip
        ensure_len(file_data, pos, ls)?;
        pos += ls;

        // btree_huge_objects_address (offset_size) — root of the v2 B-tree that
        // indexes "huge" objects (used when a stored object exceeds
        // max_managed_object_size, e.g. links/attributes with very long names).
        let btree_huge_objects_address = read_offset(file_data, pos, offset_size)?;
        pos += os;

        // Skip the remaining fixed fields: free_space_managed_blocks(ls),
        // managed_block_free_space_manager_address(os), managed_space_in_heap(ls),
        // allocated_managed_space_in_heap(ls),
        // direct_block_allocation_iterator_offset(ls)
        let skip_size = 4 * ls + os;
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

        // start_root_rows: starting # of rows in the root indirect block (2)
        ensure_len(file_data, pos, 2)?;
        let start_root_rows = u16::from_le_bytes([file_data[pos], file_data[pos + 1]]);
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
            btree_huge_objects_address,
            table_width,
            starting_block_size,
            max_direct_block_size,
            max_heap_size,
            start_root_rows,
            root_block_address,
            current_rows_in_root_indirect_block,
            managed_objects_count,
        })
    }

    /// Decode a managed heap ID into (offset_in_heap, object_length).
    ///
    /// The heap ID layout for managed objects (type 0):
    /// - Byte 0: bits 6-7 = version (0), bits 4-5 = type (0), bits 0-3 = reserved
    /// - Bytes 1+: offset (max_heap_size bits, LE) then length (remaining bits, LE)
    pub fn decode_managed_id(&self, id_bytes: &[u8]) -> Result<(u64, u64), FormatError> {
        if id_bytes.is_empty() {
            return Err(FormatError::UnexpectedEof {
                expected: 1,
                available: 0,
            });
        }

        // Object type lives in bits 4-5 of byte 0 (bits 6-7 are the version).
        let id_type = (id_bytes[0] >> 4) & 0x03;
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

        #[expect(
            clippy::cast_possible_truncation,
            reason = "payload is a single managed-object heap entry; its bit length fits u32"
        )]
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
                self.root_block_address.to_usize()?,
                self.starting_block_size,
                0, // block offset in heap = 0 for root
                heap_offset,
                obj_len.to_usize()?,
                offset_size,
            )
        } else {
            // Root is an indirect block — limit recursion to 64 levels
            self.read_from_indirect_block(
                file_data,
                self.root_block_address.to_usize()?,
                self.current_rows_in_root_indirect_block,
                0, // block offset
                heap_offset,
                obj_len.to_usize()?,
                offset_size,
                64, // max recursion depth
            )
        }
    }

    /// Classify a heap ID by its type bits (bits 4-5 of byte 0). Bits 6-7 carry
    /// the format version, which must be 0.
    fn heap_id_type(id_bytes: &[u8]) -> Result<HeapIdType, FormatError> {
        let byte0 = *id_bytes.first().ok_or(FormatError::UnexpectedEof {
            expected: 1,
            available: 0,
        })?;
        match (byte0 >> 4) & 0x03 {
            0 => Ok(HeapIdType::Managed),
            1 => Ok(HeapIdType::Huge),
            2 => Ok(HeapIdType::Tiny),
            other => Err(FormatError::InvalidHeapIdType(other)),
        }
    }

    /// Whether this heap stores "huge" object addresses and lengths inline in the
    /// heap ID (`huge_ids_direct` in HDF5) rather than behind the huge-objects
    /// v2 B-tree. The writer makes this choice from the heap ID length and the
    /// address/length widths; a reader must recompute it the same way
    /// (H5HFhuge.c, `H5HF__huge_init`).
    fn huge_ids_direct(&self, offset_size: u8, length_size: u8) -> bool {
        let avail = (self.heap_id_length as usize).saturating_sub(1);
        if self.io_filter_encoded_length > 0 {
            avail >= offset_size as usize + length_size as usize + 4 + length_size as usize
        } else {
            avail >= offset_size as usize + length_size as usize
        }
    }

    /// Read an object from the heap given its raw heap ID bytes, dispatching on
    /// the heap-ID type. Managed objects live in the doubling-table blocks; huge
    /// objects are stored directly in the file (resolved here through the
    /// huge-objects v2 B-tree); tiny objects are encoded in the ID itself.
    pub fn read_object(
        &self,
        file_data: &[u8],
        id_bytes: &[u8],
        offset_size: u8,
        length_size: u8,
    ) -> Result<Vec<u8>, FormatError> {
        match Self::heap_id_type(id_bytes)? {
            HeapIdType::Managed => self.read_managed_object(file_data, id_bytes, offset_size),
            HeapIdType::Huge => {
                self.read_huge_object(file_data, id_bytes, offset_size, length_size)
            }
            HeapIdType::Tiny => read_tiny_object(self.heap_id_length, id_bytes),
        }
    }

    /// Resolve and read a "huge" object given its heap ID.
    fn read_huge_object(
        &self,
        file_data: &[u8],
        id_bytes: &[u8],
        offset_size: u8,
        length_size: u8,
    ) -> Result<Vec<u8>, FormatError> {
        if self.io_filter_encoded_length > 0 {
            return Err(FormatError::UnsupportedFilteredHeapObject);
        }
        let payload = &id_bytes[1..];

        if self.huge_ids_direct(offset_size, length_size) {
            // The address and length are stored inline in the heap ID.
            let addr = read_offset(payload, 0, offset_size)?;
            let len = read_offset(payload, offset_size as usize, length_size)?;
            return slice_object(file_data, addr, len.to_usize()?);
        }

        // Indirect: the heap ID holds a B-tree key (the huge object ID); the
        // huge-objects v2 B-tree maps it to (address, length).
        let huge_id = read_var_le(payload);
        if is_undefined(self.btree_huge_objects_address, offset_size) {
            return Err(FormatError::HugeObjectNotFound(huge_id));
        }
        let btree_addr = self.btree_huge_objects_address.to_usize()?;
        let header = BTreeV2Header::parse(file_data, btree_addr, offset_size, length_size)?;
        let records = collect_btree_v2_records(file_data, &header, offset_size, length_size)?;
        let (addr, len) = find_huge_record(&records, huge_id, offset_size, length_size)?;
        slice_object(file_data, addr, len.to_usize()?)
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
        let local_offset = (target_offset - block_heap_offset).to_usize()?;
        let pos = block_addr
            .checked_add(local_offset)
            .ok_or(FormatError::OffsetOverflow {
                offset: block_addr as u64,
                length: target_offset - block_heap_offset,
            })?;
        ensure_len(file_data, pos, length)?;
        Ok(file_data[pos..pos + length].to_vec())
    }

    /// Locate the indirect-block child whose heap range contains `target_offset`.
    ///
    /// `block` is the indirect block's bytes (offset 0 = the "FHIB" signature).
    /// Walks the direct-block rows then the indirect-block rows, accumulating the
    /// heap-offset cursor exactly as the on-disk doubling table prescribes, and
    /// returns the matching child (or `None`). Shared by the buffered and
    /// streaming readers so this navigation lives in one place.
    fn find_child_for_offset(
        &self,
        block: &[u8],
        nrows: u16,
        iblock_heap_offset: u64,
        target_offset: u64,
        offset_size: u8,
    ) -> Result<Option<HeapChild>, FormatError> {
        ensure_len(block, 0, 4)?;
        if &block[0..4] != b"FHIB" {
            return Err(FormatError::InvalidFractalHeapSignature);
        }

        let block_offset_bytes = (self.max_heap_size as usize).div_ceil(8);
        let iblock_header = 5 + offset_size as usize + block_offset_bytes;
        let mut pos = iblock_header;
        let tw = self.table_width as u64;
        let nrows_usize = nrows as usize;
        // The direct/indirect boundary is computed from the doubling table, not
        // taken from the `start_root_rows` header field (a common confusion that
        // mis-reads any heap whose rows extend past that hint).
        let direct_rows = nrows_usize.min(self.max_direct_rows());
        let mut current_heap_offset = iblock_heap_offset;

        // Direct-block rows.
        for row in 0..direct_rows {
            let block_size = self.block_size_for_row(row);
            for _col in 0..tw {
                let child_addr = read_offset(block, pos, offset_size)?;
                pos += offset_size as usize;
                if self.io_filter_encoded_length > 0 {
                    // filtered_size + filter_mask — filtered direct blocks are not
                    // yet supported; skip the (simplified) trailing field.
                    pos += 4;
                }
                if !is_undefined(child_addr, offset_size) {
                    let block_end = current_heap_offset.saturating_add(block_size);
                    if target_offset >= current_heap_offset && target_offset < block_end {
                        return Ok(Some(HeapChild::Direct {
                            addr: child_addr,
                            block_size,
                            heap_offset: current_heap_offset,
                        }));
                    }
                }
                current_heap_offset = current_heap_offset.saturating_add(block_size);
            }
        }

        // Indirect-block rows. A child indirect block occupying parent row `row`
        // has that row's doubling-table block size; its own row count and the
        // heap space it spans follow from that size.
        for row in direct_rows..nrows_usize {
            let child_nrows = self.size_to_rows(self.block_size_for_row(row));
            let total_child_space = self.indirect_block_heap_size(child_nrows);
            for _col in 0..tw {
                let child_addr = read_offset(block, pos, offset_size)?;
                pos += offset_size as usize;
                if !is_undefined(child_addr, offset_size) {
                    let block_end = current_heap_offset.saturating_add(total_child_space);
                    if target_offset >= current_heap_offset && target_offset < block_end {
                        #[expect(
                            clippy::cast_possible_truncation,
                            reason = "fractal-heap row count is log-scale (bounded by \
                                      max_heap_size bits), so it fits u16"
                        )]
                        return Ok(Some(HeapChild::Indirect {
                            addr: child_addr,
                            nrows: child_nrows as u16,
                            heap_offset: current_heap_offset,
                        }));
                    }
                }
                current_heap_offset = current_heap_offset.saturating_add(total_child_space);
            }
        }

        Ok(None)
    }

    /// Byte length of the entry region of an indirect block with `nrows` rows
    /// (header + per-row child pointers), used to size the read window for the
    /// streaming reader.
    fn indirect_block_entries_len(&self, nrows: u16, offset_size: u8) -> usize {
        let block_offset_bytes = (self.max_heap_size as usize).div_ceil(8);
        let iblock_header = 5 + offset_size as usize + block_offset_bytes;
        let nrows_usize = nrows as usize;
        let boundary = self.max_direct_rows();
        let direct_rows = nrows_usize.min(boundary);
        let num_indirect_rows = nrows_usize.saturating_sub(boundary);
        let direct_entry = offset_size as usize
            + if self.io_filter_encoded_length > 0 {
                4
            } else {
                0
            };
        let tw = self.table_width as usize;
        iblock_header
            + direct_rows * tw * direct_entry
            + num_indirect_rows * tw * (offset_size as usize)
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
        ensure_len(file_data, iblock_addr, 4)?;
        let block = &file_data[iblock_addr..];
        match self.find_child_for_offset(
            block,
            nrows,
            iblock_heap_offset,
            target_offset,
            offset_size,
        )? {
            Some(HeapChild::Direct {
                addr,
                block_size,
                heap_offset,
            }) => self.read_from_direct_block(
                file_data,
                addr.to_usize()?,
                block_size,
                heap_offset,
                target_offset,
                length,
                offset_size,
            ),
            Some(HeapChild::Indirect {
                addr,
                nrows: child_nrows,
                heap_offset,
            }) => self.read_from_indirect_block(
                file_data,
                addr.to_usize()?,
                child_nrows,
                heap_offset,
                target_offset,
                length,
                offset_size,
                depth_remaining - 1,
            ),
            None => Err(FormatError::UnexpectedEof {
                expected: target_offset.to_usize()?.saturating_add(length),
                available: file_data.len(),
            }),
        }
    }

    // -----------------------------------------------------------------------
    // Streaming readers (fetch each block from a `FileSource` on demand)
    // -----------------------------------------------------------------------

    /// Parse a fractal heap header from a [`FileSource`] (small bounded window).
    pub fn parse_from_source<S: FileSource + ?Sized>(
        source: &S,
        address: u64,
        offset_size: u8,
        length_size: u8,
    ) -> Result<FractalHeapHeader, FormatError> {
        // The FRHP header is bounded: with 8-byte offsets/lengths it is ~170
        // bytes; 256 is a safe window.
        const MAX_HEADER: u64 = 256;
        let window = MAX_HEADER
            .min(source.len().saturating_sub(address))
            .to_usize()?;
        let buf = source.read_metadata_at(address, window)?;
        Self::parse(&buf, 0, offset_size, length_size)
    }

    /// Read a managed object from the heap (by raw heap ID) via a [`FileSource`].
    pub fn read_managed_object_from_source<S: FileSource + ?Sized>(
        &self,
        source: &S,
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
            self.read_from_direct_block_from_source(
                source,
                self.root_block_address,
                0, // root direct block starts at heap offset 0
                heap_offset,
                obj_len.to_usize()?,
            )
        } else {
            self.read_from_indirect_block_from_source(
                source,
                self.root_block_address,
                self.current_rows_in_root_indirect_block,
                0,
                heap_offset,
                obj_len.to_usize()?,
                offset_size,
                64,
            )
        }
    }

    /// Streaming counterpart to [`FractalHeapHeader::read_object`].
    pub fn read_object_from_source<S: FileSource + ?Sized>(
        &self,
        source: &S,
        id_bytes: &[u8],
        offset_size: u8,
        length_size: u8,
    ) -> Result<Vec<u8>, FormatError> {
        match Self::heap_id_type(id_bytes)? {
            HeapIdType::Managed => {
                self.read_managed_object_from_source(source, id_bytes, offset_size)
            }
            HeapIdType::Huge => {
                self.read_huge_object_from_source(source, id_bytes, offset_size, length_size)
            }
            HeapIdType::Tiny => read_tiny_object(self.heap_id_length, id_bytes),
        }
    }

    /// Resolve and read a "huge" object via a [`FileSource`].
    fn read_huge_object_from_source<S: FileSource + ?Sized>(
        &self,
        source: &S,
        id_bytes: &[u8],
        offset_size: u8,
        length_size: u8,
    ) -> Result<Vec<u8>, FormatError> {
        if self.io_filter_encoded_length > 0 {
            return Err(FormatError::UnsupportedFilteredHeapObject);
        }
        let payload = &id_bytes[1..];

        if self.huge_ids_direct(offset_size, length_size) {
            let addr = read_offset(payload, 0, offset_size)?;
            let len = read_offset(payload, offset_size as usize, length_size)?;
            return read_object_at_source(source, addr, len.to_usize()?);
        }

        let huge_id = read_var_le(payload);
        if is_undefined(self.btree_huge_objects_address, offset_size) {
            return Err(FormatError::HugeObjectNotFound(huge_id));
        }
        let header = BTreeV2Header::parse_from_source(
            source,
            self.btree_huge_objects_address,
            offset_size,
            length_size,
        )?;
        let records =
            collect_btree_v2_records_from_source(source, &header, offset_size, length_size)?;
        let (addr, len) = find_huge_record(&records, huge_id, offset_size, length_size)?;
        read_object_at_source(source, addr, len.to_usize()?)
    }

    fn read_from_direct_block_from_source<S: FileSource + ?Sized>(
        &self,
        source: &S,
        block_addr: u64,
        block_heap_offset: u64,
        target_offset: u64,
        length: usize,
    ) -> Result<Vec<u8>, FormatError> {
        let local_offset = target_offset - block_heap_offset;
        let pos = block_addr
            .checked_add(local_offset)
            .ok_or(FormatError::OffsetOverflow {
                offset: block_addr,
                length: local_offset,
            })?;
        source.read_metadata_at(pos, length)
    }

    #[allow(clippy::too_many_arguments)]
    fn read_from_indirect_block_from_source<S: FileSource + ?Sized>(
        &self,
        source: &S,
        iblock_addr: u64,
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
        let region_len = (self.indirect_block_entries_len(nrows, offset_size) as u64)
            .min(source.len().saturating_sub(iblock_addr))
            .to_usize()?;
        let block = source.read_metadata_at(iblock_addr, region_len)?;
        match self.find_child_for_offset(
            &block,
            nrows,
            iblock_heap_offset,
            target_offset,
            offset_size,
        )? {
            Some(HeapChild::Direct {
                addr, heap_offset, ..
            }) => self.read_from_direct_block_from_source(
                source,
                addr,
                heap_offset,
                target_offset,
                length,
            ),
            Some(HeapChild::Indirect {
                addr,
                nrows: child_nrows,
                heap_offset,
            }) => self.read_from_indirect_block_from_source(
                source,
                addr,
                child_nrows,
                heap_offset,
                target_offset,
                length,
                offset_size,
                depth_remaining - 1,
            ),
            None => Err(FormatError::UnexpectedEof {
                expected: target_offset.to_usize()?.saturating_add(length),
                available: source.len().to_usize().unwrap_or(usize::MAX),
            }),
        }
    }

    /// Get block size for a given row in the doubling table. Saturates rather
    /// than overflowing for an out-of-range `row`, so a malformed header (an
    /// implausibly large `current_rows`) cannot panic the shift or the multiply.
    fn block_size_for_row(&self, row: usize) -> u64 {
        let sbs = self.starting_block_size;
        if row <= 1 {
            sbs
        } else {
            match u32::try_from(row - 1)
                .ok()
                .and_then(|s| 1u64.checked_shl(s))
            {
                Some(mult) => sbs.saturating_mul(mult),
                None => u64::MAX,
            }
        }
    }

    /// Number of rows in the doubling table that hold **direct** blocks. Rows
    /// `[0, max_direct_rows)` are direct; rows at or beyond it are indirect.
    /// Computed from the doubling-table parameters exactly as HDF5 does
    /// (`H5HFdtable.c`: `(max_direct_bits - start_bits) + 2`) — not read from the
    /// `start_root_rows` header field, which is a different quantity.
    fn max_direct_rows(&self) -> usize {
        let start_bits = log2_floor(self.starting_block_size);
        let max_direct_bits = log2_floor(self.max_direct_block_size);
        (max_direct_bits.saturating_sub(start_bits) + 2) as usize
    }

    /// Number of rows in an indirect block that manages `size` bytes of heap
    /// space (HDF5's `H5HF__dtable_size_to_rows`:
    /// `(log2(size) - first_row_bits) + 1`, where
    /// `first_row_bits = log2(start_block_size) + log2(table_width)`).
    fn size_to_rows(&self, size: u64) -> usize {
        let first_row_bits =
            log2_floor(self.starting_block_size) + log2_floor(self.table_width as u64);
        (log2_floor(size).saturating_sub(first_row_bits) + 1) as usize
    }

    /// Total heap space covered by an indirect block with the given number of
    /// rows. Saturates so a malformed `nrows` cannot overflow the running total.
    fn indirect_block_heap_size(&self, nrows: usize) -> u64 {
        let tw = self.table_width as u64;
        let mut total = 0u64;
        for row in 0..nrows {
            total = total.saturating_add(self.block_size_for_row(row).saturating_mul(tw));
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
        for i in 0..os {
            buf[pos + i] = 0xFF;
        }
        pos += os;
        // free_space_managed_blocks (length_size)
        pos += ls;
        // managed_block_free_space_manager_address (offset_size) - undefined
        for i in 0..os {
            buf[pos + i] = 0xFF;
        }
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
        // start_root_rows (2) = 2
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
        let dblock_header_size = 5 + 8 + (hdr.max_heap_size as usize).div_ceil(8); // 15
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

    #[cfg(feature = "std")]
    #[test]
    fn streaming_managed_object_matches_buffered() {
        use crate::source::{BytesSource, ReadSeekSource};
        let (file_data, _) = build_simple_heap(8, 8);

        // Same heap ID as `read_managed_object_from_direct_block`.
        let hdr = FractalHeapHeader::parse(&file_data, 0, 8, 8).unwrap();
        let dblock_header_size = 5 + 8 + (hdr.max_heap_size as usize).div_ceil(8);
        let payload = (dblock_header_size as u64) | (13u64 << hdr.max_heap_size);
        let mut id = vec![0u8; 7];
        for i in 0..6 {
            id[1 + i] = ((payload >> (i * 8)) & 0xFF) as u8;
        }

        let buffered = hdr.read_managed_object(&file_data, &id, 8).unwrap();

        // Header parsed from a source, then the object fetched from a source.
        let mem = BytesSource::new(&file_data);
        let hdr_mem = FractalHeapHeader::parse_from_source(&mem, 0, 8, 8).unwrap();
        let from_mem = hdr_mem
            .read_managed_object_from_source(&mem, &id, 8)
            .unwrap();

        let seek = ReadSeekSource::new(std::io::Cursor::new(file_data)).unwrap();
        let hdr_seek = FractalHeapHeader::parse_from_source(&seek, 0, 8, 8).unwrap();
        let from_seek = hdr_seek
            .read_managed_object_from_source(&seek, &id, 8)
            .unwrap();

        assert_eq!(buffered, from_mem);
        assert_eq!(buffered, from_seek);
        assert_eq!(&from_seek, b"Hello, World!");
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
        // Type lives in bits 4-5; type 1 (huge) = 0x10. decode_managed_id only
        // accepts managed (type 0) IDs.
        let id = vec![0x10u8, 0, 0, 0, 0, 0, 0];
        let err = hdr.decode_managed_id(&id).unwrap_err();
        assert_eq!(err, FormatError::InvalidHeapIdType(1));
    }

    #[test]
    fn heap_id_type_reads_bits_4_5() {
        // Type is bits 4-5; the version (bits 6-7) must not be read as type.
        assert_eq!(
            FractalHeapHeader::heap_id_type(&[0x00]).unwrap(),
            HeapIdType::Managed
        );
        assert_eq!(
            FractalHeapHeader::heap_id_type(&[0x10]).unwrap(),
            HeapIdType::Huge
        );
        assert_eq!(
            FractalHeapHeader::heap_id_type(&[0x20]).unwrap(),
            HeapIdType::Tiny
        );
        // Reserved type 3.
        assert_eq!(
            FractalHeapHeader::heap_id_type(&[0x30]),
            Err(FormatError::InvalidHeapIdType(3))
        );
        // Version bits set (0xC0) must not change the decoded type.
        assert_eq!(
            FractalHeapHeader::heap_id_type(&[0xC0 | 0x10]).unwrap(),
            HeapIdType::Huge
        );
    }

    #[test]
    fn huge_ids_direct_matches_hdf5_rule() {
        // Unfiltered: direct when (id_len - 1) >= offset_size + length_size.
        let mut h = dtable_header(512, 65536, 4);
        h.heap_id_length = 7; // 6 payload bytes < 8 + 8 -> indirect (B-tree)
        assert!(!h.huge_ids_direct(8, 8));
        h.heap_id_length = 17; // 16 payload bytes == 8 + 8 -> direct
        assert!(h.huge_ids_direct(8, 8));
        // Filtered heaps need extra room (addr + len + filter_mask(4) + filtered
        // len); 17 is no longer enough.
        h.io_filter_encoded_length = 4;
        assert!(!h.huge_ids_direct(8, 8));
    }

    #[test]
    fn find_huge_record_matches_by_id() {
        // Type-1 record: address(8) + length(8) + id(8), little-endian.
        let rec = |addr: u64, len: u64, id: u64| {
            let mut d = Vec::new();
            d.extend_from_slice(&addr.to_le_bytes());
            d.extend_from_slice(&len.to_le_bytes());
            d.extend_from_slice(&id.to_le_bytes());
            BTreeV2Record { data: d }
        };
        let records = vec![
            rec(0x1000, 5000, 1),
            rec(0x2000, 6000, 2),
            rec(0x3000, 7000, 5),
        ];
        assert_eq!(find_huge_record(&records, 2, 8, 8).unwrap(), (0x2000, 6000));
        assert_eq!(find_huge_record(&records, 5, 8, 8).unwrap(), (0x3000, 7000));
        assert_eq!(
            find_huge_record(&records, 9, 8, 8),
            Err(FormatError::HugeObjectNotFound(9))
        );
    }

    #[test]
    fn read_tiny_object_short_and_extended() {
        // Short form (heap ID <= 16 bytes): low nibble of byte 0 is length - 1.
        let id = [0x20 | 0x03, b'a', b'b', b'c', b'd', 0, 0];
        assert_eq!(read_tiny_object(7, &id).unwrap(), b"abcd");
        // Extended form (heap ID >= 18 bytes): 12-bit length across bytes 0-1.
        // length 5 -> stored value 4 = 0x004: byte0 low nibble 0x0, byte1 0x04.
        let mut id = vec![0x20, 0x04];
        id.extend_from_slice(b"hello");
        id.resize(20, 0);
        assert_eq!(read_tiny_object(20, &id).unwrap(), b"hello");

        // Boundary: heap_id_length == 17 still uses the short form (HDF5 keeps
        // short while heap_id_length - 1 <= 16). Decoding it as extended would
        // misread the length, so this pins the off-by-one.
        let mut id = vec![0x20 | 0x04]; // short form, length 5
        id.extend_from_slice(b"world");
        id.resize(17, 0);
        assert_eq!(read_tiny_object(17, &id).unwrap(), b"world");
    }

    #[test]
    fn log2_floor_basics() {
        assert_eq!(log2_floor(0), 0);
        assert_eq!(log2_floor(1), 0);
        assert_eq!(log2_floor(512), 9);
        assert_eq!(log2_floor(65536), 16);
        assert_eq!(log2_floor(131072), 17);
        // Floor for non-powers-of-two (mirrors H5VM_log2_gen).
        assert_eq!(log2_floor(1023), 9);
        assert_eq!(log2_floor(1024), 10);
    }

    /// Build a header with the given doubling-table parameters (the only fields
    /// `max_direct_rows`/`size_to_rows` consult); other fields are irrelevant.
    fn dtable_header(
        start_block_size: u64,
        max_direct_block_size: u64,
        table_width: u16,
    ) -> FractalHeapHeader {
        FractalHeapHeader {
            heap_id_length: 7,
            io_filter_encoded_length: 0,
            max_managed_object_size: 0,
            btree_huge_objects_address: u64::MAX,
            table_width,
            starting_block_size: start_block_size,
            max_direct_block_size,
            max_heap_size: 64,
            start_root_rows: 1,
            root_block_address: 0,
            current_rows_in_root_indirect_block: 0,
            managed_objects_count: 0,
        }
    }

    #[test]
    fn max_direct_rows_matches_hdf5_formula() {
        // (log2(max_direct) - log2(start)) + 2, the boundary between direct and
        // indirect rows. Values cross-checked against HDF5's H5HFdtable.c.
        assert_eq!(dtable_header(512, 65536, 4).max_direct_rows(), 9); // (16-9)+2
        assert_eq!(dtable_header(4096, 65536, 4).max_direct_rows(), 6); // (16-12)+2
        // Degenerate: max-direct == start gives the minimum of 2 direct rows.
        assert_eq!(dtable_header(512, 512, 4).max_direct_rows(), 2);
    }

    #[test]
    fn size_to_rows_matches_hdf5_formula() {
        // first_row_bits = log2(start) + log2(width). For start=512, width=4:
        // first_row_bits = 9 + 2 = 11, rows = (log2(size) - 11) + 1.
        let h = dtable_header(512, 65536, 4);
        assert_eq!(h.size_to_rows(131072), 7); // 2^17: (17-11)+1
        assert_eq!(h.size_to_rows(4096), 2); // 2^12: (12-11)+1
        // Below first_row_bits saturates to a single row, never underflows.
        assert_eq!(h.size_to_rows(512), 1);
        assert_eq!(h.size_to_rows(1), 1);
    }
}
