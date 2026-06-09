//! HDF5 B-tree v2 parsing.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(feature = "checksum")]
use byteorder::{ByteOrder, LittleEndian};

use crate::convert::TryToUsize;
use crate::error::FormatError;
use crate::source::FileSource;

/// Parsed B-tree v2 header (signature "BTHD").
#[derive(Debug, Clone)]
pub struct BTreeV2Header {
    /// B-tree type: 5=links indexed by name, 6=links indexed by creation order, etc.
    pub tree_type: u8,
    /// Node size in bytes.
    pub node_size: u32,
    /// Record size in bytes.
    pub record_size: u16,
    /// Depth of the tree (0 = root is a leaf).
    pub depth: u16,
    /// Address of root node.
    pub root_node_address: u64,
    /// Number of records in the root node.
    pub num_records_in_root: u16,
    /// Total number of records in all nodes.
    pub total_records: u64,
}

/// A single record from a B-tree v2 node.
#[derive(Debug, Clone)]
pub struct BTreeV2Record {
    /// Raw record bytes (record_size bytes).
    pub data: Vec<u8>,
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

/// Compute the number of bytes needed to represent a count, using variable-width encoding.
/// B-tree v2 uses this for the number of records fields in internal nodes.
fn bytes_for_max_records(max_nrec: u64) -> usize {
    if max_nrec == 0 {
        return 1;
    }
    let bits = 64 - max_nrec.leading_zeros() as usize;
    bits.div_ceil(8)
}

/// Read a variable-width unsigned integer (1-8 bytes, LE).
fn read_var_uint(data: &[u8], pos: usize, width: usize) -> Result<u64, FormatError> {
    ensure_len(data, pos, width)?;
    let mut val = 0u64;
    for i in 0..width {
        val |= (data[pos + i] as u64) << (i * 8);
    }
    Ok(val)
}

impl BTreeV2Header {
    /// Parse a B-tree v2 header at the given offset.
    pub fn parse(
        file_data: &[u8],
        offset: usize,
        offset_size: u8,
        length_size: u8,
    ) -> Result<BTreeV2Header, FormatError> {
        ensure_len(file_data, offset, 4)?;
        if &file_data[offset..offset + 4] != b"BTHD" {
            return Err(FormatError::InvalidBTreeV2Signature);
        }

        ensure_len(file_data, offset, 4 + 1 + 1 + 4 + 2 + 2 + 1 + 1)?;
        let version = file_data[offset + 4];
        if version != 0 {
            return Err(FormatError::InvalidBTreeV2Version(version));
        }

        let tree_type = file_data[offset + 5];
        let node_size = u32::from_le_bytes([
            file_data[offset + 6],
            file_data[offset + 7],
            file_data[offset + 8],
            file_data[offset + 9],
        ]);
        let record_size = u16::from_le_bytes([file_data[offset + 10], file_data[offset + 11]]);
        let depth = u16::from_le_bytes([file_data[offset + 12], file_data[offset + 13]]);
        let _split_percent = file_data[offset + 14];
        let _merge_percent = file_data[offset + 15];

        let mut pos = offset + 16;
        let root_node_address = read_offset(file_data, pos, offset_size)?;
        pos += offset_size as usize;

        ensure_len(file_data, pos, 2)?;
        let num_records_in_root = u16::from_le_bytes([file_data[pos], file_data[pos + 1]]);
        pos += 2;

        let total_records = read_offset(file_data, pos, length_size)?;
        #[allow(unused_assignments)]
        {
            pos += length_size as usize;
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

        Ok(BTreeV2Header {
            tree_type,
            node_size,
            record_size,
            depth,
            root_node_address,
            num_records_in_root,
            total_records,
        })
    }

    /// Parse a B-tree v2 header from a [`FileSource`].
    ///
    /// The header is fully self-contained (signature + fixed fields + a root
    /// pointer + checksum), so only a small bounded window is read.
    pub fn parse_from_source<S: FileSource + ?Sized>(
        source: &S,
        address: u64,
        offset_size: u8,
        length_size: u8,
    ) -> Result<BTreeV2Header, FormatError> {
        // 16 fixed prefix bytes + root address + 2 (num records) + total-records
        // field + 4 checksum; <= 64 with 8-byte offsets/lengths.
        const MAX_HEADER: u64 = 64;
        let window = MAX_HEADER
            .min(source.len().saturating_sub(address))
            .to_usize()?;
        let buf = source.read_exact_at(address, window)?;
        Self::parse(&buf, 0, offset_size, length_size)
    }
}

/// Compute maximum records per node for a given depth level.
/// leaf: (node_size - overhead) / record_size
/// internal: depends on pointers
fn max_records_leaf(node_size: u32, record_size: u16) -> u64 {
    // Leaf overhead: signature(4) + version(1) + type(1) + checksum(4) = 10
    let overhead = 10u32;
    if node_size <= overhead || record_size == 0 {
        return 0;
    }
    ((node_size - overhead) / record_size as u32) as u64
}

/// Collect all records from a B-tree v2 by traversing from the root.
pub fn collect_btree_v2_records(
    file_data: &[u8],
    header: &BTreeV2Header,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<BTreeV2Record>, FormatError> {
    if header.total_records == 0 || header.num_records_in_root == 0 {
        return Ok(Vec::new());
    }

    let max_leaf_nrec = max_records_leaf(header.node_size, header.record_size);

    if header.depth == 0 {
        // Root is a leaf
        parse_leaf_records(
            file_data,
            header.root_node_address.to_usize()?,
            header.num_records_in_root,
            header.record_size,
        )
    } else {
        // Root is internal; traverse recursively
        let mut records = Vec::new();
        collect_internal_records(
            file_data,
            header.root_node_address.to_usize()?,
            header.num_records_in_root,
            header.depth,
            header.record_size,
            header.node_size,
            offset_size,
            length_size,
            max_leaf_nrec,
            &mut records,
        )?;
        Ok(records)
    }
}

/// Parse records from a leaf node (signature "BTLF").
fn parse_leaf_records(
    file_data: &[u8],
    offset: usize,
    num_records: u16,
    record_size: u16,
) -> Result<Vec<BTreeV2Record>, FormatError> {
    // signature(4) + version(1) + type(1) = 6 bytes header
    ensure_len(file_data, offset, 6)?;
    if &file_data[offset..offset + 4] != b"BTLF" {
        return Err(FormatError::InvalidBTreeV2Signature);
    }

    let pos = offset + 6;
    let rs = record_size as usize;
    let total = num_records as usize * rs;
    ensure_len(file_data, pos, total)?;

    // Validate checksum: 4 bytes after records + padding
    #[cfg(feature = "checksum")]
    {
        let checksum_pos = pos + total;
        if file_data.len() >= checksum_pos + 4 {
            let stored = LittleEndian::read_u32(&file_data[checksum_pos..checksum_pos + 4]);
            let computed = crate::checksum::jenkins_lookup3(&file_data[offset..checksum_pos]);
            if computed != stored {
                return Err(FormatError::ChecksumMismatch {
                    expected: stored,
                    computed,
                });
            }
        }
    }

    let mut records = Vec::with_capacity(num_records as usize);
    for i in 0..num_records as usize {
        let start = pos + i * rs;
        records.push(BTreeV2Record {
            data: file_data[start..start + rs].to_vec(),
        });
    }
    Ok(records)
}

/// Parse an internal node's child pointers from its node bytes (offset 0 = the
/// "BTIN" signature), returning the `(child_address, child_num_records)` list.
///
/// The node's own records sit at `node[6 + i * record_size ..]`; the caller
/// reads them while interleaving child traversals. Shared by the buffered and
/// streaming collectors so the child-pointer-width logic lives in one place.
fn parse_internal_child_pointers(
    node: &[u8],
    num_records: u16,
    depth: u16,
    record_size: u16,
    offset_size: u8,
    max_leaf_nrec: u64,
) -> Result<Vec<(u64, u16)>, FormatError> {
    // signature(4) + version(1) + type(1) = 6
    ensure_len(node, 0, 6)?;
    if &node[0..4] != b"BTIN" {
        return Err(FormatError::InvalidBTreeV2Signature);
    }

    let nr = num_records as usize;
    let rs = record_size as usize;
    // Records come first, then the child pointers.
    let mut pos = 6;
    ensure_len(node, pos, nr * rs)?;
    pos += nr * rs;

    // Child-pointer encoding widths (variable, per the HDF5 spec).
    let child_depth = depth - 1;
    let max_nrec_child = if child_depth == 0 {
        max_leaf_nrec
    } else {
        // Computing the exact max records at an internal child depth is complex;
        // a conservative upper bound from the leaf max is sufficient to size the
        // variable-width nrec field.
        max_leaf_nrec * 2
    };
    let nrec_width = bytes_for_max_records(max_nrec_child);

    let total_nrec_width = if depth > 1 {
        let max_total = header_max_total_records(max_leaf_nrec, depth - 1);
        bytes_for_max_records(max_total)
    } else {
        0
    };

    let num_children = nr + 1;
    let child_ptr_size = offset_size as usize + nrec_width + total_nrec_width;
    ensure_len(node, pos, num_children * child_ptr_size)?;

    let mut children = Vec::with_capacity(num_children);
    for _ in 0..num_children {
        let addr = read_offset(node, pos, offset_size)?;
        pos += offset_size as usize;
        let child_nrec = read_var_uint(node, pos, nrec_width)? as u16;
        pos += nrec_width;
        pos += total_nrec_width; // skip total-records-in-subtree
        children.push((addr, child_nrec));
    }

    Ok(children)
}

/// Recursively collect records from an internal node (buffered path).
#[allow(clippy::too_many_arguments, clippy::only_used_in_recursion)]
fn collect_internal_records(
    file_data: &[u8],
    offset: usize,
    num_records: u16,
    depth: u16,
    record_size: u16,
    node_size: u32,
    offset_size: u8,
    length_size: u8,
    max_leaf_nrec: u64,
    out: &mut Vec<BTreeV2Record>,
) -> Result<(), FormatError> {
    ensure_len(file_data, offset, 6)?;
    let node = &file_data[offset..];
    let children = parse_internal_child_pointers(
        node,
        num_records,
        depth,
        record_size,
        offset_size,
        max_leaf_nrec,
    )?;

    let nr = num_records as usize;
    let rs = record_size as usize;
    let child_depth = depth - 1;

    // Interleave: child[0], record[0], child[1], record[1], ..., child[nr].
    for (i, &(child_addr, child_nrec)) in children.iter().enumerate() {
        if child_depth == 0 {
            out.extend(parse_leaf_records(
                file_data,
                child_addr.to_usize()?,
                child_nrec,
                record_size,
            )?);
        } else {
            collect_internal_records(
                file_data,
                child_addr.to_usize()?,
                child_nrec,
                child_depth,
                record_size,
                node_size,
                offset_size,
                length_size,
                max_leaf_nrec,
                out,
            )?;
        }

        if i < nr {
            let start = 6 + i * rs;
            out.push(BTreeV2Record {
                data: node[start..start + rs].to_vec(),
            });
        }
    }

    Ok(())
}

/// Collect all records from a B-tree v2 by traversing from the root, reading
/// each fixed-size node from a [`FileSource`] on demand rather than indexing a
/// whole-file buffer.
pub fn collect_btree_v2_records_from_source<S: FileSource + ?Sized>(
    source: &S,
    header: &BTreeV2Header,
    offset_size: u8,
    _length_size: u8,
) -> Result<Vec<BTreeV2Record>, FormatError> {
    if header.total_records == 0 || header.num_records_in_root == 0 {
        return Ok(Vec::new());
    }
    let max_leaf_nrec = max_records_leaf(header.node_size, header.record_size);
    let mut records = Vec::new();
    collect_node_from_source(
        source,
        header.root_node_address,
        header.num_records_in_root,
        header.depth,
        header.record_size,
        header.node_size,
        offset_size,
        max_leaf_nrec,
        &mut records,
    )?;
    Ok(records)
}

/// Read and collect one node (leaf or internal) from the source, recursing into
/// children. Mirrors the buffered [`collect_btree_v2_records`] traversal and
/// produces records in the same order.
#[allow(clippy::too_many_arguments)]
fn collect_node_from_source<S: FileSource + ?Sized>(
    source: &S,
    address: u64,
    num_records: u16,
    depth: u16,
    record_size: u16,
    node_size: u32,
    offset_size: u8,
    max_leaf_nrec: u64,
    out: &mut Vec<BTreeV2Record>,
) -> Result<(), FormatError> {
    // Every node occupies `node_size` bytes; read that window (clamped to the
    // bytes available, in case the final node abuts EOF).
    let node_len = u64::from(node_size)
        .min(source.len().saturating_sub(address))
        .to_usize()?;
    let node = source.read_exact_at(address, node_len)?;

    if depth == 0 {
        out.extend(parse_leaf_records(&node, 0, num_records, record_size)?);
        return Ok(());
    }

    let children = parse_internal_child_pointers(
        &node,
        num_records,
        depth,
        record_size,
        offset_size,
        max_leaf_nrec,
    )?;

    let nr = num_records as usize;
    let rs = record_size as usize;
    let child_depth = depth - 1;
    for (i, &(child_addr, child_nrec)) in children.iter().enumerate() {
        collect_node_from_source(
            source,
            child_addr,
            child_nrec,
            child_depth,
            record_size,
            node_size,
            offset_size,
            max_leaf_nrec,
            out,
        )?;
        if i < nr {
            let start = 6 + i * rs;
            out.push(BTreeV2Record {
                data: node[start..start + rs].to_vec(),
            });
        }
    }

    Ok(())
}

/// Estimate maximum total records at a given depth (for variable-width encoding).
fn header_max_total_records(max_leaf_nrec: u64, depth: u16) -> u64 {
    // Conservative: branching factor * max_leaf at each level
    let mut total = max_leaf_nrec;
    for _ in 0..depth {
        total = total.saturating_mul(max_leaf_nrec.max(2));
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_btree_v2_header(
        tree_type: u8,
        node_size: u32,
        record_size: u16,
        depth: u16,
        root_addr: u64,
        num_records_root: u16,
        total_records: u64,
        offset_size: u8,
        length_size: u8,
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"BTHD");
        buf.push(0); // version
        buf.push(tree_type);
        buf.extend_from_slice(&node_size.to_le_bytes());
        buf.extend_from_slice(&record_size.to_le_bytes());
        buf.extend_from_slice(&depth.to_le_bytes());
        buf.push(85); // split_percent
        buf.push(40); // merge_percent
        match offset_size {
            4 => buf.extend_from_slice(&(root_addr as u32).to_le_bytes()),
            8 => buf.extend_from_slice(&root_addr.to_le_bytes()),
            _ => {}
        }
        buf.extend_from_slice(&num_records_root.to_le_bytes());
        match length_size {
            4 => buf.extend_from_slice(&(total_records as u32).to_le_bytes()),
            8 => buf.extend_from_slice(&total_records.to_le_bytes()),
            _ => {}
        }
        let checksum = crate::checksum::jenkins_lookup3(&buf);
        buf.extend_from_slice(&checksum.to_le_bytes());
        buf
    }

    fn build_leaf_node(tree_type: u8, records: &[&[u8]]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"BTLF");
        buf.push(0); // version
        buf.push(tree_type);
        for rec in records {
            buf.extend_from_slice(rec);
        }
        let checksum = crate::checksum::jenkins_lookup3(&buf);
        buf.extend_from_slice(&checksum.to_le_bytes());
        buf
    }

    #[test]
    fn parse_header() {
        let data = build_btree_v2_header(5, 512, 11, 0, 0x1000, 3, 3, 8, 8);
        let hdr = BTreeV2Header::parse(&data, 0, 8, 8).unwrap();
        assert_eq!(hdr.tree_type, 5);
        assert_eq!(hdr.node_size, 512);
        assert_eq!(hdr.record_size, 11);
        assert_eq!(hdr.depth, 0);
        assert_eq!(hdr.root_node_address, 0x1000);
        assert_eq!(hdr.num_records_in_root, 3);
        assert_eq!(hdr.total_records, 3);
    }

    #[test]
    fn parse_leaf_with_2_records() {
        let rec1 = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let rec2 = [11u8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];
        let leaf = build_leaf_node(5, &[&rec1, &rec2]);

        let leaf_offset = 256usize;
        let header = build_btree_v2_header(5, 512, 11, 0, leaf_offset as u64, 2, 2, 8, 8);

        let mut file_data = vec![0u8; 512];
        file_data[..header.len()].copy_from_slice(&header);
        file_data[leaf_offset..leaf_offset + leaf.len()].copy_from_slice(&leaf);

        let hdr = BTreeV2Header::parse(&file_data, 0, 8, 8).unwrap();
        let records = collect_btree_v2_records(&file_data, &hdr, 8, 8).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].data, rec1.to_vec());
        assert_eq!(records[1].data, rec2.to_vec());
    }

    #[test]
    fn invalid_signature() {
        let mut data = build_btree_v2_header(5, 512, 11, 0, 0, 0, 0, 8, 8);
        data[0] = b'X';
        let err = BTreeV2Header::parse(&data, 0, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidBTreeV2Signature);
    }

    #[test]
    fn invalid_version() {
        let mut data = build_btree_v2_header(5, 512, 11, 0, 0, 0, 0, 8, 8);
        data[4] = 1; // bad version
        let err = BTreeV2Header::parse(&data, 0, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidBTreeV2Version(1));
    }

    #[test]
    fn empty_tree() {
        let header = build_btree_v2_header(5, 512, 11, 0, 0, 0, 0, 8, 8);
        let hdr = BTreeV2Header::parse(&header, 0, 8, 8).unwrap();
        let records = collect_btree_v2_records(&header, &hdr, 8, 8).unwrap();
        assert!(records.is_empty());
    }

    #[cfg(feature = "std")]
    #[test]
    fn streaming_btree_matches_buffered() {
        use crate::source::{BytesSource, ReadSeekSource};
        let rec1 = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let rec2 = [11u8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];
        let leaf = build_leaf_node(5, &[&rec1, &rec2]);
        let leaf_offset = 256usize;
        let header = build_btree_v2_header(5, 512, 11, 0, leaf_offset as u64, 2, 2, 8, 8);
        let mut file_data = vec![0u8; 512];
        file_data[..header.len()].copy_from_slice(&header);
        file_data[leaf_offset..leaf_offset + leaf.len()].copy_from_slice(&leaf);

        let hdr = BTreeV2Header::parse(&file_data, 0, 8, 8).unwrap();
        let buffered: Vec<_> = collect_btree_v2_records(&file_data, &hdr, 8, 8)
            .unwrap()
            .into_iter()
            .map(|r| r.data)
            .collect();

        let mem = BytesSource::new(&file_data);
        let hdr_mem = BTreeV2Header::parse_from_source(&mem, 0, 8, 8).unwrap();
        assert_eq!(hdr_mem.root_node_address, hdr.root_node_address);
        let from_mem: Vec<_> = collect_btree_v2_records_from_source(&mem, &hdr_mem, 8, 8)
            .unwrap()
            .into_iter()
            .map(|r| r.data)
            .collect();

        let seek = ReadSeekSource::new(std::io::Cursor::new(file_data)).unwrap();
        let hdr_seek = BTreeV2Header::parse_from_source(&seek, 0, 8, 8).unwrap();
        let from_seek: Vec<_> = collect_btree_v2_records_from_source(&seek, &hdr_seek, 8, 8)
            .unwrap()
            .into_iter()
            .map(|r| r.data)
            .collect();

        assert_eq!(buffered, from_mem);
        assert_eq!(buffered, from_seek);
        assert_eq!(from_seek.len(), 2);
    }
}
