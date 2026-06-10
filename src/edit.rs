//! In-place editing of an existing HDF5 file (issue #32, Group C — first
//! milestone: add objects without rewriting the whole file).
//!
//! [`EditSession`] opens an existing file and appends new datasets to its root
//! group **in place**: the new dataset's data and object header are written at
//! the end of the file, and the root group's object header is rewritten (also
//! appended) with one extra link per added dataset, after which the superblock
//! is repointed at the new root header. Nothing already in the file is moved,
//! so the cost is proportional to what you add, not to the file size — unlike
//! the read-everything-then-rebuild path through [`FileBuilder`](crate::FileBuilder).
//!
//! # Scope of this milestone
//!
//! This is the foundation of the broader in-place edit engine (deletion and
//! object copy are planned follow-ons). It is deliberately strict: rather than
//! silently produce a degraded file, it refuses with [`Error::EditUnsupported`]
//! any case it cannot reproduce faithfully. Supported targets:
//!
//! - The file uses a latest-format (version 2/3) superblock with 8-byte
//!   offsets/lengths and **no** userblock (base address 0).
//! - The root group stores its links compactly (not in a dense fractal heap),
//!   its object header is a single chunk, and it does not track message
//!   creation order. Files written by this crate's [`FileBuilder`](crate::FileBuilder)
//!   satisfy this.
//! - Added datasets are contiguous and unfiltered (no chunking, compression,
//!   shuffle, scale-offset, or extensible dimensions), have a fixed-size
//!   datatype with a non-empty shape, and carry only fixed-size (non
//!   variable-length) attributes, few enough to stay in compact storage.
//!
//! The reclaimed-space question (the old root header becomes unreferenced dead
//! bytes after each commit) is out of scope here; it belongs to the free-space
//! work tracked separately (issue #21).

use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::checksum::jenkins_lookup3;
use crate::dataspace::{Dataspace, DataspaceType};
use crate::error::Error;
use crate::file_writer::{LENGTH_SIZE, OFFSET_SIZE, build_dataset_oh, make_link};
use crate::message_type::MessageType;
use crate::signature;
use crate::superblock::Superblock;
use crate::type_builders::{AttrValue, DatasetBuilder, build_attr_message};

/// Maximum number of compact attributes; beyond this HDF5 switches a dataset to
/// dense (fractal-heap) attribute storage, which this milestone does not emit.
/// Mirrors `DENSE_ATTR_THRESHOLD` in `file_writer`.
const MAX_COMPACT_ATTRS: usize = 8;

/// An open HDF5 file being edited in place.
///
/// Mirror the file in memory and keep a writable handle; every mutation is
/// applied to both so the on-disk file stays consistent. Stage additions with
/// [`create_dataset`](Self::create_dataset), then apply them with
/// [`commit`](Self::commit).
///
/// # Example
///
/// ```no_run
/// use hdf5_pure::EditSession;
///
/// let mut session = EditSession::open("existing.h5")?;
/// session
///     .create_dataset("new_signal")
///     .with_f64_data(&[1.0, 2.0, 3.0]);
/// session.commit()?;
/// # Ok::<(), hdf5_pure::Error>(())
/// ```
pub struct EditSession {
    handle: fs::File,
    /// In-memory mirror of the file, kept byte-for-byte in sync with `handle`.
    data: Vec<u8>,
    /// Absolute offset of the superblock signature in the file.
    sb_sig_off: usize,
    /// Parsed superblock. Addresses are as stored on disk (relative to the base
    /// address, which this editor requires to be 0).
    superblock: Superblock,
    /// Datasets staged by `create_dataset`, applied on the next `commit`.
    pending: Vec<DatasetBuilder>,
}

impl EditSession {
    /// Open an existing HDF5 file for in-place editing.
    ///
    /// Reads the file into memory and retains a read/write handle. Fails with
    /// [`Error::EditUnsupported`] if the file is not a supported target (see the
    /// [module docs](self) for the exact requirements).
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let mut handle = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path.as_ref())
            .map_err(Error::Io)?;
        let mut data = Vec::new();
        handle.read_to_end(&mut data).map_err(Error::Io)?;

        let sb_sig_off = signature::find_signature(&data)?;
        let superblock = Superblock::parse(&data, sb_sig_off)?;

        if superblock.version < 2 {
            return Err(Error::EditUnsupported(
                "superblock version 0/1 (symbol-table root group) is not editable in place",
            ));
        }
        if superblock.offset_size != OFFSET_SIZE || superblock.length_size != LENGTH_SIZE {
            return Err(Error::EditUnsupported(
                "only 8-byte offsets and lengths are supported for in-place editing",
            ));
        }
        if superblock.base_address != 0 {
            return Err(Error::EditUnsupported(
                "files with a userblock (non-zero base address) are not editable in place yet",
            ));
        }

        Ok(Self {
            handle,
            data,
            sb_sig_off,
            superblock,
            pending: Vec::new(),
        })
    }

    /// Stage a new dataset to be added to the root group on the next
    /// [`commit`](Self::commit). Returns a [`DatasetBuilder`] to configure its
    /// data, shape, and attributes — the same builder used by
    /// [`FileBuilder`](crate::FileBuilder).
    pub fn create_dataset(&mut self, name: &str) -> &mut DatasetBuilder {
        self.pending.push(DatasetBuilder::new(name));
        self.pending
            .last_mut()
            .expect("just pushed a dataset builder")
    }

    /// Apply all staged datasets to the file in place and flush to disk.
    ///
    /// Appends each new dataset (data blob + object header), then appends a
    /// rewritten root-group object header carrying the new links, then repoints
    /// the superblock at it. On success the staged set is cleared and the
    /// session can be reused for further edits. On any
    /// [`Error::EditUnsupported`] the file on disk is left untouched (the guard
    /// runs before any bytes are written).
    pub fn commit(&mut self) -> Result<(), Error> {
        if self.pending.is_empty() {
            return Ok(());
        }

        // Validate the root group up front, before writing anything, so an
        // unsupported target fails cleanly with the file unmodified.
        let root_addr = usize::try_from(self.superblock.root_group_address)
            .map_err(|_| Error::EditUnsupported("root group address exceeds this platform"))?;
        let (region_start, region_end) = self.inspect_root_group(root_addr)?;
        let old_region = self.data[region_start..region_end].to_vec();

        // Flatten every staged dataset before writing, so a rejected one (e.g.
        // chunked) does not leave a half-applied commit.
        let pending = std::mem::take(&mut self.pending);
        let mut flat = Vec::with_capacity(pending.len());
        for db in pending {
            flat.push(flatten_dataset(db)?);
        }

        // Append each dataset and collect the link messages for the root group.
        let mut new_links = Vec::new();
        for fd in flat {
            let data_addr = self.append(&fd.raw)?;
            let oh = build_dataset_oh(
                &fd.dt,
                &fd.ds,
                data_addr,
                fd.raw.len() as u64,
                &fd.attrs,
                None,
            );
            let oh_addr = self.append(&oh)?;
            new_links.push(encode_link_message(&fd.name, oh_addr));
        }

        // Rewrite the root-group object header (old messages verbatim + the new
        // links) at the end of the file, then repoint the superblock at it.
        let mut region = old_region;
        for link in &new_links {
            region.extend_from_slice(link);
        }
        let new_root_oh = build_v2_object_header(&region);
        let new_root_addr = self.append(&new_root_oh)?;

        self.superblock.root_group_address = new_root_addr;
        self.superblock.eof_address = self.data.len() as u64;
        let sb_bytes = self.superblock.serialize();
        self.write_at(self.sb_sig_off, &sb_bytes)?;

        self.handle.flush().map_err(Error::Io)?;
        Ok(())
    }

    /// Parse and validate the root group's object header, returning the byte
    /// range `[start, end)` of its chunk-0 message region (the bytes to copy
    /// when rewriting the header).
    fn inspect_root_group(&self, addr: usize) -> Result<(usize, usize), Error> {
        let d = &self.data;
        if d.len() < addr + 6 || &d[addr..addr + 4] != b"OHDR" {
            return Err(Error::EditUnsupported(
                "root group does not use a version 2 object header",
            ));
        }
        if d[addr + 4] != 2 {
            return Err(Error::EditUnsupported(
                "root group does not use a version 2 object header",
            ));
        }
        let flags = d[addr + 5];
        if flags & 0x04 != 0 {
            return Err(Error::EditUnsupported(
                "root group tracks message creation order (not supported in place yet)",
            ));
        }
        let mut pos = addr + 6;
        if flags & 0x20 != 0 {
            pos += 16; // optional timestamps
        }
        if flags & 0x10 != 0 {
            pos += 4; // optional attribute phase-change thresholds
        }
        let size_width = match flags & 0x03 {
            0 => 1usize,
            1 => 2,
            2 => 4,
            _ => 8,
        };
        if d.len() < pos + size_width {
            return Err(Error::EditUnsupported("truncated root group object header"));
        }
        let chunk0_size = read_le(&d[pos..pos + size_width]);
        pos += size_width;
        let region_start = pos;
        let region_end = region_start
            .checked_add(chunk0_size)
            .filter(|&e| e + 4 <= d.len())
            .ok_or(Error::EditUnsupported("truncated root group object header"))?;

        // Walk the chunk-0 messages: reject continuation (multi-chunk header)
        // and dense link storage, and confirm this is actually a group.
        let mut p = region_start;
        let mut has_link_info = false;
        while p + 4 <= region_end {
            let msg_type = MessageType::from_u16(d[p] as u16);
            let msg_size = u16::from_le_bytes([d[p + 1], d[p + 2]]) as usize;
            let body = p + 4;
            if body + msg_size > region_end {
                return Err(Error::EditUnsupported("malformed root group object header"));
            }
            match msg_type {
                MessageType::ObjectHeaderContinuation => {
                    return Err(Error::EditUnsupported(
                        "root group object header spans multiple chunks (not supported in place yet)",
                    ));
                }
                MessageType::LinkInfo => {
                    has_link_info = true;
                    // LinkInfo: version(1) flags(1) [max_creation_index(8) if
                    // flags&0x01] fractal_heap_addr(8) ... — dense storage has a
                    // defined fractal-heap address.
                    let mut q = body + 2;
                    if msg_size >= 2 && d[body + 1] & 0x01 != 0 {
                        q += 8;
                    }
                    if q + 8 <= region_end {
                        let heap_addr = u64::from_le_bytes(d[q..q + 8].try_into().unwrap());
                        if heap_addr != u64::MAX {
                            return Err(Error::EditUnsupported(
                                "root group uses dense (fractal-heap) link storage (not supported in place yet)",
                            ));
                        }
                    }
                }
                MessageType::DataLayout => {
                    return Err(Error::EditUnsupported(
                        "root object is a dataset, not a group",
                    ));
                }
                _ => {}
            }
            p = body + msg_size;
        }
        if !has_link_info {
            return Err(Error::EditUnsupported(
                "root group object header has no link-info message",
            ));
        }
        Ok((region_start, region_end))
    }

    /// Append `bytes` at end-of-file, updating both the mirror and the file.
    /// Returns the absolute address the bytes were written at.
    fn append(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        let addr = self.data.len() as u64;
        self.data.extend_from_slice(bytes);
        self.handle.seek(SeekFrom::Start(addr)).map_err(Error::Io)?;
        self.handle.write_all(bytes).map_err(Error::Io)?;
        Ok(addr)
    }

    /// Overwrite bytes in place at `offset`, updating both the mirror and the
    /// file. The caller guarantees the range already exists.
    fn write_at(&mut self, offset: usize, bytes: &[u8]) -> Result<(), Error> {
        self.data[offset..offset + bytes.len()].copy_from_slice(bytes);
        self.handle
            .seek(SeekFrom::Start(offset as u64))
            .map_err(Error::Io)?;
        self.handle.write_all(bytes).map_err(Error::Io)?;
        Ok(())
    }
}

/// A staged dataset reduced to the pieces the writer needs.
struct FlatDataset {
    name: String,
    dt: crate::datatype::Datatype,
    ds: Dataspace,
    raw: Vec<u8>,
    attrs: Vec<crate::attribute::AttributeMessage>,
}

/// Validate a staged dataset and reduce it to a [`FlatDataset`], rejecting any
/// feature this milestone cannot emit as contiguous, unfiltered storage.
fn flatten_dataset(db: DatasetBuilder) -> Result<FlatDataset, Error> {
    if db.chunk_options.is_chunked() || db.maxshape.is_some() {
        return Err(Error::EditUnsupported(
            "chunked / compressed / extensible datasets cannot be added in place yet",
        ));
    }
    if db.reference_targets.is_some() {
        return Err(Error::EditUnsupported(
            "object-reference datasets cannot be added in place yet",
        ));
    }
    #[cfg(feature = "provenance")]
    if db.provenance.is_some() {
        return Err(Error::EditUnsupported(
            "provenance datasets cannot be added in place yet",
        ));
    }

    let dt = db
        .datatype
        .ok_or(Error::EditUnsupported("dataset has no datatype/data"))?;
    let shape = db
        .shape
        .ok_or(Error::EditUnsupported("dataset has no shape"))?;
    if shape.contains(&0) {
        return Err(Error::EditUnsupported(
            "empty (zero-element) datasets cannot be added in place yet",
        ));
    }
    let raw = db
        .data
        .ok_or(Error::EditUnsupported("dataset has no data"))?;

    let elem = dt.type_size() as u64;
    if elem > 0 {
        let expected = shape.iter().product::<u64>().saturating_mul(elem);
        if raw.len() as u64 != expected {
            return Err(Error::EditUnsupported(
                "dataset data length does not match its shape",
            ));
        }
    }

    if db
        .attrs
        .iter()
        .any(|(_, v)| matches!(v, AttrValue::VarLenAsciiArray(_)))
    {
        return Err(Error::EditUnsupported(
            "variable-length attributes cannot be added in place yet",
        ));
    }
    if db.attrs.len() > MAX_COMPACT_ATTRS {
        return Err(Error::EditUnsupported(
            "datasets with dense (many) attributes cannot be added in place yet",
        ));
    }

    let ds = Dataspace {
        space_type: if shape.is_empty() {
            DataspaceType::Scalar
        } else {
            DataspaceType::Simple
        },
        rank: shape.len() as u8,
        dimensions: shape,
        max_dimensions: None,
    };
    let attrs = db
        .attrs
        .iter()
        .map(|(n, v)| build_attr_message(n, v))
        .collect();

    Ok(FlatDataset {
        name: db.name,
        dt,
        ds,
        raw,
        attrs,
    })
}

/// Encode a complete object-header Link message (4-byte record header + body)
/// for a hard link `name -> addr`.
fn encode_link_message(name: &str, addr: u64) -> Vec<u8> {
    let body = make_link(name, addr).serialize(OFFSET_SIZE);
    let mut m = Vec::with_capacity(4 + body.len());
    m.push(MessageType::Link.to_u16() as u8);
    m.extend_from_slice(&(body.len() as u16).to_le_bytes());
    m.push(0); // message flags
    m.extend_from_slice(&body);
    m
}

/// Wrap a chunk-0 message region in a fresh single-chunk version 2 object
/// header (`OHDR` prefix + region + Jenkins checksum). Mirrors the encoding in
/// [`crate::object_header_writer::ObjectHeaderWriter::serialize`].
fn build_v2_object_header(region: &[u8]) -> Vec<u8> {
    let total = region.len();
    let (flags, width) = if total <= 255 {
        (0u8, 1usize)
    } else if total <= 65535 {
        (1u8, 2)
    } else {
        (2u8, 4)
    };
    let mut buf = Vec::with_capacity(8 + total + 4);
    buf.extend_from_slice(b"OHDR");
    buf.push(2); // version
    buf.push(flags);
    match width {
        1 => buf.push(total as u8),
        2 => buf.extend_from_slice(&(total as u16).to_le_bytes()),
        _ => buf.extend_from_slice(&(total as u32).to_le_bytes()),
    }
    buf.extend_from_slice(region);
    let checksum = jenkins_lookup3(&buf);
    buf.extend_from_slice(&checksum.to_le_bytes());
    buf
}

/// Read a little-endian unsigned integer of `bytes.len()` (≤ 8) bytes.
fn read_le(bytes: &[u8]) -> usize {
    let mut v = 0u64;
    for (i, &b) in bytes.iter().enumerate() {
        v |= (b as u64) << (8 * i);
    }
    v as usize
}
