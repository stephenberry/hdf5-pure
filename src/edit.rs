//! In-place editing of an existing HDF5 file (issue #32, Group C).
//!
//! [`EditSession`] opens an existing file and adds objects to it **in place**:
//! new data and object headers are written at the end of the file, and the
//! object headers of the touched groups (and their ancestors up to the root)
//! are rewritten — also appended — so the superblock ends up pointing at the
//! new root header. Nothing already in the file is moved, so the cost is
//! proportional to what you add, not to the file size — unlike the
//! read-everything-then-rebuild path through [`FileBuilder`](crate::FileBuilder).
//!
//! Both new datasets and new (sub)groups are supported, at any existing group
//! path. Adding into a nested group `/a/b` rewrites `b`'s header (with the new
//! link), then `a`'s header (repointing its link to `b`'s new location), then
//! the root's — "relocation up the tree". This is always safe for *additions*
//! because no surviving object is relocated except the groups on the path being
//! edited, and those are reachable only through links this same commit rewrites
//! (the root through the superblock); absolute object-reference addresses to
//! other objects stay valid.
//!
//! Deletion ([`EditSession::delete`], the HDF5 `H5Ldelete`) is the mirror image:
//! the parent group's header is rebuilt without the removed link, relocated up
//! the tree the same way, and the unlinked object (and its subtree) is left as
//! dead bytes. Object copy ([`EditSession::copy`], the HDF5 `H5Ocopy`) deep-copies
//! a source subtree — appending fresh copies of every object, repointing internal
//! links and the contiguous data address — and links the copy in like an
//! addition; the headers are reproduced from their verbatim message bytes, so
//! datatypes, dataspaces, and attributes stay byte-exact.
//!
//! # Scope
//!
//! It is deliberately strict: rather than silently produce a degraded file, it
//! refuses with [`Error::EditUnsupported`] any case it cannot reproduce
//! faithfully. Requirements:
//!
//! - The file uses 8-byte offsets/lengths and has **no** userblock (base
//!   address 0). Any superblock version (0–3) is accepted: a version 0/1
//!   (symbol-table) file is edited by converting each group on the edited path
//!   to the latest format and repointing the superblock's root symbol-table
//!   entry.
//! - A version 2/3 group on an edited path stores its links compactly (not in a
//!   dense fractal heap) and does not track message creation order; headers
//!   split across continuation chunks (as the reference C library often writes)
//!   are collapsed into a single chunk when rewritten. A version 1 group is
//!   converted to a compact-link v2 header, carrying its links and attributes
//!   over (other group messages — symbol table, modification time — are
//!   dropped); an attribute it cannot reproduce is refused.
//! - Added datasets are contiguous and unfiltered (no chunking, compression,
//!   shuffle, scale-offset, or extensible dimensions), have a fixed-size
//!   datatype with a non-empty shape, and carry only fixed-size (non
//!   variable-length) attributes, few enough to stay in compact storage.
//! - A new group's parent must already exist or be created in the same session
//!   (each level created explicitly); intermediate groups are not auto-created.
//!
//! The reclaimed-space question (a superseded object header becomes unreferenced
//! dead bytes after each commit) is out of scope here; it belongs to the
//! free-space work tracked separately (issue #21).

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::checksum::jenkins_lookup3;
use crate::dataspace::{Dataspace, DataspaceType};
use crate::error::Error;
use crate::file_writer::{LENGTH_SIZE, OFFSET_SIZE, build_dataset_oh, make_link};
use crate::group_v2::resolve_group_entries;
use crate::link_message::{LinkMessage, LinkTarget};
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::signature;
use crate::superblock::Superblock;
use crate::type_builders::{AttrValue, DatasetBuilder, build_attr_message};

/// Maximum number of compact attributes; beyond this HDF5 switches a dataset to
/// dense (fractal-heap) attribute storage, which this engine does not emit.
/// Mirrors `DENSE_ATTR_THRESHOLD` in `file_writer`.
const MAX_COMPACT_ATTRS: usize = 8;

/// Recursion-depth cap for object copy, guarding against a stack overflow on a
/// pathological or cyclic hard-link graph (HDF5 hard links can form cycles).
/// Far deeper than any real group hierarchy.
const MAX_COPY_DEPTH: u32 = 1000;

/// Maximum number of object-header chunks to follow when gathering a header that
/// spans continuation blocks, guarding against a cyclic continuation chain.
/// Matches the reader's continuation-depth cap.
const MAX_OH_CHUNKS: usize = 256;

/// A path identified by its components (no leading/trailing empties); the root
/// group is the empty vector.
type PathKey = Vec<String>;

/// An open HDF5 file being edited in place.
///
/// Mirror the file in memory and keep a writable handle; every mutation is
/// applied to both so the on-disk file stays consistent. Stage additions with
/// [`create_dataset`](Self::create_dataset) / [`create_group`](Self::create_group),
/// then apply them with [`commit`](Self::commit).
///
/// # Example
///
/// ```no_run
/// use hdf5_pure::EditSession;
///
/// let mut session = EditSession::open("existing.h5")?;
/// session.create_group("run2");
/// session
///     .create_dataset("run2/signal")
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
    /// Datasets staged by `create_dataset`, as (parent group path, builder).
    pending_datasets: Vec<(PathKey, DatasetBuilder)>,
    /// New groups staged by `create_group`, as full paths.
    pending_groups: Vec<PathKey>,
    /// Links staged for removal by `delete`, as full paths.
    pending_deletes: Vec<PathKey>,
    /// Object copies staged by `copy`, as (source path, destination full path).
    pending_copies: Vec<(PathKey, PathKey)>,
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

        if superblock.version > 3 {
            return Err(Error::EditUnsupported("unsupported superblock version"));
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
            pending_datasets: Vec::new(),
            pending_groups: Vec::new(),
            pending_deletes: Vec::new(),
            pending_copies: Vec::new(),
        })
    }

    /// Stage a new dataset, added on the next [`commit`](Self::commit). The
    /// argument is the full path of the dataset; everything before the last
    /// component names the parent group, which must exist (or be created in this
    /// session). Returns the [`DatasetBuilder`] — the same builder used by
    /// [`FileBuilder`](crate::FileBuilder) — to configure data, shape, and
    /// attributes.
    pub fn create_dataset(&mut self, path: &str) -> &mut DatasetBuilder {
        let mut comps = split_path(path);
        let leaf = comps.pop().unwrap_or_default();
        self.pending_datasets
            .push((comps, DatasetBuilder::new(&leaf)));
        &mut self.pending_datasets.last_mut().unwrap().1
    }

    /// Stage a new (empty) group at `path`, created on the next
    /// [`commit`](Self::commit). The parent must already exist or be created in
    /// the same session; populate the group with datasets via
    /// [`create_dataset`](Self::create_dataset) using a path under it.
    pub fn create_group(&mut self, path: &str) {
        self.pending_groups.push(split_path(path));
    }

    /// Stage removal of the link at `path` (the HDF5 `H5Ldelete`), applied on the
    /// next [`commit`](Self::commit). The link's object — and, for a group, its
    /// whole subtree — becomes unreachable; the bytes it occupied are not
    /// reclaimed (that is the separate free-space concern, issue #21).
    ///
    /// The path must exist. A deletion may not overlap another staged change in
    /// the same commit (e.g. delete `/a` while adding `/a/b`); split such
    /// edits into separate commits. The link's parent group must itself be
    /// editable in place (compact links, single-chunk header); the target being
    /// removed has no such restriction.
    pub fn delete(&mut self, path: &str) {
        self.pending_deletes.push(split_path(path));
    }

    /// Stage a deep copy of the object at `src` to a new link at `dst` (the HDF5
    /// `H5Ocopy`), applied on the next [`commit`](Self::commit). The source — a
    /// dataset or a whole group subtree — is duplicated: fresh copies of every
    /// object's data and header are written, internal links and the contiguous
    /// data address are repointed to the copies, and a link named by `dst`'s last
    /// component is added to `dst`'s parent group. The original is untouched.
    ///
    /// The copy reflects the file's on-disk state at commit time. `src` must
    /// exist and `dst` must not (and may not lie inside `src`). The source
    /// subtree must be copyable in place: contiguous/compact datasets only (no
    /// chunked/compressed storage), compact links and attributes, single-chunk
    /// headers — otherwise `commit` reports [`Error::EditUnsupported`].
    pub fn copy(&mut self, src: &str, dst: &str) {
        self.pending_copies.push((split_path(src), split_path(dst)));
    }

    /// Apply all staged additions and deletions to the file in place and flush.
    ///
    /// Appends each new dataset (data blob + object header) and each new group,
    /// then appends rewritten object headers for every touched group and its
    /// ancestors up to the root (omitting any deleted links), then repoints the
    /// superblock at the new root. On success the staged set is cleared and the
    /// session can be reused. On any [`Error::EditUnsupported`] the file on disk
    /// is left untouched: every check runs before the first byte is written.
    pub fn commit(&mut self) -> Result<(), Error> {
        if self.pending_datasets.is_empty()
            && self.pending_groups.is_empty()
            && self.pending_deletes.is_empty()
            && self.pending_copies.is_empty()
        {
            return Ok(());
        }

        // --- Plan: build the tree of "dirty" groups (root plus every group on a
        // path to an addition or deletion), validating every target before any
        // write. `add_targets` records the full paths created this commit, used
        // to reject a deletion that overlaps an addition. ---
        let mut nodes: BTreeMap<PathKey, Node> = BTreeMap::new();
        nodes.entry(PathKey::new()).or_default(); // root is always dirty
        let mut add_targets: Vec<PathKey> = Vec::new();

        // Mark explicitly-created new groups, ensuring their ancestor chain.
        for path in std::mem::take(&mut self.pending_groups) {
            if path.is_empty() {
                return Err(Error::EditUnsupported("cannot create the root group"));
            }
            ensure_ancestors(&mut nodes, &path);
            nodes.entry(path.clone()).or_default().is_new = true;
            add_targets.push(path);
        }

        // Attach datasets to their parent group nodes, ensuring ancestor chains.
        for (parent, db) in std::mem::take(&mut self.pending_datasets) {
            let mut full = parent.clone();
            full.push(db.name.clone());
            add_targets.push(full);
            ensure_ancestors(&mut nodes, &parent);
            nodes.entry(parent).or_default().datasets.push(db);
        }

        // Stage copies: validate the source subtree is copyable (read-only),
        // then treat the destination like an addition to its parent group.
        for (src, dst) in std::mem::take(&mut self.pending_copies) {
            if src.is_empty() {
                return Err(Error::EditUnsupported("cannot copy the root group"));
            }
            if dst.is_empty() {
                return Err(Error::EditUnsupported("copy destination path is empty"));
            }
            if is_prefix(&src, &dst) {
                return Err(Error::EditUnsupported(
                    "cannot copy an object into itself or its own subtree",
                ));
            }
            let src_str = src.join("/");
            let src_addr =
                crate::group_v2::resolve_path_any(&self.data, &self.superblock, &src_str)
                    .map_err(|_| Error::EditUnsupported("copy source does not exist"))?;
            let src_addr = usize::try_from(src_addr)
                .map_err(|_| Error::EditUnsupported("source address exceeds this platform"))?;
            self.plan_copy(src_addr, 0)?;
            add_targets.push(dst.clone());
            let leaf = dst.last().unwrap().clone();
            let parent = dst[..dst.len() - 1].to_vec();
            ensure_ancestors(&mut nodes, &parent);
            nodes
                .entry(parent)
                .or_default()
                .copies
                .push((leaf, src_addr as u64));
        }

        // Stage deletions: each must exist, must not overlap any other staged
        // change, and is recorded against its parent group (which becomes dirty).
        let deletes = std::mem::take(&mut self.pending_deletes);
        for (i, d) in deletes.iter().enumerate() {
            if d.is_empty() {
                return Err(Error::EditUnsupported("cannot delete the root group"));
            }
            let path_str = d.join("/");
            crate::group_v2::resolve_path_any(&self.data, &self.superblock, &path_str)
                .map_err(|_| Error::EditUnsupported("nothing to delete at the given path"))?;
            for t in &add_targets {
                if is_prefix(d, t) || is_prefix(t, d) {
                    return Err(Error::EditUnsupported(
                        "a deletion overlaps an addition in the same commit; use separate commits",
                    ));
                }
            }
            for (j, d2) in deletes.iter().enumerate() {
                if i != j && is_prefix(d, d2) {
                    return Err(Error::EditUnsupported(
                        "overlapping deletions in one commit; delete the common parent only",
                    ));
                }
            }
            let parent = d[..d.len() - 1].to_vec();
            ensure_ancestors(&mut nodes, &parent);
            nodes
                .entry(parent)
                .or_default()
                .deletes
                .push(d.last().unwrap().clone());
        }

        // Resolve / validate each node's base object-header region up front.
        let keys: Vec<PathKey> = nodes.keys().cloned().collect();
        for key in &keys {
            let is_new = nodes[key].is_new;
            if is_new {
                nodes.get_mut(key).unwrap().base_region = fresh_group_region();
            } else {
                let path_str = key.join("/");
                let addr =
                    crate::group_v2::resolve_path_any(&self.data, &self.superblock, &path_str)
                        .map_err(|_| {
                            Error::EditUnsupported(
                                "a target group does not exist; create it first in this session",
                            )
                        })?;
                let addr = usize::try_from(addr)
                    .map_err(|_| Error::EditUnsupported("group address exceeds this platform"))?;
                let info = self.inspect_group(addr)?;
                let node = nodes.get_mut(key).unwrap();
                node.base_region = info.region;
                node.existing_links = info.link_names;
            }
        }

        // Map each node to its direct child group nodes (for link wiring).
        let mut children: BTreeMap<PathKey, Vec<PathKey>> = BTreeMap::new();
        for key in &keys {
            if !key.is_empty() {
                let parent = key[..key.len() - 1].to_vec();
                children.entry(parent).or_default().push(key.clone());
            }
        }

        // Validate names: no addition may collide with an existing link or with
        // another addition under the same parent.
        for key in &keys {
            let node = &nodes[key];
            let mut adding: Vec<&str> = Vec::new();
            for db in &node.datasets {
                adding.push(&db.name);
            }
            for child in children.get(key).into_iter().flatten() {
                if nodes[child].is_new {
                    adding.push(child.last().unwrap());
                }
            }
            for (leaf, _) in &node.copies {
                adding.push(leaf);
            }
            for (i, name) in adding.iter().enumerate() {
                if node.existing_links.iter().any(|n| n == name) || adding[..i].contains(name) {
                    return Err(Error::EditUnsupported(
                        "a link with this name already exists in the target group",
                    ));
                }
            }
        }

        // Flatten datasets (more guards) before any write, so a rejected one
        // leaves the commit unapplied.
        let mut flat: BTreeMap<PathKey, Vec<FlatDataset>> = BTreeMap::new();
        for key in &keys {
            let dbs = std::mem::take(&mut nodes.get_mut(key).unwrap().datasets);
            let mut v = Vec::with_capacity(dbs.len());
            for db in dbs {
                v.push(flatten_dataset(db)?);
            }
            flat.insert(key.clone(), v);
        }

        // --- Apply: process deepest groups first so each parent sees its
        // children's new addresses, then repoint the superblock last. ---
        let mut new_addr: BTreeMap<PathKey, u64> = BTreeMap::new();
        let mut by_depth = keys.clone();
        by_depth.sort_by_key(|k| std::cmp::Reverse(k.len())); // deepest first
        for key in &by_depth {
            let (mut region, deletes, copies) = {
                let node = nodes.get_mut(key).unwrap();
                (
                    std::mem::take(&mut node.base_region),
                    std::mem::take(&mut node.deletes),
                    std::mem::take(&mut node.copies),
                )
            };

            // Remove deleted links first (verbatim-preserving the rest).
            for name in &deletes {
                region = remove_link_from_region(&region, name)?;
            }

            // Deep-copy each source subtree and link its root into this group.
            for (leaf, src_addr) in copies {
                let root = self.perform_copy(src_addr as usize, 0)?;
                region.extend_from_slice(&encode_link_message(&leaf, root));
            }

            // Datasets directly under this group.
            for fd in flat.remove(key).into_iter().flatten() {
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
                region.extend_from_slice(&encode_link_message(&fd.name, oh_addr));
            }

            // Wire links to dirty child groups (new → add a link; existing →
            // patch the existing link to the child's new address).
            for child in children.get(key).into_iter().flatten() {
                let child_name = child.last().unwrap();
                let child_addr = new_addr[child];
                if nodes[child].is_new {
                    region.extend_from_slice(&encode_link_message(child_name, child_addr));
                } else {
                    patch_link_target(&mut region, child_name, child_addr)?;
                }
            }

            let oh = build_v2_object_header(&region);
            let addr = self.append(&oh)?;
            new_addr.insert(key.clone(), addr);
        }

        // Repoint the superblock at the new root last: this is the commit's
        // linearization point. Until it lands, the file on disk still points at
        // the old root (the appended objects are merely unreferenced trailing
        // bytes), so a failure here leaves a valid file.
        let new_root = new_addr[&PathKey::new()];
        let new_eof = self.data.len() as u64;
        if self.superblock.version >= 2 {
            // Build the new superblock off a clone and adopt it only once the
            // write succeeds, so a failed write does not desync the in-memory
            // state. The v2/v3 superblock carries its own checksum.
            let mut new_sb = self.superblock.clone();
            new_sb.root_group_address = new_root;
            new_sb.eof_address = new_eof;
            let sb_bytes = new_sb.serialize();
            self.write_at(self.sb_sig_off, &sb_bytes)?;
            self.handle.flush().map_err(Error::Io)?;
            self.superblock = new_sb;
        } else {
            self.repoint_v0v1_root(new_root, new_eof)?;
            self.handle.flush().map_err(Error::Io)?;
            self.superblock.root_group_address = new_root;
            self.superblock.eof_address = new_eof;
        }
        Ok(())
    }

    /// Repoint a version 0/1 superblock at the rebuilt (now v2) root group and
    /// update its end-of-file field, patching the raw bytes in place — these
    /// superblocks carry no checksum. The root symbol-table entry is switched to
    /// cache type 0 (its scratch-pad B-tree / local-heap addresses, which
    /// describe the old symbol-table group, no longer apply). The
    /// object-header-address write is done last so it is the linearization point.
    fn repoint_v0v1_root(&mut self, new_root: u64, new_eof: u64) -> Result<(), Error> {
        let os = self.superblock.offset_size as usize;
        // Field layout after the fixed prefix: base / free-space / EOF / driver
        // addresses, then the root symbol-table entry (link-name offset, object
        // header address, cache type(4), reserved(4), scratch(16)). The prefix is
        // 24 bytes for v0 and 28 for v1 (the latter adds indexed-storage-K).
        let var_start = if self.superblock.version == 0 { 24 } else { 28 };
        let base = self.sb_sig_off + var_start;
        let eof_off = base + 2 * os;
        let ste = base + 4 * os;
        let oh_addr_off = ste + os;
        let cache_off = ste + 2 * os;
        self.write_at(eof_off, &new_eof.to_le_bytes()[..os])?;
        self.write_at(cache_off, &[0u8; 4])?; // cache type = none
        self.write_at(cache_off + 8, &[0u8; 16])?; // clear scratch-pad
        self.write_at(oh_addr_off, &new_root.to_le_bytes()[..os])?;
        Ok(())
    }

    /// Parse and validate the prefix of a single-chunk version 2 object header at
    /// `addr`, returning the `[start, end)` byte range of its message region.
    /// Rejects headers that are not OHDR v2 or that track message creation order
    /// (whose 6-byte message records this engine does not emit).
    fn oh_region(&self, addr: usize) -> Result<(usize, usize), Error> {
        let d = &self.data;
        if d.len() < addr + 6 || &d[addr..addr + 4] != b"OHDR" || d[addr + 4] != 2 {
            return Err(Error::EditUnsupported(
                "an object does not use a version 2 object header",
            ));
        }
        let flags = d[addr + 5];
        if flags & 0x04 != 0 {
            return Err(Error::EditUnsupported(
                "an object tracks message creation order (not supported in place yet)",
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
            return Err(Error::EditUnsupported("truncated object header"));
        }
        let chunk0_size = read_le(&d[pos..pos + size_width]);
        pos += size_width;
        let region_start = pos;
        let region_end = region_start
            .checked_add(chunk0_size)
            .filter(|&e| e + 4 <= d.len())
            .ok_or(Error::EditUnsupported("truncated object header"))?;
        Ok((region_start, region_end))
    }

    /// Collect every message of the object header at `addr` into one contiguous
    /// region, following continuation blocks across chunks and dropping the
    /// `Continuation` messages themselves. Re-emitting the result through
    /// [`build_v2_object_header`] collapses a multi-chunk header (as the
    /// reference C library often writes) into a single chunk, which is how this
    /// editor rebuilds headers. The chunk-0 prefix is validated by
    /// [`oh_region`]; each continuation block must be a well-formed `OCHK` block
    /// within the file.
    fn gather_oh_messages(&self, addr: usize) -> Result<Vec<u8>, Error> {
        let (rs, re) = self.oh_region(addr)?;
        let d = &self.data;
        let mut out = Vec::new();
        // Worklist of (message-region start, end) per chunk, chunk 0 first.
        let mut chunks: Vec<(usize, usize)> = vec![(rs, re)];
        let mut i = 0;
        while i < chunks.len() {
            if chunks.len() > MAX_OH_CHUNKS {
                return Err(Error::EditUnsupported(
                    "object header has too many continuation chunks",
                ));
            }
            let (cs, ce) = chunks[i];
            i += 1;
            let region = &d[..ce];
            let mut p = cs;
            while let Some((msg_type, body, body_end)) = next_message(region, p)? {
                if msg_type == MessageType::ObjectHeaderContinuation {
                    // Body: block offset (offset_size) + block length (length_size).
                    if body_end - body < (OFFSET_SIZE + LENGTH_SIZE) as usize {
                        return Err(Error::EditUnsupported("malformed continuation message"));
                    }
                    let off = u64::from_le_bytes(d[body..body + 8].try_into().unwrap());
                    let len = u64::from_le_bytes(d[body + 8..body + 16].try_into().unwrap());
                    let off = usize::try_from(off).map_err(|_| {
                        Error::EditUnsupported("continuation address exceeds this platform")
                    })?;
                    let len = usize::try_from(len).map_err(|_| {
                        Error::EditUnsupported("continuation length exceeds this platform")
                    })?;
                    // An OCHK block is signature(4) + messages + checksum(4).
                    let blk_end = off
                        .checked_add(len)
                        .filter(|&e| e <= d.len() && len >= 8)
                        .ok_or(Error::EditUnsupported("continuation block out of bounds"))?;
                    if &d[off..off + 4] != b"OCHK" {
                        return Err(Error::EditUnsupported(
                            "invalid continuation block signature",
                        ));
                    }
                    chunks.push((off + 4, blk_end - 4));
                } else {
                    out.extend_from_slice(&region[p..body_end]);
                }
                p = body_end;
            }
        }
        Ok(out)
    }

    /// Reconstruct a version-1 (symbol-table) group as a fresh v2 compact-link
    /// message region: a LinkInfo message, one Link message per existing child,
    /// and the group's existing attributes (re-wrapped as v2 messages). The
    /// symbol-table message and other non-link/non-attribute messages
    /// (modification time, comment, …) are dropped — editing a v0/v1 group
    /// converts it to the latest format. Refuses an attribute it cannot
    /// reproduce (shared, or larger than a v2 message can hold).
    fn reconstruct_v1_group(&self, addr: usize) -> Result<GroupInfo, Error> {
        let os = self.superblock.offset_size;
        let ls = self.superblock.length_size;
        let base = self.superblock.base_address;
        let oh = ObjectHeader::parse_with_base(&self.data, addr, os, ls, base)?;
        if oh
            .messages
            .iter()
            .any(|m| m.msg_type == MessageType::DataLayout)
        {
            return Err(Error::EditUnsupported(
                "a target path names a dataset, not a group",
            ));
        }
        let entries = resolve_group_entries(&self.data, &oh, os, ls, base)?;

        let mut region = fresh_group_region();
        let mut link_names = Vec::with_capacity(entries.len());
        for e in &entries {
            // Group-entry addresses are stored relative to the base address (0
            // here), matching how link targets are stored.
            region.extend_from_slice(&encode_link_message(&e.name, e.object_header_address));
            link_names.push(e.name.clone());
        }
        for m in &oh.messages {
            if m.msg_type == MessageType::Attribute {
                if m.flags != 0 {
                    return Err(Error::EditUnsupported(
                        "a v0/v1 group has a shared attribute message (not convertible in place yet)",
                    ));
                }
                if m.data.len() > u16::MAX as usize {
                    return Err(Error::EditUnsupported(
                        "a v0/v1 group attribute is too large to convert in place",
                    ));
                }
                // Re-wrap the attribute message body (it is self-describing) in a
                // v2 message record.
                region.push(MessageType::Attribute.to_u16() as u8);
                region.extend_from_slice(&(m.data.len() as u16).to_le_bytes());
                region.push(0); // message flags
                region.extend_from_slice(&m.data);
            }
        }
        Ok(GroupInfo { region, link_names })
    }

    /// Parse and validate a group's object header, returning its message region
    /// — the bytes to copy when rewriting the header — and the names of its
    /// existing links. A version 2 header is rebuilt from its own message bytes
    /// (collapsing continuation chunks, preserving every message); a version 1
    /// symbol-table group is converted to v2 via [`reconstruct_v1_group`].
    fn inspect_group(&self, addr: usize) -> Result<GroupInfo, Error> {
        if self.data.len() < addr + 4 || self.data[addr..addr + 4] != *b"OHDR" {
            return self.reconstruct_v1_group(addr);
        }
        let region = self.gather_oh_messages(addr)?;
        let mut p = 0;
        let mut has_link_info = false;
        let mut link_names = Vec::new();
        while let Some((msg_type, body, body_end)) = next_message(&region, p)? {
            match msg_type {
                MessageType::LinkInfo => {
                    has_link_info = true;
                    // LinkInfo: version(1) flags(1) [max_creation_index(8) if
                    // flags&0x01] fractal_heap_addr(8) … — dense storage has a
                    // defined fractal-heap address. Bound the read by this
                    // message's own body, not just the region, so a short or
                    // malformed LinkInfo can't make us read the next message.
                    let mut q = body + 2;
                    if body_end - body >= 2 && region[body + 1] & 0x01 != 0 {
                        q += 8;
                    }
                    if q + 8 <= body_end {
                        let heap_addr = u64::from_le_bytes(region[q..q + 8].try_into().unwrap());
                        if heap_addr != u64::MAX {
                            return Err(Error::EditUnsupported(
                                "a target group uses dense (fractal-heap) link storage (not supported in place yet)",
                            ));
                        }
                    }
                }
                MessageType::Link => {
                    if let Ok(link) = LinkMessage::parse(&region[body..body_end], OFFSET_SIZE) {
                        link_names.push(link.name);
                    }
                }
                MessageType::DataLayout => {
                    return Err(Error::EditUnsupported(
                        "a target path names a dataset, not a group",
                    ));
                }
                _ => {}
            }
            p = body_end;
        }
        if !has_link_info {
            return Err(Error::EditUnsupported(
                "a target group's object header has no link-info message",
            ));
        }
        Ok(GroupInfo { region, link_names })
    }

    /// Parse the object header at `addr` into a copyable model, validating that
    /// every message can be reproduced faithfully (verbatim message bytes, with
    /// only the contiguous data address and child link targets repointed).
    /// Rejects multi-chunk headers, dense attribute storage, dense or
    /// soft/external links, chunked/old-version data layouts, and headers that
    /// are neither a dataset nor a group.
    fn read_object(&self, addr: usize) -> Result<ObjModel, Error> {
        let region = self.gather_oh_messages(addr)?;

        let mut layout: Option<(usize, usize)> = None; // (body offset, size)
        let mut has_link_info = false;
        let mut children: Vec<(String, u64)> = Vec::new();
        let mut non_link: Vec<u8> = Vec::new();

        let mut p = 0;
        while let Some((msg_type, body, body_end)) = next_message(&region, p)? {
            match msg_type {
                MessageType::AttributeInfo => {
                    return Err(Error::EditUnsupported(
                        "an object uses dense (fractal-heap) attribute storage (not supported in place yet)",
                    ));
                }
                MessageType::LinkInfo => {
                    has_link_info = true;
                    let mut q = body + 2;
                    if body_end - body >= 2 && region[body + 1] & 0x01 != 0 {
                        q += 8;
                    }
                    if q + 8 <= body_end {
                        let heap_addr = u64::from_le_bytes(region[q..q + 8].try_into().unwrap());
                        if heap_addr != u64::MAX {
                            return Err(Error::EditUnsupported(
                                "a group uses dense (fractal-heap) link storage (not supported in place yet)",
                            ));
                        }
                    }
                }
                MessageType::Link => match LinkMessage::parse(&region[body..body_end], OFFSET_SIZE)
                {
                    Ok(LinkMessage {
                        name,
                        link_target:
                            LinkTarget::Hard {
                                object_header_address,
                            },
                        ..
                    }) => children.push((name, object_header_address)),
                    _ => {
                        return Err(Error::EditUnsupported(
                            "a group contains a soft/external link (not copyable in place yet)",
                        ));
                    }
                },
                MessageType::DataLayout => layout = Some((body, body_end - body)),
                _ => {}
            }
            if msg_type != MessageType::Link {
                non_link.extend_from_slice(&region[p..body_end]);
            }
            p = body_end;
        }

        if let Some((lbody, lsize)) = layout {
            let version = region[lbody];
            if !(version == 3 || version == 4) || lsize < 2 {
                return Err(Error::EditUnsupported(
                    "an unsupported data-layout version cannot be copied in place yet",
                ));
            }
            let class = region[lbody + 1];
            match class {
                0 => Ok(ObjModel::DatasetVerbatim { region }),
                1 => {
                    if lbody + 18 > region.len() {
                        return Err(Error::EditUnsupported("malformed contiguous data layout"));
                    }
                    let data_addr =
                        u64::from_le_bytes(region[lbody + 2..lbody + 10].try_into().unwrap());
                    let data_size =
                        u64::from_le_bytes(region[lbody + 10..lbody + 18].try_into().unwrap());
                    Ok(ObjModel::DatasetContiguous {
                        region,
                        addr_off: lbody + 2,
                        data_addr,
                        data_size,
                    })
                }
                _ => Err(Error::EditUnsupported(
                    "chunked datasets cannot be copied in place yet",
                )),
            }
        } else if has_link_info {
            Ok(ObjModel::Group {
                non_link_region: non_link,
                children,
            })
        } else {
            Err(Error::EditUnsupported(
                "an object is neither a contiguous/compact dataset nor a group",
            ))
        }
    }

    /// Recursively validate that the object at `addr` (and, for a group, its
    /// whole subtree) can be copied, without writing anything.
    fn plan_copy(&self, addr: usize, depth: u32) -> Result<(), Error> {
        if depth >= MAX_COPY_DEPTH {
            return Err(Error::EditUnsupported(
                "copy source nests too deeply (possible hard-link cycle)",
            ));
        }
        if let ObjModel::Group { children, .. } = self.read_object(addr)? {
            for (_, child) in children {
                let child = usize::try_from(child)
                    .map_err(|_| Error::EditUnsupported("child address exceeds this platform"))?;
                self.plan_copy(child, depth + 1)?;
            }
        }
        Ok(())
    }

    /// Recursively deep-copy the object at `addr`, appending fresh copies of
    /// every object (data blobs and headers) at end-of-file and returning the new
    /// object-header address of the copied root.
    fn perform_copy(&mut self, addr: usize, depth: u32) -> Result<u64, Error> {
        if depth >= MAX_COPY_DEPTH {
            return Err(Error::EditUnsupported(
                "copy source nests too deeply (possible hard-link cycle)",
            ));
        }
        match self.read_object(addr)? {
            ObjModel::DatasetVerbatim { region } => {
                let oh = build_v2_object_header(&region);
                self.append(&oh)
            }
            ObjModel::DatasetContiguous {
                mut region,
                addr_off,
                data_addr,
                data_size,
            } => {
                let start = usize::try_from(data_addr)
                    .map_err(|_| Error::EditUnsupported("data address exceeds this platform"))?;
                let len = usize::try_from(data_size)
                    .map_err(|_| Error::EditUnsupported("data size exceeds this platform"))?;
                let end = start
                    .checked_add(len)
                    .filter(|&e| e <= self.data.len())
                    .ok_or(Error::EditUnsupported("dataset data is out of bounds"))?;
                let data = self.data[start..end].to_vec();
                let new_data_addr = self.append(&data)?;
                region[addr_off..addr_off + 8].copy_from_slice(&new_data_addr.to_le_bytes());
                let oh = build_v2_object_header(&region);
                self.append(&oh)
            }
            ObjModel::Group {
                non_link_region,
                children,
            } => {
                let mut region = non_link_region;
                for (name, child) in children {
                    let child = usize::try_from(child).map_err(|_| {
                        Error::EditUnsupported("child address exceeds this platform")
                    })?;
                    let new_child = self.perform_copy(child, depth + 1)?;
                    region.extend_from_slice(&encode_link_message(&name, new_child));
                }
                let oh = build_v2_object_header(&region);
                self.append(&oh)
            }
        }
    }

    /// Append `bytes` at end-of-file, updating both the mirror and the file.
    /// Returns the absolute address the bytes were written at.
    fn append(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        // Write to disk before updating the in-memory mirror, so a failed write
        // never leaves the mirror ahead of the file on disk.
        let addr = self.data.len() as u64;
        self.handle.seek(SeekFrom::Start(addr)).map_err(Error::Io)?;
        self.handle.write_all(bytes).map_err(Error::Io)?;
        self.data.extend_from_slice(bytes);
        Ok(addr)
    }

    /// Overwrite bytes in place at `offset`, updating both the mirror and the
    /// file. The caller guarantees the range already exists.
    fn write_at(&mut self, offset: usize, bytes: &[u8]) -> Result<(), Error> {
        // Write to disk before updating the in-memory mirror (see `append`).
        self.handle
            .seek(SeekFrom::Start(offset as u64))
            .map_err(Error::Io)?;
        self.handle.write_all(bytes).map_err(Error::Io)?;
        self.data[offset..offset + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }
}

/// A dirty group in the edit plan: its base object-header message region and the
/// additions targeting it.
#[derive(Default)]
struct Node {
    is_new: bool,
    datasets: Vec<DatasetBuilder>,
    /// Names of links to remove from this group (from `delete`).
    deletes: Vec<String>,
    /// Copies to add to this group: (new link name, source object-header addr).
    copies: Vec<(String, u64)>,
    base_region: Vec<u8>,
    existing_links: Vec<String>,
}

/// A source object parsed for copying. Headers are reproduced from their
/// verbatim message bytes; only the contiguous data address and child link
/// targets are repointed to the freshly-written copies.
enum ObjModel {
    /// A compact dataset (data inline in the header): copy the region verbatim.
    DatasetVerbatim { region: Vec<u8> },
    /// A contiguous dataset: copy the region, repointing the data address at
    /// `addr_off` (region-relative) to a fresh copy of `[data_addr, +data_size)`.
    DatasetContiguous {
        region: Vec<u8>,
        addr_off: usize,
        data_addr: u64,
        data_size: u64,
    },
    /// A group: every non-link message verbatim, plus its hard-link children to
    /// copy and re-link by name.
    Group {
        non_link_region: Vec<u8>,
        children: Vec<(String, u64)>,
    },
}

/// The validated, chunk-collapsed message region and existing link names of a
/// group header.
struct GroupInfo {
    region: Vec<u8>,
    link_names: Vec<String>,
}

/// A staged dataset reduced to the pieces the writer needs.
struct FlatDataset {
    name: String,
    dt: crate::datatype::Datatype,
    ds: Dataspace,
    raw: Vec<u8>,
    attrs: Vec<crate::attribute::AttributeMessage>,
}

/// Split a path into non-empty components.
fn split_path(path: &str) -> PathKey {
    path.split('/')
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

/// Ensure a node exists for every ancestor prefix of `path` (so each is rebuilt
/// and can re-wire its child link). Does not set `is_new`.
fn ensure_ancestors(nodes: &mut BTreeMap<PathKey, Node>, path: &[String]) {
    for len in 0..=path.len() {
        nodes.entry(path[..len].to_vec()).or_default();
    }
}

/// Validate a staged dataset and reduce it to a [`FlatDataset`], rejecting any
/// feature this engine cannot emit as contiguous, unfiltered storage.
fn flatten_dataset(db: DatasetBuilder) -> Result<FlatDataset, Error> {
    if db.name.is_empty() {
        return Err(Error::EditUnsupported("dataset path has an empty name"));
    }
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
    // The link message body (whose length is independent of the address) must
    // fit the object-header message's u16 size field; a pathologically long
    // name would otherwise overflow it into silent corruption.
    if make_link(&db.name, 0).serialize(OFFSET_SIZE).len() > u16::MAX as usize {
        return Err(Error::EditUnsupported(
            "dataset name is too long to encode as a link message",
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

/// The chunk-0 message region of a fresh, empty compact-link group: a single
/// LinkInfo message advertising no dense storage. Mirrors `build_group_oh`.
fn fresh_group_region() -> Vec<u8> {
    let mut li = Vec::with_capacity(18);
    li.push(0); // version
    li.push(0); // flags
    li.extend_from_slice(&u64::MAX.to_le_bytes()); // fractal heap addr = UNDEF
    li.extend_from_slice(&u64::MAX.to_le_bytes()); // btree name index addr = UNDEF
    let mut m = Vec::with_capacity(4 + li.len());
    m.push(MessageType::LinkInfo.to_u16() as u8);
    m.extend_from_slice(&(li.len() as u16).to_le_bytes());
    m.push(0); // message flags
    m.extend_from_slice(&li);
    m
}

/// Encode a complete object-header Link message (4-byte record header + body)
/// for a hard link `name -> addr`. The caller must have validated that the body
/// fits the u16 size field (see [`flatten_dataset`]); group names are short.
fn encode_link_message(name: &str, addr: u64) -> Vec<u8> {
    let body = make_link(name, addr).serialize(OFFSET_SIZE);
    let mut m = Vec::with_capacity(4 + body.len());
    m.push(MessageType::Link.to_u16() as u8);
    m.extend_from_slice(&(body.len() as u16).to_le_bytes());
    m.push(0); // message flags
    m.extend_from_slice(&body);
    m
}

/// Patch an existing hard Link message in a chunk-0 message `region`, retargeting
/// the link named `name` to `new_addr` (used to repoint a parent at a relocated
/// child group). The target address is the trailing `OFFSET_SIZE` bytes of the
/// link body for a hard link.
fn patch_link_target(region: &mut [u8], name: &str, new_addr: u64) -> Result<(), Error> {
    let mut p = 0;
    while let Some((msg_type, body, body_end)) = next_message(region, p)? {
        if msg_type == MessageType::Link {
            if let Ok(link) = LinkMessage::parse(&region[body..body_end], OFFSET_SIZE) {
                if link.name == name {
                    return match link.link_target {
                        LinkTarget::Hard { .. } => {
                            let ofs = body_end - OFFSET_SIZE as usize;
                            region[ofs..body_end].copy_from_slice(&new_addr.to_le_bytes());
                            Ok(())
                        }
                        _ => Err(Error::EditUnsupported(
                            "a group on the edited path is reached by a soft/external link",
                        )),
                    };
                }
            }
        }
        p = body_end;
    }
    Err(Error::EditUnsupported(
        "expected child link not found in parent group",
    ))
}

/// Copy a chunk-0 message `region`, dropping the single Link message named
/// `name` and preserving every other message verbatim (used by `delete`). Errors
/// if no such link is present.
fn remove_link_from_region(region: &[u8], name: &str) -> Result<Vec<u8>, Error> {
    let mut out = Vec::with_capacity(region.len());
    let mut p = 0;
    let mut removed = false;
    while let Some((msg_type, body, body_end)) = next_message(region, p)? {
        let mut skip = false;
        if msg_type == MessageType::Link {
            if let Ok(link) = LinkMessage::parse(&region[body..body_end], OFFSET_SIZE) {
                if link.name == name {
                    skip = true;
                    removed = true;
                }
            }
        }
        if !skip {
            out.extend_from_slice(&region[p..body_end]);
        }
        p = body_end;
    }
    if p < region.len() {
        out.extend_from_slice(&region[p..]);
    }
    if !removed {
        return Err(Error::EditUnsupported(
            "link to delete not found in its parent group",
        ));
    }
    Ok(out)
}

/// Whether `a` is a path prefix of (or equal to) `b`.
fn is_prefix(a: &[String], b: &[String]) -> bool {
    a.len() <= b.len() && b[..a.len()] == *a
}

/// Parse the version-2 object-header message record at `p` within a chunk-0
/// message region, returning `(message type, body start, body end)`; the next
/// record begins at `body end`. Returns `Ok(None)` once fewer than 4 bytes
/// remain (a clean end of the region), and `Err` if a record's declared body
/// runs past the region. Centralizes the bounds check shared by every walker.
fn next_message(region: &[u8], p: usize) -> Result<Option<(MessageType, usize, usize)>, Error> {
    if p + 4 > region.len() {
        return Ok(None);
    }
    let msg_type = MessageType::from_u16(region[p] as u16);
    let msg_size = u16::from_le_bytes([region[p + 1], region[p + 2]]) as usize;
    let body = p + 4;
    let body_end = body + msg_size;
    if body_end > region.len() {
        return Err(Error::EditUnsupported("malformed object header message"));
    }
    Ok(Some((msg_type, body, body_end)))
}

/// Wrap a chunk-0 message region in a fresh single-chunk version 2 object header
/// (`OHDR` prefix + region + Jenkins checksum). Mirrors the encoding in
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
