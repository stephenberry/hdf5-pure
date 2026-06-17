//! In-place editing of an existing HDF5 file (issue #32, Group C).
//!
//! [`EditSession`] opens an existing file and adds objects or edits compact
//! group attributes **in place**:
//! new data and object headers are written at the end of the file, and the
//! object headers of the touched groups (and their ancestors up to the root)
//! are rewritten — also appended — so the superblock ends up pointing at the
//! new root header. Nothing already in the file is moved, so the cost is
//! proportional to what you add, not to the file size — unlike the
//! read-everything-then-rebuild path through [`FileBuilder`](crate::FileBuilder).
//!
//! Both new datasets, new (sub)groups, and group attribute edits are supported,
//! at any existing group path. Adding into a nested group `/a/b` rewrites `b`'s
//! header (with the new link), then `a`'s header (repointing its link to `b`'s
//! new location), then the root's — "relocation up the tree". This is always
//! safe for *additions* because no surviving object is relocated except the
//! groups on the path being edited, and those are reachable only through links
//! this same commit rewrites (the root through the superblock); absolute
//! object-reference addresses to other objects stay valid.
//!
//! Deletion ([`EditSession::delete`], the HDF5 `H5Ldelete`) is the mirror image:
//! the parent group's header is rebuilt without the removed link, relocated up
//! the tree the same way, and the unlinked object (and its subtree) is freed —
//! its blocks are returned to a session-local free list (see below).
//! Object copy ([`EditSession::copy`], the HDF5 `H5Ocopy`) deep-copies
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
//! - Added datasets may be contiguous *or* chunked, with any filter the
//!   whole-file writer supports (deflate, shuffle, fletcher32, scale-offset,
//!   ZFP), and may declare extensible (maximum, optionally unlimited)
//!   dimensions. A chunked dataset's data and index — and any filtered chunks —
//!   are produced by the same builder the whole-file writer uses and appended at
//!   end-of-file, so its object header is byte-identical to a freshly written
//!   one. Every added dataset must have a fixed-size datatype with a non-empty
//!   shape and carry only fixed-size (non variable-length) attributes, few
//!   enough to stay in compact storage; object-reference and provenance
//!   datasets are not added in place. Group attribute edits have the same
//!   compact, fixed-size attribute restriction.
//! - A new group's parent must already exist or be created in the same session
//!   (each level created explicitly); intermediate groups are not auto-created.
//!
//! # Free-space reuse (issue #21)
//!
//! Each commit vacates space: the object headers it rewrites are superseded, and
//! a deletion abandons its target's blocks. Those regions are recorded in a
//! session-local free list and reused by later commits in the same session —
//! a new object is written into a fitting freed region instead of growing the
//! file, and when freed space forms a run reaching end-of-file the file is
//! physically truncated. The reuse is crash-safe: it only ever overwrites space
//! freed by an *earlier*, already-durable commit (never space the current commit
//! is mid-way through freeing), and truncation happens only after the superblock
//! recording the smaller end-of-file is itself durable.
//!
//! Reclaim is best-effort and conservative. A deleted object whose blocks cannot
//! be enumerated exhaustively — chunked or variable-length storage, dense
//! attribute/link heaps, a non–version-2 header — is left as dead bytes rather
//! than risk freeing a region that is still in use; under-reclaiming only wastes
//! space, while over-reclaiming would corrupt.
//!
//! Whether the free list outlives the session depends on how the file was
//! created. For the default (non-persisting) file it is **not** persisted: it is
//! forgotten on close, so reuse and shrinkage apply to churn within a session,
//! and a single delete-then-close shrinks the file only when the freed bytes
//! reach end-of-file. A file created with
//! `H5Pset_file_space_strategy(persist = true)` instead **persists** its free
//! space: `open` seeds the list from the on-disk free-space managers (the
//! `FSHD`/`FSSE` blocks the superblock-extension File Space Info message points
//! at), and each commit rewrites those managers, so freed regions survive
//! close/reopen and are reused across sessions — by this crate and the reference
//! C library alike. A persisting commit *retains* freed space (recording it on
//! disk) rather than truncating it; the blocks holding the managers are appended
//! past all live data and the superblock is repointed last, so a crash before the
//! repoint leaves the prior file wholly intact. Whole-file compaction that
//! reclaims every hole at once is still the separate repack path.

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::checksum::jenkins_lookup3;
use crate::chunked_write::{ChunkOptions, build_chunked_data_at_ext};
use crate::dataspace::{Dataspace, DataspaceType};
use crate::error::Error;
use crate::file_lock::{self, FileLocking};
use crate::file_space_info::{FileSpaceInfo, FileSpaceStrategy};
use crate::file_writer::{
    LENGTH_SIZE, OFFSET_SIZE, build_chunked_dataset_oh, build_dataset_oh, make_link,
};
use crate::filters::ChunkContext;
use crate::free_space::FreeList;
use crate::free_space_manager::{self, FreeSection, FsmHeader, fshd_len, serialize_file_fsm};
use crate::group_v2::resolve_group_entries;
use crate::link_message::{LinkMessage, LinkTarget};
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::signature;
use crate::superblock::Superblock;
use crate::type_builders::{AttrValue, DatasetBuilder, build_attr_message};

/// An undefined on-disk address (all bits set), HDF5's "no address" sentinel.
const UNDEF: u64 = u64::MAX;

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
/// [`create_dataset`](Self::create_dataset) / [`create_group`](Self::create_group)
/// and group attribute edits with [`set_group_attr`](Self::set_group_attr) /
/// [`remove_group_attr`](Self::remove_group_attr), then apply them with
/// [`commit`](Self::commit).
///
/// # Example
///
/// ```no_run
/// use hdf5_pure::{AttrValue, EditSession};
///
/// let mut session = EditSession::open("existing.h5")?;
/// session.create_group("run2");
/// session.set_group_attr("run2", "kind", AttrValue::AsciiString("trial".into()));
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
    /// Group attribute edits staged as (group path, operation). The path may be
    /// a group created in this same session.
    pending_group_attrs: Vec<(PathKey, GroupAttrOp)>,
    /// Links staged for removal by `delete`, as full paths.
    pending_deletes: Vec<PathKey>,
    /// Object copies staged by `copy`, as (source path, destination full path).
    pending_copies: Vec<(PathKey, PathKey)>,
    /// Session-local free-space tracker (issue #21). Holds regions vacated by
    /// prior commits in this session — superseded object headers and the blocks
    /// of deleted objects — so later commits reuse them instead of growing the
    /// file, and so a freed run reaching end-of-file can be truncated away. It
    /// starts empty on `open` for a non-persisting file: holes already present
    /// from earlier sessions or other tools are not tracked. When the file
    /// persists its free space (`persist` is `Some`), `open` instead seeds it
    /// from the on-disk free-space managers, so reuse spans sessions.
    free: FreeList,
    /// Free-space persistence read from the file's superblock extension on
    /// `open` (the file-creation `H5Pset_file_space_strategy(persist = true)`
    /// setting). `None` for the default non-persisting file; when `Some`, every
    /// [`commit`](Self::commit) rewrites the on-disk free-space managers so the
    /// free list survives close/reopen.
    persist: Option<PersistState>,
}

/// State for a file that persists its free space on disk. Carries the file's
/// fixed file-space parameters and the extents of the free-space-manager blocks
/// (and superblock extension) the *current* on-disk file uses, so the next
/// persisting commit can reclaim them when it writes fresh ones.
struct PersistState {
    strategy: FileSpaceStrategy,
    threshold: u64,
    page_size: u64,
    /// `(addr, len)` of the on-disk superblock-extension header and every
    /// free-space-manager `FSHD`/`FSSE` block currently in use. Superseded — and
    /// therefore freed — by the next persisting commit.
    old_blocks: Vec<(u64, u64)>,
}

impl EditSession {
    /// Open an existing HDF5 file for in-place editing.
    ///
    /// Reads the file into memory and retains a read/write handle. Takes an
    /// exclusive OS advisory lock so the file cannot be opened concurrently by
    /// another writer or reader; the lock is released automatically when the
    /// session is dropped or the process exits (including on a crash). Fails with
    /// [`Error::FileLocked`] if the file is already locked, or
    /// [`Error::EditUnsupported`] if the file is not a supported target (see the
    /// [module docs](self) for the exact requirements). To control or disable
    /// locking, use [`open_with_locking`](Self::open_with_locking) or set
    /// `HDF5_USE_FILE_LOCKING=FALSE`.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        Self::open_with_locking(path, FileLocking::Enabled)
    }

    /// Open an existing HDF5 file for in-place editing, choosing the file-locking
    /// policy explicitly. See [`open`](Self::open) and [`FileLocking`].
    pub fn open_with_locking<P: AsRef<Path>>(path: P, locking: FileLocking) -> Result<Self, Error> {
        let path = path.as_ref();
        let mut handle = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(Error::Io)?;
        // Acquire the exclusive lock before reading or mutating; the retained
        // `handle` holds it for the session's life.
        file_lock::acquire_exclusive(&handle, locking, path)?;
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

        let mut session = Self {
            handle,
            data,
            sb_sig_off,
            superblock,
            pending_datasets: Vec::new(),
            pending_groups: Vec::new(),
            pending_group_attrs: Vec::new(),
            pending_deletes: Vec::new(),
            pending_copies: Vec::new(),
            free: FreeList::new(),
            persist: None,
        };
        // If the file persists its free space, seed the free list from the
        // on-disk managers and arm persistence for future commits. Best-effort:
        // an unreadable or non-persisting extension simply leaves the session in
        // the default, non-persisting mode.
        session.load_persisted_free_space();
        Ok(session)
    }

    /// Read the superblock-extension File Space Info message; if it requests
    /// persistence, seed [`self.free`](Self::free) from the on-disk free-space
    /// managers and record the manager/extension block extents for reclamation on
    /// the next commit. Silent on any malformed or absent metadata — persistence
    /// is then simply off for this session.
    fn load_persisted_free_space(&mut self) {
        if self.superblock.version < 2 {
            return; // no superblock extension exists before v2
        }
        let Some(ext_rel) = self.superblock.superblock_extension_address else {
            return;
        };
        if ext_rel == UNDEF {
            return;
        }
        let Ok(ext_addr) = usize::try_from(ext_rel) else {
            return;
        };
        let Some(info) = self.extension_fsinfo(ext_addr) else {
            return;
        };
        if !info.persist {
            return;
        }
        let os = self.superblock.offset_size;

        // Seed the free list with every persisted section (addresses are stored
        // relative to the base address, which this editor requires to be 0).
        // Defensive against a malformed or corrupt manager: skip a section that is
        // empty, runs past end-of-file, or overlaps one already taken. A
        // well-formed file (this crate's or the C library's) has none of these;
        // tolerating them keeps a bad file from seeding a bogus or double-counted
        // free region that a later commit would hand out into live data.
        if let Ok(mut sections) =
            free_space_manager::read_persisted_sections(&self.data, &info.manager_addrs, 0, os)
        {
            let file_len = self.data.len() as u64;
            sections.sort_by_key(|s| s.addr);
            let mut prev_end = 0u64;
            for s in sections {
                let Some(end) = s.addr.checked_add(s.size) else {
                    continue;
                };
                if s.size == 0 || end > file_len || s.addr < prev_end {
                    continue;
                }
                prev_end = end;
                self.free.free(s.addr, s.size);
            }
        }

        // Record the byte extents of the blocks the live file uses so the next
        // persisting commit frees them when it writes replacements: the
        // extension header, and each defined manager's FSHD + FSSE.
        let mut old_blocks = Vec::new();
        if let Ok(spans) = self.oh_chunk_spans(ext_addr) {
            old_blocks.extend(spans);
        }
        for &m in &info.manager_addrs {
            if m == UNDEF {
                continue;
            }
            let Ok(m_us) = usize::try_from(m) else {
                continue;
            };
            let Some(slice) = self.data.get(m_us..) else {
                continue;
            };
            if let Ok(h) = FsmHeader::parse(slice, os) {
                // `FsmHeader::parse` succeeding guarantees the header's own bytes
                // are present, so the FSHD extent is in-bounds; validate the
                // section-info extent before recording it, so a malformed
                // `fsse_used` can't later free a region running past end-of-file.
                old_blocks.push((m, fshd_len(os)));
                if h.fsse_addr != UNDEF
                    && h.fsse_addr
                        .checked_add(h.fsse_used)
                        .is_some_and(|end| end <= self.data.len() as u64)
                {
                    old_blocks.push((h.fsse_addr, h.fsse_used));
                }
            }
        }

        self.persist = Some(PersistState {
            strategy: info.strategy,
            threshold: info.threshold,
            page_size: info.page_size,
            old_blocks,
        });
    }

    /// Parse the File Space Info message out of the superblock-extension object
    /// header at `ext_addr`, if present and readable.
    fn extension_fsinfo(&self, ext_addr: usize) -> Option<FileSpaceInfo> {
        let os = self.superblock.offset_size;
        let ls = self.superblock.length_size;
        let base = self.superblock.base_address;
        let oh = ObjectHeader::parse_with_base(&self.data, ext_addr, os, ls, base).ok()?;
        let msg = oh
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::FileSpaceInfo)?;
        FileSpaceInfo::parse(&msg.data, os, ls).ok()
    }

    /// Stage a new dataset, added on the next [`commit`](Self::commit). The
    /// argument is the full path of the dataset; everything before the last
    /// component names the parent group, which must exist (or be created in this
    /// session). Returns the [`DatasetBuilder`] — the same builder used by
    /// [`FileBuilder`](crate::FileBuilder) — to configure data, shape, and
    /// attributes.
    ///
    /// The dataset may be contiguous or chunked, and chunked datasets may be
    /// filtered (`with_deflate`, `with_shuffle`, `with_fletcher32`,
    /// `with_scale_offset`, `with_zfp`) and/or extensible (`with_maxshape`); see
    /// the [module docs](self) for what stays unsupported (variable-length or
    /// dense attributes, object-reference and provenance datasets, empty shapes).
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

    /// Stage an attribute add or replacement on a group, applied on the next
    /// [`commit`](Self::commit).
    ///
    /// `path` names the group to edit; `""` or `"/"` names the root group. The
    /// group may already exist or may be created earlier in the same session
    /// with [`create_group`](Self::create_group). Attributes are stored compactly
    /// in the rebuilt group header; variable-length attributes and edits that
    /// would exceed the compact-attribute limit are refused before any file
    /// bytes are changed.
    pub fn set_group_attr(&mut self, path: &str, name: &str, value: AttrValue) -> &mut Self {
        self.pending_group_attrs.push((
            split_path(path),
            GroupAttrOp::Set {
                name: name.to_string(),
                value,
            },
        ));
        self
    }

    /// Stage removal of a compact attribute from a group, applied on the next
    /// [`commit`](Self::commit).
    ///
    /// `path` names the group to edit; `""` or `"/"` names the root group. The
    /// named attribute must exist in the committed group state after any earlier
    /// staged attribute operations for the same group have been applied.
    pub fn remove_group_attr(&mut self, path: &str, name: &str) -> &mut Self {
        self.pending_group_attrs.push((
            split_path(path),
            GroupAttrOp::Remove {
                name: name.to_string(),
            },
        ));
        self
    }

    /// Stage removal of the link at `path` (the HDF5 `H5Ldelete`), applied on the
    /// next [`commit`](Self::commit). The link's object — and, for a group, its
    /// whole subtree — becomes unreachable. The bytes it occupied are returned to
    /// this session's free list (issue #21): a later commit reuses them for new
    /// objects instead of growing the file, and if a freed run reaches
    /// end-of-file the file is truncated. Reclaim is best-effort — an object
    /// whose blocks this engine cannot enumerate exhaustively (chunked or
    /// variable-length storage, dense attribute/link heaps) is left as dead bytes
    /// rather than risk freeing a region that is still in use. Freed space is
    /// reused within the open session; for a file created with
    /// `H5Pset_file_space_strategy(persist = true)` it is also recorded on disk so
    /// it survives reopen (see the [module docs](self)), otherwise it is forgotten
    /// on close. After reuse, an object reference to a deleted object may resolve
    /// to an unrelated object (deleting a referenced object is undefined in HDF5).
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
    /// Appends each new dataset (its data — a contiguous blob, or the chunk data
    /// and index for a chunked/filtered dataset — plus its object header) and
    /// each new group, then appends rewritten object headers for every touched
    /// group and its ancestors up to the root (omitting any deleted links), then
    /// repoints the superblock at the new root. On success the staged set is
    /// cleared and the session can be reused. On any [`Error::EditUnsupported`]
    /// the file on disk is left untouched: the checks that raise it — including
    /// each dataset's filter-pipeline and chunk-geometry validation — all run
    /// before the first byte is written. Should a later step fail mid-apply (an
    /// I/O error, or a residual build error), the superblock — repointed last —
    /// still names the prior root, so the file stays valid and the appended bytes
    /// are unreferenced slack.
    pub fn commit(&mut self) -> Result<(), Error> {
        if self.pending_datasets.is_empty()
            && self.pending_groups.is_empty()
            && self.pending_group_attrs.is_empty()
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
        let mut attr_targets: Vec<PathKey> = Vec::new();

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

        // Stage group attribute edits against their target groups. A target may
        // be a newly-created group from this same commit, but not a copied
        // destination or a dataset being added in the same commit.
        for (path, op) in std::mem::take(&mut self.pending_group_attrs) {
            ensure_ancestors(&mut nodes, &path);
            nodes.entry(path.clone()).or_default().attr_ops.push(op);
            attr_targets.push(path);
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
        // `deleted_addrs` keeps each removed object's header address so its owned
        // blocks can be reclaimed after the commit lands (issue #21).
        let deletes = std::mem::take(&mut self.pending_deletes);
        let mut deleted_addrs: Vec<usize> = Vec::new();
        for (i, d) in deletes.iter().enumerate() {
            if d.is_empty() {
                return Err(Error::EditUnsupported("cannot delete the root group"));
            }
            let path_str = d.join("/");
            let del_addr =
                crate::group_v2::resolve_path_any(&self.data, &self.superblock, &path_str)
                    .map_err(|_| Error::EditUnsupported("nothing to delete at the given path"))?;
            if let Ok(a) = usize::try_from(del_addr) {
                deleted_addrs.push(a);
            }
            for t in &add_targets {
                if is_prefix(d, t) || is_prefix(t, d) {
                    return Err(Error::EditUnsupported(
                        "a deletion overlaps an addition in the same commit; use separate commits",
                    ));
                }
            }
            for t in &attr_targets {
                if is_prefix(d, t) {
                    return Err(Error::EditUnsupported(
                        "a deletion overlaps a group-attribute edit in the same commit; use separate commits",
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
        // Every existing dirty group is rewritten to a freshly-appended header,
        // so its old header becomes dead bytes once the superblock is repointed;
        // `superseded_addrs` records those old headers for reclamation (#21).
        let keys: Vec<PathKey> = nodes.keys().cloned().collect();
        let mut superseded_addrs: Vec<usize> = Vec::new();
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
                superseded_addrs.push(addr);
                let node = nodes.get_mut(key).unwrap();
                node.base_region = info.region;
                node.existing_links = info.link_names;
            }
        }

        // Apply and validate group attribute edits before any writes. This keeps
        // unsupported attribute edits under the same all-or-nothing preflight
        // contract as unsupported dataset additions.
        for key in &keys {
            let node = nodes.get_mut(key).unwrap();
            let ops = std::mem::take(&mut node.attr_ops);
            if !ops.is_empty() {
                let region = std::mem::take(&mut node.base_region);
                node.base_region = apply_group_attr_ops(&region, &ops)?;
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

        // Gather the regions this commit will vacate, read from the current
        // on-disk layout before any byte moves: every deleted object's owned
        // blocks plus every superseded group header. These are not added to the
        // free list until after the superblock repoint (they remain live until
        // then), so the appends below never reuse them. Enumeration is
        // best-effort — `collect_free_spans` simply omits anything it cannot
        // account for exhaustively, so the worst case is unreclaimed dead bytes,
        // never a freed-but-live region.
        let mut to_free: Vec<(u64, u64)> = Vec::new();
        for &a in &deleted_addrs {
            self.collect_free_spans(a, 0, &mut to_free);
        }
        for &a in &superseded_addrs {
            if let Ok(spans) = self.oh_chunk_spans(a) {
                to_free.extend(spans);
            }
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
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "src_addr is an offset into self.data, the in-memory file image \
                              (a Vec<u8>), so it always fits usize on the running target"
                )]
                let root = self.perform_copy(src_addr as usize, 0)?;
                region.extend_from_slice(&encode_link_message(&leaf, root));
            }

            // Datasets directly under this group.
            for fd in flat.remove(key).into_iter().flatten() {
                let oh = if fd.chunk_options.is_chunked() || fd.maxshape.is_some() {
                    self.build_chunked_dataset(&fd)?
                } else {
                    let data_addr = self.alloc_or_append(&fd.raw)?;
                    build_dataset_oh(
                        &fd.dt,
                        &fd.ds,
                        data_addr,
                        fd.raw.len() as u64,
                        &fd.attrs,
                        None,
                    )
                };
                let oh_addr = self.alloc_or_append(&oh)?;
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
            let addr = self.alloc_or_append(&oh)?;
            new_addr.insert(key.clone(), addr);
        }

        // Repoint the superblock at the new root last: this is the commit's
        // linearization point. Until it lands, the file on disk still points at
        // the old root (the appended objects are merely unreferenced trailing
        // bytes), so a failure here leaves a valid file.
        //
        // That ordering is only crash-safe if the appended objects are durable
        // before the root pointer is flipped; otherwise a power loss could
        // persist the flip ahead of the data it references, leaving the root
        // pointing at bytes that never reached disk. `flush` on a plain `File`
        // does not force a write-back, so sync the appended bytes to disk first
        // (the barrier), then flip the pointer, then sync the flip.
        let new_root = new_addr[&PathKey::new()];

        // A persisting file keeps its freed space recorded on disk rather than
        // truncating it away, so its commit takes a different, append-only tail.
        if self.persist.is_some() {
            return self.commit_persisting(new_root, to_free);
        }

        // The new tree is fully written, so the regions this commit vacated are
        // now dead: hand them to the session free list. If the resulting free
        // space forms a run reaching end-of-file, the file can be physically
        // truncated to where that run starts; otherwise the end-of-file is
        // unchanged. `take_trailing` removes the trimmed run so it is not also
        // counted as reusable interior space.
        for (a, l) in to_free.drain(..) {
            self.free.free(a, l);
        }
        let cur_eof = self.data.len() as u64;
        let trunc_to = self.free.take_trailing(cur_eof);
        let new_eof = trunc_to.unwrap_or(cur_eof);

        self.handle.sync_all().map_err(Error::Io)?;
        if self.superblock.version >= 2 {
            // Build the new superblock off a clone and adopt it only once the
            // write succeeds, so a failed write does not desync the in-memory
            // state. The v2/v3 superblock carries its own checksum.
            let mut new_sb = self.superblock.clone();
            new_sb.root_group_address = new_root;
            new_sb.eof_address = new_eof;
            // Clear any write/SWMR consistency flag rather than re-emitting one
            // the source file carried (e.g. left set by a crashed SWMR writer):
            // this clean commit leaves the file properly closed for the C library
            // (issue #73). serialize() recomputes the v2/v3 checksum.
            new_sb.consistency_flags = 0;
            let sb_bytes = new_sb.serialize();
            self.write_at(self.sb_sig_off, &sb_bytes)?;
            self.handle.sync_all().map_err(Error::Io)?;
            self.superblock = new_sb;
        } else {
            self.repoint_v0v1_root(new_root, new_eof)?;
            self.handle.sync_all().map_err(Error::Io)?;
            self.superblock.root_group_address = new_root;
            self.superblock.eof_address = new_eof;
        }

        // Physically shrink the file only after the superblock — now carrying the
        // smaller end-of-file — is durable. A crash between the two leaves a file
        // whose superblock end-of-file is correct and whose trailing bytes are
        // mere unreferenced slack, which the next open ignores; the reverse order
        // could advertise an end-of-file past the actual file length.
        if let Some(cut) = trunc_to {
            self.handle.set_len(cut).map_err(Error::Io)?;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "cut is a shrink target <= the current file length, which equals \
                          self.data.len() (a usize)"
            )]
            self.data.truncate(cut as usize);
            self.handle.sync_all().map_err(Error::Io)?;
        }
        Ok(())
    }

    /// Commit tail for a file that persists its free space (issue #21). Unlike
    /// the non-persisting path, freed space is *retained* and recorded on disk —
    /// matching the reference library's persistent free-space strategy — so a
    /// later reopen (by this crate or the C library) recovers it.
    ///
    /// The post-commit free list (this commit's vacated regions plus the now-dead
    /// old free-space-manager and extension blocks) is serialized into a fresh
    /// `FSHD`/`FSSE` pair and a rewritten superblock-extension File Space Info
    /// message, all appended at the current end-of-file. Nothing live or
    /// still-referenced is overwritten: the new blocks sit strictly past the old
    /// ones, and the superblock — repointed last — is the linearization point. A
    /// crash before it leaves the prior file (root, extension, and managers)
    /// wholly intact.
    fn commit_persisting(&mut self, new_root: u64, to_free: Vec<(u64, u64)>) -> Result<(), Error> {
        let os = self.superblock.offset_size;
        let (strategy, threshold, page_size, old_blocks) = {
            // Copy what we need so no borrow of `self.persist` is held across the
            // `&mut self` writes below; the old state stays in place so a failure
            // leaves the session reusable.
            let ps = self
                .persist
                .as_ref()
                .expect("commit_persisting is only called when persistence is armed");
            (
                ps.strategy,
                ps.threshold,
                ps.page_size,
                ps.old_blocks.clone(),
            )
        };

        // The free list the new managers will record: this commit's vacated
        // regions plus the superseded FSM/extension blocks (dead once we
        // repoint), coalesced. Built in a temp so `self.free` and the on-disk old
        // blocks stay untouched until after the superblock repoint.
        let mut post = self.free.clone();
        for &(a, l) in &to_free {
            post.free(a, l);
        }
        for &(a, l) in &old_blocks {
            post.free(a, l);
        }
        let sections: Vec<FreeSection> = post
            .sections()
            .into_iter()
            .map(|(addr, size)| FreeSection { addr, size })
            .collect();

        let old_ext_rel = self
            .superblock
            .superblock_extension_address
            .filter(|&a| a != UNDEF)
            .ok_or(Error::EditUnsupported(
                "a persisting file has no superblock extension to update",
            ))?;
        let old_ext_addr = usize::try_from(old_ext_rel)
            .map_err(|_| Error::EditUnsupported("extension address exceeds this platform"))?;

        // The persist File Space Info message is fixed-size, so the rewritten
        // extension's length is independent of the addresses it will carry: size
        // it with a placeholder to place the FSM blocks that follow it.
        let placeholder =
            FileSpaceInfo::persistent_single_manager(strategy, threshold, page_size, 0, 0);
        let ext_len =
            build_v2_object_header(&self.rewrite_extension_region(old_ext_addr, &placeholder)?)
                .len() as u64;

        let ext_addr = self.data.len() as u64;
        let fshd_addr = ext_addr + ext_len;

        // Build the real extension and the FSM blocks. With no free space to
        // record we still refresh the extension (persist on, managers undefined).
        let (ext_oh, fsm_blocks, final_eof) = if sections.is_empty() {
            let info = FileSpaceInfo::persistent_empty(strategy, threshold, page_size);
            let ext_oh =
                build_v2_object_header(&self.rewrite_extension_region(old_ext_addr, &info)?);
            let final_eof = ext_addr + ext_oh.len() as u64;
            (ext_oh, None, final_eof)
        } else {
            let fsse_addr = fshd_addr + fshd_len(os);
            // `eoa_pre_fsm` is the end-of-allocation before the free-space-manager
            // section blocks (`FSHD`/`FSSE`) were allocated: a consumer may shrink
            // back to here and rebuild them. It points at the FSHD, not the
            // extension — the extension sits below it and persists, so shrinking
            // leaves the superblock and its extension pointer valid (only the
            // manager blocks, which are rewritten every commit, are discarded).
            // This matches the C library's convention of keeping the superblock
            // extension stable across closes, and is the value `H5Fget_freespace`
            // accounts for correctly (verified in the crosscheck).
            let eoa_pre_fsm = fshd_addr;
            let info = FileSpaceInfo::persistent_single_manager(
                strategy,
                threshold,
                page_size,
                fshd_addr,
                eoa_pre_fsm,
            );
            let ext_oh =
                build_v2_object_header(&self.rewrite_extension_region(old_ext_addr, &info)?);
            debug_assert_eq!(
                ext_oh.len() as u64,
                ext_len,
                "extension length must be stable across the placeholder and real messages"
            );
            let (fshd, fsse) = serialize_file_fsm(&sections, fshd_addr, fsse_addr, os);
            let final_eof = fsse_addr + fsse.len() as u64;
            (ext_oh, Some((fshd, fsse)), final_eof)
        };

        // Append the extension, then the FSM blocks, at end-of-file. They are
        // unreferenced until the superblock repoint, so a crash here is harmless.
        let written_ext = self.append(&ext_oh)?;
        debug_assert_eq!(written_ext, ext_addr);
        let mut new_old_blocks = vec![(ext_addr, ext_oh.len() as u64)];
        if let Some((fshd, fsse)) = fsm_blocks {
            let wf = self.append(&fshd)?;
            debug_assert_eq!(wf, fshd_addr);
            new_old_blocks.push((fshd_addr, fshd.len() as u64));
            let ws = self.append(&fsse)?;
            new_old_blocks.push((ws, fsse.len() as u64));
        }

        // Barrier, then repoint the superblock (root, eof, and the new extension)
        // — the linearization point — and sync it.
        self.handle.sync_all().map_err(Error::Io)?;
        let mut new_sb = self.superblock.clone();
        new_sb.root_group_address = new_root;
        new_sb.eof_address = final_eof;
        new_sb.superblock_extension_address = Some(ext_addr);
        // Clear any leftover write/SWMR consistency flag on a clean commit (see
        // the non-persisting path above and issue #73).
        new_sb.consistency_flags = 0;
        let sb_bytes = new_sb.serialize();
        self.write_at(self.sb_sig_off, &sb_bytes)?;
        self.handle.sync_all().map_err(Error::Io)?;
        self.superblock = new_sb;

        // The repoint is durable: the prior free list plus this commit's vacated
        // regions are now genuinely free, and the freshly written blocks become
        // the ones a future commit will supersede.
        self.free = post;
        self.persist = Some(PersistState {
            strategy,
            threshold,
            page_size,
            old_blocks: new_old_blocks,
        });
        Ok(())
    }

    /// Rebuild the superblock-extension object header's message region with its
    /// File Space Info message replaced by `info` (every other message preserved
    /// verbatim), ready to wrap with [`build_v2_object_header`]. The persisting
    /// message is fixed-size, so this never changes the region's length.
    fn rewrite_extension_region(
        &self,
        ext_addr: usize,
        info: &FileSpaceInfo,
    ) -> Result<Vec<u8>, Error> {
        let region = self.gather_oh_messages(ext_addr)?;
        let new_body = info.serialize();
        // The message body is the fixed-size File Space Info record (≤ 125 bytes),
        // so it always fits the u16 size field; `try_from` keeps this off the
        // 32-bit narrowing-cast ledger.
        let new_len = u16::try_from(new_body.len())
            .map_err(|_| Error::EditUnsupported("File Space Info message too large"))?;
        let mut out = Vec::with_capacity(region.len());
        let mut p = 0;
        let mut replaced = false;
        while let Some((msg_type, _body, body_end)) = next_message(&region, p)? {
            if msg_type == MessageType::FileSpaceInfo {
                out.push(region[p]); // message type byte
                out.extend_from_slice(&new_len.to_le_bytes());
                out.push(region[p + 3]); // preserve the message flags (0x14)
                out.extend_from_slice(&new_body);
                replaced = true;
            } else {
                out.extend_from_slice(&region[p..body_end]);
            }
            p = body_end;
        }
        if !replaced {
            // Persistence is armed only when the extension already carries a File
            // Space Info message, so this is unreachable; refuse rather than
            // silently restructure an extension we did not understand.
            return Err(Error::EditUnsupported(
                "a persisting file's superblock extension has no File Space Info message",
            ));
        }
        Ok(out)
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
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "message type ids are a small enum that fits the 1-byte v2 type field"
                )]
                region.push(MessageType::Attribute.to_u16() as u8);
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "attribute body length fits the 2-byte message-size field (oversized \
                              bodies are rejected above)"
                )]
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
        let mut region = self.gather_oh_messages(addr)?;
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
        // Heal headers written by older hdf5-pure releases that omitted the
        // Group Info message, so the rewritten group stays writable by the C
        // library.
        ensure_group_info(&mut region)?;
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
            // A copied group must carry a Group Info message so the copy stays
            // writable by the C library, even when the source omitted it.
            ensure_group_info(&mut non_link)?;
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
                self.alloc_or_append(&oh)
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
                let new_data_addr = self.alloc_or_append(&data)?;
                region[addr_off..addr_off + 8].copy_from_slice(&new_data_addr.to_le_bytes());
                let oh = build_v2_object_header(&region);
                self.alloc_or_append(&oh)
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
                self.alloc_or_append(&oh)
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

    /// Place `bytes` either in a reusable free region left by a prior commit
    /// (overwriting it in place) or, failing that, by appending at end-of-file.
    /// Returns the address written to.
    ///
    /// Reuse only ever draws from [`self.free`](Self::free), which holds regions
    /// vacated by *earlier* commits in this session — never space the current
    /// commit is about to free — so the bytes it overwrites are already
    /// unreachable from the on-disk root and a mid-commit crash cannot corrupt
    /// the live tree (the superblock still points at the prior, intact root).
    fn alloc_or_append(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        if let Some(addr) = self.free.alloc(bytes.len() as u64) {
            self.write_at(
                usize::try_from(addr).map_err(|_| {
                    Error::EditUnsupported("free-region address exceeds this platform")
                })?,
                bytes,
            )?;
            Ok(addr)
        } else {
            self.append(bytes)
        }
    }

    /// Lay out a chunked / filtered / extensible dataset and return its object
    /// header bytes (which the caller links into the parent group).
    ///
    /// The chunk data and index (B-tree v1 / fixed-array / extensible-array, with
    /// any filter pipeline applied) are produced as one relocatable blob by
    /// [`build_chunked_data_at_ext`], whose internal layout — and therefore total
    /// size — is independent of the base address it is given. The blob is
    /// appended at end-of-file, so passing the current end-of-file as the base
    /// makes every absolute address it embeds (chunk addresses, index-structure
    /// addresses, the addresses in the data-layout message) land exactly where
    /// the bytes are written. The header is then built with
    /// [`build_chunked_dataset_oh`] — the same function the whole-file writer
    /// uses — so the header is byte-identical to one written fresh.
    ///
    /// Unlike the contiguous path the blob is always *appended* rather than
    /// placed via [`alloc_or_append`]: reusing an interior freed region would
    /// require knowing the blob's size before building it at that region's
    /// address, and appending keeps the address known up front. Freed space is
    /// still reused for the object header and for every other object in the
    /// commit.
    fn build_chunked_dataset(&mut self, fd: &FlatDataset) -> Result<Vec<u8>, Error> {
        let base = self.data.len() as u64;
        let chunk_dims = fd.chunk_options.resolve_chunk_dims(&fd.ds.dimensions);
        let ctx = ChunkContext::from_datatype(&chunk_dims, &fd.dt);
        let result = build_chunked_data_at_ext(
            &fd.raw,
            &fd.ds.dimensions,
            ctx,
            &fd.chunk_options,
            base,
            fd.maxshape.as_deref(),
        )?;
        // `append` writes at the current end-of-file, which equals `base`: the
        // blob lands exactly where its embedded addresses expect.
        let written = self.append(&result.data_bytes)?;
        debug_assert_eq!(
            written, base,
            "chunk blob must land at the base address it was built for",
        );
        Ok(build_chunked_dataset_oh(
            &fd.dt,
            &fd.ds,
            &result.layout_message,
            result.pipeline_message.as_deref(),
            &fd.attrs,
            None,
        ))
    }

    /// On-disk byte spans `(addr, len)` of every chunk of the version 2 object
    /// header at `addr`: chunk 0 (signature, prefix, messages, checksum) plus
    /// each continuation (`OCHK`) block. Used to reclaim a header's storage when
    /// its object is deleted. An error (propagated from [`oh_region`] or a
    /// malformed continuation) means the header is not a plain v2 header this
    /// engine can fully account for, and the caller leaves it as dead bytes
    /// rather than guess its extent.
    fn oh_chunk_spans(&self, addr: usize) -> Result<Vec<(u64, u64)>, Error> {
        let (rs, re) = self.oh_region(addr)?;
        let d = &self.data;
        // Chunk 0 spans from the header start through its trailing checksum;
        // `oh_region` guarantees `re + 4 <= d.len()`.
        let mut spans: Vec<(u64, u64)> = vec![(addr as u64, (re + 4 - addr) as u64)];
        // Walk continuation messages exactly as `gather_oh_messages` does, but
        // record each OCHK block's extent instead of collecting its messages.
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
                    if body_end - body < (OFFSET_SIZE + LENGTH_SIZE) as usize {
                        return Err(Error::EditUnsupported("malformed continuation message"));
                    }
                    let off = u64::from_le_bytes(d[body..body + 8].try_into().unwrap());
                    let len = u64::from_le_bytes(d[body + 8..body + 16].try_into().unwrap());
                    let off_us = usize::try_from(off).map_err(|_| {
                        Error::EditUnsupported("continuation address exceeds this platform")
                    })?;
                    let len_us = usize::try_from(len).map_err(|_| {
                        Error::EditUnsupported("continuation length exceeds this platform")
                    })?;
                    let blk_end = off_us
                        .checked_add(len_us)
                        .filter(|&e| e <= d.len() && len_us >= 8)
                        .ok_or(Error::EditUnsupported("continuation block out of bounds"))?;
                    if d[off_us..off_us + 4] != *b"OCHK" {
                        return Err(Error::EditUnsupported(
                            "invalid continuation block signature",
                        ));
                    }
                    spans.push((off, len));
                    chunks.push((off_us + 4, blk_end - 4));
                }
                p = body_end;
            }
        }
        Ok(spans)
    }

    /// Best-effort enumeration of every on-disk block owned by the object at
    /// `addr` (and, for a group, its whole subtree), accumulating `(addr, len)`
    /// spans into `out` for reclamation after a delete.
    ///
    /// Deliberately conservative: any object whose layout it cannot fully
    /// account for — a non-v2 header, a chunked or otherwise unsupported data
    /// layout, a group holding a soft/external link, dense attribute storage —
    /// contributes nothing and is not descended into, so `out` never names a
    /// region that might still be in use. Bounded by [`MAX_COPY_DEPTH`] against a
    /// hard-link cycle. Variable-length data in global-heap collections is never
    /// reclaimed here (a collection can be shared), so it is simply left behind.
    fn collect_free_spans(&self, addr: usize, depth: u32, out: &mut Vec<(u64, u64)>) {
        if depth >= MAX_COPY_DEPTH {
            return;
        }
        // The header's own chunks. If they cannot be mapped, account for nothing.
        let spans = match self.oh_chunk_spans(addr) {
            Ok(s) => s,
            Err(_) => return,
        };
        match self.read_object(addr) {
            Ok(ObjModel::DatasetVerbatim { .. }) => out.extend(spans),
            Ok(ObjModel::DatasetContiguous {
                data_addr,
                data_size,
                ..
            }) => {
                out.extend(spans);
                // A defined, in-bounds contiguous data block is owned outright;
                // an empty dataset stores the undefined address and owns none.
                if data_addr != u64::MAX && data_size > 0 {
                    if let (Ok(start), Ok(len)) =
                        (usize::try_from(data_addr), usize::try_from(data_size))
                    {
                        if start.checked_add(len).is_some_and(|e| e <= self.data.len()) {
                            out.push((data_addr, data_size));
                        }
                    }
                }
            }
            Ok(ObjModel::Group { children, .. }) => {
                out.extend(spans);
                for (_, child) in children {
                    if let Ok(c) = usize::try_from(child) {
                        self.collect_free_spans(c, depth + 1, out);
                    }
                }
            }
            // Header maps but the content is unsupported: leak the whole object
            // rather than free a header whose owned blocks we cannot enumerate.
            Err(_) => {}
        }
    }
}

/// A dirty group in the edit plan: its base object-header message region and the
/// additions targeting it.
#[derive(Default)]
struct Node {
    is_new: bool,
    datasets: Vec<DatasetBuilder>,
    /// Compact group-attribute operations to apply to this group.
    attr_ops: Vec<GroupAttrOp>,
    /// Names of links to remove from this group (from `delete`).
    deletes: Vec<String>,
    /// Copies to add to this group: (new link name, source object-header addr).
    copies: Vec<(String, u64)>,
    base_region: Vec<u8>,
    existing_links: Vec<String>,
}

/// A staged compact attribute edit for a group.
enum GroupAttrOp {
    Set { name: String, value: AttrValue },
    Remove { name: String },
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
    /// Chunked/filtered storage options. When [`ChunkOptions::is_chunked`] is
    /// false and `maxshape` is `None`, the dataset is written as contiguous,
    /// unfiltered storage; otherwise its chunk data and index are built by
    /// [`build_chunked_data_at_ext`] and appended at end-of-file.
    chunk_options: ChunkOptions,
    /// Maximum dimensions for an extensible dataset (an unlimited dimension is
    /// `u64::MAX`), mirrored into `ds.max_dimensions`. `None` for a fixed-shape
    /// dataset. A maxshape with an unlimited dimension selects the
    /// extensible-array chunk index; a finite maxshape stays fixed-array/single.
    maxshape: Option<Vec<u64>>,
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

/// Validate a staged dataset and reduce it to a [`FlatDataset`]. Contiguous,
/// unfiltered datasets are emitted as such; chunked, filtered, or extensible
/// datasets carry their [`ChunkOptions`] and maxshape through to the commit,
/// where [`build_chunked_data_at_ext`] lays out their chunk data and index.
/// Rejects any remaining feature this engine cannot reproduce faithfully:
/// object-reference or provenance datasets, variable-length or dense
/// attributes, an empty (zero-element) shape, or a filter pipeline the build
/// cannot construct.
fn flatten_dataset(db: DatasetBuilder) -> Result<FlatDataset, Error> {
    if db.name.is_empty() {
        return Err(Error::EditUnsupported("dataset path has an empty name"));
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
        // Multiply with checked arithmetic: an absurd shape whose element count
        // (or byte size) overflows `u64` is refused rather than panicking in a
        // debug build or silently wrapping in release (which could let a wrapped
        // product spuriously match `raw.len()`).
        let expected = shape
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d))
            .and_then(|n| n.checked_mul(elem));
        match expected {
            Some(expected) if raw.len() as u64 == expected => {}
            Some(_) => {
                return Err(Error::EditUnsupported(
                    "dataset data length does not match its shape",
                ));
            }
            None => {
                return Err(Error::EditUnsupported(
                    "dataset shape is too large to address on this platform",
                ));
            }
        }
    }

    let chunked = db.chunk_options.is_chunked() || db.maxshape.is_some();
    if chunked {
        // Refuse malformed chunk geometry up front (the same validation the
        // whole-file writer applies), so a bad request — chunk dimensions of the
        // wrong rank, a zero chunk dimension, an inconsistent maximum shape, or
        // chunking a scalar — never reaches and panics the chunk splitter, nor
        // yields a dataset the reader cannot decode.
        db.chunk_options
            .validate_geometry(&shape, db.maxshape.as_deref())
            .map_err(Error::EditUnsupported)?;
        // Deflate is compiled out unless the `deflate` feature is on, but
        // `build_pipeline` emits its descriptor regardless; catch a
        // disabled-feature request here so it is refused up front rather than
        // failing mid-apply when a chunk is compressed.
        #[cfg(not(feature = "deflate"))]
        if db.chunk_options.deflate_level.is_some() {
            return Err(Error::EditUnsupported(
                "deflate compression requires the `deflate` crate feature",
            ));
        }
        // Validate the requested filter pipeline now — before any file bytes are
        // written — so an unsupported filter, an incompatible datatype, or a
        // disabled compression feature is refused up front; the chunk data
        // itself is laid out in the commit's apply phase. Chunked/filtered
        // storage flows through the very builder the normal writer uses
        // ([`build_chunked_data_at_ext`] + [`build_chunked_dataset_oh`]), so the
        // resulting object header is byte-identical to a freshly written one.
        let chunk_dims = db.chunk_options.resolve_chunk_dims(&shape);
        let ctx = ChunkContext::from_datatype(&chunk_dims, &dt);
        db.chunk_options
            .build_pipeline(
                ctx.element_size,
                &chunk_dims,
                ctx.element_type,
                ctx.scale_offset_type,
            )
            .map_err(|_| {
                Error::EditUnsupported(
                    "this dataset's filter pipeline cannot be added in place \
                     (an unsupported filter, an incompatible datatype, or a \
                     compression feature that is not enabled)",
                )
            })?;
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
        #[expect(
            clippy::cast_possible_truncation,
            reason = "dataspace rank fits the 1-byte dimensionality field (HDF5 caps rank at 32)"
        )]
        rank: shape.len() as u8,
        dimensions: shape,
        // A chunked, extensible dataset records its maximum dimensions (an
        // unlimited dimension is `u64::MAX`); a fixed-shape dataset has none.
        max_dimensions: db.maxshape.clone(),
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
        chunk_options: db.chunk_options,
        maxshape: db.maxshape,
    })
}

/// A minimal Group Info message body (type 0x000A): version 0 with neither the
/// link-phase-change nor the estimated-entry fields stored. With both absent the
/// HDF5 C library fills `max_compact`/`min_dense` from its own defaults (8 and
/// 6). See [`ensure_group_info`] for why every group needs this message.
const GROUP_INFO_BODY: [u8; 2] = [0, 0];

/// Frame one chunk-0 object-header message record: a 1-byte type, a 2-byte
/// little-endian body length, a 1-byte flags field (always 0 here), then the
/// body. This is the v2 message-record layout used throughout a group's chunk-0
/// message region. Callers pass bodies that fit the u16 length field: link
/// bodies are validated in [`flatten_dataset`], and the Link Info / Group Info
/// bodies are fixed and short.
fn region_message(msg_type: MessageType, body: &[u8]) -> Vec<u8> {
    let mut m = Vec::with_capacity(4 + body.len());
    #[expect(
        clippy::cast_possible_truncation,
        reason = "message type ids are a small enum that fits the 1-byte v2 type field"
    )]
    m.push(msg_type.to_u16() as u8);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "callers pass bodies that fit the 2-byte message-size field (see doc comment)"
    )]
    m.extend_from_slice(&(body.len() as u16).to_le_bytes());
    m.push(0); // message flags
    m.extend_from_slice(body);
    m
}

/// The chunk-0 message region of a fresh, empty compact-link group: a LinkInfo
/// message advertising no dense storage, followed by a GroupInfo message.
/// Mirrors `build_group_oh`.
fn fresh_group_region() -> Vec<u8> {
    let mut li = Vec::with_capacity(18);
    li.push(0); // version
    li.push(0); // flags
    li.extend_from_slice(&u64::MAX.to_le_bytes()); // fractal heap addr = UNDEF
    li.extend_from_slice(&u64::MAX.to_le_bytes()); // btree name index addr = UNDEF
    let mut region = region_message(MessageType::LinkInfo, &li);
    region.extend_from_slice(&region_message(MessageType::GroupInfo, &GROUP_INFO_BODY));
    region
}

/// Ensure a group's chunk-0 message `region` carries a Group Info message,
/// appending a minimal one when absent.
///
/// The HDF5 C library refuses to insert a link into a group whose object header
/// has a Link Info message but no Group Info message: on the new-format path
/// `H5G_obj_insert` reads the Group Info message unconditionally and fails with
/// "message type not found". Such a group round-trips for *reading* but cannot
/// be *modified* by the C library. Earlier hdf5-pure releases wrote groups that
/// way, so heal any such header whenever we rewrite one in place.
fn ensure_group_info(region: &mut Vec<u8>) -> Result<(), Error> {
    let mut p = 0;
    while let Some((msg_type, _body, body_end)) = next_message(region, p)? {
        if msg_type == MessageType::GroupInfo {
            return Ok(());
        }
        p = body_end;
    }
    region.extend_from_slice(&region_message(MessageType::GroupInfo, &GROUP_INFO_BODY));
    Ok(())
}

/// Encode a complete object-header Link message (4-byte record header + body)
/// for a hard link `name -> addr`. The caller must have validated that the body
/// fits the u16 size field (see [`flatten_dataset`]); group names are short.
fn encode_link_message(name: &str, addr: u64) -> Vec<u8> {
    let body = make_link(name, addr).serialize(OFFSET_SIZE);
    region_message(MessageType::Link, &body)
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

/// Apply compact attribute edits to a group message `region`, preserving every
/// non-attribute message verbatim. The result is still a compact-attribute
/// header; dense attribute storage and shared attribute messages are refused.
fn apply_group_attr_ops(region: &[u8], ops: &[GroupAttrOp]) -> Result<Vec<u8>, Error> {
    let mut out = region.to_vec();
    let mut wrote_attr = false;
    for op in ops {
        out = match op {
            GroupAttrOp::Set { name, value } => {
                wrote_attr = true;
                set_attr_in_region(&out, name, value)?
            }
            GroupAttrOp::Remove { name } => remove_attr_from_region(&out, name)?,
        };
    }
    if wrote_attr && compact_attr_count(&out)? > MAX_COMPACT_ATTRS {
        return Err(Error::EditUnsupported(
            "group attributes would exceed compact storage; dense attribute edits are not supported in place yet",
        ));
    }
    Ok(out)
}

/// Copy a message region, dropping all Attribute messages named `name` and then
/// appending a fresh compact Attribute message for `value`.
fn set_attr_in_region(region: &[u8], name: &str, value: &AttrValue) -> Result<Vec<u8>, Error> {
    let new_msg = encode_attr_message(name, value)?;
    let mut out = Vec::with_capacity(region.len() + new_msg.len());
    let mut p = 0;
    while let Some((msg_type, body, body_end)) = next_message(region, p)? {
        match msg_type {
            MessageType::AttributeInfo => {
                return Err(Error::EditUnsupported(
                    "a target group uses dense (fractal-heap) attribute storage (not supported in place yet)",
                ));
            }
            MessageType::Attribute => {
                let attr_name = parse_compact_attr_name(region, p, body, body_end)?;
                if attr_name == name {
                    p = body_end;
                    continue;
                }
            }
            _ => {}
        }
        out.extend_from_slice(&region[p..body_end]);
        p = body_end;
    }
    out.extend_from_slice(&new_msg);
    if p < region.len() {
        out.extend_from_slice(&region[p..]);
    }
    Ok(out)
}

/// Copy a message region, dropping all Attribute messages named `name`.
fn remove_attr_from_region(region: &[u8], name: &str) -> Result<Vec<u8>, Error> {
    let mut out = Vec::with_capacity(region.len());
    let mut p = 0;
    let mut removed = false;
    while let Some((msg_type, body, body_end)) = next_message(region, p)? {
        let mut skip = false;
        match msg_type {
            MessageType::AttributeInfo => {
                return Err(Error::EditUnsupported(
                    "a target group uses dense (fractal-heap) attribute storage (not supported in place yet)",
                ));
            }
            MessageType::Attribute => {
                let attr_name = parse_compact_attr_name(region, p, body, body_end)?;
                if attr_name == name {
                    skip = true;
                    removed = true;
                }
            }
            _ => {}
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
            "group attribute to remove was not found",
        ));
    }
    Ok(out)
}

fn compact_attr_count(region: &[u8]) -> Result<usize, Error> {
    let mut count = 0usize;
    let mut p = 0;
    while let Some((msg_type, _body, body_end)) = next_message(region, p)? {
        if msg_type == MessageType::AttributeInfo {
            return Err(Error::EditUnsupported(
                "a target group uses dense (fractal-heap) attribute storage (not supported in place yet)",
            ));
        }
        if msg_type == MessageType::Attribute {
            count += 1;
        }
        p = body_end;
    }
    Ok(count)
}

fn parse_compact_attr_name(
    region: &[u8],
    msg_start: usize,
    body: usize,
    body_end: usize,
) -> Result<String, Error> {
    if region[msg_start + 3] != 0 {
        return Err(Error::EditUnsupported(
            "a target group has a shared attribute message (not editable in place yet)",
        ));
    }
    crate::attribute::AttributeMessage::parse(&region[body..body_end], LENGTH_SIZE)
        .map(|attr| attr.name)
        .map_err(|_| Error::EditUnsupported("a target group has an unreadable attribute message"))
}

fn encode_attr_message(name: &str, value: &AttrValue) -> Result<Vec<u8>, Error> {
    if matches!(value, AttrValue::VarLenAsciiArray(_)) {
        return Err(Error::EditUnsupported(
            "variable-length group attributes cannot be edited in place yet",
        ));
    }
    let body = build_attr_message(name, value).serialize(LENGTH_SIZE);
    if body.len() > u16::MAX as usize {
        return Err(Error::EditUnsupported(
            "group attribute is too large to encode in place",
        ));
    }
    Ok(region_message(MessageType::Attribute, &body))
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
    #[expect(
        clippy::cast_possible_truncation,
        reason = "width was selected just above to be the smallest field that holds total"
    )]
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
#[expect(
    clippy::cast_possible_truncation,
    reason = "callers parse in-file sizes/offsets bounded by the in-memory image; downstream \
              slicing is length-checked, so a malformed oversized field errors rather than reads OOB"
)]
fn read_le(bytes: &[u8]) -> usize {
    let mut v = 0u64;
    for (i, &b) in bytes.iter().enumerate() {
        v |= (b as u64) << (8 * i);
    }
    v as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Collect the message types present in a chunk-0 region, in order.
    fn region_types(region: &[u8]) -> Vec<MessageType> {
        let mut out = Vec::new();
        let mut p = 0;
        while let Some((mt, _, end)) = next_message(region, p).unwrap() {
            out.push(mt);
            p = end;
        }
        out
    }

    #[test]
    fn fresh_group_region_pairs_link_info_with_group_info() {
        // A new-style group must carry both a Link Info and a Group Info message
        // (the C library requires the pair before it will insert a link).
        let types = region_types(&fresh_group_region());
        assert_eq!(types, vec![MessageType::LinkInfo, MessageType::GroupInfo]);
    }

    #[test]
    fn ensure_group_info_appends_when_missing() {
        // A region with a Link Info message but no Group Info message (how older
        // hdf5-pure releases wrote groups) gains exactly one Group Info message.
        let li_body = {
            let mut b = vec![0u8, 0];
            b.extend_from_slice(&u64::MAX.to_le_bytes());
            b.extend_from_slice(&u64::MAX.to_le_bytes());
            b
        };
        let mut region = region_message(MessageType::LinkInfo, &li_body);
        ensure_group_info(&mut region).unwrap();
        assert_eq!(
            region_types(&region),
            vec![MessageType::LinkInfo, MessageType::GroupInfo]
        );

        // The appended message decodes as a minimal Group Info body.
        let mut p = 0;
        while let Some((mt, body, end)) = next_message(&region, p).unwrap() {
            if mt == MessageType::GroupInfo {
                assert_eq!(&region[body..end], &GROUP_INFO_BODY);
            }
            p = end;
        }
    }

    #[test]
    fn ensure_group_info_is_idempotent() {
        // A region that already has a Group Info message is left untouched, so
        // re-editing a healed (or C-written) group does not duplicate it.
        let mut region = fresh_group_region();
        let before = region.clone();
        ensure_group_info(&mut region).unwrap();
        assert_eq!(region, before);
    }

    #[test]
    fn commit_clears_a_stale_consistency_flag() {
        // A clean in-place edit must leave the file properly closed for the C
        // library: the write/SWMR consistency flag a crashed SWMR writer left
        // behind is cleared rather than re-emitted (issue #73).
        use crate::writer::FileBuilder;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stale_flag.h5");

        let mut b = FileBuilder::new();
        b.create_dataset("d").with_i32_data(&[1, 2, 3]);
        b.write(&path).unwrap();

        // Simulate a crashed SWMR writer by stamping the on-disk write+SWMR flag
        // (0x05) into the superblock, recomputing its checksum.
        {
            let mut data = std::fs::read(&path).unwrap();
            let off = signature::find_signature(&data).unwrap();
            let mut sb = Superblock::parse(&data, off).unwrap();
            assert!(
                sb.version >= 2,
                "FileBuilder should emit a v2/v3 superblock"
            );
            sb.consistency_flags = 0x05;
            let bytes = sb.serialize();
            data[off..off + bytes.len()].copy_from_slice(&bytes);
            std::fs::write(&path, &data).unwrap();
            // Sanity: the stale flag is really set on disk now.
            assert_eq!(
                Superblock::parse(&data, off).unwrap().consistency_flags,
                0x05
            );
        }

        // A clean edit-and-commit cycle heals it.
        {
            let mut s = EditSession::open(&path).unwrap();
            s.create_dataset("e").with_i32_data(&[4, 5]);
            s.commit().unwrap();
        }

        let data = std::fs::read(&path).unwrap();
        let off = signature::find_signature(&data).unwrap();
        assert_eq!(
            Superblock::parse(&data, off).unwrap().consistency_flags,
            0,
            "commit must clear the stale consistency flag"
        );
    }
}
