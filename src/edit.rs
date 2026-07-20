//! In-place editing of an existing HDF5 file (issue #32, Group C).
//!
//! [`EditSession`] opens an existing file and adds objects, overwrites dataset
//! values, or edits compact group attributes **in place**:
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
//! datatypes, dataspaces, and attributes stay byte-exact. A chunked (and filtered)
//! dataset is copied with its chunk payloads and filter pipeline preserved
//! byte-for-byte, its index rebuilt at the new location. The same machinery,
//! [`EditSession::copy_from`], copies an object **across two open files** — the
//! source being a separate [`File`](crate::File) reader rather than the file being
//! edited. Because the copy is byte-for-byte, the cross-file path refuses anything
//! that embeds a source-file absolute address (variable-length or reference data,
//! a committed datatype), which an in-file copy keeps valid by sharing the source
//! file's heaps and objects.
//!
//! Value overwrite ([`EditSession::write_dataset`], the HDF5 `H5Dwrite`) replaces
//! an **existing** dataset's values. The replacement's datatype and shape must
//! match the on-disk dataset (an overwrite, not a reshape or retype); contiguous,
//! compact, and chunked (including filtered) datasets are all supported, the chunk
//! geometry and filter pipeline taken from the on-disk header. A same-length
//! contiguous overwrite is the cheapest edit there is — the new bytes go straight
//! into the existing data block, so no header is rewritten and the superblock root
//! is not flipped, and the synced data write is the commit's linearization point.
//! A chunked overwrite takes the same in-place path when every (re-encoded) chunk
//! still fits its slot — always for unfiltered storage (chunk sizes are fixed by
//! the unchanged shape), and for filtered storage when the re-encoded chunks match.
//! When a length differs (a resized contiguous block, a filtered chunk that no
//! longer fits, or a compact dataset) the dataset's storage is rebuilt and its
//! header relocated like an addition: the new data and a rewritten header are
//! appended, the data-layout message is repointed, the old storage is freed, and
//! the parent group's link is patched. A relocating overwrite of a dataset
//! reachable through more than one hard link is refused, since only the one named
//! link could be repointed at the moved header.
//!
//! # Scope
//!
//! It is deliberately strict: rather than silently produce a degraded file, it
//! refuses with [`Error::EditUnsupported`] any case it cannot reproduce
//! faithfully. Requirements:
//!
//! - The file uses 8-byte offsets/lengths. A **userblock** (non-zero base
//!   address, as every MATLAB v7.3 `.mat` file has) is supported: addresses are
//!   read and written relative to the base and the userblock bytes are preserved
//!   verbatim. Every edit works on a userblock file — value overwrites, additions
//!   of contiguous and chunked/filtered datasets, in-place and relocating
//!   overwrites of every layout (with the old storage reclaimed), object deletion
//!   (with base-aware subtree reclaim), in-file copy, cross-file copy into a
//!   userblock destination, group creation, compact attributes, and free-space
//!   reuse. The one userblock-specific limitation left is cross-file copy *from* a
//!   userblock source (the source must have base 0; see [`copy_from`](EditSession::copy_from)).
//!   Any superblock version (0–3) is accepted: a version 0/1
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
//!   one. A contiguous dataset may be empty (zero-element); chunking an empty
//!   shape is not supported. A provenance dataset (`with_provenance`) is
//!   supported, its attributes computed the same way the whole-file writer
//!   computes them. A contiguous dataset may carry a variable-length-string
//!   payload (`with_vlen_strings`) or per-element object-reference targets
//!   (`with_path_references`); chunking either is not supported. A
//!   path-resolved reference may target any object this commit is not itself
//!   still writing (an ancestor group, a same-depth sibling group ordered
//!   later in the same commit, a copy destination or its interior, a
//!   `write_dataset` target, or an object this commit deletes) — targeting
//!   one of those is refused, up front and before any byte of the commit is
//!   written, rather than resolved to a stale or wrong address; a path that
//!   resolves nowhere at all becomes an undefined reference, matching the
//!   whole-file writer. Every
//!   added dataset must have a fixed-size datatype, few enough attributes
//!   (compact or variable-length) to stay in compact storage. Group, root, and
//!   **dataset** attribute edits (`set_group_attr` / `set_dataset_attr`) may
//!   likewise be fixed-size or variable-length, under the same compact-storage
//!   limit; dense (fractal-heap) attribute storage is not editable. A dataset
//!   attribute edit relocates the dataset header and so requires a single hard
//!   link.
//! - A new group's parent must already exist or be created in the same session
//!   (each level created explicitly); intermediate groups are not auto-created.
//! - Rows can be appended to an existing chunked, unlimited, Extensible-Array
//!   dataset **immediately and in place** with `append_inplace` (amortized O(1),
//!   crash-atomic, no `commit`), interleaved with the staged edits above. A
//!   target the fast path cannot handle — a userblock or pre-v2 file, an
//!   unallocated index, a non-Extensible-Array or multi-hard-link dataset, a
//!   non-chunk-aligned filtered append — is refused with
//!   [`Error::AppendInPlaceUnsupported`]; use the staged `append_dataset` instead.
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
//! Reclaim is best-effort and conservative. Contiguous and chunked datasets
//! (chunk index plus chunk data) and whole group subtrees are reclaimed; a
//! deleted object whose blocks cannot be enumerated exhaustively —
//! variable-length global-heap storage, dense attribute/link heaps, a
//! non–version-2 header, a version 2 B-tree chunk index — is left as dead bytes
//! rather than risk freeing a region that is still in use; under-reclaiming only
//! wastes space, while over-reclaiming would corrupt.
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

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::checksum::jenkins_lookup3;
use crate::chunk_index_inplace::{InPlaceBytes, Located, apply_ea_append, plan_ea_append};
use crate::chunked_read::{chunk_index_spans_buffered, enumerate_chunks_buffered, plan_dense_grid};
use crate::chunked_write::{
    ChunkMeta, ChunkOptions, ChunkProvider, WrittenChunk, build_chunked_data_at_ext,
    build_extensible_array_at, emit_chunked_data_verbatim, plan_chunked_data_verbatim,
    serialize_v4_extensible_array, split_into_chunks,
};
use crate::convert::TryToUsize;
use crate::data_layout::DataLayout;
use crate::dataspace::{Dataspace, DataspaceType};
use crate::datatype::{Datatype, DatatypeByteOrder};
use crate::error::{Error, FormatError};
use crate::extensible_array::ExtensibleArrayHeader;
use crate::file_lock::{self, FileLocking};
use crate::file_space_info::{FileSpaceInfo, FileSpaceStrategy};
use crate::file_writer::{
    LENGTH_SIZE, OFFSET_SIZE, build_chunked_dataset_oh, build_dataset_oh, make_link,
};
use crate::filter_pipeline::{
    FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_SCALEOFFSET, FILTER_SHUFFLE, FilterPipeline,
};
use crate::filters::{ChunkContext, compress_chunk, decompress_chunk};
use crate::free_space::FreeList;
use crate::free_space_manager::{self, FreeSection, FsmHeader, fshd_len, serialize_file_fsm};
use crate::group_v2::resolve_group_entries;
use crate::link_message::{LinkMessage, LinkTarget};
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::signature;
use crate::superblock::Superblock;
use crate::type_builders::{
    AttrValue, DatasetBuilder, ObjectRefTarget, VlStringStaging, build_attr_message,
    build_global_heap_collection, make_f32_type, make_f64_type, make_i8_type, make_i16_type,
    make_i32_type, make_i64_type, make_u8_type, make_u16_type, make_u32_type, make_u64_type,
    patch_vl_refs, patch_vl_refs_masked,
};

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

/// Upper bound on the number of object headers walked when counting hard links
/// across the file (issue #77 / reclaim safety). Far beyond any real file; a
/// graph larger than this aborts the count, and the commit then leaves deleted
/// objects unreclaimed (a safe leak) rather than risk an unbounded walk.
const MAX_LINK_GRAPH_NODES: u32 = 1 << 24;

/// Maximum number of object-header chunks to follow when gathering a header that
/// spans continuation blocks, guarding against a cyclic continuation chain.
/// Matches the reader's continuation-depth cap.
const MAX_OH_CHUNKS: usize = 256;

/// A path identified by its components (no leading/trailing empties); the root
/// group is the empty vector.
type PathKey = Vec<String>;

/// Variable-length group/root attributes staged by [`apply_group_attr_ops`],
/// each an (attribute message still carrying a placeholder heap address, its
/// global heap collection bytes) pair, resolved in the apply loop.
type PendingVlAttrs = Vec<(crate::attribute::AttributeMessage, Vec<u8>)>;

/// Accumulates elements to append to an existing chunked, unlimited dataset via
/// [`EditSession::append_dataset`], in call order along the dataset's first
/// (axis-0) dimension.
///
/// It mirrors [`DatasetBuilder`]'s typed/generic vocabulary. Repeated typed or
/// [`append_raw`](Self::append_raw) calls concatenate; each typed method also
/// records the element datatype it implies, which `commit` checks against the
/// dataset's on-disk datatype (a mismatch — including a mix of element types in
/// one builder — is refused with [`Error::AppendUnsupported`], never written as
/// garbage).
pub struct AppendBuilder {
    /// Accumulated little-endian element bytes to append, in call order.
    raw: Vec<u8>,
    /// The element datatype implied by the typed `append_*` calls, if any were
    /// used. `None` when only [`append_raw`](Self::append_raw) was called (a raw
    /// append is checked structurally — element-size alignment and little-endian
    /// on-disk order — rather than by datatype equality).
    elem_dt: Option<Datatype>,
    /// Set when two typed calls implied different element datatypes; `commit`
    /// refuses such a builder rather than write a mix of encodings.
    dt_conflict: bool,
}

impl AppendBuilder {
    pub(crate) fn new() -> Self {
        Self {
            raw: Vec::new(),
            elem_dt: None,
            dt_conflict: false,
        }
    }

    /// Accumulated little-endian element bytes (for the general append writer,
    /// which reuses this builder to gather typed/generic appends).
    pub(crate) fn raw(&self) -> &[u8] {
        &self.raw
    }

    /// The element datatype implied by typed appends, if any.
    pub(crate) fn elem_dt(&self) -> Option<&Datatype> {
        self.elem_dt.as_ref()
    }

    /// Whether two typed appends implied conflicting element datatypes.
    pub(crate) fn dt_conflict(&self) -> bool {
        self.dt_conflict
    }

    /// Record the datatype a typed append implies, flagging a conflict if an
    /// earlier typed call implied a different one.
    fn set_dt(&mut self, dt: Datatype) {
        match &self.elem_dt {
            Some(prev) if *prev != dt => self.dt_conflict = true,
            Some(_) => {}
            None => self.elem_dt = Some(dt),
        }
    }

    /// Append already-little-endian element bytes verbatim. The concatenated
    /// length must be a whole multiple of the dataset's on-disk element size, and
    /// the dataset's element datatype must be little-endian; no datatype is
    /// otherwise inferred. Prefer the typed methods when the element type is known.
    pub fn append_raw(&mut self, bytes: &[u8]) -> &mut Self {
        self.raw.extend_from_slice(bytes);
        self
    }

    /// Generic append of a flat slice of any supported scalar type — the
    /// counterpart of [`DatasetBuilder::with_data`](crate::DatasetBuilder::with_data).
    pub fn append<T: crate::element::H5Element>(&mut self, data: &[T]) -> &mut Self {
        T::append_into(self, data);
        self
    }
}

/// Generate the typed `append_*` methods: serialize each value little-endian and
/// record the implied element datatype.
macro_rules! append_typed {
    ($($method:ident, $ty:ty, $make:ident;)*) => {
        impl AppendBuilder {
            $(
                #[doc = concat!("Append `", stringify!($ty), "` values to the dataset.")]
                pub fn $method(&mut self, data: &[$ty]) -> &mut Self {
                    self.set_dt($make());
                    for &v in data {
                        self.raw.extend_from_slice(&v.to_le_bytes());
                    }
                    self
                }
            )*
        }
    };
}

append_typed! {
    append_f64, f64, make_f64_type;
    append_f32, f32, make_f32_type;
    append_i8, i8, make_i8_type;
    append_i16, i16, make_i16_type;
    append_i32, i32, make_i32_type;
    append_i64, i64, make_i64_type;
    append_u8, u8, make_u8_type;
    append_u16, u16, make_u16_type;
    append_u32, u32, make_u32_type;
    append_u64, u64, make_u64_type;
}

/// An open HDF5 file being edited in place.
///
/// Mirror the file in memory and keep a writable handle; every mutation is
/// applied to both so the on-disk file stays consistent.
///
/// The session has **two commit models** that compose on one open file:
///
/// - **Staged** edits are batched and applied together by [`commit`](Self::commit):
///   additions ([`create_dataset`](Self::create_dataset) /
///   [`create_group`](Self::create_group)), value overwrites
///   ([`write_dataset`](Self::write_dataset)), staged appends
///   ([`append_dataset`](Self::append_dataset)), group and dataset attribute edits
///   ([`set_group_attr`](Self::set_group_attr) / [`set_dataset_attr`](Self::set_dataset_attr)
///   and their `remove_*` counterparts), object [`copy`](Self::copy), and
///   [`delete`](Self::delete). Dropping the session discards uncommitted staged
///   edits ([`has_staged_edits`](Self::has_staged_edits) reports whether any remain).
/// - **Immediate** in-place row appends ([`append_inplace`](Self::append_inplace))
///   are applied and made durable the moment they are called — amortized O(1),
///   crash-atomic, needing no `commit` — and can be freely interleaved with staged
///   edits on the same session, so a high-frequency append loop and occasional tree
///   edits share one open file with no reopening between them.
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
    /// Parsed superblock. On-disk addresses are stored relative to `base_address`;
    /// the in-memory `root_group_address` is normalized to an absolute file offset
    /// on open and converted back to a base-relative address when serialized on
    /// commit. `base_address` equals the superblock's file location (`sb_sig_off`):
    /// 0 for a plain file, the userblock size for one with a userblock.
    superblock: Superblock,
    /// Datasets staged by `create_dataset`, as (parent group path, builder).
    pending_datasets: Vec<(PathKey, DatasetBuilder)>,
    /// Value overwrites staged by `write_dataset`, as (full dataset path,
    /// builder). Each replaces an existing dataset's values in place; the new
    /// datatype and shape must match the on-disk ones byte-exactly (this is a
    /// value overwrite, not a reshape/retype). Applied on the next `commit`.
    pending_writes: Vec<(PathKey, DatasetBuilder)>,
    /// Appends staged by `append_dataset`, as (full dataset path, builder). Each
    /// grows an existing chunked, unlimited, Extensible-Array-indexed dataset
    /// along axis 0 by keeping its existing chunk data in place and rebuilding the
    /// index over the kept plus newly-appended (and any rewritten trailing) chunks.
    /// Applied on the next `commit`.
    pending_appends: Vec<(PathKey, AppendBuilder)>,
    /// New groups staged by `create_group`, as full paths.
    pending_groups: Vec<PathKey>,
    /// Group attribute edits staged as (group path, operation). The path may be
    /// a group created in this same session.
    pending_group_attrs: Vec<(PathKey, AttrOp)>,
    /// Dataset attribute edits staged as (full dataset path, operation), applied
    /// on the next `commit`. Each relocates the dataset's object header (like a
    /// relocating overwrite): the header is rebuilt with the compact-attribute
    /// change, its single naming link is patched, and the old header freed — the
    /// dataset's data and chunk index stay in place. The target must be an existing,
    /// single-hard-link dataset using compact (not dense fractal-heap) attributes.
    pending_dataset_attrs: Vec<(PathKey, AttrOp)>,
    /// Links staged for removal by `delete`, as full paths.
    pending_deletes: Vec<PathKey>,
    /// Object copies staged by `copy`, as (source path, destination full path).
    pending_copies: Vec<(PathKey, PathKey)>,
    /// Cross-file object copies staged by `copy_from`, as (destination full path,
    /// the source subtree already read out of the other file). The subtree is read
    /// — and foreign-address-screened — eagerly in `copy_from` (the source file is
    /// borrowed only for that call), then linked in at the next `commit`.
    pending_cross_copies: Vec<(PathKey, CopyTree)>,
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
    /// Per-dataset geometry cache for the immediate O(1) in-place append
    /// ([`append_inplace`](Self::append_inplace)), keyed by the dataset's resolved
    /// **object-header address** (not its path, so two hard links to one dataset
    /// share one entry). Populated on the first append to a dataset and maintained
    /// across appends; cleared wholesale at the entry of every non-trivial
    /// [`commit`](Self::commit), since a commit can relocate a cached header or
    /// free the region it points into (see `commit`).
    located: HashMap<u64, LocatedState>,
    /// True when this session was opened for SWMR writing
    /// ([`open_swmr_writer`](Self::open_swmr_writer)): the append engine then
    /// enforces the SWMR subset (unfiltered, chunk-aligned) so a concurrent
    /// reader never observes a torn view. `false` for an ordinary edit session.
    swmr_mode: bool,
}

/// Superblock consistency-flag bits raised while a SWMR writer is active: bit 0
/// (write access) | bit 2 (SWMR write access). Cleared on a clean close. Matches
/// the reference C library, h5py, and [`crate::SwmrWriter`].
const SWMR_WRITE_FLAGS: u32 = 0x05;

/// A dataset located once for [`EditSession::append_inplace`], then maintained
/// across appends. Mirrors the append writer's per-dataset state.
struct LocatedState {
    loc: Located,
    /// The dataset's on-disk element datatype (for the append type check and the
    /// filter chunk context).
    datatype: Datatype,
    /// Spatial (rank-length) chunk dimensions in elements: `[chunk_elems]`.
    spatial: Vec<u64>,
    /// Bytes per element (datatype size).
    element_size: usize,
    /// The re-encodable filter pipeline, when the dataset is filtered.
    pipeline: Option<FilterPipeline>,
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

/// A snapshot of an [`EditSession`]'s live space usage (issue #150).
///
/// This is the mutating-session counterpart of the read-only accounting on
/// [`File`](crate::File) ([`file_size`](crate::File::file_size) and
/// [`persisted_free_space`](crate::File::persisted_free_space)): it describes the
/// file *as the session currently holds it*, taken atomically at the moment of
/// the [`space_accounting`](EditSession::space_accounting) call.
///
/// It reflects the committed file plus any immediate in-place appends
/// ([`append_inplace`](EditSession::append_inplace)), but **not** edits still
/// staged for the next [`commit`](EditSession::commit) — `create_group`,
/// `create_dataset`, `write_dataset`, `append_dataset`, `delete`, `copy`,
/// `copy_from`, and attribute edits change these figures only when they are
/// applied at commit. Use [`has_staged_edits`](EditSession::has_staged_edits) to
/// tell whether such pending work exists.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct SpaceAccounting {
    /// The session's current logical size in bytes: the byte length of the file
    /// as the session holds it. It equals what
    /// [`File::file_size`](crate::File::file_size) reports for the file on disk
    /// right now (the HDF5 `H5Fget_filesize` value), because the session keeps its
    /// in-memory mirror byte-for-byte identical to the file — every committed
    /// write and every immediate in-place append
    /// ([`append_inplace`](EditSession::append_inplace)) updates both together.
    ///
    /// It is not monotonic: [`commit`](EditSession::commit) can reclaim trailing
    /// free space and *shrink* the file. It can also exceed the superblock's
    /// recorded end-of-file address when the file was opened carrying unaccounted
    /// trailing bytes (the same slack [`File::file_size`](crate::File::file_size)
    /// surfaces), since opening does not rewrite that address.
    pub logical_size: u64,
    /// Total reusable free bytes the next allocation or [`commit`](EditSession::commit)
    /// can draw from before the file has to grow — the summed length of
    /// [`reusable_free_space`](Self::reusable_free_space).
    ///
    /// Counts holes left inside [`logical_size`](Self::logical_size) by this
    /// session's earlier commits (superseded object headers, the blocks of
    /// deleted objects) and, for a file created with
    /// `H5Pset_file_space_strategy(persist = true)` and no userblock, the regions
    /// seeded from the on-disk free-space managers when the session was opened (so
    /// reuse spans sessions). A fresh non-persisting session reports `0` even if
    /// the file contains holes left by other tools — those are never tracked. It
    /// is neither a lower bound on the next write's growth nor a promise of
    /// shrinkage: a region counted here may be truncated away at commit — rather
    /// than reused — if adjacent space is later freed and the coalesced run
    /// reaches end-of-file.
    pub reusable_free_bytes: u64,
    /// The reusable free regions as `(offset, length)` pairs, sorted ascending by
    /// offset and fully coalesced (no two regions touch or overlap).
    ///
    /// The offsets are **absolute** file offsets (from byte 0, including any
    /// userblock prefix), matching [`logical_size`](Self::logical_size). This
    /// differs from [`File::persisted_free_space`](crate::File::persisted_free_space),
    /// whose pairs are relative to the superblock base address; the two coincide
    /// for a file with no userblock (base address 0), which is the only kind whose
    /// persisted free space a session seeds. Empty when nothing is reusable.
    pub reusable_free_space: Vec<(u64, u64)>,
}

impl EditSession {
    /// Open an existing HDF5 file for in-place editing.
    ///
    /// Reads the file into memory and retains a read/write handle. Takes an
    /// exclusive OS advisory lock so the file cannot be opened concurrently by
    /// another writer or reader; the lock is released automatically when the
    /// session is dropped or the process exits (including on a crash). Fails with
    /// [`Error::FileLocked`] if the file is already locked, or
    /// [`Error::EditUnsupported`] if the file is not a supported target; its
    /// documentation enumerates the exact requirements. To control or disable
    /// locking, use [`open_with_locking`](Self::open_with_locking) or set
    /// `HDF5_USE_FILE_LOCKING=FALSE`.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        Self::open_with_locking(path, FileLocking::Enabled)
    }

    /// Open an existing HDF5 file for in-place editing, choosing the file-locking
    /// policy explicitly. See [`open`](Self::open) and [`FileLocking`].
    pub fn open_with_locking<P: AsRef<Path>>(path: P, locking: FileLocking) -> Result<Self, Error> {
        Self::open_inner(path.as_ref(), Some(locking))
    }

    /// Open an existing file for SWMR (single-writer/multiple-reader) writing:
    /// take **no** OS lock at all and raise the superblock's SWMR-write
    /// consistency flag. Backs [`File::open_swmr_writer`](crate::File::open_swmr_writer).
    ///
    /// The no-lock is unconditional — `lock = None` never reaches
    /// `acquire_exclusive`, so `HDF5_USE_FILE_LOCKING` cannot reintroduce a lock
    /// that would block the concurrent readers SWMR exists to permit (fatally so
    /// on Windows, where OS locks are mandatory). Requires a latest-format
    /// (version-2/3 superblock) file with no userblock and no persisted
    /// free-space, so the superblock can be rewritten in place.
    pub(crate) fn open_swmr_writer<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let mut session = Self::open_inner(path.as_ref(), None)?;
        if session.superblock.version < 2
            || session.superblock.base_address != 0
            || session.persist.is_some()
        {
            return Err(Error::SwmrAppendUnsupported(
                "SWMR writing requires a latest-format file (v2/v3 superblock) with no userblock \
                 and no persisted free-space",
            ));
        }
        session.swmr_mode = true;
        session.set_consistency_flags(SWMR_WRITE_FLAGS)?;
        Ok(session)
    }

    /// Set the superblock's consistency flags in the mirror and on disk, then
    /// flush. Used to raise the SWMR-write flag on open and clear it on close.
    /// Requires a base-0, version-2/3 file (checked by `open_swmr_writer`), since
    /// [`Superblock::serialize`] emits the v2/v3 layout at the base address.
    pub(crate) fn set_consistency_flags(&mut self, flags: u32) -> Result<(), Error> {
        self.superblock.consistency_flags = flags;
        let bytes = self.superblock.serialize();
        self.data[self.sb_sig_off..self.sb_sig_off + bytes.len()].copy_from_slice(&bytes);
        self.handle
            .seek(SeekFrom::Start(self.sb_sig_off as u64))
            .map_err(Error::Io)?;
        self.handle.write_all(&bytes).map_err(Error::Io)?;
        self.handle.flush().map_err(Error::Io)?;
        self.handle.sync_data().map_err(Error::Io)?;
        Ok(())
    }

    /// Shared open path. `lock = Some(policy)` acquires an exclusive OS lock under
    /// that policy (the ordinary read-write session); `lock = None` takes no lock
    /// at all (the SWMR writer — see [`open_swmr_writer`](Self::open_swmr_writer)).
    fn open_inner(path: &Path, lock: Option<FileLocking>) -> Result<Self, Error> {
        let mut handle = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(Error::Io)?;
        // Acquire the exclusive lock before reading or mutating; the retained
        // `handle` holds it for the session's life. A `None` policy (SWMR) never
        // reaches `acquire_exclusive`, so no lock is ever taken.
        if let Some(policy) = lock {
            file_lock::acquire_exclusive(&handle, policy, path)?;
        }
        let mut data = Vec::new();
        handle.read_to_end(&mut data).map_err(Error::Io)?;

        let sb_sig_off = signature::find_signature(&data)?;
        let mut superblock = Superblock::parse(&data, sb_sig_off)?;

        if superblock.version > 3 {
            return Err(Error::EditUnsupported("unsupported superblock version"));
        }
        if superblock.offset_size != OFFSET_SIZE || superblock.length_size != LENGTH_SIZE {
            return Err(Error::EditUnsupported(
                "only 8-byte offsets and lengths are supported for in-place editing",
            ));
        }
        // A userblock shifts the whole HDF5 image forward by `base_address`: the
        // superblock sits at the base address and every stored address is relative
        // to it (the end-of-file address is the sole absolute field). The editor
        // supports this by reading at `stored + base` and writing back
        // `file_offset - base`. Only the canonical layout — superblock located
        // exactly at the base address (e.g. a MATLAB v7.3 `.mat` file's 512-byte
        // userblock) — is accepted; a base address that disagrees with the
        // superblock's location is a relocated or malformed file we will not rewrite.
        if superblock.base_address != sb_sig_off as u64 {
            return Err(Error::EditUnsupported(
                "a file whose superblock is not located at its base address is not editable in place",
            ));
        }
        // Normalize the root group address to an absolute file offset, exactly as
        // the reader does (`reader::parse_superblock`), so `resolve_path_any` and
        // the link-graph walk index `self.data` correctly. It is converted back to a
        // stored (base-relative) address only when the superblock is serialized on
        // commit.
        superblock.root_group_address = superblock
            .root_group_address
            .checked_add(superblock.base_address)
            .ok_or(FormatError::OffsetOverflow {
                offset: superblock.root_group_address,
                length: superblock.base_address,
            })?;

        let mut session = Self {
            handle,
            data,
            sb_sig_off,
            superblock,
            pending_datasets: Vec::new(),
            pending_writes: Vec::new(),
            pending_appends: Vec::new(),
            pending_groups: Vec::new(),
            pending_group_attrs: Vec::new(),
            pending_dataset_attrs: Vec::new(),
            pending_deletes: Vec::new(),
            pending_copies: Vec::new(),
            pending_cross_copies: Vec::new(),
            free: FreeList::new(),
            persist: None,
            located: HashMap::new(),
            swmr_mode: false,
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
        // Free-space reuse and persistence are not yet base-address aware: the
        // persisted section addresses (and the extension/manager block walk below)
        // are read as absolute, so on a userblock file they would seed `self.free`
        // with wrong regions that `alloc_or_append` could later hand out into live
        // data. Leave persistence off for such a file — the on-disk managers stay
        // untouched and valid, this session simply appends rather than reusing.
        if self.superblock.base_address != 0 {
            return;
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
    /// `with_scale_offset`, `with_zfp`) and/or extensible (`with_maxshape`). An
    /// empty (zero-element) contiguous dataset is supported (chunking one is
    /// not), a provenance dataset (`with_provenance`) is supported, and a
    /// contiguous dataset may carry variable-length attributes, a
    /// variable-length-string payload (`with_vlen_strings`), or path-resolved
    /// object-reference elements (`with_path_references`; chunking any of
    /// these is not supported, and dense attributes remain unsupported).
    pub fn create_dataset(&mut self, path: &str) -> &mut DatasetBuilder {
        let mut comps = split_path(path);
        let leaf = comps.pop().unwrap_or_default();
        self.pending_datasets
            .push((comps, DatasetBuilder::new(&leaf)));
        &mut self.pending_datasets.last_mut().unwrap().1
    }

    /// Stage an in-place overwrite of an **existing** dataset's values (the HDF5
    /// `H5Dwrite` whole-dataset write), applied on the next
    /// [`commit`](Self::commit). `path` is the full path of a dataset that must
    /// already exist; the returned [`DatasetBuilder`] — the same builder used by
    /// [`create_dataset`](Self::create_dataset) — supplies the replacement data.
    ///
    /// This is a *value* overwrite, not a reshape or retype: the new data's
    /// datatype and shape must match the on-disk dataset's exactly (byte-for-byte
    /// after serialization, so endianness and compound layout must agree), or
    /// `commit` reports [`Error::EditUnsupported`]. Contiguous, compact, and
    /// chunked (including filtered) datasets are all supported; the dataset's
    /// existing chunk geometry, filter pipeline, and chunk index are taken from the
    /// on-disk header (a builder that itself requests chunking/filtering is refused
    /// as "not a value overwrite"). A chunk index this engine cannot enumerate (a
    /// version-2 B-tree) is refused. Partial / sub-region writes are out of scope —
    /// the whole dataset is replaced.
    ///
    /// When the new data is the same length as the existing contiguous data block
    /// (the common case), the bytes are written straight into that block: no
    /// object header is rewritten and the superblock root is not flipped, so the
    /// commit's linearization point is the synced data write itself. A chunked
    /// dataset is handled the same way when every (re-encoded) chunk is the same
    /// byte length as the slot it replaces — an unfiltered overwrite (chunk sizes
    /// are fixed by the unchanged shape) or a filtered one whose re-encoded chunks
    /// match — so it too writes straight into the existing chunk slots. When the
    /// length differs (a resized contiguous block, or a filtered chunk that no
    /// longer fits), the dataset's storage is rebuilt at end-of-file (or in
    /// reusable freed space), the old extent is freed, the data-layout message is
    /// repointed, the object header is rewritten, and the parent group's link is
    /// patched — exactly like an addition relocates the path up to the root. A
    /// relocating overwrite moves the object header, so it is refused unless the
    /// dataset has a single hard link.
    pub fn write_dataset(&mut self, path: &str) -> &mut DatasetBuilder {
        let comps = split_path(path);
        let leaf = comps.last().cloned().unwrap_or_default();
        self.pending_writes
            .push((comps, DatasetBuilder::new(&leaf)));
        &mut self.pending_writes.last_mut().unwrap().1
    }

    /// Stage an append of new elements to an **existing** chunked, unlimited
    /// dataset, applied on the next [`commit`](Self::commit). `path` names a
    /// dataset that must already exist; the returned [`AppendBuilder`] supplies
    /// the elements to add via its typed / generic / raw `append_*` methods.
    ///
    /// Unlike [`write_dataset`](Self::write_dataset) (a value overwrite that
    /// forbids any shape change) this **grows** the dataset along its first
    /// (axis-0) dimension. It works on **filtered** datasets: the appended chunks
    /// are compressed through the dataset's own on-disk filter pipeline
    /// (deflate / shuffle / fletcher32 / scale-offset, and ZFP with the `zfp`
    /// feature), and the pipeline, datatype, fill value, and attributes are
    /// preserved verbatim. Appends of any length are supported — when the
    /// dataset's current length is not a whole multiple of the chunk length, the
    /// single trailing partial chunk is read, extended, and re-encoded; every
    /// other existing chunk is carried by metadata alone, so the existing data is
    /// not rewritten and the file does not grow by the whole dataset per append.
    ///
    /// This does **not** use SWMR and sets no consistency flag. Like every other
    /// [`EditSession`] edit it commits by appending the new chunks and a rebuilt
    /// index at end-of-file and repointing the superblock last (under the
    /// session's exclusive lock), so a crash leaves either the original dataset or
    /// the fully-grown one, never a torn state.
    ///
    /// The first release supports the Extensible-Array chunk index (the index the
    /// reference C library and h5py select for a single unlimited dimension under
    /// the latest format, and the one this crate writes for every unlimited
    /// dataset), rank-1 datasets, and datasets with a single hard link. A dataset
    /// that is not chunked, not unlimited along axis 0, not Extensible-Array
    /// indexed, higher than rank 1, uses a filter this engine cannot re-encode,
    /// has a sparse chunk grid, or (for [`append_raw`](AppendBuilder::append_raw))
    /// has a big-endian element datatype is refused with
    /// [`Error::AppendUnsupported`]. Use [`Dataset::is_chunked`](crate::Dataset::is_chunked),
    /// [`maxshape`](crate::Dataset::maxshape), and [`filters`](crate::Dataset::filters)
    /// to check eligibility up front.
    pub fn append_dataset(&mut self, path: &str) -> &mut AppendBuilder {
        self.pending_appends
            .push((split_path(path), AppendBuilder::new()));
        &mut self.pending_appends.last_mut().unwrap().1
    }

    /// Immediately append rows to an **existing** chunked, unlimited,
    /// Extensible-Array-indexed dataset **in place**, at amortized O(1) index cost
    /// — the throughput-oriented, self-committing counterpart to the staged
    /// [`append_dataset`](Self::append_dataset).
    ///
    /// Unlike every other [`EditSession`] edit, an in-place append is **not**
    /// staged: it is applied and made durable before it returns (writes ordered
    /// child-before-parent with `fsync` barriers, the dataspace dimension
    /// published last as the single commit point), exactly like
    /// [`AppendWriter`](crate::AppendWriter). It needs no [`commit`](Self::commit),
    /// and it composes with staged tree edits on the same session: append rows,
    /// stage `create_group` / `create_dataset` / attribute / `delete` edits,
    /// `commit` them, and keep appending — all without reopening the file between
    /// the fast appends and the tree edits.
    ///
    /// # Length rules and crash safety
    ///
    /// Identical to [`AppendWriter`](crate::AppendWriter): an **unfiltered** dataset
    /// accepts any-length appends (a partial trailing chunk is rewritten and its
    /// single-address index element repointed with one atomic write); a
    /// **filtered** dataset accepts whole-chunk appends only (its multi-field index
    /// element cannot be repointed power-loss-atomically). Every append is
    /// crash-atomic — a crash between appends leaves the previous length or the new
    /// one, never a torn view.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendInPlaceUnsupported`] — deliberately distinct from
    /// [`Error::AppendUnsupported`] so a caller can catch it and fall back to the
    /// staged [`append_dataset`](Self::append_dataset) — when the file has a
    /// userblock or a pre-v2 superblock, the dataset's Extensible-Array index is
    /// not yet allocated, the dataset is not rank-1 / unlimited / Extensible-Array
    /// indexed, a filtered append is not chunk-aligned, the append datatype does
    /// not match the dataset, or the target path (or an ancestor) has a staged edit
    /// still pending in this session (commit those first). A dataset reachable
    /// through more than one hard link is handled through the staged
    /// [`append_dataset`](Self::append_dataset), not here.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use hdf5_pure::EditSession;
    ///
    /// let mut session = EditSession::open("log.h5")?;
    /// session.append_inplace_i32("samples", &[8, 9, 10, 11])?; // immediate + durable
    /// session.create_group("run2"); // staged
    /// session.append_inplace_i32("samples", &[12, 13])?;
    /// session.commit()?; // applies the staged group; appends already durable
    /// # Ok::<(), hdf5_pure::Error>(())
    /// ```
    pub fn append_inplace<T: crate::element::H5Element>(
        &mut self,
        dataset: &str,
        data: &[T],
    ) -> Result<(), Error> {
        let mut b = AppendBuilder::new();
        b.append(data);
        self.append_inplace_gathered(dataset, &b, 4)
    }

    /// Append raw little-endian element bytes to `dataset` in place. See
    /// [`append_inplace`](Self::append_inplace) for the full contract; the byte
    /// length must be a whole multiple of the dataset's on-disk element size and
    /// the element datatype must be little-endian (and neither variable-length nor
    /// a reference). Prefer the typed methods when the element type is known.
    pub fn append_inplace_raw(&mut self, dataset: &str, bytes: &[u8]) -> Result<(), Error> {
        let mut b = AppendBuilder::new();
        b.append_raw(bytes);
        self.append_inplace_gathered(dataset, &b, 4)
    }

    /// Whether any staged tree edit is still uncommitted. In-place appends
    /// ([`append_inplace`](Self::append_inplace)) are applied immediately and are
    /// never staged, so they never affect this; it reflects only edits awaiting
    /// [`commit`](Self::commit) — `create_group`, `create_dataset`,
    /// `write_dataset`, `append_dataset`, group and dataset attribute edits,
    /// `delete`, `copy`, and `copy_from`. Dropping the session silently discards
    /// any staged edits.
    pub fn has_staged_edits(&self) -> bool {
        !self.pending_datasets.is_empty()
            || !self.pending_writes.is_empty()
            || !self.pending_appends.is_empty()
            || !self.pending_groups.is_empty()
            || !self.pending_group_attrs.is_empty()
            || !self.pending_dataset_attrs.is_empty()
            || !self.pending_deletes.is_empty()
            || !self.pending_copies.is_empty()
            || !self.pending_cross_copies.is_empty()
    }

    /// The session's whole-file in-memory mirror, reflecting committed state plus
    /// immediate in-place appends (not edits still staged for `commit`). Used by
    /// the owned read-write [`File`](crate::File) to serve reads on a mirror
    /// backend.
    pub(crate) fn mirror_bytes(&self) -> &[u8] {
        &self.data
    }

    /// A snapshot of this session's live space usage — the current file size and
    /// the free space it can reuse — as a [`SpaceAccounting`].
    ///
    /// This is the mutating-session analogue of the read-only accounting on
    /// [`File`](crate::File): it answers "how big is the file right now, and how
    /// much space can be reused before it must grow?" from the session's own live
    /// state. The snapshot reflects the committed file plus any immediate in-place
    /// appends ([`append_inplace`](Self::append_inplace)) but excludes edits still
    /// staged for the next [`commit`](Self::commit); see [`SpaceAccounting`] for
    /// the field-by-field semantics and [`has_staged_edits`](Self::has_staged_edits)
    /// for detecting pending work.
    ///
    /// ```no_run
    /// use hdf5_pure::EditSession;
    ///
    /// let session = EditSession::open("existing.h5")?;
    /// let acct = session.space_accounting();
    /// println!(
    ///     "{} bytes on disk, {} reusable in {} free region(s)",
    ///     acct.logical_size,
    ///     acct.reusable_free_bytes,
    ///     acct.reusable_free_space.len(),
    /// );
    /// # Ok::<(), hdf5_pure::Error>(())
    /// ```
    #[must_use]
    pub fn space_accounting(&self) -> SpaceAccounting {
        let reusable_free_space = self.free.sections();
        let reusable_free_bytes = reusable_free_space.iter().map(|(_, len)| len).sum();
        SpaceAccounting {
            logical_size: self.data.len() as u64,
            reusable_free_bytes,
            reusable_free_space,
        }
    }

    /// Apply a gathered in-place append (typed / generic / raw bytes) to `dataset`,
    /// immediately and crash-atomically, driving the shared Extensible-Array engine
    /// against the session's own mirror through an [`EditMirror`] adapter. Runs only
    /// the first `max_phase` durability phases; production callers pass 4, the
    /// crash-consistency tests stop at a boundary to simulate a crash.
    fn append_inplace_gathered(
        &mut self,
        dataset: &str,
        b: &AppendBuilder,
        max_phase: u8,
    ) -> Result<(), Error> {
        if b.dt_conflict() {
            return Err(Error::AppendInPlaceUnsupported(
                "append mixes element types in one call; use one element type per append",
            ));
        }
        // The fast in-place append is only sound on a base-0 latest-format file:
        // the slot math assumes absolute addresses and the superblock is patched in
        // place per call. A userblock or pre-v2 file falls back to the staged
        // `append_dataset`, which rebuilds the index and repoints the superblock
        // last.
        if self.superblock.base_address != 0 {
            return Err(Error::AppendInPlaceUnsupported(
                "in-place append does not support a file with a userblock (non-zero base \
                 address); use EditSession::append_dataset",
            ));
        }
        if self.superblock.version < 2 {
            return Err(Error::AppendInPlaceUnsupported(
                "in-place append requires a latest-format file (v2/v3 superblock); use \
                 EditSession::append_dataset",
            ));
        }
        // A file that persists its free space keeps on-disk free-space managers (and,
        // for a paged strategy, a page-aligned end-of-file) that only a staged
        // `commit` rewrites consistently. An immediate EOF append bypasses that
        // rewrite, so fall back to the staged path, which rebuilds the managers and
        // repoints the superblock last.
        if self.persist.is_some() {
            return Err(Error::AppendInPlaceUnsupported(
                "in-place append is not supported on a file that persists its free space \
                 (H5Pset_file_space_strategy persist=true); use EditSession::append_dataset",
            ));
        }

        // Refuse an append against a dataset (or a subtree) that a still-staged edit
        // in this same session will relocate, replace, or delete — which would
        // strand the durably-appended rows or plan against a header the commit
        // moves. The caller must commit those edits first.
        let target = split_path(dataset);
        if self.append_conflicts_with_pending(&target) {
            return Err(Error::AppendInPlaceUnsupported(
                "the dataset or an ancestor has a staged edit pending in this session; commit \
                 the staged edits before appending in place, or use EditSession::append_dataset",
            ));
        }

        // Resolve the dataset's object-header address — the geometry cache key.
        // base == 0 here, so the resolved address is absolute; two hard links to
        // one dataset share the one entry.
        let oh_addr = crate::group_v2::resolve_path_any(&self.data, &self.superblock, dataset)
            .map_err(|_| {
                Error::AppendInPlaceUnsupported("nothing to append to at the given path")
            })?;

        // Locate the dataset on the first append (cache miss) against the session's
        // own borrowed mirror — no second lock, no second mirror, no re-read.
        if !self.located.contains_key(&oh_addr) {
            let mirror = EditMirror {
                handle: &mut self.handle,
                data: &mut self.data,
                superblock: &mut self.superblock,
                sb_sig_off: self.sb_sig_off,
            };
            let state = locate_dataset_state(&mirror, dataset)?;
            self.located.insert(oh_addr, state);
        }

        // Validate the appended bytes against the on-disk datatype.
        let raw = b.raw();
        {
            let st = &self.located[&oh_addr];
            if raw.len() % st.element_size != 0 {
                return Err(Error::AppendInPlaceUnsupported(
                    "appended byte length is not a whole number of elements",
                ));
            }
            match b.elem_dt() {
                Some(expected) if *expected != st.datatype => {
                    return Err(Error::AppendInPlaceUnsupported(
                        "append datatype does not match the on-disk dataset (wrong element \
                         type or byte order)",
                    ));
                }
                Some(_) => {}
                None => {
                    if !datatype_is_raw_appendable(&st.datatype) {
                        return Err(Error::AppendInPlaceUnsupported(
                            "append_raw onto this dataset's datatype (non-little-endian, \
                             variable-length, or reference) could misencode the bytes; use a \
                             typed append",
                        ));
                    }
                }
            }
        }

        let new_elems = (raw.len() / self.located[&oh_addr].element_size) as u64;
        if new_elems == 0 {
            return Ok(());
        }

        // In SWMR mode, hold to the subset a concurrent reader can follow safely:
        // unfiltered (a filtered element is a multi-field record whose in-place
        // repoint is not power-loss atomic) and chunk-aligned (so an append only
        // ever inserts new, not-yet-visible elements and never rewrites a visible
        // trailing chunk out from under a reader).
        if self.swmr_mode {
            let st = &self.located[&oh_addr];
            if st.pipeline.is_some() {
                return Err(Error::SwmrAppendUnsupported(
                    "filtered datasets are not supported for SWMR append",
                ));
            }
            let chunk_elems = st.loc.chunk_elems;
            if chunk_elems == 0
                || st.loc.current_dim % chunk_elems != 0
                || new_elems % chunk_elems != 0
            {
                return Err(Error::SwmrAppendUnsupported(
                    "SWMR append must be chunk-aligned: the current length and the appended \
                     length must both be whole multiples of the chunk length",
                ));
            }
        }

        // Read/plan phase (immutable borrows only, nothing published yet), then the
        // ordered, fsync-barriered write phase — both shared with `AppendWriter`
        // through the chunk-index engine. `EditMirror` borrows only the
        // mirror-carrying fields, so `self.located` stays independently borrowable.
        let plan_result = {
            let st = &self.located[&oh_addr];
            let mirror = EditMirror {
                handle: &mut self.handle,
                data: &mut self.data,
                superblock: &mut self.superblock,
                sb_sig_off: self.sb_sig_off,
            };
            plan_ea_append(
                &mirror,
                &st.loc,
                &st.datatype,
                &st.spatial,
                st.element_size,
                st.pipeline.as_ref(),
                raw,
                new_elems,
            )
        };
        let plan = plan_result.map_err(as_inplace_error)?;
        let st = self
            .located
            .get_mut(&oh_addr)
            .expect("dataset located above");
        let mut mirror = EditMirror {
            handle: &mut self.handle,
            data: &mut self.data,
            superblock: &mut self.superblock,
            sb_sig_off: self.sb_sig_off,
        };
        apply_ea_append(&mut mirror, &mut st.loc, &plan, max_phase).map_err(as_inplace_error)
    }

    /// Test-only phased in-place append (stops after `max_phase` durability phases)
    /// used by the crash-consistency tests, mirroring `AppendWriter`'s harness.
    #[cfg(test)]
    fn append_inplace_i32_phased(
        &mut self,
        dataset: &str,
        values: &[i32],
        max_phase: u8,
    ) -> Result<(), Error> {
        let mut b = AppendBuilder::new();
        b.append_i32(values);
        self.append_inplace_gathered(dataset, &b, max_phase)
    }

    /// Whether `target` (an [`append_inplace`](Self::append_inplace) dataset path)
    /// or any of its ancestors is named by a staged edit that a later
    /// [`commit`](Self::commit) would relocate, replace, or delete. `create_group`
    /// and group-attribute edits are excluded: they rewrite a group header without
    /// moving a descendant dataset's header or freeing its storage, so they cannot
    /// stale the append geometry cache.
    fn append_conflicts_with_pending(&self, target: &[String]) -> bool {
        let hits = |p: &[String]| paths_overlap(target, p);
        self.pending_writes.iter().any(|(p, _)| hits(p))
            || self.pending_appends.iter().any(|(p, _)| hits(p))
            || self.pending_deletes.iter().any(|p| hits(p))
            || self.pending_copies.iter().any(|(_, dst)| hits(dst))
            || self.pending_cross_copies.iter().any(|(dst, _)| hits(dst))
            || self.pending_dataset_attrs.iter().any(|(p, _)| hits(p))
            || self.pending_datasets.iter().any(|(parent, db)| {
                let mut full = parent.clone();
                full.push(db.name.clone());
                paths_overlap(target, &full)
            })
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
    /// with [`create_group`](Self::create_group). Attributes — fixed-size or
    /// variable-length (`AttrValue::VarLenAsciiArray`) — are stored compactly in
    /// the rebuilt group header; an edit that would exceed the compact-attribute
    /// limit, or a group using dense (fractal-heap) attribute storage, is
    /// refused before any file bytes are changed.
    pub fn set_group_attr(&mut self, path: &str, name: &str, value: AttrValue) -> &mut Self {
        self.pending_group_attrs.push((
            split_path(path),
            AttrOp::Set {
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
            AttrOp::Remove {
                name: name.to_string(),
            },
        ));
        self
    }

    /// Stage an attribute add or replacement on an **existing dataset**, applied on
    /// the next [`commit`](Self::commit).
    ///
    /// `path` names the dataset to edit. Attributes — fixed-size or variable-length
    /// (`AttrValue::VarLenAsciiArray`) — are stored compactly in the rebuilt dataset
    /// header. Applying it relocates the dataset's object header (the header is
    /// rewritten and its single naming link repointed; the dataset's data and chunk
    /// index stay in place), so it is supported only when the dataset has a **single
    /// hard link**. An edit that would exceed the compact-attribute limit, or a
    /// dataset using dense (fractal-heap) attribute storage, is refused before any
    /// file bytes change. To set attributes on a dataset being *created* in this
    /// session, use the builder's [`set_attr`](crate::DatasetBuilder::set_attr)
    /// instead.
    pub fn set_dataset_attr(&mut self, path: &str, name: &str, value: AttrValue) -> &mut Self {
        self.pending_dataset_attrs.push((
            split_path(path),
            AttrOp::Set {
                name: name.to_string(),
                value,
            },
        ));
        self
    }

    /// Stage removal of a compact attribute from an **existing dataset**, applied on
    /// the next [`commit`](Self::commit).
    ///
    /// `path` names the dataset to edit; the named attribute must exist in the
    /// committed dataset state after any earlier staged attribute operations for the
    /// same dataset have been applied. Like [`set_dataset_attr`](Self::set_dataset_attr)
    /// it relocates the dataset header and requires a single hard link.
    pub fn remove_dataset_attr(&mut self, path: &str, name: &str) -> &mut Self {
        self.pending_dataset_attrs.push((
            split_path(path),
            AttrOp::Remove {
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
    /// end-of-file the file is truncated. Contiguous and chunked datasets (their
    /// chunk index and chunk data blocks) and whole group subtrees are all
    /// reclaimed. Reclaim is best-effort — an object whose blocks this engine
    /// cannot enumerate exhaustively (variable-length global-heap storage, dense
    /// attribute/link heaps, a version 2 B-tree chunk index) is left as dead
    /// bytes rather than risk freeing a region that is still in use. Freed space is
    /// reused within the open session; for a file created with
    /// `H5Pset_file_space_strategy(persist = true)` it is also recorded on disk so
    /// it survives reopen (see [`EditSession`]), otherwise it is forgotten
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
    /// exist and `dst` must not (and may not lie inside `src`). A chunked (and
    /// filtered) dataset is copied with its chunk payloads and filter pipeline
    /// preserved byte-for-byte (the index is rebuilt at the new location, so a
    /// source using a B-tree-v1 or implicit index is reproduced with an equivalent
    /// v4 index). The source subtree must otherwise be copyable in place: compact
    /// links and attributes, single-chunk headers, and a chunk index this engine
    /// can enumerate (a version-2 B-tree, or a sparse/unallocated chunk grid, is
    /// refused) — otherwise `commit` reports [`Error::EditUnsupported`].
    pub fn copy(&mut self, src: &str, dst: &str) {
        self.pending_copies.push((split_path(src), split_path(dst)));
    }

    /// Stage a deep copy of the object at `src` in another open file `source` to a
    /// new link at `dst` in this file — a *cross-file* HDF5 `H5Ocopy` — applied on
    /// the next [`commit`](Self::commit). Like [`copy`](Self::copy) but the source
    /// lives in a separate, independently-opened [`File`](crate::File) reader
    /// rather than the file being edited.
    ///
    /// The source — a dataset or a whole group subtree — is duplicated faithfully:
    /// fresh, byte-identical copies of every object's header and data are appended
    /// to this file, internal links repointed, and a link named by `dst`'s last
    /// component added to `dst`'s parent group (which must already exist or be
    /// created earlier in this session). Both files are left otherwise untouched;
    /// the destination only changes on `commit`.
    ///
    /// Unlike the same-file [`copy`](Self::copy), the source is read **eagerly**
    /// here (the `source` borrow need not outlive the call), so this returns
    /// `Result`: the source subtree is resolved, validated, and read out before
    /// returning, and only an already-validated copy is queued for `commit`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the copy cannot be reproduced exactly
    /// in another file. Because the copy is byte-for-byte verbatim, anything that
    /// embeds a *source-file* absolute address is refused (it would dangle here):
    /// **variable-length** or **reference** datasets and attributes (including a
    /// chunked dataset whose elements are variable-length or references, whose
    /// chunk payloads embed such addresses), and any **shared header message** (a
    /// committed datatype, or an SOHM-shared dataspace, fill value, or filter
    /// pipeline). As with [`copy`](Self::copy) a chunked/filtered source is copied
    /// with its chunk payloads and pipeline preserved (index rebuilt at the new
    /// location); the source must use compact links and attributes, single-chunk
    /// version-2 headers, and a chunk index this engine can enumerate (a
    /// version-2 B-tree, or a sparse chunk grid, is refused). The
    /// `source` must be a buffered file ([`File::open`](crate::File::open) or
    /// [`File::from_bytes`](crate::File::from_bytes), not
    /// [`open_streaming`](crate::File::open_streaming)) using 8-byte offsets and no
    /// userblock, and `src` must exist in it and not be the root group.
    pub fn copy_from(
        &mut self,
        source: &crate::reader::File,
        src: &str,
        dst: &str,
    ) -> Result<(), Error> {
        // The source bytes must be addressable: a streaming file is refused.
        let src_data = source.in_memory_image().ok_or(Error::EditUnsupported(
            "cross-file copy requires a buffered source file (File::open or File::from_bytes), not a streaming one",
        ))?;
        let src_sb = source.superblock();
        if src_sb.offset_size != OFFSET_SIZE || src_sb.length_size != LENGTH_SIZE {
            return Err(Error::EditUnsupported(
                "cross-file copy requires the source file to use 8-byte offsets and lengths",
            ));
        }
        if source.base_address() != 0 {
            return Err(Error::EditUnsupported(
                "cross-file copy requires the source file to have no userblock (base address 0)",
            ));
        }

        let src = split_path(src);
        if src.is_empty() {
            return Err(Error::EditUnsupported("cannot copy the root group"));
        }
        let dst = split_path(dst);
        if dst.is_empty() {
            return Err(Error::EditUnsupported("copy destination path is empty"));
        }

        let src_addr = crate::group_v2::resolve_path_any(src_data, src_sb, &src.join("/"))
            .map_err(|_| Error::EditUnsupported("copy source does not exist in the source file"))?;
        let src_addr = usize::try_from(src_addr)
            .map_err(|_| Error::EditUnsupported("source address exceeds this platform"))?;
        // Read (and foreign-address-screen) the whole subtree now, while `source`
        // is borrowed; the owned tree carries every byte the commit will write. The
        // source is gated to base 0 above, so its stored addresses are absolute.
        let tree = Self::read_copy_subtree(src_data, src_addr, 0, true, 0)?;
        self.pending_cross_copies.push((dst, tree));
        Ok(())
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
            && self.pending_writes.is_empty()
            && self.pending_appends.is_empty()
            && self.pending_groups.is_empty()
            && self.pending_group_attrs.is_empty()
            && self.pending_dataset_attrs.is_empty()
            && self.pending_deletes.is_empty()
            && self.pending_copies.is_empty()
            && self.pending_cross_copies.is_empty()
        {
            return Ok(());
        }

        // Invalidate the in-place-append geometry cache before doing any work. A
        // commit that reaches here rewrites and relocates object headers, frees
        // vacated regions into `self.free`, and may truncate the file — any of
        // which can leave a cached `Located` pointing at a moved header or into a
        // now-free-eligible region. Clearing at *entry* (rather than the success
        // tail) means a later failure — including one after the durable root flip,
        // which leaves the session reusable — never strands a stale cache. The
        // no-op fast return above does no such work, so it keeps the cache. The
        // next `append_inplace` re-locates against the fresh mirror.
        self.located.clear();

        // On a file with a userblock, stored addresses are relative to this base
        // and the editor converts at every disk boundary (read `stored + base`,
        // write `file_offset - base`). Userblock support covers value overwrites,
        // additions of contiguous and chunked/filtered datasets, in-place and
        // relocating overwrites of every layout (chunked, contiguous, compact) with
        // reclaim, object deletion (with base-aware subtree reclaim), object copy
        // (in-file, and cross-file into a userblock destination), group creation,
        // and compact group attributes. Cross-file copy still requires a base-0
        // *source* (see [`copy_from`](Self::copy_from)).
        let base = self.superblock.base_address;

        // --- Preflight value overwrites (`write_dataset`) before any write, under
        // the same all-or-nothing contract as additions. Each is resolved,
        // validated (datatype and shape must match the on-disk dataset exactly),
        // and classified: a same-length contiguous overwrite is applied straight
        // in place (no header rewrite, no superblock flip), while a resize or
        // compact rewrite relocates the header and is staged against its parent
        // group so the commit below rebuilds it and patches the link. ---
        let writes = std::mem::take(&mut self.pending_writes);
        let mut inplace_writes: Vec<(usize, Vec<u8>)> = Vec::new();
        let mut moving_writes: Vec<(PathKey, String, MovingWrite)> = Vec::new();
        let mut write_targets: Vec<PathKey> = Vec::new();
        // The file-wide hard-link count, computed lazily the first time a write
        // relocates a header: such a write moves the dataset's object header and
        // patches only the one parent link that names it, so a dataset reachable
        // through more than one hard link would have its other links left pointing
        // at the stale header. Refuse that rather than silently diverge the aliases
        // (a same-length in-place overwrite is unaffected — it rewrites the shared
        // data block, which every link sees).
        let mut incoming_links: Option<Option<HashMap<u64, u32>>> = None;
        for (full, db) in writes {
            if full.is_empty() {
                return Err(Error::EditUnsupported("cannot overwrite the root group"));
            }
            // A path named twice in one commit would write it twice (and double-
            // free a resized extent); require separate commits.
            if write_targets.contains(&full) {
                return Err(Error::EditUnsupported(
                    "the same dataset is overwritten twice in one commit; use separate commits",
                ));
            }
            let path_str = full.join("/");
            let addr = crate::group_v2::resolve_path_any(&self.data, &self.superblock, &path_str)
                .map_err(|_| {
                Error::EditUnsupported("nothing to overwrite at the given path")
            })?;
            let addr = usize::try_from(addr)
                .map_err(|_| Error::EditUnsupported("dataset address exceeds this platform"))?;
            let fd = flatten_dataset(db)?;
            match Self::prepare_write(&self.data, addr, &fd, base)? {
                WritePlan::InPlace { data_addr, raw } => inplace_writes.push((data_addr, raw)),
                WritePlan::InPlaceChunks { writes } => inplace_writes.extend(writes),
                WritePlan::Moving(mw) => {
                    // A relocating overwrite rewrites the dataset's header and data
                    // address. Every variant is base-aware on a userblock file: the
                    // chunked one rebuilds the chunk blob with stored addresses and
                    // reclaims the old storage base-relative, the contiguous one
                    // stores the relocated data address base-relative (and frees the
                    // old extent at its absolute offset), and the compact one carries
                    // its data inline. The parent link to the rewritten header is
                    // patched base-relative below.
                    //
                    // A relocating overwrite is safe only when this is the
                    // dataset's sole hard link. Compute the link graph once.
                    let counts = incoming_links
                        .get_or_insert_with(|| self.count_incoming_hard_links())
                        .as_ref();
                    match counts.and_then(|c| c.get(&(addr as u64))) {
                        Some(&1) => {}
                        _ => {
                            return Err(Error::EditUnsupported(
                                "overwriting a dataset that resizes or relocates its header is \
                                 only supported when it has a single hard link",
                            ));
                        }
                    }
                    let leaf = full.last().unwrap().clone();
                    let parent = full[..full.len() - 1].to_vec();
                    moving_writes.push((parent, leaf, mw));
                }
            }
            write_targets.push(full);
        }

        // --- Preflight appends (`append_dataset`) under the same all-or-nothing,
        // single-hard-link contract. Each plans a relocating append — existing
        // chunk data stays in place; the appended (and any rewritten trailing)
        // chunks and a rebuilt Extensible-Array index are staged, and the whole is
        // treated like a relocating overwrite of the dataset's header (staged
        // against its parent group so the commit patches the link). A zero-length
        // append is a no-op and is dropped here. ---
        let appends = std::mem::take(&mut self.pending_appends);
        for (full, ab) in appends {
            if full.is_empty() {
                return Err(Error::AppendUnsupported("cannot append to the root group"));
            }
            if ab.raw.is_empty() {
                continue; // nothing to append
            }
            // A dataset overwritten or appended earlier in this commit would be
            // planned against a stale header and its old storage double-freed;
            // require separate commits.
            if write_targets.contains(&full) {
                return Err(Error::AppendUnsupported(
                    "the same dataset is edited more than once in one commit; use separate commits",
                ));
            }
            let path_str = full.join("/");
            let addr = crate::group_v2::resolve_path_any(&self.data, &self.superblock, &path_str)
                .map_err(|_| {
                Error::AppendUnsupported("nothing to append to at the given path")
            })?;
            let addr = usize::try_from(addr)
                .map_err(|_| Error::AppendUnsupported("dataset address exceeds this platform"))?;
            let mw = Self::prepare_append(&self.data, addr, &ab, base)?;
            // A relocating append moves the dataset's object header and patches only
            // the one parent link that names it, so it is safe only when this is the
            // dataset's sole hard link (same rule as a relocating overwrite).
            let counts = incoming_links
                .get_or_insert_with(|| self.count_incoming_hard_links())
                .as_ref();
            match counts.and_then(|c| c.get(&(addr as u64))) {
                Some(&1) => {}
                _ => {
                    return Err(Error::AppendUnsupported(
                        "appending relocates the dataset header; only supported when it \
                         has a single hard link",
                    ));
                }
            }
            let leaf = full.last().unwrap().clone();
            let parent = full[..full.len() - 1].to_vec();
            moving_writes.push((parent, leaf, mw));
            write_targets.push(full);
        }

        // --- Preflight dataset attribute edits (`set_dataset_attr` /
        // `remove_dataset_attr`) under the same all-or-nothing, single-hard-link
        // contract. Each gathers the dataset's verbatim object-header region,
        // applies the compact attribute ops to it, and stages a relocating
        // `AttrEdit` header rewrite against the parent group — like a value
        // overwrite, but the data-layout message (and thus the chunk data and index)
        // is preserved verbatim, so only the header moves. ---
        let dataset_attrs = std::mem::take(&mut self.pending_dataset_attrs);
        if !dataset_attrs.is_empty() {
            // Collect the ops per dataset in first-seen path order, so multiple edits
            // to one dataset produce a single relocating header rewrite.
            let mut order: Vec<PathKey> = Vec::new();
            let mut ops_by_path: HashMap<PathKey, Vec<AttrOp>> = HashMap::new();
            for (path, op) in dataset_attrs {
                if !ops_by_path.contains_key(&path) {
                    order.push(path.clone());
                }
                ops_by_path.entry(path).or_default().push(op);
            }
            for full in order {
                let ops = ops_by_path.remove(&full).unwrap();
                if full.is_empty() {
                    return Err(Error::EditUnsupported(
                        "cannot set a dataset attribute on the root group; use set_group_attr",
                    ));
                }
                // A dataset already overwritten or appended in this commit would be
                // planned against a stale header; require separate commits.
                if write_targets.contains(&full) {
                    return Err(Error::EditUnsupported(
                        "the same dataset is edited more than once in one commit (an attribute \
                         edit plus another edit); use separate commits",
                    ));
                }
                let path_str = full.join("/");
                let addr =
                    crate::group_v2::resolve_path_any(&self.data, &self.superblock, &path_str)
                        .map_err(|_| {
                            Error::EditUnsupported(
                                "nothing to set an attribute on at the given path",
                            )
                        })?;
                let addr = usize::try_from(addr)
                    .map_err(|_| Error::EditUnsupported("dataset address exceeds this platform"))?;
                // An attribute edit relocates the dataset's object header and patches
                // only the one naming link, so it is safe only when this is the
                // dataset's sole hard link (same rule as a relocating overwrite).
                let counts = incoming_links
                    .get_or_insert_with(|| self.count_incoming_hard_links())
                    .as_ref();
                match counts.and_then(|c| c.get(&(addr as u64))) {
                    Some(&1) => {}
                    _ => {
                        return Err(Error::EditUnsupported(
                            "editing a dataset attribute relocates its header; only supported \
                             when it has a single hard link",
                        ));
                    }
                }
                let region = Self::gather_oh_messages(&self.data, addr, base)?;
                let (region, pending_vl_attrs) = apply_group_attr_ops(&region, &ops)?;
                let leaf = full.last().unwrap().clone();
                let parent = full[..full.len() - 1].to_vec();
                moving_writes.push((
                    parent,
                    leaf,
                    MovingWrite::AttrEdit {
                        region,
                        pending_vl_attrs,
                    },
                ));
                write_targets.push(full);
            }
        }

        // Fast path: when the only staged edits are same-length in-place
        // overwrites, apply them straight to their data blocks and return without
        // rebuilding any header or flipping the superblock root. The commit's
        // linearization point is the synced data write — there is no tree to
        // repoint, so each overwrite stands alone. (A persisting file takes the
        // same path: no free-space change occurs.)
        //
        // Because this path never rewrites the superblock, it deliberately leaves
        // it untouched — including a pre-existing stale consistency flag (e.g. one
        // left by a crashed SWMR writer). A lone same-length value overwrite does
        // not introduce any inconsistency, so it does not clear one either; an edit
        // that takes the full path below (any header/root change) clears the flag
        // as usual.
        if moving_writes.is_empty()
            && self.pending_datasets.is_empty()
            && self.pending_groups.is_empty()
            && self.pending_group_attrs.is_empty()
            && self.pending_deletes.is_empty()
            && self.pending_copies.is_empty()
            && self.pending_cross_copies.is_empty()
        {
            for (data_addr, raw) in &inplace_writes {
                self.write_at(*data_addr, raw)?;
            }
            self.handle.sync_all().map_err(Error::Io)?;
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

        // Attach relocating value overwrites (resized contiguous or compact) to
        // their parent group nodes: the new header is written below and the
        // parent's existing link patched to it, like an existing child group.
        for (parent, leaf, mw) in moving_writes {
            ensure_ancestors(&mut nodes, &parent);
            nodes.entry(parent).or_default().writes.push((leaf, mw));
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
            // Read the source subtree from this file's own mirror (`cross_file`
            // false: same address space, so verbatim addresses stay valid). On a
            // userblock file the stored addresses are base-relative, so pass this
            // session's base for the read to absolutize them.
            let tree = Self::read_copy_subtree(&self.data, src_addr, 0, false, base)?;
            add_targets.push(dst.clone());
            let leaf = dst.last().unwrap().clone();
            let parent = dst[..dst.len() - 1].to_vec();
            ensure_ancestors(&mut nodes, &parent);
            nodes.entry(parent).or_default().copies.push((leaf, tree));
        }

        // Stage cross-file copies: their subtrees were already read out of the
        // source file (with foreign-address screening) when `copy_from` was
        // called, so here they are simply linked into the destination parent like
        // any other addition.
        for (dst, tree) in std::mem::take(&mut self.pending_cross_copies) {
            if dst.is_empty() {
                return Err(Error::EditUnsupported("copy destination path is empty"));
            }
            add_targets.push(dst.clone());
            let leaf = dst.last().unwrap().clone();
            let parent = dst[..dst.len() - 1].to_vec();
            ensure_ancestors(&mut nodes, &parent);
            nodes.entry(parent).or_default().copies.push((leaf, tree));
        }

        // Stage deletions: each must exist, must not overlap any other staged
        // change, and is recorded against its parent group (which becomes dirty).
        // `deleted_addrs` keeps each removed object's header address so its owned
        // blocks can be reclaimed after the commit lands (issue #21).
        let delete_targets = std::mem::take(&mut self.pending_deletes);
        let mut deleted_addrs: Vec<usize> = Vec::new();
        for (i, d) in delete_targets.iter().enumerate() {
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
            for t in &write_targets {
                if is_prefix(d, t) {
                    return Err(Error::EditUnsupported(
                        "a deletion overlaps a value overwrite in the same commit; use separate commits",
                    ));
                }
            }
            for (j, d2) in delete_targets.iter().enumerate() {
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
        // contract as unsupported dataset additions. A variable-length attribute
        // is not fully resolved here — its global heap collection is built (it
        // is self-contained, no address needed yet) but placed and patched into
        // `base_region` only in the apply loop below, once its address is known.
        for key in &keys {
            let node = nodes.get_mut(key).unwrap();
            let ops = std::mem::take(&mut node.attr_ops);
            if !ops.is_empty() {
                let region = std::mem::take(&mut node.base_region);
                let (region, pending_vl_attrs) = apply_group_attr_ops(&region, &ops)?;
                node.base_region = region;
                node.pending_vl_attrs = pending_vl_attrs;
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

        // Prove every object-reference target resolves before any write (see
        // `preflight_reference_targets`'s doc comment): otherwise a reference
        // resolution failure discovered mid-apply-loop would leave every
        // earlier-processed group's real writes (headers, data, copied
        // subtrees) orphaned in the file despite `commit()` returning `Err`.
        Self::preflight_reference_targets(
            &keys,
            &flat,
            &nodes,
            &add_targets,
            &write_targets,
            &delete_targets,
            &self.data,
            &self.superblock,
        )?;

        // Gather the regions this commit will vacate, read from the current
        // on-disk layout before any byte moves: every deleted object's owned
        // blocks plus every superseded group header. These are not added to the
        // free list until after the superblock repoint (they remain live until
        // then), so the appends below never reuse them. Enumeration is
        // best-effort — `collect_free_spans` simply omits anything it cannot
        // account for exhaustively, so the worst case is unreclaimed dead bytes,
        // never a freed-but-live region.
        let mut to_free: Vec<(u64, u64)> = Vec::new();

        // An object's storage is reclaimed only when the link being removed is
        // its LAST hard link: HDF5 objects can have several hard links, and one
        // reachable through a surviving link is still live (freeing it would
        // corrupt the survivor). Count every hard link in the pre-commit file
        // and reclaim a deleted object only when its count is exactly 1.
        // `deleted_addrs` is de-duplicated first so two delete paths that are
        // hard links to the same object are not visited (and freed) twice. If
        // the link graph cannot be walked in full, no deleted object is
        // reclaimed (a safe leak), but superseded headers — always dead once the
        // root is repointed — still are.
        deleted_addrs.sort_unstable();
        deleted_addrs.dedup();
        if !deleted_addrs.is_empty() {
            if let Some(incoming) = self.count_incoming_hard_links() {
                for &a in &deleted_addrs {
                    self.collect_free_spans(a, 0, &incoming, &mut to_free);
                }
            }
        }
        // A superseded group header is dead once the root is repointed. Its chunk
        // spans are enumerated base-aware (`oh_chunk_spans` shifts continuation
        // addresses by the userblock base and returns absolute file offsets), as is
        // the delete path (`collect_free_spans`), so all of this reclamation works
        // on userblock files too.
        for &a in &superseded_addrs {
            if let Ok(spans) = self.oh_chunk_spans(a) {
                to_free.extend(spans);
            }
        }

        // A relocating overwrite (`write_dataset` resize, or any compact rewrite)
        // vacates the dataset's old object header, and a resized contiguous one
        // also vacates its old data block: both become dead once the parent's
        // relinked header lands. `superseded_addrs` covers only the rebuilt group
        // headers, not the relocated dataset's own header, so record that here too.
        // The pre-commit dataset-header address is resolved from the live file; its
        // chunks and old data extent are freed only after the superblock repoint.
        // The single-hard-link guard in the write preflight makes freeing the old
        // header safe (no surviving link still points at it).
        for key in &keys {
            for (leaf, mw) in &nodes[key].writes {
                match mw {
                    MovingWrite::Contiguous {
                        old_extent: Some(extent),
                        ..
                    } => to_free.push(*extent),
                    // A relocated chunked dataset vacates its old chunk index and
                    // chunk data blocks. `chunked_storage_spans` returns `None` for
                    // anything it cannot enumerate exhaustively (leaving dead bytes
                    // rather than freeing a region still in use); the old header
                    // chunks are freed generically below.
                    MovingWrite::Chunked { old_addr, .. } => {
                        if let Ok(a) = usize::try_from(*old_addr) {
                            if let Some(spans) = self.chunked_storage_spans(a) {
                                to_free.extend(spans);
                            }
                        }
                    }
                    // A relocating append keeps the existing chunk *data* in place
                    // (shared by both indexes during the commit), so only the old
                    // index structure and the relocated old trailing chunk are dead.
                    // The old header chunks are freed by the generic path below.
                    MovingWrite::AppendedChunks {
                        old_addr,
                        old_tail_extent,
                        ..
                    } => {
                        if let Ok(a) = usize::try_from(*old_addr) {
                            if let Some(spans) = self.chunked_index_spans(a) {
                                to_free.extend(spans);
                            }
                        }
                        if let Some(ext) = old_tail_extent {
                            to_free.push(*ext);
                        }
                    }
                    _ => {}
                }
                // The relocated dataset's old header chunks are dead too.
                let mut full = key.clone();
                full.push(leaf.clone());
                let path_str = full.join("/");
                if let Ok(addr) =
                    crate::group_v2::resolve_path_any(&self.data, &self.superblock, &path_str)
                {
                    if let Ok(a) = usize::try_from(addr) {
                        if let Ok(spans) = self.oh_chunk_spans(a) {
                            to_free.extend(spans);
                        }
                    }
                }
            }
        }

        // Defense in depth: never hand the free list an out-of-bounds or
        // overlapping span. The last-link guard plus the per-object checks
        // should already make the accumulated spans disjoint; this enforces it
        // as a whole-commit invariant against the pre-commit end-of-file. Any
        // dropped span (which should not occur for a well-formed file) only
        // leaks, never corrupts.
        retain_disjoint_in_bounds(&mut to_free, self.data.len() as u64);

        // --- Apply: process deepest groups first so each parent sees its
        // children's new addresses, then repoint the superblock last.
        // `path_addr` accumulates every group's and dataset's address as it is
        // placed — read by `resolve_reference_target` to resolve a same-commit
        // object-reference target (see the dataset-placement loop below for the
        // group/dataset key convention: a group's own path, or a dataset's
        // full parent+name path). ---
        let mut path_addr: BTreeMap<PathKey, u64> = BTreeMap::new();
        let mut by_depth = keys.clone();
        by_depth.sort_by_key(|k| std::cmp::Reverse(k.len())); // deepest first
        for key in &by_depth {
            let (mut region, deletes, copies, writes, pending_vl_attrs) = {
                let node = nodes.get_mut(key).unwrap();
                (
                    std::mem::take(&mut node.base_region),
                    std::mem::take(&mut node.deletes),
                    std::mem::take(&mut node.copies),
                    std::mem::take(&mut node.writes),
                    std::mem::take(&mut node.pending_vl_attrs),
                )
            };

            // Remove deleted links first (verbatim-preserving the rest).
            for name in &deletes {
                region = remove_link_from_region(&region, name)?;
            }

            // Write each staged source subtree and link its root into this group.
            // `write_copy_subtree` returns an absolute header address; the parent
            // link stores it relative to the userblock base.
            for (leaf, tree) in copies {
                let root = self.write_copy_subtree(&tree)?;
                region.extend_from_slice(&encode_link_message(&leaf, root - base));
            }

            // Datasets directly under this group. Appended addresses are absolute
            // file offsets; the contiguous data-layout address and the parent link
            // target are stored relative to the base address (`- base`). Placed
            // non-reference datasets first (recording each into `path_addr`), then
            // reference datasets — a reference to a *non-reference* sibling added
            // in the same group's batch resolves regardless of `pending_datasets`
            // call order (`Vec::sort_by_key` is stable, so within each of the two
            // groups the original order is preserved). Two reference datasets that
            // target each other in the same batch are still call-order-dependent —
            // whichever is placed first resolves the other, and the reverse
            // direction is safely refused as "still writing" (never corrupted),
            // caught up front by `preflight_reference_targets`.
            let mut group_datasets: Vec<FlatDataset> =
                flat.remove(key).into_iter().flatten().collect();
            group_datasets.sort_by_key(|fd| fd.reference_targets.is_some());
            for mut fd in group_datasets {
                // Place each variable-length attribute's global heap collection
                // and patch its placeholder heap address. Unlike VL-string
                // *data* (`vl_string_staging`, refused when chunked below), a
                // chunked/extensible dataset can carry a VL *attribute* just
                // fine — attributes live in the object header, not inside a
                // chunk, so patching them here before either apply branch runs
                // covers both.
                for (idx, collection_bytes) in std::mem::take(&mut fd.vl_attrs) {
                    let addr = self.place_vl_collection(&collection_bytes)?;
                    patch_vl_refs(&mut fd.attrs[idx].raw_data, addr);
                }
                // Resolve an object-reference dataset's per-element targets now
                // that every earlier-placed object in this commit is in
                // `path_addr` (chunked datasets never carry these —
                // `flatten_dataset` refuses that combination).
                if let Some(targets) = fd.reference_targets.take() {
                    let mut patched = Vec::with_capacity(targets.len() * 8);
                    for target in &targets {
                        let addr = Self::resolve_reference_target(
                            target,
                            &path_addr,
                            &nodes,
                            &add_targets,
                            &write_targets,
                            &delete_targets,
                            &self.data,
                            &self.superblock,
                        )?;
                        patched.extend_from_slice(&addr.to_le_bytes());
                    }
                    fd.raw = patched;
                }
                let oh = if fd.chunk_options.is_chunked() || fd.maxshape.is_some() {
                    self.build_chunked_dataset(&fd)?
                } else {
                    // A staged variable-length-string dataset's element
                    // references still carry a placeholder heap address; place
                    // its collection and patch them before `raw` is appended
                    // (chunked datasets never carry staging — refused above).
                    if let Some(staging) = fd.vl_string_staging.take() {
                        if !staging.collection_bytes.is_empty() {
                            let addr = self.place_vl_collection(&staging.collection_bytes)?;
                            patch_vl_refs_masked(&mut fd.raw, &staging.patch_mask, addr);
                        }
                    }
                    // A zero-element dataset has no data block to allocate; its
                    // layout address is the undefined-address sentinel (never
                    // base-relative — see `build_dataset_oh`'s empty-data callers
                    // in the whole-file writer), matching every reader's and the
                    // reference C library's convention for "no storage allocated".
                    let data_addr = if fd.raw.is_empty() {
                        u64::MAX
                    } else {
                        self.alloc_or_append(&fd.raw)? - base
                    };
                    build_dataset_oh(
                        &fd.dt,
                        &fd.ds,
                        data_addr,
                        fd.raw.len() as u64,
                        &fd.attrs,
                        None,
                        fd.fill.as_deref(),
                    )
                };
                let oh_addr = self.alloc_or_append(&oh)?;
                region.extend_from_slice(&encode_link_message(&fd.name, oh_addr - base));
                let mut full = key.clone();
                full.push(fd.name.clone());
                path_addr.insert(full, oh_addr);
            }

            // Relocating value overwrites under this group: write the new data and
            // rewritten header, then patch this group's existing link to it. The
            // link target is stored relative to the base address (`- base`); on a
            // userblock file only the chunked variant reaches here (contiguous and
            // compact resizes are refused in the write preflight).
            for (leaf, mw) in &writes {
                let new_oh = self.write_moving(mw)?;
                patch_link_target(&mut region, leaf, new_oh - base)?;
            }

            // Wire links to dirty child groups (new → add a link; existing →
            // patch the existing link to the child's new address). Link targets are
            // stored relative to the base address.
            for child in children.get(key).into_iter().flatten() {
                let child_name = child.last().unwrap();
                let child_addr = path_addr[child] - base;
                if nodes[child].is_new {
                    region.extend_from_slice(&encode_link_message(child_name, child_addr));
                } else {
                    patch_link_target(&mut region, child_name, child_addr)?;
                }
            }

            // Variable-length group/root attributes staged by
            // `apply_group_attr_ops`: place each collection and patch its
            // attribute message's placeholder heap address, then append the
            // resolved message to this group's header region.
            for (mut msg, collection_bytes) in pending_vl_attrs {
                let addr = self.place_vl_collection(&collection_bytes)?;
                patch_vl_refs(&mut msg.raw_data, addr);
                region.extend_from_slice(&region_message(
                    MessageType::Attribute,
                    &msg.serialize(LENGTH_SIZE),
                ));
            }

            let oh = build_v2_object_header(&region);
            let addr = self.alloc_or_append(&oh)?;
            path_addr.insert(key.clone(), addr);
        }

        // Same-length in-place overwrites (`write_dataset`) write straight into
        // their existing, already-referenced data blocks. Those blocks are
        // reachable from both the old and the new root (the dataset's header is
        // unchanged), so the write is independent of the superblock flip; it is
        // ordered before the barrier sync below so the new bytes are durable
        // alongside everything else this commit appended.
        for (data_addr, raw) in &inplace_writes {
            self.write_at(*data_addr, raw)?;
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
        let new_root = path_addr[&PathKey::new()];

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
        // The root address is stored relative to the base address; the end-of-file
        // address is absolute. After writing the relative root to disk, keep the
        // in-memory `root_group_address` absolute (the open-time convention).
        if self.superblock.version >= 2 {
            // Build the new superblock off a clone and adopt it only once the
            // write succeeds, so a failed write does not desync the in-memory
            // state. The v2/v3 superblock carries its own checksum.
            let mut new_sb = self.superblock.clone();
            new_sb.root_group_address = new_root - base;
            new_sb.eof_address = new_eof;
            // Clear any write/SWMR consistency flag rather than re-emitting one
            // the source file carried (e.g. left set by a crashed SWMR writer):
            // this clean commit leaves the file properly closed for the C library
            // (issue #73). serialize() recomputes the v2/v3 checksum.
            new_sb.consistency_flags = 0;
            let sb_bytes = new_sb.serialize();
            self.write_at(self.sb_sig_off, &sb_bytes)?;
            self.handle.sync_all().map_err(Error::Io)?;
            new_sb.root_group_address = new_root;
            self.superblock = new_sb;
        } else {
            self.repoint_v0v1_root(new_root - base, new_eof)?;
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
        let region = Self::gather_oh_messages(&self.data, ext_addr, self.superblock.base_address)?;
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
    fn oh_region(d: &[u8], addr: usize) -> Result<(usize, usize), Error> {
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
    fn gather_oh_messages(d: &[u8], addr: usize, base: u64) -> Result<Vec<u8>, Error> {
        let (rs, re) = Self::oh_region(d, addr)?;
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
                    // The continuation block address is stored relative to the base
                    // address; convert to an absolute file offset to index `d`.
                    let off = off
                        .checked_add(base)
                        .ok_or(Error::EditUnsupported("continuation address overflow"))?;
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
            // Group-entry addresses are already stored relative to the base address,
            // matching how `encode_link_message` stores link targets — so they are
            // re-emitted verbatim, no base conversion needed.
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
        let mut region = Self::gather_oh_messages(&self.data, addr, self.superblock.base_address)?;
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

    /// Preflight a staged value overwrite (`write_dataset`): resolve the dataset
    /// at `addr`, validate that the staged `fd` matches it byte-exactly in
    /// datatype and shape, and classify how the bytes will be applied. No file
    /// bytes are written here — this is part of the all-or-nothing preflight, so a
    /// rejected write leaves the commit unapplied.
    ///
    /// Contiguous, compact, and chunked (including filtered) datasets are all
    /// supported; the chunk geometry, filter pipeline, and chunk index come from
    /// the on-disk header (a staged builder that itself requests chunking/filters/an
    /// extensible shape is refused as "not a value overwrite", and a chunk index
    /// this engine cannot enumerate — a version-2 B-tree — is refused too). A
    /// datatype or shape that differs from the on-disk dataset's is likewise
    /// refused — this is a value overwrite, not a reshape or retype.
    fn prepare_write(
        d: &[u8],
        addr: usize,
        fd: &FlatDataset,
        base: u64,
    ) -> Result<WritePlan, Error> {
        // A value overwrite never introduces chunking, filters, or an extensible
        // shape: those would change the storage layout, not just the bytes.
        if fd.chunk_options.is_chunked() || fd.maxshape.is_some() {
            return Err(Error::EditUnsupported(
                "write_dataset overwrites values only; it cannot make a dataset \
                 chunked, filtered, or extensible",
            ));
        }

        // `write_dataset` overwrites element bytes only; it does not touch the
        // object header's attribute messages. Attributes staged on the returned
        // builder would otherwise be silently dropped (the in-place path rewrites
        // only the data block, and the moving path reuses the verbatim on-disk
        // header), so refuse rather than degrade — set them in a separate edit.
        if !fd.attrs.is_empty() {
            return Err(Error::EditUnsupported(
                "write_dataset overwrites values only; it cannot set attributes \
                 (set them with a separate edit)",
            ));
        }

        // `write_dataset` overwrites element bytes only; it reuses the dataset's
        // existing Fill Value message (the in-place path rewrites only the data
        // block, and the moving path keeps every header message but the layout
        // verbatim). A fill value staged on the returned builder would otherwise
        // be silently ignored, so refuse rather than degrade — set the fill value
        // when the dataset is first created.
        if fd.fill.is_some() {
            return Err(Error::EditUnsupported(
                "write_dataset overwrites values only; it cannot change the fill \
                 value (set it when the dataset is created)",
            ));
        }

        // `with_vlen_strings` stages placeholder element references that only the
        // add path's apply loop knows how to resolve (place the global heap
        // collection, then patch the placeholders once its address is known,
        // before the data block itself is written). `prepare_write` runs during
        // preflight, before any bytes are written and without `&mut self`
        // access to place a heap collection, and its result can be flushed by
        // the same-length fast path with no apply loop at all — so refuse
        // rather than write unpatched (heap address 0) placeholders as if they
        // were final.
        if fd.vl_string_staging.is_some() {
            return Err(Error::EditUnsupported(
                "write_dataset cannot overwrite a variable-length-string dataset's \
                 data in place yet",
            ));
        }

        let region = Self::gather_oh_messages(d, addr, base)?;

        // Locate the datatype, dataspace, and data-layout messages, and detect a
        // filter pipeline (filtered storage is always chunked, never contiguous).
        let mut datatype: Option<(usize, usize)> = None;
        let mut dataspace: Option<(usize, usize)> = None;
        let mut layout: Option<(usize, usize)> = None;
        let mut filter: Option<(usize, usize)> = None;
        let mut has_link = false;
        let mut p = 0;
        while let Some((msg_type, body, body_end)) = next_message(&region, p)? {
            match msg_type {
                MessageType::Datatype => datatype = Some((body, body_end)),
                MessageType::Dataspace => dataspace = Some((body, body_end)),
                MessageType::DataLayout => layout = Some((body, body_end)),
                MessageType::FilterPipeline => filter = Some((body, body_end)),
                MessageType::Link | MessageType::LinkInfo | MessageType::SymbolTable => {
                    has_link = true;
                }
                _ => {}
            }
            p = body_end;
        }

        if has_link {
            return Err(Error::EditUnsupported(
                "write_dataset target is a group, not a dataset",
            ));
        }
        let (dt_b, dt_e) =
            datatype.ok_or(Error::EditUnsupported("dataset header has no datatype"))?;
        let (ds_b, ds_e) =
            dataspace.ok_or(Error::EditUnsupported("dataset header has no dataspace"))?;
        let (lb, le) = layout.ok_or(Error::EditUnsupported("dataset header has no data layout"))?;

        // Compare datatype and shape structurally against the staged data. A
        // value overwrite must keep both exactly: the datatype (including its
        // class, size, endianness, and any compound/array/enumeration layout) so
        // the bytes are interpreted the same, and the *current* dimensions so the
        // byte count is unchanged. Parsing both sides and comparing the decoded
        // values — rather than the raw message bytes — tolerates the harmless
        // encoding differences between this crate's writer and the reference C
        // library (e.g. the C library records a maximum-dimensions array equal to
        // the current dimensions, which this crate omits) while still refusing any
        // real retype or reshape.
        let (disk_dt, _) = crate::datatype::Datatype::parse(&region[dt_b..dt_e])
            .map_err(|_| Error::EditUnsupported("dataset header datatype could not be parsed"))?;
        if disk_dt != fd.dt {
            return Err(Error::EditUnsupported(
                "write_dataset datatype does not match the on-disk dataset (overwrite, not retype)",
            ));
        }
        let disk_ds = Dataspace::parse(&region[ds_b..ds_e], LENGTH_SIZE)
            .map_err(|_| Error::EditUnsupported("dataset header dataspace could not be parsed"))?;
        if disk_ds.space_type != fd.ds.space_type
            || disk_ds.rank != fd.ds.rank
            || disk_ds.dimensions != fd.ds.dimensions
        {
            return Err(Error::EditUnsupported(
                "write_dataset shape does not match the on-disk dataset (overwrite, not reshape)",
            ));
        }

        // Classify the layout. Version 3/4 compact (class 0), contiguous (class
        // 1), and chunked (class 2) are supported; an old-version layout or a
        // virtual layout (class 3) is refused.
        if le - lb < 2 {
            return Err(Error::EditUnsupported("malformed data-layout message"));
        }
        let version = region[lb];
        if version != 3 && version != 4 {
            return Err(Error::EditUnsupported(
                "an unsupported data-layout version cannot be overwritten in place yet",
            ));
        }
        match region[lb + 1] {
            // Compact: the data is inline in the header. Rebuild the header with
            // the new inline bytes (relocating it), patching the parent link.
            0 => Ok(WritePlan::Moving(MovingWrite::Compact {
                region,
                raw: fd.raw.clone(),
            })),
            1 => {
                if le - lb < 18 {
                    return Err(Error::EditUnsupported("malformed contiguous data layout"));
                }
                let addr_off = lb + 2;
                let data_addr =
                    u64::from_le_bytes(region[addr_off..addr_off + 8].try_into().unwrap());
                let data_size = u64::from_le_bytes(region[lb + 10..lb + 18].try_into().unwrap());

                // Same length and a defined, in-bounds data block: overwrite the
                // bytes straight in place. No header rewrite, no relink. The stored
                // address is base-relative; the in-place write targets the absolute
                // file offset `data_addr + base`.
                if data_addr != UNDEF && data_size == fd.raw.len() as u64 {
                    if let Some(start) = data_addr
                        .checked_add(base)
                        .and_then(|a| usize::try_from(a).ok())
                    {
                        if start
                            .checked_add(fd.raw.len())
                            .is_some_and(|e| e <= d.len())
                        {
                            return Ok(WritePlan::InPlace {
                                data_addr: start,
                                raw: fd.raw.clone(),
                            });
                        }
                    }
                }

                // Length differs or the block was undefined/out of bounds: the new
                // data goes elsewhere and the old extent (if any) is freed. The
                // freed extent is recorded as an absolute file offset (`+ base`) to
                // match the session free list.
                let old_extent = if data_addr != UNDEF && data_size > 0 {
                    Some((data_addr + base, data_size))
                } else {
                    None
                };
                Ok(WritePlan::Moving(MovingWrite::Contiguous {
                    region,
                    addr_off,
                    raw: fd.raw.clone(),
                    old_extent,
                }))
            }
            // Chunked: overwrite each chunk in place when every new (re-encoded)
            // chunk is the same byte length as its slot, else rebuild and relocate
            // the whole chunk storage. The chunk geometry, filter pipeline, and
            // index type all come from the existing on-disk header (the staged
            // builder carries none — chunked/filtered/extensible builders are
            // refused at the top of this function as "not a value overwrite").
            2 => {
                // Chunked overwrite (in-place or relocating). On a userblock file
                // every stored chunk-index and chunk address is relative to `base`:
                // the in-place path below walks the index on a base-relative view of
                // the file and shifts the resulting write offsets back by `base`,
                // and the relocating path rebuilds the chunk blob with stored
                // addresses (see `write_chunked_relocatable`).
                let dl =
                    DataLayout::parse(&region[lb..le], OFFSET_SIZE, LENGTH_SIZE).map_err(|_| {
                        Error::EditUnsupported("dataset header data layout could not be parsed")
                    })?;
                let DataLayout::Chunked {
                    version: lversion,
                    chunk_index_type,
                    ..
                } = dl
                else {
                    return Err(Error::EditUnsupported("dataset is not chunked"));
                };
                if !chunk_index_enumerable(lversion, chunk_index_type) {
                    return Err(Error::EditUnsupported(
                        "a chunked dataset with a version-2 B-tree or unknown chunk index \
                         cannot be overwritten in place yet",
                    ));
                }

                let ChunkedGeometry {
                    spatial,
                    element_size,
                    raw_size,
                    maxshape,
                } = chunked_geometry(&fd.dt, &disk_ds, &dl)?;

                // Split the new value into full-size chunk buffers in dense
                // row-major grid order (edge overhang zero-filled, matching how
                // unfiltered chunks are stored), then re-encode through the on-disk
                // pipeline when the dataset is filtered.
                let split = split_into_chunks(&fd.raw, &disk_ds.dimensions, &spatial, element_size);
                let pipeline_message: Option<Vec<u8>> =
                    filter.map(|(fb, fe)| region[fb..fe].to_vec());

                let new_chunk_bytes: Vec<Vec<u8>> = if let Some(pm) = &pipeline_message {
                    let pipeline = FilterPipeline::parse(pm).map_err(|_| {
                        Error::EditUnsupported("dataset filter pipeline could not be parsed")
                    })?;
                    if !pipeline_reencodable(&pipeline) {
                        return Err(Error::EditUnsupported(
                            "a chunked dataset using a filter this engine cannot re-encode \
                             cannot be overwritten in place yet",
                        ));
                    }
                    let ctx = ChunkContext::from_datatype(&spatial, &fd.dt);
                    let mut encoded = Vec::with_capacity(split.len());
                    for (_, buf) in &split {
                        encoded.push(compress_chunk(buf, &pipeline, ctx)?);
                    }
                    encoded
                } else {
                    split.into_iter().map(|(_, buf)| buf).collect()
                };

                // Fast path: overwrite each chunk straight in its slot when every
                // new chunk fits. No header rewrite and no superblock flip — the
                // chunk (and index) blocks are reachable from both roots. The index
                // is left untouched when chunks keep their size and rebuilt in place
                // when they shrink. The index walk runs on a base-relative view of
                // the file (so the layout's stored addresses index correctly), and
                // the returned write offsets are shifted back to absolute file
                // offsets by adding `base` (a no-op on a base-0 file).
                let base_off = usize::try_from(base).map_err(|_| {
                    Error::EditUnsupported("userblock base address exceeds this platform")
                })?;
                if let Some(writes) = try_inplace_chunk_writes(
                    &d[base_off..],
                    &dl,
                    &disk_ds,
                    &spatial,
                    raw_size,
                    &new_chunk_bytes,
                ) {
                    let writes = writes
                        .into_iter()
                        .map(|(off, b)| (off + base_off, b))
                        .collect();
                    return Ok(WritePlan::InPlaceChunks { writes });
                }

                // Otherwise relocate: rebuild a fresh chunk blob + index at
                // end-of-file (carrying the re-encoded chunk bytes and the source
                // pipeline verbatim), swap the data-layout message in the verbatim
                // header, and free the old chunk storage after the commit lands.
                let meta = new_chunk_bytes
                    .iter()
                    .map(|c| ChunkMeta {
                        compressed_size: c.len() as u64,
                        filter_mask: 0,
                    })
                    .collect();
                Ok(WritePlan::Moving(MovingWrite::Chunked {
                    region,
                    chunk_dims: spatial,
                    element_size,
                    raw_size,
                    maxshape,
                    pipeline_message,
                    meta,
                    chunk_bytes: new_chunk_bytes,
                    old_addr: addr as u64,
                }))
            }
            _ => Err(Error::EditUnsupported(
                "an unsupported data-layout class cannot be overwritten in place yet",
            )),
        }
    }

    /// Plan a relocating append to an existing chunked, unlimited,
    /// Extensible-Array-indexed dataset at `addr`. Validates the target, splits
    /// the appended elements into new (and one rewritten trailing) chunks —
    /// compressed through the on-disk pipeline when filtered — and gathers the
    /// existing complete chunks by metadata alone. Returns the
    /// [`MovingWrite::AppendedChunks`] plan; the commit machinery appends the new
    /// chunks and a rebuilt index and repoints the header (see
    /// [`write_appended_chunks`](Self::write_appended_chunks)).
    ///
    /// Reads only `d` (the file mirror); no bytes are written here. `d` is the
    /// whole file image and `base` its userblock base; the dataset's stored
    /// (base-relative) structures are read through a `base`-shifted view.
    fn prepare_append(
        d: &[u8],
        addr: usize,
        ab: &AppendBuilder,
        base: u64,
    ) -> Result<MovingWrite, Error> {
        if ab.dt_conflict {
            return Err(Error::AppendUnsupported(
                "append mixes element types in one builder; use one element type per \
                 append_dataset call",
            ));
        }

        let region = Self::gather_oh_messages(d, addr, base)?;

        // Locate the datatype, dataspace, data-layout, and filter-pipeline
        // messages, and detect a group (link) header.
        let mut datatype: Option<(usize, usize)> = None;
        let mut dataspace: Option<(usize, usize)> = None;
        let mut layout: Option<(usize, usize)> = None;
        let mut filter: Option<(usize, usize)> = None;
        let mut has_link = false;
        let mut p = 0;
        while let Some((msg_type, body, body_end)) = next_message(&region, p)? {
            match msg_type {
                MessageType::Datatype => datatype = Some((body, body_end)),
                MessageType::Dataspace => dataspace = Some((body, body_end)),
                MessageType::DataLayout => layout = Some((body, body_end)),
                MessageType::FilterPipeline => filter = Some((body, body_end)),
                MessageType::Link | MessageType::LinkInfo | MessageType::SymbolTable => {
                    has_link = true;
                }
                _ => {}
            }
            p = body_end;
        }
        if has_link {
            return Err(Error::AppendUnsupported(
                "append target is a group, not a dataset",
            ));
        }
        let (dt_b, dt_e) =
            datatype.ok_or(Error::AppendUnsupported("dataset header has no datatype"))?;
        let (ds_b, ds_e) =
            dataspace.ok_or(Error::AppendUnsupported("dataset header has no dataspace"))?;
        let (lb, le) = layout.ok_or(Error::AppendUnsupported(
            "dataset header has no data layout",
        ))?;

        let (disk_dt, _) = Datatype::parse(&region[dt_b..dt_e])
            .map_err(|_| Error::AppendUnsupported("dataset header datatype could not be parsed"))?;
        let disk_ds = Dataspace::parse(&region[ds_b..ds_e], LENGTH_SIZE).map_err(|_| {
            Error::AppendUnsupported("dataset header dataspace could not be parsed")
        })?;
        let dl = DataLayout::parse(&region[lb..le], OFFSET_SIZE, LENGTH_SIZE).map_err(|_| {
            Error::AppendUnsupported("dataset header data layout could not be parsed")
        })?;

        // Require chunked, data-layout version 4, Extensible-Array index (type 4).
        let DataLayout::Chunked {
            version: lversion,
            chunk_index_type,
            btree_address,
            ..
        } = &dl
        else {
            return Err(Error::AppendUnsupported(
                "append requires a chunked dataset",
            ));
        };
        if *lversion != 4 || *chunk_index_type != Some(4) {
            return Err(Error::AppendUnsupported(
                "append requires an Extensible-Array-indexed chunked dataset (a single \
                 unlimited dimension under the latest format)",
            ));
        }

        // Require rank 1, unlimited along axis 0.
        if disk_ds.space_type != DataspaceType::Simple || disk_ds.dimensions.len() != 1 {
            return Err(Error::AppendUnsupported(
                "append requires a rank-1 dataset in this release",
            ));
        }
        match &disk_ds.max_dimensions {
            Some(md) if md.first() == Some(&u64::MAX) => {}
            _ => {
                return Err(Error::AppendUnsupported(
                    "append requires a dataset that is unlimited along its first dimension",
                ));
            }
        }

        let ChunkedGeometry {
            spatial,
            element_size,
            raw_size,
            ..
        } = chunked_geometry(&disk_dt, &disk_ds, &dl)?;
        let chunk_elems = spatial[0];
        if chunk_elems == 0 {
            return Err(Error::AppendUnsupported(
                "append requires a nonzero chunk length",
            ));
        }

        // Validate the appended bytes against the on-disk element type.
        if ab.raw.len() % element_size != 0 {
            return Err(Error::AppendUnsupported(
                "appended byte length is not a whole number of elements",
            ));
        }
        match &ab.elem_dt {
            // A typed append must match the on-disk datatype exactly (class, size,
            // and byte order) — this is a value append, not a retype.
            Some(expected) if *expected != disk_dt => {
                return Err(Error::AppendUnsupported(
                    "append datatype does not match the on-disk dataset (wrong element \
                     type or byte order)",
                ));
            }
            Some(_) => {}
            // A raw append trusts the caller's bytes but still refuses any datatype
            // whose flat little-endian bytes cannot be written verbatim: a
            // big-endian numeric leaf would silently misencode, and a
            // variable-length or reference leaf embeds heap/object addresses a byte
            // append cannot reproduce. A typed append is byte-order- and
            // class-checked by the datatype-equality arm above.
            None => {
                if !datatype_is_raw_appendable(&disk_dt) {
                    return Err(Error::AppendUnsupported(
                        "append_raw onto this dataset's datatype (non-little-endian, \
                         variable-length, or reference) could misencode the bytes; use a \
                         typed append",
                    ));
                }
            }
        }

        let new_elems = (ab.raw.len() / element_size) as u64;
        let current_dim0 = disk_ds.dimensions[0];
        let new_dim0 = current_dim0
            .checked_add(new_elems)
            .ok_or(Error::AppendUnsupported(
                "append would overflow the dataset dimension",
            ))?;

        // The filter pipeline is preserved verbatim in the rebuilt header; parse it
        // to re-encode the new chunks. An engine-unencodable filter is refused.
        let pipeline_message: Option<Vec<u8>> = filter.map(|(fb, fe)| region[fb..fe].to_vec());
        let has_filters = pipeline_message.is_some();
        let pipeline = match &pipeline_message {
            Some(pm) => {
                let parsed = FilterPipeline::parse(pm).map_err(|_| {
                    Error::AppendUnsupported("dataset filter pipeline could not be parsed")
                })?;
                if !pipeline_reencodable(&parsed) {
                    return Err(Error::AppendUnsupported(
                        "dataset uses a filter this engine cannot re-encode",
                    ));
                }
                Some(parsed)
            }
            None => None,
        };

        let base_off = usize::try_from(base).map_err(|_| {
            Error::AppendUnsupported("userblock base address exceeds this platform")
        })?;
        let view = d.get(base_off..).ok_or(Error::AppendUnsupported(
            "userblock base address past end-of-file",
        ))?;

        // The rebuilt index's element format (bare address vs address+size+mask) is
        // chosen by `has_filters`; it must agree with the source index's client id,
        // or the kept chunks — carried by metadata into the new index — would be
        // re-encoded in the wrong element width.
        if let Some(idx_addr) = *btree_address {
            let src = crate::source::BytesSource::new(view);
            let hdr =
                ExtensibleArrayHeader::parse_from_source(&src, idx_addr, OFFSET_SIZE, LENGTH_SIZE)
                    .map_err(|_| {
                        Error::AppendUnsupported(
                            "dataset extensible-array header could not be parsed",
                        )
                    })?;
            if (hdr.client_id == 1) != has_filters {
                return Err(Error::AppendUnsupported(
                    "dataset filter metadata is inconsistent (chunk-index client id \
                     disagrees with the filter pipeline)",
                ));
            }
        }

        // Enumerate the existing chunks (base-relative addresses) and require a
        // dense grid: `plan_dense_grid` returns the chunks in index order and
        // `None` on any hole, duplicate, or count mismatch against the dimension.
        let infos = enumerate_chunks_buffered(view, &dl, &disk_ds, OFFSET_SIZE, LENGTH_SIZE)
            .map_err(|_| Error::AppendUnsupported("dataset chunk index could not be enumerated"))?;
        let grid = plan_dense_grid(infos, &disk_ds.dimensions, &spatial).ok_or(
            Error::AppendUnsupported(
                "dataset has a sparse or inconsistent chunk grid; cannot append",
            ),
        )?;
        let grid_order = grid.grid_order;

        // Complete chunks are kept by metadata; a trailing partial chunk (when the
        // current length is not chunk-aligned) is rewritten.
        let n_full = usize::try_from(current_dim0 / chunk_elems)
            .map_err(|_| Error::AppendUnsupported("chunk count exceeds this platform"))?;
        let has_partial = current_dim0 % chunk_elems != 0;

        let mut kept_chunks: Vec<WrittenChunk> = Vec::with_capacity(n_full);
        for ci in grid_order.iter().take(n_full) {
            kept_chunks.push(WrittenChunk {
                address: ci.address,
                compressed_size: u64::from(ci.chunk_size),
                raw_size,
                // Preserve the source mask verbatim: a C/h5py file records a nonzero
                // mask for a chunk whose filter was skipped (e.g. deflate on
                // incompressible data), and forcing it to 0 would corrupt that chunk.
                filter_mask: ci.filter_mask,
            });
        }

        // Build the raw byte region for the tail (from the last chunk boundary to
        // the new end): the live prefix of any rewritten partial chunk, then the
        // appended bytes.
        let mut tail_raw: Vec<u8> = Vec::new();
        let mut old_tail_extent: Option<(u64, u64)> = None;
        if has_partial {
            let partial = &grid_order[n_full];
            let start = usize::try_from(partial.address)
                .map_err(|_| Error::AppendUnsupported("chunk address exceeds this platform"))?;
            let len = partial.chunk_size as usize;
            let end = start.checked_add(len).filter(|&e| e <= view.len()).ok_or(
                Error::AppendUnsupported("trailing chunk extends past end-of-file"),
            )?;
            let stored = &view[start..end];
            let full = if let Some(pl) = &pipeline {
                let ctx = ChunkContext::from_datatype(&spatial, &disk_dt);
                decompress_chunk(stored, pl, ctx, partial.filter_mask).map_err(Error::Format)?
            } else {
                stored.to_vec()
            };
            let live_elems = usize::try_from(current_dim0 % chunk_elems)
                .map_err(|_| Error::AppendUnsupported("chunk length exceeds this platform"))?;
            let live_bytes = live_elems * element_size;
            if full.len() < live_bytes {
                return Err(Error::AppendUnsupported(
                    "trailing chunk decoded shorter than its live element count",
                ));
            }
            tail_raw.extend_from_slice(&full[..live_bytes]);
            // The old partial chunk's data block is dead once the new index lands.
            old_tail_extent = Some((partial.address + base, u64::from(partial.chunk_size)));
        }
        tail_raw.extend_from_slice(&ab.raw);

        // Split the tail into full chunk buffers (edge overhang zero-filled) and
        // compress each through the pipeline when filtered.
        let tail_len_elems = new_dim0 - (n_full as u64) * chunk_elems;
        let split = split_into_chunks(&tail_raw, &[tail_len_elems], &spatial, element_size);
        let new_chunk_bytes: Vec<Vec<u8>> = if let Some(pl) = &pipeline {
            let ctx = ChunkContext::from_datatype(&spatial, &disk_dt);
            let mut out = Vec::with_capacity(split.len());
            for (_, buf) in &split {
                out.push(compress_chunk(buf, pl, ctx).map_err(Error::Format)?);
            }
            out
        } else {
            split.into_iter().map(|(_, buf)| buf).collect()
        };

        // Grow the dataspace along axis 0, preserving the (unlimited) max-dims.
        let mut grown = disk_ds.clone();
        grown.dimensions[0] = new_dim0;
        let new_dataspace_body = grown.serialize(LENGTH_SIZE);

        #[expect(
            clippy::cast_possible_truncation,
            reason = "spatial chunk dims come from the on-disk u32 chunk_dimensions, so they fit u32"
        )]
        let chunk_dims_u32: Vec<u32> = spatial.iter().map(|&dm| dm as u32).collect();

        Ok(MovingWrite::AppendedChunks {
            region,
            new_dataspace_body,
            chunk_dims_u32,
            element_size,
            raw_size,
            has_filters,
            kept_chunks,
            new_chunk_bytes,
            old_addr: addr as u64,
            old_tail_extent,
        })
    }

    /// Parse the object header at `addr` into a copyable model, validating that
    /// every message can be reproduced faithfully (verbatim message bytes, with
    /// only the contiguous data address and child link targets repointed).
    /// Dense (fractal-heap) attribute storage is read out of the source heap into
    /// a parsed attribute set carried on the model (`dense_attrs`) and re-emitted
    /// into a fresh heap on write, provided it fits the single-direct-block layout
    /// the emitter can build; an oversized set is refused. Rejects multi-chunk
    /// headers, dense or soft/external links, chunked/old-version data layouts, and
    /// headers that are neither a dataset nor a group.
    fn read_object(d: &[u8], addr: usize, base: u64) -> Result<ObjModel, Error> {
        let region = Self::gather_oh_messages(d, addr, base)?;

        // First pass: detect whether attributes are stored densely (a defined
        // fractal-heap address in the Attribute Info message). A dense object is
        // copied by reading its attributes out of the source heap and rebuilding
        // a fresh heap on write, so its Attribute Info message and any inline
        // Attribute messages are dropped from the verbatim region — the rebuilt
        // region carries neither, and `dense_attrs` carries the parsed set.
        let mut dense = false;
        let mut p = 0;
        while let Some((msg_type, body, body_end)) = next_message(&region, p)? {
            if msg_type == MessageType::AttributeInfo {
                // An Attribute Info message does not by itself mean dense
                // storage: the reference C library and h5py emit one (with an
                // *undefined* fractal-heap address) even for compact, inline
                // attributes in the latest format, to carry attribute
                // creation-order metadata. Only a *defined* heap address is real
                // dense (fractal-heap) storage. A message that cannot be parsed
                // is refused conservatively.
                let ai = crate::attribute_info::AttributeInfoMessage::parse(
                    &region[body..body_end],
                    OFFSET_SIZE,
                )
                .map_err(|_| {
                    Error::EditUnsupported(
                        "a source attribute-info message could not be parsed for copying",
                    )
                })?;
                if ai.fractal_heap_address.is_some() {
                    dense = true;
                }
            }
            p = body_end;
        }

        // If dense, read the attribute set out of the source fractal heap now (so
        // the source buffer need not outlive the read) and validate it can be
        // re-emitted into a fresh heap on write. `extract_attributes_full` reads
        // both compact and dense attributes; a dense object carries no inline
        // Attribute messages, so it returns exactly the heap-resident set.
        let dense_attrs = if dense {
            let header =
                ObjectHeader::parse_with_base(d, addr, OFFSET_SIZE, LENGTH_SIZE, base).map_err(|_| {
                    Error::EditUnsupported(
                        "a source object header with dense attributes could not be parsed for copying",
                    )
                })?;
            let attrs = crate::attribute::extract_attributes_full(
                d,
                &header,
                OFFSET_SIZE,
                LENGTH_SIZE,
            )
            .map_err(|_| {
                Error::EditUnsupported(
                    "a source object's dense (fractal-heap) attributes could not be read for copying",
                )
            })?;
            if !crate::file_writer::dense_attrs_fit(&attrs) {
                return Err(Error::EditUnsupported(
                    "an object's dense (fractal-heap) attribute set is too large to reproduce (would need fractal-heap indirect blocks)",
                ));
            }
            attrs
        } else {
            Vec::new()
        };

        let mut layout: Option<(usize, usize)> = None; // (body offset in kept, size)
        let mut has_link_info = false;
        let mut children: Vec<(String, u64)> = Vec::new();
        // The rebuilt chunk-0 region: every message kept verbatim except hard
        // Link messages (carried as `children`) and, when dense, the Attribute
        // Info message and inline Attribute messages (carried as `dense_attrs`).
        let mut kept: Vec<u8> = Vec::new();

        let mut p = 0;
        while let Some((msg_type, body, body_end)) = next_message(&region, p)? {
            let mut keep = true;
            match msg_type {
                MessageType::AttributeInfo => {
                    // Already parsed in the first pass; drop the dense Attribute
                    // Info message so the rebuilt header references the fresh heap
                    // (spliced in on write) rather than the source one. A compact
                    // (undefined-heap) Attribute Info message is kept verbatim.
                    if dense {
                        keep = false;
                    }
                }
                MessageType::Attribute => {
                    // A dense object should carry no inline Attribute messages,
                    // but drop any defensively so the rebuilt header's only
                    // attribute storage is the fresh heap.
                    if dense {
                        keep = false;
                    }
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
                MessageType::Link => {
                    keep = false;
                    match LinkMessage::parse(&region[body..body_end], OFFSET_SIZE) {
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
                    }
                }
                MessageType::DataLayout => {
                    // Record the layout body offset within the *kept* region so a
                    // contiguous dataset's data-address field can be repointed
                    // even after earlier messages were dropped.
                    layout = Some((kept.len() + (body - p), body_end - body));
                }
                _ => {}
            }
            if keep {
                kept.extend_from_slice(&region[p..body_end]);
            }
            p = body_end;
        }

        if let Some((lbody, lsize)) = layout {
            let version = kept[lbody];
            if !(version == 3 || version == 4) || lsize < 2 {
                return Err(Error::EditUnsupported(
                    "an unsupported data-layout version cannot be copied in place yet",
                ));
            }
            let class = kept[lbody + 1];
            match class {
                0 => Ok(ObjModel::DatasetVerbatim {
                    region: kept,
                    dense_attrs,
                }),
                1 => {
                    if lbody + 18 > kept.len() {
                        return Err(Error::EditUnsupported("malformed contiguous data layout"));
                    }
                    let data_addr =
                        u64::from_le_bytes(kept[lbody + 2..lbody + 10].try_into().unwrap());
                    let data_size =
                        u64::from_le_bytes(kept[lbody + 10..lbody + 18].try_into().unwrap());
                    Ok(ObjModel::DatasetContiguous {
                        region: kept,
                        addr_off: lbody + 2,
                        data_addr,
                        data_size,
                        dense_attrs,
                    })
                }
                // Chunked: the verbatim header carries the data-layout and filter-
                // pipeline messages; `read_copy_subtree` (which holds the source
                // buffer) enumerates and captures the chunk bytes and rebuilds the
                // index on write.
                2 => Ok(ObjModel::DatasetChunked {
                    region: kept,
                    dense_attrs,
                }),
                _ => Err(Error::EditUnsupported(
                    "an unsupported data-layout class cannot be copied in place yet",
                )),
            }
        } else if has_link_info {
            // A copied group must carry a Group Info message so the copy stays
            // writable by the C library, even when the source omitted it.
            ensure_group_info(&mut kept)?;
            Ok(ObjModel::Group {
                non_link_region: kept,
                children,
                dense_attrs,
            })
        } else {
            Err(Error::EditUnsupported(
                "an object is neither a contiguous/compact dataset nor a group",
            ))
        }
    }

    /// Read the object at `addr` in the source buffer `d` — and, for a group, its
    /// whole subtree — into an owned [`CopyTree`], the read half of an object copy.
    /// No bytes are written; this both validates that the subtree is copyable and
    /// captures the bytes the write half ([`write_copy_subtree`](Self::write_copy_subtree))
    /// later appends, so the source buffer need not outlive the read.
    ///
    /// `d` is the buffer the source object lives in: this session's own mirror for
    /// an in-file [`copy`](Self::copy), or another file's image for a cross-file
    /// [`copy_from`](Self::copy_from). `base` is that buffer's userblock base (the
    /// session's own base for an in-file copy, always 0 for a cross-file copy, whose
    /// source is gated to base 0): the stored, base-relative addresses read out of
    /// the source headers are shifted by it to index `d`. When `cross_file` is set,
    /// every copied object header is additionally screened by
    /// [`reject_foreign_addresses`] — verbatim bytes that embed a *source-file*
    /// absolute address (variable-length or reference data, a committed datatype)
    /// would dangle in another file and are refused, whereas an in-file copy keeps
    /// them valid by sharing the source file's heaps and objects.
    fn read_copy_subtree(
        d: &[u8],
        addr: usize,
        depth: u32,
        cross_file: bool,
        base: u64,
    ) -> Result<CopyTree, Error> {
        if depth >= MAX_COPY_DEPTH {
            return Err(Error::EditUnsupported(
                "copy source nests too deeply (possible hard-link cycle)",
            ));
        }
        // `base` is the userblock base of the buffer `d`: this session's own base
        // for an in-file copy, and always 0 for a cross-file copy (the source is
        // gated to base 0 in `copy_from`). `addr` is an absolute offset into `d`;
        // the stored (base-relative) addresses `read_object` returns for contiguous
        // data, chunk storage, and child links are converted to absolute offsets by
        // adding `base` before `d` is indexed or a child is descended into.
        let base_off = usize::try_from(base)
            .map_err(|_| Error::EditUnsupported("userblock base address exceeds this platform"))?;
        match Self::read_object(d, addr, base)? {
            ObjModel::DatasetVerbatim {
                region,
                dense_attrs,
            } => {
                if cross_file {
                    reject_foreign_addresses(&region)?;
                    reject_foreign_dense_attrs(&dense_attrs)?;
                }
                Ok(CopyTree::DatasetVerbatim {
                    region,
                    dense_attrs,
                })
            }
            ObjModel::DatasetContiguous {
                region,
                addr_off,
                data_addr,
                data_size,
                dense_attrs,
            } => {
                if cross_file {
                    reject_foreign_addresses(&region)?;
                    reject_foreign_dense_attrs(&dense_attrs)?;
                }
                // The stored data address is base-relative; shift it to an absolute
                // offset into `d` before slicing out the data block.
                let start = data_addr
                    .checked_add(base)
                    .and_then(|a| usize::try_from(a).ok())
                    .ok_or(Error::EditUnsupported("data address exceeds this platform"))?;
                let len = usize::try_from(data_size)
                    .map_err(|_| Error::EditUnsupported("data size exceeds this platform"))?;
                let end = start
                    .checked_add(len)
                    .filter(|&e| e <= d.len())
                    .ok_or(Error::EditUnsupported("dataset data is out of bounds"))?;
                Ok(CopyTree::DatasetContiguous {
                    region,
                    addr_off,
                    data: d[start..end].to_vec(),
                    dense_attrs,
                })
            }
            ObjModel::DatasetChunked {
                region,
                dense_attrs,
            } => {
                // Screen the verbatim header on the cross-file path. This refuses a
                // variable-length or reference datatype (whose chunk payload embeds
                // source-file global-heap / object addresses that would dangle in
                // another file) and any shared message — exactly the forms repack
                // also refuses for a cross-file verbatim chunk copy. An in-file copy
                // keeps them valid by sharing the source file's heaps.
                if cross_file {
                    reject_foreign_addresses(&region)?;
                    reject_foreign_dense_attrs(&dense_attrs)?;
                }
                let ChunkedHeaderParts {
                    dt,
                    ds,
                    layout,
                    pipeline_message,
                } = parse_chunked_header(&region)?;
                let DataLayout::Chunked {
                    version: lversion,
                    chunk_index_type,
                    ..
                } = layout
                else {
                    return Err(Error::EditUnsupported("dataset is not chunked"));
                };
                if !chunk_index_enumerable(lversion, chunk_index_type) {
                    return Err(Error::EditUnsupported(
                        "a chunked dataset with a version-2 B-tree or unknown chunk index \
                         cannot be copied in place yet",
                    ));
                }
                let ChunkedGeometry {
                    spatial: chunk_dims,
                    element_size,
                    raw_size,
                    maxshape,
                } = chunked_geometry(&dt, &ds, &layout)?;

                // The layout's chunk-index address and every chunk address it leads
                // to are stored base-relative, so enumerate and read on a
                // base-relative view of the source buffer (a no-op slice on a base-0
                // file). The returned addresses are then offsets into `dview`.
                let dview = &d[base_off..];

                // Enumerate the source chunks and map them onto a dense grid; a
                // sparse (holed/unallocated) dataset cannot be reproduced by the
                // verbatim layout path, which needs every grid slot filled.
                let infos =
                    enumerate_chunks_buffered(dview, &layout, &ds, OFFSET_SIZE, LENGTH_SIZE)?;
                let grid = plan_dense_grid(infos, &ds.dimensions, &chunk_dims).ok_or(
                    Error::EditUnsupported(
                        "a chunked dataset with unallocated (sparse) chunks cannot be copied in place yet",
                    ),
                )?;
                if grid.grid_order.is_empty() {
                    return Err(Error::EditUnsupported(
                        "an empty chunked dataset cannot be copied in place yet",
                    ));
                }

                // Capture each chunk's already-compressed bytes (no decode) into an
                // owned buffer, in dense row-major grid order, so the copy can be
                // written after the source buffer is gone (cross-file copy reads at
                // staging time). Sizes and masks are carried verbatim.
                let mut meta = Vec::with_capacity(grid.grid_order.len());
                let mut chunk_bytes = Vec::with_capacity(grid.grid_order.len());
                for ci in &grid.grid_order {
                    let start = usize::try_from(ci.address).map_err(|_| {
                        Error::EditUnsupported("chunk address exceeds this platform")
                    })?;
                    let len = ci.chunk_size as usize;
                    let end = start
                        .checked_add(len)
                        .filter(|&e| e <= dview.len())
                        .ok_or(Error::EditUnsupported("chunk data is out of bounds"))?;
                    chunk_bytes.push(dview[start..end].to_vec());
                    meta.push(ChunkMeta {
                        compressed_size: ci.chunk_size as u64,
                        filter_mask: ci.filter_mask,
                    });
                }

                Ok(CopyTree::DatasetChunked {
                    region,
                    chunk_dims,
                    element_size,
                    raw_size,
                    maxshape,
                    pipeline_message,
                    meta,
                    chunk_bytes,
                    dense_attrs,
                })
            }
            ObjModel::Group {
                non_link_region,
                children,
                dense_attrs,
            } => {
                if cross_file {
                    reject_foreign_addresses(&non_link_region)?;
                    reject_foreign_dense_attrs(&dense_attrs)?;
                }
                let mut kids = Vec::with_capacity(children.len());
                for (name, child) in children {
                    // Child link targets are stored base-relative; re-absolutize
                    // before descending so `addr` stays an absolute offset into `d`.
                    let child = child
                        .checked_add(base)
                        .and_then(|a| usize::try_from(a).ok())
                        .ok_or(Error::EditUnsupported(
                            "child address exceeds this platform",
                        ))?;
                    kids.push((
                        name,
                        Self::read_copy_subtree(d, child, depth + 1, cross_file, base)?,
                    ));
                }
                Ok(CopyTree::Group {
                    non_link_region,
                    children: kids,
                    dense_attrs,
                })
            }
        }
    }

    /// Append the fresh copies described by `node` (data blobs and headers) into
    /// this session at end-of-file or into reusable freed regions, returning the
    /// new object-header address of the copied root. The write half of an object
    /// copy; children are written before their parent group so each parent links
    /// its children's new addresses, and a contiguous dataset's data-address field
    /// is repointed at the freshly-written copy. Every address the copy writes into
    /// a header (a contiguous data block, a child link) is stored relative to the
    /// userblock base (`- base`, a no-op on a base-0 file); the chunked storage and
    /// dense attribute heaps are laid out base-relative by their own builders.
    fn write_copy_subtree(&mut self, node: &CopyTree) -> Result<u64, Error> {
        let base = self.superblock.base_address;
        match node {
            CopyTree::DatasetVerbatim {
                region,
                dense_attrs,
            } => {
                let mut region = region.clone();
                self.append_dense_attrs(&mut region, dense_attrs)?;
                let oh = build_v2_object_header(&region);
                self.alloc_or_append(&oh)
            }
            CopyTree::DatasetContiguous {
                region,
                addr_off,
                data,
                dense_attrs,
            } => {
                let new_data_addr = self.alloc_or_append(data)?;
                let mut region = region.clone();
                // `alloc_or_append` returns an absolute offset; the data-layout
                // address field stores it relative to the userblock base.
                region[*addr_off..*addr_off + 8]
                    .copy_from_slice(&(new_data_addr - base).to_le_bytes());
                // Append the dense heap *after* the data so the heap's base
                // equals end-of-file (see `append_dense_attrs`).
                self.append_dense_attrs(&mut region, dense_attrs)?;
                let oh = build_v2_object_header(&region);
                self.alloc_or_append(&oh)
            }
            CopyTree::DatasetChunked {
                region,
                chunk_dims,
                element_size,
                raw_size,
                maxshape,
                pipeline_message,
                meta,
                chunk_bytes,
                dense_attrs,
            } => self.write_chunked_relocatable(
                region,
                chunk_dims,
                *element_size,
                *raw_size,
                maxshape.as_deref(),
                pipeline_message.as_deref(),
                meta,
                chunk_bytes,
                dense_attrs,
            ),
            CopyTree::Group {
                non_link_region,
                children,
                dense_attrs,
            } => {
                let mut region = non_link_region.clone();
                for (name, child) in children {
                    let new_child = self.write_copy_subtree(child)?;
                    // The link target is stored relative to the userblock base.
                    region.extend_from_slice(&encode_link_message(name, new_child - base));
                }
                // Append the dense heap after the children's headers/data so its
                // base equals end-of-file (see `append_dense_attrs`).
                self.append_dense_attrs(&mut region, dense_attrs)?;
                let oh = build_v2_object_header(&region);
                self.alloc_or_append(&oh)
            }
        }
    }

    /// Write a chunked dataset's storage at end-of-file and return its new
    /// object-header address — the shared write half of a chunked copy
    /// ([`CopyTree::DatasetChunked`]) and a relocating chunked overwrite
    /// ([`MovingWrite::Chunked`]).
    ///
    /// A fresh chunk-data blob and index are laid out relocatably at the current
    /// end-of-file via [`plan_chunked_data_verbatim`] / [`emit_chunked_data_verbatim`],
    /// pulling each chunk's already-compressed bytes from `chunk_bytes` (in dense
    /// row-major grid order) and carrying `meta`'s sizes and filter masks and the
    /// source `pipeline_message` verbatim — no recompression, no filter-parameter
    /// reconstruction. The blob is *appended* (not placed via [`alloc_or_append`])
    /// because its embedded addresses assume `base == end-of-file`, exactly like
    /// [`build_chunked_dataset`](Self::build_chunked_dataset). The verbatim header
    /// `region`'s data-layout message is then swapped for the one the planner
    /// produced (every other message preserved), any dense attribute heap is
    /// appended after the blob, and the header is written into reusable freed space
    /// or at end-of-file.
    #[expect(
        clippy::too_many_arguments,
        reason = "the chunked rebuild needs the full geometry, \
        pipeline, and chunk payloads; bundling them into a struct would only move the list"
    )]
    fn write_chunked_relocatable(
        &mut self,
        region: &[u8],
        chunk_dims: &[u64],
        element_size: usize,
        raw_size: u64,
        maxshape: Option<&[u64]>,
        pipeline_message: Option<&[u8]>,
        meta: &[ChunkMeta],
        chunk_bytes: &[Vec<u8>],
        dense_attrs: &[crate::attribute::AttributeMessage],
    ) -> Result<u64, Error> {
        let eof = self.data.len() as u64;
        // Build with the *stored* (base-relative) address the blob will occupy, so
        // its embedded addresses resolve to its real file offset once the reader adds
        // the userblock base back (see `build_chunked_dataset`). On a base-0 file this
        // equals `eof`.
        let stored_base = eof - self.superblock.base_address;
        let layout = plan_chunked_data_verbatim(
            meta,
            chunk_dims,
            element_size,
            raw_size,
            pipeline_message,
            stored_base,
            maxshape,
        )?;
        let mut buf = Vec::with_capacity(usize::try_from(layout.plan.total_len).unwrap_or(0));
        emit_chunked_data_verbatim(
            &mut buf,
            &layout.plan,
            &SliceChunkProvider {
                chunks: chunk_bytes,
            },
        )?;
        let written = self.append(&buf)?;
        debug_assert_eq!(written, eof, "chunk blob must land at end-of-file",);
        // Swap the data-layout message for the rebuilt one; keep every other header
        // message (datatype, dataspace, fill value, filter pipeline, attributes)
        // verbatim. A dense attribute heap, if any, is appended after the blob so
        // its base equals end-of-file (see `append_dense_attrs`).
        let mut new_region = replace_layout_message(region, &layout.layout_message)?;
        self.append_dense_attrs(&mut new_region, dense_attrs)?;
        let oh = build_v2_object_header(&new_region);
        self.alloc_or_append(&oh)
    }

    /// When `attrs` is non-empty, build a fresh dense (fractal-heap) attribute
    /// blob for it, append it at end-of-file, and splice the matching Attribute
    /// Info message onto `region`. A no-op for an empty set.
    ///
    /// The blob produced by [`file_writer::build_dense_attrs`] is fully
    /// relocatable: every address it embeds is `base + fixed offset`, so passing
    /// the current end-of-file as the base makes those addresses land exactly
    /// where the bytes are written. Like [`build_chunked_dataset`](Self::build_chunked_dataset)
    /// the blob is therefore *appended* (never placed into an interior freed
    /// region), and the caller must append it before any later append in the same
    /// node so `base == end-of-file` still holds. The freshly built heap is
    /// always same-file, so it never aliases the source heap even for an in-file
    /// copy. The caller has already validated [`file_writer::dense_attrs_fit`].
    fn append_dense_attrs(
        &mut self,
        region: &mut Vec<u8>,
        attrs: &[crate::attribute::AttributeMessage],
    ) -> Result<(), Error> {
        if attrs.is_empty() {
            return Ok(());
        }
        let eof = self.data.len() as u64;
        // Build with the *stored* (base-relative) address the blob will occupy, so
        // every address it embeds resolves to its real file offset once the reader
        // adds the userblock base back (see `build_chunked_dataset`). On a base-0
        // file this equals `eof`.
        let stored_base = eof - self.superblock.base_address;
        let blob = crate::file_writer::build_dense_attrs(attrs, stored_base);
        let written = self.append(&blob.blob)?;
        debug_assert_eq!(
            written, eof,
            "dense attribute blob must land at end-of-file",
        );
        region.extend_from_slice(&region_message(
            MessageType::AttributeInfo,
            &blob.attr_info_message,
        ));
        Ok(())
    }

    /// Apply a relocating value overwrite (`write_dataset` resize / compact
    /// rewrite): write the new data and a rewritten object header at end-of-file
    /// (or into reusable freed space) and return the new header address. The
    /// caller patches the parent group's link to this address. The old data
    /// extent (for a resized contiguous dataset) is freed separately, after the
    /// commit's superblock repoint, so it is never reused mid-commit.
    fn write_moving(&mut self, mw: &MovingWrite) -> Result<u64, Error> {
        let base = self.superblock.base_address;
        match mw {
            MovingWrite::Contiguous {
                region,
                addr_off,
                raw,
                ..
            } => {
                let new_data_addr = self.alloc_or_append(raw)?;
                let mut region = region.clone();
                // `alloc_or_append` returns an absolute file offset; the contiguous
                // data-layout field stores it relative to the userblock base (`-
                // base`, a no-op on a base-0 file).
                region[*addr_off..*addr_off + 8]
                    .copy_from_slice(&(new_data_addr - base).to_le_bytes());
                // The data size field follows the 8-byte address in the contiguous
                // layout body; keep it in sync with the new length.
                let size_off = *addr_off + 8;
                region[size_off..size_off + 8].copy_from_slice(&(raw.len() as u64).to_le_bytes());
                let oh = build_v2_object_header(&region);
                self.alloc_or_append(&oh)
            }
            MovingWrite::Compact { region, raw } => {
                let region = rebuild_compact_layout_region(region, raw)?;
                let oh = build_v2_object_header(&region);
                self.alloc_or_append(&oh)
            }
            MovingWrite::Chunked {
                region,
                chunk_dims,
                element_size,
                raw_size,
                maxshape,
                pipeline_message,
                meta,
                chunk_bytes,
                ..
            } => self.write_chunked_relocatable(
                region,
                chunk_dims,
                *element_size,
                *raw_size,
                maxshape.as_deref(),
                pipeline_message.as_deref(),
                meta,
                chunk_bytes,
                &[],
            ),
            MovingWrite::AppendedChunks {
                region,
                new_dataspace_body,
                chunk_dims_u32,
                element_size,
                raw_size,
                has_filters,
                kept_chunks,
                new_chunk_bytes,
                ..
            } => self.write_appended_chunks(
                region,
                new_dataspace_body,
                chunk_dims_u32,
                *element_size,
                *raw_size,
                *has_filters,
                kept_chunks,
                new_chunk_bytes,
            ),
            MovingWrite::AttrEdit {
                region,
                pending_vl_attrs,
            } => {
                // `region` already carries the fixed-size attribute edits (applied
                // in the commit preflight). Place each variable-length attribute's
                // global heap collection, patch its placeholder heap address, and
                // append the resolved message — exactly as the group-attribute apply
                // loop does — then build and place the relocated dataset header. The
                // data-layout message is untouched, so the dataset's chunk data and
                // index stay in place; only the header moves.
                let mut region = region.clone();
                for (msg, collection_bytes) in pending_vl_attrs {
                    let mut msg = msg.clone();
                    let addr = self.place_vl_collection(collection_bytes)?;
                    patch_vl_refs(&mut msg.raw_data, addr);
                    region.extend_from_slice(&region_message(
                        MessageType::Attribute,
                        &msg.serialize(LENGTH_SIZE),
                    ));
                }
                let oh = build_v2_object_header(&region);
                self.alloc_or_append(&oh)
            }
        }
    }

    /// Apply a relocating append ([`MovingWrite::AppendedChunks`]): append the new
    /// (and any rewritten trailing) chunk bytes at end-of-file, rebuild a fresh
    /// Extensible Array over the kept plus appended chunks, grow the dataspace and
    /// repoint the data layout in the verbatim header `region`, and write the
    /// relocated header. Returns the new header address; the caller patches the
    /// parent link. The kept chunk data is untouched (referenced by both the old
    /// and new index during the commit); the old index/header/trailing chunk are
    /// freed only after the superblock repoint.
    #[expect(
        clippy::too_many_arguments,
        reason = "the append rebuild needs the header region, grown dataspace, chunk \
        geometry, and both chunk sets; bundling them into a struct would only move the list"
    )]
    fn write_appended_chunks(
        &mut self,
        region: &[u8],
        new_dataspace_body: &[u8],
        chunk_dims_u32: &[u32],
        element_size: usize,
        raw_size: u64,
        has_filters: bool,
        kept_chunks: &[WrittenChunk],
        new_chunk_bytes: &[Vec<u8>],
    ) -> Result<u64, Error> {
        let base = self.superblock.base_address;
        // Append each new chunk at true end-of-file (never `alloc_or_append`: the
        // rebuilt index below records base-relative addresses computed from the
        // end-of-file the appends land at). Existing chunks keep their in-place
        // addresses and are carried by metadata alone.
        let mut combined: Vec<WrittenChunk> = kept_chunks.to_vec();
        for cb in new_chunk_bytes {
            let abs = self.append(cb)?;
            combined.push(WrittenChunk {
                address: abs - base,
                compressed_size: cb.len() as u64,
                raw_size,
                // This engine applies every filter to a new chunk (no per-chunk
                // skipping), so an appended chunk's mask is always 0. Kept chunks
                // carry their own (possibly nonzero) mask in `combined` already.
                filter_mask: 0,
            });
        }

        // Build the fresh Extensible Array at the current end-of-file. Its embedded
        // block addresses are computed from `ea_base` (base-relative), so appending
        // the blob at the matching file offset makes them resolve correctly, on a
        // userblock (`base != 0`) file too.
        let ea_base = self.data.len() as u64 - base;
        let ea_bytes =
            build_extensible_array_at(&combined, OFFSET_SIZE, LENGTH_SIZE, has_filters, ea_base)
                .map_err(Error::Format)?;
        let written = self.append(&ea_bytes)?;
        debug_assert_eq!(
            written,
            ea_base + base,
            "extensible-array index must land at end-of-file",
        );

        // Swap the dataspace (grown) and data-layout (repointed at the new index)
        // messages; every other header message is preserved verbatim.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "element size is a datatype byte width that fits u32"
        )]
        let layout_body = serialize_v4_extensible_array(
            chunk_dims_u32,
            ea_base,
            OFFSET_SIZE,
            element_size as u32,
        );
        let region = replace_dataspace_message(region, new_dataspace_body)?;
        let region = replace_layout_message(&region, &layout_body)?;
        let oh = build_v2_object_header(&region);
        self.alloc_or_append(&oh)
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

    /// Place an already-built, self-contained global heap collection (from
    /// [`build_global_heap_collection`] or a [`VlStringStaging::collection_bytes`])
    /// and return the base-relative address a variable-length reference into it
    /// should be patched to. A `GCOL` blob embeds no addresses of its own, so it
    /// can be appended (or dropped into reused free space) at any point in the
    /// apply loop, unlike a group or dataset header, which must be built last so
    /// it can name its children's real addresses.
    fn place_vl_collection(&mut self, collection_bytes: &[u8]) -> Result<u64, Error> {
        let addr = self.alloc_or_append(collection_bytes)?;
        Ok(addr - self.superblock.base_address)
    }

    /// Resolve one object-reference element's target to the base-relative
    /// address that should be stored on disk. [`ObjectRefTarget::Raw`] is
    /// written back verbatim (a null or undefined reference is a sentinel, not
    /// a real address, so it needs no base adjustment — mirrors the whole-file
    /// writer). [`ObjectRefTarget::Path`] resolves, in order:
    ///
    /// 1. Against `path_addr` — every group and dataset this commit has
    ///    already placed (a sibling dataset placed earlier in the same
    ///    group's batch — see the apply loop's non-reference-first ordering —
    ///    or a descendant subtree fully processed earlier in the deepest-first
    ///    walk).
    /// 2. Against the pre-commit on-disk file
    ///    ([`resolve_path_any`](crate::group_v2::resolve_path_any)), but only
    ///    when the path is untouched by this commit, so its pre-commit
    ///    address is guaranteed to still be valid post-commit. "Touched"
    ///    means: a dirty group (`nodes`, new or merely rewritten because an
    ///    addition lives under it — its own address changes either way); a
    ///    path this commit adds, or that lies under a subtree this commit
    ///    copies in (`add_targets`, checked by prefix so a copy's interior is
    ///    covered even though only its root is enumerated there); or a
    ///    `write_dataset` target (`write_targets`) — conservatively refused
    ///    even for a same-length overwrite that does not actually relocate,
    ///    since resolving that distinction here is not worth the complexity.
    /// 3. If the path resolves nowhere at all (neither this commit nor the
    ///    pre-commit file has ever heard of it), as an undefined reference
    ///    (`HADDR_UNDEF`) — mirroring [`ObjectRefTarget::Path`]'s existing
    ///    whole-file-writer resolution convention for the same builder type.
    ///
    /// A path that step 1 misses but step 2 identifies as commit-touched is
    /// refused with a clear [`Error::EditUnsupported`] rather than resolved to
    /// a stale or wrong address — the one case this engine cannot resolve
    /// without the whole-file writer's two-pass dummy/real-address scheme.
    /// "Touched" also covers a path this same commit deletes (`pending_deletes`):
    /// without that check the deleted object's pre-commit address would still
    /// resolve via step 2, and the reference would end up pointing at storage
    /// this same commit is about to reclaim and hand out to something else.
    fn resolve_reference_target(
        target: &ObjectRefTarget,
        path_addr: &BTreeMap<PathKey, u64>,
        nodes: &BTreeMap<PathKey, Node>,
        add_targets: &[PathKey],
        write_targets: &[PathKey],
        pending_deletes: &[PathKey],
        data: &[u8],
        superblock: &Superblock,
    ) -> Result<u64, Error> {
        let path = match target {
            ObjectRefTarget::Raw(addr) => return Ok(*addr),
            ObjectRefTarget::Path(path) => path,
        };
        let base = superblock.base_address;
        let key = split_path(path);
        if let Some(&addr) = path_addr.get(&key) {
            return Ok(addr - base);
        }
        if nodes.contains_key(&key)
            || add_targets.iter().any(|t| is_prefix(t, &key))
            || write_targets.contains(&key)
            || pending_deletes.contains(&key)
        {
            return Err(Error::EditUnsupported(
                "an object-reference dataset targets a path this commit is still writing; \
                 use separate commits",
            ));
        }
        match crate::group_v2::resolve_path_any(data, superblock, path) {
            Ok(addr) => Ok(addr - base),
            Err(_) => Ok(UNDEF),
        }
    }

    /// Prove, before any byte of this commit is written, that every
    /// object-reference target across every staged dataset will resolve
    /// successfully — either against a pre-existing untouched object or
    /// against something this same commit places. [`resolve_reference_target`]
    /// classifies a target purely from *whether* a `PathKey` has been placed
    /// yet (`path_addr.get`), never from the address *value*, so replaying the
    /// apply loop's placement order here with placeholder addresses (`0`)
    /// standing in for "already placed" reproduces the exact same verdict the
    /// apply loop's own calls will reach later, without writing anything. If
    /// this preflight pass returns `Ok`, none of the apply loop's own
    /// `resolve_reference_target` calls can fail, so a reference-resolution
    /// error can no longer leave earlier-processed groups' real writes
    /// orphaned in the file (the failure surfaces here instead, before the
    /// apply loop's first `alloc_or_append`/`write_at`).
    fn preflight_reference_targets(
        keys: &[PathKey],
        flat: &BTreeMap<PathKey, Vec<FlatDataset>>,
        nodes: &BTreeMap<PathKey, Node>,
        add_targets: &[PathKey],
        write_targets: &[PathKey],
        pending_deletes: &[PathKey],
        data: &[u8],
        superblock: &Superblock,
    ) -> Result<(), Error> {
        let mut by_depth = keys.to_vec();
        by_depth.sort_by_key(|k| std::cmp::Reverse(k.len()));
        let mut sim_addr: BTreeMap<PathKey, u64> = BTreeMap::new();
        for key in &by_depth {
            if let Some(datasets) = flat.get(key) {
                // Mirrors the apply loop's `group_datasets.sort_by_key(|fd|
                // fd.reference_targets.is_some())`: non-reference datasets are
                // placed (and so become resolvable) before any reference
                // dataset in the same group.
                let mut ordered: Vec<&FlatDataset> = datasets.iter().collect();
                ordered.sort_by_key(|fd| fd.reference_targets.is_some());
                for fd in ordered {
                    if let Some(targets) = &fd.reference_targets {
                        for target in targets {
                            Self::resolve_reference_target(
                                target,
                                &sim_addr,
                                nodes,
                                add_targets,
                                write_targets,
                                pending_deletes,
                                data,
                                superblock,
                            )?;
                        }
                    }
                    let mut full = key.clone();
                    full.push(fd.name.clone());
                    sim_addr.insert(full, 0);
                }
            }
            sim_addr.insert(key.clone(), 0);
        }
        Ok(())
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
        let eof = self.data.len() as u64;
        // The blob embeds *stored* (base-relative) addresses, so the planner base is
        // the stored address the blob will occupy: its end-of-file offset minus the
        // userblock base. The reader recovers each as `stored + base_address`, which
        // resolves back to the blob's real file offset. On a base-0 file this is just
        // `eof`.
        let stored_base = eof - self.superblock.base_address;
        let chunk_dims = fd.chunk_options.resolve_chunk_dims(&fd.ds.dimensions);
        let ctx = ChunkContext::from_datatype(&chunk_dims, &fd.dt);
        let result = build_chunked_data_at_ext(
            &fd.raw,
            &fd.ds.dimensions,
            ctx,
            &fd.chunk_options,
            stored_base,
            fd.maxshape.as_deref(),
        )?;
        // `append` writes at the current end-of-file, which equals `eof`: the blob
        // lands exactly where its embedded (stored) addresses expect once the reader
        // adds the base back.
        let written = self.append(&result.data_bytes)?;
        debug_assert_eq!(written, eof, "chunk blob must land at end-of-file",);
        Ok(build_chunked_dataset_oh(
            &fd.dt,
            &fd.ds,
            &result.layout_message,
            result.pipeline_message.as_deref(),
            &fd.attrs,
            None,
            fd.fill.as_deref(),
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
        let (rs, re) = Self::oh_region(&self.data, addr)?;
        let d = &self.data;
        // A continuation message records the OCHK block's address relative to the
        // userblock base, so it is shifted to an absolute file offset before
        // indexing the file or recording the span (a no-op on a base-0 file). Chunk
        // 0 sits at the absolute header address itself, which is already absolute.
        let base = self.superblock.base_address;
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
                    let off_abs = off
                        .checked_add(base)
                        .ok_or(Error::EditUnsupported("continuation address overflow"))?;
                    let off_us = usize::try_from(off_abs).map_err(|_| {
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
                    spans.push((off_abs, len));
                    chunks.push((off_us + 4, blk_end - 4));
                }
                p = body_end;
            }
        }
        Ok(spans)
    }

    /// Count, for every object-header address reachable from the root, how many
    /// hard links in the *pre-commit* file point to it. The result drives the
    /// last-hard-link reclaim guard in [`collect_free_spans`](Self::collect_free_spans):
    /// an object is freed only when its count is 1.
    ///
    /// Walks the whole link graph from the root, following hard links through
    /// groups of any on-disk format (v0/v1 symbol-table, v2 compact, v2 dense)
    /// via [`resolve_group_entries`], tallying each hard-link edge. Datasets and
    /// other leaves contribute no edges. Returns `None` — so the caller reclaims
    /// nothing for the deletions, a safe leak — if the graph cannot be walked in
    /// full: an unparseable header, a group whose links cannot be enumerated, or
    /// more than [`MAX_LINK_GRAPH_NODES`] objects. Cycles are handled by visiting
    /// each object once. Base-aware: stored child addresses are shifted by the
    /// userblock base, so the returned keys are absolute file offsets.
    fn count_incoming_hard_links(&self) -> Option<HashMap<u64, u32>> {
        let os = self.superblock.offset_size;
        let ls = self.superblock.length_size;
        let base = self.superblock.base_address;
        let mut counts: HashMap<u64, u32> = HashMap::new();
        let mut visited: HashSet<u64> = HashSet::new();
        let mut stack: Vec<u64> = vec![self.superblock.root_group_address];
        let mut budget = MAX_LINK_GRAPH_NODES;
        while let Some(addr) = stack.pop() {
            if !visited.insert(addr) {
                continue; // already expanded (also breaks hard-link cycles)
            }
            if budget == 0 {
                return None; // graph larger than we will walk; leak conservatively
            }
            budget -= 1;
            let off = usize::try_from(addr).ok()?;
            let header = ObjectHeader::parse_with_base(&self.data, off, os, ls, base).ok()?;
            // Datasets and other leaves are not groups and own no links.
            let is_group = header.messages.iter().any(|m| {
                matches!(
                    m.msg_type,
                    MessageType::SymbolTable | MessageType::Link | MessageType::LinkInfo
                )
            });
            if !is_group {
                continue;
            }
            // A group we cannot enumerate fully would undercount incoming links
            // and risk over-reclaim; bail to the safe-leak fallback instead.
            let entries = resolve_group_entries(&self.data, &header, os, ls, base).ok()?;
            for e in entries {
                let child = e.object_header_address.checked_add(base)?;
                *counts.entry(child).or_insert(0) += 1;
                stack.push(child);
            }
        }
        Some(counts)
    }

    /// Best-effort enumeration of every on-disk block owned by the object at
    /// `addr` (and, for a group, its whole subtree), accumulating `(addr, len)`
    /// spans into `out` for reclamation after a delete.
    ///
    /// Contiguous datasets (header + data block), chunked datasets (header +
    /// chunk index + chunk data, via [`chunked_storage_spans`](Self::chunked_storage_spans)),
    /// and whole group subtrees are reclaimed. Deliberately conservative: any
    /// object whose layout it cannot fully account for — a non-v2 header, an
    /// unsupported or only-partially-enumerable chunk index, a group holding a
    /// soft/external link, dense attribute storage — contributes nothing and is
    /// not descended into, so `out` never names a region that might still be in
    /// use. Bounded by [`MAX_COPY_DEPTH`] against a hard-link cycle.
    /// Variable-length data in global-heap collections is never reclaimed here (a
    /// collection can be shared between objects), so it is simply left behind.
    ///
    /// `incoming` is the file-wide hard-link count per object-header address
    /// (from [`count_incoming_hard_links`](Self::count_incoming_hard_links)). An
    /// object is reclaimed — and, for a group, descended into — only when its
    /// count is exactly 1, i.e. the link being removed is its last: an object
    /// still reachable through another hard link is live and is left untouched
    /// (so is everything below a surviving group), which is what keeps deleting
    /// one of several hard links from corrupting the survivor.
    fn collect_free_spans(
        &self,
        addr: usize,
        depth: u32,
        incoming: &HashMap<u64, u32>,
        out: &mut Vec<(u64, u64)>,
    ) {
        // `addr` is an absolute file offset (the caller resolves it from the live
        // file, and the group recursion below re-absolutizes each child). `incoming`
        // is keyed by absolute offset, and `oh_chunk_spans`/`chunked_storage_spans`
        // both take an absolute address and return absolute spans, so the whole
        // walk works in absolute file offsets. The one shift this method must apply
        // itself is on the *stored* (base-relative) addresses `read_object` returns
        // for a contiguous data block and a group's child links: each is converted
        // to an absolute offset by adding `base` (a no-op on a base-0 file) before
        // it is bounds-checked, recorded, or descended into.
        let base = self.superblock.base_address;
        if depth >= MAX_COPY_DEPTH {
            return;
        }
        // Reclaim only when this delete removes the object's last hard link. A
        // count other than 1 (it has surviving links, or the graph walk could
        // not account for it) means the object — and a group's whole subtree —
        // stays live and must not be freed.
        if incoming.get(&(addr as u64)) != Some(&1) {
            return;
        }
        // The header's own chunks. If they cannot be mapped, account for nothing.
        let spans = match self.oh_chunk_spans(addr) {
            Ok(s) => s,
            Err(_) => return,
        };
        match Self::read_object(&self.data, addr, self.superblock.base_address) {
            Ok(ObjModel::DatasetVerbatim { .. }) => out.extend(spans),
            Ok(ObjModel::DatasetContiguous {
                data_addr,
                data_size,
                ..
            }) => {
                out.extend(spans);
                // A defined, in-bounds contiguous data block is owned outright;
                // an empty dataset stores the undefined address and owns none. The
                // stored address is base-relative, so shift it to an absolute file
                // offset before bounds-checking and recording it.
                if data_addr != u64::MAX && data_size > 0 {
                    if let (Some(abs), Ok(len)) =
                        (data_addr.checked_add(base), usize::try_from(data_size))
                    {
                        if let Ok(start) = usize::try_from(abs) {
                            if start.checked_add(len).is_some_and(|e| e <= self.data.len()) {
                                out.push((abs, data_size));
                            }
                        }
                    }
                }
            }
            Ok(ObjModel::Group { children, .. }) => {
                out.extend(spans);
                // Child link targets are stored base-relative; re-absolutize each
                // before descending so the recursion keeps working in absolute
                // offsets (matching `incoming`'s keys and `oh_chunk_spans`).
                for (_, child) in children {
                    if let Some(c) = child
                        .checked_add(base)
                        .and_then(|a| usize::try_from(a).ok())
                    {
                        self.collect_free_spans(c, depth + 1, incoming, out);
                    }
                }
            }
            // A chunked dataset: reclaim its chunk index and chunk data blocks
            // alongside its header. `chunked_storage_spans` returns `None` for
            // anything it cannot account for exhaustively (an index type with no
            // walker, an undefined index address, or spans that fail the
            // bounds/overlap check), leaving the whole dataset as dead bytes
            // rather than freeing a region that might still be in use.
            Ok(ObjModel::DatasetChunked { .. }) => {
                if let Some(storage) = self.chunked_storage_spans(addr) {
                    out.extend(spans);
                    out.extend(storage);
                }
            }
            // A truly unsupported object (one `read_object` cannot model): leave
            // its bytes in place rather than guess its extent.
            Err(_) => {}
        }
    }

    /// Best-effort enumeration of every on-disk block a *chunked* dataset at
    /// `addr` owns: its chunk index structure (B-tree v1 nodes, or fixed- /
    /// extensible-array header, index, super, and data blocks) plus every
    /// allocated chunk data block. The object-header chunks are freed by the
    /// caller ([`collect_free_spans`](Self::collect_free_spans)); this returns
    /// only the storage the data-layout message points at.
    ///
    /// Returns `None` — contribute nothing, leave the object as dead bytes —
    /// whenever the dataset cannot be enumerated *exhaustively* and safely: a
    /// header that does not parse or is not a chunked dataset, a chunk index
    /// with no walker (a version 2 B-tree, index type 5), an undefined index
    /// address (an empty, never-written dataset), or any resulting span that
    /// falls outside the file image or overlaps another. This upholds the
    /// editor's invariant that reclaimed space is never a region still in use:
    /// under-reclaiming only wastes space, while over-reclaiming would corrupt.
    ///
    /// Chunk data addresses and sizes come from the same index walkers the
    /// reader uses, so they match the bytes the writer laid down exactly. The
    /// per-layout enumeration lives in
    /// [`chunked_read::collect_chunked_storage_spans`](crate::chunked_read::collect_chunked_storage_spans);
    /// this method only locates the layout and dataspace messages and validates
    /// the result. Variable-length data in global-heap collections is still
    /// never reclaimed (a collection can be shared between objects); see the
    /// [module docs](self).
    fn chunked_storage_spans(&self, addr: usize) -> Option<Vec<(u64, u64)>> {
        // Locate the data-layout and dataspace messages in the object header.
        let region =
            Self::gather_oh_messages(&self.data, addr, self.superblock.base_address).ok()?;
        let mut layout_msg: Option<(usize, usize)> = None;
        let mut dataspace_msg: Option<(usize, usize)> = None;
        let mut p = 0;
        loop {
            match next_message(&region, p) {
                Ok(Some((msg_type, body, body_end))) => {
                    match msg_type {
                        MessageType::DataLayout => layout_msg = Some((body, body_end)),
                        MessageType::Dataspace => dataspace_msg = Some((body, body_end)),
                        _ => {}
                    }
                    p = body_end;
                }
                Ok(None) => break,
                Err(_) => return None,
            }
        }
        let (lb, le) = layout_msg?;
        let (db, de) = dataspace_msg?;

        let layout = DataLayout::parse(&region[lb..le], OFFSET_SIZE, LENGTH_SIZE).ok()?;
        if !matches!(layout, DataLayout::Chunked { .. }) {
            return None;
        }
        let dataspace = Dataspace::parse(&region[db..de], LENGTH_SIZE).ok()?;

        // Delegate the per-index-type enumeration to the chunked reader (the
        // single owner of chunk-storage layout knowledge), then validate: every
        // span must lie inside the current file image and be pairwise disjoint,
        // or the free list would later hand out live bytes (and a debug build
        // would panic on the double-free). On any error or violation, leave the
        // whole dataset unreclaimed rather than free a region still in use.
        //
        // The layout's stored addresses are relative to the userblock base, so the
        // enumeration runs on a base-relative view of the file and each returned
        // span address is shifted back to an absolute file offset by adding `base`
        // (a no-op on a base-0 file). The free list and the bounds check below both
        // work in absolute file offsets.
        let base = self.superblock.base_address;
        let base_off = usize::try_from(base).ok()?;
        let mut spans = crate::chunked_read::collect_chunked_storage_spans(
            &self.data[base_off..],
            &layout,
            &dataspace,
            OFFSET_SIZE,
            LENGTH_SIZE,
        )
        .ok()?;
        for (addr, _) in &mut spans {
            *addr = addr.checked_add(base)?;
        }
        if !spans_disjoint_in_bounds(&mut spans, self.data.len() as u64) {
            return None;
        }
        Some(spans)
    }

    /// Every on-disk byte span of a chunked dataset's *index structure only* (not
    /// its chunk data), for reclaiming the old index after a relocating append
    /// ([`MovingWrite::AppendedChunks`]) that keeps the chunk data in place. Mirror
    /// of [`chunked_storage_spans`](Self::chunked_storage_spans) but delegating to
    /// [`chunk_index_spans_buffered`], which enumerates only the EA header/index/
    /// data/super blocks and never a chunk-data address, so the shared kept chunk
    /// data is never freed. Base-aware and validated disjoint/in-bounds; returns
    /// `None` (leave unreclaimed) on any error or violation.
    fn chunked_index_spans(&self, addr: usize) -> Option<Vec<(u64, u64)>> {
        let region =
            Self::gather_oh_messages(&self.data, addr, self.superblock.base_address).ok()?;
        let mut layout_msg: Option<(usize, usize)> = None;
        let mut p = 0;
        loop {
            match next_message(&region, p) {
                Ok(Some((msg_type, body, body_end))) => {
                    if msg_type == MessageType::DataLayout {
                        layout_msg = Some((body, body_end));
                    }
                    p = body_end;
                }
                Ok(None) => break,
                Err(_) => return None,
            }
        }
        let (lb, le) = layout_msg?;
        let layout = DataLayout::parse(&region[lb..le], OFFSET_SIZE, LENGTH_SIZE).ok()?;
        if !matches!(layout, DataLayout::Chunked { .. }) {
            return None;
        }
        let base = self.superblock.base_address;
        let base_off = usize::try_from(base).ok()?;
        let mut spans =
            chunk_index_spans_buffered(&self.data[base_off..], &layout, OFFSET_SIZE, LENGTH_SIZE)
                .ok()?;
        for (a, _) in &mut spans {
            *a = a.checked_add(base)?;
        }
        if !spans_disjoint_in_bounds(&mut spans, self.data.len() as u64) {
            return None;
        }
        Some(spans)
    }
}

/// A dirty group in the edit plan: its base object-header message region and the
/// additions targeting it.
#[derive(Default)]
struct Node {
    is_new: bool,
    datasets: Vec<DatasetBuilder>,
    /// Compact group-attribute operations to apply to this group.
    attr_ops: Vec<AttrOp>,
    /// Names of links to remove from this group (from `delete`).
    deletes: Vec<String>,
    /// Copies to add to this group: (new link name, the source subtree read out
    /// for writing). Built at staging time from either this file (an in-file
    /// [`copy`](EditSession::copy)) or another open file (a cross-file
    /// [`copy_from`](EditSession::copy_from)).
    copies: Vec<(String, CopyTree)>,
    /// Value overwrites whose dataset header relocates (a resize or compact
    /// rewrite by `write_dataset`), as (child link name, the relocation plan). On
    /// apply, the new data and header are written and this group's existing link
    /// to the moved header is patched to its new address — exactly like an
    /// existing child group's link.
    writes: Vec<(String, MovingWrite)>,
    base_region: Vec<u8>,
    existing_links: Vec<String>,
    /// Variable-length group/root attributes staged by [`apply_group_attr_ops`],
    /// each still carrying a placeholder heap address: (the attribute message,
    /// its global heap collection bytes). Resolved in the apply loop right
    /// before this node's header is built — [`EditSession::place_vl_collection`]
    /// appends the collection, then the patched message is appended to
    /// `base_region`.
    pending_vl_attrs: PendingVlAttrs,
}

/// A staged compact attribute edit for a group or dataset (shared by
/// [`EditSession::set_group_attr`]/`remove_group_attr` and
/// [`EditSession::set_dataset_attr`]/`remove_dataset_attr`).
enum AttrOp {
    Set { name: String, value: AttrValue },
    Remove { name: String },
}

/// A source object parsed for copying. Headers are reproduced from their
/// verbatim message bytes; only the contiguous data address and child link
/// targets are repointed to the freshly-written copies.
enum ObjModel {
    /// A compact dataset (data inline in the header): copy the region verbatim.
    /// `dense_attrs` is empty unless the source stored its attributes densely, in
    /// which case the Attribute Info message and inline Attribute messages have
    /// been stripped from `region` and the parsed set is carried here to be
    /// re-emitted into a fresh fractal heap on write.
    DatasetVerbatim {
        region: Vec<u8>,
        dense_attrs: Vec<crate::attribute::AttributeMessage>,
    },
    /// A contiguous dataset: copy the region, repointing the data address at
    /// `addr_off` (region-relative) to a fresh copy of `[data_addr, +data_size)`.
    /// See [`DatasetVerbatim`](ObjModel::DatasetVerbatim) for `dense_attrs`.
    DatasetContiguous {
        region: Vec<u8>,
        addr_off: usize,
        data_addr: u64,
        data_size: u64,
        dense_attrs: Vec<crate::attribute::AttributeMessage>,
    },
    /// A chunked (and possibly filtered) dataset: the verbatim header `region`
    /// (datatype, dataspace, fill value, data layout, and filter pipeline kept as
    /// written). The chunk data is not captured here — [`read_copy_subtree`](EditSession::read_copy_subtree)
    /// enumerates and reads the chunks (it holds the source buffer), repointing the
    /// rebuilt index on write. See [`DatasetVerbatim`](ObjModel::DatasetVerbatim)
    /// for `dense_attrs`.
    DatasetChunked {
        region: Vec<u8>,
        dense_attrs: Vec<crate::attribute::AttributeMessage>,
    },
    /// A group: every non-link message verbatim, plus its hard-link children to
    /// copy and re-link by name. See
    /// [`DatasetVerbatim`](ObjModel::DatasetVerbatim) for `dense_attrs`.
    Group {
        non_link_region: Vec<u8>,
        children: Vec<(String, u64)>,
        dense_attrs: Vec<crate::attribute::AttributeMessage>,
    },
}

/// An object subtree fully read out of a source buffer and owning every byte it
/// will write, the read result of [`EditSession::read_copy_subtree`] and the
/// input to [`EditSession::write_copy_subtree`]. Unlike [`ObjModel`] (a single
/// object still referencing source addresses) it is recursive and self-contained:
/// a contiguous dataset owns its data bytes, and a group owns its children, so it
/// can be written into the destination without the source buffer still in hand —
/// which is what lets a cross-file copy read the source at staging time and apply
/// it at commit time.
enum CopyTree {
    /// A compact dataset: the header region is written verbatim (data is inline).
    /// `dense_attrs`, when non-empty, is re-emitted into a freshly built fractal
    /// heap appended just before the header, whose Attribute Info message is
    /// spliced into the region on write.
    DatasetVerbatim {
        region: Vec<u8>,
        dense_attrs: Vec<crate::attribute::AttributeMessage>,
    },
    /// A contiguous dataset: `data` is written first and its new address patched
    /// into the header `region` at `addr_off` before the header is written. See
    /// [`DatasetVerbatim`](CopyTree::DatasetVerbatim) for `dense_attrs`.
    DatasetContiguous {
        region: Vec<u8>,
        addr_off: usize,
        data: Vec<u8>,
        dense_attrs: Vec<crate::attribute::AttributeMessage>,
    },
    /// A chunked (and possibly filtered) dataset. The header `region` is written
    /// verbatim except its data-layout message, which is swapped for one naming the
    /// freshly rebuilt index; `chunk_bytes` (each chunk's already-compressed bytes,
    /// in dense row-major grid order, with sizes/masks in `meta`) and the source
    /// `pipeline_message` are carried unchanged, so the copy preserves the filter
    /// pipeline and chunk payloads byte-for-byte. The on-disk index *type* is
    /// reselected from `maxshape`/chunk count (single / fixed-array / extensible-
    /// array), so a B-tree-v1 or implicit source is reproduced with a v4 index. See
    /// [`DatasetVerbatim`](CopyTree::DatasetVerbatim) for `dense_attrs`.
    DatasetChunked {
        region: Vec<u8>,
        chunk_dims: Vec<u64>,
        element_size: usize,
        raw_size: u64,
        maxshape: Option<Vec<u64>>,
        pipeline_message: Option<Vec<u8>>,
        meta: Vec<ChunkMeta>,
        chunk_bytes: Vec<Vec<u8>>,
        dense_attrs: Vec<crate::attribute::AttributeMessage>,
    },
    /// A group: every non-link message verbatim, plus the (name, child) subtrees
    /// to write first and re-link by name. See
    /// [`DatasetVerbatim`](CopyTree::DatasetVerbatim) for `dense_attrs`.
    Group {
        non_link_region: Vec<u8>,
        children: Vec<(String, CopyTree)>,
        dense_attrs: Vec<crate::attribute::AttributeMessage>,
    },
}

/// The validated, chunk-collapsed message region and existing link names of a
/// group header.
struct GroupInfo {
    region: Vec<u8>,
    link_names: Vec<String>,
}

/// How a staged value overwrite (`write_dataset`) will be applied, decided by
/// [`EditSession::prepare_write`] during the all-or-nothing preflight.
enum WritePlan {
    /// A contiguous dataset whose new data is the same length as its existing,
    /// defined data block: overwrite the bytes straight in place at `data_addr`.
    /// No object header is rewritten and the superblock root is not flipped.
    InPlace { data_addr: usize, raw: Vec<u8> },
    /// A chunked dataset overwritten chunk-by-chunk in place: each `(addr, bytes)`
    /// pair is written straight over an existing chunk slot. Used when every new
    /// (re-encoded) chunk is the same byte length as the slot it replaces — an
    /// unfiltered chunked overwrite (chunk sizes are fixed by the unchanged shape)
    /// or a filtered one whose re-encoded chunks happen to match. Like
    /// [`InPlace`](WritePlan::InPlace) it touches no header and no chunk index, so
    /// the superblock root is not flipped.
    InPlaceChunks { writes: Vec<(usize, Vec<u8>)> },
    /// The dataset's header relocates: a contiguous resize, a compact rewrite, or
    /// a chunked rebuild. The parent group is rebuilt and its link patched. See
    /// [`MovingWrite`].
    Moving(MovingWrite),
}

/// A value overwrite that relocates the dataset's object header — a contiguous
/// dataset whose data length changed (or had no data block) or a compact dataset
/// whose inline bytes are replaced. On apply the new data and a rewritten header
/// are written at end-of-file (or into reusable freed space), and the parent
/// group's link is repointed at the new header address.
enum MovingWrite {
    /// A contiguous dataset: write `raw` elsewhere, patch the data-layout address
    /// at `addr_off` in the verbatim header `region`, rewrite the header, and free
    /// `old_extent` (the prior data block, if any) after the commit lands.
    Contiguous {
        region: Vec<u8>,
        addr_off: usize,
        raw: Vec<u8>,
        old_extent: Option<(u64, u64)>,
    },
    /// A compact dataset: rebuild the header `region` with `raw` inline.
    Compact { region: Vec<u8>, raw: Vec<u8> },
    /// A chunked dataset whose new (re-encoded) chunks do not all fit their
    /// existing slots, so its whole storage is rebuilt and relocated. A fresh
    /// chunk-data blob and index are appended at end-of-file (via the verbatim
    /// layout path, carrying `chunk_bytes` and the source filter `pipeline_message`
    /// unchanged — no recompression and no filter-parameter reconstruction), the
    /// data-layout message in the verbatim header `region` is swapped for the new
    /// one (every other header message — datatype, dataspace, fill value, filter
    /// pipeline, and attributes, including a dense attribute heap referenced by an
    /// untouched Attribute Info message — is preserved verbatim), and the old
    /// chunk storage at `old_addr` is freed after the commit lands.
    Chunked {
        region: Vec<u8>,
        chunk_dims: Vec<u64>,
        element_size: usize,
        raw_size: u64,
        maxshape: Option<Vec<u64>>,
        pipeline_message: Option<Vec<u8>>,
        meta: Vec<ChunkMeta>,
        chunk_bytes: Vec<Vec<u8>>,
        old_addr: u64,
    },
    /// A relocating **append** to a chunked, unlimited, Extensible-Array-indexed
    /// dataset (`append_dataset`). The dataset's existing chunk *data* stays in
    /// place; only the newly-appended chunks and any rewritten trailing partial
    /// chunk (`new_chunk_bytes`, already compressed through the on-disk pipeline)
    /// are appended at end-of-file, a fresh Extensible Array is rebuilt over
    /// `kept_chunks ++ new_chunk_bytes`, the verbatim header `region`'s dataspace
    /// message is grown (`new_dataspace_body`) and its data-layout message
    /// repointed at the new index (every other message — datatype, filter
    /// pipeline, fill value, attributes — preserved verbatim), and the header is
    /// relocated. After the commit lands, only the old index structure at
    /// `old_addr`, the old header, and the relocated old trailing chunk
    /// (`old_tail_extent`) are freed — never the kept chunk data, which both the
    /// old and new index share during the commit.
    AppendedChunks {
        region: Vec<u8>,
        /// The grown dataspace message body (v2-serialized), current axis-0
        /// dimension increased, maximum dimensions (unlimited) preserved.
        new_dataspace_body: Vec<u8>,
        /// Rank-only spatial chunk dimensions, for the rebuilt v4 layout message.
        chunk_dims_u32: Vec<u32>,
        element_size: usize,
        /// Full (uncompressed) chunk byte size = product(spatial) * element_size.
        raw_size: u64,
        has_filters: bool,
        /// Existing complete chunks, in index order, carried by metadata alone —
        /// their base-relative addresses, on-disk stored sizes, and filter masks
        /// preserved exactly (a nonzero mask from a C/h5py-skipped filter is kept).
        kept_chunks: Vec<WrittenChunk>,
        /// The appended chunks in index order: the recompressed trailing partial
        /// chunk first (when present), then the remaining new full chunks.
        new_chunk_bytes: Vec<Vec<u8>>,
        /// The dataset header address, for old-index and old-header reclaim.
        old_addr: u64,
        /// The absolute `(addr, len)` of the old trailing partial chunk's data
        /// block when it was rewritten, freed after the commit lands. `None` when
        /// the append was chunk-aligned (no partial chunk to rewrite).
        old_tail_extent: Option<(u64, u64)>,
    },
    /// A compact dataset-attribute edit (`set_dataset_attr` / `remove_dataset_attr`).
    /// The verbatim header `region` already carries the fixed-size attribute change
    /// (applied by [`apply_group_attr_ops`] in the commit preflight); any
    /// variable-length attribute is placed and patched in [`EditSession::write_moving`]
    /// via `pending_vl_attrs`. The rewritten header is relocated and the parent link
    /// repointed, exactly like the other relocating writes — but the data-layout
    /// message is preserved verbatim, so the dataset's chunk data and index stay in
    /// place; only the old header is freed.
    AttrEdit {
        region: Vec<u8>,
        pending_vl_attrs: PendingVlAttrs,
    },
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
    /// Variable-length attributes still carrying a placeholder heap address:
    /// (index into `attrs`, that attribute's global heap collection bytes).
    /// Resolved in the apply loop right before this dataset's header is built.
    vl_attrs: Vec<(usize, Vec<u8>)>,
    /// A staged variable-length-string dataset's element references (still
    /// carrying placeholder heap addresses in `raw`) and global heap collection.
    /// Resolved in the apply loop right before `raw` is appended.
    vl_string_staging: Option<VlStringStaging>,
    /// An object-reference dataset's per-element targets, still unresolved.
    /// Resolved (see [`EditSession::resolve_reference_target`]) and patched
    /// into `raw` in the apply loop, once every object this commit places has
    /// a known address. `None` for an ordinary dataset.
    reference_targets: Option<Vec<ObjectRefTarget>>,
    /// A user-defined fill value, encoded in the dataset's datatype, or `None`
    /// for the library default. Validated against the datatype element size in
    /// [`flatten_dataset`].
    fill: Option<Vec<u8>>,
}

/// A borrow adapter that drives the shared Extensible-Array append engine
/// ([`crate::chunk_index_inplace`]) against an [`EditSession`]'s *own* mirror,
/// handle, and superblock, so a session runs an immediate O(1) in-place append
/// without constructing a second `InPlaceFile` (which would take a second exclusive
/// lock and keep a divergent mirror). It borrows only the mirror-carrying fields,
/// leaving [`EditSession::located`] independently borrowable. Its primitives write
/// to disk *before* the mirror — the session's discipline, the opposite of the
/// append/SWMR writers' mirror-before-disk order; the [`InPlaceBytes`] trait lets
/// each owner keep its own failure-path discipline while sharing the checksummed
/// slot/block mechanics.
struct EditMirror<'a> {
    handle: &'a mut fs::File,
    data: &'a mut Vec<u8>,
    superblock: &'a mut Superblock,
    sb_sig_off: usize,
}

impl InPlaceBytes for EditMirror<'_> {
    fn data(&self) -> &[u8] {
        &self.data[..]
    }
    fn offset_size(&self) -> u8 {
        self.superblock.offset_size
    }
    fn length_size(&self) -> u8 {
        self.superblock.length_size
    }
    fn superblock(&self) -> &Superblock {
        &*self.superblock
    }
    fn append_bytes(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        // Disk before mirror, so a failed write never leaves the mirror ahead of
        // the file (matching `EditSession::append`).
        let addr = self.data.len() as u64;
        self.handle.seek(SeekFrom::Start(addr)).map_err(Error::Io)?;
        self.handle.write_all(bytes).map_err(Error::Io)?;
        self.data.extend_from_slice(bytes);
        Ok(addr)
    }
    fn write_at(&mut self, offset: usize, bytes: &[u8]) -> Result<(), Error> {
        // Disk before mirror (see `append_bytes`).
        self.handle
            .seek(SeekFrom::Start(offset as u64))
            .map_err(Error::Io)?;
        self.handle.write_all(bytes).map_err(Error::Io)?;
        self.data[offset..offset + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }
    fn patch_superblock_eof(&mut self) -> Result<(), Error> {
        // Advance only the recorded end-of-file and re-serialize the superblock in
        // place. Unlike `EditSession::commit`, this deliberately does NOT clear the
        // consistency flags and does NOT repoint the root group: base_address is 0
        // for every in-place-append-eligible file, so the normalized-absolute root
        // address serializes back to the same stored value.
        let eof = self.data.len() as u64;
        self.superblock.eof_address = eof;
        let bytes = self.superblock.serialize();
        self.write_at(self.sb_sig_off, &bytes)
    }
    fn sync(&mut self) -> Result<(), Error> {
        self.handle.flush().map_err(Error::Io)?;
        self.handle.sync_data().map_err(Error::Io)?;
        Ok(())
    }
}

/// Whether two object paths are equal or one is an ancestor of the other.
fn paths_overlap(a: &[String], b: &[String]) -> bool {
    a.starts_with(b) || b.starts_with(a)
}

/// Re-tag a refusal from the shared append engine (`AppendUnsupported`) as the
/// fast-path [`Error::AppendInPlaceUnsupported`], so a caller can catch it and fall
/// back to the staged [`append_dataset`](EditSession::append_dataset) — which
/// handles the non-chunk-aligned filtered case, index-geometry limits, and
/// platform-width limits that the engine reports this way. Genuine I/O and format
/// errors pass through unchanged.
fn as_inplace_error(e: Error) -> Error {
    match e {
        Error::AppendUnsupported(m) => Error::AppendInPlaceUnsupported(m),
        other => other,
    }
}

/// Locate `dataset` in `file` and build its [`LocatedState`], validating in-place
/// append eligibility (rank-1 / unlimited / Extensible-Array indexed, a nonzero
/// chunk length, and a re-encodable filter pipeline). Mirrors the append writer's
/// `ensure_located`, reporting through [`Error::AppendInPlaceUnsupported`].
fn locate_dataset_state<F: InPlaceBytes>(file: &F, dataset: &str) -> Result<LocatedState, Error> {
    let result = Located::locate(file, dataset, Error::AppendInPlaceUnsupported)?;
    if result.located.chunk_elems == 0 {
        return Err(Error::AppendInPlaceUnsupported(
            "in-place append requires a nonzero chunk length",
        ));
    }
    let data = file.data();
    let (dt_off, dt_size) = result.spans.datatype;
    let (datatype, _) = Datatype::parse(&data[dt_off..dt_off + dt_size])
        .map_err(|_| Error::AppendInPlaceUnsupported("dataset datatype could not be parsed"))?;
    let pipeline = match result.spans.filter {
        Some((fb, fsize)) => {
            let parsed = FilterPipeline::parse(&data[fb..fb + fsize]).map_err(|_| {
                Error::AppendInPlaceUnsupported("dataset filter pipeline could not be parsed")
            })?;
            if !pipeline_reencodable(&parsed) {
                return Err(Error::AppendInPlaceUnsupported(
                    "dataset uses a filter this engine cannot re-encode",
                ));
            }
            Some(parsed)
        }
        None => None,
    };
    let element_size = result.located.elem_bytes;
    let spatial = vec![result.located.chunk_elems];
    Ok(LocatedState {
        loc: result.located,
        datatype,
        spatial,
        element_size,
        pipeline,
    })
}

/// Generate the typed `append_inplace_*` methods, mirroring [`AppendBuilder`]'s
/// vocabulary: each gathers into a builder and applies the append immediately.
macro_rules! append_inplace_typed {
    ($($method:ident, $builder:ident, $ty:ty;)*) => {
        impl EditSession {
            $(
                #[doc = concat!("Append `", stringify!($ty), "` values to `dataset` in \
                    place. See [`append_inplace`](Self::append_inplace) for the contract.")]
                pub fn $method(&mut self, dataset: &str, data: &[$ty]) -> Result<(), Error> {
                    let mut b = AppendBuilder::new();
                    b.$builder(data);
                    self.append_inplace_gathered(dataset, &b, 4)
                }
            )*
        }
    };
}

append_inplace_typed! {
    append_inplace_f64, append_f64, f64;
    append_inplace_f32, append_f32, f32;
    append_inplace_i8, append_i8, i8;
    append_inplace_i16, append_i16, i16;
    append_inplace_i32, append_i32, i32;
    append_inplace_i64, append_i64, i64;
    append_inplace_u8, append_u8, u8;
    append_inplace_u16, append_u16, u16;
    append_inplace_u32, append_u32, u32;
    append_inplace_u64, append_u64, u64;
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

/// Validate that every reclaim span `(addr, len)` is non-empty, ends at or
/// before `eof`, and that no two overlap; sorts `spans` by address as a side
/// effect. Returns `false` on any violation so the caller can decline to
/// reclaim the object rather than feed the free list an out-of-bounds or
/// overlapping (double-free) region. Touching spans are allowed — the free list
/// coalesces them.
fn spans_disjoint_in_bounds(spans: &mut [(u64, u64)], eof: u64) -> bool {
    for &(addr, len) in spans.iter() {
        match addr.checked_add(len) {
            Some(end) if len > 0 && end <= eof => {}
            _ => return false,
        }
    }
    spans.sort_unstable_by_key(|&(addr, _)| addr);
    spans.windows(2).all(|w| w[0].0 + w[0].1 <= w[1].0)
}

/// Sanitize the accumulated free spans for a whole commit so the free list never
/// sees an out-of-bounds or overlapping (double-free) region: drop empty or
/// past-`eof` spans, sort by address, then drop any span overlapping one already
/// kept. Dropping only leaks (the bytes stay allocated); it never frees a live
/// region. With the last-hard-link guard in force nothing should be dropped for
/// a well-formed file — this is a backstop, not the primary defense.
fn retain_disjoint_in_bounds(spans: &mut Vec<(u64, u64)>, eof: u64) {
    spans.retain(|&(addr, len)| len > 0 && addr.checked_add(len).is_some_and(|e| e <= eof));
    spans.sort_unstable_by_key(|&(addr, _)| addr);
    let mut kept_end = 0u64;
    spans.retain(|&(addr, len)| {
        if addr >= kept_end {
            kept_end = addr + len;
            true
        } else {
            false // overlaps a span already kept; leak it rather than double-free
        }
    });
}

/// Validate a staged dataset and reduce it to a [`FlatDataset`]. Contiguous,
/// unfiltered datasets are emitted as such; chunked, filtered, or extensible
/// datasets carry their [`ChunkOptions`] and maxshape through to the commit,
/// where [`build_chunked_data_at_ext`] lays out their chunk data and index. An
/// empty (zero-element) shape is allowed for contiguous storage (mirroring the
/// whole-file writer, its data address is `HADDR_UNDEF` — see the apply loop),
/// but chunking one stays refused via the geometry validation below. A
/// `provenance` dataset has its SHA-256/creator/timestamp/source attributes
/// computed here from `raw`, exactly as the whole-file writer does. A
/// variable-length attribute's global heap collection is built here (it is
/// fully self-contained — no address of its own) but placed and patched later,
/// in the apply loop, once its final address is known; likewise a
/// variable-length-string dataset's staged references and collection
/// (`db.vl_string_staging`) are carried through unresolved. An object-reference
/// dataset's per-element targets (`db.reference_targets`) are likewise carried
/// through unresolved — resolving a path target requires knowing every other
/// object this commit places, which is only known well into the apply loop
/// (see [`EditSession::resolve_reference_target`]). Rejects any remaining
/// feature this engine cannot reproduce faithfully: dense attributes, a
/// chunked/extensible variable-length-string or object-reference dataset, or a
/// filter pipeline the build cannot construct.
fn flatten_dataset(db: DatasetBuilder) -> Result<FlatDataset, Error> {
    if db.name.is_empty() {
        return Err(Error::EditUnsupported("dataset path has an empty name"));
    }
    let dt = db
        .datatype
        .ok_or(Error::EditUnsupported("dataset has no datatype/data"))?;
    let shape = db
        .shape
        .ok_or(Error::EditUnsupported("dataset has no shape"))?;
    let is_empty = shape.contains(&0);
    let chunked = db.chunk_options.is_chunked() || db.maxshape.is_some();
    if is_empty && chunked {
        return Err(Error::EditUnsupported(
            "chunked or extensible empty (zero-element) datasets cannot be added in place yet",
        ));
    }
    // Variable-length string element references live in the global heap, whose
    // address is only known once the apply loop places the collection. For
    // chunked/filtered/resizable storage the references sit inside chunks
    // written before that address exists, so patching them in is impossible —
    // mirrors the whole-file writer's `ChunkedVlenStringUnsupported` refusal.
    if db.vl_string_staging.is_some() && chunked {
        return Err(Error::EditUnsupported(
            "chunked or extensible variable-length-string datasets cannot be added in place yet",
        ));
    }
    // Object-reference elements are resolved (see `resolve_reference_target`)
    // and patched into `raw` right before it is appended; for chunked storage
    // that patch would need to reach inside already-built chunk data, which
    // this engine does not support (mirrors the variable-length-string
    // refusal above — untested and unneeded combination for v1).
    if db.reference_targets.is_some() && chunked {
        return Err(Error::EditUnsupported(
            "chunked or extensible object-reference datasets cannot be added in place yet",
        ));
    }
    let raw = if is_empty {
        db.data.unwrap_or_default()
    } else {
        db.data
            .ok_or(Error::EditUnsupported("dataset has no data"))?
    };

    let elem = dt.type_size() as u64;
    if elem > 0 {
        // Multiply with checked arithmetic: an absurd shape whose element count
        // (or byte size) overflows `u64` is refused rather than panicking in a
        // debug build or silently wrapping in release (which could let a wrapped
        // product spuriously match `raw.len()`). For a zero-element shape this
        // expected length is always 0 (a `0` dimension makes every checked
        // multiplication `Some(0)` regardless of the other dimensions), so this
        // also catches data mistakenly supplied for a shape that holds nothing.
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
    let mut attrs: Vec<crate::attribute::AttributeMessage> = Vec::with_capacity(db.attrs.len());
    for (n, v) in &db.attrs {
        attrs.push(build_attr_message(n, v));
    }
    // `build_attr_message` already writes a placeholder (heap address 0) for a
    // `VarLenAsciiArray` attribute; stage its self-contained global heap
    // collection here (no address of its own to resolve yet) and record which
    // `attrs` slot it patches once the apply loop places it.
    let vl_attrs: Vec<(usize, Vec<u8>)> = db
        .attrs
        .iter()
        .enumerate()
        .filter_map(|(i, (_, v))| match v {
            AttrValue::VarLenAsciiArray(strings) => {
                let str_refs: Vec<&str> = strings.iter().map(String::as_str).collect();
                Some((i, build_global_heap_collection(&str_refs)))
            }
            _ => None,
        })
        .collect();
    #[cfg(feature = "provenance")]
    if let Some(ref prov) = db.provenance {
        let p = crate::provenance::Provenance {
            creator: prov.creator.clone(),
            timestamp: prov.timestamp.clone(),
            source: prov.source.clone(),
        };
        attrs.extend(p.build_attrs(&raw));
    }
    // The object-header message-size field is 2 bytes wide, so an oversized
    // attribute (most reachable via a `VarLenAsciiArray` with many/long
    // strings) would silently truncate and corrupt the header if written
    // as-is; refuse it instead, mirroring `apply_group_attr_ops`'s and
    // `encode_attr_message`'s equivalent checks for group/root attributes.
    for a in &attrs {
        if a.serialize(LENGTH_SIZE).len() > u16::MAX as usize {
            return Err(Error::EditUnsupported(
                "dataset attribute is too large to encode in place",
            ));
        }
    }
    if attrs.len() > MAX_COMPACT_ATTRS {
        return Err(Error::EditUnsupported(
            "datasets with dense (many) attributes cannot be added in place yet",
        ));
    }

    // A user-defined fill value is one element wide, so its byte length must
    // equal the datatype's element size (mirrors the whole-file writer's check).
    if let Some(fill) = &db.fill {
        let expected = elem.to_usize()?;
        if fill.len() != expected {
            return Err(Error::Format(FormatError::FillValueSizeMismatch {
                expected,
                actual: fill.len(),
            }));
        }
    }

    Ok(FlatDataset {
        name: db.name,
        dt,
        ds,
        raw,
        attrs,
        chunk_options: db.chunk_options,
        maxshape: db.maxshape,
        vl_attrs,
        vl_string_staging: db.vl_string_staging,
        reference_targets: db.reference_targets,
        fill: db.fill,
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
/// Whether a chunked dataset with this data-layout version and chunk index type
/// can be enumerated chunk-by-chunk (and therefore overwritten or copied in
/// place). Mirrors the dispatch in
/// [`chunked_read::collect_chunks_for_layout_from_source`](crate::chunked_read):
/// version-3 B-tree v1 and the version-4 single / implicit / fixed-array /
/// extensible-array indexes have walkers; a version-2 B-tree (index type 5) or
/// any unknown index type does not.
fn chunk_index_enumerable(version: u8, chunk_index_type: Option<u8>) -> bool {
    matches!((version, chunk_index_type), (3, _) | (4, Some(1..=4)))
}

/// Whether every filter in `pipeline` is one this crate can *apply* (re-encode a
/// chunk through) — not merely decode. A pipeline with any other filter cannot be
/// re-encoded for an in-place overwrite, so the caller refuses with a typed error
/// rather than letting [`compress_chunk`] surface a raw `UnsupportedFilter`.
pub(crate) fn pipeline_reencodable(pipeline: &FilterPipeline) -> bool {
    pipeline.filters.iter().all(|f| match f.filter_id {
        FILTER_DEFLATE | FILTER_SHUFFLE | FILTER_FLETCHER32 | FILTER_SCALEOFFSET => true,
        #[cfg(feature = "zfp")]
        crate::filter_pipeline::FILTER_ZFP => true,
        _ => false,
    })
}

/// Rebuild a header message `region`, replacing the single Data Layout message's
/// record with one carrying `new_layout_body` and leaving every other message
/// (datatype, dataspace, fill value, filter pipeline, attributes, attribute info)
/// byte-for-byte. The replacement may differ in length from the original — a
/// chunked rebuild can change the index type and thus the layout message size — so
/// the record is rebuilt via [`region_message`] rather than patched in place. The
/// chunked overwrite and copy paths use this to relocate a dataset's chunk storage
/// while preserving the rest of its header exactly.
fn replace_layout_message(region: &[u8], new_layout_body: &[u8]) -> Result<Vec<u8>, Error> {
    let mut out = Vec::with_capacity(region.len());
    let mut p = 0;
    let mut replaced = false;
    while let Some((msg_type, _body, body_end)) = next_message(region, p)? {
        if msg_type == MessageType::DataLayout && !replaced {
            out.extend_from_slice(&region_message(MessageType::DataLayout, new_layout_body));
            replaced = true;
        } else {
            out.extend_from_slice(&region[p..body_end]);
        }
        p = body_end;
    }
    if !replaced {
        return Err(Error::EditUnsupported(
            "chunked dataset header has no data-layout message to relocate",
        ));
    }
    Ok(out)
}

/// Rebuild a header message `region`, replacing the single Dataspace message's
/// record with one carrying `new_dataspace_body` (the grown current dimensions,
/// v2-serialized, maximum dimensions preserved) and leaving every other message
/// byte-for-byte. Used by the append path to grow a dataset's axis-0 dimension.
/// The replacement may differ in length from the original (a v1 on-disk
/// dataspace is normalized to v2 in the rebuilt header), so the record is rebuilt
/// via [`region_message`] rather than patched in place.
fn replace_dataspace_message(region: &[u8], new_dataspace_body: &[u8]) -> Result<Vec<u8>, Error> {
    let mut out = Vec::with_capacity(region.len());
    let mut p = 0;
    let mut replaced = false;
    while let Some((msg_type, _body, body_end)) = next_message(region, p)? {
        if msg_type == MessageType::Dataspace && !replaced {
            out.extend_from_slice(&region_message(MessageType::Dataspace, new_dataspace_body));
            replaced = true;
        } else {
            out.extend_from_slice(&region[p..body_end]);
        }
        p = body_end;
    }
    if !replaced {
        return Err(Error::AppendUnsupported(
            "dataset header has no dataspace message to grow",
        ));
    }
    Ok(out)
}

/// Whether a datatype's raw on-disk bytes can be appended verbatim from a caller
/// via [`AppendBuilder::append_raw`]. True only when every scalar leaf is safe to
/// write as flat little-endian bytes:
///
/// - numeric leaves (fixed-point, floating-point, time, bit field) must be
///   little-endian, or the caller's little-endian bytes would silently misencode
///   into a big-endian (or VAX) field;
/// - string and opaque leaves are byte arrays with no numeric byte order, so they
///   are order-agnostic and safe;
/// - aggregates (enumeration, array, compound) are appendable iff every leaf is;
/// - variable-length and reference leaves embed global-heap or object addresses
///   that a flat byte append cannot reproduce, so they are never raw-appendable.
///
/// A typed `append_*` bypasses this: it checks full datatype equality instead, so
/// it already refuses every non-little-endian and non-scalar dataset.
pub(crate) fn datatype_is_raw_appendable(dt: &Datatype) -> bool {
    match dt {
        Datatype::FixedPoint { byte_order, .. }
        | Datatype::FloatingPoint { byte_order, .. }
        | Datatype::Time { byte_order, .. }
        | Datatype::BitField { byte_order, .. } => *byte_order == DatatypeByteOrder::LittleEndian,
        Datatype::String { .. } | Datatype::Opaque { .. } => true,
        Datatype::Enumeration { base_type, .. } | Datatype::Array { base_type, .. } => {
            datatype_is_raw_appendable(base_type)
        }
        Datatype::Compound { members, .. } => members
            .iter()
            .all(|m| datatype_is_raw_appendable(&m.datatype)),
        Datatype::VariableLength { .. } | Datatype::Reference { .. } => false,
    }
}

/// The datatype, dataspace, parsed chunked data layout, and verbatim filter-
/// pipeline message bytes (if any) of a chunked dataset header, parsed by
/// [`parse_chunked_header`].
struct ChunkedHeaderParts {
    dt: crate::datatype::Datatype,
    ds: Dataspace,
    layout: DataLayout,
    pipeline_message: Option<Vec<u8>>,
}

/// Parse the datatype, dataspace, chunked data layout, and verbatim filter-
/// pipeline message bytes (if any) from a chunked dataset header `region`. Used by
/// the chunked copy path to derive chunk geometry and the on-disk filter pipeline.
/// Errors if any required message is missing or the layout is not chunked.
fn parse_chunked_header(region: &[u8]) -> Result<ChunkedHeaderParts, Error> {
    let mut datatype: Option<(usize, usize)> = None;
    let mut dataspace: Option<(usize, usize)> = None;
    let mut layout: Option<(usize, usize)> = None;
    let mut pipeline: Option<(usize, usize)> = None;
    let mut p = 0;
    while let Some((msg_type, body, body_end)) = next_message(region, p)? {
        match msg_type {
            MessageType::Datatype => datatype = Some((body, body_end)),
            MessageType::Dataspace => dataspace = Some((body, body_end)),
            MessageType::DataLayout => layout = Some((body, body_end)),
            MessageType::FilterPipeline => pipeline = Some((body, body_end)),
            _ => {}
        }
        p = body_end;
    }
    let (dt_b, dt_e) = datatype.ok_or(Error::EditUnsupported("dataset header has no datatype"))?;
    let (ds_b, ds_e) =
        dataspace.ok_or(Error::EditUnsupported("dataset header has no dataspace"))?;
    let (lb, le) = layout.ok_or(Error::EditUnsupported("dataset header has no data layout"))?;
    let (dt, _) = crate::datatype::Datatype::parse(&region[dt_b..dt_e])
        .map_err(|_| Error::EditUnsupported("dataset header datatype could not be parsed"))?;
    let ds = Dataspace::parse(&region[ds_b..ds_e], LENGTH_SIZE)
        .map_err(|_| Error::EditUnsupported("dataset header dataspace could not be parsed"))?;
    let dl = DataLayout::parse(&region[lb..le], OFFSET_SIZE, LENGTH_SIZE)
        .map_err(|_| Error::EditUnsupported("dataset header data layout could not be parsed"))?;
    if !matches!(dl, DataLayout::Chunked { .. }) {
        return Err(Error::EditUnsupported("dataset is not chunked"));
    }
    let pipeline_message = pipeline.map(|(b, e)| region[b..e].to_vec());
    Ok(ChunkedHeaderParts {
        dt,
        ds,
        layout: dl,
        pipeline_message,
    })
}

/// The chunk geometry a verbatim chunked rebuild needs, derived by
/// [`chunked_geometry`] from a chunked dataset's datatype, dataspace, and parsed
/// [`DataLayout::Chunked`].
struct ChunkedGeometry {
    /// Rank-only spatial chunk dimensions.
    spatial: Vec<u64>,
    /// Element size in bytes.
    element_size: usize,
    /// Full (uncompressed) chunk byte size, `product(spatial) * element_size`.
    raw_size: u64,
    /// The on-disk maximum dimensions when they differ from the current shape; an
    /// unlimited dimension selects the extensible-array index, a finite one the
    /// fixed-array index. `None` keeps the fixed-array / single-chunk index.
    maxshape: Option<Vec<u64>>,
}

/// Derive the [`ChunkedGeometry`] for a chunked dataset from its datatype,
/// dataspace, and parsed [`DataLayout::Chunked`].
fn chunked_geometry(
    dt: &crate::datatype::Datatype,
    ds: &Dataspace,
    layout: &DataLayout,
) -> Result<ChunkedGeometry, Error> {
    let DataLayout::Chunked {
        chunk_dimensions, ..
    } = layout
    else {
        return Err(Error::EditUnsupported("dataset is not chunked"));
    };
    let rank = ds.dimensions.len();
    if chunk_dimensions.len() <= rank {
        return Err(Error::EditUnsupported(
            "chunked layout has malformed dimensions",
        ));
    }
    let spatial: Vec<u64> = chunk_dimensions[..rank]
        .iter()
        .map(|&c| u64::from(c))
        .collect();
    let element_size = dt.type_size() as usize;
    if element_size == 0 {
        return Err(Error::EditUnsupported(
            "chunked dataset has a zero element size",
        ));
    }
    let raw_size = spatial
        .iter()
        .copied()
        .product::<u64>()
        .saturating_mul(element_size as u64);
    let maxshape = ds
        .max_dimensions
        .as_ref()
        .filter(|ms| *ms != &ds.dimensions)
        .cloned();
    Ok(ChunkedGeometry {
        spatial,
        element_size,
        raw_size,
        maxshape,
    })
}

/// Try to overwrite a chunked dataset's chunks in place. When the dataset's
/// on-disk chunks form a dense grid aligned with `new_bytes` (dense row-major
/// order), every slot is unmasked (`filter_mask == 0`), and every new chunk
/// **fits** the slot it replaces (`new_len <= slot`), return the in-place
/// `(address, bytes)` writes:
///
/// - When every new chunk is **exactly** its slot's size, only the chunk data is
///   written; the index is untouched (so any enumerable index type works, and a
///   crash can tear at most a chunk's value bytes, not the structure).
/// - When some new chunks are **smaller** (fit with slack), the chunk index
///   records each chunk's stored size, so the index is rebuilt in place to record
///   the new sizes (see [`try_rebuild_index_in_place`]). This is supported only
///   for a v4 fixed-array or extensible-array index occupying a single contiguous
///   on-disk region; any other case returns `None` to relocate.
///
/// Returns `None` — so the caller relocates the dataset instead — when the index
/// cannot be enumerated, the grid is sparse, a slot is masked, a new chunk does
/// not fit, the index cannot be rebuilt in place, or any write would be out of
/// bounds or overlap another.
fn try_inplace_chunk_writes(
    d: &[u8],
    layout: &DataLayout,
    ds: &Dataspace,
    spatial: &[u64],
    raw_size: u64,
    new_bytes: &[Vec<u8>],
) -> Option<Vec<(usize, Vec<u8>)>> {
    let infos = enumerate_chunks_buffered(d, layout, ds, OFFSET_SIZE, LENGTH_SIZE).ok()?;
    let grid = plan_dense_grid(infos, &ds.dimensions, spatial)?;
    if grid.grid_order.len() != new_bytes.len() {
        return None;
    }
    let mut writes = Vec::with_capacity(new_bytes.len() + 1);
    let mut spans: Vec<(u64, u64)> = Vec::with_capacity(new_bytes.len() + 1);
    let mut any_shrunk = false;
    for (ci, bytes) in grid.grid_order.iter().zip(new_bytes.iter()) {
        // A nonzero filter mask means the source left some filter unapplied for
        // this chunk; re-encoding always applies every filter (mask 0), so an
        // in-place overwrite would desync the index-recorded mask. Relocate.
        if ci.filter_mask != 0 {
            return None;
        }
        let new_len = bytes.len() as u64;
        let slot = u64::from(ci.chunk_size);
        // A chunk that no longer fits its slot must relocate.
        if new_len > slot {
            return None;
        }
        if new_len < slot {
            any_shrunk = true;
        }
        let start = usize::try_from(ci.address).ok()?;
        start.checked_add(bytes.len()).filter(|&e| e <= d.len())?;
        writes.push((start, bytes.clone()));
        spans.push((ci.address, new_len));
    }

    // A shrinking overwrite changes the index-recorded chunk sizes, so the index
    // must be rebuilt in place to match; an equal-size one leaves it untouched.
    if any_shrunk {
        let (index_addr, index_bytes) =
            try_rebuild_index_in_place(d, layout, raw_size, &grid.grid_order, new_bytes)?;
        spans.push((index_addr as u64, index_bytes.len() as u64));
        writes.push((index_addr, index_bytes));
    }

    // Refuse to perform overlapping in-place writes (a malformed source index, or
    // an index region that overlaps a chunk slot); relocate instead so two writes
    // never clobber each other.
    if !spans_disjoint_in_bounds(&mut spans, d.len() as u64) {
        return None;
    }
    Some(writes)
}

/// Rebuild a chunked dataset's index **in place** so it records the new
/// (smaller) per-chunk stored sizes after a fits-with-slack overwrite, returning
/// the `(address, bytes)` write that replaces it. The chunks keep their existing
/// addresses (only their stored bytes shrank), so the rebuilt index points at the
/// same slots with the new sizes.
///
/// Supported only for a v4 **fixed-array** or **extensible-array** index whose
/// on-disk structure is a single contiguous region starting at the index address
/// — the layout this crate's own writer produces. The element width derives from
/// the unchanged raw chunk size, so the rebuilt structure is byte-for-byte the
/// same length as the original; this is required to match exactly, which rejects a
/// scattered or differently-laid-out (e.g. C-written) index, leaving the caller
/// to relocate. Single-chunk (size in the layout message) and B-tree-v1 (no
/// writer) indexes are not rebuilt here.
///
/// Like any in-place value overwrite (the HDF5 `H5Dwrite` model) this is not
/// atomic: a crash mid-write can tear the index and leave the dataset needing a
/// rewrite. It is used only on the in-place path, whose linearization point is the
/// synced data write.
fn try_rebuild_index_in_place(
    d: &[u8],
    layout: &DataLayout,
    raw_size: u64,
    grid_order: &[crate::chunked_read::ChunkInfo],
    new_bytes: &[Vec<u8>],
) -> Option<(usize, Vec<u8>)> {
    let DataLayout::Chunked {
        btree_address: Some(index_addr),
        chunk_index_type,
        version,
        ..
    } = layout
    else {
        return None;
    };
    let written: Vec<crate::chunked_write::WrittenChunk> = grid_order
        .iter()
        .zip(new_bytes)
        .map(|(ci, b)| crate::chunked_write::WrittenChunk {
            address: ci.address,
            compressed_size: b.len() as u64,
            raw_size,
            filter_mask: 0,
        })
        .collect();
    let new_index = match (version, chunk_index_type) {
        (4, Some(3)) => crate::chunked_write::build_fixed_array_at(
            &written,
            OFFSET_SIZE,
            LENGTH_SIZE,
            true,
            *index_addr,
        ),
        (4, Some(4)) => crate::chunked_write::build_extensible_array_at(
            &written,
            OFFSET_SIZE,
            LENGTH_SIZE,
            true,
            *index_addr,
        )
        .ok()?,
        // Single-chunk records its size in the layout message (a header rewrite),
        // and a B-tree-v1 index has no writer; both relocate instead.
        _ => return None,
    };

    // The on-disk index must be a single contiguous region starting at the index
    // address, and the rebuilt structure must be exactly the same length (true for
    // an index this crate wrote). A scattered or different on-disk layout fails
    // the check and the caller relocates.
    let mut spans =
        crate::chunked_read::chunk_index_spans_buffered(d, layout, OFFSET_SIZE, LENGTH_SIZE)
            .ok()?;
    if spans.is_empty() {
        return None;
    }
    spans.sort_unstable_by_key(|&(a, _)| a);
    if spans[0].0 != *index_addr {
        return None;
    }
    let mut end = *index_addr;
    for &(a, l) in &spans {
        if a != end {
            return None; // a gap means the index is not contiguous
        }
        end = a.checked_add(l)?;
    }
    if new_index.len() as u64 != end - *index_addr {
        return None;
    }
    let start = usize::try_from(*index_addr).ok()?;
    start
        .checked_add(new_index.len())
        .filter(|&e| e <= d.len())?;
    Some((start, new_index))
}

/// A [`ChunkProvider`] over chunk bytes already held in memory, in dense
/// row-major grid order. Used by the editor's chunked copy and relocating
/// overwrite, which own each chunk's bytes (a [`CopyTree`] or [`MovingWrite`]
/// captured them) rather than streaming from a source file like repack.
struct SliceChunkProvider<'a> {
    chunks: &'a [Vec<u8>],
}

impl ChunkProvider for SliceChunkProvider<'_> {
    fn chunk_bytes(&self, index: usize) -> Result<Vec<u8>, FormatError> {
        self.chunks.get(index).cloned().ok_or_else(|| {
            FormatError::ChunkedReadError("chunk index out of range for in-memory provider".into())
        })
    }
}

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

/// Copy a chunk-0 message `region`, replacing the single (compact) Data Layout
/// message's inline data with `raw` and preserving every other message verbatim.
/// Used by `write_dataset` to overwrite a compact dataset's values. The message
/// header (type and flags) and version byte are kept; only the inline data — and
/// the message size and 2-byte inline-size fields — change. `raw` must fit the
/// compact layout's 2-byte size field (HDF5's 64 KiB compact-storage limit),
/// which an overwrite of an existing compact dataset always satisfies.
fn rebuild_compact_layout_region(region: &[u8], raw: &[u8]) -> Result<Vec<u8>, Error> {
    if raw.len() > u16::MAX as usize {
        return Err(Error::EditUnsupported(
            "compact dataset data is too large to overwrite in place",
        ));
    }
    let mut out = Vec::with_capacity(region.len() + raw.len());
    let mut p = 0;
    let mut replaced = false;
    while let Some((msg_type, body, body_end)) = next_message(region, p)? {
        if msg_type == MessageType::DataLayout {
            if body_end - body < 2 || region[body + 1] != 0 {
                return Err(Error::EditUnsupported(
                    "compact-layout overwrite found a non-compact data layout",
                ));
            }
            // New compact layout body: version (kept), class=0, 2-byte inline
            // size, then the data.
            let mut layout = Vec::with_capacity(4 + raw.len());
            layout.push(region[body]); // version (3 or 4)
            layout.push(0); // class = compact
            #[expect(
                clippy::cast_possible_truncation,
                reason = "raw.len() bounded to u16::MAX above"
            )]
            layout.extend_from_slice(&(raw.len() as u16).to_le_bytes());
            layout.extend_from_slice(raw);
            // Message record: type byte, 2-byte size (LE), flags byte (kept).
            out.push(region[p]);
            #[expect(
                clippy::cast_possible_truncation,
                reason = "layout body length is 4 + raw.len() <= u16::MAX + 4, and an OH \
                          message size that overflows u16 is itself malformed"
            )]
            out.extend_from_slice(&(layout.len() as u16).to_le_bytes());
            out.push(region[p + 3]);
            out.extend_from_slice(&layout);
            replaced = true;
        } else {
            out.extend_from_slice(&region[p..body_end]);
        }
        p = body_end;
    }
    if p < region.len() {
        out.extend_from_slice(&region[p..]);
    }
    if !replaced {
        return Err(Error::EditUnsupported(
            "compact dataset header has no data-layout message",
        ));
    }
    Ok(out)
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
/// non-attribute message verbatim. A fixed-size `Set`/`Remove` is resolved
/// into `region` directly; a variable-length `Set` (`VarLenAsciiArray`) is
/// instead collected into the returned `pending_vl_attrs` — its placeholder
/// heap address is only patched, and the message appended to the group's
/// header, by the apply loop once its global heap collection's real address
/// is known (see [`EditSession::place_vl_collection`]). A later op for the
/// same name (another `Set`, fixed-size or not, or a `Remove`) replaces or
/// cancels an earlier still-pending variable-length entry, keeping the net
/// effect the same regardless of op order within one commit. `region`'s
/// fixed-size portion is a complete compact-attribute header on return; dense
/// attribute storage and shared attribute messages are refused.
fn apply_group_attr_ops(region: &[u8], ops: &[AttrOp]) -> Result<(Vec<u8>, PendingVlAttrs), Error> {
    let mut out = region.to_vec();
    let mut pending_vl: PendingVlAttrs = Vec::new();
    let mut wrote_attr = false;
    for op in ops {
        match op {
            AttrOp::Set { name, value } => {
                wrote_attr = true;
                pending_vl.retain(|(msg, _)| &msg.name != name);
                if let AttrValue::VarLenAsciiArray(strings) = value {
                    // Nothing yet to remove from `region` if this name has
                    // never been set as a fixed-size attribute.
                    out = remove_attr_from_region(&out, name, false)?;
                    let msg = build_attr_message(name, value);
                    if msg.serialize(LENGTH_SIZE).len() > u16::MAX as usize {
                        return Err(Error::EditUnsupported(
                            "attribute is too large to encode in place",
                        ));
                    }
                    let str_refs: Vec<&str> = strings.iter().map(String::as_str).collect();
                    pending_vl.push((msg, build_global_heap_collection(&str_refs)));
                } else {
                    out = set_attr_in_region(&out, name, value)?;
                }
            }
            AttrOp::Remove { name } => {
                let before = pending_vl.len();
                pending_vl.retain(|(msg, _)| &msg.name != name);
                if pending_vl.len() == before {
                    out = remove_attr_from_region(&out, name, true)?;
                }
            }
        }
    }
    if wrote_attr && compact_attr_count(&out)? + pending_vl.len() > MAX_COMPACT_ATTRS {
        return Err(Error::EditUnsupported(
            "attributes would exceed compact storage; dense attribute edits are not supported in place yet",
        ));
    }
    Ok((out, pending_vl))
}

/// Whether an Attribute Info (0x0015) message body denotes *dense* (fractal-heap)
/// attribute storage — a *defined* heap address. The reference C library and h5py
/// emit an Attribute Info message with an *undefined* heap address even for
/// compact, inline attributes in the latest format (to carry creation-order
/// metadata), so its mere presence is not dense storage; only a defined heap
/// address is. An unparseable message is treated as dense (refused conservatively).
/// Mirrors the copy path's dense detection so the compact-attribute editors accept
/// the undefined-address message that nearly every real-world object carries.
fn attribute_info_is_dense(body: &[u8]) -> bool {
    match crate::attribute_info::AttributeInfoMessage::parse(body, OFFSET_SIZE) {
        Ok(ai) => ai.fractal_heap_address.is_some(),
        Err(_) => true,
    }
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
                if attribute_info_is_dense(&region[body..body_end]) {
                    return Err(Error::EditUnsupported(
                        "a target object uses dense (fractal-heap) attribute storage (not supported in place yet)",
                    ));
                }
                // An undefined-heap Attribute Info message is creation-order
                // metadata, not dense storage; preserve it verbatim (fall through
                // to copy the message below).
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

/// Copy a message region, dropping all Attribute messages named `name`. When
/// `required` is true, an absent `name` is an [`Error::EditUnsupported`] (a
/// `Remove` of a nonexistent attribute); when false, it is not an error (a
/// `Set` of a fresh variable-length attribute may have no fixed-size message
/// to remove from the region yet).
fn remove_attr_from_region(region: &[u8], name: &str, required: bool) -> Result<Vec<u8>, Error> {
    let mut out = Vec::with_capacity(region.len());
    let mut p = 0;
    let mut removed = false;
    while let Some((msg_type, body, body_end)) = next_message(region, p)? {
        let mut skip = false;
        match msg_type {
            MessageType::AttributeInfo => {
                if attribute_info_is_dense(&region[body..body_end]) {
                    return Err(Error::EditUnsupported(
                        "a target object uses dense (fractal-heap) attribute storage (not supported in place yet)",
                    ));
                }
                // An undefined-heap Attribute Info message is creation-order
                // metadata, not dense storage; preserve it verbatim.
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
    if !removed && required {
        return Err(Error::EditUnsupported("attribute to remove was not found"));
    }
    Ok(out)
}

fn compact_attr_count(region: &[u8]) -> Result<usize, Error> {
    let mut count = 0usize;
    let mut p = 0;
    while let Some((msg_type, body, body_end)) = next_message(region, p)? {
        if msg_type == MessageType::AttributeInfo
            && attribute_info_is_dense(&region[body..body_end])
        {
            return Err(Error::EditUnsupported(
                "a target object uses dense (fractal-heap) attribute storage (not supported in place yet)",
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
            "a target object has a shared attribute message (not editable in place yet)",
        ));
    }
    crate::attribute::AttributeMessage::parse(&region[body..body_end], LENGTH_SIZE)
        .map(|attr| attr.name)
        .map_err(|_| Error::EditUnsupported("a target object has an unreadable attribute message"))
}

fn encode_attr_message(name: &str, value: &AttrValue) -> Result<Vec<u8>, Error> {
    // `apply_group_attr_ops`'s `Set` branch — this function's only caller —
    // handles `VarLenAsciiArray` itself (staging it into `pending_vl` instead
    // of calling `set_attr_in_region`/here), so this value is always
    // fixed-size by construction, not by a check made at this call site.
    debug_assert!(
        !matches!(value, AttrValue::VarLenAsciiArray(_)),
        "VarLenAsciiArray must be intercepted by apply_group_attr_ops before reaching encode_attr_message"
    );
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

/// Version-2 object-header message flag bit marking a message as *shared* (stored
/// once in the shared-message table and referenced by an object-header address or
/// fractal-heap id) rather than inline. Whatever the message type, that reference
/// points into the source file and is meaningless after a cross-file copy.
const MSG_FLAG_SHARED: u8 = 0x02;

/// Refuse to copy an object whose header embeds a *source-file* absolute address
/// that a verbatim copy into another file cannot translate. An in-file copy keeps
/// these valid by sharing the source file's heaps and objects; a cross-file copy
/// cannot. Three things qualify:
///
/// - a **variable-length** datatype, whose element bytes are global-heap
///   references (collection address + index) into the source file's heap;
/// - a **reference** datatype (object or dataset-region), whose element bytes are
///   absolute object addresses in the source file;
/// - any **shared message** (the `MSG_FLAG_SHARED` bit set) — a committed datatype,
///   but also a shared dataspace, fill value, or filter-pipeline message — whose
///   body is a reference into the source file's shared-message storage.
///
/// The scan covers a copied object's whole message region (a dataset's or a
/// group's): it refuses any shared message outright, and inspects Datatype
/// messages (the element type) and Attribute messages (their own datatype),
/// recursing through compound members, array elements, and enumeration bases so a
/// nested variable-length or reference occurrence is caught too. It is applied
/// only on the cross-file path; the same-file [`copy`](EditSession::copy)
/// deliberately keeps these forms (their addresses stay valid in one file).
fn reject_foreign_addresses(region: &[u8]) -> Result<(), Error> {
    let mut p = 0;
    while let Some((msg_type, body, body_end)) = next_message(region, p)? {
        // A *shared* message stores, in place of its real body, a reference into
        // the source file's shared-message storage — an object-header address or a
        // fractal-heap (SOHM) id — which means nothing in another file. This
        // catches committed (shared) datatypes and shared attributes as well as a
        // shared dataspace, fill value, or filter-pipeline message, all of which
        // HDF5 may place in the shared-message table. Refuse any of them, whatever
        // the message type. The flags byte is the 4th of the record header (type,
        // size, flags); `next_message` returning `Some` guarantees
        // `p + 4 <= region.len()`.
        if region[p + 3] & MSG_FLAG_SHARED != 0 {
            return Err(Error::EditUnsupported(
                "a shared (committed/SOHM) object-header message cannot be copied to another file yet",
            ));
        }
        match msg_type {
            MessageType::Datatype => {
                let (dt, _) =
                    crate::datatype::Datatype::parse(&region[body..body_end]).map_err(|_| {
                        Error::EditUnsupported("a source datatype could not be parsed for copying")
                    })?;
                if datatype_copies_foreign_address(&dt) {
                    return Err(Error::EditUnsupported(
                        "variable-length or reference datasets cannot be copied to another file yet",
                    ));
                }
            }
            MessageType::Attribute => {
                let attr =
                    crate::attribute::AttributeMessage::parse(&region[body..body_end], LENGTH_SIZE)
                        .map_err(|_| {
                            Error::EditUnsupported(
                                "a source attribute could not be parsed for copying",
                            )
                        })?;
                if datatype_copies_foreign_address(&attr.datatype) {
                    return Err(Error::EditUnsupported(
                        "variable-length or reference attributes cannot be copied to another file yet",
                    ));
                }
            }
            _ => {}
        }
        p = body_end;
    }
    Ok(())
}

/// Cross-file screen for a dense (fractal-heap) attribute set. The bytes parsed
/// out of the source heap can embed source-file absolute addresses just as inline
/// attribute messages can — variable-length (global-heap) or reference attribute
/// data — which would dangle in another file. [`reject_foreign_addresses`] screens
/// the verbatim object-header region but not heap-resident attribute bytes, so a
/// dense attribute set is screened here instead. Same-file copies skip this (their
/// addresses stay valid); the fresh heap built on write is same-file by
/// construction, so only the source datatypes matter.
fn reject_foreign_dense_attrs(attrs: &[crate::attribute::AttributeMessage]) -> Result<(), Error> {
    for attr in attrs {
        if datatype_copies_foreign_address(&attr.datatype) {
            return Err(Error::EditUnsupported(
                "variable-length or reference dense (fractal-heap) attributes cannot be copied to another file yet",
            ));
        }
    }
    Ok(())
}

/// Whether `dt` stores, anywhere in its structure, a value that is a source-file
/// absolute address: a variable-length (global-heap) or reference datatype, or a
/// compound / array / enumeration built over one. See [`reject_foreign_addresses`].
fn datatype_copies_foreign_address(dt: &crate::datatype::Datatype) -> bool {
    use crate::datatype::Datatype;
    match dt {
        Datatype::VariableLength { .. } | Datatype::Reference { .. } => true,
        Datatype::Compound { members, .. } => members
            .iter()
            .any(|m| datatype_copies_foreign_address(&m.datatype)),
        Datatype::Array { base_type, .. } | Datatype::Enumeration { base_type, .. } => {
            datatype_copies_foreign_address(base_type)
        }
        _ => false,
    }
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

    /// Stopping an in-place append (`append_inplace`) at any phase boundary must
    /// leave the file readable as a consistent prefix — the old length until the
    /// phase-4 dimension commit, the new length after it — even though a
    /// partial-tail append repoints the visible trailing element in place. Mirrors
    /// `AppendWriter`'s crash-consistency harness, but driven through
    /// `EditSession`'s own mirror (disk-before-mirror ordering) to prove the shared
    /// engine is crash-safe under both owners. Two starting layouts: the trailing
    /// element inline in the index block (chunk 4, n 6), and in a data block
    /// (chunk 2, n 9, slot 0).
    #[test]
    fn append_inplace_crash_consistency_partial_tail_prefix() {
        use crate::reader::File as PureFile;
        use crate::writer::FileBuilder;
        use tempfile::tempdir;

        let build = |path: &std::path::Path, n: i32, chunk: u64| {
            let data: Vec<i32> = (0..n).collect();
            let mut b = FileBuilder::new();
            b.create_dataset("d")
                .with_i32_data(&data)
                .with_shape(&[n as u64])
                .with_maxshape(&[u64::MAX])
                .with_chunks(&[chunk]);
            b.write(path).unwrap();
        };

        for (n, chunk, add) in [(6i32, 4u64, 5i32), (9, 2, 6)] {
            let dir = tempdir().unwrap();
            let base = dir.path().join("base.h5");
            build(&base, n, chunk);

            for max_phase in 1u8..=4 {
                let p = dir.path().join(format!("crash_{n}_{chunk}_{max_phase}.h5"));
                std::fs::copy(&base, &p).unwrap();
                {
                    let mut s = EditSession::open(&p).unwrap();
                    s.append_inplace_i32_phased("d", &(n..n + add).collect::<Vec<_>>(), max_phase)
                        .unwrap();
                    // session dropped here, simulating a crash after `max_phase`
                }
                let expected_len = if max_phase == 4 { n + add } else { n };
                let f = PureFile::from_bytes(std::fs::read(&p).unwrap()).unwrap();
                assert_eq!(
                    f.dataset("d").unwrap().read_i32().unwrap(),
                    (0..expected_len).collect::<Vec<_>>(),
                    "inconsistent view after crash at phase {max_phase} (n={n}, chunk={chunk})"
                );
            }
        }
    }

    #[test]
    fn raw_appendable_recurses_into_aggregates() {
        use crate::datatype::{CompoundMember, DatatypeByteOrder};

        let f64_with = |byte_order| Datatype::FloatingPoint {
            size: 8,
            byte_order,
            bit_offset: 0,
            bit_precision: 64,
            exponent_location: 52,
            exponent_size: 11,
            mantissa_location: 0,
            mantissa_size: 52,
            exponent_bias: 1023,
        };
        let le_f64 = f64_with(DatatypeByteOrder::LittleEndian);
        let be_f64 = f64_with(DatatypeByteOrder::BigEndian);

        // Little-endian scalar: appendable. Big-endian scalar: not.
        assert!(datatype_is_raw_appendable(&le_f64));
        assert!(!datatype_is_raw_appendable(&be_f64));

        // The confirmed bug: a compound / array whose leaf is big-endian must be
        // refused (it was wrongly accepted before recursion was added).
        let be_member = Datatype::Compound {
            size: 8,
            members: vec![CompoundMember {
                name: "x".into(),
                byte_offset: 0,
                datatype: be_f64.clone(),
            }],
        };
        assert!(!datatype_is_raw_appendable(&be_member));
        let le_member = Datatype::Compound {
            size: 8,
            members: vec![CompoundMember {
                name: "x".into(),
                byte_offset: 0,
                datatype: le_f64.clone(),
            }],
        };
        assert!(datatype_is_raw_appendable(&le_member));
        assert!(!datatype_is_raw_appendable(&Datatype::Array {
            base_type: Box::new(be_f64.clone()),
            dimensions: vec![4],
        }));

        // Variable-length / reference leaves are never raw-appendable, even LE.
        assert!(!datatype_is_raw_appendable(&Datatype::VariableLength {
            is_string: false,
            padding: None,
            charset: None,
            base_type: Box::new(le_f64.clone()),
        }));
        assert!(!datatype_is_raw_appendable(&Datatype::Reference {
            size: 8,
            ref_type: crate::datatype::ReferenceType::Object,
        }));
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
    fn reject_foreign_addresses_refuses_any_shared_message() {
        // A shared (SOHM) message of *any* type — here a Dataspace — stores a
        // source-file reference in place of its body, so a verbatim cross-file
        // copy must refuse it, not only shared datatypes/attributes. (A plain,
        // non-shared dataspace embeds no foreign address and is accepted.)
        let mut shared = region_message(MessageType::Dataspace, &[0u8; 8]);
        shared[3] = MSG_FLAG_SHARED; // set the message's shared flag
        let err = reject_foreign_addresses(&shared).unwrap_err();
        assert!(err.to_string().contains("shared"), "got: {err}");

        let plain = region_message(MessageType::Dataspace, &[0u8; 8]);
        reject_foreign_addresses(&plain).unwrap();
    }

    /// Build a compact data-layout message body: version, class=0, 2-byte inline
    /// size, then the data.
    fn compact_layout_body(version: u8, data: &[u8]) -> Vec<u8> {
        let mut b = vec![version, 0];
        b.extend_from_slice(&(data.len() as u16).to_le_bytes());
        b.extend_from_slice(data);
        b
    }

    #[test]
    fn rebuild_compact_layout_replaces_inline_data_only() {
        // A region with a Dataspace message, a compact Data Layout, and a trailing
        // Attribute message: rewriting the inline data must replace exactly the
        // layout's bytes and leave every other message verbatim.
        let mut region = region_message(MessageType::Dataspace, &[0xAB; 8]);
        region.extend_from_slice(&region_message(
            MessageType::DataLayout,
            &compact_layout_body(3, &[1, 2, 3, 4]),
        ));
        region.extend_from_slice(&region_message(MessageType::Attribute, &[0xCD; 5]));

        let out = rebuild_compact_layout_region(&region, &[9, 8, 7, 6]).unwrap();

        // Same messages in the same order; only the layout's inline data changed.
        assert_eq!(
            region_types(&out),
            vec![
                MessageType::Dataspace,
                MessageType::DataLayout,
                MessageType::Attribute,
            ]
        );
        let mut p = 0;
        while let Some((mt, body, end)) = next_message(&out, p).unwrap() {
            match mt {
                MessageType::Dataspace => assert_eq!(&out[body..end], &[0xAB; 8]),
                MessageType::DataLayout => {
                    assert_eq!(out[body], 3, "version preserved");
                    assert_eq!(out[body + 1], 0, "still compact");
                    let size = u16::from_le_bytes([out[body + 2], out[body + 3]]) as usize;
                    assert_eq!(size, 4);
                    assert_eq!(&out[body + 4..body + 4 + size], &[9, 8, 7, 6]);
                }
                MessageType::Attribute => assert_eq!(&out[body..end], &[0xCD; 5]),
                other => panic!("unexpected message {other:?}"),
            }
            p = end;
        }
    }

    #[test]
    fn rebuild_compact_layout_refuses_non_compact() {
        // A contiguous (class 1) data layout is not compact, so the rebuild refuses
        // rather than corrupt it.
        let mut region = region_message(MessageType::DataLayout, &{
            let mut b = vec![3u8, 1]; // version 3, class 1 (contiguous)
            b.extend_from_slice(&0u64.to_le_bytes());
            b.extend_from_slice(&0u64.to_le_bytes());
            b
        });
        region.extend_from_slice(&region_message(MessageType::Dataspace, &[0; 8]));
        let err = rebuild_compact_layout_region(&region, &[1, 2]).unwrap_err();
        assert!(err.to_string().contains("non-compact"), "got: {err}");
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

    #[test]
    fn add_vlen_string_dataset_with_null_elements_via_edit_session() {
        // Regression test for a silent-corruption bug (issue #105): a
        // VL-string dataset added via `EditSession` used to commit `Ok(())`
        // without ever writing its global heap collection or patching its
        // placeholder references, so the dataset failed to read back. A null
        // element (no heap object at all, distinct from an empty string) must
        // stay untouched by the patch — only `patch_mask`-flagged elements'
        // placeholder addresses are resolved; exercising both keeps the mask
        // itself, not just the common all-`Bytes` case, under test.
        use crate::type_builders::VlStringElement;
        use crate::writer::FileBuilder;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vlen_null.h5");

        let mut b = FileBuilder::new();
        b.create_dataset("seed").with_i32_data(&[0]);
        b.write(&path).unwrap();

        let datatype =
            crate::type_builders::make_vlen_string_type(crate::datatype::CharacterSet::Utf8);
        let elements = vec![
            VlStringElement::Bytes(b"alpha".to_vec()),
            VlStringElement::Null,
            VlStringElement::Bytes(b"gamma".to_vec()),
        ];

        {
            let mut s = EditSession::open(&path).unwrap();
            s.create_dataset("labels")
                .with_vlen_string_elements(datatype, &elements)
                .unwrap();
            s.commit().unwrap();
        }

        let file = crate::reader::File::open(&path).unwrap();
        let ds = file.dataset("labels").unwrap();
        assert_eq!(
            ds.read_string().unwrap(),
            vec!["alpha".to_string(), String::new(), "gamma".to_string()]
        );
    }

    #[test]
    fn edit_session_root_group_base_address_overflow_is_rejected() {
        // The edit-path sibling of issue #137. A userblock file has a nonzero base
        // address that `EditSession::open` adds to the stored root-group address.
        // A crafted address of HADDR_UNDEF must be rejected rather than overflow
        // (panicking in debug, wrapping in release).
        use crate::writer::FileBuilder;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("edit_root_overflow.h5");

        const UB: u64 = 512;
        let mut b = FileBuilder::new();
        b.with_userblock(UB);
        b.create_dataset("d").with_i32_data(&[1, 2, 3]);
        b.write(&path).unwrap();

        // Rewrite the stored (base-relative) root-group address to HADDR_UNDEF,
        // recomputing the superblock checksum via `serialize`. The base address
        // still equals the superblock offset, so the file stays editable and the
        // editor reaches the `root_group_address + base` normalization.
        let mut data = std::fs::read(&path).unwrap();
        let off = signature::find_signature(&data).unwrap();
        let mut sb = Superblock::parse(&data, off).unwrap();
        assert_eq!(sb.base_address, UB, "userblock file must have base == UB");
        sb.root_group_address = u64::MAX;
        let bytes = sb.serialize();
        data[off..off + bytes.len()].copy_from_slice(&bytes);
        std::fs::write(&path, &data).unwrap();

        let err = EditSession::open(&path).err().expect("open must fail");
        match err {
            Error::Format(FormatError::OffsetOverflow { offset, length }) => {
                assert_eq!(offset, u64::MAX);
                assert_eq!(length, UB);
            }
            other => panic!("expected root-group address overflow, got {other:?}"),
        }
    }
}
