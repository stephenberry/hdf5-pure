//! In-place editing of an existing HDF5 file (issue #32, Group C).
//!
//! The in-place edit engine opens an existing file and adds objects, overwrites dataset
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
//! Deletion ([`Group::delete`](crate::Group::delete), the HDF5 `H5Ldelete`) is the mirror image:
//! the parent group's header is rebuilt without the removed link, relocated up
//! the tree the same way, and the unlinked object (and its subtree) is freed —
//! its blocks are returned to a session-local free list (see below). A deletion
//! and an addition at the *same* path in one commit is a **replacement** (issue
//! #305): the link is removed from the rebuilt parent before the new object's is
//! appended, and the one superblock write publishes both, so a rotating store
//! expresses a rotation as one commit and the path is never momentarily absent.
//! The new object's storage is still appended rather than laid over the old
//! one's — the original stays live until the superblock repoint, which is what
//! makes a crash during the rotation land on one side or the other — so the
//! space the deletion released is what a *later* commit draws on, by reuse or by
//! truncation.
//! Object copy ([`File::copy`](crate::File::copy), the HDF5 `H5Ocopy`) deep-copies
//! a source subtree — appending fresh copies of every object, repointing internal
//! links and the contiguous data address — and links the copy in like an
//! addition; the headers are reproduced from their verbatim message bytes, so
//! datatypes, dataspaces, and attributes stay byte-exact. A chunked (and filtered)
//! dataset is copied with its chunk payloads and filter pipeline preserved
//! byte-for-byte, its index rebuilt at the new location. The same machinery,
//! [`File::copy_from`](crate::File::copy_from), copies an object **across two open files** — the
//! source being a separate [`File`](crate::File) reader rather than the file being
//! edited. Because the copy is byte-for-byte, the cross-file path refuses anything
//! that embeds a source-file absolute address (variable-length or reference data,
//! a committed datatype), which an in-file copy keeps valid by sharing the source
//! file's heaps and objects.
//!
//! Value overwrite ([`Dataset::write`](crate::Dataset::write), the HDF5 `H5Dwrite`) replaces
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
//!   userblock source (the source must have base 0; see [`copy_from`](crate::File::copy_from)).
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
//!   LZF, ZFP), and may declare extensible (maximum, optionally unlimited)
//!   dimensions. A chunked dataset's data and index — and any filtered chunks —
//!   are produced by the same builder the whole-file writer uses and appended at
//!   end-of-file, so its object header is byte-identical to a freshly written
//!   one. A dataset may be empty (zero-element) under either storage, which is
//!   how an extensible dataset is created before the first
//!   [`append_staged`](crate::Dataset::append_staged) fills it: a contiguous one
//!   gets the undefined data address, a chunked one an index over zero chunks.
//!   A provenance dataset (`with_provenance`) is
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
//!   dataset **immediately and in place** with an in-place append (amortized O(1),
//!   crash-atomic, no `commit`), interleaved with the staged edits above. A
//!   target the fast path cannot handle — a userblock or pre-v2 file, an
//!   unallocated index, a non-Extensible-Array or multi-hard-link dataset, a
//!   a filtered dataset sitting on a partial trailing chunk — is refused with
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
//! recording the smaller end-of-file is itself durable. A commit that fails
//! before its repoint gives back what it drew, so a failed attempt costs the
//! session nothing.
//!
//! Everything a commit places goes through one allocator
//! ([`WriteEngine::reserve`] and [`WriteEngine::place`]), including a chunked
//! dataset's data region and a dense attribute heap. Those carry addresses of
//! their own, so they are sized before they are placed and then built for the
//! address they got — which is why they can land in a freed region at all
//! (issue #261). A paged file (`H5F_FSPACE_STRATEGY_PAGE`) draws only from free
//! space of the page type it is placing, so reuse cannot make metadata and raw
//! data share a page — except from pages that are *wholly* free, which hold
//! nothing of either type to be mixed with and so may be opened for whichever
//! type asks ([`PagedEdit::alloc_typed`]). Every free-space rewrite a paged
//! commit performs is placed through that same allocator where anything fits,
//! rather than into a page of its own, which is what keeps a delete-and-recreate
//! workload from growing a page per commit (issue #286).
//!
//! Reclaim is best-effort and conservative. Contiguous and chunked datasets
//! (chunk index plus chunk data) and whole group subtrees are reclaimed; a
//! deleted object whose blocks cannot be enumerated exhaustively —
//! variable-length global-heap storage, dense attribute/link heaps, a
//! non–version-2 header, a version 2 B-tree chunk index — is left as dead bytes
//! rather than risk freeing a region that is still in use; under-reclaiming only
//! wastes space, while over-reclaiming would corrupt.
//!
//! On a paged file that conservatism extends to anything whose *page type* is not
//! established. A file another writer produced records free space this one cannot
//! place — the reference library's generic-large manager holds metadata and raw
//! alike, and it puts a chunk index among its metadata where this crate puts one
//! beside the chunk data. Such space is kept and written back where it was found,
//! but never handed to an allocation
//! ([`PagedEdit::unclassified`], [`WriteEngine::index_is_provably_raw`]), because
//! placing a byte of the wrong kind in a page is the one thing paging exists to
//! prevent and no reader would report it.
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
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use core::num::NonZeroUsize;

use crate::checksum::jenkins_lookup3;
use crate::chunk_index_inplace::{Located, Store, apply_ea_append, plan_ea_append};
use crate::chunked_read::{
    chunk_index_spans_from_source, enumerate_chunks_from_source, plan_dense_grid,
};
use crate::chunked_write::{
    ChunkMeta, ChunkOptions, ChunkProvider, WrittenChunk, assemble_chunked_at,
    build_extensible_array_at, chunked_data_len, compress_chunks, emit_chunked_data_verbatim,
    extensible_array_len, full_chunk_bytes, plan_chunked_data_verbatim,
    serialize_v4_extensible_array, split_into_chunks,
};
use crate::convert::TryToUsize;
use crate::data_layout::DataLayout;
use crate::dataspace::{Dataspace, DataspaceType};
use crate::datatype::{Datatype, DatatypeByteOrder, embedded_reference_slots};
use crate::error::{Error, FormatError, OBJECT_HEADER_MESSAGE_MAX};
use crate::extensible_array::ExtensibleArrayHeader;
use crate::file_create_properties::FileCreateProperties;
use crate::file_lock::{self, FileLocking};
use crate::file_space_info::{FileSpaceInfo, FileSpaceStrategy, NUM_FILE_FSM_MANAGERS};
use crate::file_writer::{
    LENGTH_SIZE, OFFSET_SIZE, build_chunked_dataset_oh, build_dataset_oh, make_link,
};
use crate::filter_pipeline::{
    FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_LZF, FILTER_SCALEOFFSET, FILTER_SHUFFLE,
    FilterPipeline,
};
use crate::filters::{ChunkContext, FilterScratch, compress_chunk_with, decompress_chunk};
use crate::free_space::FreeList;
use crate::free_space_manager::{
    self, FreeSection, FsmHeader, PageType, PagedManagerPlan, SECT_CLASS_SIMPLE, align_up,
    free_sections, fshd_len, plan_paged_managers, serialize_file_fsm,
};
use crate::group_v2::resolve_group_entries_from_source;
use crate::image::{FileImage, HandleImage, MirrorImage, WriteBuffering};
use crate::libver::LibVer;
use crate::link_message::{LinkMessage, LinkTarget};
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::reader::FileAccessProperties;
use crate::shared_message::DatatypeLocation;
use crate::signature;
use crate::source::{BaseOffsetSource, BytesSource, MetadataCacheConfig, Source};
use crate::superblock::Superblock;
use crate::type_builders::{
    AttrValue, DatasetBuilder, ObjectRefPatch, ObjectRefTarget, VlStringStaging,
    build_attr_message, build_global_heap_collections, make_f32_type, make_f64_type, make_i8_type,
    make_i16_type, make_i32_type, make_i64_type, make_u8_type, make_u16_type, make_u32_type,
    make_u64_type, patch_vl_refs, patch_vl_refs_masked, write_reference_address,
};

/// An undefined on-disk address (all bits set), HDF5's "no address" sentinel.
const UNDEF: u64 = u64::MAX;

/// The refusal both address-side reference screens report: an object reference
/// this commit writes names an object the same commit removes.
///
/// One constant because the two screens differ only in where the address came
/// from — [`WriteEngine::resolve_reference_target`] takes it from a builder's
/// [`ObjectRefTarget::Raw`], [`screen_resolved_references`] reads it out of a
/// dataset's element bytes — and a caller must not be able to tell them apart
/// by the message. It is the address-side twin of the by-name refusal
/// `resolve_reference_target` reports for a path, and shares its second clause.
const REFERENCE_INTO_RECLAIMED_SPACE: &str = "an object-reference dataset holds the address of an object this commit deletes, or of \
     one under it; the reference would be left pointing at storage the delete can reclaim";

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

/// Maximum length of a version 2 object header's fixed prefix: signature (4) +
/// version (1) + flags (1) + optional access/modification/change/birth times
/// (16) + optional attribute phase-change thresholds (4) + the chunk-0 size
/// field (up to 8). Reading this many bytes always covers the prefix, so
/// [`oh_region_at`] can be handed one bounded window instead of a whole-file
/// image.
const OH_PREFIX_MAX: usize = 34;

/// A path identified by its components (no leading/trailing empties); the root
/// group is the empty vector.
type PathKey = Vec<String>;

/// A live [`BufferedAppender`](crate::BufferedAppender)'s hold on a dataset.
///
/// The appender accepts elements into memory and is the only thing that can
/// write them, so any staged edit that would make its flush refuse turns
/// accepted data into lost data at drop time. The claim lets the engine refuse
/// that edit up front instead.
struct AppenderClaim {
    /// Identifies this claim for release; ids are never reused within a session.
    token: u64,
    /// The appender's dataset path, or `None` for a handle reached by object
    /// reference. Such a handle is named by object-header address, which *any*
    /// staged edit may move, so a path-less claim conflicts with everything —
    /// exactly the rule `append_prepare` already applies to that target.
    path: Option<PathKey>,
    /// Whether this appender still owes a staged, index-rebuilding realignment
    /// (a filtered dataset sitting on a partial trailing chunk). That write
    /// commits, and a commit refuses to run beside unrelated staged edits, so
    /// while it is outstanding the claim conflicts with everything.
    needs_commit: bool,
}

/// Variable-length group/root attributes staged by [`apply_group_attr_ops`],
/// each an (attribute message still carrying a placeholder heap address, its
/// global heap collections) pair, resolved in the apply loop.
type PendingVlAttrs = Vec<(crate::attribute::AttributeMessage, Vec<Vec<u8>>)>;

/// Accumulates elements to append to an existing chunked, unlimited dataset via
/// [`Dataset::append_staged`](crate::Dataset::append_staged), in call order along the dataset's first
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

    /// A builder holding a copy of this one's first `byte_len` bytes, carrying
    /// the same element datatype so the prefix is type-checked exactly as the
    /// whole would have been. Used by [`BufferedAppender`](crate::BufferedAppender)
    /// to write out the chunk-aligned prefix of its buffer; it copies rather
    /// than splits so a failed write leaves the buffer intact and the appender
    /// can report precisely which elements did not land.
    pub(crate) fn head(&self, byte_len: usize) -> Self {
        Self {
            raw: self.raw[..byte_len.min(self.raw.len())].to_vec(),
            elem_dt: self.elem_dt.clone(),
            dt_conflict: self.dt_conflict,
        }
    }

    /// Discard the first `byte_len` buffered bytes (a prefix that reached the
    /// file), keeping the element datatype.
    pub(crate) fn drop_front(&mut self, byte_len: usize) {
        self.raw.drain(..byte_len.min(self.raw.len()));
    }

    /// Consume the builder, yielding its accumulated element bytes. Used by
    /// [`BufferedAppender::discard`](crate::BufferedAppender::discard) to hand
    /// abandoned elements back to the caller.
    pub(crate) fn into_raw(self) -> Vec<u8> {
        self.raw
    }

    /// Cut the buffer back to `byte_len` bytes, keeping the element datatype.
    /// [`BufferedAppender`](crate::BufferedAppender) uses this to undo a call
    /// whose write was refused before it touched the file, so the refusal leaves
    /// nothing buffered and a retry cannot append the same elements twice.
    pub(crate) fn truncate(&mut self, byte_len: usize) {
        self.raw.truncate(byte_len);
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
                    // Reserved up front: serializing element by element into a
                    // growing `Vec` re-allocated it ten times per call and copied
                    // the batch twice over, which for an append loop is the whole
                    // per-call cost (issue #228).
                    self.raw.reserve(data.len() * core::mem::size_of::<$ty>());
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

/// Every edit a session has staged and not yet committed, as one value.
///
/// [`WriteEngine::commit`] takes the whole set out for the duration of an
/// attempt and puts it back if that attempt refuses, so a refused commit costs
/// the session no staged work — the same guarantee [`FreeSnapshot`] gives the
/// free lists, for the same reason (issue #316). Keeping the vectors together
/// rather than beside the engine's other fields is what makes that total: a
/// staged kind added later participates by construction, where a tenth field
/// would have to be remembered in a hand-written restore.
#[derive(Default)]
struct StagedEdits {
    /// Datasets staged by `create_dataset`, as (parent group path, dataset).
    ///
    /// Flattened at staging rather than at commit: [`flatten_dataset`] is the
    /// one step of the commit's preflight that *consumes* what it validates, so
    /// running it here is what lets the preflight read the staged set without
    /// destroying it (issue #316). It is a pure function of the builder, so the
    /// guards it raises — a missing shape, data that does not match it, a
    /// feature this engine cannot reproduce — are answered at the call that
    /// stages the dataset, where the caller still has the context to fix them.
    datasets: Vec<(PathKey, FlatDataset)>,
    /// Value overwrites staged by `write_dataset`, as (full dataset path,
    /// dataset). Each replaces an existing dataset's values in place; the new
    /// datatype and shape must match the on-disk ones byte-exactly (this is a
    /// value overwrite, not a reshape/retype). Applied on the next `commit`.
    /// Flattened at staging, for the reason given on [`datasets`](Self::datasets);
    /// the match against the on-disk dataset is a commit-time check and stays
    /// one, since it reads the file.
    writes: Vec<(PathKey, FlatDataset)>,
    /// Appends staged by `append_dataset`, as (full dataset path, builder). Each
    /// grows an existing chunked, unlimited, Extensible-Array-indexed dataset
    /// along axis 0 by keeping its existing chunk data in place and rebuilding the
    /// index over the kept plus newly-appended (and any rewritten trailing) chunks.
    /// Applied on the next `commit`.
    appends: Vec<(PathKey, AppendBuilder)>,
    /// New groups staged by `create_group`, as full paths.
    groups: Vec<PathKey>,
    /// Group attribute edits staged as (group path, operation). The path may be
    /// a group created in this same session.
    group_attrs: Vec<(PathKey, AttrOp)>,
    /// Dataset attribute edits staged as (full dataset path, operation), applied
    /// on the next `commit`. Each relocates the dataset's object header (like a
    /// relocating overwrite): the header is rebuilt with the compact-attribute
    /// change, its single naming link is patched, and the old header freed — the
    /// dataset's data and chunk index stay in place. The target must be an existing,
    /// single-hard-link dataset using compact (not dense fractal-heap) attributes.
    dataset_attrs: Vec<(PathKey, AttrOp)>,
    /// Links staged for removal by `delete`, as full paths.
    deletes: Vec<PathKey>,
    /// Object copies staged by `copy`, as (source path, destination full path).
    copies: Vec<(PathKey, PathKey)>,
    /// Cross-file object copies staged by `copy_from`, as (destination full path,
    /// the source subtree already read out of the other file). The subtree is read
    /// — and foreign-address-screened — eagerly in `copy_from` (the source file is
    /// borrowed only for that call), then linked in at the next `commit`.
    cross_copies: Vec<(PathKey, CopyTree)>,
}

/// Where a [`StagedEdits`] stood before a batch of staging calls, so a batch
/// that fails partway can be undone (see [`StagedEdits::rewind`]).
#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct StagedMark {
    datasets: usize,
    writes: usize,
    appends: usize,
    groups: usize,
    group_attrs: usize,
    dataset_attrs: usize,
    deletes: usize,
    copies: usize,
    cross_copies: usize,
}

impl StagedEdits {
    /// The length of every staged vector right now, for
    /// [`rewind`](Self::rewind).
    ///
    /// Destructured rather than read field by field, here and in
    /// [`rewind`](Self::rewind), so that the "a staged kind added later
    /// participates by construction" claim on [`StagedEdits`] is enforced rather
    /// than hoped for: a struct pattern naming fewer fields than the struct has
    /// does not compile, and the unused binding a tenth kind would leave in
    /// `rewind` is an error under the crate's `-D warnings`.
    fn mark(&self) -> StagedMark {
        let Self {
            datasets,
            writes,
            appends,
            groups,
            group_attrs,
            dataset_attrs,
            deletes,
            copies,
            cross_copies,
        } = self;
        StagedMark {
            datasets: datasets.len(),
            writes: writes.len(),
            appends: appends.len(),
            groups: groups.len(),
            group_attrs: group_attrs.len(),
            dataset_attrs: dataset_attrs.len(),
            deletes: deletes.len(),
            copies: copies.len(),
            cross_copies: cross_copies.len(),
        }
    }

    /// Drop everything staged since `mark`.
    ///
    /// Staging only ever appends — every `stage_*` entry point validates and
    /// then pushes — so truncating to the recorded lengths is an exact undo of
    /// the calls made in between, and leaves anything staged before them alone.
    fn rewind(&mut self, mark: StagedMark) {
        let StagedMark {
            datasets,
            writes,
            appends,
            groups,
            group_attrs,
            dataset_attrs,
            deletes,
            copies,
            cross_copies,
        } = mark;
        self.datasets.truncate(datasets);
        self.writes.truncate(writes);
        self.appends.truncate(appends);
        self.groups.truncate(groups);
        self.group_attrs.truncate(group_attrs);
        self.dataset_attrs.truncate(dataset_attrs);
        self.deletes.truncate(deletes);
        self.copies.truncate(copies);
        self.cross_copies.truncate(cross_copies);
    }

    /// Whether nothing at all is staged.
    ///
    /// Asked as "is the mark the zero mark", which is exact — nothing is staged
    /// exactly when every vector has length zero — and leaves [`mark`](Self::mark)
    /// as the single place that names every vector. The two callers,
    /// [`WriteEngine::has_staged_edits`] and the commit's own no-op return, then
    /// cannot come to disagree, and a staged kind added later reaches both. (The
    /// commit's *fast path* asks a narrower question and spells its own subset
    /// out.)
    fn is_empty(&self) -> bool {
        self.mark() == StagedMark::default()
    }
}

/// The in-place write engine behind the owned read-write [`File`](crate::File)
/// (its `Backend::Edit`).
///
/// Reads and edits the file through a [`FileImage`], which owns the writable
/// handle and decides how much of the file is resident. It carries two commit
/// models: staged tree edits applied by [`commit`](Self::commit), and immediate
/// crash-atomic in-place appends ([`append_inplace_gathered`](Self::append_inplace_gathered)).
pub(crate) struct WriteEngine {
    /// The file bytes this session reads and edits, behind the [`FileImage`]
    /// abstraction: reads go through its [`Source`] impl, and the write side —
    /// the end-of-file cursor, `append`, `write_at`, `truncate`, and the
    /// durability barriers — through its own primitives.
    ///
    /// Nothing in the engine assumes the whole file is resident, so one engine
    /// serves both a whole-file mirror and a file-backed image that holds only
    /// what it is reading (issue #198). [`image_slice`](Self::image_slice)
    /// exposes the mirror's buffer where a caller can exploit it.
    image: Box<dyn FileImage>,
    /// Absolute offset of the superblock signature in the file.
    sb_sig_off: usize,
    /// Parsed superblock. On-disk addresses are stored relative to `base_address`;
    /// the in-memory `root_group_address` is normalized to an absolute file offset
    /// on open and converted back to a base-relative address when serialized on
    /// commit. `base_address` equals the superblock's file location (`sb_sig_off`):
    /// 0 for a plain file, the userblock size for one with a userblock.
    superblock: Superblock,
    /// Every edit this session has staged and not yet applied. Held as one
    /// value so that [`commit`](WriteEngine::commit) can take the whole set out
    /// for the duration of an attempt and put it back when that attempt
    /// refuses; see [`StagedEdits`].
    staged: StagedEdits,
    /// Datasets with a live [`BufferedAppender`](crate::BufferedAppender), which
    /// holds accepted elements only it can write. A staged edit that would stop
    /// that appender from flushing is refused while the claim stands, rather
    /// than left to fail in the appender's `Drop`, where there is no caller to
    /// report it to and the buffer is simply lost. See `refuse_if_claimed`.
    appender_claims: Vec<AppenderClaim>,
    /// Monotonic id for the next claim, so releasing one is exact even when two
    /// appenders on different datasets are live at once.
    next_appender_token: u64,
    /// Set while an appender is driving its own staged realignment through
    /// `stage_dataset_append` + `commit`, so its claim does not refuse its own
    /// write. Re-entrancy only; never observable outside that call.
    appender_commit_in_progress: bool,
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
    /// ([`append_inplace_gathered`](Self::append_inplace_gathered)), keyed by the dataset's resolved
    /// **object-header address** (not its path, so two hard links to one dataset
    /// share one entry). Populated on the first append to a dataset and maintained
    /// across appends; cleared wholesale at the entry of every non-trivial
    /// [`commit`](Self::commit), since a commit can relocate a cached header or
    /// free the region it points into (see `commit`).
    located: HashMap<u64, LocatedState>,
    /// True when this engine was opened for SWMR writing
    /// ([`open_swmr_writer`](Self::open_swmr_writer)): the append engine then
    /// enforces the SWMR subset (unfiltered, chunk-aligned) so a concurrent
    /// reader never observes a torn view. `false` for an ordinary edit session.
    swmr_mode: bool,
    /// Paged-file state (`H5F_FSPACE_STRATEGY_PAGE`), read from the superblock
    /// extension at `open` regardless of the persist flag; `None` for the common
    /// non-paged file. When `Some`, [`commit`](Self::commit) takes a page-aware
    /// tail that keeps pages homogeneous and rewrites the per-page-type managers
    /// (issue #198). A paged file that does not *persist* its free space is still
    /// refused: see [`PagedEdit`].
    paged: Option<PagedEdit>,
    /// Set by the first [`commit`](Self::commit) that does any work. A commit can
    /// relocate an object header, and nothing on disk distinguishes a relocated
    /// header from the intact bytes it vacated — the old header still parses, and
    /// its data-layout message still points at the live chunk index. An
    /// [`AppendTarget::Header`] captured before that commit would therefore append
    /// successfully *into the dead header*, growing its dataspace while the live
    /// dataset stayed put, and report `Ok`. A path is re-resolved on every append
    /// and so survives a commit; a raw address does not, so it is refused once one
    /// has run.
    committed: bool,
    /// Object-header address for each path an in-place append has resolved in
    /// this session. A single `Dataset::append` asks for the target's geometry
    /// and then appends to it, and a loop of appends repeats that; without this
    /// the path would be walked from the root on every one of those steps.
    ///
    /// An in-place append never moves an object header — that is why
    /// [`located`](Self::located) can be keyed by address and survive appends —
    /// so only a commit can stale an entry, and it clears both together.
    resolved: HashMap<String, u64>,
    /// Whether this session splits a large in-place append into batches, trading
    /// whole-call crash atomicity for a peak memory that does not scale with the
    /// call. Set by [`open_rw_with_strategy`](Self::open_rw_with_strategy); see
    /// [`batch_elems`](Self::batch_elems).
    batched_appends: bool,
    /// Whether this session reads through a handle rather than a whole-file
    /// mirror. Set by [`open_rw_with_strategy`](Self::open_rw_with_strategy), and reported by
    /// [`File::edit_backing`](crate::File::edit_backing) so a caller who
    /// asked for [`MemoryStrategy::Auto`] can tell which one it got. Distinct
    /// from [`batched_appends`](Self::batched_appends), which is a crash-atomicity
    /// trade the bounded engine happens to make, not a statement about memory.
    bounded: bool,
    /// The on-disk format this session may write, resolved from the fapl's
    /// [`FileAccessProperties::with_libver_bounds`]. `None` — the default —
    /// means unconstrained: the session adds whatever the content needs, which
    /// is what lets a file the C library wrote under its own bounds be edited at
    /// all.
    ///
    /// Set below [`LibVer::V110`] it refuses content the older format cannot
    /// carry, the way the whole-file writer does. The file's *own* superblock
    /// version cannot stand in for this: a version 2 superblock says the file is
    /// 1.8-readable today, not that the caller wants it to stay that way, and
    /// deriving the ceiling from it would refuse the C-library-file edits of
    /// issue #101.
    ///
    /// [`FileAccessProperties::with_libver_bounds`]: crate::FileAccessProperties::with_libver_bounds
    libver_ceiling: Option<LibVer>,
    /// The file length when the on-disk free-space managers were last written,
    /// for a file that persists them. Every immediate in-place append grows the
    /// file past those managers and leaves them mid-file, so a session that ends
    /// with `image.len() != fsm_len` owes a rewrite; that is what
    /// [`finalize_persist`](Self::finalize_persist) settles at close. Meaningless
    /// (and untouched) when `persist` is `None`.
    fsm_len: u64,
    /// Whether the commit currently running has repointed the superblock at its
    /// new root. Set by whichever tail performs that write, read only by
    /// [`commit`](Self::commit) to decide whether a failure may hand this
    /// commit's allocations back to the free lists (see [`FreeSnapshot`]).
    ///
    /// It is a property of one commit, not of the session, so `commit` clears it
    /// on entry rather than trusting the previous run to have left it false.
    repointed: bool,
    /// Who owns this session's `fsync` cadence, from the fapl's
    /// [`FileAccessProperties::with_sync_policy`]. Consulted by
    /// [`barrier`](Self::barrier) and [`barrier_data`](Self::barrier_data) —
    /// every durability point the write paths define — and by nothing else, so
    /// [`sync_now`](Self::sync_now) can serve an explicit
    /// [`File::sync`](crate::File::sync) whatever it says.
    ///
    /// [`FileAccessProperties::with_sync_policy`]: crate::FileAccessProperties::with_sync_policy
    sync_policy: SyncPolicy,
}

/// The free lists as they stood before a commit's apply loop drew from them.
///
/// A commit allocates as it goes — a free region it hands to an object header or
/// a chunk blob leaves the list immediately, so no two objects in the same commit
/// can be handed the same region. If the commit then fails, those regions are as
/// dead as they were before it started: nothing the commit wrote is reachable,
/// because the superblock still names the old root. Restoring this snapshot gives
/// them back instead of leaking them for the rest of the session.
///
/// The restore is sound only *before* the repoint. After it, the objects written
/// into those regions are live, and handing the same addresses out again would
/// overwrite the tree the commit just published — which is why
/// [`WriteEngine::repointed`] gates it.
struct FreeSnapshot {
    free: FreeList,
    paged: Option<(FreeList, FreeList)>,
}

/// How much memory a read-write open may use to hold the file being edited.
///
/// The two read-write backends differ in memory, not in what they can express: a
/// *bounded* session reads through a handle and holds only what a commit is
/// building, while a *mirrored* session materializes the whole file in memory.
/// Bounded is the better default when it applies, but it cannot yet edit every
/// file — a pre-v2 (non-latest-format) superblock or a userblock still needs the
/// mirror.
///
/// This is what a caller says about that trade-off, on
/// [`FileAccessProperties::with_memory_strategy`](crate::FileAccessProperties::with_memory_strategy).
/// Leaving it unset lets the entry point decide: [`File::open_rw`](crate::File::open_rw)
/// prefers the bounded engine and falls back to the mirror ([`Auto`](Self::Auto)).
/// Stating [`Bounded`](Self::Bounded) refuses instead of falling back.
///
/// This is a *request*, so it is deliberately not the type a file answers with:
/// [`File::edit_backing`](crate::File::edit_backing) returns an [`EditBacking`],
/// which cannot express [`Auto`](Self::Auto).
///
/// Sealed: unlike [`FileLocking`] or [`FileSpaceStrategy`](crate::FileSpaceStrategy),
/// whose variant sets mirror a closed C-library enum, this is a policy this crate
/// invented, so a fourth strategy must not be a breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MemoryStrategy {
    /// Never build a whole-file mirror. A file the bounded engine cannot edit is
    /// refused at open with [`Error::EditUnsupported`], before anything is
    /// staged.
    Bounded,
    /// Prefer the bounded engine, but fall back to the whole-file mirror for a
    /// file it cannot edit, rather than refusing. Memory then scales with the
    /// file, which is the cost of the file opening at all. What
    /// [`File::open_rw`](crate::File::open_rw) uses when nothing is asked for.
    Auto,
    /// Always build the whole-file mirror, whatever the file looks like. What
    /// [`File::open_rw`](crate::File::open_rw) did before it learned to dispatch.
    Mirrored,
}

/// Which of the two read-write backends a file's editing session is actually
/// using, from [`File::edit_backing`](crate::File::edit_backing).
///
/// Deliberately a different type from [`MemoryStrategy`]: that one is what a
/// caller *asks* for and includes [`Auto`](MemoryStrategy::Auto), which is a
/// preference between these two rather than a third thing a session can be. A
/// single shared type would make `file.backing() == Auto` a comparison that
/// compiles and is false forever.
///
/// The two also evolve at different rates. A future `MemoryStrategy` may name a
/// new *policy* — a byte budget, a size threshold — without the set of backends
/// changing at all. Sealed for the rarer case that a third backend does appear.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EditBacking {
    /// Reads through a file handle, holding only what a commit is building.
    /// Memory does not scale with the file.
    Bounded,
    /// Holds the whole file in memory for the life of the session.
    Mirrored,
}

impl From<EditBacking> for MemoryStrategy {
    /// Turns an outcome back into the request that pins it, so a caller can
    /// reopen a file onto the backing it got the first time:
    /// `with_memory_strategy(file.edit_backing().unwrap().into())`.
    fn from(backing: EditBacking) -> Self {
        match backing {
            EditBacking::Bounded => Self::Bounded,
            EditBacking::Mirrored => Self::Mirrored,
        }
    }
}

/// When a read-write session forces its writes to durable storage — who owns the
/// `fsync` cadence, this crate or the application.
///
/// This is *not* about whether a write reaches the file. Every commit and every
/// append has gone to the operating system by the time it returns, whatever this
/// policy says, so a committed edit is visible to any other process on the same
/// machine and survives this process crashing. What it governs is the `fsync` on
/// top of that, which is what makes those bytes survive the *machine* losing
/// power.
///
/// A session does gather the many small writes *inside* one such operation and
/// issue them a page at a time (issue #288). That gathering releases at every
/// ordering barrier, so it keeps the guarantee in the paragraph above rather
/// than trading it.
///
/// The reference C library does not `fsync` on its normal path either: the
/// default `sec2` driver installs no flush callback at all, so `H5Fflush` drains
/// libhdf5's own caches with `write` and stops there, leaving power-loss
/// durability to the application. [`OnClose`](Self::OnClose) is that behavior;
/// [`Always`](Self::Always), the default here, is the stronger one.
///
/// The whole-file writer ([`FileBuilder`](crate::FileBuilder)) and
/// [`repack`](crate::repack) are outside this: they never `fsync` under either
/// policy, since each writes a file and hands it over rather than holding an
/// editing session.
///
/// Sealed: a future policy (syncing on a timer, or once per N bytes) must not be
/// a breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum SyncPolicy {
    /// Force durability at every point the write paths define one: after each
    /// immediate [`Dataset::append`](crate::Dataset::append) batch, at each
    /// ordering barrier inside a [`File::commit`](crate::File::commit), and when
    /// the file is closed or dropped.
    ///
    /// This is what makes an append crash-atomic and a commit all-or-nothing
    /// against power loss: the barriers order the appended data before the
    /// superblock repoint that publishes it, so an interrupted write leaves the
    /// previous state rather than a torn one.
    #[default]
    Always,
    /// No `fsync` during the session; one when the file is finished. Nothing an
    /// append or a commit does is forced to durable storage — writes still reach
    /// the operating system immediately, so another process on the same machine
    /// sees them and a crash of *this* process loses nothing — and
    /// [`File::close`](crate::File::close), or dropping the last handle, issues
    /// a single barrier at the end. [`File::sync`](crate::File::sync) adds a
    /// checkpoint wherever the application wants one.
    ///
    /// The terminal barrier is not an exception grafted onto "never sync": it is
    /// the point past which the application *cannot* act. `close` and `drop`
    /// both write — they apply staged edits, re-home the free-space managers of
    /// a file that persists them, and clear a SWMR writer's flag — and they
    /// destroy the handle that would have ordered those writes. A policy that
    /// skipped the barrier there would not hand the caller a cadence; it would
    /// take one away. The cost is one `fsync` per session against the five per
    /// append batch and two or three per commit that this policy removes.
    ///
    /// Three things are given up in exchange. **Power-loss ordering** during the
    /// session: with the barriers gone, a machine that loses power mid-commit
    /// can have the superblock repoint on disk without the data it points at.
    /// **Deferred write errors**: on a filesystem that allocates late, a write
    /// that will fail at writeback still returns success, and `fsync` is where
    /// the `ENOSPC`/`EIO` surfaces, so a commit the filesystem cannot complete
    /// returns `Ok` until the terminal barrier reports it. **Cross-host
    /// visibility**: "another process sees it" is the page cache, so under NFS's
    /// close-to-open semantics a reader on another host — a SWMR reader
    /// included — may not see writes a client is holding.
    OnClose,
}

/// Why the bounded engine cannot edit a file, when the whole-file mirror can.
///
/// Kept separate from the refusals that apply to *both* engines: a fallback is
/// only ever worth taking for a limitation the mirror does not share. A paged
/// file with no persisted free-space managers, for instance, is refused by the
/// staged commit as well, so mirroring it would trade a clear error at open for
/// the same error later with work already staged.
fn bounded_only_limitation(session: &WriteEngine) -> Option<&'static str> {
    if session.superblock.version < 2 {
        return Some(
            "bounded read-write access requires a latest-format file (v2/v3 superblock); \
             leave MemoryStrategy unset, or pass MemoryStrategy::Auto, to fall back to \
             the whole-file mirror here",
        );
    }
    if session.superblock.base_address != 0 {
        return Some(
            "bounded read-write access does not support a file with a userblock \
             (non-zero base address); leave MemoryStrategy unset, or pass \
             MemoryStrategy::Auto, to fall back to the whole-file mirror here",
        );
    }
    None
}

/// Whether a file built from `create` could not then be opened read-write under
/// `access`, and why.
///
/// [`File::create_with_options`](crate::File::create_with_options) writes a file
/// and hands back an open read-write handle, so a creation/access pair that
/// cannot survive that second half must be caught *before* the write — otherwise
/// the call leaves a file on disk and returns `Err`, which reads like a failed
/// create but is not one.
///
/// This mirrors the open-time refusals above and must be kept in step with them:
/// [`bounded_only_limitation`] for the userblock, and the shared paged check in
/// [`open_rw_with_strategy`](WriteEngine::open_rw_with_strategy) for a paged file
/// with no persisted free space. Both are stated here in terms of the properties
/// that *cause* them, because the open-time wording tells the caller to recreate
/// the file — advice that is circular when the caller is creating it.
pub(crate) fn create_would_refuse_reopen(
    create: &FileCreateProperties,
    access: &FileAccessProperties,
) -> Option<&'static str> {
    if let Some((FileSpaceStrategy::Page, false, _)) = create.file_space_strategy() {
        return Some(
            "a paged file (FileSpaceStrategy::Page) with persist = false cannot be reopened \
             read-write, so creating one this way would write the file and then fail to open \
             it; pass persist = true to with_file_space_strategy, or build the file with \
             FileBuilder if it is only ever going to be read",
        );
    }
    if create.userblock() != 0 && access.memory_strategy() == Some(MemoryStrategy::Bounded) {
        return Some(
            "a userblock cannot be combined with MemoryStrategy::Bounded: the bounded engine \
             cannot edit a file with a non-zero base address, so creating one this way would \
             write the file and then refuse to open it; drop the userblock, or leave \
             MemoryStrategy unset to mirror this file",
        );
    }
    None
}

/// Paged-file bookkeeping for the whole-file editor (issue #198, step 1).
///
/// A paged file never mixes metadata and raw data within one page, so this tracks
/// free space per page *type* — the one distinction that governs where a byte may
/// be placed — and keeps the commit's appends homogeneous by padding a tail page
/// whenever the page type changes.
///
/// One list per type, not one per on-disk manager. A paged file records its free
/// space in three managers (metadata-small, raw-small, and a generic-large one for
/// whole free pages), but for space this session can place, that split is a size
/// classification of the same raw or metadata space, and [`plan_paged_managers`]
/// recomputes it from scratch at every commit. Tracking it here as well would only
/// stop a freed chunk-data run and the freed index that abuts it from coalescing —
/// leaving two neighboring holes neither of which fits the dataset that just
/// vacated both (issue #261).
///
/// The free lists are seeded only when the file *persists* its free space. A paged
/// non-persisting file has no on-disk record of which pages hold metadata and
/// which hold raw data, so there is nothing to seed and no way to stay segregated;
/// [`commit`](EditSession::commit) refuses it outright, exactly as the bounded
/// backend does.
struct PagedEdit {
    page_size: u64,
    /// Free space inside metadata pages, plus whole free pages this session last
    /// saw as metadata. A page holding nothing belongs to no type, so
    /// [`alloc_typed`](Self::alloc_typed) lets raw data claim one from here.
    meta: FreeList,
    /// Free space inside raw-data pages, plus whole free pages this session last
    /// saw as raw — including every one seeded from the generic-large manager,
    /// which records no type. Metadata claims from here on the same terms.
    raw: FreeList,
    /// Free space this session may record but must never hand out, because the
    /// page it sits in is of unknown type. Seeded at open and constant
    /// thereafter — nothing this engine frees is ever of unknown type — and
    /// written back to the generic-large manager verbatim, so the space stays
    /// available to the writer that recorded it.
    ///
    /// It exists because the reference C library's generic-large manager is
    /// exactly that: `H5F_MEM_PAGE_GENERIC` is *"large-sized generic: meta and
    /// raw"*, and under paged aggregation on a contiguous-address driver every
    /// allocation of a page or more lands there whatever its type. Its
    /// page-alignment tail (`H5MF__alloc_pagefs`) is then recorded as a free
    /// section in the same manager — a sub-page fragment sitting in a page whose
    /// earlier bytes are live. Handing such a fragment to raw data would put raw
    /// bytes in a metadata page, so a section that is not a whole run of aligned
    /// pages is kept here rather than guessed at.
    unclassified: FreeList,
    /// Page type of the current tail page. `None` until this session's first
    /// typed append; the file is page-aligned at open, so the first append never
    /// needs to pad regardless of this.
    last: Option<PageType>,
    /// Free tails left by padding a metadata page before a raw append.
    meta_pad: Vec<(u64, u64)>,
    /// Free tails left by padding a raw page before a metadata append.
    raw_pad: Vec<(u64, u64)>,
}

impl PagedEdit {
    /// Ensure the next allocation on `image` begins in a page holding page type
    /// `ty`: when the tail page holds the *other* type and is only partially
    /// filled, pad it to a page boundary and record the padding as free space of
    /// the outgoing type.
    ///
    /// This is the whole of the paged-append rule, and it lives here so the two
    /// places that grow a paged file — the staged commit through
    /// [`WriteEngine::begin_page`](WriteEngine::begin_page), and the shared
    /// Extensible-Array append engine through [`EditStore`] — cannot drift. They
    /// used to keep separate copies of this state, one per engine, which is what
    /// made an in-place append to a paged file unsafe from the whole-file editor
    /// (issue #198).
    ///
    /// Call it **before** reading the image's end-of-file to compute an address
    /// that will be embedded in the bytes being built: several callers build
    /// content whose interior addresses assume it lands at the current
    /// end-of-file, and padding inserted after that read would shift the landing
    /// address out from under them.
    fn begin(&mut self, image: &mut dyn FileImage, ty: PageType) -> Result<(), Error> {
        let len = image.len();
        if len % self.page_size != 0 {
            let pad_len = self.page_size - len % self.page_size;
            // `prev` is the outgoing page type to record the padding under, or
            // `None` for a crash-recovery pad whose tail-page type is unknown.
            let pad = match self.last {
                // Normal case: the tail page holds a known type; pad only on a
                // type switch, recording the tail as free of the outgoing type.
                Some(prev) if prev != ty => Some(Some(prev)),
                Some(_) => None, // same type: keep packing the tail page
                // A previous session grew this paged file and was killed before
                // its tail was page-aligned, so the file opened non-page-aligned
                // with no known tail type. Pad it up (extending whatever the tail
                // page holds, so the page stays homogeneous) and leave the padding
                // untracked, since recording it under the wrong page type could
                // let a reader reuse it and mix the page.
                None => Some(None),
            };
            if let Some(prev) = pad {
                let pad_at = len;
                image.append(&vec![0u8; pad_len.to_usize()?])?;
                match prev {
                    Some(PageType::Meta) => self.meta_pad.push((pad_at, pad_len)),
                    Some(PageType::Raw) => self.raw_pad.push((pad_at, pad_len)),
                    None => {} // crash-recovery pad: untracked (tail type unknown)
                }
            }
        }
        self.last = Some(ty);
        Ok(())
    }

    fn new(page_size: u64) -> Self {
        PagedEdit {
            page_size,
            meta: FreeList::new(),
            raw: FreeList::new(),
            unclassified: FreeList::new(),
            last: None,
            meta_pad: Vec::new(),
            raw_pad: Vec::new(),
        }
    }

    /// Which list a persisted section from File Space Info `slot` belongs in.
    ///
    /// Slot 0 (SUPER) is small metadata and slot 2 (DRAW) is small raw: both name
    /// a page type outright. Slot 6 is the generic-large manager, which holds
    /// space of *either* type (see [`unclassified`](Self::unclassified)), so a
    /// section from it is only safe to reuse when it covers whole aligned pages —
    /// and such a section belongs to no type at all, since nothing lives in those
    /// pages to be mixed with. It is filed under raw and reached from either side
    /// through [`alloc_typed`](Self::alloc_typed), rather than kept in a third
    /// list, so that a run of free pages still coalesces with the sub-page
    /// fragment beside it. Everything else, including the per-type large managers
    /// a multi/split driver would populate, is left unclassified rather than
    /// guessed at.
    fn slot_list(slot: usize, addr: u64, size: u64, page_size: u64) -> Option<PageType> {
        match slot {
            0 => Some(PageType::Meta),
            2 => Some(PageType::Raw),
            6 if addr % page_size == 0 && size % page_size == 0 => Some(PageType::Raw),
            _ => None,
        }
    }

    /// Record `(addr, size)` as free space of page type `ty` in the given lists.
    /// [`plan_paged_managers`] splits and classes into the on-disk managers at
    /// serialization time, so this only has to route by page type.
    ///
    /// Takes the lists rather than `&mut self` because a commit routes into
    /// *copies* of them — nothing is free until the superblock repoint — and the
    /// rule must not be restated at that call site.
    fn route_free(meta: &mut FreeList, raw: &mut FreeList, addr: u64, size: u64, ty: PageType) {
        match ty {
            PageType::Meta => meta.free(addr, size),
            PageType::Raw => raw.free(addr, size),
        }
    }

    /// Draw `len` bytes of page type `ty` from this file's free space, or `None`
    /// when nothing can serve it.
    ///
    /// Its own list first, which is the only place a *partly* free page can serve
    /// it. Failing that, whole pages inside the other type's free space: a page
    /// with nothing in it belongs to neither type, so opening it for `ty` cannot
    /// mix it. Enough whole pages to cover `len` are claimed, and the remainder
    /// joins `ty`'s list, since those pages now hold `ty`.
    ///
    /// The cross-type claim is what lets space survive a close. A whole free page
    /// is written to the generic-large manager, which records no page type, so
    /// every one of them comes back from disk as raw ([`slot_list`](Self::slot_list));
    /// without this, metadata could never reuse a page again after a reopen, and
    /// each session's commits would append past all of them (issue #286).
    fn alloc_typed(&mut self, len: u64, ty: PageType) -> Option<u64> {
        let (own, other) = match ty {
            PageType::Meta => (&mut self.meta, &mut self.raw),
            PageType::Raw => (&mut self.raw, &mut self.meta),
        };
        if let Some(addr) = own.alloc(len) {
            return Some(addr);
        }
        let span = align_up(len, self.page_size);
        let addr = other.alloc_whole_units(span, self.page_size)?;
        if span > len {
            own.free(addr + len, span - len);
        }
        Some(addr)
    }

    /// Every free region this session could still hand out, ascending by address.
    /// Used for space accounting, where the caller wants one total rather than a
    /// per-page-type breakdown.
    ///
    /// [`unclassified`](Self::unclassified) is excluded: it is free space, and
    /// [`File::persisted_free_space`](crate::File::persisted_free_space) reports
    /// it as such, but this session will never place anything in it, which is
    /// what the accounting field it feeds is about.
    fn reusable_sections(&self) -> Vec<(u64, u64)> {
        let mut out = self.meta.sections();
        out.extend(self.raw.sections());
        out.sort_by_key(|&(addr, _)| addr);
        out
    }
}

/// The free space a paged commit is about to write to disk, as
/// [`WriteEngine::paged_post_free`] computes it: the session's lists with this
/// commit's frees folded in, held apart from the session until the repoint makes
/// them true.
struct PagedPostFree {
    meta: FreeList,
    raw: FreeList,
    /// Already flattened: nothing in a commit adds to or draws from this one.
    unclassified: Vec<FreeSection>,
}

/// Where a commit has decided to put one allocation's bytes, handed out by
/// [`WriteEngine::reserve`] and consumed by [`WriteEngine::place`].
///
/// The two arms differ in how the bytes are written — over a dead interior region
/// or past the end of the file — and in nothing else the caller sees, so a caller
/// that needs the address before the bytes (a relocatable blob) never has to know
/// which it got.
#[derive(Clone, Copy, Debug)]
enum Placement {
    /// A region an earlier commit freed, already removed from its free list.
    Reused { addr: u64, len: u64 },
    /// The end-of-file, with the page for the requested type already open.
    Appended { addr: u64, len: u64 },
}

impl Placement {
    /// The absolute file address the bytes will occupy.
    fn address(self) -> u64 {
        match self {
            Placement::Reused { addr, .. } | Placement::Appended { addr, .. } => addr,
        }
    }

    /// The reserved byte count, which the placed bytes must match exactly.
    fn len(self) -> u64 {
        match self {
            Placement::Reused { len, .. } | Placement::Appended { len, .. } => len,
        }
    }
}

/// What an in-place append names its target by.
///
/// A handle with a resolvable path uses it, which lets the session compare the
/// target against its own staged edits. A handle reached by object reference has
/// no path, so it names the dataset by the object-header address it was reached
/// through — the same key the geometry cache uses.
#[derive(Clone, Copy)]
pub(crate) enum AppendTarget<'a> {
    Path(&'a str),
    Header(u64),
}

/// Byte budget for one append batch on a session that batches: a large append is
/// split into whole-chunk batches of at most this many raw bytes (always at least
/// one chunk), each applied as its own crash-atomic fsync-barriered sequence, so
/// peak append memory never scales with the caller's slice.
const APPEND_BATCH_BYTES: u64 = 1 << 20;

/// The data-only durability barrier: an ordering point always, and an `fsync`
/// only when this session's [`SyncPolicy`] keeps the cadence.
///
/// One function rather than one per caller because the commit path
/// ([`WriteEngine::barrier_data`]) and the append engine's phase boundaries
/// ([`EditStore::sync`]) are the same point reached from two owners of the
/// image, and a barrier that orders at one and not the other is a
/// crash-consistency defect no test on the default policy can see (issue #288).
///
/// The match is exhaustive on purpose, as at [`WriteEngine::barrier`]:
/// `SyncPolicy` is sealed to the outside but not to this crate, so a policy
/// added later fails to compile here until someone decides what it means.
fn barrier_data(image: &mut dyn FileImage, sync_policy: SyncPolicy) -> Result<(), Error> {
    match sync_policy {
        SyncPolicy::Always => image.sync_data(),
        SyncPolicy::OnClose => image.ordering_barrier(),
    }
}

/// Byte budget for the writes one operation may gather before they are issued
/// (see [`WriteBuffering::Operation`]).
///
/// A megabyte, the same figure as [`APPEND_BATCH_BYTES`] and for a related
/// reason: that is already what a bounded session spends holding one batch of
/// raw append data, so this is a familiar quantum for the write path rather than
/// a new one. The two are not derived from each other and either may be tuned
/// alone. An operation larger than this is flushed part-way, which costs it
/// little — its writes are long and contiguous by then.
const WRITE_GATHER_BYTES: usize = 1 << 20;

/// Page the write gatherer merges within on a file that is not paged.
///
/// A non-paged file has no page size of its own, and this is the same figure
/// HDF5 defaults `H5Pset_file_space_page_size` to, so the merge quantum matches
/// what the file *would* have used had it been paged.
const DEFAULT_GATHER_PAGE: u64 = crate::file_space_info::DEFAULT_PAGE_SIZE;

/// One dataset's append geometry, handed to the public append path so it can
/// slice a large call into aligned batches without materializing the whole
/// call's bytes first.
pub(crate) struct AppendGeometry {
    /// Elements per chunk along axis 0 (>= 1).
    pub(crate) chunk_elems: u64,
    /// Bytes per on-disk element, proven non-zero.
    pub(crate) element_size: NonZeroUsize,
    /// Current length along the unlimited dimension.
    pub(crate) current_dim: u64,
    /// Whether a filter pipeline applies (an in-place append then requires a
    /// chunk-aligned starting length).
    pub(crate) filtered: bool,
    /// Whole-chunk elements in one full batch (>= one chunk's worth), or
    /// [`u64::MAX`] when the session does not batch.
    pub(crate) full_batch_elems: u64,
}

/// Superblock consistency-flag bits raised while a SWMR writer is active: bit 0
/// (write access) | bit 2 (SWMR write access). Cleared on a clean close. Matches
/// the reference C library, h5py, and [`crate::File::open_swmr_writer`]. These
/// are the bits every open path checks — see [`file_lock::check_status_flags`].
const SWMR_WRITE_FLAGS: u32 = file_lock::WRITE_ACCESS | file_lock::SWMR_WRITE_ACCESS;

/// A dataset located once for [`Dataset::append`](crate::Dataset::append) (or the bounded
/// backend's immediate append), then maintained across appends. Mirrors the
/// append writer's per-dataset state.
pub(crate) struct LocatedState {
    pub(crate) loc: Located,
    /// The dataset's on-disk element datatype (for the append type check and the
    /// filter chunk context).
    pub(crate) datatype: Datatype,
    /// Spatial (rank-length) chunk dimensions in elements: `[chunk_elems]`.
    pub(crate) spatial: Vec<u64>,
    /// Bytes per element (datatype size), proven non-zero.
    pub(crate) element_size: NonZeroUsize,
    /// The re-encodable filter pipeline, when the dataset is filtered.
    pub(crate) pipeline: Option<FilterPipeline>,
    /// What the slots past the new dimension must hold when an append completes
    /// a partial chunk (issue #296).
    pub(crate) fill: crate::fill_value::PaddingFill,
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

/// A snapshot of a writable file's live space usage (issue #150).
///
/// This is the mutating-session counterpart of the read-only accounting on
/// [`File`](crate::File) ([`file_size`](crate::File::file_size) and
/// [`persisted_free_space`](crate::File::persisted_free_space)): it describes the
/// file *as the session currently holds it*, taken atomically at the moment of
/// the [`space_accounting`](crate::File::space_accounting) call.
///
/// It reflects the committed file plus any immediate in-place appends
/// ([`append`](crate::Dataset::append)), but **not** edits still
/// staged for the next [`commit`](crate::File::commit) — `create_group`,
/// `create_dataset`, `write_dataset`, `append_dataset`, `delete`, `copy`,
/// `copy_from`, and attribute edits change these figures only when they are
/// applied at commit. Use [`has_staged_edits`](crate::File::has_staged_edits) to
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
    /// ([`append`](crate::Dataset::append)) updates both together.
    ///
    /// It is not monotonic: [`commit`](crate::File::commit) can reclaim trailing
    /// free space and *shrink* the file. It can also exceed the superblock's
    /// recorded end-of-file address when the file was opened carrying unaccounted
    /// trailing bytes (the same slack [`File::file_size`](crate::File::file_size)
    /// surfaces), since opening does not rewrite that address.
    pub logical_size: u64,
    /// Total reusable free bytes the next allocation or [`commit`](crate::File::commit)
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

impl WriteEngine {
    /// Open an existing HDF5 file for in-place editing under an explicit
    /// file-locking policy.
    ///
    /// Reads the file into memory and retains a read/write handle. Under
    /// [`FileLocking::Enabled`] it takes an exclusive OS advisory lock so the file
    /// cannot be opened concurrently by another writer or reader; the lock is
    /// released automatically when the session is dropped or the process exits
    /// (including on a crash). Fails with [`Error::FileLocked`] if the file is
    /// already locked, or [`Error::EditUnsupported`] if the file is not a
    /// supported target; its documentation enumerates the exact requirements.
    /// `HDF5_USE_FILE_LOCKING` overrides the requested policy, as in the C
    /// library.
    pub fn open_with_locking<P: AsRef<Path>>(path: P, locking: FileLocking) -> Result<Self, Error> {
        Self::open_inner(path.as_ref(), Some(locking))
    }

    /// Open exactly as [`open_with_locking`](Self::open_with_locking) does, but
    /// behind an image that withholds its whole-file slice, so every read takes
    /// the [`Source`] path rather than the slice fast path.
    ///
    /// Distinct from [`open_rw_with_strategy`](Self::open_rw_with_strategy), which withholds the
    /// slice *and* the residency: this one still mirrors the file, so a test can
    /// compare the two read forms on a file the bounded open would refuse.
    #[cfg(test)]
    pub(crate) fn open_source_only(path: &Path) -> Result<Self, Error> {
        Self::open_imaged(path, Some(FileLocking::Enabled), |handle, _len| {
            Ok(Box::new(crate::image::SourceOnlyImage::new(
                Self::read_mirror(handle)?,
            )))
        })
    }

    /// Open a bounded session whose image counts the bytes read through it, so a
    /// test can assert that an operation touches only a small part of the file.
    /// The counter is shared with the caller.
    #[cfg(test)]
    pub(crate) fn open_bounded_counting(
        path: &Path,
        read_bytes: std::sync::Arc<std::sync::atomic::AtomicU64>,
    ) -> Result<Self, Error> {
        let mut session = Self::open_imaged(path, Some(FileLocking::Enabled), |handle, len| {
            Ok(Box::new(crate::image::CountingImage::new(
                Box::new(HandleImage::new(
                    handle,
                    len,
                    crate::source::MetadataCacheConfig::disabled(),
                )),
                read_bytes,
                std::sync::Arc::default(),
            )))
        })?;
        session.batched_appends = true;
        Ok(session)
    }

    /// Open a session under `policy` whose image counts the `fsync`s issued
    /// through it, so a test can assert what a write path actually costs. The
    /// counter is shared with the caller.
    ///
    /// It takes the bounded backing and its flags, so the session matches what
    /// [`File::open_rw`](crate::File::open_rw) builds for a latest-format file;
    /// the barrier sites live on the engine, above the choice of image, so one
    /// backing exercises them all.
    #[cfg(test)]
    pub(crate) fn open_sync_counting(
        path: &Path,
        policy: SyncPolicy,
        syncs: std::sync::Arc<std::sync::atomic::AtomicU64>,
    ) -> Result<Self, Error> {
        let mut session = Self::open_imaged(path, Some(FileLocking::Enabled), |handle, len| {
            Ok(Box::new(crate::image::CountingImage::new(
                Box::new(HandleImage::new(
                    handle,
                    len,
                    crate::source::MetadataCacheConfig::disabled(),
                )),
                std::sync::Arc::default(),
                syncs,
            )))
        })?;
        session.batched_appends = true;
        session.bounded = true;
        session.set_sync_policy(policy);
        Ok(session)
    }

    /// Open an existing file for read-write editing under `strategy`: the one
    /// place that picks between the bounded backing (a [`HandleImage`] keeping no
    /// whole-file mirror, so resident memory is the metadata-cache budget plus
    /// whatever is being parsed) and the whole-file mirror. Backs
    /// [`File::open_rw`](crate::File::open_rw), which selects between them by
    /// the strategy it is given.
    ///
    /// The eligibility rules are checked here rather than deferred, because a
    /// caller who asked for bounded memory cannot be silently given the mirror
    /// instead. Under [`MemoryStrategy::Bounded`] a file the bounded engine
    /// cannot edit is refused up front; [`MemoryStrategy::Auto`] opts in to
    /// falling back to the mirror instead (issue #198, steps 3 and 4).
    ///
    /// Only a *bounded-only* limitation is worth falling back for — see
    /// [`bounded_only_limitation`]. Non-8-byte offsets or lengths are refused by
    /// [`open_imaged`](Self::open_imaged) for both engines, and so is an
    /// unsupported superblock version; a paged file without persisted free space
    /// is refused below for both.
    pub(crate) fn open_rw_with_strategy(
        path: &Path,
        cache: MetadataCacheConfig,
        locking: FileLocking,
        strategy: MemoryStrategy,
    ) -> Result<Self, Error> {
        if strategy == MemoryStrategy::Mirrored {
            return Self::open_with_locking(path, locking);
        }
        let mut session = Self::open_imaged(path, Some(locking), |handle, len| {
            Ok(Box::new(HandleImage::new(handle, len, cache)))
        })?;
        session.batched_appends = true;
        session.bounded = true;
        // Refusals that apply to *both* backings come first, or falling back for a
        // bounded-only limitation would skip them and hand back a session that
        // cannot commit. A paged file with no persisted managers has no on-disk
        // record of which pages hold metadata and which hold raw data, so nothing
        // can keep the pages segregated; the staged commit refuses it too, so
        // deferring would only trade this error for the same one later, with work
        // already staged. A userblock is one way to reach this state without the
        // file saying `persist = false`: persisted free space is declined for a
        // non-zero base address, which leaves the managers unseeded all the same.
        if session.paged.is_some() && session.persist.is_none() {
            return Err(Error::EditUnsupported(
                "read-write access to a paged file (H5F_FSPACE_STRATEGY_PAGE) requires \
                 persisted free space; recreate the file with \
                 with_file_space_strategy(FileSpaceStrategy::Page, true, ..) to grow it in place",
            ));
        }
        if let Some(reason) = bounded_only_limitation(&session) {
            if strategy == MemoryStrategy::Bounded {
                return Err(Error::EditUnsupported(reason));
            }
            // Release the handle and its exclusive lock before reopening, or the
            // mirrored open would contend with the probe we are discarding —
            // fatally so on Windows, where the OS lock is mandatory. Dropping a
            // bare `WriteEngine` writes nothing: the free-space finalize that a
            // dropped writer owes lives on `FileInner`, which this is not yet.
            // Another writer can take the lock in that window; the reopen then
            // reports `Error::FileLocked`, which is the truthful answer.
            drop(session);
            return Self::open_with_locking(path, locking);
        }
        Ok(session)
    }

    /// Open an existing file for SWMR (single-writer/multiple-reader) writing:
    /// take **no** OS lock at all and raise the superblock's SWMR-write
    /// consistency flag. Backs [`File::open_swmr_writer`](crate::File::open_swmr_writer).
    ///
    /// The no-lock is unconditional — `lock = None` never reaches
    /// `acquire_exclusive`, so `HDF5_USE_FILE_LOCKING` cannot reintroduce a lock
    /// that would block the concurrent readers SWMR exists to permit (fatally so
    /// on Windows, where OS locks are mandatory). Requires a latest-format
    /// (version-3 superblock) file with no userblock and no persisted
    /// free-space, so the superblock can be rewritten in place.
    ///
    /// The version-3 requirement is the C library's (`H5F__super_read`: "superblock
    /// version for SWMR is less than 3"), and it is what keeps the SWMR-write flag
    /// meaningful: neither library reads the status-flags byte back on an older
    /// superblock, so a flag raised there would announce a live writer to nobody.
    /// This crate's writer emits version 3, so no file it produces is affected.
    pub(crate) fn open_swmr_writer<P: AsRef<Path>>(
        path: P,
        sync_policy: SyncPolicy,
    ) -> Result<Self, Error> {
        let mut session = Self::open_inner(path.as_ref(), None)?;
        // Before the flag write below, which is a durability point like any
        // other: a caller who asked for no `fsync` gets none, and the flag still
        // reaches every other process, which reads it from the operating system.
        session.set_sync_policy(sync_policy);
        if session.superblock.version < 3
            || session.superblock.base_address != 0
            || session.persist.is_some()
        {
            return Err(Error::SwmrAppendUnsupported(
                "SWMR writing requires a latest-format file (v3 superblock) with no userblock \
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
        self.write_at(self.sb_sig_off, &bytes)?;
        self.barrier_data()?;
        Ok(())
    }

    /// Shared open path. `lock = Some(policy)` acquires an exclusive OS lock under
    /// that policy (the ordinary read-write session); `lock = None` takes no lock
    /// at all (the SWMR writer — see [`open_swmr_writer`](Self::open_swmr_writer)).
    fn open_inner(path: &Path, lock: Option<FileLocking>) -> Result<Self, Error> {
        Self::open_imaged(path, lock, |handle, _len| {
            Ok(Box::new(Self::read_mirror(handle)?))
        })
    }

    /// Read `handle` whole into a [`MirrorImage`]. The one place the engine
    /// still assumes it can hold the file, kept behind a named constructor so
    /// the mirrorless opens visibly do not call it.
    ///
    /// `read_to_end` reads from the handle's *current* cursor, and the open path
    /// has already read the superblock through it, so this rewinds first. Reading
    /// from wherever the last read landed would mirror a truncated file — with no
    /// error to say so, since a short mirror is a valid `Vec<u8>`.
    fn read_mirror(mut handle: fs::File) -> Result<MirrorImage, Error> {
        handle.seek(SeekFrom::Start(0)).map_err(Error::Io)?;
        let mut data = Vec::new();
        handle.read_to_end(&mut data).map_err(Error::Io)?;
        Ok(MirrorImage::new(handle, data))
    }

    /// Shared open path over any backing: acquire the handle (and, when asked,
    /// the exclusive lock), parse and validate the superblock through a borrowed
    /// view of the handle, and only then let `build` decide how the bytes are
    /// held.
    ///
    /// Every refusal comes before `build`, because `build` may read the whole
    /// file: reaching a refusal after it would spend `O(file size)` on a file
    /// that is then rejected — a 20 GB flagged file read into memory and thrown
    /// away. The superblock reads themselves are a few bounded windows either
    /// way, so nothing is read twice.
    ///
    /// `build` receives the file's length as well as the handle because a
    /// mirrorless image has to be told its end-of-file — it has no buffer whose
    /// length implies it.
    ///
    /// Nothing below this point reads the file as a slice, which is what lets
    /// one engine open a whole-file mirror, a mirrorless handle, and (in tests)
    /// a mirror that withholds its slice.
    fn open_imaged(
        path: &Path,
        lock: Option<FileLocking>,
        build: impl FnOnce(fs::File, u64) -> Result<Box<dyn FileImage>, Error>,
    ) -> Result<Self, Error> {
        let handle = fs::OpenOptions::new()
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
        let len = handle.metadata().map_err(Error::Io)?.len();
        // Read the superblock through the handle itself, before any image owns
        // it. `probe` borrows, so it is gone by the time `build` takes the
        // handle; it leaves the handle's cursor wherever its last read ended,
        // which is why the mirror positions the handle before reading it whole.
        let probe = crate::image::BorrowedHandle::new(&handle, len);
        let sb_sig_off = signature::find_signature_in(&probe)?.to_usize()?;
        let mut superblock = Superblock::parse_from_source(&probe, sb_sig_off as u64)?;

        if superblock.version > 3 {
            return Err(Error::EditUnsupported("unsupported superblock version"));
        }
        // Refuse a file a writer already holds, before anything is mutated. This
        // is the one exclusion the OS lock above cannot make: a SWMR writer
        // takes no lock, so its file is lock-free but flagged (issue #245).
        file_lock::check_status_flags(&superblock, file_lock::OpenIntent::Write, path)?;
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
        // the link-graph walk index the image correctly. It is converted back to a
        // stored (base-relative) address only when the superblock is serialized on
        // commit.
        superblock.root_group_address = superblock
            .root_group_address
            .checked_add(superblock.base_address)
            .ok_or(FormatError::OffsetOverflow {
                offset: superblock.root_group_address,
                length: superblock.base_address,
            })?;

        // Everything that can refuse this file has run; only now is it worth
        // holding the bytes.
        let image = build(handle, len)?;

        let mut session = Self {
            image,
            sb_sig_off,
            superblock,
            staged: StagedEdits::default(),
            appender_claims: Vec::new(),
            next_appender_token: 0,
            appender_commit_in_progress: false,
            free: FreeList::new(),
            persist: None,
            located: HashMap::new(),
            swmr_mode: false,
            paged: None,
            committed: false,
            resolved: HashMap::new(),
            batched_appends: false,
            bounded: false,
            libver_ceiling: None,
            fsm_len: len,
            repointed: false,
            sync_policy: SyncPolicy::Always,
        };
        // If the file persists its free space, seed the free list from the
        // on-disk managers and arm persistence for future commits. Best-effort:
        // an unreadable or non-persisting extension simply leaves the session in
        // the default, non-persisting mode.
        session.load_persisted_free_space();
        // Gather this session's writes, now that the page size the file was laid
        // out on is known. Only an exclusively locked session: `lock = None` is
        // the SWMR writer, whose concurrent readers observe the order its ordered
        // phases become visible in, and so must see every write as it is made
        // (issue #288).
        //
        // `lock.is_some()` is a proxy for "not the SWMR writer", used because
        // `swmr_mode` is not set until after this. It is exact only while
        // `open_swmr_writer` is the sole lock-free entry point: a future one that
        // took a lock would silently *gain* gathering, which is the direction
        // that costs a reader. The other direction — a lock-free non-SWMR open
        // losing gathering — costs only writes. Whoever adds such an entry point
        // owns this condition.
        if lock.is_some() {
            let page_size = session.gather_page_size();
            session
                .image
                .set_write_buffering(WriteBuffering::Operation {
                    page_size,
                    max_bytes: WRITE_GATHER_BYTES,
                })?;
        }
        Ok(session)
    }

    /// The page this session's writes are merged within: the file's own
    /// file-space page size when it is paged, and the format's default otherwise.
    fn gather_page_size(&self) -> u64 {
        self.paged
            .as_ref()
            .map_or(DEFAULT_GATHER_PAGE, |pg| pg.page_size)
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
        // The extension address is stored relative to the base address, so it is
        // shifted to an absolute file offset before the header is read. This is a
        // no-op on the base-0 file every path below the userblock check sees, but
        // that check itself needs the strategy of a *userblock* file.
        let Ok(ext_addr) = ext_rel
            .checked_add(self.superblock.base_address)
            .ok_or(())
            .and_then(|a| usize::try_from(a).map_err(|_| ()))
        else {
            return;
        };
        let Some(info) = self.extension_fsinfo(ext_addr) else {
            return;
        };
        // Free-space reuse and persistence are not yet base-address aware: the
        // persisted section addresses (and the extension/manager block walk below)
        // are read as absolute, so on a userblock file they would seed `self.free`
        // with wrong regions that a later allocation could hand out into live
        // data. Leave persistence off for such a file — the on-disk managers stay
        // untouched and valid, this session simply appends rather than reusing.
        //
        // A *paged* userblock file is a different matter: appending without page
        // awareness would mix metadata and raw data in its pages and leave its end
        // of allocation unaligned, quietly producing a file that still claims the
        // paged strategy but no longer satisfies it. Install the paged marker
        // without persistence so the commit refusal below catches it, which is the
        // same rule a paged non-persisting file already takes.
        if self.superblock.base_address != 0 {
            if info.strategy == FileSpaceStrategy::Page && info.page_size > 0 {
                self.paged = Some(PagedEdit::new(info.page_size));
            }
            return;
        }
        // Record the paged strategy regardless of the persist flag: a paged commit
        // needs page-aware bookkeeping, and a paged file that does not persist its
        // free space is refused outright (see `PagedEdit` and the commit refusal).
        //
        // A zero page size is refused rather than installed: every page calculation
        // divides by it, so a corrupt or hostile file declaring `Page` with a page
        // size of 0 would panic the editor. Leaving `paged` unset makes the file
        // take the ordinary flat path, which needs no page geometry.
        let paged = info.strategy == FileSpaceStrategy::Page && info.page_size > 0;
        if paged {
            self.paged = Some(PagedEdit::new(info.page_size));
        }
        if !info.persist {
            return;
        }
        let os = self.superblock.offset_size;
        let file_len = self.image.len();

        // Seed the free list(s) with every persisted section (addresses are stored
        // relative to the base address, which this editor requires to be 0).
        // Defensive against a malformed or corrupt manager: skip a section that is
        // empty, runs past end-of-file, or overlaps one already taken. A
        // well-formed file (this crate's or the C library's) has none of these;
        // tolerating them keeps a bad file from seeding a bogus or double-counted
        // free region that a later commit would hand out into live data.
        if paged {
            // A paged file's free space is segregated across per-page-type
            // managers, so read each slot on its own and keep the page type its
            // slot implies ([`PagedEdit::slot_list`]). Flattening them (as the
            // non-paged path below does) would lose exactly the distinction the
            // commit has to preserve. A section whose slot does not settle its page
            // type is recorded but never handed out.
            let page_size = info.page_size;
            let mut tagged: Vec<(FreeSection, Option<PageType>)> = Vec::new();
            for (slot, &m) in info.manager_addrs.iter().enumerate() {
                if m == UNDEF {
                    continue;
                }
                let Ok(sections) =
                    free_space_manager::read_persisted_sections_source(&self.image(), &[m], 0, os)
                        .map(|(sections, _)| sections)
                else {
                    continue;
                };
                for s in sections {
                    let ty = PagedEdit::slot_list(slot, s.addr, s.size, page_size);
                    tagged.push((s, ty));
                }
            }
            tagged.sort_by_key(|(s, _)| s.addr);
            let mut prev_end = 0u64;
            for (s, ty) in tagged {
                let Some(end) = s.addr.checked_add(s.size) else {
                    continue;
                };
                if s.size == 0 || end > file_len || s.addr < prev_end {
                    continue;
                }
                prev_end = end;
                let pg = self
                    .paged
                    .as_mut()
                    .expect("the paged state was just installed");
                match ty {
                    Some(ty) => {
                        PagedEdit::route_free(&mut pg.meta, &mut pg.raw, s.addr, s.size, ty)
                    }
                    None => pg.unclassified.free(s.addr, s.size),
                }
            }
        } else if let Ok(mut sections) = free_space_manager::read_persisted_sections_source(
            &self.image(),
            &info.manager_addrs,
            0,
            os,
        )
        .map(|(sections, _)| sections)
        {
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
            let Ok(hdr_len) = fshd_len(os).to_usize() else {
                continue;
            };
            let Ok(fshd) = self.image().read_metadata_at(m, hdr_len) else {
                continue;
            };
            if let Ok(h) = FsmHeader::parse(&fshd, os) {
                // `FsmHeader::parse` succeeding guarantees the header's own bytes
                // are present, so the FSHD extent is in-bounds; validate the
                // section-info extent before recording it, so a malformed
                // `fsse_used` can't later free a region running past end-of-file.
                old_blocks.push((m, fshd_len(os)));
                if h.fsse_addr != UNDEF
                    && h.fsse_addr
                        .checked_add(h.fsse_used)
                        .is_some_and(|end| end <= file_len)
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
        let oh =
            ObjectHeader::parse_from_source(&self.image(), ext_addr as u64, os, ls, base).ok()?;
        let msg = oh
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::FileSpaceInfo)?;
        FileSpaceInfo::parse(&msg.data, os, ls).ok()
    }

    /// Stage a new dataset, added on the next [`commit`](Self::commit).
    ///
    /// `path` is the dataset's full path; everything before the last component
    /// names the parent group, which must exist (or be created in this session).
    /// `builder` is the same [`DatasetBuilder`] [`FileBuilder`](crate::FileBuilder)
    /// uses, configured by the caller; its name is taken from `path`, so the two
    /// cannot disagree.
    ///
    /// The builder is passed in finished rather than handed out as a `&mut` into
    /// this engine. That is deliberate: a borrow into the engine forces the
    /// caller to hold it — and, above this layer, its lock — for as long as the
    /// builder is being configured, which is what let a user closure deadlock
    /// against the same file it was reading (issue #200).
    ///
    /// The dataset may be contiguous or chunked, and chunked datasets may be
    /// filtered (`with_deflate`, `with_shuffle`, `with_fletcher32`,
    /// `with_scale_offset`, `with_zfp`) and/or extensible (`with_maxshape`). An
    /// empty (zero-element) dataset is supported under either storage, a
    /// provenance dataset (`with_provenance`) is supported, and a
    /// contiguous dataset may carry variable-length attributes, a
    /// variable-length-string payload (`with_vlen_strings`), or path-resolved
    /// object-reference elements (`with_path_references`; chunking any of
    /// these is not supported, and dense attributes remain unsupported).
    pub(crate) fn stage_created_dataset(
        &mut self,
        path: &str,
        mut builder: DatasetBuilder,
    ) -> Result<(), Error> {
        self.refuse_if_claimed(&split_path(path))?;
        let mut comps = split_path(path);
        builder.name = comps.pop().unwrap_or_default();
        self.staged
            .datasets
            .push((comps, flatten_dataset(builder)?));
        Ok(())
    }

    /// Stage an in-place overwrite of an **existing** dataset's values (the HDF5
    /// `H5Dwrite` whole-dataset write), applied on the next
    /// [`commit`](Self::commit).
    ///
    /// `path` is the full path of a dataset that must already exist; `builder`
    /// supplies the replacement data and is named from `path`, as in
    /// [`stage_created_dataset`](Self::stage_created_dataset).
    ///
    /// This is a *value* overwrite, not a reshape or retype: the new data's
    /// datatype and shape must match the on-disk dataset's exactly (byte-for-byte
    /// after serialization, so endianness and compound layout must agree), or
    /// `commit` reports [`Error::EditUnsupported`]. Contiguous, compact, and
    /// chunked (including filtered) datasets are all supported; the dataset's
    /// existing chunk geometry, filter pipeline, and chunk index are taken from the
    /// on-disk header. A chunk index this engine cannot enumerate (a version-2
    /// B-tree) is refused. Partial / sub-region writes are out of scope — the
    /// whole dataset is replaced.
    ///
    /// What the builder alone can rule out —
    /// [`refuse_unsupported_overwrite`](Self::refuse_unsupported_overwrite) —
    /// is refused *here* rather than at `commit`.
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
    pub(crate) fn stage_dataset_write(
        &mut self,
        path: &str,
        mut builder: DatasetBuilder,
    ) -> Result<(), Error> {
        self.refuse_if_claimed(&split_path(path))?;
        let comps = split_path(path);
        // Before the flatten below, which would otherwise report the root as a
        // dataset with an empty name: the leaf of an empty path is what names
        // the builder.
        let Some(leaf) = comps.last() else {
            return Err(Error::EditUnsupported("cannot overwrite the root group"));
        };
        builder.name = leaf.clone();
        let fd = flatten_dataset(builder)?;
        Self::refuse_unsupported_overwrite(&fd)?;
        self.staged.writes.push((comps, fd));
        Ok(())
    }

    /// Stage an append of new elements to an **existing** chunked, unlimited
    /// dataset, applied on the next [`commit`](Self::commit).
    ///
    /// `path` names a dataset that must already exist; `builder` supplies the
    /// elements to add via its typed / generic / raw `append_*` methods.
    ///
    /// Unlike [`stage_dataset_write`](Self::stage_dataset_write) (a value
    /// overwrite that forbids any shape change) this **grows** the dataset along
    /// its first (axis-0) dimension. It works on **filtered** datasets: the
    /// appended chunks are compressed through the dataset's own on-disk filter
    /// pipeline (deflate / shuffle / fletcher32 / scale-offset / LZF, and ZFP
    /// with the `zfp` feature), and the pipeline, datatype, fill value, and attributes are
    /// preserved verbatim. Appends of any length are supported — when the
    /// dataset's current length is not a whole multiple of the chunk length, the
    /// single trailing partial chunk is read, extended, and re-encoded; every
    /// other existing chunk is carried by metadata alone, so the existing data is
    /// not rewritten and the file does not grow by the whole dataset per append.
    ///
    /// This does **not** use SWMR and sets no consistency flag. Like every other
    /// staged edit it commits by appending the new chunks and a rebuilt
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
    pub(crate) fn stage_dataset_append(
        &mut self,
        path: &str,
        builder: AppendBuilder,
    ) -> Result<(), Error> {
        let comps = split_path(path);
        self.refuse_if_claimed(&comps)?;
        self.staged.appends.push((comps, builder));
        Ok(())
    }

    /// Register a live [`BufferedAppender`](crate::BufferedAppender) on `path`
    /// (`None` for a handle reached by object reference), returning the token
    /// that releases it.
    ///
    /// Refused when the claim could not be honored: a second appender on the
    /// same dataset would interleave the two buffers a chunk at a time, and a
    /// staged edit already pending on that path — or any staged edit at all, when
    /// the appender still owes a realignment — is one the appender's own flush
    /// would later refuse.
    pub(crate) fn claim_for_appender(
        &mut self,
        path: Option<&str>,
        needs_commit: bool,
    ) -> Result<u64, Error> {
        let path = path.map(split_path);
        if self
            .appender_claims
            .iter()
            .any(|c| claims_conflict(c.path.as_deref(), path.as_deref()))
        {
            return Err(Error::EditUnsupported(
                "this dataset already has a live buffered appender; two of them would interleave \
                 their buffers a chunk at a time",
            ));
        }
        let blocked = if needs_commit {
            self.has_staged_edits()
        } else {
            match path.as_deref() {
                Some(p) => self.append_conflicts_with_pending(p),
                None => self.has_staged_edits() || self.committed,
            }
        };
        if blocked {
            return Err(Error::EditUnsupported(
                "this session holds staged edits that would stop a buffered appender from \
                 flushing; commit or discard them before opening one",
            ));
        }
        let token = self.next_appender_token;
        self.next_appender_token += 1;
        self.appender_claims.push(AppenderClaim {
            token,
            path,
            needs_commit,
        });
        Ok(token)
    }

    /// Record whether the claim still owes a staged realignment. The appender
    /// calls this after every write, since flushing a partial trailing chunk on
    /// a filtered dataset puts the debt back.
    pub(crate) fn set_appender_needs_commit(&mut self, token: u64, needs_commit: bool) {
        if let Some(c) = self.appender_claims.iter_mut().find(|c| c.token == token) {
            c.needs_commit = needs_commit;
        }
    }

    /// Drop a claim. Called from the appender's `Drop`, after its final flush.
    pub(crate) fn release_appender_claim(&mut self, token: u64) {
        self.appender_claims.retain(|c| c.token != token);
    }

    /// Run `f` with this session's claims suspended, so an appender's own staged
    /// realignment is not refused by its own claim.
    pub(crate) fn within_appender_commit<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        let prev = self.appender_commit_in_progress;
        self.appender_commit_in_progress = true;
        let out = f(self);
        self.appender_commit_in_progress = prev;
        out
    }

    /// Refuse a staged edit at `path` that a live appender could not survive.
    ///
    /// An appender holds elements a caller has already been told were accepted,
    /// and only its own flush can write them; that flush goes through the
    /// immediate append path, which refuses a dataset with a staged edit on it or
    /// an ancestor. Left unchecked, staging such an edit turns accepted data into
    /// data lost silently in `Drop`. Refusing here moves the failure to the call
    /// that creates the conflict, where there is someone to report it to.
    fn refuse_if_claimed(&self, path: &[String]) -> Result<(), Error> {
        if self.appender_commit_in_progress {
            return Ok(());
        }
        let conflicts = self.appender_claims.iter().any(|c| match &c.path {
            // A claim owing a realignment needs a commit, and a commit refuses to
            // run beside unrelated staged edits, so it conflicts with every path.
            _ if c.needs_commit => true,
            Some(p) => paths_overlap(p, path),
            None => true,
        });
        if conflicts {
            return Err(Error::EditUnsupported(
                "this dataset has a live buffered appender holding elements only its own flush \
                 can write, and this edit would stop that flush; finish or discard the appender \
                 first",
            ));
        }
        Ok(())
    }

    /// Whether this session is the SWMR writer, whose append rules are a strict
    /// subset of the ordinary ones (see `append_inplace_gathered`).
    pub(crate) fn is_swmr(&self) -> bool {
        self.swmr_mode
    }

    /// Whether any staged tree edit is still uncommitted. In-place appends
    /// ([`append_inplace_gathered`](Self::append_inplace_gathered)) are applied immediately and are
    /// never staged, so they never affect this; it reflects only edits awaiting
    /// [`commit`](Self::commit) — `create_group`, `create_dataset`,
    /// `write_dataset`, `append_dataset`, group and dataset attribute edits,
    /// `delete`, `copy`, and `copy_from`. Dropping the session silently discards
    /// any staged edits.
    ///
    /// A [`commit`](Self::commit) that refuses leaves this answering `true`: the
    /// staged set it declined to apply is put back exactly as it was.
    pub fn has_staged_edits(&self) -> bool {
        !self.staged.is_empty()
    }

    /// Run `f`, and if it fails, drop whatever it managed to stage.
    ///
    /// One caller-facing call can stage many edits — `create_group_with` stages
    /// a group, its attributes and its whole subtree — and each is validated as
    /// it is staged, so the fifth dataset can be refused after four have been
    /// recorded. Without this, such a call would return `Err` having changed the
    /// session anyway, which is the half of issue #316 that lives one level
    /// above `commit`: a refused operation must cost the session nothing.
    pub(crate) fn stage_atomically<R>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<R, Error>,
    ) -> Result<R, Error> {
        let mark = self.staged.mark();
        let result = f(self);
        if result.is_err() {
            self.staged.rewind(mark);
        }
        result
    }

    /// This session's file image as one slice, when its backing holds the whole
    /// file in memory; `None` for a file-backed image.
    ///
    /// The slice reflects committed state plus immediate in-place appends, not
    /// edits still staged for `commit`. The owned read-write
    /// [`File`](crate::File) uses it to serve reads by borrowing rather than
    /// copying, and falls back to [`image`](Self::image) when it is absent.
    pub(crate) fn image_slice(&self) -> Option<&[u8]> {
        self.image.as_slice()
    }

    /// A random-access [`Source`] view of this session's file image, for the
    /// parsers the edit engine drives.
    ///
    /// Every read the engine performs against the file goes through this, so the
    /// image needs to be no more than a source of bytes — whole-file mirror or
    /// not — without the parsers knowing which (issue #198).
    pub(crate) fn image(&self) -> &dyn Source {
        self.image.as_ref()
    }

    /// This session's parsed superblock, with `root_group_address` normalized to
    /// an absolute file offset (the open-time convention) and `base_address` the
    /// userblock size. A relocating commit updates it, so a caller holding a
    /// clone from an earlier moment may be reading a stale root.
    pub(crate) fn superblock(&self) -> &Superblock {
        &self.superblock
    }

    /// The on-disk format this session writes, read from the file it opened
    /// rather than chosen fresh, since a commit preserves the superblock version
    /// it found.
    ///
    /// This decides the *contiguous* data-layout message version, where the
    /// older number costs nothing: versions 3 and 4 have identical bodies there,
    /// so a dataset added to a 1.8 file stays readable by a 1.8 library.
    ///
    /// It does not gate chunked storage. A chunked dataset needs the version 4
    /// layout and a 1.10 chunk index whatever the superblock says, so adding one
    /// to an older file does make that file need 1.10 — deliberately, since the
    /// alternative is a version 1 B-tree index this crate does not write, and
    /// refusing instead would take away in-place editing of every file the C
    /// library wrote with its own default bounds. `tests/edit_crosscheck.rs`
    /// covers exactly that case (issue #101).
    ///
    /// That is a default, not a verdict: a caller who needs the file to stay
    /// loadable by an old reader asks for it with
    /// [`FileAccessProperties::with_libver_bounds`], which sets
    /// [`libver_ceiling`](Self::libver_ceiling) and turns the addition into a
    /// refusal at commit rather than a silent format bump.
    ///
    /// [`FileAccessProperties::with_libver_bounds`]: crate::FileAccessProperties::with_libver_bounds
    pub(crate) fn libver(&self) -> LibVer {
        LibVer::from_superblock_version(self.superblock.version)
    }

    /// Constrain what this session may add, from the fapl's library-version
    /// bounds. See [`libver_ceiling`](Self::libver_ceiling).
    ///
    /// Resolved through [`LibVer::resolve_writable`], the same rule the
    /// whole-file writer applies, so bounds admitting no format this crate
    /// writes are refused here as they are there rather than silently ignored on
    /// the editing path.
    pub(crate) fn set_libver_bounds(
        &mut self,
        bounds: Option<(LibVer, LibVer)>,
    ) -> Result<(), Error> {
        self.libver_ceiling = match bounds {
            Some(_) => Some(LibVer::resolve_writable(bounds).map_err(Error::Format)?),
            None => None,
        };
        Ok(())
    }

    /// Refuse staged content the session's [`libver_ceiling`](Self::libver_ceiling)
    /// cannot express, before the commit writes anything.
    ///
    /// Only chunked storage is at stake: `build_chunked_dataset_oh` writes a
    /// version 4 data-layout message and a 1.10 chunk index unconditionally,
    /// while everything else this session adds is expressible in both formats
    /// (`libver` already picks the contiguous layout version). A filter or an
    /// unlimited dimension arrives as chunked storage, so they are covered here
    /// too.
    fn check_libver_admits<'a>(
        &self,
        datasets: impl IntoIterator<Item = &'a FlatDataset>,
    ) -> Result<(), Error> {
        let Some(ceiling) = self.libver_ceiling else {
            return Ok(());
        };
        if ceiling >= LibVer::V110 {
            return Ok(());
        }
        let chunked = datasets
            .into_iter()
            .any(|fd| fd.chunk_options.is_chunked() || fd.maxshape.is_some());
        if chunked {
            return Err(Error::Format(FormatError::LibverTooOldForContent {
                content: "a chunked, filtered, or resizable dataset",
                needs: LibVer::V110.name(),
                writing: ceiling.name(),
            }));
        }
        Ok(())
    }

    /// Which backend this session resolved to: [`Bounded`] when it reads through
    /// a handle, [`Mirrored`] when it holds a whole-file image.
    ///
    /// [`Bounded`]: EditBacking::Bounded
    /// [`Mirrored`]: EditBacking::Mirrored
    pub(crate) fn edit_backing(&self) -> EditBacking {
        if self.bounded {
            EditBacking::Bounded
        } else {
            EditBacking::Mirrored
        }
    }

    /// A snapshot of this session's live space usage — the current file size and
    /// the free space it can reuse — as a [`SpaceAccounting`].
    ///
    /// This is the mutating-session analogue of the read-only accounting on
    /// [`File`](crate::File): it answers "how big is the file right now, and how
    /// much space can be reused before it must grow?" from the session's own live
    /// state. The snapshot reflects the committed file plus any immediate in-place
    /// appends ([`append_inplace_gathered`](Self::append_inplace_gathered)) but excludes edits still
    /// staged for the next [`commit`](Self::commit); see [`SpaceAccounting`] for
    /// the field-by-field semantics and [`has_staged_edits`](Self::has_staged_edits)
    /// for detecting pending work.
    ///
    /// On a paged file (`H5F_FSPACE_STRATEGY_PAGE`) the reported regions are the
    /// union of the per-page-type managers. They are recorded and handed back to
    /// the reference library, but a commit does not draw on them: a hole belongs
    /// to one page type, and reusing it for the other kind of allocation would
    /// re-mix its page, so such a commit appends instead.
    ///
    /// ```no_run
    /// use hdf5_pure::File;
    ///
    /// let file = File::open_rw("existing.h5")?;
    /// let acct = file.space_accounting()?;
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
        // A paged file tracks its free space per page type; report the union, since
        // the caller wants one total rather than a per-manager breakdown.
        let reusable_free_space = match &self.paged {
            Some(pg) => pg.reusable_sections(),
            None => self.free.sections(),
        };
        let reusable_free_bytes = reusable_free_space.iter().map(|(_, len)| len).sum();
        SpaceAccounting {
            logical_size: self.image.len(),
            reusable_free_bytes,
            reusable_free_space,
        }
    }

    /// The shared Extensible-Array append engine's view of this session: the
    /// image, the superblock, and the paged-file state, paired as [`EditStore`].
    ///
    /// Takes `&mut self`, so a caller that also needs [`located`](Self::located)
    /// borrowed at the same time must destructure the fields itself rather than
    /// call this.
    fn store(&mut self) -> EditStore<'_> {
        EditStore {
            image: self.image.as_mut(),
            superblock: &mut self.superblock,
            sb_sig_off: self.sb_sig_off,
            paged: self.paged.as_mut(),
            sync_policy: self.sync_policy,
        }
    }

    /// Adopt the fapl's `fsync` cadence. Called once, on the funnel every
    /// read-write open passes through, before the session is handed out.
    pub(crate) fn set_sync_policy(&mut self, policy: SyncPolicy) {
        self.sync_policy = policy;
    }

    /// The cadence this session adopted. Nothing in the crate branches on this —
    /// [`barrier`](Self::barrier) does that — but the fapl reaching the engine is
    /// otherwise invisible from outside, and an entry point that forgot to pass
    /// it on would look exactly like one that did.
    #[cfg(test)]
    pub(crate) fn sync_policy(&self) -> SyncPolicy {
        self.sync_policy
    }

    /// Override how long this session's writes may sit in memory.
    ///
    /// The image is private to this module, and the only caller outside it is
    /// [`crate::crash_replay`], which sweeps the *same* workload under gathering
    /// and without it. That comparison is the answer to the obvious worry about
    /// making writes larger, so it is worth an accessor; nothing outside a test
    /// build has one.
    #[cfg(test)]
    pub(crate) fn set_write_buffering(&mut self, mode: WriteBuffering) -> Result<(), Error> {
        self.image.set_write_buffering(mode)
    }

    /// Every write this session issued, as `(offset, length)` in the order it
    /// went out. Same reasoning as [`set_write_buffering`](Self::set_write_buffering):
    /// the image is private, and a test that asks whether a publish left the
    /// engine as one write has nowhere else to look.
    #[cfg(test)]
    pub(crate) fn issued_write_order(&self) -> Vec<(u64, u64)> {
        self.image.issued_write_order()
    }

    /// The durability barrier a write path calls for, data and metadata both —
    /// issued unless this session's [`SyncPolicy`] leaves the `fsync` cadence to
    /// the application.
    ///
    /// Every such point routes through here rather than touching the image, so
    /// the policy is honored by construction at sites yet to be written.
    ///
    /// A barrier is an **ordering** point first and a durability point second,
    /// and only the second half is the policy's to skip. The writes gathered
    /// before it are issued here whatever the policy says, because the image
    /// issues gathered writes in address order: leave them gathered across a
    /// barrier and a commit's superblock — address 0 — goes to the disk *before*
    /// the content it names, which is the one order the whole tail is built to
    /// avoid. A write that then fails, or a process that dies mid-flush, would
    /// leave a superblock naming bytes that are not in the file, where before it
    /// left the previous file intact (issue #288). No buffering mode this engine
    /// takes holds across one; a mode that did would be trading exactly that
    /// guarantee, which is why [#308] proposes to pair one with an on-disk mark
    /// that makes the resulting file refuse to open.
    ///
    /// [#308]: https://github.com/stephenberry/hdf5-pure/issues/308
    ///
    /// The teardown barrier is deliberately *not* one of these points:
    /// [`File::close`](crate::File::close) and `FileInner::drop` write after the
    /// last barrier a caller could have asked for, so they take
    /// [`force_sync`](Self::force_sync) instead.
    fn barrier(&mut self) -> Result<(), Error> {
        // Exhaustive on purpose, here and at the two sites below: `SyncPolicy` is
        // sealed to the outside but not to this crate, so a policy added later
        // fails to compile at every durability point until someone decides what
        // it means there.
        match self.sync_policy {
            // `sync_all` issues the gathered writes on its way to the disk.
            SyncPolicy::Always => self.image.sync_all(),
            SyncPolicy::OnClose => self.image.ordering_barrier(),
        }
    }

    /// The data-only counterpart to [`barrier`](Self::barrier), for a write that
    /// does not move end-of-file and so needs no metadata flush. It orders the
    /// gathered writes for the same reason.
    fn barrier_data(&mut self) -> Result<(), Error> {
        barrier_data(self.image.as_mut(), self.sync_policy)
    }

    /// Force this session's writes to durable storage *whatever* the policy
    /// says. The counterpart to [`barrier`](Self::barrier), which the policy may
    /// skip.
    ///
    /// Two callers: [`File::sync`](crate::File::sync), the application naming
    /// its own cadence, and the teardown path — `File::close` and
    /// `FileInner::drop` — whose own writes no caller can order, because both
    /// destroy the handle that would have done it.
    pub(crate) fn force_sync(&mut self) -> Result<(), Error> {
        self.image.sync_all()
    }

    /// Rewrite the on-disk free-space managers into canonical (manager-at-tail)
    /// shape for a file that persists them, if this session left them stale.
    ///
    /// Immediate in-place appends grow the file at end-of-file, which pushes the
    /// managers into the middle of it with live data after them. A staged
    /// [`commit`](Self::commit) re-homes them as part of its tail, but a session
    /// that only appends never runs one, so [`File::close`](crate::File::close)
    /// and `FileInner::drop` call this instead. It is the same tail the commit
    /// writes — appended past everything live, with the superblock repoint as the
    /// crash-atomic linearization point — with nothing to free and the root
    /// unchanged.
    ///
    /// A no-op for a non-persisting file, and skipped when the file has not grown
    /// past the managers since they were last written, so an unchanged session
    /// never grows the file.
    ///
    /// If a session that grew the file ends without `close` or `drop` running (a
    /// true crash — `SIGKILL`, power loss), the managers are left mid-file. Every
    /// append was durable and crash-atomic, so no data is lost, and both this
    /// crate and the reference C library reopen the file and read it correctly;
    /// the managers are simply non-canonical until a clean rewrite.
    pub(crate) fn finalize_persist(&mut self) -> Result<(), Error> {
        if self.persist.is_none() || self.image.len() == self.fsm_len {
            return Ok(());
        }
        self.commit_persisting(self.superblock.root_group_address, Vec::new())
    }

    /// Resolve and locate an in-place append target, applying every rule that
    /// does not depend on the bytes being appended: the file-level eligibility
    /// guards, the staged-edit conflict check, and the geometry lookup that
    /// populates [`located`](Self::located). Returns the dataset's object-header
    /// address, which is that cache's key.
    ///
    /// Split out from the append itself so [`append_geometry`](Self::append_geometry)
    /// can report a dataset's batching geometry under exactly the same rules the
    /// append will apply — a caller slicing a large append into batches must be
    /// refused before the first batch, not part-way through.
    fn append_prepare(&mut self, target: AppendTarget<'_>) -> Result<u64, Error> {
        // The fast in-place append is only sound on a base-0 latest-format file:
        // the slot math assumes absolute addresses and the superblock is patched in
        // place per call. A userblock or pre-v2 file falls back to the staged
        // `append_dataset`, which rebuilds the index and repoints the superblock
        // last.
        if self.superblock.base_address != 0 {
            return Err(Error::AppendInPlaceUnsupported(
                "in-place append does not support a file with a userblock (non-zero base \
                 address); use Dataset::append_staged",
            ));
        }
        if self.superblock.version < 2 {
            return Err(Error::AppendInPlaceUnsupported(
                "in-place append requires a latest-format file (v2/v3 superblock); use \
                 Dataset::append_staged",
            ));
        }
        // A paged file (`H5F_FSPACE_STRATEGY_PAGE`) that does not persist its free
        // space has no on-disk record of which pages hold metadata and which hold
        // raw data, so neither this immediate append nor the staged commit can keep
        // the two segregated; refuse it outright. A paged *persisting* file appends
        // through the page-aware `EditStore`, which pads a tail page whenever the
        // page type changes, and has its managers rewritten at the next commit or
        // at close (issue #198).
        if self.paged.is_some() && self.persist.is_none() {
            return Err(Error::AppendInPlaceUnsupported(
                "in-place append is not supported on a paged file \
                 (H5F_FSPACE_STRATEGY_PAGE) without persisted free space; recreate the \
                 file with with_file_space_strategy(FileSpaceStrategy::Page, true, ..)",
            ));
        }

        // Refuse an append against a dataset (or a subtree) that a still-staged edit
        // in this same session will relocate, replace, or delete — which would
        // strand the durably-appended rows or plan against a header the commit
        // moves. The caller must commit those edits first.
        //
        // A target named by object-header address cannot be compared against the
        // staged paths, so any staged edit at all disqualifies it. That is a
        // superset of the path check, and the remedy is the same one.
        match target {
            AppendTarget::Path(dataset) => {
                if self.append_conflicts_with_pending(&split_path(dataset)) {
                    return Err(Error::AppendInPlaceUnsupported(
                        "the dataset or an ancestor has a staged edit pending in this session; \
                         commit the staged edits before appending in place, or use \
                         Dataset::append_staged",
                    ));
                }
            }
            AppendTarget::Header(_) if self.has_staged_edits() || self.committed => {
                return Err(Error::AppendInPlaceUnsupported(
                    "this append target was reached by object reference, so it names a dataset \
                     by object-header address, and this session has staged or committed edits \
                     that can move that header; re-open the dataset by path to append to it",
                ));
            }
            AppendTarget::Header(_) => {}
        }

        // Resolve the dataset's object-header address — the geometry cache key.
        // base == 0 here, so a resolved address is absolute; two hard links to
        // one dataset share the one entry.
        let oh_addr = match target {
            AppendTarget::Path(dataset) => match self.resolved.get(dataset) {
                Some(&addr) => addr,
                None => {
                    let addr = crate::group_v2::resolve_path_any_from_source(
                        &self.image(),
                        &self.superblock,
                        dataset,
                    )
                    .map_err(|_| {
                        Error::AppendInPlaceUnsupported("nothing to append to at the given path")
                    })?;
                    self.resolved.insert(dataset.to_string(), addr);
                    addr
                }
            },
            AppendTarget::Header(addr) => addr,
        };

        // Locate the dataset on the first append (cache miss) against the session's
        // own image — no second lock, no second view of the file, no re-read.
        if !self.located.contains_key(&oh_addr) {
            let store = self.store();
            let state = locate_dataset_state(&store, oh_addr)?;
            self.located.insert(oh_addr, state);
        }
        Ok(oh_addr)
    }

    /// The append geometry of the dataset `target` names, so a caller can slice a
    /// large append into aligned batches *before* materializing each batch's
    /// bytes — which is what keeps a bounded session's peak memory at one batch
    /// rather than the whole call.
    pub(crate) fn append_geometry(
        &mut self,
        target: AppendTarget<'_>,
    ) -> Result<AppendGeometry, Error> {
        let oh_addr = self.append_prepare(target)?;
        let st = &self.located[&oh_addr];
        let chunk_elems = st.loc.chunk_elems.max(1);
        Ok(AppendGeometry {
            chunk_elems,
            element_size: st.element_size,
            current_dim: st.loc.current_dim,
            filtered: st.pipeline.is_some(),
            full_batch_elems: self.batch_elems(st.loc.chunk_bytes, chunk_elems),
        })
    }

    /// Whole-chunk elements in one append batch.
    ///
    /// A bounded session caps a batch at [`APPEND_BATCH_BYTES`] of raw data (at
    /// least one chunk) so peak memory is independent of the call size, at the
    /// cost of splitting one crash-atomic append into several: a crash between
    /// batches leaves a valid shorter dataset, exactly as if the caller had
    /// looped. A mirror session already holds the whole file, so bounding the
    /// call buys nothing there and would trade that atomicity away for free;
    /// it takes the whole append as one batch.
    fn batch_elems(&self, chunk_bytes: usize, chunk_elems: u64) -> u64 {
        if !self.batched_appends {
            return u64::MAX;
        }
        (APPEND_BATCH_BYTES / (chunk_bytes.max(1) as u64)).max(1) * chunk_elems
    }

    /// Apply a gathered in-place append (typed / generic / raw bytes) to the
    /// dataset `target` names, immediately and crash-atomically, driving the
    /// shared Extensible-Array engine against the session's own image through an
    /// [`EditStore`] adapter. Runs only the first `max_phase` durability phases;
    /// production callers pass 4, the crash-consistency tests stop at a boundary
    /// to simulate a crash.
    ///
    /// A bounded session splits the call into whole-chunk batches (see
    /// [`batch_elems`](Self::batch_elems)), each its own crash-atomic apply.
    /// Every predictable refusal is raised before the first batch, so a rejected
    /// append leaves the file untouched rather than partly grown.
    ///
    /// `Dataset::append` slices its own call the same way before it reaches here,
    /// so through the public API this loop runs once per call; it batches for the
    /// benefit of a caller that hands the engine one large builder directly, whose
    /// bytes are already materialized but whose plan need not be.
    pub(crate) fn append_inplace_gathered(
        &mut self,
        target: AppendTarget<'_>,
        b: &AppendBuilder,
        max_phase: u8,
    ) -> Result<(), Error> {
        if b.dt_conflict() {
            return Err(Error::AppendInPlaceUnsupported(
                "append mixes element types in one call; use one element type per append",
            ));
        }
        let oh_addr = self.append_prepare(target)?;

        // Validate the appended bytes against the on-disk datatype.
        let raw = b.raw();
        let new_elems = validate_gathered_append(&self.located[&oh_addr], b)?;
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

        let (chunk_elems, elem_bytes, full_batch_elems, filtered, current_dim) = {
            let st = &self.located[&oh_addr];
            (
                st.loc.chunk_elems.max(1),
                st.element_size.get() as u64,
                self.batch_elems(st.loc.chunk_bytes, st.loc.chunk_elems.max(1)),
                st.pipeline.is_some(),
                st.loc.current_dim,
            )
        };
        // Refuse a filtered append onto an unaligned length before ANY batch
        // applies, so the refusal is as atomic as an unbatched one. Left to
        // `plan_ea_append` it would surface only when the first batch was reached.
        // The appended length is unconstrained: `batch_elems` is a whole-chunk
        // multiple, so an unaligned remainder can only ever be the *last* batch,
        // by which point every earlier batch has left the length chunk-aligned.
        if filtered && current_dim % chunk_elems != 0 {
            return Err(Error::AppendInPlaceUnsupported(
                "a filtered dataset whose length is not a whole multiple of the chunk length \
                 cannot be appended in place: growing its trailing partial chunk would repoint \
                 an index element a reader can already see. Use Dataset::append_staged, or a \
                 BufferedAppender, which keeps the on-disk length chunk-aligned",
            ));
        }

        let mut done = 0u64;
        while done < new_elems {
            // Fill the trailing partial chunk first (so later batches start
            // chunk-aligned and never rewrite it again), then whole-chunk batches.
            // Filtered datasets are chunk-aligned by contract, so every batch stays
            // chunk-aligned there too.
            let current_dim = self.located[&oh_addr].loc.current_dim;
            let to_boundary = (chunk_elems - current_dim % chunk_elems) % chunk_elems;
            let take = (new_elems - done).min(to_boundary.saturating_add(full_batch_elems));
            let batch =
                &raw[(done * elem_bytes).to_usize()?..((done + take) * elem_bytes).to_usize()?];

            // Read/plan phase (immutable borrows only, nothing published yet), then
            // the ordered, fsync-barriered write phase — both shared with
            // `Dataset::append` through the chunk-index engine. `EditStore` borrows
            // only the image-carrying fields, so `self.located` stays independently
            // borrowable.
            let plan_result = {
                let Self {
                    image,
                    superblock,
                    sb_sig_off,
                    paged,
                    located,
                    sync_policy,
                    ..
                } = self;
                let st = &located[&oh_addr];
                let store = EditStore {
                    image: image.as_mut(),
                    superblock,
                    sb_sig_off: *sb_sig_off,
                    paged: paged.as_mut(),
                    sync_policy: *sync_policy,
                };
                plan_ea_append(
                    &store,
                    &st.loc,
                    &st.datatype,
                    &st.spatial,
                    st.element_size,
                    st.pipeline.as_ref(),
                    batch,
                    take,
                    st.fill.pattern(st.element_size),
                )
            };
            let plan = plan_result.map_err(as_inplace_error)?;
            {
                let Self {
                    image,
                    superblock,
                    sb_sig_off,
                    paged,
                    located,
                    sync_policy,
                    ..
                } = self;
                let st = located.get_mut(&oh_addr).expect("dataset located above");
                let mut store = EditStore {
                    image: image.as_mut(),
                    superblock,
                    sb_sig_off: *sb_sig_off,
                    paged: paged.as_mut(),
                    sync_policy: *sync_policy,
                };
                apply_ea_append(&mut store, &mut st.loc, &plan, max_phase)
                    .map_err(as_inplace_error)?;
            }
            if max_phase < 4 {
                // Crash-consistency hook: the caller asked to stop inside the
                // first batch's durability sequence, so there is no next batch.
                // Each phase's own barrier has already issued the writes up to it
                // under either policy, which is what leaves a phase boundary a
                // real one for the tests that stop at it.
                return Ok(());
            }
            done += take;
        }
        Ok(())
    }

    /// Test-only phased in-place append (stops after `max_phase` durability phases)
    /// used by the crash-consistency tests, mirroring `Dataset::append`'s harness.
    #[cfg(test)]
    fn append_inplace_i32_phased(
        &mut self,
        dataset: &str,
        values: &[i32],
        max_phase: u8,
    ) -> Result<(), Error> {
        let mut b = AppendBuilder::new();
        b.append_i32(values);
        self.append_inplace_gathered(AppendTarget::Path(dataset), &b, max_phase)
    }

    /// Whether `target` (an [`append_inplace_gathered`](Self::append_inplace_gathered) dataset path)
    /// or any of its ancestors is named by a staged edit that a later
    /// [`commit`](Self::commit) would relocate, replace, or delete. `create_group`
    /// and group-attribute edits are excluded: they rewrite a group header without
    /// moving a descendant dataset's header or freeing its storage, so they cannot
    /// stale the append geometry cache.
    fn append_conflicts_with_pending(&self, target: &[String]) -> bool {
        let hits = |p: &[String]| paths_overlap(target, p);
        self.staged.writes.iter().any(|(p, _)| hits(p))
            || self.staged.appends.iter().any(|(p, _)| hits(p))
            || self.staged.deletes.iter().any(|p| hits(p))
            || self.staged.copies.iter().any(|(_, dst)| hits(dst))
            || self.staged.cross_copies.iter().any(|(dst, _)| hits(dst))
            || self.staged.dataset_attrs.iter().any(|(p, _)| hits(p))
            || self.staged.datasets.iter().any(|(parent, fd)| {
                let mut full = parent.clone();
                full.push(fd.name.clone());
                paths_overlap(target, &full)
            })
    }

    /// Stage a new (empty) group at `path`, created on the next
    /// [`commit`](Self::commit). The parent must already exist or be created in
    /// the same session; populate the group with datasets via
    /// [`create_dataset`](Self::create_dataset) using a path under it.
    pub fn create_group(&mut self, path: &str) -> Result<(), Error> {
        let comps = split_path(path);
        self.refuse_if_claimed(&comps)?;
        self.staged.groups.push(comps);
        Ok(())
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
    pub fn set_group_attr(
        &mut self,
        path: &str,
        name: &str,
        value: AttrValue,
    ) -> Result<(), Error> {
        let comps = split_path(path);
        self.refuse_if_claimed(&comps)?;
        self.staged.group_attrs.push((
            comps,
            AttrOp::Set {
                name: name.to_string(),
                value,
            },
        ));
        Ok(())
    }

    /// Stage removal of a compact attribute from a group, applied on the next
    /// [`commit`](Self::commit).
    ///
    /// `path` names the group to edit; `""` or `"/"` names the root group. The
    /// named attribute must exist in the committed group state after any earlier
    /// staged attribute operations for the same group have been applied.
    pub fn remove_group_attr(&mut self, path: &str, name: &str) -> Result<(), Error> {
        let comps = split_path(path);
        self.refuse_if_claimed(&comps)?;
        self.staged.group_attrs.push((
            comps,
            AttrOp::Remove {
                name: name.to_string(),
            },
        ));
        Ok(())
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
    pub fn set_dataset_attr(
        &mut self,
        path: &str,
        name: &str,
        value: AttrValue,
    ) -> Result<(), Error> {
        let comps = split_path(path);
        self.refuse_if_claimed(&comps)?;
        self.staged.dataset_attrs.push((
            comps,
            AttrOp::Set {
                name: name.to_string(),
                value,
            },
        ));
        Ok(())
    }

    /// Stage removal of a compact attribute from an **existing dataset**, applied on
    /// the next [`commit`](Self::commit).
    ///
    /// `path` names the dataset to edit; the named attribute must exist in the
    /// committed dataset state after any earlier staged attribute operations for the
    /// same dataset have been applied. Like [`set_dataset_attr`](Self::set_dataset_attr)
    /// it relocates the dataset header and requires a single hard link.
    pub fn remove_dataset_attr(&mut self, path: &str, name: &str) -> Result<(), Error> {
        let comps = split_path(path);
        self.refuse_if_claimed(&comps)?;
        self.staged.dataset_attrs.push((
            comps,
            AttrOp::Remove {
                name: name.to_string(),
            },
        ));
        Ok(())
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
    /// it survives reopen, otherwise it is forgotten
    /// on close. After reuse, an object reference to a deleted object may resolve
    /// to an unrelated object (deleting a referenced object is undefined in HDF5).
    ///
    /// The path must exist. The link's parent group must itself be editable in
    /// place (compact links, single-chunk header); the target being removed has
    /// no such restriction.
    ///
    /// # Replacing an object
    ///
    /// Deleting a path and creating a new object at that same path in one commit
    /// is a *replacement*, and is accepted (issue #305): the removal is applied
    /// before the addition, and the commit's single superblock write publishes
    /// both, so the path is occupied at every instant a crash could interrupt.
    /// The new object need not resemble the old one — a dataset may replace a
    /// group, or the reverse — and replacing a group discards its whole subtree
    /// rather than inheriting it.
    ///
    /// Everything this commit adds *below* a replaced path lands in the
    /// replacement, whatever order the calls were made in: staging
    /// `create_dataset("g/x")` and then replacing `g` puts `x` in the new group,
    /// not the one being removed. A staged edit that could only mean the
    /// *original* is refused instead — a group under the replaced path that this
    /// commit does not itself create, an attribute set on the object being
    /// removed, a value overwrite inside it, or a [`copy`](crate::File::copy)
    /// reading from it.
    ///
    /// See [`Group::delete`](crate::Group::delete) for the public form and a
    /// worked example.
    ///
    /// A deletion may not overlap a staged change at a *different* path
    /// (deleting `/a` while adding `/a/b`, unless `/a` is itself replaced in the
    /// same commit), nor an edit to the object being removed (an attribute set
    /// on it, or a value overwrite of something inside it); split those into
    /// separate commits.
    pub fn delete(&mut self, path: &str) -> Result<(), Error> {
        let comps = split_path(path);
        self.refuse_if_claimed(&comps)?;
        self.staged.deletes.push(comps);
        Ok(())
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
    pub fn copy(&mut self, src: &str, dst: &str) -> Result<(), Error> {
        let (s, d) = (split_path(src), split_path(dst));
        // Both ends matter: the source is read and the destination is written,
        // and a commit relocates headers along either path.
        self.refuse_if_claimed(&s)?;
        self.refuse_if_claimed(&d)?;
        self.staged.copies.push((s, d));
        Ok(())
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
        // Read (and foreign-address-screen) the whole subtree now, while `source`
        // is borrowed; the owned tree carries every byte the commit will write. The
        // source is gated to base 0 above, so its stored addresses are absolute.
        let tree = Self::read_copy_subtree(&BytesSource::new(src_data), src_addr, 0, true, 0)?;
        self.refuse_if_claimed(&dst)?;
        self.staged.cross_copies.push((dst, tree));
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
    ///
    /// **A refused commit costs the session nothing it had staged.** The
    /// staged set is whole afterwards — so the batch can be committed again,
    /// and refuses again identically rather than applying the part of itself
    /// the refusal was not about — and the free regions the attempt drew from
    /// are given back, so an attempt that never became visible costs the
    /// session neither staged work nor reusable space (issue #316; see
    /// [`StagedEdits`] and [`FreeSnapshot`]).
    ///
    /// A failure *past* the first write is the other case, and it clears the
    /// staged set rather than restoring it: the file is valid but some of the
    /// batch is in it as slack, and a later `commit` must not re-issue the rest
    /// as if nothing had happened.
    pub fn commit(&mut self) -> Result<(), Error> {
        let snapshot = self.snapshot_free();
        self.repointed = false;
        // The staged set comes out for the duration of the attempt and goes back
        // if the attempt refuses, so a refusal costs the session no staged work
        // (issue #316). What goes back is whatever the attempt did not take: its
        // preflight only reads, and its apply phase takes the whole set in one
        // move at the point of no return, so a preflight refusal restores every
        // edit and a failure past that point restores nothing.
        let mut staged = std::mem::take(&mut self.staged);
        let result = self.commit_inner(&mut staged);
        if result.is_err() && !self.repointed {
            self.restore_free(snapshot);
            self.staged = staged;
        }
        result
    }

    /// The free lists as they stand now, for [`commit`](Self::commit) to restore
    /// if its apply loop draws from them and then fails.
    fn snapshot_free(&self) -> FreeSnapshot {
        FreeSnapshot {
            free: self.free.clone(),
            paged: self
                .paged
                .as_ref()
                .map(|pg| (pg.meta.clone(), pg.raw.clone())),
        }
    }

    /// Put the free lists back as [`snapshot_free`](Self::snapshot_free) found
    /// them. Called only for a commit that failed before publishing anything, so
    /// every region it restores is dead again.
    fn restore_free(&mut self, snapshot: FreeSnapshot) {
        self.free = snapshot.free;
        if let (Some(pg), Some((meta, raw))) = (self.paged.as_mut(), snapshot.paged) {
            pg.meta = meta;
            pg.raw = raw;
        }
    }

    fn commit_inner(&mut self, staged: &mut StagedEdits) -> Result<(), Error> {
        if staged.is_empty() {
            return Ok(());
        }

        // A paged file (`H5F_FSPACE_STRATEGY_PAGE`) that does not persist its free
        // space has no on-disk record of which pages hold metadata and which hold
        // raw data, so this commit could not keep the two segregated and would
        // silently degrade the paging. Refuse up front, before any writes, exactly
        // as the bounded backend does. A paged *persisting* file is committed
        // through the page-aware tail below (issue #198).
        if self.paged.is_some() && self.persist.is_none() {
            return Err(Error::EditUnsupported(
                "committing an edit to a paged file (H5F_FSPACE_STRATEGY_PAGE) requires \
                 persisted free space; recreate the file with \
                 with_file_space_strategy(FileSpaceStrategy::Page, true, ..) to edit it in place",
            ));
        }

        // Invalidate the in-place-append geometry cache before doing any work. A
        // commit that reaches here rewrites and relocates object headers, frees
        // vacated regions into `self.free`, and may truncate the file — any of
        // which can leave a cached `Located` pointing at a moved header or into a
        // now-free-eligible region. Clearing at *entry* (rather than the success
        // tail) means a later failure — including one after the durable root flip,
        // which leaves the session reusable — never strands a stale cache. The
        // no-op fast return above does no such work, so it keeps the cache. The
        // The next in-place append re-locates against the fresh file.
        self.located.clear();
        self.resolved.clear();
        // Past this point the commit may relocate object headers, so every address
        // a caller captured earlier is suspect; see `committed`.
        self.committed = true;

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
        for (full, fd) in &staged.writes {
            // A path named twice in one commit would write it twice (and double-
            // free a resized extent); require separate commits.
            if write_targets.contains(full) {
                return Err(Error::EditUnsupported(
                    "the same dataset is overwritten twice in one commit; use separate commits",
                ));
            }
            let path_str = full.join("/");
            let addr = crate::group_v2::resolve_path_any_from_source(
                &self.image(),
                &self.superblock,
                &path_str,
            )
            .map_err(|_| Error::EditUnsupported("nothing to overwrite at the given path"))?;
            let addr = usize::try_from(addr)
                .map_err(|_| Error::EditUnsupported("dataset address exceeds this platform"))?;
            match Self::prepare_write(&self.image(), addr as u64, fd, base)? {
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
            write_targets.push(full.clone());
        }

        // --- Preflight appends (`append_dataset`) under the same all-or-nothing,
        // single-hard-link contract. Each plans a relocating append — existing
        // chunk data stays in place; the appended (and any rewritten trailing)
        // chunks and a rebuilt Extensible-Array index are staged, and the whole is
        // treated like a relocating overwrite of the dataset's header (staged
        // against its parent group so the commit patches the link). A zero-length
        // append is a no-op and is dropped here. ---
        for (full, ab) in &staged.appends {
            if full.is_empty() {
                return Err(Error::AppendUnsupported("cannot append to the root group"));
            }
            if ab.raw.is_empty() {
                continue; // nothing to append
            }
            // A dataset overwritten or appended earlier in this commit would be
            // planned against a stale header and its old storage double-freed;
            // require separate commits.
            if write_targets.contains(full) {
                return Err(Error::AppendUnsupported(
                    "the same dataset is edited more than once in one commit; use separate commits",
                ));
            }
            let path_str = full.join("/");
            let addr = crate::group_v2::resolve_path_any_from_source(
                &self.image(),
                &self.superblock,
                &path_str,
            )
            .map_err(|_| Error::AppendUnsupported("nothing to append to at the given path"))?;
            let addr = usize::try_from(addr)
                .map_err(|_| Error::AppendUnsupported("dataset address exceeds this platform"))?;
            let mw = Self::prepare_append(&self.image(), addr as u64, ab, base)?;
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
            write_targets.push(full.clone());
        }

        // --- Preflight dataset attribute edits (`set_dataset_attr` /
        // `remove_dataset_attr`) under the same all-or-nothing, single-hard-link
        // contract. Each gathers the dataset's verbatim object-header region,
        // applies the compact attribute ops to it, and stages a relocating
        // `AttrEdit` header rewrite against the parent group — like a value
        // overwrite, but the data-layout message (and thus the chunk data and index)
        // is preserved verbatim, so only the header moves. ---
        if !staged.dataset_attrs.is_empty() {
            // Collect the ops per dataset in first-seen path order, so multiple edits
            // to one dataset produce a single relocating header rewrite.
            let mut order: Vec<PathKey> = Vec::new();
            let mut ops_by_path: HashMap<&PathKey, Vec<&AttrOp>> = HashMap::new();
            for (path, op) in &staged.dataset_attrs {
                if !ops_by_path.contains_key(path) {
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
                let addr = crate::group_v2::resolve_path_any_from_source(
                    &self.image(),
                    &self.superblock,
                    &path_str,
                )
                .map_err(|_| {
                    Error::EditUnsupported("nothing to set an attribute on at the given path")
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
                let region = Self::gather_oh_messages(&self.image(), addr as u64, base)?;
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
            && staged.datasets.is_empty()
            && staged.groups.is_empty()
            && staged.group_attrs.is_empty()
            && staged.deletes.is_empty()
            && staged.copies.is_empty()
            && staged.cross_copies.is_empty()
        {
            // This path's own point of no return, and it takes the staged set
            // for the same reason the main one below does (issue #316): every
            // refusal that applies to these overwrites has already run, and a
            // batch some of which has been written is not one a later `commit`
            // may re-issue. It is also what this path already did, back when the
            // write preflight drained the staged set on its way here.
            //
            // Leaving the set in place would make a half-written batch
            // retryable — these overwrites are same-length writes to fixed
            // addresses, so repeating them is idempotent — but that reads the
            // rule the other way round for one path, on an argument nothing
            // enforces, and the only case that can tell the two apart is a
            // `write_at` that fails partway, which no test here can produce.
            drop(std::mem::take(staged));
            for (data_addr, raw) in &inplace_writes {
                self.write_at(*data_addr, raw)?;
            }
            self.barrier()?;
            return Ok(());
        }

        // --- Plan: build the tree of "dirty" groups (root plus every group on a
        // path to an addition or deletion), validating every target before any
        // write. `add_targets` records the full paths created this commit, used
        // to reject a deletion that overlaps an addition. ---
        let mut nodes: BTreeMap<PathKey, Node> = BTreeMap::new();
        nodes.entry(PathKey::new()).or_default(); // root is always dirty
        let mut add_targets: Vec<PathKey> = Vec::new();
        // Where each in-file `copy` reads from. A copy takes its bytes from the
        // *pre-commit* file, so a source this same commit replaces would copy the
        // object being removed while the replacement lands at the same path — see
        // the delete-staging loop, which refuses that. Cross-file copies are not
        // tracked: their source is in another file, so no path here can name it.
        let mut copy_sources: Vec<PathKey> = Vec::new();
        let mut attr_targets: Vec<PathKey> = Vec::new();

        // Mark explicitly-created new groups, ensuring their ancestor chain.
        for path in &staged.groups {
            if path.is_empty() {
                return Err(Error::EditUnsupported("cannot create the root group"));
            }
            ensure_ancestors(&mut nodes, path);
            nodes.entry(path.clone()).or_default().is_new = true;
            add_targets.push(path.clone());
        }

        // Make each staged dataset's parent group a node (with its ancestor
        // chain). The datasets themselves stay in the staged set, which the
        // preflight reads but must not empty (issue #316); `datasets_by_group`
        // below is the grouped view both the preflight and the apply loop use.
        for (parent, fd) in &staged.datasets {
            let mut full = parent.clone();
            full.push(fd.name.clone());
            add_targets.push(full);
            ensure_ancestors(&mut nodes, parent);
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
        // destination or a dataset being added in the same commit. The ops
        // themselves stay in the staged set and are looked up by path where they
        // are applied, so a refusal below still gives them back (issue #316).
        for (path, _) in &staged.group_attrs {
            ensure_ancestors(&mut nodes, path);
            attr_targets.push(path.clone());
        }

        // Stage copies: validate the source subtree is copyable (read-only),
        // then treat the destination like an addition to its parent group.
        for (src, dst) in &staged.copies {
            if src.is_empty() {
                return Err(Error::EditUnsupported("cannot copy the root group"));
            }
            if dst.is_empty() {
                return Err(Error::EditUnsupported("copy destination path is empty"));
            }
            if is_prefix(src, dst) {
                return Err(Error::EditUnsupported(
                    "cannot copy an object into itself or its own subtree",
                ));
            }
            let src_str = src.join("/");
            let src_addr = crate::group_v2::resolve_path_any_from_source(
                &self.image(),
                &self.superblock,
                &src_str,
            )
            .map_err(|_| Error::EditUnsupported("copy source does not exist"))?;
            let src_addr = usize::try_from(src_addr)
                .map_err(|_| Error::EditUnsupported("source address exceeds this platform"))?;
            // Read the source subtree from this file's own mirror (`cross_file`
            // false: same address space, so verbatim addresses stay valid). On a
            // userblock file the stored addresses are base-relative, so pass this
            // session's base for the read to absolutize them.
            let tree = Self::read_copy_subtree(&self.image(), src_addr as u64, 0, false, base)?;
            copy_sources.push(src.clone());
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
        for (dst, _) in &staged.cross_copies {
            if dst.is_empty() {
                return Err(Error::EditUnsupported("copy destination path is empty"));
            }
            add_targets.push(dst.clone());
            let leaf = dst.last().unwrap().clone();
            let parent = dst[..dst.len() - 1].to_vec();
            ensure_ancestors(&mut nodes, &parent);
            nodes.entry(parent).or_default().cross_copies.push(leaf);
        }

        // Stage deletions: each must exist, must not overlap any other staged
        // change *unless* this commit replaces what it removes, and is recorded
        // against its parent group (which becomes dirty). `deleted_addrs` keeps
        // each removed object's header address so its owned blocks can be
        // reclaimed after the commit lands (issue #21).
        let delete_targets = &staged.deletes;
        let mut deleted_addrs: Vec<usize> = Vec::new();
        for (i, d) in delete_targets.iter().enumerate() {
            if d.is_empty() {
                return Err(Error::EditUnsupported("cannot delete the root group"));
            }
            let path_str = d.join("/");
            let del_addr = crate::group_v2::resolve_path_any_from_source(
                &self.image(),
                &self.superblock,
                &path_str,
            )
            .map_err(|_| Error::EditUnsupported("nothing to delete at the given path"))?;
            if let Ok(a) = usize::try_from(del_addr) {
                deleted_addrs.push(a);
            }
            // A deletion may overlap other staged work when this commit
            // *replaces* what it removes: an addition names exactly `d`, so the
            // removal and the new object at the same path are one rotation
            // rather than an edit of something being deleted (issue #305).
            let recreated = add_targets.iter().any(|t| t == d);
            // A replacement also requires that every group node at or below `d`
            // is one this commit builds fresh. A node that is *not* new is
            // rebuilt from the old object's on-disk header — the object being
            // replaced — and both ways that lands are wrong, in the two shapes
            // this guard was measured against:
            //
            // * At `d` itself (a group attribute set on a path this commit
            //   replaces with a *dataset*), the node and the replacement share a
            //   `path_addr` key. The replacement's address overwrites the group's,
            //   the parent's link is then patched to the address it already had,
            //   and the rebuilt group header is left orphaned — a commit that
            //   returns `Ok` having **silently discarded** the staged attribute.
            // * Strictly below `d`, the parent is a freshly built region with no
            //   existing link to patch, so `patch_link_target` reports a missing
            //   child link partway through the apply. That leaves a valid file
            //   (the superblock is never repointed) but appends dead bytes and
            //   reports the wrong thing.
            //
            // Refusing here keeps both in the preflight, where the commit is
            // all-or-nothing and the message can name the actual conflict.
            if recreated
                && !nodes
                    .iter()
                    .all(|(key, node)| !is_prefix(d, key) || node.is_new)
            {
                return Err(Error::EditUnsupported(
                    "a staged edit names a group at or under a replaced path that this \
                     commit does not itself create; create it in the same commit, or use \
                     separate commits",
                ));
            }
            // A copy reading from a path this commit replaces is the same conflict
            // seen from the other side: `read_copy_subtree` already took its bytes
            // from the pre-commit file, so the commit would place the *original*
            // at the copy's destination while placing something else at `d` — two
            // different objects from one path, in one commit, with no error. A
            // source that is merely deleted and not replaced is unambiguous (it is
            // a move) and stays allowed.
            if recreated {
                for t in &copy_sources {
                    if is_prefix(d, t) {
                        return Err(Error::EditUnsupported(
                            "a copy in this commit reads from a path the same commit \
                             replaces; use separate commits",
                        ));
                    }
                }
            }
            // Past that return `recreated` means the whole touched subtree is
            // fresh, so an addition at or below `d` lands in the replacement. The
            // apply loop already removes a group's deleted links before appending
            // any new one (`remove_link_from_region` runs first), so a
            // replacement needs no further sequencing here.
            for t in &add_targets {
                if recreated && is_prefix(d, t) {
                    continue;
                }
                if is_prefix(d, t) || is_prefix(t, d) {
                    return Err(Error::EditUnsupported(
                        "a deletion overlaps an addition in the same commit; \
                         replace the path instead, or use separate commits",
                    ));
                }
            }
            for t in &attr_targets {
                if recreated && is_prefix(d, t) {
                    continue;
                }
                if is_prefix(d, t) {
                    return Err(Error::EditUnsupported(
                        "a deletion overlaps a group-attribute edit in the same commit; use separate commits",
                    ));
                }
            }
            for t in &write_targets {
                if is_prefix(d, t) {
                    // `write_targets` holds three kinds — a value overwrite, a
                    // dataset-attribute edit, and a staged append — so the
                    // message names what they have in common rather than only
                    // the first of them.
                    return Err(Error::EditUnsupported(
                        "a deletion overlaps a staged edit to a dataset in the same \
                         commit; use separate commits",
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
                let addr = crate::group_v2::resolve_path_any_from_source(
                    &self.image(),
                    &self.superblock,
                    &path_str,
                )
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
        let attrs_by_group = group_by_parent(staged.group_attrs.iter().map(|(p, op)| (p, op)));
        for key in &keys {
            if let Some(ops) = attrs_by_group.get(key) {
                let node = nodes.get_mut(key).unwrap();
                let region = std::mem::take(&mut node.base_region);
                let (region, pending_vl_attrs) = apply_group_attr_ops(&region, ops)?;
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

        // Each group's added datasets, in the order the apply loop will place
        // them, for the guards below to read. Borrowed from the staged set:
        // every one of those guards may still refuse, and a refusal gives the
        // caller back every edit it was holding (issue #316). The apply loop
        // rebuilds the same grouping, by the same rule, once it owns the set.
        let datasets_by_group = group_by_parent(staged.datasets.iter().map(|(p, fd)| (p, fd)));

        // Validate names: no addition may collide with an existing link or with
        // another addition under the same parent. A link this same commit
        // deletes is not one of the existing ones — the apply loop removes it
        // from the region before any addition is appended, so replacing an
        // object at its own path is a rotation, not a collision (issue #305).
        for key in &keys {
            let node = &nodes[key];
            let mut adding: Vec<&str> = Vec::new();
            for fd in datasets_by_group.get(key).into_iter().flatten() {
                adding.push(&fd.name);
            }
            for child in children.get(key).into_iter().flatten() {
                if nodes[child].is_new {
                    adding.push(child.last().unwrap());
                }
            }
            for (leaf, _) in &node.copies {
                adding.push(leaf);
            }
            for leaf in &node.cross_copies {
                adding.push(leaf);
            }
            for (i, name) in adding.iter().enumerate() {
                let survives = node.existing_links.iter().any(|n| n == name)
                    && !node.deletes.iter().any(|n| n == name);
                if survives || adding[..i].contains(name) {
                    return Err(Error::EditUnsupported(
                        "a link with this name already exists in the target group",
                    ));
                }
            }
        }

        // Content the caller's library-version bound cannot carry is refused
        // here, before any write, so a rejected addition leaves the commit
        // unapplied.
        self.check_libver_admits(staged.datasets.iter().map(|(_, fd)| fd))?;

        // Enumerate what this commit's deletions reclaim, from the current
        // on-disk layout and before any byte moves. It is read here, ahead of
        // every remaining refusal, because two of them screen against it: an
        // object reference this commit writes must not name space the same
        // commit frees (issue #317). The spans are carried to `to_free` below
        // rather than walked a second time, so the screen and the allocator
        // always mean the same thing by "removed".
        //
        // An object's storage is reclaimed only when the link being removed is
        // its LAST hard link: HDF5 objects can have several hard links, and one
        // reachable through a surviving link is still live (freeing it would
        // corrupt the survivor). Count every hard link in the pre-commit file
        // and reclaim a deleted object only when its count is exactly 1.
        // `deleted_addrs` is de-duplicated first so two delete paths that are
        // hard links to the same object are not visited (and freed) twice. If
        // the link graph cannot be walked in full, no deleted object is
        // reclaimed (a safe leak) — and none is screened either, which is sound
        // in the same direction: nothing is freed, so no reference dangles.
        let mut deleted_free: Vec<(u64, u64, PageType)> = Vec::new();
        deleted_addrs.sort_unstable();
        deleted_addrs.dedup();
        if !deleted_addrs.is_empty() {
            if let Some(incoming) = self.count_incoming_hard_links() {
                for &a in &deleted_addrs {
                    self.collect_free_spans(a, 0, &incoming, &mut deleted_free);
                }
            }
        }
        let reclaimed = ReclaimedSpace::new(&deleted_free, self.superblock.base_address);

        // An in-file copy re-emits its source's element bytes verbatim, so a
        // copied object reference keeps naming whatever it named in the source —
        // including an object this same commit is removing (issue #317).
        {
            // Framed at the base address: a shared-message address is stored
            // relative to it, and `SourceResolver` reads its references as
            // absolute within the view it is given.
            let image = self.image();
            let framed = BaseOffsetSource {
                inner: image,
                base: self.superblock.base_address,
            };
            for key in &keys {
                for (_, tree) in &nodes[key].copies {
                    screen_copied_references(tree, &reclaimed, &framed)?;
                }
            }
        }

        // A value overwrite never reaches `preflight_reference_targets` — it
        // replaces an existing dataset's bytes rather than placing a new object —
        // so its elements are screened here. Resolved addresses are the only
        // references an overwrite can carry: `refuse_unsupported_overwrite`
        // refuses one whose builder staged unresolved `reference_targets`
        // (issue #318). Cross-file copies need no screen for the opposite
        // reason — `reject_foreign_addresses` refuses a reference datatype
        // outright on that path.
        for (_, fd) in &staged.writes {
            screen_resolved_references(&fd.dt, &fd.raw, &reclaimed)?;
        }

        // Prove every object-reference target resolves before any write (see
        // `preflight_reference_targets`'s doc comment): otherwise a reference
        // resolution failure discovered mid-apply-loop would leave every
        // earlier-processed group's real writes (headers, data, copied
        // subtrees) orphaned in the file despite `commit()` returning `Err`.
        Self::preflight_reference_targets(
            &keys,
            &datasets_by_group,
            &nodes,
            &add_targets,
            &write_targets,
            delete_targets,
            &reclaimed,
            &self.image(),
            &self.superblock,
        )?;

        // --- The point of no return. Every refusal is behind us, so the staged
        // set is taken here rather than at the top of this function: a refusal
        // above restores *every* edit the caller staged, and a failure below —
        // an I/O error mid-apply, which leaves the file valid because the
        // superblock is repointed last — restores none of them, so no later
        // `commit` can finish a batch this one abandoned (issue #316). Nothing
        // above this line may consume from `staged`. ---
        let mut taken = std::mem::take(staged);
        // The same paths as the borrow above, owned now that the set has moved.
        let delete_targets = std::mem::take(&mut taken.deletes);
        // The same grouping as `datasets_by_group` above, by the same rule and
        // so in the same order — which is what keeps the apply loop's placement
        // order the one `preflight_reference_targets` proved.
        let mut flat = group_by_parent(taken.datasets.drain(..));
        // A cross-file copy's subtree can move now, so it joins its destination
        // group's in-file copies. The parent node exists: the preflight made one
        // for every destination.
        for (dst, tree) in taken.cross_copies.drain(..) {
            let leaf = dst.last().unwrap().clone();
            let parent = dst[..dst.len() - 1].to_vec();
            nodes.get_mut(&parent).unwrap().copies.push((leaf, tree));
        }

        // Gather the regions this commit will vacate, read from the current
        // on-disk layout before any byte moves: every deleted object's owned
        // blocks plus every superseded group header. These are not added to the
        // free list until after the superblock repoint (they remain live until
        // then), so the appends below never reuse them. Enumeration is
        // best-effort — `collect_free_spans` simply omits anything it cannot
        // account for exhaustively, so the worst case is unreclaimed dead bytes,
        // never a freed-but-live region.
        let mut to_free: Vec<(u64, u64, PageType)> = Vec::new();

        // The deleted objects' blocks, enumerated before the preflight because a
        // refusal there depends on them (`deleted_free`).
        to_free.extend_from_slice(&deleted_free);
        // A superseded group header is dead once the root is repointed. Its chunk
        // spans are enumerated base-aware (`oh_chunk_spans` shifts continuation
        // addresses by the userblock base and returns absolute file offsets), as is
        // the delete path (`collect_free_spans`), so all of this reclamation works
        // on userblock files too.
        for &a in &superseded_addrs {
            if let Ok(spans) = self.oh_chunk_spans(a) {
                to_free.extend(spans.into_iter().map(|(a, l)| (a, l, PageType::Meta)));
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
                    } => to_free.push((extent.0, extent.1, PageType::Raw)),
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
                        kept_chunks,
                        ..
                    } => {
                        if let Ok(a) = usize::try_from(*old_addr) {
                            if let Some(spans) = self.chunked_index_spans(a) {
                                // The old index is reclaimed as raw only where it
                                // provably sits in a raw page; see
                                // `index_is_provably_raw`. The dataset's old chunk
                                // data is the kept chunks (base-relative) plus the
                                // trailing partial chunk this append relocated —
                                // both already in hand, so the proof needs no second
                                // walk of the index.
                                let data_end = kept_chunks
                                    .iter()
                                    .filter_map(|c| {
                                        c.address.checked_add(base)?.checked_add(c.compressed_size)
                                    })
                                    .chain(old_tail_extent.and_then(|(a, l)| a.checked_add(l)))
                                    .max();
                                if self.index_is_provably_raw(data_end, &spans) {
                                    to_free.extend(
                                        spans.into_iter().map(|(a, l)| (a, l, PageType::Raw)),
                                    );
                                }
                            }
                        }
                        if let Some(ext) = old_tail_extent {
                            // The relocated old trailing chunk is raw data.
                            to_free.push((ext.0, ext.1, PageType::Raw));
                        }
                    }
                    _ => {}
                }
                // The relocated dataset's old header chunks are dead too.
                let mut full = key.clone();
                full.push(leaf.clone());
                let path_str = full.join("/");
                if let Ok(addr) = crate::group_v2::resolve_path_any_from_source(
                    &self.image(),
                    &self.superblock,
                    &path_str,
                ) {
                    if let Ok(a) = usize::try_from(addr) {
                        if let Ok(spans) = self.oh_chunk_spans(a) {
                            to_free.extend(spans.into_iter().map(|(a, l)| (a, l, PageType::Meta)));
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
        retain_disjoint_in_bounds(&mut to_free, self.image.len());

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
            //
            // First is a correctness requirement rather than a convention, and
            // has been since a replacement became one commit (issue #305):
            // `remove_link_from_region` matches by *name*, so a removal running
            // after the additions below would take the replacement's link with
            // the original's and leave the path gone. Every link this loop could
            // collide with — a copy's, a dataset's, a new child group's — is
            // appended after it, which is what keeps that unreachable. (A
            // relocating write and an existing child group *patch* a link rather
            // than appending one, so neither can be a replacement's.)
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
            // in the same group's batch resolves regardless of `staged.datasets`
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
                for (idx, collections) in std::mem::take(&mut fd.vl_attrs) {
                    let addrs = self.place_vl_collections(&collections)?;
                    patch_vl_refs(&mut fd.attrs[idx].raw_data, &addrs);
                }
                // Resolve an object-reference dataset's per-element targets now
                // that every earlier-placed object in this commit is in
                // `path_addr` (chunked datasets never carry these —
                // `flatten_dataset` refuses that combination).
                if let Some(patches) = fd.reference_targets.take() {
                    for patch in &patches {
                        let addr = Self::resolve_reference_target(
                            &patch.target,
                            &path_addr,
                            &nodes,
                            &add_targets,
                            &write_targets,
                            &delete_targets,
                            &reclaimed,
                            &self.image(),
                            &self.superblock,
                        )?;
                        write_reference_address(&mut fd.raw, patch.byte_offset, addr);
                    }
                }
                let oh = if fd.chunk_options.is_chunked() || fd.maxshape.is_some() {
                    self.build_chunked_dataset(&fd)?
                } else {
                    // A staged variable-length-string dataset's element
                    // references still carry a placeholder heap address; place
                    // its collection and patch them before `raw` is appended
                    // (chunked datasets never carry staging — refused above).
                    if let Some(staging) = fd.vl_string_staging.take() {
                        if !staging.collections.is_empty() {
                            let addrs = self.place_vl_collections(&staging.collections)?;
                            patch_vl_refs_masked(&mut fd.raw, &staging.patch_offsets, &addrs);
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
                        self.alloc_or_append_typed(&fd.raw, PageType::Raw)? - base
                    };
                    build_dataset_oh(
                        &fd.dt,
                        // Committed datatypes are refused when a dataset is
                        // staged (`flatten_dataset`), so every type here is
                        // written into the dataset's own header.
                        &DatatypeLocation::Inline,
                        &fd.ds,
                        data_addr,
                        fd.raw.len() as u64,
                        &fd.attrs,
                        None,
                        fd.fill.as_deref(),
                        self.libver(),
                    )?
                };
                let oh_addr = self.alloc_or_append_typed(&oh, PageType::Meta)?;
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
            for (mut msg, collections) in pending_vl_attrs {
                let addrs = self.place_vl_collections(&collections)?;
                patch_vl_refs(&mut msg.raw_data, &addrs);
                region.extend_from_slice(&region_message(
                    MessageType::Attribute,
                    &msg.serialize(LENGTH_SIZE),
                ));
            }

            let oh = build_v2_object_header(&region)?;
            let addr = self.alloc_or_append_typed(&oh, PageType::Meta)?;
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
        for (a, l, _) in to_free.drain(..) {
            self.free.free(a, l);
        }
        let cur_eof = self.image.len();
        let trunc_to = self.free.take_trailing(cur_eof);
        let new_eof = trunc_to.unwrap_or(cur_eof);

        self.barrier()?;
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
            self.repointed = true;
            self.barrier()?;
            new_sb.root_group_address = new_root;
            self.superblock = new_sb;
        } else {
            self.repoint_v0v1_root(new_root - base, new_eof)?;
            self.repointed = true;
            self.barrier()?;
            self.superblock.root_group_address = new_root;
            self.superblock.eof_address = new_eof;
        }

        // Physically shrink the file only after the superblock — now carrying the
        // smaller end-of-file — is durable. A crash between the two leaves a file
        // whose superblock end-of-file is correct and whose trailing bytes are
        // mere unreferenced slack, which the next open ignores; the reverse order
        // could advertise an end-of-file past the actual file length.
        if let Some(cut) = trunc_to {
            self.image.truncate(cut)?;
            self.barrier()?;
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
    fn commit_persisting(
        &mut self,
        new_root: u64,
        to_free: Vec<(u64, u64, PageType)>,
    ) -> Result<(), Error> {
        // A paged file records its free space in per-page-type managers and keeps
        // its allocation page-aligned, so it takes its own tail (issue #198).
        if self.paged.is_some() {
            return self.commit_persisting_paged(new_root, to_free);
        }
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
        for &(a, l, _) in &to_free {
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
            build_v2_object_header(&self.rewrite_extension_region(old_ext_addr, &placeholder)?)?
                .len() as u64;

        let ext_addr = self.image.len();
        let fshd_addr = ext_addr + ext_len;

        // Build the real extension and the FSM blocks. With no free space to
        // record we still refresh the extension (persist on, managers undefined).
        let (ext_oh, fsm_blocks, final_eof) = if sections.is_empty() {
            let info = FileSpaceInfo::persistent_empty(strategy, threshold, page_size);
            let ext_oh =
                build_v2_object_header(&self.rewrite_extension_region(old_ext_addr, &info)?)?;
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
                build_v2_object_header(&self.rewrite_extension_region(old_ext_addr, &info)?)?;
            debug_assert_eq!(
                ext_oh.len() as u64,
                ext_len,
                "extension length must be stable across the placeholder and real messages"
            );
            let (fshd, fsse) =
                serialize_file_fsm(&sections, fshd_addr, fsse_addr, os, SECT_CLASS_SIMPLE);
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
        self.barrier()?;
        let mut new_sb = self.superblock.clone();
        new_sb.root_group_address = new_root;
        new_sb.eof_address = final_eof;
        new_sb.superblock_extension_address = Some(ext_addr);
        // Clear any leftover write/SWMR consistency flag on a clean commit (see
        // the non-persisting path above and issue #73).
        new_sb.consistency_flags = 0;
        let sb_bytes = new_sb.serialize();
        self.write_at(self.sb_sig_off, &sb_bytes)?;
        self.repointed = true;
        self.barrier()?;
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
        // The managers now sit at the tail, so nothing is owed until the file
        // grows past them again.
        self.fsm_len = self.image.len();
        Ok(())
    }

    /// Commit tail for a genuine paged file (`H5F_FSPACE_STRATEGY_PAGE`, issue
    /// #198). The paged counterpart of [`commit_persisting`](Self::commit_persisting).
    ///
    /// Two things differ from the flat tail. Free space is recorded in *per-page-type*
    /// managers — SUPER (slot 0) for metadata, DRAW (slot 2) for small raw, and the
    /// generic-large manager (slot 6) for whole free pages and large-raw fragments —
    /// rather than one generic manager, so a paged file reopened by the reference
    /// library still finds its free space segregated. And the file is padded to a
    /// page before the tail is laid down, so the end-of-allocation stays a whole
    /// number of pages, matching the paged file the from-scratch writer produces.
    ///
    /// The tail itself is placed like any other run of metadata, in free space
    /// where some fits ([`tail_layout`](Self::tail_layout)), and only opens a page
    /// at end-of-file when none does. It used to always append into a page of its
    /// own and pad the remainder out untracked, which cost a page per commit and
    /// recorded none of it — a paged file under delete-and-recreate churn grew
    /// without bound while reporting a fraction of that as reusable (issue #286).
    /// The reference library does not page-align these blocks either: its manager
    /// headers sit at whatever offset within a metadata page they are allocated at.
    ///
    /// Crash atomicity is identical to the flat path: the tail is either past the
    /// live file or inside space an earlier commit freed, and is unreferenced
    /// either way until the superblock repoint, which is the linearization point.
    fn commit_persisting_paged(
        &mut self,
        new_root: u64,
        to_free: Vec<(u64, u64, PageType)>,
    ) -> Result<(), Error> {
        let os = self.superblock.offset_size;
        let (strategy, threshold, page_size, old_blocks) = {
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

        // Page-align the file before anything else, so the rewritten extension and
        // the manager blocks begin on a page boundary and stay in metadata pages.
        // The padded tail becomes free space of whatever type that page held.
        self.pad_to_page()?;

        let old_ext_rel = self
            .superblock
            .superblock_extension_address
            .filter(|&a| a != UNDEF)
            .ok_or(Error::EditUnsupported(
                "a persisting file has no superblock extension to update",
            ))?;
        let old_ext_addr = usize::try_from(old_ext_rel)
            .map_err(|_| Error::EditUnsupported("extension address exceeds this platform"))?;

        // The 12-slot persist message is fixed-size, so a placeholder sizes the
        // rewritten extension before its manager addresses are known — and before
        // its address is, which is what lets the tail be sized before it is placed.
        let placeholder = FileSpaceInfo::persistent_managers(
            strategy,
            threshold,
            page_size,
            [UNDEF; NUM_FILE_FSM_MANAGERS],
            0,
        );
        let ext_len =
            build_v2_object_header(&self.rewrite_extension_region(old_ext_addr, &placeholder)?)?
                .len() as u64;

        // Place the tail — the rewritten extension and the manager blocks — as an
        // ordinary run of metadata, in a hole an *earlier* commit freed where one
        // fits. That is what stops a file under delete-and-recreate churn from
        // growing: every commit frees its predecessor's tail exactly, so in the
        // steady state each tail lands in the hole the one before it left, and the
        // file never has to open a page for it (issue #286). `pg`'s lists hold only
        // durable free space — this commit's own frees stay in `to_free` until the
        // repoint — so the bytes overwritten are already unreachable from the
        // on-disk root, the guarantee [`reserve`](Self::reserve) relies on.
        //
        // The tail is sized from the very lists it draws on, so its length and its
        // placement define each other: reserving space removes sections, and fewer
        // sections need fewer bytes to record. `tail_layout` settles that by
        // proposing a length, planning against it, and accepting the proposal only
        // when the plan comes out exactly that long — no shorter, which would leave
        // untracked bytes inside the reservation, and no longer, which would run
        // past it into whatever lives next. A few rounds settle it; a proposal that
        // will not converge falls through to the append below, which has no length
        // to satisfy.
        let placed = self.tail_layout(&to_free, &old_blocks, ext_len, page_size, os);
        let reused = placed.is_some();
        let (post, plan, ext_addr, blocks_len) = match placed {
            Some(layout) => layout,
            None => {
                // Nothing fits: open a metadata page at end-of-file. `pad_to_page`
                // above already left the file page-aligned, so this only records the
                // tail page's type.
                self.begin_page(PageType::Meta)?;
                let at = self.image.len();
                let post = self.paged_post_free(&to_free, &old_blocks);
                let plan = plan_paged_managers(
                    &free_sections(&post.meta),
                    &free_sections(&post.raw),
                    &post.unclassified,
                    page_size,
                    at + ext_len,
                    os,
                );
                let blocks_len = plan.end_of_managers.max(at + ext_len) - at;
                (post, plan, at, blocks_len)
            }
        };
        let final_eof = if reused {
            // The tail landed inside the file; the end-of-allocation is where this
            // commit's appends already left it, page-aligned by `pad_to_page`.
            self.image.len()
        } else {
            align_up(ext_addr + blocks_len, page_size)
        };

        let ext_oh = if plan.is_empty() {
            // No free space to track: an empty persist message, page-aligned.
            let info = FileSpaceInfo::persistent_empty(strategy, threshold, page_size);
            build_v2_object_header(&self.rewrite_extension_region(old_ext_addr, &info)?)?
        } else {
            // Paged convention (matching the from-scratch writer): the managers are
            // ordinary metadata below a page-aligned end-of-allocation.
            let info = FileSpaceInfo::persistent_managers(
                strategy, threshold, page_size, plan.slots, final_eof,
            );
            build_v2_object_header(&self.rewrite_extension_region(old_ext_addr, &info)?)?
        };
        debug_assert_eq!(
            ext_oh.len() as u64,
            ext_len,
            "extension length must be stable across the placeholder and real messages"
        );

        // Write the extension, then every manager block. Both forms are safe against
        // a crash here: an appended tail is past everything live, and a reused one
        // sits in space an earlier commit freed. Neither is referenced until the
        // repoint below.
        let region = (ext_addr, blocks_len);
        self.write_tail_block(region, ext_addr, &ext_oh)?;
        for b in &plan.blocks {
            let (fshd, fsse) =
                serialize_file_fsm(&b.sections, b.fshd_addr, b.fsse_addr, os, b.class);
            self.write_tail_block(region, b.fshd_addr, &fshd)?;
            self.write_tail_block(region, b.fsse_addr, &fsse)?;
        }
        // An appended tail ends mid-page; pad it out, so the end-of-allocation stays
        // a whole number of pages. A reused tail is already inside the file, and
        // matched its reservation exactly, so there is nothing to pad.
        self.pad_zeros_to(final_eof)?;
        // Exactly the bytes written, contiguous from the extension. A session that
        // reopens this file records the same extents from the message and the
        // manager headers, so nothing depends on remembering this across a close.
        let new_old_blocks = vec![(ext_addr, blocks_len)];

        // Barrier, then repoint the superblock (root, eof, and the new extension)
        // — the linearization point — and sync it.
        self.barrier()?;
        let mut new_sb = self.superblock.clone();
        new_sb.root_group_address = new_root;
        new_sb.eof_address = final_eof;
        new_sb.superblock_extension_address = Some(ext_addr);
        new_sb.consistency_flags = 0;
        let sb_bytes = new_sb.serialize();
        self.write_at(self.sb_sig_off, &sb_bytes)?;
        self.repointed = true;
        self.barrier()?;
        self.superblock = new_sb;

        // The repoint is durable. Only now are this commit's vacated regions
        // genuinely free, so adopt the lists built above and drop the padding tails
        // they already account for. The blocks just written become the ones the next
        // commit supersedes.
        if let Some(pg) = self.paged.as_mut() {
            pg.meta = post.meta;
            pg.raw = post.raw;
            pg.meta_pad.clear();
            pg.raw_pad.clear();
            if !reused {
                // The tail page is fresh metadata: the managers sit in it, so a
                // following append of metadata may keep packing that page, and the
                // rest of it is free metadata this session can spend. The managers
                // just written cannot say so — they would have to describe space
                // whose size their own length decides — so the next commit records
                // it, from here. A reused tail leaves end-of-file where the data put
                // it, so the outgoing page type stands and there is no padding.
                pg.last = Some(PageType::Meta);
                pg.meta
                    .free(ext_addr + blocks_len, final_eof - (ext_addr + blocks_len));
            }
        }
        self.persist = Some(PersistState {
            strategy,
            threshold,
            page_size,
            old_blocks: new_old_blocks,
        });
        // The managers now sit at the tail, so nothing is owed until the file
        // grows past them again.
        self.fsm_len = self.image.len();
        Ok(())
    }

    /// Pad a paged file to a page boundary if its tail page is partially filled,
    /// recording the padding as free space of the tail page's type. A no-op on a
    /// non-paged file or an already-aligned one.
    fn pad_to_page(&mut self) -> Result<(), Error> {
        let len = self.image.len();
        let pad = match &self.paged {
            Some(pg) if len % pg.page_size != 0 => {
                Some((pg.last, pg.page_size - len % pg.page_size))
            }
            _ => None,
        };
        if let Some((last, pad_len)) = pad {
            let pad_at = len;
            self.append(&vec![0u8; pad_len.to_usize()?])?;
            if let Some(pg) = self.paged.as_mut() {
                match last {
                    // A partially-filled tail page at commit time is a raw page:
                    // the last thing the apply loop writes for a dataset is its
                    // header, but a commit that only wrote raw data ends on one.
                    Some(PageType::Meta) => pg.meta_pad.push((pad_at, pad_len)),
                    Some(PageType::Raw) => pg.raw_pad.push((pad_at, pad_len)),
                    // No typed append this commit, so the tail page is one a
                    // previous session left non-aligned — a crash, since a clean
                    // close pads. Its type is unknown, and recording the padding
                    // under a guess would advertise it for reuse of that type and
                    // mix the page, so leave it untracked (see `PagedEdit::begin`,
                    // which makes the same call for the same reason). Reuse made
                    // this reachable: before it, every commit appended at least the
                    // root group header, so `last` was always known here.
                    None => {}
                }
            }
        }
        Ok(())
    }

    /// The free lists a paged commit is about to persist: the session's durable
    /// lists plus the page-padding tails this commit's appends left behind, the
    /// regions it vacated, and the superseded extension and manager blocks (all
    /// metadata, dead once the superblock is repointed).
    ///
    /// Returned as temporaries rather than folded into the session, exactly as the
    /// flat path builds `post`: every region gathered here is still *live* until
    /// that repoint, so the session's own lists must not learn about it until the
    /// repoint succeeds. Everything in between can fail (the extension rewrite,
    /// each write, each barrier), and a session that survived a failed commit
    /// while believing live extents were free would hand them out on the next
    /// commit — silently destroying the objects still occupying them.
    ///
    /// Called once to *size* the tail and again for each round the tail's
    /// placement takes to settle, since what remains depends on the space the tail
    /// has drawn for itself ([`tail_layout`](Self::tail_layout)). Each call clones
    /// both lists; they hold one region per hole, not per byte, so the copies are
    /// small.
    fn paged_post_free(
        &self,
        to_free: &[(u64, u64, PageType)],
        old_blocks: &[(u64, u64)],
    ) -> PagedPostFree {
        let pg = self
            .paged
            .as_ref()
            .expect("commit_persisting_paged is only called on a paged file");
        let (mut meta, mut raw) = (pg.meta.clone(), pg.raw.clone());
        // Carried through untouched: nothing this engine frees is of unknown
        // page type, and nothing it places comes out of that list.
        let unclassified = free_sections(&pg.unclassified);
        let mut free = |a: u64, l: u64, ty: PageType| {
            PagedEdit::route_free(&mut meta, &mut raw, a, l, ty);
        };
        for &(a, l) in &pg.meta_pad {
            free(a, l, PageType::Meta);
        }
        for &(a, l) in &pg.raw_pad {
            free(a, l, PageType::Raw);
        }
        for &(a, l, ty) in to_free {
            free(a, l, ty);
        }
        // The superseded extension and manager blocks, which are metadata wherever
        // they sat: the tail is placed as metadata, and a page it had to open for
        // itself was opened as metadata too.
        for &(a, l) in old_blocks {
            free(a, l, PageType::Meta);
        }
        PagedPostFree {
            meta,
            raw,
            unclassified,
        }
    }

    /// Reserve free metadata space for a paged commit's tail and plan the manager
    /// blocks for the address it got, or `None` when nothing reusable fits — the
    /// caller then appends instead.
    ///
    /// The reservation and the plan are mutually dependent: drawing `len` bytes out
    /// of the free lists changes the sections those very blocks record, and a
    /// different section set is a different length. This proposes a length, plans
    /// against the lists as they stand once that length is taken, and accepts only
    /// an exact agreement. Anything else hands the reservation back and proposes
    /// the length the plan just came out at, which is the natural next candidate.
    ///
    /// Both kinds of disagreement have to be rejected, not just the obvious one. A
    /// plan **longer** than its reservation would write past it into whatever lives
    /// after. A plan **shorter** would leave bytes inside the reservation that no
    /// manager records and the next commit does not free — the leak this whole
    /// change exists to remove, reintroduced a few bytes at a time.
    ///
    /// Iteration is capped rather than trusted to converge: the length is a
    /// step function of the section set, so a proposal can in principle oscillate
    /// between two values that each imply the other. Appending is always available
    /// and always correct, so giving up costs a page rather than a guarantee.
    fn tail_layout(
        &mut self,
        to_free: &[(u64, u64, PageType)],
        old_blocks: &[(u64, u64)],
        ext_len: u64,
        page_size: u64,
        os: u8,
    ) -> Option<(PagedPostFree, PagedManagerPlan, u64, u64)> {
        /// Enough rounds for the section set to settle after a reservation shrinks
        /// it, without letting an oscillating proposal spin.
        const ROUNDS: usize = 4;

        // The first proposal: what the blocks would measure with nothing reserved.
        // A manager block's length depends only on the sections it records, never
        // on its address, so a start of 0 measures the blocks alone.
        let probe = self.paged_post_free(to_free, old_blocks);
        let mut proposed = ext_len
            + plan_paged_managers(
                &free_sections(&probe.meta),
                &free_sections(&probe.raw),
                &probe.unclassified,
                page_size,
                0,
                os,
            )
            .end_of_managers;

        for _ in 0..ROUNDS {
            let pg = self
                .paged
                .as_mut()
                .expect("commit_persisting_paged is only called on a paged file");
            let at = pg.alloc_typed(proposed, PageType::Meta)?;
            let post = self.paged_post_free(to_free, old_blocks);
            // Class the free space into its managers and place their blocks after
            // the extension. Shared with the bounded backend so both lay out
            // identically.
            let plan = plan_paged_managers(
                &free_sections(&post.meta),
                &free_sections(&post.raw),
                &post.unclassified,
                page_size,
                at + ext_len,
                os,
            );
            // An empty plan writes no blocks at all, leaving the tail the extension
            // alone; `end_of_managers` is then its own start.
            let blocks_len = plan.end_of_managers.max(at + ext_len) - at;
            if blocks_len == proposed {
                return Some((post, plan, at, blocks_len));
            }
            let pg = self
                .paged
                .as_mut()
                .expect("the paged state outlives this loop");
            PagedEdit::route_free(&mut pg.meta, &mut pg.raw, at, proposed, PageType::Meta);
            proposed = blocks_len;
        }
        None
    }

    /// Write one block of a paged commit's tail at the address its plan gave it,
    /// which is either the current end-of-file (the tail is being appended) or
    /// inside `region` — the span reserved from the free lists — when the tail is
    /// being reused into space an earlier commit left.
    ///
    /// The region is checked rather than asserted, for the reason
    /// [`place`](Self::place) checks its own reservation: a reused span is followed
    /// by live bytes, so a block running past it would silently destroy a
    /// neighboring object, and this is the one write in a commit that lands in the
    /// middle of a live file. `blocks_len` is derived from the same plan that
    /// placed these blocks, so the two cannot disagree today; the comparison is
    /// what keeps that true if the plan ever learns a length this sizing does not
    /// model.
    fn write_tail_block(
        &mut self,
        region: (u64, u64),
        addr: u64,
        bytes: &[u8],
    ) -> Result<(), Error> {
        let (start, len) = region;
        if addr < start || addr + bytes.len() as u64 > start + len {
            return Err(Error::Format(FormatError::SerializationError(format!(
                "a paged commit's tail reserved [{start}, {}) but placed {} bytes at {addr}",
                start + len,
                bytes.len()
            ))));
        }
        if addr == self.image.len() {
            let written = self.append(bytes)?;
            debug_assert_eq!(written, addr, "an appended block must land at end-of-file");
            return Ok(());
        }
        self.write_at(
            usize::try_from(addr)
                .map_err(|_| Error::EditUnsupported("tail address exceeds this platform"))?,
            bytes,
        )
    }

    /// Extend the file with zeros up to `target` (>= the current length), used by
    /// the paged tail to pad the final metadata page to its boundary.
    fn pad_zeros_to(&mut self, target: u64) -> Result<(), Error> {
        let len = self.image.len();
        if target > len {
            let pad = (target - len).to_usize()?;
            self.append(&vec![0u8; pad])?;
        }
        debug_assert_eq!(self.image.len(), target);
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
        let region =
            Self::gather_oh_messages(&self.image(), ext_addr as u64, self.superblock.base_address)?;
        rewrite_extension_region_bytes(&region, info)
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

    /// Collect every message of the object header at `addr` into one contiguous
    /// region, following continuation blocks across chunks and dropping the
    /// `Continuation` messages themselves. Re-emitting the result through
    /// [`build_v2_object_header`] collapses a multi-chunk header (as the
    /// reference C library often writes) into a single chunk, which is how this
    /// editor rebuilds headers. The chunk-0 prefix is validated by
    /// [`oh_region_at`]; each continuation block must be a well-formed `OCHK`
    /// block within the file.
    ///
    /// Reads each header chunk out of `src` as one bounded buffer rather than
    /// indexing a whole-file image, so this serves a session whose file is not
    /// mirrored in memory (issue #198).
    fn gather_oh_messages<S: Source + ?Sized>(
        src: &S,
        addr: u64,
        base: u64,
    ) -> Result<Vec<u8>, Error> {
        let mut out = Vec::new();
        for chunk in read_oh_chunks(src, addr, base)? {
            let (region, mut p) = chunk.message_region();
            while let Some((msg_type, _body, body_end)) = next_message(region, p)? {
                if msg_type != MessageType::ObjectHeaderContinuation {
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
        let oh = ObjectHeader::parse_from_source(&self.image(), addr as u64, os, ls, base)?;
        if oh
            .messages
            .iter()
            .any(|m| m.msg_type == MessageType::DataLayout)
        {
            return Err(Error::EditUnsupported(
                "a target path names a dataset, not a group",
            ));
        }
        let entries = resolve_group_entries_from_source(&self.image(), &oh, os, ls, base)?;

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
                if m.data.len() > OBJECT_HEADER_MESSAGE_MAX {
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
        let sig = self.image().read_metadata_at(addr as u64, 4);
        if sig.as_deref() != Ok(&b"OHDR"[..]) {
            return self.reconstruct_v1_group(addr);
        }
        let mut region =
            Self::gather_oh_messages(&self.image(), addr as u64, self.superblock.base_address)?;
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

    /// The refusals a staged value overwrite (`write_dataset`) makes from the
    /// staged builder alone, with no on-disk header involved.
    ///
    /// [`stage_dataset_write`](Self::stage_dataset_write) applies these as the
    /// write is staged, so the call that configured the builder is the one that
    /// reports the mistake. A refusal that reads only `fd` belongs here; one that
    /// needs the target's header belongs in
    /// [`prepare_write`](Self::prepare_write).
    fn refuse_unsupported_overwrite(fd: &FlatDataset) -> Result<(), Error> {
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
        // add path knows how to resolve: place the global heap collection, then
        // patch the placeholders once its address is known, before the data block
        // itself is written. The overwrite path has no such step, and a
        // same-length overwrite is flushed straight over the existing data block
        // with no apply loop at all — so refuse rather than write unpatched (heap
        // address 0) placeholders as if they were final.
        if fd.vl_string_staging.is_some() {
            return Err(Error::EditUnsupported(
                "write_dataset cannot overwrite a variable-length-string dataset's \
                 data in place yet",
            ));
        }

        // `reference_targets` holds elements only the add path resolves, in
        // `preflight_reference_targets` and the apply loop after it; the
        // overwrite path never reads the field. Checked on the field rather than
        // on `with_path_references`, since every producer stages elements that
        // are equally unresolved. Left unrefused, a staged overwrite writes
        // address-zero placeholders over a working reference dataset and
        // `commit` reports `Ok` (issue #318). `with_reference_data` supplies
        // resolved addresses and overwrites like any other value.
        if fd.reference_targets.is_some() {
            return Err(Error::EditUnsupported(
                "write_dataset cannot overwrite an object-reference dataset's \
                 data in place yet",
            ));
        }

        Ok(())
    }

    /// Preflight a staged value overwrite (`write_dataset`): resolve the dataset
    /// at `addr`, validate that the staged `fd` matches it byte-exactly in
    /// datatype and shape, and classify how the bytes will be applied. No file
    /// bytes are written here — this is part of the all-or-nothing preflight, so a
    /// rejected write leaves the commit unapplied.
    ///
    /// Contiguous, compact, and chunked (including filtered) datasets are all
    /// supported; the chunk geometry, filter pipeline, and chunk index come from
    /// the on-disk header (a chunk index this engine cannot enumerate — a
    /// version-2 B-tree — is refused). A datatype or shape that differs from the
    /// on-disk dataset's is likewise refused — this is a value overwrite, not a
    /// reshape or retype.
    ///
    /// Every refusal `fd` can decide on its own has been made by then, where the
    /// write is staged.
    fn prepare_write<S: Source + ?Sized>(
        src: &S,
        addr: u64,
        fd: &FlatDataset,
        base: u64,
    ) -> Result<WritePlan, Error> {
        // Enforced by construction: `staged.writes` has one producer, and it
        // refuses there. Asserted rather than re-refused because a second caller
        // that skipped it would not fail here — a chunked builder would fall
        // through and be written as a plain contiguous overwrite, with
        // `chunk_options` silently dropped.
        debug_assert!(
            Self::refuse_unsupported_overwrite(fd).is_ok(),
            "a staged write reached prepare_write without refuse_unsupported_overwrite"
        );

        let region = Self::gather_oh_messages(src, addr, base)?;

        // Locate the datatype, dataspace, and data-layout messages, and detect a
        // filter pipeline (filtered storage is always chunked, never contiguous).
        let mut datatype: Option<(usize, usize)> = None;
        let mut dataspace: Option<(usize, usize)> = None;
        let mut layout: Option<(usize, usize)> = None;
        let mut filter: Option<(usize, usize)> = None;
        // The dataset's own fill value, which the edge overhang of a partial
        // chunk must hold (issue #296). Both message forms, newest wins.
        let mut fill_msg: Option<(MessageType, usize, usize)> = None;
        let mut has_link = false;
        let mut p = 0;
        while let Some((msg_type, body, body_end)) = next_message(&region, p)? {
            match msg_type {
                MessageType::Datatype => datatype = Some((body, body_end)),
                MessageType::Dataspace => dataspace = Some((body, body_end)),
                MessageType::DataLayout => layout = Some((body, body_end)),
                MessageType::FilterPipeline => filter = Some((body, body_end)),
                // Versioned beats legacy, and within a type the first wins —
                // the same rule `Dataset::fill_bytes` and `Located::from_walk`
                // apply, so all three agree on a header carrying more than one.
                MessageType::FillValue
                    if !matches!(fill_msg, Some((MessageType::FillValue, ..))) =>
                {
                    fill_msg = Some((msg_type, body, body_end));
                }
                MessageType::FillValueOld if fill_msg.is_none() => {
                    fill_msg = Some((msg_type, body, body_end));
                }
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
                            .is_some_and(|e| e as u64 <= src.len())
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
                // row-major grid order, then re-encode through the on-disk
                // pipeline when the dataset is filtered.
                // The overhang past the dataset's edge holds the dataset's own
                // fill value, not zeros: an allocated chunk is expected to carry
                // it wherever nothing was written, and those slots are what a
                // reader returns once the dataset is extended into them (#296).
                let padding = fill_msg
                    .map_or(crate::fill_value::PaddingFill::Zero, |(mt, b, e)| {
                        crate::fill_value::PaddingFill::from_message(mt, &region[b..e])
                    });
                let split = split_into_chunks(
                    &fd.raw,
                    &disk_ds.dimensions,
                    &spatial,
                    element_size,
                    padding.pattern(element_size),
                )
                .map_err(Error::Format)?;
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
                    let ctx = ChunkContext::from_datatype(&spatial, &fd.dt)?;
                    let mut encoded = Vec::with_capacity(split.len());
                    // One encoder across the rewrite; see `FilterScratch`.
                    let mut scratch = FilterScratch::new();
                    for buf in &split {
                        encoded.push(compress_chunk_with(&mut scratch, buf, &pipeline, ctx)?);
                    }
                    encoded
                } else {
                    split
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
                    &BaseOffsetSource { inner: src, base },
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
                    shape: disk_ds.dimensions.clone(),
                    chunk_dims: spatial,
                    element_size,
                    maxshape,
                    pipeline_message,
                    meta,
                    chunk_bytes: new_chunk_bytes,
                    old_addr: addr,
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
    /// Reads only; no bytes are written here. `src` is the file image and `base`
    /// its userblock base; the dataset's stored (base-relative) structures are read
    /// through a `base`-shifted view.
    fn prepare_append<S: Source + ?Sized>(
        src: &S,
        addr: u64,
        ab: &AppendBuilder,
        base: u64,
    ) -> Result<MovingWrite, Error> {
        if ab.dt_conflict {
            return Err(Error::AppendUnsupported(
                "append mixes element types in one builder; use one element type per \
                 append_dataset call",
            ));
        }

        let region = Self::gather_oh_messages(src, addr, base)?;

        // Locate the datatype, dataspace, data-layout, and filter-pipeline
        // messages, and detect a group (link) header.
        let mut datatype: Option<(usize, usize)> = None;
        let mut dataspace: Option<(usize, usize)> = None;
        let mut layout: Option<(usize, usize)> = None;
        let mut filter: Option<(usize, usize)> = None;
        // The dataset's own fill value, which the edge overhang of a partial
        // chunk must hold (issue #296). Both message forms, newest wins.
        let mut fill_msg: Option<(MessageType, usize, usize)> = None;
        let mut has_link = false;
        let mut p = 0;
        while let Some((msg_type, body, body_end)) = next_message(&region, p)? {
            match msg_type {
                MessageType::Datatype => datatype = Some((body, body_end)),
                MessageType::Dataspace => dataspace = Some((body, body_end)),
                MessageType::DataLayout => layout = Some((body, body_end)),
                MessageType::FilterPipeline => filter = Some((body, body_end)),
                // Versioned beats legacy, and within a type the first wins —
                // the same rule `Dataset::fill_bytes` and `Located::from_walk`
                // apply, so all three agree on a header carrying more than one.
                MessageType::FillValue
                    if !matches!(fill_msg, Some((MessageType::FillValue, ..))) =>
                {
                    fill_msg = Some((msg_type, body, body_end));
                }
                MessageType::FillValueOld if fill_msg.is_none() => {
                    fill_msg = Some((msg_type, body, body_end));
                }
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

        if base > src.len() {
            return Err(Error::AppendUnsupported(
                "userblock base address past end-of-file",
            ));
        }
        let view = BaseOffsetSource { inner: src, base };

        // The rebuilt index's element format (bare address vs address+size+mask) is
        // chosen by `has_filters`; it must agree with the source index's client id,
        // or the kept chunks — carried by metadata into the new index — would be
        // re-encoded in the wrong element width.
        if let Some(idx_addr) = *btree_address {
            let hdr =
                ExtensibleArrayHeader::parse_from_source(&view, idx_addr, OFFSET_SIZE, LENGTH_SIZE)
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
        let infos = enumerate_chunks_from_source(&view, &dl, &disk_ds, OFFSET_SIZE, LENGTH_SIZE)
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
            let len = partial.chunk_size as usize;
            partial
                .address
                .checked_add(len as u64)
                .filter(|&e| e <= view.len())
                .ok_or(Error::AppendUnsupported(
                    "trailing chunk extends past end-of-file",
                ))?;
            let stored = view
                .read_exact_at(partial.address, len)
                .map_err(|_| Error::AppendUnsupported("trailing chunk could not be read"))?;
            let full = if let Some(pl) = &pipeline {
                let ctx = ChunkContext::from_datatype(&spatial, &disk_dt)?;
                decompress_chunk(&stored, pl, ctx, partial.filter_mask).map_err(Error::Format)?
            } else {
                stored
            };
            let live_elems = usize::try_from(current_dim0 % chunk_elems)
                .map_err(|_| Error::AppendUnsupported("chunk length exceeds this platform"))?;
            let live_bytes = live_elems * element_size.get();
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

        // Split the tail into full chunk buffers and compress each through the
        // pipeline when filtered. The final chunk's overhang takes the dataset's
        // fill value, which is what a reader returns from those slots once a
        // later append or resize reaches them (#296).
        let tail_len_elems = new_dim0 - (n_full as u64) * chunk_elems;
        let padding = fill_msg.map_or(crate::fill_value::PaddingFill::Zero, |(mt, b, e)| {
            crate::fill_value::PaddingFill::from_message(mt, &region[b..e])
        });
        let split = split_into_chunks(
            &tail_raw,
            &[tail_len_elems],
            &spatial,
            element_size,
            padding.pattern(element_size),
        )
        .map_err(Error::Format)?;
        let new_chunk_bytes: Vec<Vec<u8>> = if let Some(pl) = &pipeline {
            let ctx = ChunkContext::from_datatype(&spatial, &disk_dt)?;
            let mut out = Vec::with_capacity(split.len());
            // One encoder across the appended tail; see `FilterScratch`.
            let mut scratch = FilterScratch::new();
            for buf in &split {
                out.push(compress_chunk_with(&mut scratch, buf, pl, ctx).map_err(Error::Format)?);
            }
            out
        } else {
            split
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
            has_filters,
            kept_chunks,
            new_chunk_bytes,
            old_addr: addr,
            old_tail_extent,
        })
    }

    /// Parse the object header at `addr` into a copyable model, validating that
    /// every message can be reproduced faithfully (verbatim message bytes, with
    /// only the contiguous data address and child link targets repointed).
    /// Dense (fractal-heap) attribute storage is read out of the source heap into
    /// a parsed attribute set carried on the model (`dense_attrs`) and re-emitted
    /// into a fresh heap on write, within the bounds that heap declares (see
    /// `file_writer::dense_attrs_check`); an attribute too large to hold as a
    /// managed object is re-emitted as a *huge* object. Rejects multi-chunk
    /// headers, dense or soft/external links, chunked/old-version data layouts, and
    /// headers that are neither a dataset nor a group.
    fn read_object<S: Source + ?Sized>(src: &S, addr: u64, base: u64) -> Result<ObjModel, Error> {
        let region = Self::gather_oh_messages(src, addr, base)?;

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
                ObjectHeader::parse_from_source(src, addr, OFFSET_SIZE, LENGTH_SIZE, base).map_err(|_| {
                    Error::EditUnsupported(
                        "a source object header with dense attributes could not be parsed for copying",
                    )
                })?;
            // The heap address in the Attribute Info message is stored relative to
            // the base address, so the walk gets the source framed past its
            // userblock — the same view the reader uses. `base` is 0 for a plain
            // file, where this is `src` itself.
            if base > src.len() {
                return Err(Error::EditUnsupported(
                    "a source file's userblock is larger than the file itself",
                ));
            }
            let framed = BaseOffsetSource { inner: src, base };
            let attrs = crate::attribute::extract_attributes_full_from_source(
                &framed,
                &header,
                OFFSET_SIZE,
                LENGTH_SIZE,
            )
            .map_err(|_| {
                Error::EditUnsupported(
                    "a source object's dense (fractal-heap) attributes could not be read for copying",
                )
            })?;
            // The typed error names the offending attribute, which the previous
            // blanket `EditUnsupported` message could not.
            crate::file_writer::dense_attrs_check(&attrs).map_err(Error::Format)?;
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
    /// `src` is the image the source object lives in: this session's own file image
    /// for an in-file [`copy`](Self::copy), or another file's image for a cross-file
    /// [`copy_from`](Self::copy_from). `base` is that image's userblock base (the
    /// session's own base for an in-file copy, always 0 for a cross-file copy, whose
    /// source is gated to base 0): the stored, base-relative addresses read out of
    /// the source headers are shifted by it to index `src`. When `cross_file` is set,
    /// every copied object header is additionally screened by
    /// [`reject_foreign_addresses`] — verbatim bytes that embed a *source-file*
    /// absolute address (variable-length or reference data, a committed datatype)
    /// would dangle in another file and are refused, whereas an in-file copy keeps
    /// them valid by sharing the source file's heaps and objects.
    fn read_copy_subtree<S: Source + ?Sized>(
        src: &S,
        addr: u64,
        depth: u32,
        cross_file: bool,
        base: u64,
    ) -> Result<CopyTree, Error> {
        if depth >= MAX_COPY_DEPTH {
            return Err(Error::EditUnsupported(
                "copy source nests too deeply (possible hard-link cycle)",
            ));
        }
        // `base` is the userblock base of the image `src`: this session's own base
        // for an in-file copy, and always 0 for a cross-file copy (the source is
        // gated to base 0 in `copy_from`). `addr` is an absolute offset into `src`;
        // the stored (base-relative) addresses `read_object` returns for contiguous
        // data, chunk storage, and child links are converted to absolute offsets by
        // adding `base` before `src` is read or a child is descended into.
        match Self::read_object(src, addr, base)? {
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
                // offset into `src` before reading the data block out.
                let start = data_addr
                    .checked_add(base)
                    .ok_or(Error::EditUnsupported("data address exceeds this platform"))?;
                let len = usize::try_from(data_size)
                    .map_err(|_| Error::EditUnsupported("data size exceeds this platform"))?;
                start
                    .checked_add(len as u64)
                    .filter(|&e| e <= src.len())
                    .ok_or(Error::EditUnsupported("dataset data is out of bounds"))?;
                Ok(CopyTree::DatasetContiguous {
                    region,
                    addr_off,
                    data: src
                        .read_exact_at(start, len)
                        .map_err(|_| Error::EditUnsupported("dataset data is out of bounds"))?,
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
                    raw_size: _,
                    maxshape,
                } = chunked_geometry(&dt, &ds, &layout)?;

                // The layout's chunk-index address and every chunk address it leads
                // to are stored base-relative, so enumerate and read on a
                // base-relative view of the source image (the identity on a base-0
                // file). The returned addresses are then offsets into `dview`.
                let dview = BaseOffsetSource { inner: src, base };

                // Enumerate the source chunks and map them onto a dense grid; a
                // sparse (holed/unallocated) dataset cannot be reproduced by the
                // verbatim layout path, which needs every grid slot filled.
                let infos =
                    enumerate_chunks_from_source(&dview, &layout, &ds, OFFSET_SIZE, LENGTH_SIZE)?;
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
                    let len = ci.chunk_size as usize;
                    ci.address
                        .checked_add(len as u64)
                        .filter(|&e| e <= dview.len())
                        .ok_or(Error::EditUnsupported("chunk data is out of bounds"))?;
                    chunk_bytes.push(
                        dview
                            .read_exact_at(ci.address, len)
                            .map_err(|_| Error::EditUnsupported("chunk data is out of bounds"))?,
                    );
                    meta.push(ChunkMeta {
                        compressed_size: ci.chunk_size as u64,
                        filter_mask: ci.filter_mask,
                    });
                }

                Ok(CopyTree::DatasetChunked {
                    region,
                    shape: ds.dimensions.clone(),
                    chunk_dims,
                    element_size,
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
                    // before descending so `addr` stays an absolute offset into `src`.
                    let child = child.checked_add(base).ok_or(Error::EditUnsupported(
                        "child address exceeds this platform",
                    ))?;
                    kids.push((
                        name,
                        Self::read_copy_subtree(src, child, depth + 1, cross_file, base)?,
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
                let oh = build_v2_object_header(&region)?;
                self.alloc_or_append_typed(&oh, PageType::Meta)
            }
            CopyTree::DatasetContiguous {
                region,
                addr_off,
                data,
                dense_attrs,
            } => {
                let new_data_addr = self.alloc_or_append_typed(data, PageType::Raw)?;
                let mut region = region.clone();
                // The placement is an absolute offset; the data-layout
                // address field stores it relative to the userblock base.
                region[*addr_off..*addr_off + 8]
                    .copy_from_slice(&(new_data_addr - base).to_le_bytes());
                // The dense heap is placed independently of the data — it is
                // built for whatever address it gets (see `append_dense_attrs`),
                // so no ordering between the two is owed.
                self.append_dense_attrs(&mut region, dense_attrs)?;
                let oh = build_v2_object_header(&region)?;
                self.alloc_or_append_typed(&oh, PageType::Meta)
            }
            CopyTree::DatasetChunked {
                region,
                shape,
                chunk_dims,
                element_size,
                maxshape,
                pipeline_message,
                meta,
                chunk_bytes,
                dense_attrs,
            } => self.write_chunked_relocatable(
                region,
                shape,
                chunk_dims,
                *element_size,
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
                // The dense heap is built for whatever address it is placed at
                // (see `append_dense_attrs`), so it needs no ordering against the
                // children's headers and data.
                self.append_dense_attrs(&mut region, dense_attrs)?;
                let oh = build_v2_object_header(&region)?;
                self.alloc_or_append_typed(&oh, PageType::Meta)
            }
        }
    }

    /// Write a chunked dataset's storage and return its new object-header address
    /// — the shared write half of a chunked copy ([`CopyTree::DatasetChunked`])
    /// and a relocating chunked overwrite ([`MovingWrite::Chunked`]).
    ///
    /// A fresh chunk-data blob and index are laid out relocatably via
    /// [`plan_chunked_data_verbatim`] / [`emit_chunked_data_verbatim`], pulling
    /// each chunk's already-compressed bytes from `chunk_bytes` (in dense
    /// row-major grid order) and carrying `meta`'s sizes and filter masks and the
    /// source `pipeline_message` verbatim — no recompression, no filter-parameter
    /// reconstruction. Like [`build_chunked_dataset`](Self::build_chunked_dataset)
    /// the blob is sized from its plan before it is placed, so it can go into a
    /// freed region that fits it as readily as at end-of-file. The verbatim header
    /// `region`'s data-layout message is then swapped for the one the planner
    /// produced (every other message preserved), any dense attribute heap is
    /// placed, and the header is written into reusable freed space or at
    /// end-of-file.
    #[expect(
        clippy::too_many_arguments,
        reason = "the chunked rebuild needs the full geometry, \
        pipeline, and chunk payloads; bundling them into a struct would only move the list"
    )]
    fn write_chunked_relocatable(
        &mut self,
        region: &[u8],
        shape: &[u64],
        chunk_dims: &[u64],
        element_size: NonZeroUsize,
        maxshape: Option<&[u64]>,
        pipeline_message: Option<&[u8]>,
        meta: &[ChunkMeta],
        chunk_bytes: &[Vec<u8>],
        dense_attrs: &[crate::attribute::AttributeMessage],
    ) -> Result<u64, Error> {
        // Plan once at a provisional base purely to size the data region: the plan
        // walks chunk *sizes* and sizes the index from its layout, so it touches
        // no bytes at all, and every address it embeds sits in a fixed-width
        // field, so its total length is the same wherever the blob lands. That is
        // what lets the address be chosen — a freed region or end-of-file — before
        // the bytes exist. This plan is discarded; the one built at the real base
        // below is what the emit works from.
        let sizing = plan_chunked_data_verbatim(
            meta,
            shape,
            chunk_dims,
            element_size,
            pipeline_message,
            0,
            maxshape,
        )?;
        let (_addr, layout_message) =
            self.place_relocatable(sizing.plan.total_len, PageType::Raw, |stored_base| {
                // Re-plan at the address the blob really occupies, so its embedded
                // addresses resolve to their real file offsets once the reader adds
                // the userblock base back (see `build_chunked_dataset`).
                let layout = plan_chunked_data_verbatim(
                    meta,
                    shape,
                    chunk_dims,
                    element_size,
                    pipeline_message,
                    stored_base,
                    maxshape,
                )?;
                let mut buf =
                    Vec::with_capacity(usize::try_from(layout.plan.total_len).unwrap_or(0));
                emit_chunked_data_verbatim(
                    &mut buf,
                    &layout.plan,
                    &SliceChunkProvider {
                        chunks: chunk_bytes,
                    },
                )?;
                Ok((buf, layout.layout_message))
            })?;
        // Swap the data-layout message for the rebuilt one; keep every other header
        // message (datatype, dataspace, fill value, filter pipeline, attributes)
        // verbatim.
        let mut new_region = replace_layout_message(region, &layout_message)?;
        self.append_dense_attrs(&mut new_region, dense_attrs)?;
        let oh = build_v2_object_header(&new_region)?;
        self.alloc_or_append_typed(&oh, PageType::Meta)
    }

    /// When `attrs` is non-empty, build a fresh dense (fractal-heap) attribute
    /// blob for it, place it, and splice the matching Attribute Info message onto
    /// `region`. A no-op for an empty set.
    ///
    /// The blob produced by [`file_writer::DenseAttrPlan::build`] is fully
    /// relocatable: every address it embeds is `base + fixed offset`, and its
    /// length is the same for every base, so it can go into a freed metadata
    /// region as readily as at end-of-file — the base it is built for is whichever
    /// address it gets. The reservation comes from the plan the blob is then
    /// built from, so sizing it costs no bytes. The freshly built heap is always
    /// same-file, so it never aliases the source heap even for an in-file copy.
    /// The caller has already validated [`file_writer::dense_attrs_check`].
    fn append_dense_attrs(
        &mut self,
        region: &mut Vec<u8>,
        attrs: &[crate::attribute::AttributeMessage],
    ) -> Result<(), Error> {
        if attrs.is_empty() {
            return Ok(());
        }
        let plan = crate::file_writer::dense_attrs_plan(attrs);
        let (_addr, attr_info_message) =
            self.place_relocatable(plan.blob_len(), PageType::Meta, |stored_base| {
                let blob = plan.build(stored_base);
                Ok((blob.blob, blob.attr_info_message))
            })?;
        region.extend_from_slice(&region_message(
            MessageType::AttributeInfo,
            &attr_info_message,
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
                let new_data_addr = self.alloc_or_append_typed(raw, PageType::Raw)?;
                let mut region = region.clone();
                // The placement is an absolute file offset; the contiguous
                // data-layout field stores it relative to the userblock base (`-
                // base`, a no-op on a base-0 file).
                region[*addr_off..*addr_off + 8]
                    .copy_from_slice(&(new_data_addr - base).to_le_bytes());
                // The data size field follows the 8-byte address in the contiguous
                // layout body; keep it in sync with the new length.
                let size_off = *addr_off + 8;
                region[size_off..size_off + 8].copy_from_slice(&(raw.len() as u64).to_le_bytes());
                let oh = build_v2_object_header(&region)?;
                self.alloc_or_append_typed(&oh, PageType::Meta)
            }
            MovingWrite::Compact { region, raw } => {
                let region = rebuild_compact_layout_region(region, raw)?;
                let oh = build_v2_object_header(&region)?;
                self.alloc_or_append_typed(&oh, PageType::Meta)
            }
            MovingWrite::Chunked {
                region,
                shape,
                chunk_dims,
                element_size,
                maxshape,
                pipeline_message,
                meta,
                chunk_bytes,
                ..
            } => self.write_chunked_relocatable(
                region,
                shape,
                chunk_dims,
                *element_size,
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
                has_filters,
                kept_chunks,
                new_chunk_bytes,
                ..
            } => self.write_appended_chunks(
                region,
                new_dataspace_body,
                chunk_dims_u32,
                *element_size,
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
                for (msg, collections) in pending_vl_attrs {
                    let mut msg = msg.clone();
                    let addrs = self.place_vl_collections(collections)?;
                    patch_vl_refs(&mut msg.raw_data, &addrs);
                    region.extend_from_slice(&region_message(
                        MessageType::Attribute,
                        &msg.serialize(LENGTH_SIZE),
                    ));
                }
                let oh = build_v2_object_header(&region)?;
                self.alloc_or_append_typed(&oh, PageType::Meta)
            }
        }
    }

    /// Apply a relocating append ([`MovingWrite::AppendedChunks`]): place the new
    /// (and any rewritten trailing) chunk bytes, rebuild a fresh
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
        element_size: NonZeroUsize,
        has_filters: bool,
        kept_chunks: &[WrittenChunk],
        new_chunk_bytes: &[Vec<u8>],
    ) -> Result<u64, Error> {
        let base = self.superblock.base_address;
        // Place each new chunk, reusing freed raw space where it fits. A chunk
        // embeds no addresses of its own, so it can go anywhere and the rebuilt
        // index below simply records where it went. Existing chunks keep their
        // in-place addresses and are carried by metadata alone.
        let mut combined: Vec<WrittenChunk> = kept_chunks.to_vec();
        for cb in new_chunk_bytes {
            let abs = self.alloc_or_append_typed(cb, PageType::Raw)?;
            combined.push(WrittenChunk {
                address: abs - base,
                compressed_size: cb.len() as u64,
                // This engine applies every filter to a new chunk (no per-chunk
                // skipping), so an appended chunk's mask is always 0. Kept chunks
                // carry their own (possibly nonzero) mask in `combined` already.
                filter_mask: 0,
            });
        }

        // Build the fresh Extensible Array for wherever it is placed: its embedded
        // block addresses are computed from `ea_base` (base-relative), so they
        // resolve correctly on a userblock (`base != 0`) file too. Its length does
        // not depend on that base, so `extensible_array_len` gives the reservation
        // from the array's layout without emitting a byte of it.
        //
        // The index goes in a *raw* page, not a metadata one: every other writer in
        // this crate places a chunk index in the same run as the chunk data, and
        // `chunked_storage_spans` reclaims every index as raw on that basis. Placing
        // this one in a metadata page would make it the single exception the reclaim
        // side then mis-files, advertising a metadata hole inside a raw page.
        //
        // The element width comes from the chunk geometry rather than from
        // `combined`, so a rebuild declares the same width the original index
        // did — including when the dataset it grows was created empty.
        let chunk_bytes =
            full_chunk_bytes(chunk_dims_u32.iter().map(|&d| u64::from(d)), element_size);
        // Rank 1 and unlimited along axis 0 (`prepare_append` refuses anything
        // else), so every chunk's index slot is its position in the grid and the
        // array is dense from zero.
        let slots = crate::chunked_write::IndexSlots::dense(&combined);
        let ea_len =
            extensible_array_len(&slots, chunk_bytes, OFFSET_SIZE, LENGTH_SIZE, has_filters);
        let (ea_addr, ()) = self.place_relocatable(ea_len, PageType::Raw, |ea_base| {
            let bytes = build_extensible_array_at(
                &slots,
                chunk_bytes,
                OFFSET_SIZE,
                LENGTH_SIZE,
                has_filters,
                ea_base,
            )
            .map_err(Error::Format)?;
            Ok((bytes, ()))
        })?;
        let ea_base = ea_addr - base;

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
            element_size.get() as u32,
        );
        let region = replace_dataspace_message(region, new_dataspace_body)?;
        let region = replace_layout_message(&region, &layout_body)?;
        let oh = build_v2_object_header(&region)?;
        self.alloc_or_append_typed(&oh, PageType::Meta)
    }

    /// Append `bytes` at end-of-file, returning the absolute address they were
    /// written at.
    fn append(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        self.image.append(bytes)
    }

    /// Overwrite bytes in place at `offset`. The caller guarantees the range
    /// already exists.
    fn write_at(&mut self, offset: usize, bytes: &[u8]) -> Result<(), Error> {
        self.image.write_at(offset as u64, bytes)
    }

    /// Ensure the next allocation begins in a page holding page type `ty`, on a
    /// paged file. A no-op on the common non-paged file.
    ///
    /// A paged file never mixes metadata and raw data within one page, so when the
    /// tail page holds the *other* type and is only partially filled it is first
    /// padded to a page boundary, the padding being recorded as free space of the
    /// outgoing type.
    ///
    /// Call this **before** reading the image's end-of-file ([`Source::len`]) to compute an
    /// address that will be embedded in the bytes being built: several callers
    /// (the chunk blob, the extensible-array index, the dense-attribute blob)
    /// build content whose interior addresses assume it lands at the current
    /// end-of-file, and padding inserted after that read would shift the landing
    /// address out from under them.
    fn begin_page(&mut self, ty: PageType) -> Result<(), Error> {
        // Destructure so the page state and the image are borrowed as the
        // separate fields they are.
        let Self { image, paged, .. } = self;
        match paged.as_mut() {
            Some(pg) => pg.begin(image.as_mut(), ty),
            None => Ok(()),
        }
    }

    /// Place `bytes` as page type `ty`, reusing a free region where one fits.
    /// Equivalent to [`reserve`](Self::reserve) followed by [`place`](Self::place),
    /// for the callers whose bytes are already built.
    fn alloc_or_append_typed(&mut self, bytes: &[u8], ty: PageType) -> Result<u64, Error> {
        let at = self.reserve(bytes.len() as u64, ty)?;
        self.place(at, bytes)
    }

    /// Choose where `len` bytes of page type `ty` will go: a reusable free region
    /// left by a prior commit, or the current end-of-file. The returned
    /// [`Placement`] must be handed to [`place`](Self::place) with exactly `len`
    /// bytes.
    ///
    /// Splitting the choice from the write is what lets a *relocatable* blob — one
    /// whose interior addresses are all `base + fixed offset`, like a chunked
    /// dataset's data region or a dense attribute heap — be built for a freed
    /// region rather than only for end-of-file: its size is known from its plan
    /// before its bytes exist, so the address can be settled first
    /// ([`place_relocatable`](Self::place_relocatable) is that sequence).
    ///
    /// Reuse only ever draws from free space vacated by *earlier* commits in this
    /// session (or seeded from the on-disk managers at open) — never space the
    /// current commit is about to free, which stays in its `to_free` list until
    /// after the superblock repoint. The bytes it overwrites are therefore already
    /// unreachable from the on-disk root, so a mid-commit crash cannot corrupt the
    /// live tree: the superblock still points at the prior, intact root.
    ///
    /// A paged file reuses within the matching page type, or out of a page holding
    /// nothing at all, which belongs to no type ([`PagedEdit::alloc_typed`]); every
    /// page stays homogeneous either way. It never opens a page for a reused region
    /// — the tail page is untouched by a write into the middle of the file.
    fn reserve(&mut self, len: u64, ty: PageType) -> Result<Placement, Error> {
        if let Some(addr) = self.alloc_free(len, ty) {
            return Ok(Placement::Reused { addr, len });
        }
        self.begin_page(ty)?;
        Ok(Placement::Appended {
            addr: self.image.len(),
            len,
        })
    }

    /// Draw `len` bytes from the free space of page type `ty`, or `None` when no
    /// single region is large enough.
    ///
    /// A non-paged file keeps one list for the whole file. A paged file keeps one
    /// per page type and must be served from the list matching `ty`: handing a
    /// metadata hole to raw data (or the reverse) would mix the two within a page,
    /// which is the single invariant the paged strategy exists to hold. A page
    /// holding nothing at all is the exception, and
    /// [`alloc_typed`](PagedEdit::alloc_typed) is where it is spent.
    fn alloc_free(&mut self, len: u64, ty: PageType) -> Option<u64> {
        let Some(pg) = self.paged.as_mut() else {
            return self.free.alloc(len);
        };
        pg.alloc_typed(len, ty)
    }

    /// Write `bytes` at the address [`reserve`](Self::reserve) handed out,
    /// returning that address.
    ///
    /// The length is checked rather than asserted: a reused region is sized to the
    /// reservation and is followed by live bytes, so writing more than was reserved
    /// would silently destroy a neighboring object. That makes it the one internal
    /// miscount worth paying a comparison to catch in every build.
    fn place(&mut self, at: Placement, bytes: &[u8]) -> Result<u64, Error> {
        if bytes.len() as u64 != at.len() {
            return Err(Error::Format(FormatError::SerializationError(format!(
                "a placement reserved {} bytes but built {}",
                at.len(),
                bytes.len()
            ))));
        }
        match at {
            Placement::Reused { addr, .. } => {
                self.write_at(
                    usize::try_from(addr).map_err(|_| {
                        Error::EditUnsupported("free-region address exceeds this platform")
                    })?,
                    bytes,
                )?;
                Ok(addr)
            }
            Placement::Appended { addr, .. } => {
                let written = self.append(bytes)?;
                debug_assert_eq!(
                    written, addr,
                    "an appended placement must land at end-of-file"
                );
                Ok(written)
            }
        }
    }

    /// Place a *relocatable* blob of `len` bytes as page type `ty`: pick its
    /// address first, then build it for that address with `build`, then write it
    /// there. Returns the address and whatever else `build` produced (typically the
    /// object-header message naming the blob, which embeds the same address).
    ///
    /// `build` receives the *stored* (base-relative) address the blob will occupy,
    /// since every address a blob embeds is stored base-relative and the reader
    /// recovers it as `stored + base_address`. On a file without a userblock the
    /// two are equal.
    ///
    /// `len` must be the length `build` will produce; [`place`](Self::place)
    /// rejects a mismatch. For every blob placed this way the length is a function
    /// of the content alone — the addresses sit in fixed-width fields — so every
    /// caller derives it, and none builds the blob to measure it:
    /// [`DenseAttrPlan::blob_len`](crate::file_writer::DenseAttrPlan::blob_len)
    /// for a dense attribute heap, [`extensible_array_len`] for an appended
    /// chunk index, and [`chunked_data_len`] / [`plan_chunked_data_verbatim`]
    /// for a whole chunked data region, both of which size their index through
    /// [`chunk_index_len`](crate::chunked_write::chunk_index_len).
    fn place_relocatable<T>(
        &mut self,
        len: u64,
        ty: PageType,
        build: impl FnOnce(u64) -> Result<(Vec<u8>, T), Error>,
    ) -> Result<(u64, T), Error> {
        let at = self.reserve(len, ty)?;
        let (bytes, extra) = build(at.address() - self.superblock.base_address)?;
        let addr = self.place(at, &bytes)?;
        Ok((addr, extra))
    }

    /// Place one variable-length dataset's or attribute's already-built,
    /// self-contained global heap collections (from
    /// [`build_global_heap_collections`] or a
    /// [`VlStringStaging::collections`]) and return, in the same order, the
    /// base-relative addresses its variable-length references should be patched
    /// to. A `GCOL` blob embeds no addresses of its own, so it can be appended
    /// (or dropped into reused free space) at any point in the apply loop,
    /// unlike a group or dataset header, which must be built last so it can name
    /// its children's real addresses. Each collection is placed independently,
    /// so they need not land contiguously.
    fn place_vl_collections(&mut self, collections: &[Vec<u8>]) -> Result<Vec<u64>, Error> {
        collections
            .iter()
            .map(|collection| {
                let addr = self.alloc_or_append_typed(collection, PageType::Meta)?;
                Ok(addr - self.superblock.base_address)
            })
            .collect()
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
    ///
    /// A path this same commit *deletes* (`delete_targets`) is refused for a
    /// different reason, and so carries its own message: the address step 2
    /// would resolve is not stale but doomed, and the next commit to reuse the
    /// span leaves a reference that dereferenced cleanly reading whatever
    /// landed there. The test is by **prefix**, because a deletion takes the
    /// whole subtree with it (`collect_free_spans` walks it): a reference to a
    /// child of a deleted group dangles exactly as a reference to the group
    /// itself does (issue #314).
    ///
    /// It is a conservative test in the same sense `write_targets` is. A child
    /// whose object survives the delete through another hard link keeps its
    /// address — `collect_free_spans` reclaims nothing when the incoming count
    /// is not 1 — and is refused here anyway, so the message states what the
    /// commit does rather than what the allocator will conclude.
    ///
    /// The delete test runs *after* the other three so that a replacement this
    /// commit has merely not placed yet — a sibling group later in the same
    /// depth band — is reported as an ordering problem rather than as a
    /// deletion. That ordering is a better default, not a partition: because
    /// `add_targets` claims a replaced path's whole subtree by prefix, a child
    /// the replacement does *not* recreate is genuinely doomed and still
    /// reports "still writing". Distinguishing it would mean enumerating what
    /// a replacement actually rebuilds, which neither list holds. A path the
    /// commit puts back is resolved by step 1 before either test is reached.
    fn resolve_reference_target(
        target: &ObjectRefTarget,
        path_addr: &BTreeMap<PathKey, u64>,
        nodes: &BTreeMap<PathKey, Node>,
        add_targets: &[PathKey],
        write_targets: &[PathKey],
        delete_targets: &[PathKey],
        reclaimed: &ReclaimedSpace,
        src: &(impl Source + ?Sized),
        superblock: &Superblock,
    ) -> Result<u64, Error> {
        let path = match target {
            // An address carries no name to test, so the delete check the path
            // arm reaches by prefix is made here on the address itself: whoever
            // staged this resolved it against the pre-commit file, and this
            // commit may be about to take that object away (issue #317).
            ObjectRefTarget::Raw(addr) => {
                if reclaimed.covers(*addr) {
                    return Err(Error::EditUnsupported(REFERENCE_INTO_RECLAIMED_SPACE));
                }
                return Ok(*addr);
            }
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
        {
            return Err(Error::EditUnsupported(
                "an object-reference dataset targets a path this commit is still writing; \
                 use separate commits",
            ));
        }
        // After the three above; see this function's doc for why, and for what
        // that ordering does and does not buy.
        if delete_targets.iter().any(|d| is_prefix(d, &key)) {
            return Err(Error::EditUnsupported(
                "an object-reference dataset targets an object this commit deletes, or one \
                 under it; the reference would be left pointing at storage the delete can \
                 reclaim",
            ));
        }
        match crate::group_v2::resolve_path_any_from_source(src, superblock, path) {
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
    /// apply loop's first `place`/`write_at`).
    fn preflight_reference_targets(
        keys: &[PathKey],
        flat: &BTreeMap<&PathKey, Vec<&FlatDataset>>,
        nodes: &BTreeMap<PathKey, Node>,
        add_targets: &[PathKey],
        write_targets: &[PathKey],
        delete_targets: &[PathKey],
        reclaimed: &ReclaimedSpace,
        src: &(impl Source + ?Sized),
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
                let mut ordered: Vec<&FlatDataset> = datasets.to_vec();
                ordered.sort_by_key(|fd| fd.reference_targets.is_some());
                for fd in ordered {
                    if let Some(patches) = &fd.reference_targets {
                        for patch in patches {
                            Self::resolve_reference_target(
                                &patch.target,
                                &sim_addr,
                                nodes,
                                add_targets,
                                write_targets,
                                delete_targets,
                                reclaimed,
                                src,
                                superblock,
                            )?;
                        }
                    }
                    // References a builder already resolved to addresses never
                    // reach `resolve_reference_target`, so they are screened out
                    // of the element bytes instead. This is the only pass over
                    // them: the apply loop patches unresolved slots and re-reads
                    // none of what it did not write.
                    screen_resolved_references(&fd.dt, &fd.raw, reclaimed)?;
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
    /// The chunk data and index (fixed-array / extensible-array, with any filter
    /// pipeline applied) are produced as one relocatable blob, whose internal
    /// layout — and therefore total size — is independent of the base address it
    /// is given. That is what lets the dataset be *sized before it is placed*
    /// ([`chunked_data_len`]): the blob goes into a freed region big enough to
    /// hold it, or at end-of-file when there is none, and is then assembled for
    /// whichever address it got, so every absolute address it embeds (chunk
    /// addresses, index-structure addresses, the addresses in the data-layout
    /// message) lands exactly where the bytes are written. The header is then
    /// built with [`build_chunked_dataset_oh`] — the same function the whole-file
    /// writer uses — so the header is byte-identical to one written fresh.
    ///
    /// The dataset's single pass of the filter pipeline happens in
    /// [`compress_chunks`], before either decision; sizing and assembly work from
    /// the compressed set, so choosing an address never costs a recompression
    /// (issue #261).
    fn build_chunked_dataset(&mut self, fd: &FlatDataset) -> Result<Vec<u8>, Error> {
        let chunk_dims = fd.chunk_options.resolve_chunk_dims(&fd.ds.dimensions);
        let ctx = ChunkContext::from_datatype(&chunk_dims, &fd.dt)?;
        // The overhang of a partial edge chunk holds the staged fill value
        // (issue #296).
        let elem = crate::convert::nonzero_usize_from(ctx.element_size)?;
        let fill = crate::fill_value::FillPattern::new(fd.fill.as_deref(), elem);
        let set = compress_chunks(
            &fd.raw,
            &fd.ds.dimensions,
            ctx,
            &fd.chunk_options,
            fd.maxshape.as_deref(),
            fill,
        )?;
        // Chunk data and the index beside it are raw (see `chunked_storage_spans`,
        // which reclaims both as raw).
        let (_addr, (layout_message, pipeline_message)) =
            self.place_relocatable(chunked_data_len(&set)?, PageType::Raw, |stored_base| {
                let result = assemble_chunked_at(&set, stored_base)?;
                Ok((
                    result.data_bytes,
                    (result.layout_message, result.pipeline_message),
                ))
            })?;
        Ok(build_chunked_dataset_oh(
            &fd.dt,
            &DatatypeLocation::Inline,
            &fd.ds,
            &layout_message,
            pipeline_message.as_deref(),
            &fd.attrs,
            None,
            fd.fill.as_deref(),
        )?)
    }

    /// On-disk byte spans `(addr, len)` of every chunk of the version 2 object
    /// header at `addr`: chunk 0 (signature, prefix, messages, checksum) plus
    /// each continuation (`OCHK`) block. Used to reclaim a header's storage when
    /// its object is deleted. An error (propagated from [`oh_region_at`] or a
    /// malformed continuation) means the header is not a plain v2 header this
    /// engine can fully account for, and the caller leaves it as dead bytes
    /// rather than guess its extent.
    fn oh_chunk_spans(&self, addr: usize) -> Result<Vec<(u64, u64)>, Error> {
        Ok(
            read_oh_chunks(&self.image(), addr as u64, self.superblock.base_address)?
                .into_iter()
                .map(|chunk| chunk.span)
                .collect(),
        )
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
            let header =
                ObjectHeader::parse_from_source(&self.image(), off as u64, os, ls, base).ok()?;
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
            let entries =
                resolve_group_entries_from_source(&self.image(), &header, os, ls, base).ok()?;
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
        out: &mut Vec<(u64, u64, PageType)>,
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
        let file_len = self.image().len();
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
        match Self::read_object(&self.image(), addr as u64, self.superblock.base_address) {
            Ok(ObjModel::DatasetVerbatim { .. }) => out.extend(meta_spans(spans)),
            Ok(ObjModel::DatasetContiguous {
                data_addr,
                data_size,
                ..
            }) => {
                out.extend(meta_spans(spans));
                // A defined, in-bounds contiguous data block is owned outright;
                // an empty dataset stores the undefined address and owns none. The
                // stored address is base-relative, so shift it to an absolute file
                // offset before bounds-checking and recording it.
                if data_addr != u64::MAX && data_size > 0 {
                    if let (Some(abs), Ok(len)) =
                        (data_addr.checked_add(base), usize::try_from(data_size))
                    {
                        if let Ok(start) = usize::try_from(abs) {
                            if start.checked_add(len).is_some_and(|e| e as u64 <= file_len) {
                                // A contiguous data block is raw data.
                                out.push((abs, data_size, PageType::Raw));
                            }
                        }
                    }
                }
            }
            Ok(ObjModel::Group { children, .. }) => {
                out.extend(meta_spans(spans));
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
                    out.extend(meta_spans(spans));
                    // Already page-typed: chunk data raw, index structure metadata.
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
    fn chunked_storage_spans(&self, addr: usize) -> Option<Vec<(u64, u64, PageType)>> {
        // Locate the data-layout and dataspace messages in the object header.
        let region =
            Self::gather_oh_messages(&self.image(), addr as u64, self.superblock.base_address)
                .ok()?;
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
        let split = crate::chunked_read::collect_chunked_storage_spans(
            &BaseOffsetSource {
                inner: &self.image(),
                base,
            },
            &layout,
            &dataspace,
            OFFSET_SIZE,
            LENGTH_SIZE,
        )
        .ok()?;
        // Chunk data is raw under every writer, so its spans are raw outright. The
        // index is only raw where this crate placed it — see
        // [`index_is_provably_raw`](Self::index_is_provably_raw) — and is left
        // unreclaimed otherwise.
        let mut data: Vec<(u64, u64)> = Vec::with_capacity(split.data.len());
        for (addr, len) in split.data {
            data.push((addr.checked_add(base)?, len));
        }
        let mut index: Vec<(u64, u64)> = Vec::with_capacity(split.index.len());
        for (addr, len) in split.index {
            index.push((addr.checked_add(base)?, len));
        }
        // Validate both halves together — they must be disjoint from each other as
        // well as internally — before either is trusted, including by the proof.
        let mut plain: Vec<(u64, u64)> = data.iter().chain(index.iter()).copied().collect();
        if !spans_disjoint_in_bounds(&mut plain, self.image.len()) {
            return None;
        }
        let data_end = data.iter().filter_map(|&(a, l)| a.checked_add(l)).max();
        let mut spans: Vec<(u64, u64, PageType)> =
            data.iter().map(|&(a, l)| (a, l, PageType::Raw)).collect();
        if self.index_is_provably_raw(data_end, &index) {
            spans.extend(index.into_iter().map(|(a, l)| (a, l, PageType::Raw)));
        }
        Some(spans)
    }

    /// Whether a chunked dataset's `index` provably occupies raw pages, given the
    /// `data` spans it accompanies. Both are absolute file offsets.
    ///
    /// A chunk index is *metadata* by the format's taxonomy, and the reference C
    /// library allocates one as such — out of metadata pages, nowhere near the
    /// chunk data. This crate instead lays a dataset's index down in the same run
    /// as its chunk data, inside raw pages, so that a reader following the layout
    /// message walks one contiguous blob. Both are valid; what matters on a paged
    /// file is that freeing an index records it under the page type it actually
    /// sits in, since that decides what a later allocation may overwrite there.
    ///
    /// Nothing in the file says which writer produced it, so the index is placed
    /// from the layout itself: it must begin exactly where the chunk data ends,
    /// which is this crate's single blob and is not a layout the reference library
    /// produces. A chunk index is metadata to that library, allocated out of
    /// metadata pages — and those are allocated from the bottom of the file, while
    /// large raw data goes above, so its index lands far below the chunk data it
    /// indexes rather than against it. Measured across seven files it wrote
    /// (Extensible and Fixed Array indexes, page sizes 512/4096/8192, 8 to 16
    /// chunks): the index began in page 0 every time, with the chunk data starting
    /// at page 9 or higher, so the join address was never within reach of it.
    ///
    /// An index the test does not place is left unreclaimed by the caller, wasting
    /// its bytes rather than risking advertising space inside a metadata page for
    /// raw reuse (issue #261) — the same trade the reclaim walk already makes for
    /// storage it cannot enumerate exhaustively.
    ///
    /// A non-paged file has no page types to keep apart, so everything is
    /// reclaimable there; the tag is ignored by its commit tail entirely.
    ///
    /// `data_end` is one past the last chunk-data byte, absolute.
    fn index_is_provably_raw(&self, data_end: Option<u64>, index: &[(u64, u64)]) -> bool {
        if self.paged.is_none() || index.is_empty() {
            return true;
        }
        let (Some(data_end), Some(index_start)) = (data_end, index.iter().map(|&(a, _)| a).min())
        else {
            return false;
        };
        index_start == data_end
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
            Self::gather_oh_messages(&self.image(), addr as u64, self.superblock.base_address)
                .ok()?;
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
        let mut spans = chunk_index_spans_from_source(
            &BaseOffsetSource {
                inner: &self.image(),
                base,
            },
            &layout,
            OFFSET_SIZE,
            LENGTH_SIZE,
        )
        .ok()?;
        for (a, _) in &mut spans {
            *a = a.checked_add(base)?;
        }
        if !spans_disjoint_in_bounds(&mut spans, self.image.len()) {
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
    /// Names of links to remove from this group (from `delete`).
    deletes: Vec<String>,
    /// Copies to add to this group: (new link name, the source subtree read out
    /// for writing). Built at staging time from either this file (an in-file
    /// [`copy`](crate::File::copy)) or another open file (a cross-file
    /// [`copy_from`](crate::File::copy_from)).
    copies: Vec<(String, CopyTree)>,
    /// New link names this commit adds to this group by cross-file copy. The
    /// subtrees themselves stay in the staged set — which the preflight may
    /// still refuse, and must not empty (issue #316) — and join `copies` at the
    /// point of no return.
    cross_copies: Vec<String>,
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
    /// before this node's header is built — [`WriteEngine::place_vl_collection`]
    /// appends the collection, then the patched message is appended to
    /// `base_region`.
    pending_vl_attrs: PendingVlAttrs,
}

/// A staged compact attribute edit for a group or dataset (shared by
/// [`Group::set_attr`](crate::Group::set_attr)/`remove_group_attr` and
/// [`Dataset::set_attr`](crate::Dataset::set_attr)/`remove_dataset_attr`).
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
    /// written). The chunk data is not captured here — [`read_copy_subtree`](WriteEngine::read_copy_subtree)
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
/// will write, the read result of [`WriteEngine::read_copy_subtree`] and the
/// input to [`WriteEngine::write_copy_subtree`]. Unlike [`ObjModel`] (a single
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
        /// The dataset's current shape. Held because the index's element
        /// numbering is taken over the *maximum* chunk grid, which needs both
        /// extents (see [`crate::chunk_grid`]).
        shape: Vec<u64>,
        chunk_dims: Vec<u64>,
        element_size: NonZeroUsize,
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

/// The file spans this commit's deletions reclaim, and the base address the
/// references screened against them are stored relative to.
///
/// A deletion frees its object's header and storage, and for a group its whole
/// subtree — [`WriteEngine::collect_free_spans`] is the walk that enumerates
/// it. An object reference *is* an object-header address, so a reference this
/// commit writes that lands inside one of these spans names something the same
/// commit removes: it survives the commit only until the next allocation reuses
/// the span, and then dereferences to whatever landed there (issue #317).
///
/// These are the very spans the commit later hands to the free-space manager,
/// taken from one walk rather than two, so the screen and the reclaimer cannot
/// come to disagree about what this commit removes.
struct ReclaimedSpace {
    /// Absolute `(offset, length)` file spans, as
    /// [`collect_free_spans`](WriteEngine::collect_free_spans) reports them.
    spans: Vec<(u64, u64)>,
    /// The superblock base address. A stored object reference is *base-relative*
    /// and these spans are absolute, so the comparison needs it; it is zero for
    /// every file without a userblock.
    base: u64,
}

impl ReclaimedSpace {
    /// Build the screen from one delete walk's output. The page types the walk
    /// records are for the allocator, not for this.
    fn new(freed: &[(u64, u64, PageType)], base: u64) -> Self {
        Self {
            spans: freed.iter().map(|&(off, len, _)| (off, len)).collect(),
            base,
        }
    }

    /// Whether this commit reclaims anything at all. A commit that deletes
    /// nothing — or whose deletes reclaim nothing, because the objects keep a
    /// surviving hard link — screens no reference, so every caller can skip its
    /// own work when this is true.
    fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Whether `stored`, one object-reference element exactly as it is written
    /// to disk, names an object this commit removes.
    ///
    /// The null (`0`) and undefined ([`UNDEF`]) references name no object at
    /// all, so neither is screened — the same two values
    /// [`crate::repack`](crate::repack) carries through verbatim rather than
    /// resolving.
    fn covers(&self, stored: u64) -> bool {
        if stored == 0 || stored == UNDEF {
            return false;
        }
        let Some(abs) = stored.checked_add(self.base) else {
            return false;
        };
        self.spans
            .iter()
            .any(|&(off, len)| abs >= off && abs - off < len)
    }
}

/// The validated, chunk-collapsed message region and existing link names of a
/// group header.
struct GroupInfo {
    region: Vec<u8>,
    link_names: Vec<String>,
}

/// How a staged value overwrite (`write_dataset`) will be applied, decided by
/// [`WriteEngine::prepare_write`] during the all-or-nothing preflight.
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
    /// chunk-data blob and index are placed — in a freed region that fits, else at
    /// end-of-file (via the verbatim
    /// layout path, carrying `chunk_bytes` and the source filter `pipeline_message`
    /// unchanged — no recompression and no filter-parameter reconstruction), the
    /// data-layout message in the verbatim header `region` is swapped for the new
    /// one (every other header message — datatype, dataspace, fill value, filter
    /// pipeline, and attributes, including a dense attribute heap referenced by an
    /// untouched Attribute Info message — is preserved verbatim), and the old
    /// chunk storage at `old_addr` is freed after the commit lands.
    Chunked {
        region: Vec<u8>,
        /// See [`CopyTree::DatasetChunked::shape`].
        shape: Vec<u64>,
        chunk_dims: Vec<u64>,
        element_size: NonZeroUsize,
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
    /// are placed (reusing freed raw space where it fits), a fresh Extensible
    /// Array is rebuilt over
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
        element_size: NonZeroUsize,
        /// Full (uncompressed) chunk byte size = product(spatial) * element_size.
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
    /// variable-length attribute is placed and patched in [`WriteEngine::write_moving`]
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
    /// [`WriteEngine::build_chunked_dataset`].
    chunk_options: ChunkOptions,
    /// Maximum dimensions for an extensible dataset (an unlimited dimension is
    /// `u64::MAX`), mirrored into `ds.max_dimensions`. `None` for a fixed-shape
    /// dataset. A maxshape with an unlimited dimension selects the
    /// extensible-array chunk index; a finite maxshape stays fixed-array/single.
    maxshape: Option<Vec<u64>>,
    /// Variable-length attributes still carrying a placeholder heap address:
    /// (index into `attrs`, that attribute's global heap collections).
    /// Resolved in the apply loop right before this dataset's header is built.
    vl_attrs: Vec<(usize, Vec<Vec<u8>>)>,
    /// A staged variable-length-string dataset's element references (still
    /// carrying placeholder heap addresses in `raw`) and global heap
    /// collections. Resolved in the apply loop right before `raw` is appended.
    vl_string_staging: Option<VlStringStaging>,
    /// An object-reference dataset's per-element targets, still unresolved.
    /// Resolved (see [`WriteEngine::resolve_reference_target`]) and patched
    /// into `raw` in the apply loop, once every object this commit places has
    /// a known address. `None` for an ordinary dataset.
    reference_targets: Option<Vec<ObjectRefPatch>>,
    /// A user-defined fill value, encoded in the dataset's datatype, or `None`
    /// for the library default. Validated against the datatype element size in
    /// [`flatten_dataset`].
    fill: Option<Vec<u8>>,
}

/// A borrow adapter that drives the shared Extensible-Array append engine
/// ([`crate::chunk_index_inplace`]) against the engine's *own* image and
/// superblock, so a session runs an immediate O(1) in-place append without
/// constructing a second writable handle (which would take a second exclusive
/// lock and keep a divergent view of the file). It borrows only those two
/// fields, leaving [`WriteEngine::located`] independently borrowable.
///
/// [`Store`] is the append engine's view of a file — an image *plus* the
/// superblock, which the image itself knows nothing about. Pairing them here is
/// all this adapter does; every primitive delegates, so the image's own
/// write-ordering discipline is what applies.
///
/// It carries the session's paged-file state too, so `append_raw` keeps a paged
/// file's pages homogeneous through exactly the rule the staged commit uses
/// ([`PagedEdit::begin`]). Before issue #198 there were two copies of that state —
/// one per engine — and the whole-file editor's copy was reachable only from the
/// commit path, so it had to refuse an in-place append to a paged file outright.
struct EditStore<'a> {
    image: &'a mut dyn FileImage,
    superblock: &'a mut Superblock,
    sb_sig_off: usize,
    /// The session's paged state when the file is paged, `None` otherwise. A
    /// borrow rather than a copy: padding recorded here has to reach the manager
    /// rewrite at the next commit or at close.
    paged: Option<&'a mut PagedEdit>,
    /// The session's `fsync` cadence, carried by value: the append engine's own
    /// ordered barriers ([`apply_ea_append`]) are durability points like the
    /// commit's, and answer to the same policy.
    sync_policy: SyncPolicy,
}

impl EditStore<'_> {
    /// Append `bytes` into a raw page, padding the tail page first when a paged
    /// file's tail holds metadata. A plain append on the common non-paged file.
    ///
    /// Raw is the only page type this adapter allocates: see
    /// [`Store::append_raw`](crate::chunk_index_inplace::Store::append_raw) for why
    /// an extensible-array index block belongs in a raw page here.
    fn append_into_raw_page(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        if let Some(pg) = self.paged.as_deref_mut() {
            pg.begin(self.image, PageType::Raw)?;
        }
        self.image.append(bytes)
    }
}

impl crate::source::Source for EditStore<'_> {
    fn len(&self) -> u64 {
        self.image.len()
    }
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), crate::error::FormatError> {
        self.image.read_at(offset, buf)
    }
    fn read_metadata_at(
        &self,
        offset: u64,
        len: usize,
    ) -> Result<Vec<u8>, crate::error::FormatError> {
        self.image.read_metadata_at(offset, len)
    }
}

impl Store for EditStore<'_> {
    fn offset_size(&self) -> u8 {
        self.superblock.offset_size
    }
    fn length_size(&self) -> u8 {
        self.superblock.length_size
    }
    fn append_bytes(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        self.image.append(bytes)
    }
    fn append_raw(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        self.append_into_raw_page(bytes)
    }
    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
        self.image.write_at(offset, bytes)
    }
    fn patch_superblock_eof(&mut self) -> Result<(), Error> {
        // Advance only the recorded end-of-file and re-serialize the superblock in
        // place. Unlike `WriteEngine::commit`, this deliberately does NOT clear the
        // consistency flags and does NOT repoint the root group: base_address is 0
        // for every in-place-append-eligible file, so the normalized-absolute root
        // address serializes back to the same stored value.
        let eof = self.image.len();
        self.superblock.eof_address = eof;
        let bytes = self.superblock.serialize();
        self.write_at(self.sb_sig_off as u64, &bytes)
    }
    fn sync(&mut self) -> Result<(), Error> {
        barrier_data(self.image, self.sync_policy)
    }
}

/// Whether two object paths are equal or one is an ancestor of the other.
fn paths_overlap(a: &[String], b: &[String]) -> bool {
    a.starts_with(b) || b.starts_with(a)
}

/// Whether two appender claims cover the same dataset. A path-less claim (a
/// handle reached by object reference) names its dataset by an address nothing
/// else can compare against, so it conflicts with every other claim.
fn claims_conflict(a: Option<&[String]>, b: Option<&[String]>) -> bool {
    match (a, b) {
        (Some(x), Some(y)) => x == y,
        _ => true,
    }
}

/// Re-tag a refusal from the shared append engine (`AppendUnsupported`) as the
/// fast-path [`Error::AppendInPlaceUnsupported`], so a caller can catch it and fall
/// back to the staged [`append_dataset`](WriteEngine::append_dataset) — which
/// handles the filtered partial-trailing-chunk case, index-geometry limits, and
/// platform-width limits that the engine reports this way. Genuine I/O and format
/// errors pass through unchanged.
pub(crate) fn as_inplace_error(e: Error) -> Error {
    match e {
        Error::AppendUnsupported(m) => Error::AppendInPlaceUnsupported(m),
        other => other,
    }
}

/// Validate a gathered append's bytes against a located dataset: the byte
/// length must be a whole number of elements, and the element datatype must
/// match the on-disk datatype (or, for a raw append, be raw-appendable).
/// Returns the appended element count (`0` = nothing to do). Shared by
/// [`WriteEngine::append_inplace_gathered`]'s path and the bounded backend's immediate
/// append so the acceptance rules stay identical.
pub(crate) fn validate_gathered_append(st: &LocatedState, b: &AppendBuilder) -> Result<u64, Error> {
    let raw = b.raw();
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
    Ok((raw.len() / st.element_size) as u64)
}

/// Locate the dataset at `oh_addr` in `file` and build its [`LocatedState`],
/// validating in-place append eligibility (rank-1 / unlimited / Extensible-Array
/// indexed, a nonzero chunk length, and a re-encodable filter pipeline). Mirrors
/// the append writer's `ensure_located`, reporting through
/// [`Error::AppendInPlaceUnsupported`].
pub(crate) fn locate_dataset_state<F: Store>(
    file: &F,
    oh_addr: u64,
) -> Result<LocatedState, Error> {
    let result = Located::locate_at(file, oh_addr, Error::AppendInPlaceUnsupported)?;
    if result.located.chunk_elems == 0 {
        return Err(Error::AppendInPlaceUnsupported(
            "in-place append requires a nonzero chunk length",
        ));
    }
    let (dt_off, dt_size) = result.spans.datatype;
    let dt_bytes = file
        .read_metadata_at(dt_off, dt_size)
        .map_err(|_| Error::AppendInPlaceUnsupported("dataset datatype could not be parsed"))?;
    let (datatype, _) = Datatype::parse(&dt_bytes)
        .map_err(|_| Error::AppendInPlaceUnsupported("dataset datatype could not be parsed"))?;
    let pipeline = match result.spans.filter {
        Some((fb, fsize)) => {
            let fp_bytes = file.read_metadata_at(fb, fsize).map_err(|_| {
                Error::AppendInPlaceUnsupported("dataset filter pipeline could not be parsed")
            })?;
            let parsed = FilterPipeline::parse(&fp_bytes).map_err(|_| {
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
    // A fill message that cannot be read leaves `PaddingFill::Unknown`, which
    // fails only where a chunk actually needs padding — an append that lands on
    // a chunk boundary asks nothing of it.
    let fill = match result.spans.fill {
        Some((msg_type, off, size)) => match file.read_metadata_at(off, size) {
            Ok(body) => crate::fill_value::PaddingFill::from_message(msg_type, &body),
            Err(_) => crate::fill_value::PaddingFill::Unknown,
        },
        None => crate::fill_value::PaddingFill::Zero,
    };
    Ok(LocatedState {
        loc: result.located,
        datatype,
        spatial,
        element_size,
        pipeline,
        fill,
    })
}

/// Split a path into non-empty components.
fn split_path(path: &str) -> PathKey {
    path.split('/')
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

/// Group `(parent group, item)` pairs by their parent, preserving the input
/// order within each group.
///
/// The one rule by which a commit's staged datasets become per-group batches:
/// the preflight groups borrowed ones to prove its guards, and the apply loop
/// groups the owned ones it places, so the order
/// [`preflight_reference_targets`](WriteEngine::preflight_reference_targets)
/// replays is the order the apply loop uses by construction.
fn group_by_parent<K: Ord, T>(items: impl IntoIterator<Item = (K, T)>) -> BTreeMap<K, Vec<T>> {
    let mut out: BTreeMap<K, Vec<T>> = BTreeMap::new();
    for (parent, item) in items {
        out.entry(parent).or_default().push(item);
    }
    out
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
fn retain_disjoint_in_bounds(spans: &mut Vec<(u64, u64, PageType)>, eof: u64) {
    spans.retain(|&(addr, len, _)| len > 0 && addr.checked_add(len).is_some_and(|e| e <= eof));
    spans.sort_unstable_by_key(|&(addr, _, _)| addr);
    let mut kept_end = 0u64;
    spans.retain(|&(addr, len, _)| {
        if addr >= kept_end {
            kept_end = addr + len;
            true
        } else {
            false // overlaps a span already kept; leak it rather than double-free
        }
    });
}

/// Tag object-header chunk spans as file metadata. Every span
/// [`oh_chunk_spans`](EditSession::oh_chunk_spans) returns is part of an object
/// header, so the page type is the same for all of them.
fn meta_spans(spans: Vec<(u64, u64)>) -> impl Iterator<Item = (u64, u64, PageType)> {
    spans.into_iter().map(|(a, l)| (a, l, PageType::Meta))
}

/// Validate a staged dataset and reduce it to a [`FlatDataset`]. Contiguous,
/// unfiltered datasets are emitted as such; chunked, filtered, or extensible
/// datasets carry their [`ChunkOptions`] and maxshape through to the commit,
/// where [`WriteEngine::build_chunked_dataset`] lays out their chunk data and
/// index. An empty (zero-element) shape is allowed under either storage,
/// mirroring the whole-file writer: a contiguous one takes the `HADDR_UNDEF`
/// data address (see the apply loop) and a chunked one an index over zero
/// chunks, which is what an extensible dataset is created as before the first
/// append fills it. The geometry validation below still requires explicit chunk
/// dimensions for it — auto-chunking has no shape to derive them from. A
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
/// (see [`WriteEngine::resolve_reference_target`]). Rejects any remaining
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
    // Variable-length string element references live in the global heap, whose
    // address is only known once the apply loop places the collection. For
    // chunked/filtered/resizable storage the references sit inside chunks
    // written before that address exists, so patching them in is impossible.
    //
    // The whole-file writer lifted the same restriction by placing such a
    // dataset's collections ahead of everything else (issue #109); this engine
    // appends into an existing layout, where there is no "ahead" to place them
    // in, so the equivalent fix is a separate piece of work.
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

    // Refused for the same reason the whole-file writer refuses it: nothing
    // occupies zero bytes per element, the writers divide by the element size,
    // and a caller-built `Datatype` never passes through `Datatype::parse`.
    // Taking it as a `NonZeroUsize` hands the proof to the staging below rather
    // than leaving each step to re-derive it.
    let elem_size = dt.element_size_usize()?;

    let elem = elem_size.get() as u64;
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
        // ([`compress_chunks`] + [`assemble_chunked_at`] + [`build_chunked_dataset_oh`]),
        // so the
        // resulting object header is byte-identical to a freshly written one.
        //
        // The fill value goes in for the same reason the apply phase passes it:
        // scale-offset records it in the filter's parameters, so a validation
        // that left it out would be checking a pipeline the commit does not
        // build — and would pass a fill value the encoder later refuses.
        let chunk_dims = db.chunk_options.resolve_chunk_dims(&shape);
        let ctx = ChunkContext::from_datatype(&chunk_dims, &dt)?;
        db.chunk_options
            .build_pipeline(
                &ctx,
                crate::fill_value::FillPattern::new(
                    db.fill.as_deref(),
                    crate::convert::nonzero_usize_from(ctx.element_size)?,
                ),
            )
            .map_err(|_| {
                Error::EditUnsupported(
                    "this dataset's filter pipeline cannot be added in place \
                     (an unsupported filter, an incompatible datatype, a fill \
                     value the filter cannot record, or a compression feature \
                     that is not enabled)",
                )
            })?;
    }

    // The link message body (whose length is independent of the address) must
    // fit the object-header message's u16 size field; a pathologically long
    // name would otherwise overflow it into silent corruption.
    if make_link(&db.name, 0).serialize(OFFSET_SIZE).len() > OBJECT_HEADER_MESSAGE_MAX {
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
        attrs.push(v.to_message(n));
    }
    // The message above already carries a placeholder (heap address 0) for each
    // element of a variable-length string attribute; stage its self-contained
    // global heap collections here (no address of their own to resolve yet) and
    // record which `attrs` slot they patch once the apply loop places them.
    let mut vl_attrs: Vec<(usize, Vec<Vec<u8>>)> = Vec::new();
    for (i, (_, v)) in db.attrs.iter().enumerate() {
        if let Some(strings) = v.var_len_strings() {
            let str_refs: Vec<&str> = strings.iter().map(String::as_str).collect();
            vl_attrs.push((i, build_global_heap_collections(&str_refs)));
        }
    }
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
        if a.serialize(LENGTH_SIZE).len() > OBJECT_HEADER_MESSAGE_MAX {
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

    // A committed datatype is an object of its own, which an in-place edit has no
    // way to place: it appends into an existing file rather than laying one out,
    // so there is nothing to resolve the named path against. Writing the type
    // inline instead would produce a dataset that reads correctly but no longer
    // shares the named type, so refuse by name. The whole-file writer places
    // them; [`crate::repack`] is the route from an edited file to one.
    if db.datatype_location.is_committed()
        || attrs.iter().any(|a| a.datatype_location.is_committed())
    {
        return Err(Error::EditUnsupported(
            "a dataset or attribute naming a committed (shared) datatype cannot be added in \
             place; write the file with FileBuilder instead",
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
/// rather than letting `compress_chunk` surface a raw `UnsupportedFilter`.
pub(crate) fn pipeline_reencodable(pipeline: &FilterPipeline) -> bool {
    pipeline.filters.iter().all(|f| match f.filter_id {
        FILTER_DEFLATE | FILTER_SHUFFLE | FILTER_FLETCHER32 | FILTER_SCALEOFFSET | FILTER_LZF => {
            true
        }
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
    /// Element size in bytes, proven non-zero: the chunk splitter divides by
    /// it, and so does the append path's element-count arithmetic.
    element_size: NonZeroUsize,
    /// Full (uncompressed) chunk byte size, `product(spatial) * element_size`.
    /// This is what the chunk index's element width is derived from, so it has
    /// to be the geometry's product and not any written chunk's size.
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
    let element_size = dt.element_size_usize()?;
    let raw_size = spatial
        .iter()
        .copied()
        .product::<u64>()
        .saturating_mul(element_size.get() as u64);
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
fn try_inplace_chunk_writes<S: Source + ?Sized>(
    src: &S,
    layout: &DataLayout,
    ds: &Dataspace,
    spatial: &[u64],
    raw_size: u64,
    new_bytes: &[Vec<u8>],
) -> Option<Vec<(usize, Vec<u8>)>> {
    let infos = enumerate_chunks_from_source(src, layout, ds, OFFSET_SIZE, LENGTH_SIZE).ok()?;
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
        start
            .checked_add(bytes.len())
            .filter(|&e| e as u64 <= src.len())?;
        writes.push((start, bytes.clone()));
        spans.push((ci.address, new_len));
    }

    // A shrinking overwrite changes the index-recorded chunk sizes, so the index
    // must be rebuilt in place to match; an equal-size one leaves it untouched.
    if any_shrunk {
        let (index_addr, index_bytes) = try_rebuild_index_in_place(
            src,
            layout,
            ds,
            spatial,
            raw_size,
            &grid.grid_order,
            new_bytes,
        )?;
        spans.push((index_addr as u64, index_bytes.len() as u64));
        writes.push((index_addr, index_bytes));
    }

    // Refuse to perform overlapping in-place writes (a malformed source index, or
    // an index region that overlaps a chunk slot); relocate instead so two writes
    // never clobber each other.
    if !spans_disjoint_in_bounds(&mut spans, src.len()) {
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
#[allow(clippy::too_many_arguments)]
fn try_rebuild_index_in_place<S: Source + ?Sized>(
    src: &S,
    layout: &DataLayout,
    ds: &Dataspace,
    spatial: &[u64],
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
            filter_mask: 0,
        })
        .collect();
    // `grid_order` is dense row-major over the *current* shape; where each of
    // those chunks sits in the index is the maximum grid's business, and the
    // rebuild has to reproduce the numbering the original index was written
    // with. Same rule, same function as the writer.
    let (_, slot_of_chunk, index_slots) = crate::chunked_write::plan_index_slots(
        &ds.dimensions,
        spatial,
        ds.max_dimensions.as_deref(),
        raw_size,
        true,
    )
    .ok()?;
    let slots =
        crate::chunked_write::IndexSlots::new(&written, &slot_of_chunk, index_slots).ok()?;
    let new_index = match (version, chunk_index_type) {
        // `raw_size` is the whole-chunk byte size, which is what the element
        // width derives from — the same value the original index was built with,
        // so the rebuilt structure matches its length. (An index written by a
        // version that derived the width from the written chunks instead can
        // disagree; the length check below then rejects it and the caller
        // relocates, which is the safe direction.)
        (4, Some(3)) => crate::chunked_write::build_fixed_array_at(
            &slots,
            raw_size,
            OFFSET_SIZE,
            LENGTH_SIZE,
            true,
            *index_addr,
        ),
        (4, Some(4)) => crate::chunked_write::build_extensible_array_at(
            &slots,
            raw_size,
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
        crate::chunked_read::chunk_index_spans_from_source(src, layout, OFFSET_SIZE, LENGTH_SIZE)
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
        .filter(|&e| e as u64 <= src.len())?;
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
    fn chunk_bytes(&self, index: usize, out: &mut Vec<u8>) -> Result<(), FormatError> {
        let chunk = self.chunks.get(index).ok_or_else(|| {
            FormatError::ChunkedReadError("chunk index out of range for in-memory provider".into())
        })?;
        out.extend_from_slice(chunk);
        Ok(())
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

/// Ensure a chunk-0 message `region` that carries inline Attribute messages also
/// carries an Attribute Info message, appending the compact-storage one when
/// absent.
///
/// On a version 2 object header the reference C library never counts attribute
/// messages: `H5O__attr_count_real` reads the count out of the Attribute Info
/// message, and reports zero when there is none, even though `H5Aiterate` and
/// `H5Aopen_by_name` still find every attribute. Tools that size their work by
/// that count then skip the attributes silently — `h5repack` copies such an
/// object with none of them. Releases through 0.33.0 wrote every compact
/// attribute set that way, so heal any such header whenever we rewrite one, the
/// same reason [`ensure_group_info`] exists.
///
/// A region already carrying an Attribute Info message is left alone, whichever
/// storage it names: a defined heap address means dense storage, whose count the
/// C library takes from the heap's B-tree instead.
fn ensure_attribute_info(region: &mut Vec<u8>) -> Result<(), Error> {
    let mut has_attrs = false;
    let mut p = 0;
    while let Some((msg_type, _body, body_end)) = next_message(region, p)? {
        match msg_type {
            MessageType::AttributeInfo => return Ok(()),
            MessageType::Attribute => has_attrs = true,
            _ => {}
        }
        p = body_end;
    }
    if has_attrs {
        region.extend_from_slice(&region_message(
            MessageType::AttributeInfo,
            &crate::file_writer::compact_attribute_info_message(),
        ));
    }
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

/// Bytes a compact Data Layout message carries ahead of its inline data:
/// version(1) + class(1) + the 2-byte inline size.
const COMPACT_LAYOUT_PREAMBLE: usize = 4;

/// Copy a chunk-0 message `region`, replacing the single (compact) Data Layout
/// message's inline data with `raw` and preserving every other message verbatim.
/// Used by `write_dataset` to overwrite a compact dataset's values. The message
/// header (type and flags) and version byte are kept; only the inline data — and
/// the message size and 2-byte inline-size fields — change. `raw` must fit both
/// the compact layout's own 2-byte size field (HDF5's 64 KiB compact-storage
/// limit) and, once the 4-byte layout preamble is added, the object header's
/// 2-byte message-size field — the tighter of the two, which an overwrite of an
/// existing compact dataset always satisfies.
fn rebuild_compact_layout_region(region: &[u8], raw: &[u8]) -> Result<Vec<u8>, Error> {
    // The bound is on the *message body* the layout becomes — version, class,
    // and the 2-byte inline size ahead of the data — not on `raw` alone, or the
    // last four lengths below the limit would truncate the size field written
    // for them.
    if raw.len() > OBJECT_HEADER_MESSAGE_MAX - COMPACT_LAYOUT_PREAMBLE {
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
            let mut layout = Vec::with_capacity(COMPACT_LAYOUT_PREAMBLE + raw.len());
            layout.push(region[body]); // version (3 or 4)
            layout.push(0); // class = compact
            #[expect(
                clippy::cast_possible_truncation,
                reason = "raw.len() bounded below the u16 inline-size field above"
            )]
            layout.extend_from_slice(&(raw.len() as u16).to_le_bytes());
            layout.extend_from_slice(raw);
            // Message record: type byte, 2-byte size (LE), flags byte (kept).
            out.push(region[p]);
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the guard above bounds COMPACT_LAYOUT_PREAMBLE + raw.len(), this \
                          body's exact length, to the 2-byte message-size field"
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
/// is known (see [`WriteEngine::place_vl_collection`]). A later op for the
/// same name (another `Set`, fixed-size or not, or a `Remove`) replaces or
/// cancels an earlier still-pending variable-length entry, keeping the net
/// effect the same regardless of op order within one commit. `region`'s
/// fixed-size portion is a complete compact-attribute header on return; dense
/// attribute storage and shared attribute messages are refused.
fn apply_group_attr_ops(
    region: &[u8],
    ops: &[&AttrOp],
) -> Result<(Vec<u8>, PendingVlAttrs), Error> {
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
                    if msg.serialize(LENGTH_SIZE).len() > OBJECT_HEADER_MESSAGE_MAX {
                        return Err(Error::EditUnsupported(
                            "attribute is too large to encode in place",
                        ));
                    }
                    let str_refs: Vec<&str> = strings.iter().map(String::as_str).collect();
                    pending_vl.push((msg, build_global_heap_collections(&str_refs)));
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
    // Only the name is wanted here, and it is the one field that never depends on
    // the datatype or dataspace — either of which may be a reference to a
    // committed message this walk has no file context to follow. Reading the name
    // alone lets an edit pass over such an attribute instead of refusing the
    // whole object because one of its neighbours is committed.
    crate::attribute::message_name(&region[body..body_end])
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
    if body.len() > OBJECT_HEADER_MESSAGE_MAX {
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
/// Rebuild a superblock-extension object header's message region (as collapsed by
/// [`WriteEngine::gather_oh_messages`]) with
/// its File Space Info message replaced by `info`, preserving every other message
/// verbatim. The persisting message is fixed-size, so the region length is stable.
/// Shared by the whole-file mirror commit and the bounded finalize so both write
/// the same extension bytes.
pub(crate) fn rewrite_extension_region_bytes(
    region: &[u8],
    info: &FileSpaceInfo,
) -> Result<Vec<u8>, Error> {
    let new_body = info.serialize();
    // The message body is the fixed-size File Space Info record (≤ 125 bytes),
    // so it always fits the u16 size field; `try_from` keeps this off the
    // 32-bit narrowing-cast ledger.
    let new_len = u16::try_from(new_body.len())
        .map_err(|_| Error::EditUnsupported("File Space Info message too large"))?;
    let mut out = Vec::with_capacity(region.len());
    let mut p = 0;
    let mut replaced = false;
    while let Some((msg_type, _body, body_end)) = next_message(region, p)? {
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

/// Parse and validate a version 2 object header's prefix, returning the absolute
/// `[start, end)` byte range of its chunk-0 message region.
///
/// `prefix` holds the bytes at `[addr, addr + prefix.len())` — up to
/// [`OH_PREFIX_MAX`], fewer when the header sits near the end of the image — and
/// `file_len` is the length of the image the header lives in, which bounds the
/// region. Rejects headers that are not OHDR v2 and headers that track message
/// creation order, whose 6-byte message records this engine does not emit.
fn oh_region_at(prefix: &[u8], addr: u64, file_len: u64) -> Result<(u64, u64), Error> {
    if prefix.len() < 6 || &prefix[..4] != b"OHDR" || prefix[4] != 2 {
        return Err(Error::EditUnsupported(
            "an object does not use a version 2 object header",
        ));
    }
    let flags = prefix[5];
    if flags & 0x04 != 0 {
        return Err(Error::EditUnsupported(
            "an object tracks message creation order (not supported in place yet)",
        ));
    }
    let mut pos = 6usize;
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
    if prefix.len() < pos + size_width {
        return Err(Error::EditUnsupported("truncated object header"));
    }
    let chunk0_size = read_le(&prefix[pos..pos + size_width]) as u64;
    pos += size_width;
    let region_start = addr
        .checked_add(pos as u64)
        .ok_or(Error::EditUnsupported("truncated object header"))?;
    // The region is followed by a 4-byte checksum, which must also be present.
    let region_end = region_start
        .checked_add(chunk0_size)
        .filter(|e| e.checked_add(4).is_some_and(|end| end <= file_len))
        .ok_or(Error::EditUnsupported("truncated object header"))?;
    Ok((region_start, region_end))
}

/// One chunk of a version 2 object header, read out of a file image.
///
/// The buffer stops at the end of the chunk's message region: the trailing
/// checksum is never walked, and it has already been confirmed present. `span`
/// covers the *whole* on-disk chunk including that checksum, so it can be handed
/// to the free list when the header is reclaimed.
struct OhChunk {
    /// Absolute file address and full on-disk length of the chunk.
    span: (u64, u64),
    /// The chunk's bytes, from `span.0` through the end of its message region.
    buf: Vec<u8>,
    /// Offset of the first message within [`buf`](Self::buf).
    messages_start: usize,
}

impl OhChunk {
    /// The slice to walk messages in, and the offset to start at. The two are
    /// returned together because [`next_message`] must not read past the end of
    /// the message region into the checksum.
    fn message_region(&self) -> (&[u8], usize) {
        (&self.buf, self.messages_start)
    }
}

/// Read chunk 0 of the version 2 object header at `addr` out of `src`.
fn read_oh_chunk0<S: Source + ?Sized>(src: &S, addr: u64) -> Result<OhChunk, Error> {
    let file_len = src.len();
    let window = file_len
        .saturating_sub(addr)
        .min(OH_PREFIX_MAX as u64)
        .to_usize()?;
    let prefix = src.read_metadata_at(addr, window)?;
    let (rs, re) = oh_region_at(&prefix, addr, file_len)?;
    // `re >= rs > addr`, so both differences are non-negative, and `oh_region_at`
    // has checked that the 4-byte checksum past `re` is present.
    let len = (re - addr).to_usize()?;
    Ok(OhChunk {
        span: (addr, len as u64 + 4),
        buf: src.read_metadata_at(addr, len)?,
        messages_start: (rs - addr).to_usize()?,
    })
}

/// Read every chunk of the version 2 object header at `addr`, chunk 0 first,
/// following each `Continuation` message to its `OCHK` block.
///
/// This is the one traversal of a header's chunk chain: [`gather_oh_messages`]
/// collects the messages out of the result and
/// [`oh_chunk_spans`](WriteEngine::oh_chunk_spans) collects the extents, so the
/// two cannot disagree about what a header occupies.
fn read_oh_chunks<S: Source + ?Sized>(
    src: &S,
    addr: u64,
    base: u64,
) -> Result<Vec<OhChunk>, Error> {
    let mut chunks = vec![read_oh_chunk0(src, addr)?];
    let mut i = 0;
    while i < chunks.len() {
        if chunks.len() > MAX_OH_CHUNKS {
            return Err(Error::EditUnsupported(
                "object header has too many continuation chunks",
            ));
        }
        // Collect this chunk's continuations before extending the worklist, so the
        // borrow of `chunks[i]` ends first.
        let mut found = Vec::new();
        let (region, mut p) = chunks[i].message_region();
        while let Some((msg_type, body, body_end)) = next_message(region, p)? {
            if msg_type == MessageType::ObjectHeaderContinuation {
                found.push(read_oh_continuation(src, region, body, body_end, base)?);
            }
            p = body_end;
        }
        i += 1;
        chunks.extend(found);
    }
    Ok(chunks)
}

/// Read the `OCHK` continuation block a continuation message points at.
///
/// `region[body..body_end]` is the continuation message's body: the block's
/// base-relative address followed by its length.
fn read_oh_continuation<S: Source + ?Sized>(
    src: &S,
    region: &[u8],
    body: usize,
    body_end: usize,
    base: u64,
) -> Result<OhChunk, Error> {
    if body_end - body < (OFFSET_SIZE + LENGTH_SIZE) as usize {
        return Err(Error::EditUnsupported("malformed continuation message"));
    }
    let off = u64::from_le_bytes(region[body..body + 8].try_into().unwrap());
    let len = u64::from_le_bytes(region[body + 8..body + 16].try_into().unwrap());
    // The block address is stored relative to the base address; shift it to an
    // absolute file offset before reading.
    let off = off
        .checked_add(base)
        .ok_or(Error::EditUnsupported("continuation address overflow"))?;
    // An OCHK block is signature(4) + messages + checksum(4).
    let end = off
        .checked_add(len)
        .filter(|&e| e <= src.len() && len >= 8)
        .ok_or(Error::EditUnsupported("continuation block out of bounds"))?;
    let want = (end - off)
        .to_usize()
        .map_err(|_| Error::EditUnsupported("continuation length exceeds this platform"))?;
    let mut buf = src.read_metadata_at(off, want)?;
    if buf[..4] != *b"OCHK" {
        return Err(Error::EditUnsupported(
            "invalid continuation block signature",
        ));
    }
    // Trim the trailing checksum so the message walk stops at the last message.
    buf.truncate(want - 4);
    Ok(OhChunk {
        span: (off, len),
        buf,
        messages_start: 4,
    })
}

pub(crate) fn next_message(
    region: &[u8],
    p: usize,
) -> Result<Option<(MessageType, usize, usize)>, Error> {
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
/// only on the cross-file path; the same-file [`copy`](crate::File::copy)
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
                // An attribute's *own* datatype or dataspace field can be a
                // reference to a committed message, which the record's shared
                // flag above does not report: that flag describes the attribute
                // message, not the fields inside it. The reference addresses the
                // source file, so it cannot travel any more than a shared record
                // can — and the parse below cannot resolve it here in any case.
                if crate::attribute::message_shares_a_field(&region[body..body_end]) {
                    return Err(Error::EditUnsupported(
                        "an attribute with a committed (shared) datatype cannot be copied to another file yet",
                    ));
                }
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

/// Whether `dt` reaches an 8-byte **object reference** anywhere in its structure,
/// directly or through a compound member, array entry, or enumeration base.
///
/// The narrow sibling of [`datatype_copies_foreign_address`], which asks the
/// wider question the cross-file copy path needs. Variable-length data is
/// deliberately *not* included here: a deletion frees object headers and dataset
/// storage ([`WriteEngine::collect_free_spans`]) and never a global heap
/// collection, so variable-length elements keep pointing at data that is still
/// there.
fn datatype_holds_object_reference(dt: &Datatype) -> bool {
    use crate::datatype::ReferenceType;
    match dt {
        Datatype::Reference {
            ref_type: ReferenceType::Object,
            ..
        } => true,
        Datatype::Compound { members, .. } => members
            .iter()
            .any(|m| datatype_holds_object_reference(&m.datatype)),
        Datatype::Array { base_type, .. } | Datatype::Enumeration { base_type, .. } => {
            datatype_holds_object_reference(base_type)
        }
        _ => false,
    }
}

/// Refuse a staged dataset whose element bytes already hold *resolved* object
/// references naming space this commit reclaims (issue #317).
///
/// This is the address-side half of the rule
/// [`WriteEngine::resolve_reference_target`] enforces on the path side. A target
/// named as a path is screened there, by name; a target supplied as an address —
/// `DatasetBuilder::with_reference_data`, or `with_raw_data` over a datatype that
/// holds a reference — never reaches that function at all, and before this screen
/// existed it was written straight through to disk.
///
/// Element bytes still carrying placeholders are unaffected: an unresolved slot
/// holds zero, which [`ReclaimedSpace::covers`] never screens, and the address
/// that replaces it is screened by `resolve_reference_target` when it resolves.
/// So this runs over every staged dataset's `raw` without asking which builder
/// filled it — the rule is about the bytes, not the door they came through.
fn screen_resolved_references(
    dt: &Datatype,
    raw: &[u8],
    reclaimed: &ReclaimedSpace,
) -> Result<(), Error> {
    if reclaimed.is_empty() || !datatype_holds_object_reference(dt) {
        return Ok(());
    }
    // A datatype that declares a reference but whose slots do not fit its own
    // element size cannot be walked, so its addresses cannot be screened. That
    // is a datatype this crate would not itself build; refuse rather than write
    // references past a screen that could not read them.
    let Some(slots) = embedded_reference_slots(dt) else {
        return Err(Error::EditUnsupported(
            "a staged dataset's object-reference slots do not fit its datatype, so they \
             cannot be screened against this commit's deletions",
        ));
    };
    if slots.is_empty() {
        return Ok(());
    }
    // Non-empty slots mean the datatype has room for at least one 8-byte
    // address, so the element size is at least 8 and `chunks_exact` is safe.
    let element_size = dt.type_size() as usize;
    for element in raw.chunks_exact(element_size) {
        for &at in &slots {
            let stored =
                u64::from_le_bytes(element[at..at + 8].try_into().expect(
                    "embedded_reference_slots keeps every slot 8 bytes inside the element",
                ));
            if reclaimed.covers(stored) {
                return Err(Error::EditUnsupported(REFERENCE_INTO_RECLAIMED_SPACE));
            }
        }
    }
    Ok(())
}

/// Screen an in-file copy's subtree against the space this commit reclaims
/// (issue #317).
///
/// An in-file copy re-emits its source's element bytes verbatim, which is what
/// keeps a copied variable-length or reference dataset valid: the addresses
/// still name the same file, so nothing has to be rewritten. A deletion in the
/// same commit takes that away for object references alone — a global heap
/// collection is never reclaimed by a delete, so variable-length elements keep
/// pointing at data that is still there.
///
/// Every element that can be read is screened by address, through the same
/// [`screen_resolved_references`] a staged dataset's bytes go through: a
/// contiguous dataset's data block, a compact dataset's inline data, and every
/// attribute, inline or dense. A committed (shared) datatype is resolved through
/// `src` first, so a copy of an object with a named type is screened like any
/// other rather than refused for carrying a type this could not read.
///
/// One form cannot be read at all and so is refused by *datatype*, and only when
/// it holds an object reference: a **chunked** dataset, whose addresses sit
/// inside chunks this path carries compressed and never decodes — the same
/// obstacle that makes [`crate::repack`] refuse a chunked object-reference
/// dataset outright.
///
/// `src` is the session's image framed at its base address, the view a stored
/// (base-relative) shared-message address indexes directly.
fn screen_copied_references(
    tree: &CopyTree,
    reclaimed: &ReclaimedSpace,
    src: &(impl Source + ?Sized),
) -> Result<(), Error> {
    if reclaimed.is_empty() {
        return Ok(());
    }
    use crate::shared_message::SharedResolver as _;
    let resolver = crate::shared_message::SourceResolver::new(src, OFFSET_SIZE, LENGTH_SIZE);
    let (region, dense_attrs) = match tree {
        CopyTree::DatasetVerbatim {
            region,
            dense_attrs,
        }
        | CopyTree::DatasetContiguous {
            region,
            dense_attrs,
            ..
        }
        | CopyTree::DatasetChunked {
            region,
            dense_attrs,
            ..
        }
        | CopyTree::Group {
            non_link_region: region,
            dense_attrs,
            ..
        } => (region, dense_attrs),
    };
    for attr in dense_attrs {
        screen_resolved_references(&attr.datatype, &attr.raw_data, reclaimed)?;
    }

    // One walk of the header: the object's own element datatype, the compact
    // data the layout message may carry, and every inline attribute.
    let mut element_dt: Option<Datatype> = None;
    let mut compact: Option<Vec<u8>> = None;
    let mut p = 0;
    while let Some((msg_type, body, body_end)) = next_message(region, p)? {
        // The flags byte is the 4th of the record header (type, size, flags);
        // `next_message` returning `Some` guarantees it is in bounds.
        let shared = region[p + 3] & MSG_FLAG_SHARED != 0;
        match msg_type {
            MessageType::Datatype => {
                // A committed datatype's message body is a pointer into the
                // file's shared-message storage rather than an encoded type, so
                // read the type it names before parsing.
                let committed;
                let encoded = if shared {
                    committed = resolver
                        .resolve(&region[body..body_end], MessageType::Datatype)
                        .map_err(|_| {
                            Error::EditUnsupported(
                                "a copy in this commit names a committed (shared) datatype that \
                                 could not be read, so its elements cannot be screened against \
                                 the same commit's deletions; use separate commits",
                            )
                        })?;
                    &committed[..]
                } else {
                    &region[body..body_end]
                };
                let (dt, _) = Datatype::parse(encoded).map_err(|_| {
                    Error::EditUnsupported("a source datatype could not be parsed for copying")
                })?;
                element_dt = Some(dt);
            }
            MessageType::DataLayout => {
                if let Ok(DataLayout::Compact { data }) =
                    DataLayout::parse(&region[body..body_end], OFFSET_SIZE, LENGTH_SIZE)
                {
                    compact = Some(data);
                }
            }
            MessageType::Attribute => {
                // A *shared record* is the whole attribute message held in the
                // file's shared-message table, which is a different indirection
                // from the committed datatype `parse_resolving` follows inside
                // the fields — and a rare one this path has never modelled. Its
                // elements cannot be reached here, so it is refused.
                if shared {
                    return Err(Error::EditUnsupported(
                        "a copy in this commit carries a shared (SOHM) attribute message, whose \
                         elements cannot be screened against the same commit's deletions; use \
                         separate commits",
                    ));
                }
                // `parse_resolving` rather than `parse`: an attribute's own
                // datatype field can name a committed message, which the record's
                // shared flag does not report — that flag describes the attribute
                // message, not the fields inside it — and `parse` refuses one.
                let attr = crate::attribute::AttributeMessage::parse_resolving(
                    &region[body..body_end],
                    LENGTH_SIZE,
                    &resolver,
                )
                .map_err(|_| {
                    Error::EditUnsupported("a source attribute could not be parsed for copying")
                })?;
                screen_resolved_references(&attr.datatype, &attr.raw_data, reclaimed)?;
            }
            _ => {}
        }
        p = body_end;
    }

    match tree {
        // Compact: the elements are inline in the data-layout message. A layout
        // that did not yield them leaves a reference datatype unscreened, so it
        // is refused for the same reason a chunked one is.
        CopyTree::DatasetVerbatim { .. } => match (&element_dt, &compact) {
            (Some(dt), Some(data)) => screen_resolved_references(dt, data, reclaimed)?,
            (Some(dt), None) if datatype_holds_object_reference(dt) => {
                return Err(Error::EditUnsupported(
                    "a compact object-reference dataset's elements could not be read to screen \
                     them against this commit's deletions; use separate commits",
                ));
            }
            _ => {}
        },
        CopyTree::DatasetContiguous { data, .. } => {
            if let Some(dt) = &element_dt {
                screen_resolved_references(dt, data, reclaimed)?;
            }
        }
        // Chunked: `chunk_bytes` are carried exactly as the source stored them,
        // filters and all, so there is nothing here to decode addresses out of.
        CopyTree::DatasetChunked { .. } => {
            if element_dt
                .as_ref()
                .is_some_and(datatype_holds_object_reference)
            {
                return Err(Error::EditUnsupported(
                    "a chunked object-reference dataset cannot be copied in a commit that also \
                     deletes objects: its addresses live inside chunks this path does not \
                     decode; use separate commits",
                ));
            }
        }
        CopyTree::Group { children, .. } => {
            for (_, child) in children {
                screen_copied_references(child, reclaimed, src)?;
            }
        }
    }
    Ok(())
}

/// Wrap a chunk-0 message region in a fresh single-chunk version 2 object header
/// (`OHDR` prefix + region + Jenkins checksum), first normalizing the region's
/// attribute storage with [`ensure_attribute_info`]. Mirrors the encoding in
/// [`crate::object_header_writer::ObjectHeaderWriter::serialize`].
///
/// The normalization belongs here rather than at the fifteen call sites because
/// carrying an Attribute Info message is a property of a version 2 header holding
/// inline attributes, not of any one edit operation — and a site that forgot it
/// would reintroduce the zero-count defect silently.
///
/// A region that cannot be walked is reported, not asserted away. Every region
/// reaching here should have been built by this crate or already walked
/// message-by-message on the way in, but "should" is a claim about a file this
/// session did not write: a header whose message size field overruns the region
/// is a malformed *file*, which is the caller's input and so takes an
/// [`Error::EditUnsupported`], the way every other malformed-header path in this
/// module does. The `debug_assert!(false)` this replaced made the two build
/// profiles disagree about whether such a file was writable at all — a panic in
/// a test build, and in a release build a header silently missing its Attribute
/// Info message, which is the zero-count defect this function exists to prevent.
pub(crate) fn build_v2_object_header(region: &[u8]) -> Result<Vec<u8>, Error> {
    let mut owned = region.to_vec();
    ensure_attribute_info(&mut owned)?;
    Ok(build_v2_object_header_verbatim(&owned))
}

/// [`build_v2_object_header`] without the attribute-storage normalization, for
/// the region that function has already normalized.
fn build_v2_object_header_verbatim(region: &[u8]) -> Vec<u8> {
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

    /// An object-reference attribute named `name` pointing at `address`.
    ///
    /// Built by taking a `u64` attribute — whose value is already the 8 little-
    /// endian bytes an object reference is stored as — and relabelling its
    /// datatype, because no public API stages one: [`AttrValue`] has no
    /// reference variant, so a file carrying such an attribute was written by
    /// the reference C library, and the copy path re-emits its bytes verbatim
    /// like any other attribute's.
    fn reference_attr(name: &str, address: u64) -> crate::attribute::AttributeMessage {
        let mut attr = crate::type_builders::build_attr_message(name, &AttrValue::U64(address));
        attr.datatype = Datatype::Reference {
            size: 8,
            ref_type: crate::datatype::ReferenceType::Object,
        };
        assert_eq!(
            attr.raw_data,
            address.to_le_bytes(),
            "the value is the address"
        );
        attr
    }

    /// Wrap one attribute in the object-header message record a header region
    /// holds it in: type, body size, flags, body.
    fn inline_attr_region(attr: &crate::attribute::AttributeMessage) -> Vec<u8> {
        let body = attr.serialize_v3(LENGTH_SIZE);
        let mut region = vec![0x0C, 0, 0, 0];
        region[1..3].copy_from_slice(&(body.len() as u16).to_le_bytes());
        region.extend_from_slice(&body);
        region
    }

    /// A reference target supplied as a *raw address* is screened exactly as one
    /// supplied as a path (issue #317), and the address it would have resolved
    /// to is returned unchanged when it names space this commit keeps.
    ///
    /// Driven directly because no session API stages one: `ObjectRefTarget::Raw`
    /// is what [`crate::repack`]'s faithful re-emit builds, through
    /// `with_embedded_object_references`, and a builder reachable from a session
    /// produces only [`ObjectRefTarget::Path`]. The arm is here so the two ways
    /// a target can name an object answer to the same rule rather than to
    /// whichever one a caller happened to use.
    #[test]
    fn a_raw_reference_target_is_screened_like_a_path_one() {
        use tempfile::tempdir;
        let dir = tempdir().unwrap();
        let path = dir.path().join("raw_target.h5");
        let mut b = crate::writer::FileBuilder::new();
        b.create_dataset("d").with_i32_data(&[1, 2, 3]);
        b.write(&path).unwrap();
        let engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();

        let nodes: BTreeMap<PathKey, Node> = BTreeMap::new();
        let path_addr: BTreeMap<PathKey, u64> = BTreeMap::new();
        let resolve = |address: u64, spans: Vec<(u64, u64)>| {
            WriteEngine::resolve_reference_target(
                &ObjectRefTarget::Raw(address),
                &path_addr,
                &nodes,
                &[],
                &[],
                &[],
                &ReclaimedSpace { spans, base: 0 },
                &engine.image(),
                engine.superblock(),
            )
        };

        assert!(
            resolve(300, vec![(248, 71)]).is_err(),
            "an address inside a reclaimed span is refused"
        );
        assert_eq!(
            resolve(300, vec![(400, 71)]).unwrap(),
            300,
            "an address outside every reclaimed span is carried through"
        );
        assert_eq!(
            resolve(300, Vec::new()).unwrap(),
            300,
            "a commit that reclaims nothing screens nothing"
        );
        // The two sentinels name no object, so they are carried through even
        // when they fall inside a reclaimed span.
        assert_eq!(resolve(0, vec![(0, 4096)]).unwrap(), 0);
        assert_eq!(resolve(UNDEF, vec![(0, u64::MAX)]).unwrap(), UNDEF);
    }

    /// A copied object's *attributes* are screened against the space the commit
    /// reclaims, in either storage an object can hold them in — inline in the
    /// header region, or in the fractal heap a dense object uses — and an
    /// address outside those spans passes in both (issue #317).
    #[test]
    fn a_copied_reference_attribute_is_screened_in_both_storages() {
        let empty = BytesSource::new(Vec::new());
        let reclaimed = ReclaimedSpace {
            spans: vec![(248, 71)],
            base: 0,
        };
        // 248 is the first byte of the reclaimed span, 318 its last, 319 the
        // byte after it.
        for (address, refused) in [(248u64, true), (318, true), (319, false), (247, false)] {
            let attr = reference_attr("target", address);
            let dense = CopyTree::DatasetVerbatim {
                region: Vec::new(),
                dense_attrs: vec![attr.clone()],
            };
            let inline = CopyTree::DatasetVerbatim {
                region: inline_attr_region(&attr),
                dense_attrs: Vec::new(),
            };
            for (storage, tree) in [("dense", &dense), ("inline", &inline)] {
                let got = screen_copied_references(tree, &reclaimed, &empty);
                assert_eq!(
                    got.is_err(),
                    refused,
                    "{storage} attribute at {address}: {got:?}"
                );
            }
        }
    }

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
    /// `Dataset::append`'s crash-consistency harness, but driven through
    /// the in-place edit engine's own mirror (disk-before-mirror ordering) to prove the shared
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
                    let mut s = WriteEngine::open_with_locking(&p, FileLocking::Enabled).unwrap();
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

    /// A *filtered* append whose length is not a whole number of chunks writes a
    /// partial last chunk. That chunk's index element is a fresh insert past the
    /// old dimension — the same position every whole chunk beside it occupies —
    /// so stopping anywhere in the durability sequence must still read back as
    /// the old prefix, exactly as an aligned filtered append does. This is the
    /// case the in-place refusal used to cover and no longer does; without the
    /// phase sweep, "it reads back fine" would only be testing phase 4.
    #[test]
    fn append_inplace_crash_consistency_filtered_partial_last_chunk() {
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
                .with_chunks(&[chunk])
                .with_shuffle()
                .with_deflate(4);
            b.write(path).unwrap();
        };

        // Aligned starts (a filtered append requires one), unaligned lengths:
        // one that stays inside a single new chunk, and one that spans several
        // and ends partway through the last.
        for (n, chunk, add) in [(8i32, 4u64, 2i32), (8, 4, 9), (0, 4, 3)] {
            let dir = tempdir().unwrap();
            let base = dir.path().join("base.h5");
            build(&base, n, chunk);

            for max_phase in 1u8..=4 {
                let p = dir
                    .path()
                    .join(format!("crash_f_{n}_{chunk}_{max_phase}.h5"));
                std::fs::copy(&base, &p).unwrap();
                {
                    let mut s = WriteEngine::open_with_locking(&p, FileLocking::Enabled).unwrap();
                    s.append_inplace_i32_phased("d", &(n..n + add).collect::<Vec<_>>(), max_phase)
                        .unwrap();
                    // session dropped here, simulating a crash after `max_phase`
                }
                let expected_len = if max_phase == 4 { n + add } else { n };
                let f = PureFile::from_bytes(std::fs::read(&p).unwrap()).unwrap();
                assert_eq!(
                    f.dataset("d").unwrap().read_i32().unwrap(),
                    (0..expected_len).collect::<Vec<_>>(),
                    "inconsistent view after crash at phase {max_phase} (n={n}, chunk={chunk}, \
                     add={add})"
                );
            }
        }
    }

    /// Build a one-element-per-chunk unlimited `d` holding `0..n`, the shape the
    /// crash-consistency harnesses below grow.
    fn build_unit_chunked(path: &std::path::Path, n: i32) {
        use crate::writer::FileBuilder;
        let data: Vec<i32> = (0..n).collect();
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&data)
            .with_shape(&[n as u64])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[1]);
        b.write(path).unwrap();
    }

    /// Stop an append after `max_phase` durability phases and hand back the
    /// resulting file. Dropping the engine inside is the simulated crash: no
    /// further phases run and no close barrier is written.
    fn append_stopped_at(
        base: &std::path::Path,
        out: &std::path::Path,
        values: std::ops::Range<i32>,
        max_phase: u8,
    ) {
        std::fs::copy(base, out).unwrap();
        let mut s = WriteEngine::open_with_locking(out, FileLocking::Enabled).unwrap();
        s.append_inplace_i32_phased("d", &values.collect::<Vec<_>>(), max_phase)
            .unwrap();
    }

    /// Growing an Extensible-Array index across its inline -> direct-block ->
    /// super-block boundaries touches far more index structure than the
    /// partial-tail case above. Stopping at any phase boundary must still read
    /// back as a consistent prefix.
    ///
    /// Restores coverage lost with the deprecated `SwmrWriter` (issue #202); the
    /// owned path drives the same `apply_ea_append` engine.
    #[test]
    fn append_inplace_crash_consistency_across_ea_boundaries() {
        use crate::reader::File as PureFile;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let base = dir.path().join("base.h5");
        let (n, target) = (50i32, 250i32);
        build_unit_chunked(&base, n);

        for max_phase in 1u8..=4 {
            let p = dir.path().join(format!("crash_ea_{max_phase}.h5"));
            append_stopped_at(&base, &p, n..target, max_phase);
            let expected_len = if max_phase == 4 { target } else { n };
            let f = PureFile::from_bytes(std::fs::read(&p).unwrap()).unwrap();
            assert_eq!(
                f.dataset("d").unwrap().read_i32().unwrap(),
                (0..expected_len).collect::<Vec<_>>(),
                "inconsistent view after crash at phase {max_phase}"
            );
        }
    }

    /// The same guarantee for an append that crosses the paged-data-block
    /// boundary (~131,060 chunks), where phase 1 allocates a paged super block,
    /// paged data blocks, the per-page checksums, and the page-init bitmap. This
    /// is the most intricate in-place growth the engine performs, and truncating
    /// it partway is exactly what a power loss does.
    ///
    /// Restores coverage lost with the deprecated `SwmrWriter` (issue #202).
    /// Opening a paged persisting file must seed each free section into the list
    /// for the page type its *slot* names, not one derived from its size.
    ///
    /// The managers a paged file uses mean different things — SUPER (slot 0) is
    /// metadata, DRAW (slot 2) is raw — and a freed section's size says nothing
    /// about which it came from. Getting this wrong is invisible from the outside:
    /// the total free space is unchanged, the reference library still opens the
    /// file, and only a later allocation drawn from the wrong list would mix a
    /// page. So assert the routing directly.
    #[test]
    fn paged_open_seeds_each_manager_by_slot() {
        use crate::writer::FileBuilder;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let path = dir.path().join("paged_seed.h5");
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&(0..1000).collect::<Vec<i32>>())
            .with_shape(&[1000]);
        b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
            .with_file_space_page_size(4096);
        b.write(&path).unwrap();

        // Read the file's recorded free space *before* opening the session: the
        // session holds an exclusive OS lock, and on Windows those locks are
        // mandatory, so a concurrent `File::open` would fail outright.
        let on_disk: u64 = crate::reader::File::open(&path)
            .unwrap()
            .persisted_free_space()
            .iter()
            .map(|&(_, l)| l)
            .sum();

        let s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
        let pg = s.paged.as_ref().expect("a paged file installs paged state");
        assert_eq!(pg.page_size, 4096);

        // The from-scratch writer leaves a page tail free in both the metadata and
        // the raw pages, so both per-type managers are populated. If every slot
        // were funnelled into one list, one of these would be empty.
        assert!(
            !pg.meta.sections().is_empty(),
            "SUPER (slot 0) sections seed the metadata list"
        );
        assert!(
            !pg.raw.sections().is_empty(),
            "DRAW (slot 2) sections seed the raw list, not the metadata list"
        );

        // Nothing is double-counted or dropped: the lists partition exactly the
        // free space the file records, and no two sections overlap. A file this
        // crate wrote has nothing unclassified — it only ever files a whole
        // aligned page under the generic-large manager — so summing it in is a
        // statement about that too.
        let mut all = pg.reusable_sections();
        all.extend(pg.unclassified.sections());
        assert!(
            pg.unclassified.sections().is_empty(),
            "our own paged writer files nothing whose page type is unknown"
        );
        let flat: u64 = all.iter().map(|&(_, l)| l).sum();
        assert_eq!(
            flat, on_disk,
            "the split lists hold exactly the file's free space"
        );
        all.sort_by_key(|&(a, _)| a);
        let mut prev_end = 0u64;
        for (addr, len) in all {
            assert!(addr >= prev_end, "the per-type lists do not overlap");
            prev_end = addr + len;
        }
    }

    /// Deleting a chunked dataset from a paged file the *reference library* wrote
    /// must not offer its chunk index for raw reuse.
    ///
    /// A chunk index is metadata by the format's taxonomy, and the C library
    /// allocates one accordingly — out of metadata pages, which on a small file
    /// means page 0, alongside the superblock, the root group, and every other
    /// object's header. This crate places its own indexes in raw pages beside the
    /// chunk data instead, and the reclaim path used to assume that of every file.
    /// Freeing a C-written index under that assumption puts its bytes on the raw
    /// list, and the next commit writes a dataset's values into the middle of a
    /// live metadata page.
    ///
    /// Both libraries still read such a file — every address in it is still
    /// correct — so nothing downstream reports the damage. The assertion has to be
    /// about placement itself: no byte this commit writes may land in a page that
    /// held live metadata beforehand.
    #[test]
    // Builds the fixture with the reference HDF5 C library (`hdf5-metno`), a
    // 64-bit little-endian-only dev-dependency; skip elsewhere so the lib tests
    // still run there.
    #[cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
    fn a_c_written_chunk_index_is_not_reclaimed_as_raw() {
        use hdf5::plist::file_create::FileSpaceStrategy as CStrategy;
        use tempfile::tempdir;

        const PAGE: u64 = 4096;
        let dir = tempdir().unwrap();
        let path = dir.path().join("c_paged_index.h5");
        {
            let f = hdf5::FileBuilder::new()
                .with_fapl(|fapl| fapl.libver_v110())
                .with_fcpl(|fcpl| {
                    fcpl.file_space_strategy(CStrategy::FreeSpaceManager {
                        paged: true,
                        persist: true,
                        threshold: 1,
                    })
                    .file_space_page_size(PAGE)
                })
                .create(&path)
                .unwrap();
            let ds = f
                .new_dataset::<i32>()
                .shape(hdf5::SimpleExtents::resizable(vec![8192]))
                .chunk((512,))
                .create("victim")
                .unwrap();
            ds.write_raw(&(0..8192i32).collect::<Vec<i32>>()).unwrap();
            f.new_dataset::<i32>()
                .shape((4,))
                .create("keep")
                .unwrap()
                .write_raw(&[1i32, 2, 3, 4])
                .unwrap();
            f.close().unwrap();
        }

        let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
        let page_size = s.paged.as_ref().expect("a paged file").page_size;
        assert_eq!(page_size, PAGE);
        // Where the C library put the index: every page it touches is a metadata
        // page, since the C library allocates an index as metadata.
        let victim_addr =
            crate::group_v2::resolve_path_any(s.image.as_slice().unwrap(), &s.superblock, "victim")
                .unwrap();
        let index_spans = s
            .chunked_index_spans(usize::try_from(victim_addr).unwrap())
            .expect("the C library's extensible-array index is enumerable");
        assert!(!index_spans.is_empty());
        // Page 0 always holds the superblock and the root group header; the index
        // pages are what this test is really about.
        let mut meta_pages: Vec<u64> = vec![0];
        for (addr, len) in &index_spans {
            for p in (addr / PAGE)..=((addr + len - 1) / PAGE) {
                meta_pages.push(p);
            }
        }
        meta_pages.sort_unstable();
        meta_pages.dedup();

        // Delete the dataset, then write a small dataset that would fit in the
        // index's freed bytes.
        s.delete("/victim").unwrap();
        s.commit().unwrap();
        let mut db = crate::type_builders::DatasetBuilder::new("added");
        db.with_f64_data(&[2.5f64; 8]).with_shape(&[8]);
        s.stage_created_dataset("/added", db).unwrap();
        s.commit().unwrap();

        // Release the session's exclusive OS lock before reading the file back;
        // those locks are mandatory on Windows.
        drop(s);

        // Every raw byte of the new dataset must be outside those pages.
        let f = crate::reader::File::open(&path).unwrap();
        let added = f.dataset("added").unwrap();
        let crate::Layout::Contiguous {
            address: Some(addr),
            size,
        } = added.layout().unwrap()
        else {
            panic!("a small f64 dataset is stored contiguously");
        };
        for p in (addr / PAGE)..=((addr + size - 1) / PAGE) {
            assert!(
                !meta_pages.contains(&p),
                "the new dataset's raw data landed at ({addr}, {size}), in page {p}, \
                 which held live metadata before the edit: {meta_pages:?}"
            );
        }

        assert_eq!(added.read_f64().unwrap(), vec![2.5f64; 8]);
        assert_eq!(
            f.dataset("keep").unwrap().read_i32().unwrap(),
            vec![1, 2, 3, 4]
        );
    }

    /// Padding a tail page whose type is unknown must leave the padding
    /// untracked, not guess a type for it.
    ///
    /// A commit pads the file to a page boundary before laying down its manager
    /// blocks. When this session has made no typed allocation, the tail page is
    /// one a previous session left non-aligned — only a crash does that, since a
    /// clean close pads — and nothing says what it holds. Recording the padding
    /// under a guess would advertise those bytes for reuse of that type, and half
    /// the time they sit in a page of the other one. Under-reporting is the safe
    /// direction, and it is the call [`PagedEdit::begin`] already makes for the
    /// same situation.
    ///
    /// Reuse is what made this reachable: before it, every commit appended at
    /// least the root group's header, so the tail type was always known by the
    /// time the padding ran.
    #[test]
    fn padding_a_tail_page_of_unknown_type_records_nothing() {
        use crate::writer::FileBuilder;
        use tempfile::tempdir;

        const PAGE: u64 = 4096;
        let dir = tempdir().unwrap();

        let build = |path: &std::path::Path| {
            let mut b = FileBuilder::new();
            b.create_dataset("d")
                .with_i32_data(&(0..1000).collect::<Vec<i32>>())
                .with_shape(&[1000]);
            b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
                .with_file_space_page_size(PAGE);
            b.write(path).unwrap();
        };

        // Unknown tail type: pad, but record nothing.
        let unknown = dir.path().join("unknown_tail.h5");
        build(&unknown);
        let mut s = WriteEngine::open_with_locking(&unknown, FileLocking::Enabled).unwrap();
        s.append(&[0u8; 100]).unwrap(); // leaves the image non-page-aligned
        assert!(s.paged.as_ref().unwrap().last.is_none());
        s.pad_to_page().unwrap();
        let pg = s.paged.as_ref().unwrap();
        assert_eq!(s.image.len() % PAGE, 0, "the file is padded to a page");
        assert!(
            pg.meta_pad.is_empty() && pg.raw_pad.is_empty(),
            "padding a tail page of unknown type must claim no page type"
        );
        drop(s);

        // The control: a known tail type is recorded, so the test above is about
        // the unknown case and not about padding never being tracked at all.
        let known = dir.path().join("known_tail.h5");
        build(&known);
        let mut s = WriteEngine::open_with_locking(&known, FileLocking::Enabled).unwrap();
        s.begin_page(PageType::Meta).unwrap();
        s.append(&[0u8; 100]).unwrap();
        s.pad_to_page().unwrap();
        let pg = s.paged.as_ref().unwrap();
        assert_eq!(
            pg.meta_pad.len(),
            1,
            "a known metadata tail records its padding as metadata free space"
        );
        assert!(pg.raw_pad.is_empty());
    }

    /// The reference C library's generic-large manager holds free space of *both*
    /// page types, so a section from it is only reusable when it covers whole
    /// aligned pages.
    ///
    /// `H5F_MEM_PAGE_GENERIC` is aliased to `H5F_MEM_PAGE_LARGE_SUPER` and
    /// commented in the C library's own header as *"large-sized generic: meta and
    /// raw"*. Under paged aggregation on a contiguous-address driver — the default
    /// — `H5MF__alloc_to_fs_type` sends **every** allocation of a page or more
    /// there whatever its type, and `H5MF__alloc_pagefs` then records that
    /// allocation's page-alignment tail as a free section in the same manager. The
    /// tail is smaller than a page and sits in a page whose earlier bytes are the
    /// live object, so filing it as raw and handing it to chunk data would put raw
    /// bytes inside a metadata page.
    ///
    /// This measures that the C library really does write such a section — the
    /// assertion is worthless if it does not — and then pins that this engine
    /// keeps it out of reach: it is recorded, so a rewrite gives it back to the
    /// manager it came from, but never offered to an allocation.
    #[test]
    // Builds the fixture with the reference HDF5 C library (`hdf5-metno`), a
    // 64-bit little-endian-only dev-dependency; skip elsewhere so the lib tests
    // still run there.
    #[cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
    fn a_generic_large_section_is_only_reusable_as_whole_pages() {
        use hdf5::plist::file_create::FileSpaceStrategy as CStrategy;
        use tempfile::tempdir;

        const PAGE: u64 = 512;
        let dir = tempdir().unwrap();
        let path = dir.path().join("c_generic_large.h5");
        {
            let f = hdf5::FileBuilder::new()
                .with_fapl(|fapl| fapl.libver_v110())
                .with_fcpl(|fcpl| {
                    fcpl.file_space_strategy(CStrategy::FreeSpaceManager {
                        paged: true,
                        persist: true,
                        threshold: 1,
                    })
                    .file_space_page_size(PAGE)
                })
                .create(&path)
                .unwrap();
            let ds = f.new_dataset::<f64>().shape((64,)).create("d").unwrap();
            ds.write_raw(&vec![1.0f64; 64]).unwrap();
            // One attribute far larger than a page, so the object-header chunk
            // holding it is a "large" *metadata* allocation.
            let a = ds
                .new_attr::<i64>()
                .shape((512,))
                .create("big_attr")
                .unwrap();
            a.write_raw(&vec![7i64; 512]).unwrap();
            f.close().unwrap();
        }

        // The premise: at least one section in the generic-large manager (slot 6)
        // is a sub-page fragment. Read the managers slot by slot, since the
        // flattened public view cannot say which manager a section came from.
        let opened = crate::reader::File::open(&path).unwrap();
        let info = opened.file_space_info().expect("a persisting file").clone();
        drop(opened);
        assert_eq!(info.page_size, PAGE);
        let bytes = std::fs::read(&path).unwrap();
        let src = crate::source::BytesSource::new(bytes.as_slice());
        let slot6 = info.manager_addrs[6];
        assert_ne!(slot6, UNDEF, "the C library populated the large manager");
        let (sections, _) =
            free_space_manager::read_persisted_sections_source(&src, &[slot6], 0, 8).unwrap();
        let fragments: Vec<&FreeSection> = sections
            .iter()
            .filter(|s| s.addr % PAGE != 0 || s.size % PAGE != 0)
            .collect();
        assert!(
            !fragments.is_empty(),
            "the premise of this test: the C library files sub-page fragments in \
             its generic-large manager, but this file has none ({sections:?})"
        );

        // The behavior: every such fragment is recorded but unreachable, and only
        // whole aligned pages from that manager are placeable.
        let s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
        let pg = s.paged.as_ref().expect("a paged file installs paged state");
        let unclassified = pg.unclassified.sections();
        let reusable = pg.reusable_sections();
        for f in &fragments {
            assert!(
                unclassified.contains(&(f.addr, f.size)),
                "fragment ({}, {}) must be recorded as unclassified, not lost",
                f.addr,
                f.size
            );
            assert!(
                !reusable
                    .iter()
                    .any(|&(a, l)| a < f.addr + f.size && f.addr < a + l),
                "fragment ({}, {}) must not be offered to any allocation: {reusable:?}",
                f.addr,
                f.size
            );
        }
    }

    /// A paged allocation may be served from the other page type's free list only
    /// where that space covers whole free pages — never from a hole in a page the
    /// other type still occupies.
    ///
    /// This is the allocation-side half of the rule
    /// [`paged_open_seeds_each_manager_by_slot`] pins on the seeding side, and it
    /// is just as invisible from the outside: handing a raw allocation a hole in a
    /// live metadata page puts chunk bytes inside that page, which every reader —
    /// this crate's and the reference library's — resolves correctly, since the
    /// file's addresses all still point where they should. Only the paging
    /// degrades. So assert the choice directly, in both directions, with a
    /// same-type control proving the free region really was big enough to be
    /// taken, and the whole-page case proving the exception is reached rather than
    /// merely permitted.
    #[test]
    fn a_paged_allocation_only_crosses_page_types_over_whole_free_pages() {
        use crate::writer::FileBuilder;
        use tempfile::tempdir;

        const PAGE: u64 = 4096;

        /// A paged session whose free lists hold exactly one region each, of the
        /// page types named. Both regions are interior, so an allocation that
        /// takes one is distinguishable from one that appends.
        fn session(
            path: &std::path::Path,
            meta: Option<(u64, u64)>,
            raw: Option<(u64, u64)>,
        ) -> WriteEngine {
            let mut b = FileBuilder::new();
            b.create_dataset("d")
                .with_i32_data(&(0..4000).collect::<Vec<i32>>())
                .with_shape(&[4000]);
            b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
                .with_file_space_page_size(PAGE);
            b.write(path).unwrap();
            let mut s = WriteEngine::open_with_locking(path, FileLocking::Enabled).unwrap();
            let pg = s.paged.as_mut().expect("a paged file installs paged state");
            pg.meta = FreeList::new();
            pg.raw = FreeList::new();
            if let Some((addr, len)) = meta {
                pg.meta.free(addr, len);
            }
            if let Some((addr, len)) = raw {
                pg.raw.free(addr, len);
            }
            s
        }

        let dir = tempdir().unwrap();
        // Two interior regions well inside the file the builder wrote (16 KiB of
        // raw data alone), each a *fragment* of a page whose other bytes are live,
        // so either could physically hold the request and only the page-type rule
        // decides. Deliberately not page-aligned and not a whole page: that is the
        // case the rule forbids outright.
        let hole_a = (PAGE + 512, 2048);
        let hole_b = (2 * PAGE + 512, 2048);

        // Raw request, only a metadata fragment free: appends rather than mixing
        // the page.
        let mut s = session(&dir.path().join("a.h5"), Some(hole_a), None);
        assert!(
            matches!(
                s.reserve(1024, PageType::Raw).unwrap(),
                Placement::Appended { .. }
            ),
            "a raw allocation must not be served out of a live metadata page"
        );
        drop(s);

        // Metadata request, only a raw fragment free: likewise.
        let mut s = session(&dir.path().join("b.h5"), None, Some(hole_b));
        assert!(
            matches!(
                s.reserve(1024, PageType::Meta).unwrap(),
                Placement::Appended { .. }
            ),
            "a metadata allocation must not be served out of a live raw page"
        );
        drop(s);

        // The control: with the matching type free, the same request is reused —
        // so the two refusals above are the page-type rule, not a size failure.
        let mut s = session(&dir.path().join("c.h5"), Some(hole_a), Some(hole_b));
        assert!(
            matches!(
                s.reserve(1024, PageType::Raw).unwrap(),
                Placement::Reused { addr, .. } if addr == hole_b.0
            ),
            "a raw allocation takes the raw hole"
        );
        assert!(
            matches!(
                s.reserve(1024, PageType::Meta).unwrap(),
                Placement::Reused { addr, .. } if addr == hole_a.0
            ),
            "a metadata allocation takes the metadata hole"
        );
        drop(s);

        // The exception, and the whole reason the file stops growing (issue #286):
        // a page with *nothing* in it holds no type to contradict, so either kind
        // may open it. The page is claimed whole and what the request does not use
        // becomes free space of the claiming type.
        let mut s = session(&dir.path().join("d.h5"), Some((PAGE, PAGE)), None);
        assert!(
            matches!(
                s.reserve(1024, PageType::Raw).unwrap(),
                Placement::Reused { addr, .. } if addr == PAGE
            ),
            "a raw allocation may open an empty metadata page"
        );
        let pg = s.paged.as_ref().expect("still paged");
        assert_eq!(
            pg.raw.sections(),
            [(PAGE + 1024, PAGE - 1024)],
            "the rest of the claimed page is free space of the claiming type"
        );
        assert!(
            pg.meta.sections().is_empty(),
            "the page left the list it was claimed from"
        );
        drop(s);

        // And the claim starts at the empty page, not at the free run's own start.
        // A run that spans from mid-page into whole pages beyond it is the common
        // shape — the tail of a live page, then pages nothing is left in — and
        // taking it from the front would put the request in the live page.
        let mut s = session(
            &dir.path().join("e.h5"),
            None,
            Some((PAGE + 512, 3 * PAGE - 512)),
        );
        assert!(
            matches!(
                s.reserve(1024, PageType::Meta).unwrap(),
                Placement::Reused { addr, .. } if addr == 2 * PAGE
            ),
            "the claim must begin at the empty page, not at the run's start"
        );
        // Both edges the claim leaves behind survive it: they are ordinary free
        // space of the type that already held them, and dropping either is the same
        // silent leak this change exists to remove.
        let pg = s.paged.as_ref().expect("still paged");
        assert_eq!(
            pg.raw.sections(),
            [(PAGE + 512, PAGE - 512), (3 * PAGE, PAGE)],
            "the fragment below the claimed page and the page above it both stay free"
        );
        assert_eq!(
            pg.meta.sections(),
            [(2 * PAGE + 1024, PAGE - 1024)],
            "the rest of the claimed page is free space of the claiming type"
        );
    }

    /// Every byte a paged session could still hand out, across both page types.
    fn free_total(s: &WriteEngine) -> u64 {
        let pg = s.paged.as_ref().expect("a paged session");
        pg.meta
            .sections()
            .into_iter()
            .chain(pg.raw.sections())
            .map(|(_, len)| len)
            .sum()
    }

    /// A paged commit's tail removes from the free lists exactly the bytes it goes
    /// on to write — no more, and nothing at all when it declines to place itself.
    ///
    /// This is the invariant the whole change rests on, and the one an integration
    /// test cannot reach: the tail's length and the hole it lands in determine each
    /// other, so which settlement the arithmetic reaches depends on the shape of
    /// the free list, and a file the writer builds only ever produces some of them.
    /// In particular, best fit prefers the *smallest* region that fits, so a hole
    /// whose length is exactly the length the tail proposed is the one it will
    /// choose — and consuming a region outright drops a section from the managers,
    /// making the plan come out *shorter* than the space just reserved for it. A
    /// tail that accepted that would retire the difference from the free list with
    /// nothing recording it, which is issue #286 again, a few bytes at a time.
    ///
    /// Hence a contiguous sweep of hole sizes rather than a chosen one: the length
    /// the tail proposes is a property of the file and its free list, not something
    /// this test should have to know, and the sweep is certain to cross it.
    #[test]
    fn a_paged_tail_takes_exactly_the_space_it_fills() {
        use crate::writer::FileBuilder;
        use tempfile::tempdir;

        const PAGE: u64 = 4096;
        /// Any plausible extension length exercises the same arithmetic, and the
        /// invariant holds for every one of them; the tail's real length is settled
        /// by the commit, which is not what is under test here.
        const EXT_LEN: u64 = 100;

        let dir = tempdir().unwrap();
        let path = dir.path().join("tail_exact.h5");
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&(0..4000).collect::<Vec<i32>>())
            .with_shape(&[4000]);
        b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
            .with_file_space_page_size(PAGE);
        b.write(&path).unwrap();
        let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
        let os = s.superblock.offset_size;

        let mut placed = 0usize;
        for hole in 120..420u64 {
            {
                let pg = s.paged.as_mut().expect("a paged file installs paged state");
                pg.meta = FreeList::new();
                pg.raw = FreeList::new();
                pg.unclassified = FreeList::new();
                pg.meta.free(PAGE, hole);
            }
            let free_before = free_total(&s);
            let layout = s.tail_layout(&[], &[], EXT_LEN, PAGE, os);
            let free_after = free_total(&s);
            match layout {
                Some((_, _, at, blocks_len)) => {
                    placed += 1;
                    assert_eq!(
                        free_before - free_after,
                        blocks_len,
                        "hole {hole}: the tail took {} bytes at {at} to write {blocks_len}",
                        free_before - free_after
                    );
                }
                None => assert_eq!(
                    free_after, free_before,
                    "hole {hole}: a tail that declines to place itself must hand back \
                     everything it tried"
                ),
            }
        }
        assert!(
            placed > 0,
            "the sweep must reach holes the tail can actually use, or it asserts \
             nothing about placement"
        );
    }

    /// A paged commit whose tail finds no free space to sit in opens a page for it
    /// — and hands the rest of that page back as free metadata.
    ///
    /// Every paged file this crate writes has holes in it from the start, so the
    /// suite's ordinary fixtures always reuse and this branch is never taken. It is
    /// reachable in the wild, by a file whose free space is all spoken for, and the
    /// page it opens is the same page a page-per-commit leak used to strand. Emptying
    /// the free lists by hand is what puts the file in that state deliberately.
    #[test]
    fn a_paged_tail_with_nowhere_to_go_opens_a_page_and_frees_the_rest() {
        use crate::writer::FileBuilder;
        use tempfile::tempdir;

        const PAGE: u64 = 4096;
        let dir = tempdir().unwrap();
        let path = dir.path().join("tail_appends.h5");
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&(0..4000).collect::<Vec<i32>>())
            .with_shape(&[4000]);
        b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
            .with_file_space_page_size(PAGE);
        b.write(&path).unwrap();
        let before = std::fs::metadata(&path).unwrap().len();

        let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
        {
            let pg = s.paged.as_mut().expect("a paged file installs paged state");
            pg.meta = FreeList::new();
            pg.raw = FreeList::new();
            pg.unclassified = FreeList::new();
        }
        // A commit with nothing staged returns without writing, so give it one
        // small metadata object to place. It appends for want of anywhere else,
        // and the tail follows it into the same page.
        s.create_group("g").unwrap();
        s.commit().unwrap();

        let after = std::fs::metadata(&path).unwrap().len();
        assert!(
            after > before && (after - before) % PAGE == 0,
            "the commit opened whole pages ({before} -> {after})"
        );
        // What the tail did not fill of the page it opened is metadata this session
        // can still spend. Losing it is how the file used to give up most of a page
        // on every commit; the managers just written cannot record it, so the
        // session carries it to the next commit.
        let pg = s.paged.as_ref().expect("still paged");
        let (addr, len) = pg
            .meta
            .sections()
            .into_iter()
            .find(|&(addr, len)| addr + len == after)
            .expect("the page the tail opened leaves a free remainder at end-of-file");
        assert!(
            len > 0 && len < PAGE && addr >= before,
            "the tail's blocks take the front of the page it opened and the rest is \
             free ({len} of {PAGE} at {addr}, file {before} -> {after})"
        );
    }

    /// Deleting a chunked dataset from a paged file must not record its freed
    /// chunk index in the *metadata* manager.
    ///
    /// Every writer in this crate emits a chunk index in the same run as the chunk
    /// data it indexes, so the index sits in a raw page. Recording it as metadata
    /// would advertise a metadata-sized hole inside a page that still holds another
    /// dataset's live chunk data, and the reference library placing metadata there
    /// would mix the page — the one thing a paged file forbids. Page homogeneity is
    /// preserved on disk either way, so the mis-filing is invisible to a signature
    /// scan of the file and to the C library; the manager a section lands in has to
    /// be checked directly.
    #[test]
    fn deleted_chunk_index_is_freed_into_a_raw_manager() {
        use crate::writer::FileBuilder;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let path = dir.path().join("paged_chunk_index_free.h5");
        let page = 4096u64;
        let mut b = FileBuilder::new();
        for name in ["drop", "keep"] {
            b.create_dataset(name)
                .with_i32_data(&(0..200).collect::<Vec<i32>>())
                .with_shape(&[200])
                .with_chunks(&[50]);
        }
        b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
            .with_file_space_page_size(page);
        b.write(&path).unwrap();

        {
            let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
            s.delete("/drop").unwrap();
            s.commit().unwrap();
        }

        // Pages still occupied by the surviving dataset's chunk data. Read with the
        // session closed: its lock is mandatory on Windows.
        let live_raw_pages: Vec<u64> = {
            let f = crate::reader::File::open(&path).unwrap();
            let ds = f.dataset("keep").unwrap();
            let mut pages: Vec<u64> = ds
                .chunks()
                .unwrap()
                .iter()
                .filter(|c| c.storage_size > 0)
                .flat_map(|c| (c.address / page)..=((c.address + c.storage_size - 1) / page))
                .collect();
            pages.sort_unstable();
            pages.dedup();
            pages
        };
        assert!(!live_raw_pages.is_empty(), "expected live raw pages");

        let s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
        let pg = s.paged.as_ref().expect("a paged file installs paged state");
        for (addr, len) in pg.meta.sections() {
            for p in (addr / page)..=((addr + len - 1) / page) {
                assert!(
                    !live_raw_pages.contains(&p),
                    "metadata free section ({addr}, {len}) sits in page {p}, which still \
                     holds live raw chunk data"
                );
            }
        }
        // The index really was reclaimed somewhere, so this is not vacuous.
        let reclaimed: u64 = pg.reusable_sections().iter().map(|&(_, l)| l).sum();
        assert!(reclaimed > 0, "the delete reclaimed nothing");
    }

    /// A paged commit that fails partway must leave the session's free lists
    /// exactly as it found them.
    ///
    /// Everything the commit gathers to free is still *live* until the superblock
    /// repoint: the objects occupying those regions are reachable from the old
    /// root, which a failed commit never replaces. A session that recorded them as
    /// free anyway would hand them out on the next commit, and the file would lose
    /// data with no error anywhere — in a release build, where the free list's
    /// double-free `debug_assert` is compiled out, silently.
    ///
    /// The failure is induced by pointing the superblock extension at a byte range
    /// that is not an object header, which fails the extension rewrite immediately
    /// after the regions are gathered.
    #[test]
    fn failed_paged_commit_leaves_the_free_lists_untouched() {
        use crate::writer::FileBuilder;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let path = dir.path().join("paged_failed_commit.h5");
        let mut b = FileBuilder::new();
        b.create_dataset("keep")
            .with_i32_data(&(0..200).collect::<Vec<i32>>())
            .with_shape(&[200]);
        b.create_dataset("drop")
            .with_i32_data(&(0..200).collect::<Vec<i32>>())
            .with_shape(&[200]);
        b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
            .with_file_space_page_size(4096);
        b.write(&path).unwrap();

        let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
        let before = s.space_accounting().reusable_free_space;

        // Break the extension so the commit fails *after* it has gathered the
        // regions `drop` vacates and *before* the superblock repoint.
        let good_ext = s.superblock.superblock_extension_address;
        s.superblock.superblock_extension_address = Some(0);
        s.delete("/drop").unwrap();
        assert!(
            s.commit().is_err(),
            "a commit with an unreadable extension must fail"
        );

        assert_eq!(
            s.space_accounting().reusable_free_space,
            before,
            "a failed commit must not record still-live regions as free"
        );

        // The session stays usable: repair the extension and commit for real. If
        // the failed commit had folded its regions in, this second commit would
        // double-free them (a debug assertion) and publish `keep`'s live extent.
        s.superblock.superblock_extension_address = good_ext;
        s.delete("/drop").unwrap();
        s.commit()
            .expect("the session is usable after a failed commit");

        // Release the session's exclusive OS lock before reading the file back.
        // Those locks are mandatory on Windows, so a `File::open` overlapping the
        // session fails outright there (advisory locks elsewhere would allow it).
        drop(s);

        let f = crate::reader::File::open(&path).unwrap();
        let kept = f.dataset("keep").unwrap().read_i32().unwrap();
        assert_eq!(kept, (0..200).collect::<Vec<i32>>(), "keep survives intact");
        let freed: u64 = f.persisted_free_space().iter().map(|&(_, l)| l).sum();
        let live_end = f.file_size();
        assert!(
            freed < live_end,
            "the recorded free space cannot cover the whole file"
        );
    }

    /// A commit that writes into reused free space and then dies must leave the
    /// file exactly as it found it.
    ///
    /// This is the safety argument for reuse, stated as a test: a commit may
    /// overwrite a freed region *before* the superblock repoint precisely because
    /// nothing reachable from the on-disk root lives there any more, so an attempt
    /// that never reaches the repoint is invisible. A chunked dataset is the case
    /// worth pinning — it is the largest thing a commit places, so it overwrites
    /// the most, and it is the one that used only to append.
    ///
    /// The failure is induced the same way as
    /// [`failed_paged_commit_leaves_the_free_lists_untouched`]: an unreadable
    /// superblock extension, which the persisting tail hits *after* the apply loop
    /// has written every object.
    #[test]
    fn a_failed_commit_that_reused_free_space_leaves_the_file_intact() {
        use crate::writer::FileBuilder;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let path = dir.path().join("reuse_failed_commit.h5");
        let victim: Vec<f64> = (0..4096).map(|i| (i % 13) as f64).collect();
        let ceiling: Vec<i32> = (0..500).collect();
        let mut b = FileBuilder::new();
        b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
        b.create_dataset("victim")
            .with_f64_data(&victim)
            .with_shape(&[4096])
            .with_chunks(&[512]);
        // Above the victim, so the delete leaves an interior hole and the reuse
        // has live bytes on both sides of what it overwrites.
        b.create_dataset("ceiling")
            .with_i32_data(&ceiling)
            .with_shape(&[500]);
        b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
        b.write(&path).unwrap();

        let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
        s.delete("/victim").unwrap();
        s.commit().unwrap();
        let free_before = s.space_accounting().reusable_free_space;
        let len_before = std::fs::metadata(&path).unwrap().len();
        assert!(
            free_before.iter().any(|&(_, l)| l > 4096 * 8 / 2),
            "the deleted chunked dataset left a hole worth reusing: {free_before:?}"
        );

        // Break the extension so the commit fails in its persisting tail, after
        // the apply loop has written the new dataset into that hole.
        s.superblock.superblock_extension_address = Some(0);
        let mut db = crate::type_builders::DatasetBuilder::new("fresh");
        db.with_f64_data(&vec![7.5f64; 4096])
            .with_shape(&[4096])
            .with_chunks(&[512]);
        s.stage_created_dataset("/fresh", db).unwrap();
        assert!(
            s.commit().is_err(),
            "a commit with an unreadable extension must fail"
        );
        assert_eq!(
            s.space_accounting().reusable_free_space,
            free_before,
            "the failed commit gives back the region it drew from"
        );
        // Release the session's exclusive OS lock before reading the file back;
        // those locks are mandatory on Windows.
        drop(s);

        // The file still describes the tree the last *successful* commit left: the
        // survivors read exactly, the half-written dataset is not linked, and the
        // superblock's end-of-file still matches the file.
        assert_eq!(std::fs::metadata(&path).unwrap().len(), len_before);
        let f = crate::reader::File::open(&path).unwrap();
        assert_eq!(f.file_size(), len_before);
        assert_eq!(
            f.dataset("keep").unwrap().read_i32().unwrap(),
            vec![1, 2, 3]
        );
        assert_eq!(f.dataset("ceiling").unwrap().read_i32().unwrap(), ceiling);
        assert!(f.dataset("victim").is_err());
        assert!(f.dataset("fresh").is_err());
    }

    #[test]
    fn append_inplace_crash_consistency_paged_prefix() {
        use crate::reader::File as PureFile;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let base = dir.path().join("base.h5");
        let (start, target) = (131_000i32, 132_000i32);
        build_unit_chunked(&base, start);

        for max_phase in 1u8..=4 {
            let p = dir.path().join(format!("crash_paged_{max_phase}.h5"));
            append_stopped_at(&base, &p, start..target, max_phase);
            let expected_len = if max_phase == 4 { target } else { start };
            let f = PureFile::from_bytes(std::fs::read(&p).unwrap()).unwrap();
            assert_eq!(
                f.dataset("d").unwrap().read_i32().unwrap(),
                (0..expected_len).collect::<Vec<_>>(),
                "inconsistent paged view after crash at phase {max_phase}"
            );
        }
    }

    /// The consistent-prefix guarantee must hold for the *reference C library*,
    /// not only this crate's reader — and this crate's reader is the more lenient
    /// of the two, so it cannot stand in for it.
    ///
    /// The pure reader bounds chunk reads by `min(EA count, dimension)`, which
    /// makes it tolerate a phase-3 state where the element count has advanced
    /// past the dimension. The C library instead walks strictly by the dataspace
    /// dimension and re-validates block checksums, so a stale end-of-file, a
    /// half-grown index, or a mis-checksummed block could satisfy the reader here
    /// and still break C or h5py. That gap is the whole point of the test:
    /// crash-safety for the append path is an interop guarantee.
    ///
    /// Both starting layouts are covered — a partial trailing chunk and the
    /// EA-boundary growth above — since they exercise different index writes.
    ///
    /// Restores coverage lost with the deprecated `SwmrWriter` and `AppendWriter`
    /// (issue #202).
    #[test]
    // Reads back with the reference HDF5 C library (`hdf5-metno`), a
    // 64-bit little-endian-only dev-dependency; skip elsewhere so the lib
    // tests still run there.
    #[cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
    fn append_inplace_crash_consistency_c_library_reads_prefix() {
        use tempfile::tempdir;

        // (initial length, chunk length, appended length): a partial trailing
        // chunk, a chunk-aligned start, and the EA-boundary crossing.
        for (n, chunk, add) in [(6i32, 4u64, 5i32), (8, 2, 6), (50, 1, 200)] {
            let dir = tempdir().unwrap();
            let base = dir.path().join("base.h5");
            {
                use crate::writer::FileBuilder;
                let mut b = FileBuilder::new();
                b.create_dataset("d")
                    .with_i32_data(&(0..n).collect::<Vec<i32>>())
                    .with_shape(&[n as u64])
                    .with_maxshape(&[u64::MAX])
                    .with_chunks(&[chunk]);
                b.write(&base).unwrap();
            }

            for max_phase in 1u8..=4 {
                let p = dir
                    .path()
                    .join(format!("crash_c_{n}_{chunk}_{max_phase}.h5"));
                append_stopped_at(&base, &p, n..n + add, max_phase);
                let expected_len = if max_phase == 4 { n + add } else { n };
                let f = hdf5::File::open(&p).unwrap();
                assert_eq!(
                    f.dataset("d").unwrap().read_raw::<i32>().unwrap(),
                    (0..expected_len).collect::<Vec<_>>(),
                    "C library saw an inconsistent view after crash at phase {max_phase} \
                     (n={n}, chunk={chunk})"
                );
                f.close().unwrap();
            }
        }
    }

    /// Crash recovery across the phase-3/phase-4 gap.
    ///
    /// A writer that crashes after publishing the Extensible-Array element count
    /// (phase 3) but before publishing the dataspace dimension (phase 4) leaves
    /// the on-disk count ahead of the committed dimension. A fresh writer must
    /// roll forward from the *committed dimension*, overwriting the uncommitted
    /// slots, rather than appending past them and leaving a gap.
    ///
    /// The crashed and recovering appends deliberately write different values at
    /// the overlapping positions, so a regression that seeds the chunk count from
    /// the stale EA header surfaces the crashed writer's values rather than
    /// merely producing plausible-looking data.
    ///
    /// Restores coverage lost with the deprecated `SwmrWriter` (issue #202); the
    /// surviving `recover_and_reappend_after_clean_phase4` covers only the clean
    /// case.
    #[test]
    // Reads back with the reference HDF5 C library (`hdf5-metno`), a
    // 64-bit little-endian-only dev-dependency; skip elsewhere so the lib
    // tests still run there.
    #[cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
    fn append_inplace_recover_and_reappend_after_phase3_crash() {
        use crate::reader::File as PureFile;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let path = dir.path().join("phase3_recover.h5");
        let n = 50i32;
        build_unit_chunked(&path, n);

        // Writer 1 crashes after phase 3: the element count advances but the
        // dimension stays at `n`. Its values are far from the correct
        // continuation, so a leak is unmistakable.
        {
            let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
            s.append_inplace_i32_phased("d", &(1000..1200).collect::<Vec<_>>(), 3)
                .unwrap();
        }
        let committed: Vec<i32> = (0..n).collect();
        let pf = PureFile::from_bytes(std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(
            pf.dataset("d").unwrap().read_i32().unwrap(),
            committed,
            "phase-3 crash exposed uncommitted data to the pure reader"
        );
        {
            let f = hdf5::File::open(&path).unwrap();
            assert_eq!(
                f.dataset("d").unwrap().read_raw::<i32>().unwrap(),
                committed,
                "phase-3 crash exposed uncommitted data to the C library"
            );
            f.close().unwrap();
        }

        // Writer 2 recovers: roll forward from the committed dimension,
        // overwriting the uncommitted slots with the real continuation.
        {
            let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
            s.append_inplace_i32_phased("d", &(n..150).collect::<Vec<_>>(), 4)
                .unwrap();
        }

        let expected: Vec<i32> = (0..150).collect();
        let pf = PureFile::from_bytes(std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(
            pf.dataset("d").unwrap().read_i32().unwrap(),
            expected,
            "recovery did not roll forward correctly (pure reader)"
        );
        let f = hdf5::File::open(&path).unwrap();
        assert_eq!(
            f.dataset("d").unwrap().read_raw::<i32>().unwrap(),
            expected,
            "recovery did not roll forward correctly (C library)"
        );
        f.close().unwrap();
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
    fn a_refused_open_never_builds_the_image() {
        // The image is what may cost `O(file size)` — the mirror reads the whole
        // file — so every refusal has to come first. Asserting on the build
        // closure states that directly; measuring memory or elapsed time would
        // only correlate with it.
        use crate::writer::FileBuilder;

        let dir = tempfile::tempdir().unwrap();

        // One refusal from each family: the superblock's status flags (issue
        // #245), and an unsupported superblock version, which has always been
        // refused after the read this reorders.
        let flagged = dir.path().join("flagged.h5");
        let mut b = FileBuilder::new();
        b.create_dataset("d").with_i32_data(&[1, 2, 3]);
        b.write(&flagged).unwrap();
        let ancient = dir.path().join("ancient.h5");
        std::fs::copy(&flagged, &ancient).unwrap();
        for (path, version, flags) in [(&flagged, 3, SWMR_WRITE_FLAGS), (&ancient, 9, 0)] {
            let mut data = std::fs::read(path).unwrap();
            let off = signature::find_signature(&data).unwrap();
            let mut sb = Superblock::parse(&data, off).unwrap();
            sb.version = version;
            sb.consistency_flags = flags;
            let bytes = sb.serialize();
            data[off..off + bytes.len()].copy_from_slice(&bytes);
            std::fs::write(path, &data).unwrap();
        }

        for path in [&flagged, &ancient] {
            let built = std::cell::Cell::new(false);
            let err = match WriteEngine::open_imaged(path, Some(FileLocking::Enabled), |h, len| {
                built.set(true);
                Ok(Box::new(HandleImage::new(
                    h,
                    len,
                    MetadataCacheConfig::disabled(),
                )))
            }) {
                Err(e) => e,
                Ok(_) => panic!("{} must be refused", path.display()),
            };
            assert!(
                !built.get(),
                "{} was refused with {err:?}, but the image was built first",
                path.display()
            );
        }
    }

    #[test]
    fn a_stale_consistency_flag_is_refused_then_cleared_by_a_commit() {
        // A v3 file a crashed SWMR writer left flagged is refused by the editor
        // (issue #245) rather than edited under a writer the file still records.
        // On a v2 file, where the check is gated off to match the C library, the
        // editor opens — and the commit clears the stale flag rather than
        // re-emitting it, so the file stays properly closed for the C library
        // (issue #73).
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

        // The editor refuses it while the flag stands.
        match WriteEngine::open_with_locking(&path, FileLocking::Enabled) {
            Err(Error::FileMarkedInUse(_)) => {}
            Err(e) => panic!("expected the flag refusal, got {e:?}"),
            Ok(_) => panic!("a flagged file must not be edited in place"),
        }

        // The flag survives a *version-2* superblock, where the check is gated
        // off to match the C library — which is the one state that still carries
        // a stale flag into a commit, and so the one that keeps the healing below
        // load-bearing. (v2 and v3 superblocks share a byte layout, so restamping
        // the version is the whole difference.) A crashed C writer leaves plain
        // write access, without the SWMR bit.
        {
            let mut data = std::fs::read(&path).unwrap();
            let off = signature::find_signature(&data).unwrap();
            let mut sb = Superblock::parse(&data, off).unwrap();
            sb.version = 2;
            sb.consistency_flags = crate::file_lock::WRITE_ACCESS;
            let bytes = sb.serialize();
            data[off..off + bytes.len()].copy_from_slice(&bytes);
            std::fs::write(&path, &data).unwrap();
        }

        // A clean edit-and-commit cycle heals it.
        {
            let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled)
                .expect("the gate skips a v2 superblock, so this opens");
            let mut b = DatasetBuilder::new("e");
            b.with_i32_data(&[4, 5]);
            s.stage_created_dataset("e", b).unwrap();
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
        // VL-string dataset added via the in-place edit engine used to commit `Ok(())`
        // without ever writing its global heap collection or patching its
        // placeholder references, so the dataset failed to read back. A null
        // element (no heap object at all, distinct from an empty string) must
        // stay untouched by the patch — only heap-backed elements'
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
            let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
            let mut b = DatasetBuilder::new("labels");
            b.with_vlen_string_elements(datatype, &elements).unwrap();
            s.stage_created_dataset("labels", b).unwrap();
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
        // address that `WriteEngine::open` adds to the stored root-group address.
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

        let err = WriteEngine::open_with_locking(&path, FileLocking::Enabled)
            .err()
            .expect("open must fail");
        match err {
            Error::Format(FormatError::OffsetOverflow { offset, length }) => {
                assert_eq!(offset, u64::MAX);
                assert_eq!(length, UB);
            }
            other => panic!("expected root-group address overflow, got {other:?}"),
        }
    }

    use tempfile::tempdir;

    // -----------------------------------------------------------------------
    // Bounded sessions: the same engine over a `HandleImage`, which holds no
    // whole-file mirror. These came across from the standalone bounded engine
    // deleted in issue #198; what they cover is unchanged, but they now exercise
    // the shared code the mirror sessions use.
    // -----------------------------------------------------------------------

    /// Build a rank-1 unlimited chunked i32 dataset `d` seeded with `0..n`.
    fn build_appendable(path: &Path, n: i32, chunk: u64) {
        let data: Vec<i32> = (0..n).collect();
        let mut b = crate::writer::FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&data)
            .with_shape(&[n as u64])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[chunk]);
        b.write(path).unwrap();
    }

    fn open_bounded_session(path: &Path) -> WriteEngine {
        WriteEngine::open_rw_with_strategy(
            path,
            crate::source::MetadataCacheConfig::disabled(),
            FileLocking::Enabled,
            MemoryStrategy::Bounded,
        )
        .unwrap()
    }

    fn dataset_addr(engine: &WriteEngine) -> u64 {
        crate::group_v2::resolve_path_any_from_source(&engine.image(), engine.superblock(), "d")
            .unwrap()
    }

    /// Crash consistency on a bounded session: stop the append after only the
    /// first `max_phase` durability phases (simulating a crash at that boundary)
    /// and assert the reopened file reads either the old length (phases 1-3) or
    /// the new one (phase 4), never a torn view. Layouts cover a partial trailing
    /// chunk (relocated tail) and a chunk-aligned start.
    #[test]
    fn bounded_append_crash_consistency_partial_tail_prefix() {
        let dir = tempdir().unwrap();
        for (case, (n, chunk, add)) in [(0usize, (6i32, 4u64, 5i32)), (1, (8, 2, 6))] {
            let base = dir.path().join(std::format!("base_{case}.h5"));
            build_appendable(&base, n, chunk);
            for max_phase in 1u8..=4 {
                let p = dir.path().join(std::format!("crash_{case}_{max_phase}.h5"));
                std::fs::copy(&base, &p).unwrap();
                {
                    let mut engine = open_bounded_session(&p);
                    let addr = dataset_addr(&engine);
                    let mut b = AppendBuilder::new();
                    b.append_i32(&(n..n + add).collect::<Vec<_>>());
                    engine
                        .append_inplace_gathered(AppendTarget::Header(addr), &b, max_phase)
                        .unwrap();
                    // Dropping the engine simulates the crash: no further phases,
                    // no close barrier.
                }
                let expected_len = if max_phase == 4 { n + add } else { n };
                let got = crate::File::open(&p)
                    .unwrap()
                    .dataset("d")
                    .unwrap()
                    .read_i32()
                    .unwrap();
                assert_eq!(
                    got,
                    (0..expected_len).collect::<Vec<_>>(),
                    "case {case} phase {max_phase}"
                );
            }
        }
    }

    /// The batching loop only honors `max_phase < 4` on its first batch, and a
    /// full multi-batch append leaves every batch fully committed: after a large
    /// append the file reads the complete sequence.
    #[test]
    fn bounded_multi_batch_append_commits_every_batch() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("multibatch.h5");
        build_appendable(&p, 5, 512);
        let total = 700_000i32;
        {
            let mut engine = open_bounded_session(&p);
            let addr = dataset_addr(&engine);
            let mut b = AppendBuilder::new();
            b.append_i32(&(5..total).collect::<Vec<_>>());
            engine
                .append_inplace_gathered(AppendTarget::Header(addr), &b, 4)
                .unwrap();
        }
        let got = crate::File::open(&p)
            .unwrap()
            .dataset("d")
            .unwrap()
            .read_i32()
            .unwrap();
        assert_eq!(got.len(), total as usize);
        assert!(got.iter().enumerate().all(|(i, &v)| v == i as i32));
    }

    /// A bounded session batches; a mirror session does not. The distinction is
    /// a deliberate trade — bounded peak memory against whole-call crash
    /// atomicity — so it is asserted rather than left to the batching code's
    /// arithmetic.
    #[test]
    fn only_a_bounded_session_batches_a_large_append() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("batching.h5");
        build_appendable(&p, 8, 4);

        let bounded_batch = {
            let mut engine = open_bounded_session(&p);
            engine
                .append_geometry(AppendTarget::Path("d"))
                .unwrap()
                .full_batch_elems
        };
        let mirror_batch = {
            let mut engine = WriteEngine::open_with_locking(&p, FileLocking::Enabled).unwrap();
            engine
                .append_geometry(AppendTarget::Path("d"))
                .unwrap()
                .full_batch_elems
        };

        assert_eq!(
            mirror_batch,
            u64::MAX,
            "a mirror session must take the whole append as one crash-atomic batch"
        );
        assert!(
            bounded_batch < u64::MAX,
            "a bounded session must cap a batch, got {bounded_batch}"
        );
        assert_eq!(
            bounded_batch % 4,
            0,
            "a batch must be a whole number of chunks"
        );
    }

    /// A persisting file appended through a bounded session and dropped WITHOUT
    /// `finalize_persist` (the true-crash case) still reads back every durable
    /// append. Dropping the engine releases the exclusive lock, so the reopen is
    /// portable (no leaked lock). The finalize-at-close path is covered by the
    /// `tests/bounded_append.rs` integration tests.
    #[test]
    fn bounded_persist_append_without_finalize_is_readable() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("persist_crash.h5");
        let mut b = crate::writer::FileBuilder::new();
        b.with_file_space_strategy(crate::FileSpaceStrategy::FsmAggr, true, 1);
        b.create_dataset("d")
            .with_i32_data(&(0..6).collect::<Vec<i32>>())
            .with_shape(&[6])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[4]);
        b.write(&p).unwrap();
        {
            let mut engine = open_bounded_session(&p);
            assert!(engine.persist.is_some(), "persist state is armed at open");
            let addr = dataset_addr(&engine);
            let mut ab = AppendBuilder::new();
            ab.append_i32(&(6..20).collect::<Vec<_>>());
            engine
                .append_inplace_gathered(AppendTarget::Header(addr), &ab, 4)
                .unwrap();
            // Drop without finalizing: models a true crash and releases the lock.
        }
        let got = crate::File::open(&p)
            .unwrap()
            .dataset("d")
            .unwrap()
            .read_i32()
            .unwrap();
        assert_eq!(got, (0..20).collect::<Vec<_>>());
    }

    /// A bounded session grows a PAGED persisting file and is killed before
    /// finalize (models a crash), leaving the file non-page-aligned. Reopening it
    /// must not panic, and the next append must re-align the crashed tail page
    /// before writing raw data (so no page mixes metadata and raw); a clean close
    /// then re-page-aligns the file and every row reads back.
    #[test]
    fn bounded_paged_reopen_after_crash_realigns_and_stays_readable() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("paged_crash.h5");
        let mut b = crate::writer::FileBuilder::new();
        b.with_file_space_strategy(crate::FileSpaceStrategy::Page, true, 0)
            .with_file_space_page_size(4096);
        b.create_dataset("d")
            .with_i32_data(&(0..64).collect::<Vec<i32>>())
            .with_shape(&[64])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[64]);
        b.write(&p).unwrap();

        // Grow enough to force extensible-array index growth, so the last write of
        // the session is metadata and the tail page is a partial metadata page.
        {
            let mut engine = open_bounded_session(&p);
            let addr = dataset_addr(&engine);
            let mut ab = AppendBuilder::new();
            ab.append_i32(&(64..2000).collect::<Vec<_>>());
            engine
                .append_inplace_gathered(AppendTarget::Header(addr), &ab, 4)
                .unwrap();
            // Drop without finalize: models a crash and releases the OS lock.
        }
        assert_ne!(
            std::fs::metadata(&p).unwrap().len() % 4096,
            0,
            "a crashed (un-finalized) paged session leaves the file non-page-aligned"
        );

        // Reopen must not panic on the non-aligned file; the next append re-aligns
        // the crashed tail page, and finalize re-page-aligns the whole file.
        {
            let mut engine = open_bounded_session(&p);
            let addr = dataset_addr(&engine);
            let mut ab = AppendBuilder::new();
            ab.append_i32(&(2000..2500).collect::<Vec<_>>());
            engine
                .append_inplace_gathered(AppendTarget::Header(addr), &ab, 4)
                .unwrap();
            engine.finalize_persist().unwrap();
            engine.barrier().unwrap();
        }
        assert_eq!(
            std::fs::metadata(&p).unwrap().len() % 4096,
            0,
            "reopen + append + finalize re-aligns the paged file"
        );
        let got = crate::File::open(&p)
            .unwrap()
            .dataset("d")
            .unwrap()
            .read_i32()
            .unwrap();
        assert_eq!(got, (0..2500).collect::<Vec<_>>());
    }

    /// A staged commit on a bounded session must stay bounded: it may read the
    /// metadata it edits, but never the file's bulk. Measured rather than
    /// asserted from the design — the engine is shared with the mirror sessions
    /// now, and a single slice-taking read added anywhere on the commit path
    /// would silently make a bounded open cost as much as a mirrored one.
    #[test]
    fn a_bounded_commit_reads_far_less_than_the_file() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU64, Ordering};

        let dir = tempdir().unwrap();
        let p = dir.path().join("bulk.h5");
        // ~8 MiB of chunked data, so "reads the whole file" and "reads only the
        // metadata" differ by orders of magnitude rather than by a margin.
        let rows = 2_000_000i32;
        build_appendable(&p, rows, 8192);
        let file_len = std::fs::metadata(&p).unwrap().len();
        assert!(file_len > 4 << 20, "file is only {file_len} bytes");

        let read_bytes = Arc::new(AtomicU64::new(0));
        {
            let mut engine =
                WriteEngine::open_bounded_counting(&p, Arc::clone(&read_bytes)).unwrap();
            engine.create_group("g").unwrap();
            engine.commit().unwrap();
        }
        let read = read_bytes.load(Ordering::Relaxed);

        assert!(
            read > 0,
            "the commit read nothing, so the test proves nothing"
        );
        // Measured at 310 bytes here. The bound is loose enough to survive a
        // changed header layout and still orders of magnitude below the file.
        assert!(
            read < 64 << 10,
            "a bounded commit read {read} bytes of a {file_len}-byte file"
        );
    }

    /// An in-place append leaves a partially-filled **raw** page, so the next
    /// commit's metadata must pad it rather than pack into it.
    ///
    /// This is what keeps [`PagedEdit::begin`] reachable from [`EditStore`] now
    /// that an append allocates raw pages only: the append's job is to record that
    /// the tail page turned raw, and the commit's job is to act on it. It is also
    /// the interleaving that a single session-level page tracker makes possible —
    /// with a tracker per engine, the commit path could not see what the append
    /// path had done, which is why the whole-file editor refused an in-place append
    /// to a paged file at all (issue #198).
    #[test]
    fn a_commit_after_an_append_pads_the_raw_page_the_append_left() {
        const PAGE: u64 = 4096;
        let dir = tempdir().unwrap();
        let p = dir.path().join("paged_interleave.h5");
        let mut b = crate::writer::FileBuilder::new();
        b.with_file_space_strategy(crate::FileSpaceStrategy::Page, true, 0)
            .with_file_space_page_size(PAGE);
        b.create_dataset("d")
            .with_i32_data(&(0..64).collect::<Vec<i32>>())
            .with_shape(&[64])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[64]);
        b.write(&p).unwrap();

        let mut engine = WriteEngine::open_with_locking(&p, FileLocking::Enabled).unwrap();
        let mut ab = AppendBuilder::new();
        ab.append_i32(&(64..2000).collect::<Vec<_>>());
        engine
            .append_inplace_gathered(AppendTarget::Path("d"), &ab, 4)
            .unwrap();

        assert_eq!(
            engine.paged.as_ref().unwrap().last,
            Some(PageType::Raw),
            "the append must record that the tail page now holds raw data"
        );
        assert_ne!(
            engine.image.len() % PAGE,
            0,
            "the append must leave a partially-filled page for the commit to pad"
        );

        engine.create_group("g").unwrap();
        engine.commit().unwrap();

        // The commit padded the raw tail before laying down metadata, and folded
        // that padding into the raw list (`meta_pad`/`raw_pad` are cleared into the
        // free lists as part of the paged tail).
        let pg = engine.paged.as_ref().expect("the file is paged");
        let raw_free = pg.raw.sections();
        assert!(
            !raw_free.is_empty(),
            "the commit packed metadata into the raw page the append left open"
        );
        for (addr, len) in raw_free {
            assert_eq!(
                (addr + len) % PAGE,
                0,
                "padding {addr}+{len} does not reach a page boundary"
            );
        }

        drop(engine);
        assert_eq!(
            crate::File::open(&p)
                .unwrap()
                .dataset("d")
                .unwrap()
                .read_i32()
                .unwrap(),
            (0..2000).collect::<Vec<_>>()
        );
    }

    /// An in-place append to a paged file must allocate **only raw pages** — the
    /// chunk data and the extensible-array blocks indexing it alike.
    ///
    /// The reclaim path (`chunked_storage_spans`) reports both halves of a chunked
    /// dataset as raw free space, because that is where this crate places them. An
    /// append that put its index blocks in a metadata page instead would make the
    /// reclaim advertise metadata-page bytes for raw reuse, mixing the page a paged
    /// file exists to keep homogeneous. Measured here rather than through the
    /// reference C library, which reads a mixed-page file without complaint: an
    /// interop test proves interop and says nothing about segregation.
    #[test]
    fn an_inplace_append_to_a_paged_file_allocates_only_raw_pages() {
        const PAGE: u64 = 4096;
        let dir = tempdir().unwrap();
        let p = dir.path().join("paged_raw.h5");
        let mut b = crate::writer::FileBuilder::new();
        b.with_file_space_strategy(crate::FileSpaceStrategy::Page, true, 0)
            .with_file_space_page_size(PAGE);
        b.create_dataset("d")
            .with_i32_data(&(0..64).collect::<Vec<i32>>())
            .with_shape(&[64])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[64]);
        b.write(&p).unwrap();

        let mut engine = WriteEngine::open_with_locking(&p, FileLocking::Enabled).unwrap();
        let before = engine.image().len();
        // Two appends, each large enough to grow the extensible-array index, so the
        // run allocates index blocks as well as chunk data.
        for range in [64..2000, 2000..4000] {
            let mut ab = AppendBuilder::new();
            ab.append_i32(&range.collect::<Vec<_>>());
            engine
                .append_inplace_gathered(AppendTarget::Path("d"), &ab, 4)
                .unwrap();
        }

        let pg = engine.paged.as_ref().expect("the file is paged");
        assert_eq!(
            pg.last,
            Some(PageType::Raw),
            "the append left the tail page holding something other than raw data"
        );
        assert!(
            pg.meta_pad.is_empty() && pg.raw_pad.is_empty(),
            "an in-place append switched page type: meta_pad={:?} raw_pad={:?}",
            pg.meta_pad,
            pg.raw_pad
        );

        // Not vacuous: the append really did allocate index structure above the
        // pre-append end-of-file, which is what would have opened a metadata page.
        let addr = crate::group_v2::resolve_path_any_from_source(
            &engine.image(),
            engine.superblock(),
            "d",
        )
        .unwrap();
        let spans = engine
            .chunked_storage_spans(addr.to_usize().unwrap())
            .expect("a chunked dataset has reclaimable spans");
        let fresh = spans.iter().filter(|&&(a, _, _)| a >= before).count();
        assert!(
            fresh > 0,
            "the append allocated nothing above {before}, so the assertion above proves nothing"
        );
        assert!(
            spans.iter().all(|&(_, _, ty)| ty == PageType::Raw),
            "the reclaim tags every chunked span raw; a metadata tag here would need \
             the placement rule above to change with it"
        );
    }

    /// Write a small file of `tables` unlimited chunked datasets, paged when
    /// asked, for the write-gathering tests below.
    fn gather_fixture(path: &std::path::Path, tables: usize, paged: bool) {
        use crate::writer::FileBuilder;
        let mut b = FileBuilder::new();
        if paged {
            // Deliberately *not* DEFAULT_GATHER_PAGE: a paged fixture at the
            // default page size cannot tell a session that reads the file's page
            // size from one that assumes the default.
            b.with_file_space_strategy(FileSpaceStrategy::Page, true, 1)
                .with_file_space_page_size(16 * 1024);
        }
        for t in 0..tables {
            b.create_dataset(&std::format!("t{t}"))
                .with_i32_data(&(0..256).collect::<Vec<_>>())
                .with_shape(&[256])
                .with_maxshape(&[u64::MAX])
                .with_chunks(&[64]);
        }
        b.write(path).unwrap();
    }

    /// Run the same appends and the same commit on `session`, and report what
    /// each cost in writes: the in-place appends, then the staged commit that
    /// follows them. They are counted apart because the gathering earns its keep
    /// in only one of them — see the caller. The file the workload leaves is the
    /// other half of what the callers compare, and they read it off the path
    /// themselves.
    fn gather_workload(session: &mut WriteEngine) -> (u64, u64) {
        let before = session.image.issued_writes();
        for round in 0..4 {
            for t in 0..4 {
                session
                    .append_inplace_i32_phased(&std::format!("t{t}"), &[round; 64], 4)
                    .unwrap();
            }
        }
        for t in 0..4 {
            let mut db = crate::type_builders::DatasetBuilder::new(&std::format!("n{t}"));
            db.with_f64_data(&[2.5f64; 32]).with_shape(&[32]);
            session
                .stage_created_dataset(&std::format!("/n{t}"), db)
                .unwrap();
        }
        let after_appends = session.image.issued_writes();
        session.commit().unwrap();
        (
            after_appends - before,
            session.image.issued_writes() - after_appends,
        )
    }

    /// Gathering a session's writes lowers what it costs and changes nothing
    /// about what it produces (issue #288).
    ///
    /// Both halves matter and neither implies the other. A gatherer that dropped
    /// a run, wrote one twice in the wrong order, or filled the space between two
    /// runs sharing a page with zeros would lower the count exactly as well — so
    /// the two files are compared **byte for byte**, which is the only assertion
    /// a wrong merge cannot pass. And a gatherer that merged nothing would keep
    /// them identical, which is what the count is for.
    ///
    /// The comparison is against this same engine with the gathering turned off,
    /// rather than against a recorded number: how many writes a commit needs is
    /// an implementation detail that should be free to fall further.
    #[test]
    fn gathering_writes_costs_fewer_of_them_and_changes_no_byte() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        for paged in [false, true] {
            let straight = dir.path().join(std::format!("straight_{paged}.h5"));
            let gathered = dir.path().join(std::format!("gathered_{paged}.h5"));
            gather_fixture(&straight, 4, paged);
            gather_fixture(&gathered, 4, paged);

            let mut a = WriteEngine::open_with_locking(&straight, FileLocking::Enabled).unwrap();
            a.set_sync_policy(SyncPolicy::OnClose);
            a.image
                .set_write_buffering(WriteBuffering::Unbuffered)
                .unwrap();
            let (straight_appends, straight_commit) = gather_workload(&mut a);
            a.force_sync().unwrap();
            drop(a);

            let mut b = WriteEngine::open_with_locking(&gathered, FileLocking::Enabled).unwrap();
            b.set_sync_policy(SyncPolicy::OnClose);
            let (gathered_appends, gathered_commit) = gather_workload(&mut b);
            b.force_sync().unwrap();
            drop(b);

            // The commit tail is where the gathering earns its keep, and the half
            // to assert a ratio on. Measured: 2 writes against 10 unpaged, 4
            // against 16 paged. A commit rebuilds a group, repoints a root and
            // re-homes the free-space managers, all inside one phase and all into
            // a handful of pages, which is what merging within a barrier is for.
            // Merging *across* barriers would go further and is issue #308, held
            // back on what it would cost a crashed session.
            assert!(
                gathered_commit * 3 < straight_commit,
                "paged={paged}: gathering must cost meaningfully fewer writes for a \
                 commit, but cost {gathered_commit} against {straight_commit}"
            );
            // The appends are the other half, and since issue #307 they are a
            // near-tie: 88 against 92 unpaged, 92 against 92 paged. Publishing a
            // checksummed structure is one write from the engine now rather than
            // two the gatherer had to rejoin, so the buffering has almost nothing
            // left to merge here — before that fix this half was 92 against 144.
            // What is still worth pinning is that it never costs *more*. Note the
            // limit of that: a publish write made wider still merges the same way,
            // so widening one to the whole structure it sits in changes no count
            // here or anywhere — `a_publish_writes_from_the_byte_it_changed` is
            // what holds that, by counting bytes rather than writes.
            assert!(
                gathered_appends <= straight_appends,
                "paged={paged}: gathering must not cost more writes for the appends, \
                 but cost {gathered_appends} against {straight_appends}"
            );
            assert_eq!(
                std::fs::read(&straight).unwrap(),
                std::fs::read(&gathered).unwrap(),
                "paged={paged}: gathering changed the file it produced"
            );
        }
    }

    /// A session merges writes within the page its *file* was laid out on, and
    /// falls back to the format default only when the file is not paged.
    ///
    /// Asserted directly because the fixtures cannot assert it indirectly: the
    /// merge quantum is only visible in which writes coalesce, and any fixture
    /// built at the default page size makes the two answers identical. Reading
    /// the file's page size is the whole point of resolving this after the open
    /// rather than at it.
    #[test]
    fn the_merge_page_follows_the_file_rather_than_the_default() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let paged = dir.path().join("paged.h5");
        let plain = dir.path().join("plain.h5");
        gather_fixture(&paged, 1, true);
        gather_fixture(&plain, 1, false);

        let s = WriteEngine::open_with_locking(&paged, FileLocking::Enabled).unwrap();
        assert_eq!(
            s.gather_page_size(),
            16 * 1024,
            "a paged file merges within its own file-space page"
        );
        assert_ne!(
            16 * 1024,
            DEFAULT_GATHER_PAGE,
            "the fixture must not be built at the default, or the assertion above \
             holds for a session that ignores the file entirely"
        );
        drop(s);

        let s = WriteEngine::open_with_locking(&plain, FileLocking::Enabled).unwrap();
        assert_eq!(
            s.gather_page_size(),
            DEFAULT_GATHER_PAGE,
            "an unpaged file has no page size of its own"
        );
    }

    /// A barrier issues what has been gathered under **every** policy, so the
    /// bytes a publish point names are on the disk before the publish point is.
    ///
    /// Gathered writes are issued in address order, and the superblock lives at
    /// address 0 — so a commit whose barriers issued nothing would put its new
    /// root pointer on the disk *first* and the content it names last. A write
    /// that then failed, or a process that died mid-flush, would leave a
    /// superblock naming bytes that are not in the file, where before this
    /// gathering existed it left the previous file intact.
    ///
    /// `SyncPolicy::Always` cannot see this: its barriers are `fsync`s, which
    /// flush on their way out. Every crash-consistency test in this crate runs on
    /// that default, which is exactly why nothing caught it (issue #288). So this
    /// runs on `OnClose`, and asserts the order rather than a count — a count
    /// would be satisfied by a session that issued everything at the wrong time.
    ///
    /// It covers two of the three ordering sites: `barrier` (the commit tail) and
    /// `EditStore::sync` (the append phases). The third is `barrier_data`, which
    /// `EditStore::sync` now shares rather than duplicates, so it is correct by
    /// identity rather than by argument. It is covered too, one test over:
    /// mutating its `OnClose` arm to `Ok(())` fails
    /// `sync_policy_governs_the_persisting_and_flag_barriers` and nothing else.
    #[test]
    fn a_barrier_orders_the_publish_point_last_under_every_policy() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        for policy in [SyncPolicy::Always, SyncPolicy::OnClose] {
            let path = dir.path().join(std::format!("order_{policy:?}.h5"));
            gather_fixture(&path, 1, false);

            let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
            s.set_sync_policy(policy);

            let mut db = crate::type_builders::DatasetBuilder::new("added");
            db.with_f64_data(&[2.5f64; 64]).with_shape(&[64]);
            s.stage_created_dataset("/added", db).unwrap();
            let before = s.image.issued_write_order().len();
            s.commit().unwrap();

            let order = s.image.issued_write_order()[before..].to_vec();
            let superblock = order
                .iter()
                .position(|&(at, _)| at == s.sb_sig_off as u64)
                .unwrap_or_else(|| panic!("{policy:?}: the commit never wrote the superblock"));
            let content = order
                .iter()
                .rposition(|&(at, _)| at != s.sb_sig_off as u64)
                .unwrap_or_else(|| panic!("{policy:?}: the commit wrote nothing but a superblock"));
            assert!(
                superblock > content,
                "{policy:?}: the superblock was issued at position {superblock} of \
                 {order:?}, ahead of content at {content} — a failure in that window \
                 leaves a root pointing at bytes that are not in the file"
            );

            // The same for the append engine's four phases, whose publish point is
            // the dataspace dimension in the object header — a *low* address, with
            // the chunk bytes it makes visible at end-of-file. Stated as "the last
            // write is not the highest one", which is precisely what a single
            // address-ordered flush of the whole append would make it.
            let before = s.image.issued_write_order().len();
            s.append_inplace_i32_phased("t0", &[7; 64], 4).unwrap();
            let order = s.image.issued_write_order()[before..].to_vec();
            let highest = order
                .iter()
                .map(|&(at, _)| at)
                .max()
                .expect("the append wrote");
            assert!(
                order.last().expect("the append wrote").0 < highest,
                "{policy:?}: the append's last write is its highest-addressed one, so \
                 the whole append went out in address order and the dimension that \
                 publishes the new rows preceded the chunk bytes: {order:?}"
            );
        }
    }

    /// Gathering changes no byte through the **bounded** backing either, which is
    /// the one `File::open_rw` actually picks for a latest-format file.
    ///
    /// The test above drives the whole-file mirror, whose reads come from memory
    /// and so never meet the pending writes at all. The bounded image has no
    /// mirror: every read it serves goes to the disk and is then patched with
    /// whatever is still gathered, so it is the backing where a wrong overlay
    /// silently plans the next edit against bytes that are neither on the disk nor
    /// in the buffer — and the defect is invisible in a write *count*. Byte
    /// identity against the same session with the gathering off is the assertion
    /// a wrong overlay cannot pass.
    #[test]
    fn gathering_changes_no_byte_through_the_bounded_backing() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let run = |name: &str, gathered: bool| {
            let path = dir.path().join(name);
            gather_fixture(&path, 4, true);
            let mut engine = open_bounded_session(&path);
            engine.set_sync_policy(SyncPolicy::OnClose);
            if !gathered {
                engine
                    .image
                    .set_write_buffering(WriteBuffering::Unbuffered)
                    .unwrap();
            }
            gather_workload(&mut engine);
            engine.force_sync().unwrap();
            drop(engine);
            std::fs::read(&path).unwrap()
        };

        assert_eq!(
            run("bounded_straight.h5", false),
            run("bounded_gathered.h5", true),
            "gathering changed the file the bounded backing produced"
        );
    }

    /// An operation that has returned has put its bytes in the operating system,
    /// whatever the [`SyncPolicy`] says.
    ///
    /// This is what makes the default gathering free rather than a trade: the
    /// bytes are held only *inside* a commit or an append, never across one. It
    /// is asserted as "a forced sync afterwards finds nothing left to write",
    /// because the alternative — reading the file through a second handle — is
    /// what the session's own exclusive lock exists to prevent, mandatorily so on
    /// Windows.
    #[test]
    fn a_finished_operation_has_nothing_left_to_write() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let path = dir.path().join("finished_op.h5");
        gather_fixture(&path, 1, false);

        let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
        // The policy that issues no `fsync` at all, so the ordering barriers are
        // the only thing that can be draining the buffer.
        s.set_sync_policy(SyncPolicy::OnClose);

        let before_append = s.image.issued_writes();
        s.append_inplace_i32_phased("t0", &[7; 64], 4).unwrap();
        let after_append = s.image.issued_writes();
        assert!(
            after_append > before_append,
            "the append issued nothing at all, so the equality below holds for a \
             session that did no work"
        );
        s.force_sync().unwrap();
        assert_eq!(
            s.image.issued_writes(),
            after_append,
            "a finished append left writes in this process's memory"
        );

        let mut db = crate::type_builders::DatasetBuilder::new("added");
        db.with_f64_data(&[1.5f64; 8]).with_shape(&[8]);
        s.stage_created_dataset("/added", db).unwrap();
        s.commit().unwrap();
        let after_commit = s.image.issued_writes();
        assert!(
            after_commit > after_append,
            "the commit issued nothing at all"
        );
        s.force_sync().unwrap();
        assert_eq!(
            s.image.issued_writes(),
            after_commit,
            "a finished commit left writes in this process's memory"
        );

        // And the bytes are the ones they were meant to be — an image that issued
        // writes at the right moments but the wrong contents passes everything
        // above.
        drop(s);
        let f = crate::reader::File::open(&path).unwrap();
        assert_eq!(f.dataset("t0").unwrap().read_i32().unwrap().len(), 320);
        assert_eq!(
            f.dataset("added").unwrap().read_f64().unwrap(),
            vec![1.5f64; 8]
        );
    }

    /// Overwriting the root group is refused by name.
    ///
    /// The refusal lives in `stage_dataset_write` rather than in the commit,
    /// because the staged dataset is flattened as it is staged and the root's
    /// empty path would otherwise be reported as a dataset with no name. No
    /// public entry point can reach it — `Dataset::write_staged` runs off a
    /// resolved dataset path — so this is where it stays covered.
    #[test]
    fn overwriting_the_root_group_is_refused_at_staging() {
        use crate::writer::FileBuilder;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let path = dir.path().join("root_overwrite.h5");
        let mut b = FileBuilder::new();
        b.create_dataset("d").with_i32_data(&[1]);
        b.write(&path).unwrap();

        let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
        let mut db = crate::type_builders::DatasetBuilder::new("whatever");
        db.with_i32_data(&[1]);
        let err = s.stage_dataset_write("/", db).unwrap_err();
        assert!(
            matches!(&err, Error::EditUnsupported(m) if m.contains("root group")),
            "unexpected error: {err:?}"
        );
        assert!(!s.has_staged_edits());
    }

    /// The same obligation on the one operation that does not go through
    /// [`commit`](WriteEngine::commit): the free-space finalize a session owes at
    /// teardown.
    ///
    /// Its two callers force a sync straight after it, so nothing observable
    /// breaks when it forgets — which is exactly why it needs its own test. It is
    /// `pub(crate)`, and the next caller would inherit the omission silently.
    #[test]
    fn finalize_persist_has_nothing_left_to_write() {
        use crate::writer::FileBuilder;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let path = dir.path().join("persisting.h5");
        let mut b = FileBuilder::new();
        b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
        b.create_dataset("t0")
            .with_i32_data(&(0..256).collect::<Vec<_>>())
            .with_shape(&[256])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[64]);
        b.write(&path).unwrap();

        let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
        s.set_sync_policy(SyncPolicy::OnClose);
        // Grow the file past the managers, which is what gives the finalize
        // something to re-home.
        s.append_inplace_i32_phased("t0", &[7; 64], 4).unwrap();

        s.finalize_persist().unwrap();
        let after = s.image.issued_writes();
        s.force_sync().unwrap();
        assert_eq!(
            s.image.issued_writes(),
            after,
            "finalize_persist returned with writes still gathered"
        );
    }

    /// One in-place append costs a small constant number of writes even where
    /// nothing gathers them — the case the Extensible-Array header's six
    /// statistics were written separately for.
    ///
    /// Under gathering, six adjacent writes and one merge into the same page
    /// write, so the whole suite is blind to which one the engine made. The SWMR
    /// writer is the regime where it still shows: it gathers nothing by design,
    /// so every write the engine makes is a syscall. Stated as a ceiling on the
    /// count rather than as an exact figure, since what an append needs is free to
    /// fall — and it has: 12 before issue #307 published each checksummed
    /// structure in one write, 8 after. Six statistics written singly puts it five
    /// over; a checksum written apart from the value it covers, four.
    #[test]
    fn an_unbuffered_append_costs_a_small_constant_number_of_writes() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let path = dir.path().join("swmr_cost.h5");
        gather_fixture(&path, 1, false);

        let mut s = WriteEngine::open_swmr_writer(&path, SyncPolicy::OnClose).unwrap();
        let before = s.image.issued_writes();
        s.append_inplace_i32_phased("t0", &[7; 64], 4).unwrap();
        let cost = s.image.issued_writes() - before;

        assert!(
            cost > 0,
            "the append issued nothing, so the ceiling below proves nothing"
        );
        assert!(
            cost <= 10,
            "an unbuffered append costs {cost} writes; the array header's six \
             statistics belong in one write, not six, and each checksum belongs in \
             the write that changed what it covers (measured at 8)"
        );
    }

    /// A SWMR writer gathers nothing: its readers follow the ordered phases as
    /// they become visible, and a phase that has not reached the operating system
    /// is a phase the reader cannot see.
    ///
    /// Stopping inside the durability sequence is what makes this selective. A
    /// completed append ends with a barrier, so it would land on disk under
    /// either setting; only a stop *inside* the sequence distinguishes a writer
    /// that gathers from one that does not. The file is read through a second handle, which SWMR
    /// permits precisely because it takes no lock.
    #[test]
    fn the_swmr_writer_holds_no_write_back() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let path = dir.path().join("swmr.h5");
        gather_fixture(&path, 1, false);

        let mut s = WriteEngine::open_swmr_writer(&path, SyncPolicy::OnClose).unwrap();
        let before = std::fs::read(&path).unwrap().len();
        // Phase 1 only: the chunk bytes and the superblock's end-of-file.
        s.append_inplace_i32_phased("t0", &[7; 64], 1).unwrap();

        assert!(
            std::fs::read(&path).unwrap().len() > before,
            "a SWMR reader must see the phase-1 chunk bytes as soon as they are written"
        );
    }

    /// The `fsync` cadence belongs to whoever the fapl says: every durability
    /// point in the engine — an immediate append's ordered barriers, a commit's
    /// barrier and repoint, the barrier a commit issues after truncating, the
    /// same-length-overwrite fast path, and the barrier `close` issues — answers
    /// to the session's [`SyncPolicy`], while `force_sync` (the explicit
    /// `File::sync`) answers to nobody (issue #263).
    ///
    /// The counts are compared as a table across the two policies rather than
    /// pinned to literals: how many `fsync`s a commit costs is an implementation
    /// detail that may fall, but that `OnClose` costs *none* and `Always` costs
    /// some at each of those points is the contract. The file is read back
    /// under both, since a skipped barrier must cost durability and nothing else
    /// — the bytes have already reached the operating system.
    ///
    /// Every stage here is a *distinct* barrier site. A commit tail that syncs
    /// twice does not stand in for the fast path that syncs once, nor for the
    /// post-truncate barrier that only a shrinking commit reaches: each is its
    /// own `self.barrier()` call that a later edit could regress to a bare
    /// `self.image.sync_all()` on its own — verified by mutating each site
    /// separately and watching this fail. The persisting tails and the
    /// consistency-flag write are covered by the test below.
    ///
    /// One site is left uncovered: the version 0/1 repoint branch of the
    /// non-persisting tail. A pre-v2 file cannot be produced by this crate's own
    /// writer — the crosscheck tests get one from the C library — and the
    /// counting image is a bounded one, which such a file needs the mirror
    /// instead of. Covering it needs a checked-in fixture and a second counting
    /// opener; it is named here so the gap is a known one rather than an
    /// assumed-covered one.
    #[test]
    fn sync_policy_governs_every_barrier() {
        use crate::writer::FileBuilder;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU64, Ordering};
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        // [after an immediate append, after a staged commit, after a same-length
        // overwrite, after a commit that truncates, after the close barrier,
        // after an explicit sync].
        let run = |name: &str, policy: SyncPolicy| -> [u64; 5] {
            let path = dir.path().join(name);
            let mut b = FileBuilder::new();
            b.create_dataset("d")
                .with_i32_data(&(0..8).collect::<Vec<_>>())
                .with_shape(&[8])
                .with_maxshape(&[u64::MAX])
                .with_chunks(&[4]);
            b.write(&path).unwrap();

            let syncs = Arc::new(AtomicU64::new(0));
            let mut s = WriteEngine::open_sync_counting(&path, policy, Arc::clone(&syncs)).unwrap();
            s.append_inplace_i32_phased("d", &[8, 9, 10, 11], 4)
                .unwrap();
            let after_append = syncs.load(Ordering::Relaxed);
            let mut db = crate::type_builders::DatasetBuilder::new("added");
            db.with_f64_data(&[2.5f64; 8]).with_shape(&[8]);
            s.stage_created_dataset("/added", db).unwrap();
            s.commit().unwrap();
            let after_commit = syncs.load(Ordering::Relaxed);

            // Same-length value overwrite: the commit fast path, which patches
            // the bytes where they lie and syncs without repointing anything.
            let mut ow = crate::type_builders::DatasetBuilder::new("added");
            ow.with_f64_data(&[4.5f64; 8]).with_shape(&[8]);
            s.stage_dataset_write("/added", ow).unwrap();
            s.commit().unwrap();
            let after_overwrite = syncs.load(Ordering::Relaxed);

            // A delete whose freed run reaches end-of-file, so the commit
            // truncates and takes the barrier that only a shrinking commit does.
            s.delete("/added").unwrap();
            s.commit().unwrap();
            let after_truncate = syncs.load(Ordering::Relaxed);

            s.force_sync().unwrap();
            let after_forced = syncs.load(Ordering::Relaxed);

            // Release the exclusive OS lock before reading the file back; those
            // locks are mandatory on Windows.
            drop(s);
            let f = crate::reader::File::open(&path).unwrap();
            assert_eq!(
                f.dataset("d").unwrap().read_i32().unwrap(),
                (0..12).collect::<Vec<_>>(),
                "the append must land under {policy:?}"
            );
            assert!(
                f.dataset("added").is_err(),
                "the deleting commit must land under {policy:?}"
            );
            [
                after_append,
                after_commit,
                after_overwrite,
                after_truncate,
                after_forced,
            ]
        };

        let always = run("always.h5", SyncPolicy::Always);
        assert!(always[0] > 0, "an immediate append syncs under Always");
        assert!(always[1] > always[0], "so does a commit");
        assert!(
            always[2] > always[1],
            "so does the same-length-overwrite fast path"
        );
        assert!(
            always[3] > always[2],
            "so does a commit that truncates the file"
        );
        assert_eq!(
            always[4],
            always[3] + 1,
            "and a forced sync is exactly one more"
        );

        let deferred = run("on_close.h5", SyncPolicy::OnClose);
        assert_eq!(
            &deferred[..4],
            &[0, 0, 0, 0],
            "OnClose must leave every one of those in-session barriers unissued"
        );
        assert_eq!(
            deferred[4], 1,
            "a forced sync is issued whatever the policy says — it is what the \
             teardown path and File::sync both take"
        );
    }

    /// The two barrier sites the table above cannot reach: the commit tail of a
    /// file that *persists* its free space (a different tail, with its own
    /// barrier and repoint, plus the manager re-homing `close` owes), and the
    /// superblock consistency-flag write a SWMR session makes on open and close.
    ///
    /// Both are `self.barrier()`/`self.barrier_data()` calls on paths the
    /// ordinary non-persisting session never executes, so without this a
    /// regression at either would pass the whole suite (issue #263).
    #[test]
    fn sync_policy_governs_the_persisting_and_flag_barriers() {
        use crate::writer::FileBuilder;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU64, Ordering};
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        // [after a commit on a persisting file, after the flag write, after the
        // close barrier and its manager re-homing].
        let run = |name: &str, strategy: FileSpaceStrategy, policy: SyncPolicy| -> [u64; 3] {
            let path = dir.path().join(name);
            let mut b = FileBuilder::new();
            b.create_dataset("d")
                .with_i32_data(&(0..8).collect::<Vec<_>>())
                .with_shape(&[8])
                .with_maxshape(&[u64::MAX])
                .with_chunks(&[4]);
            b.create_dataset("victim")
                .with_f64_data(&[1.5f64; 64])
                .with_shape(&[64]);
            // A paged file takes the page-aware tail, a different pair of barrier
            // calls from the plain persisting one; `0` asks for the default page
            // size, which the plain strategy ignores.
            b.with_file_space_strategy(
                strategy,
                true,
                if strategy == FileSpaceStrategy::Page {
                    0
                } else {
                    1
                },
            );
            b.write(&path).unwrap();

            let syncs = Arc::new(AtomicU64::new(0));
            let mut s = WriteEngine::open_sync_counting(&path, policy, Arc::clone(&syncs)).unwrap();
            assert!(
                s.persist.is_some(),
                "the fixture must persist its free space, or this tests the wrong tail"
            );
            // A delete on a persisting file takes `commit_persisting`: the free
            // space is recorded on disk rather than truncated away.
            s.delete("/victim").unwrap();
            s.commit().unwrap();
            let after_commit = syncs.load(Ordering::Relaxed);

            s.set_consistency_flags(0).unwrap();
            let after_flags = syncs.load(Ordering::Relaxed);
            // The flag write's barrier orders as well as syncs, like the other
            // two. Its production callers all force a sync straight after, which
            // is why nothing else would notice it stopping — the same reason
            // `finalize_persist_has_nothing_left_to_write` exists.
            let issued = s.image.issued_writes();
            s.force_sync().unwrap();
            assert_eq!(
                s.image.issued_writes(),
                issued,
                "{policy:?}: the consistency-flag write was left gathered"
            );

            // An immediate append leaves the on-disk managers mid-file, which is
            // the debt `finalize_persist` settles with a second commit tail at
            // close — the writes no earlier `sync` can have covered.
            s.append_inplace_i32_phased("d", &[8, 9, 10, 11], 4)
                .unwrap();
            let before_close = syncs.load(Ordering::Relaxed);
            s.finalize_persist().unwrap();
            let after_close = syncs.load(Ordering::Relaxed) - before_close;

            drop(s);
            let f = crate::reader::File::open(&path).unwrap();
            assert!(
                f.dataset("victim").is_err(),
                "the delete must land under {policy:?}"
            );
            assert_eq!(
                f.dataset("d").unwrap().read_i32().unwrap(),
                (0..12).collect::<Vec<_>>(),
                "and so must the append under {policy:?}"
            );
            [after_commit, after_flags, after_close]
        };

        // Both persisting tails: the plain one and the page-aware one, which is a
        // separate pair of barrier calls reached only by a paged file.
        for (label, strategy) in [
            ("fsmaggr", FileSpaceStrategy::FsmAggr),
            ("paged", FileSpaceStrategy::Page),
        ] {
            let always = run(&format!("{label}_always.h5"), strategy, SyncPolicy::Always);
            assert!(
                always[0] > 0,
                "the {label} persisting commit tail syncs under Always"
            );
            assert!(
                always[1] > always[0],
                "so does the consistency-flag write a SWMR session makes ({label})"
            );
            assert!(
                always[2] > 0,
                "so does the manager re-homing close owes ({label})"
            );

            assert_eq!(
                run(
                    &format!("{label}_on_close.h5"),
                    strategy,
                    SyncPolicy::OnClose
                ),
                [0, 0, 0],
                "OnClose must leave the {label} persisting tail, the flag write, and the \
                 close-time manager re-homing with no fsync at all"
            );
        }
    }
}

#[cfg(test)]
mod object_header_wrap_tests {
    use super::*;

    /// A region whose messages cannot be walked is reported, not asserted away.
    ///
    /// The `debug_assert!(false)` this replaced split the behavior by build
    /// profile: a test build panicked, and a release build wrote the header with
    /// no Attribute Info message — the zero-`num_attrs` defect the normalization
    /// exists to prevent. Asserted as an `Err` because that is the one answer
    /// both profiles can give.
    #[test]
    fn an_unwalkable_region_is_refused_rather_than_wrapped() {
        // One version 2 header message — type byte, 2-byte size, flags byte —
        // whose size field claims far more body than the region holds.
        let mut region = vec![0x0Cu8]; // Attribute
        region.extend_from_slice(&0xFFFFu16.to_le_bytes());
        region.push(0); // flags
        region.extend_from_slice(&[0u8; 4]); // a body far shorter than declared

        let err = build_v2_object_header(&region).unwrap_err();
        assert!(
            matches!(err, Error::EditUnsupported(_)),
            "an unwalkable region gave {err:?}"
        );
    }

    /// The walkable case still normalizes: a header carrying inline attributes
    /// comes back with the Attribute Info message that declares their count.
    #[test]
    fn a_walkable_region_still_gains_its_attribute_info() {
        let body = [0u8; 8];
        let mut region = vec![0x0Cu8]; // Attribute
        region.extend_from_slice(&(body.len() as u16).to_le_bytes());
        region.push(0); // flags
        region.extend_from_slice(&body);

        let oh = build_v2_object_header(&region).unwrap();
        assert_eq!(&oh[..4], b"OHDR");
        assert!(
            oh.len() > 8 + region.len() + 4,
            "the wrapped header did not grow by an Attribute Info message"
        );
    }
}
