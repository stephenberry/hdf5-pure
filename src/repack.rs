//! Whole-file repack (issue #21): copy an existing HDF5 file into a fresh,
//! compact one, optionally dropping objects.
//!
//! [`EditSession`](crate::EditSession) deletes objects in place but reclaims
//! space only within a session and cannot return a single deleted-and-closed
//! file's bytes to the OS. Repack is the complementary answer — the same one the
//! HDF5 C ecosystem ships as `h5repack`: it reads every surviving object and
//! rewrites the whole file from scratch through [`FileBuilder`], so the result
//! has no dead space and is strictly smaller when objects are dropped.
//!
//! # Fidelity contract
//!
//! Repack never silently degrades data. Every surviving object is reproduced
//! faithfully — datatype, shape, max-shape, chunking, filters, and byte-exact
//! element data — or the whole operation fails with [`Error::RepackUnsupported`]
//! naming the object and the reason. It refuses rather than approximate.
//! Currently reproducible:
//!
//! - Datasets with fixed-point, floating-point, time, fixed-length string,
//!   bit-field, opaque, compound, enumeration, and array datatypes,
//!   contiguous/compact or chunked.
//! - **Chunked** datasets copy their compressed chunks **verbatim** (chunk by
//!   chunk, never decoded), so *every* filter is preserved byte-exact: deflate,
//!   shuffle, fletcher32, integer **and** float scale-offset, ZFP, SZIP, and
//!   even filters this crate cannot itself apply. The destination always uses a
//!   v4 chunk index (single-chunk / fixed-array / extensible-array) regardless
//!   of the source index type.
//! - Contiguous/compact **variable-length** datasets (1D and ND): string-shaped
//!   (`is_string: true` and the MATLAB VLEN-of-1-byte-ASCII-string shape) and
//!   non-string sequences over any base type that embeds no addresses. Each
//!   element's exact heap bytes are read and re-staged through a fresh global
//!   heap, preserving charset, padding, the null-vs-empty distinction, embedded
//!   NULs, and non-UTF-8 payloads.
//! - Contiguous/compact **object-reference** datasets: each stored address is
//!   rewritten to its target object's new location in the compacted file (null
//!   and undefined references are carried verbatim).
//! - Group hierarchy of arbitrary depth.
//! - Attributes representable as [`AttrValue`] (numbers, fixed and
//!   variable-length strings and their arrays), on datasets, groups, and root.
//! - The source file's file-space management strategy (with its page size and
//!   threshold), carried into the compact output as non-persistent — a repacked
//!   file has no free space to persist.
//!
//! The verbatim chunk copy never decodes, so it eliminates the
//! decompress→recompress round-trip and the per-dataset decompression blowup,
//! and a lossy filter survives byte-exact. Two paths still re-encode and so
//! require **lossless** filters: a *contiguous/compact* filtered dataset, and a
//! *sparse* chunked dataset (one with unallocated chunk-grid holes, which the
//! dense verbatim path cannot lay out). A lossy pipeline on either of those is
//! refused.
//!
//! Refused (named, never dropped silently): chunked, filtered, or resizable
//! variable-length (string or sequence) and object-reference datasets (their
//! element references live inside compressed chunks written before the global
//! heap addresses are known, so they cannot be patched in); region references and
//! non-8-byte object references; an object reference to a dropped object or to a
//! target outside the hard-link hierarchy (a dangling, named-datatype, or region
//! target), and object references in a userblock file (non-zero base address); a
//! non-string vlen sequence whose base type embeds an address (nested vlen or
//! reference); virtual and external data layouts; a lossy filter on the
//! contiguous re-encode or sparse-chunked fallback path; and any attribute whose
//! datatype the reader cannot decode into an [`AttrValue`] (e.g. an enumeration,
//! compound, reference, or boolean attribute). An object that cannot be
//! reproduced fails the repack by name rather than being silently dropped.
//!
//! # Memory
//!
//! Repack is **out-of-core** (issue [#82]): it opens the source with
//! [`File::open_streaming`], reading metadata and one working chunk on demand
//! rather than buffering the whole file, copies each chunked dataset's
//! compressed chunks verbatim one at a time, and streams the output straight to
//! the destination. Peak memory is therefore bounded by a single chunk plus the
//! file's metadata, independent of dataset (or file) size, so a file whose data
//! exceeds available RAM repacks successfully.
//!
//! [#82]: https://github.com/stephenberry/hdf5-pure/issues/82

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;
use std::sync::Arc;

use crate::chunked_read::ChunkInfo;
use crate::chunked_write::{ChunkMeta, ChunkProvider};
use crate::convert::TryToUsize;
use crate::data_layout::DataLayout;
use crate::datatype::{Datatype, ReferenceType};
use crate::error::{Error, FormatError};
use crate::filter_pipeline::{
    FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_SCALEOFFSET, FILTER_SHUFFLE, FilterPipeline,
};
use crate::reader::{Dataset, File, Group};
use crate::scaleoffset::{self, ScaleOffset};
use crate::source::FileSource;
use crate::type_builders::{
    AttrValue, DatasetBuilder, FinishedGroup, GroupBuilder, ObjectRefTarget, VlStringElement,
};
use crate::vl_data::{VlByteObject, VlenStringReadOptions, is_vlen_string_datatype};
use crate::writer::FileBuilder;

/// Options controlling a [`repack`].
#[derive(Debug, Default, Clone)]
pub struct RepackOptions {
    /// Full paths of objects to omit from the output (e.g. `"grp/old"` or
    /// `"/grp/old"`; leading and trailing slashes are ignored). Dropping a group
    /// drops its whole subtree. Every listed path must exist in the source, or
    /// the repack fails — a no-op drop is treated as a mistake rather than
    /// silently ignored.
    pub drop: Vec<String>,
}

impl RepackOptions {
    /// Options that drop nothing — a pure compaction copy.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a path to omit from the output. Chainable.
    pub fn drop_path(mut self, path: &str) -> Self {
        self.drop.push(path.to_string());
        self
    }
}

/// Repack `src` into a new file at `dst`, applying `options`.
///
/// Reads every object of `src` not excluded by [`RepackOptions::drop`] and
/// writes them into a fresh, compact file at `dst`. On success `dst` is a normal
/// HDF5 file holding exactly the surviving objects with no dead space.
///
/// The fidelity checks run first: every object is validated while the output is
/// staged, so an [`Error::RepackUnsupported`] (an object that cannot be
/// reproduced faithfully, or a drop path that does not exist) is reported before
/// any byte is written to `dst`. Dataset *chunk bytes*, by contrast, are streamed
/// from `src` to `dst` during the write rather than buffered, so an I/O error
/// reading the source or writing the destination partway through can leave a
/// partial `dst` (remove it and retry).
///
/// See the [module documentation](self) for the exact fidelity contract.
pub fn repack<P: AsRef<Path>, Q: AsRef<Path>>(
    src: P,
    dst: Q,
    options: &RepackOptions,
) -> Result<(), Error> {
    // Open the source for on-demand streaming reads: metadata and one working
    // chunk are resident at a time, never the whole file. Shared so each streamed
    // dataset's chunk provider can pull from the same handle during the write
    // without an extra open.
    let file = Arc::new(File::open_streaming(src)?);

    // Normalize the drop set to canonical slash-free paths and remember which
    // ones actually match, so an unmatched drop can be reported as an error.
    let drop: BTreeSet<String> = options.drop.iter().map(|p| normalize(p)).collect();
    let mut matched: BTreeSet<String> = BTreeSet::new();

    let mut builder = FileBuilder::new();
    // Carry the source's file-space strategy forward. The repacked file is
    // compact with no free space, so the strategy and its page size/threshold
    // are preserved but `persist` is reset to false — there is nothing to
    // persist, and writing persistent free-space blocks is a separate feature.
    if let Some(info) = file.file_space_info() {
        builder
            .with_file_space_strategy(info.strategy, false, info.threshold)
            .with_file_space_page_size(info.page_size);
    }
    // Map every source object's (relative) header address to its path, so an
    // object-reference dataset can be rewritten to point at the same objects in
    // the compacted output rather than at their stale source addresses.
    let addr_map = build_object_address_map(&file)?;

    let root = file.root();
    populate(
        &mut builder,
        &root,
        "",
        &drop,
        &mut matched,
        &file,
        &addr_map,
    )?;

    // Every requested drop must have named a real object.
    if let Some(missing) = drop.iter().find(|d| !matched.contains(*d)) {
        return Err(Error::RepackUnsupported(format!(
            "drop path does not exist in the source: {missing}"
        )));
    }

    builder.write(dst)?;
    Ok(())
}

/// A destination that group contents can be added to. Implemented for both the
/// top-level [`FileBuilder`] (the root group) and [`GroupBuilder`] (subgroups)
/// so one recursive walk handles every level.
trait GroupSink {
    fn sink_dataset(&mut self, name: &str) -> &mut DatasetBuilder;
    fn sink_add_group(&mut self, group: FinishedGroup);
    fn sink_set_attr(&mut self, name: &str, value: AttrValue);
}

impl GroupSink for FileBuilder {
    fn sink_dataset(&mut self, name: &str) -> &mut DatasetBuilder {
        self.create_dataset(name)
    }
    fn sink_add_group(&mut self, group: FinishedGroup) {
        self.add_group(group);
    }
    fn sink_set_attr(&mut self, name: &str, value: AttrValue) {
        self.set_attr(name, value);
    }
}

impl GroupSink for GroupBuilder {
    fn sink_dataset(&mut self, name: &str) -> &mut DatasetBuilder {
        self.create_dataset(name)
    }
    fn sink_add_group(&mut self, group: FinishedGroup) {
        self.add_group(group);
    }
    fn sink_set_attr(&mut self, name: &str, value: AttrValue) {
        self.set_attr(name, value);
    }
}

/// Copy `src`'s attributes, datasets, and subgroups (recursively) into `sink`,
/// skipping anything whose path is in `drop`. `path` is the slash-free path of
/// `src` itself (empty for the root).
fn populate<S: GroupSink>(
    sink: &mut S,
    src: &Group,
    path: &str,
    drop: &BTreeSet<String>,
    matched: &mut BTreeSet<String>,
    file: &Arc<File>,
    addr_map: &HashMap<u64, String>,
) -> Result<(), Error> {
    // Attributes, in name order for a deterministic output. Refuse if any
    // attribute on this group cannot be represented (and would be dropped).
    let attrs = src.attrs()?;
    let owner = if path.is_empty() {
        "root group".to_string()
    } else {
        format!("group {path}")
    };
    check_attr_completeness(&attrs, &src.attr_names()?, &owner)?;
    for (name, value) in sorted(attrs) {
        sink.sink_set_attr(&name, value);
    }

    // Datasets, sorted by name.
    let mut dataset_names = src.datasets()?;
    dataset_names.sort();
    for name in dataset_names {
        let child_path = join(path, &name);
        if drop.contains(&child_path) {
            matched.insert(child_path);
            continue;
        }
        let ds = src.dataset(&name)?;
        emit_dataset(
            sink.sink_dataset(&name),
            &ds,
            &child_path,
            file,
            drop,
            addr_map,
        )?;
    }

    // Subgroups, sorted by name; built depth-first into a FinishedGroup.
    let mut group_names = src.groups()?;
    group_names.sort();
    for name in group_names {
        let child_path = join(path, &name);
        if drop.contains(&child_path) {
            matched.insert(child_path);
            continue;
        }
        let child = src.group(&name)?;
        let mut gb = GroupBuilder::new(&name);
        populate(&mut gb, &child, &child_path, drop, matched, file, addr_map)?;
        sink.sink_add_group(gb.finish());
    }
    Ok(())
}

/// Capture one dataset's full description and stage it on `db`, or fail with a
/// named [`Error::RepackUnsupported`] if any part cannot be reproduced.
fn emit_dataset(
    db: &mut DatasetBuilder,
    ds: &Dataset,
    path: &str,
    file: &Arc<File>,
    drop: &BTreeSet<String>,
    addr_map: &HashMap<u64, String>,
) -> Result<(), Error> {
    let datatype = ds.datatype()?;
    let dataspace = ds.dataspace()?;
    let layout = ds.data_layout()?;
    let pipeline = ds.filter_pipeline();

    check_datatype(&datatype, path)?;
    check_layout(&layout, path)?;

    let dims = dataspace.dimensions.clone();
    let n_elements: u64 = dims.iter().product();

    // Variable-length string datasets take a dedicated path: their element
    // references point into the global heap, so they are re-emitted by reading
    // each element's exact heap bytes and re-staging them, not by copying raw
    // element bytes (whose stored heap addresses would go stale on rewrite).
    if is_vlen_string_datatype(&datatype) {
        emit_vlen_string_dataset(db, ds, path, &datatype, &dims, &layout)?;
        // VL-string datasets carry attributes the same way as any other.
        let attrs = ds.attrs()?;
        check_attr_completeness(&attrs, &ds.attr_names()?, &format!("dataset {path}"))?;
        for (name, value) in sorted(attrs) {
            db.set_attr(&name, value);
        }
        return Ok(());
    }

    // Non-string variable-length (sequence) datasets take the same global-heap
    // re-staging path as VL strings: each element's exact heap bytes are read and
    // re-emitted through a fresh global heap, so the stored heap addresses are
    // rebuilt rather than copied stale. Routed here before the verbatim chunk-copy
    // path so a chunked one is refused (not copied with stale references).
    if is_nonstring_vlen(&datatype) {
        emit_vlen_sequence_dataset(db, ds, path, &datatype, &dims, &layout)?;
        let attrs = ds.attrs()?;
        check_attr_completeness(&attrs, &ds.attr_names()?, &format!("dataset {path}"))?;
        for (name, value) in sorted(attrs) {
            db.set_attr(&name, value);
        }
        return Ok(());
    }

    // Object-reference datasets store absolute object-header addresses that would
    // go stale on rewrite, so each reference is resolved to its target's *new*
    // address (via the source address->path map and the writer's path resolution)
    // rather than copied. Routed here before the verbatim chunk-copy path so a
    // chunked one is refused (not copied with stale addresses).
    if is_object_reference(&datatype) {
        emit_object_reference_dataset(db, ds, path, &dims, &layout, file, drop, addr_map)?;
        let attrs = ds.attrs()?;
        check_attr_completeness(&attrs, &ds.attr_names()?, &format!("dataset {path}"))?;
        for (name, value) in sorted(attrs) {
            db.set_attr(&name, value);
        }
        return Ok(());
    }

    // A chunked dataset with allocated chunks is copied chunk-by-chunk, verbatim:
    // each compressed chunk is laid into the output without decoding, so any
    // filter — including lossy ones (float scale-offset, ZFP) and ones this crate
    // cannot itself apply (SZIP, unknown) — is reproduced byte-exact. This avoids
    // the decompress→recompress round-trip and the whole-dataset decompression
    // blowup of the read-raw path. `check_pipeline` is intentionally skipped here:
    // never decoding makes every filter safe to carry. The datatype check above
    // still refuses time/variable-length/reference types, whose reproduction or
    // embedded addresses are unsafe even when copied verbatim.
    if let DataLayout::Chunked {
        chunk_dimensions, ..
    } = &layout
        && n_elements > 0
    {
        let rank = dims.len();
        let chunk_dims: Vec<u64> = chunk_dimensions
            .iter()
            .take(rank)
            .map(|&c| c as u64)
            .collect();

        if let Some(DenseChunkPlan { meta, grid_order }) =
            try_plan_dense_chunks(ds, &dims, &chunk_dims)?
        {
            let maxshape = dataspace
                .max_dimensions
                .as_ref()
                .filter(|ms| *ms != &dims)
                .map(|ms| ms.as_slice());
            let elem_size = datatype.type_size() as usize;
            // Stream the chunks from the source at write time rather than reading
            // them all now: the provider holds an `Arc<File>` and fetches one
            // chunk at a time, so a huge dataset never sits in memory.
            let provider = DatasetChunkProvider {
                file: Arc::clone(file),
                grid_order,
            };
            db.with_raw_chunks_lazy(
                datatype,
                &dims,
                maxshape,
                &chunk_dims,
                elem_size,
                ds.filter_pipeline_message_bytes(),
                meta,
                Box::new(provider),
            );

            // Carry the dataset's attributes, refusing any that cannot be
            // represented.
            let attrs = ds.attrs()?;
            check_attr_completeness(&attrs, &ds.attr_names()?, &format!("dataset {path}"))?;
            for (name, value) in sorted(attrs) {
                db.set_attr(&name, value);
            }
            return Ok(());
        }

        // Sparse (holes) chunked dataset: the verbatim path needs a dense grid,
        // so fall through to the read-raw + re-encode path below. That path
        // re-encodes, so it is only faithful for lossless filters; a lossy
        // pipeline on a sparse dataset is refused by `check_pipeline`.
    }

    // Contiguous/compact, or a sparse chunked dataset: read the decompressed
    // bytes and re-encode. This path can only reproduce lossless filters, so
    // refuse a lossy pipeline before reading.
    check_pipeline(pipeline.as_ref(), path)?;

    if n_elements == 0 {
        // An empty dataset owns no element bytes: carry just the datatype and
        // shape so the reconstructed dataset has the same signature.
        db.with_dtype(datatype).with_shape(&dims);
    } else {
        let raw = ds.read_raw()?;
        db.with_raw_data(datatype, raw, n_elements)
            .with_shape(&dims);
    }

    // A max-shape that differs from the current shape means a resizable dataset.
    if let Some(maxshape) = &dataspace.max_dimensions
        && maxshape != &dims
    {
        db.with_maxshape(maxshape);
    }

    // Chunking: the v3 layout appends the element size as a trailing chunk
    // dimension, so keep only the first `rank` entries; v4 already stores `rank`.
    if let DataLayout::Chunked {
        chunk_dimensions, ..
    } = &layout
    {
        let rank = dims.len();
        let logical: Vec<u64> = chunk_dimensions
            .iter()
            .take(rank)
            .map(|&c| c as u64)
            .collect();
        db.with_chunks(&logical);
    }

    // Re-apply supported filters in their stored order. `check_pipeline` has
    // already rejected anything not in this set, so the match is exhaustive.
    if let Some(p) = &pipeline {
        for f in &p.filters {
            match f.filter_id {
                FILTER_SHUFFLE => {
                    db.with_shuffle();
                }
                FILTER_FLETCHER32 => {
                    db.with_fletcher32();
                }
                FILTER_DEFLATE => {
                    // Client-data[0] is the deflate level; default to 6 if absent.
                    db.with_deflate(f.client_data.first().copied().unwrap_or(6));
                }
                FILTER_SCALEOFFSET => {
                    // `check_pipeline` guarantees integer (lossless) mode here.
                    // Re-apply with the source's minbits parameter; integer
                    // scale-offset reconstructs the exact element bytes.
                    if let Some(mode @ ScaleOffset::Integer(_)) =
                        scaleoffset::scale_offset_mode(&f.client_data)
                    {
                        db.with_scale_offset(mode);
                    } else {
                        unreachable!("check_pipeline rejected non-integer scale-offset");
                    }
                }
                _ => unreachable!("check_pipeline rejected unsupported filters"),
            }
        }
    }

    // Carry the dataset's attributes, refusing if any cannot be represented.
    let attrs = ds.attrs()?;
    check_attr_completeness(&attrs, &ds.attr_names()?, &format!("dataset {path}"))?;
    for (name, value) in sorted(attrs) {
        db.set_attr(&name, value);
    }

    Ok(())
}

/// Re-emit a variable-length string dataset faithfully: read each element's
/// exact heap bytes (preserving null-vs-empty, charset, padding, and the source
/// VL datatype shape) and re-stage them through the writer's VL-string path.
///
/// Chunked/filtered/resizable VL-string layouts are refused by name: their
/// element references live inside compressed chunks written before the global
/// heap addresses are known, so they cannot be patched in.
fn emit_vlen_string_dataset(
    db: &mut DatasetBuilder,
    ds: &Dataset,
    path: &str,
    datatype: &Datatype,
    dims: &[u64],
    layout: &DataLayout,
) -> Result<(), Error> {
    if matches!(layout, DataLayout::Chunked { .. }) {
        return Err(Error::RepackUnsupported(format!(
            "dataset {path}: chunked or filtered variable-length string datasets cannot be \
             repacked (their element references live inside compressed chunks before the global \
             heap addresses are known)"
        )));
    }
    if let Some(maxshape) = &ds.dataspace()?.max_dimensions
        && maxshape != dims
    {
        return Err(Error::RepackUnsupported(format!(
            "dataset {path}: resizable variable-length string datasets cannot be repacked"
        )));
    }

    // Read each element's exact heap bytes, preserving the null-vs-empty
    // distinction. Reading bytes (not the lossily UTF-8-decoded `String`) keeps
    // embedded NULs and non-UTF-8 payloads byte-exact.
    let objects = ds.read_vlen_string_bytes(VlenStringReadOptions::default())?;
    let elements: Vec<VlStringElement> = objects
        .into_iter()
        .map(|o| match o {
            VlByteObject::Null => VlStringElement::Null,
            VlByteObject::Bytes(bytes) => VlStringElement::Bytes(bytes),
        })
        .collect();

    // Re-stage with the exact source datatype, then set the shape. ND datasets
    // round-trip because the element references are stored row-major, matching
    // the order `read_vlen_string_bytes` returns.
    db.with_vlen_string_elements(datatype.clone(), &elements)
        .map_err(Error::Format)?;
    db.with_shape(dims);
    Ok(())
}

/// Re-emit a non-string variable-length (sequence) dataset faithfully: read each
/// element's exact heap bytes and re-stage them through a fresh global heap, so
/// the rewritten file's heap addresses are rebuilt rather than copied stale.
///
/// Chunked/filtered/resizable layouts are refused by name for the same reason as
/// VL strings: the element references live inside compressed chunks written
/// before the global heap addresses are known.
fn emit_vlen_sequence_dataset(
    db: &mut DatasetBuilder,
    ds: &Dataset,
    path: &str,
    datatype: &Datatype,
    dims: &[u64],
    layout: &DataLayout,
) -> Result<(), Error> {
    if matches!(layout, DataLayout::Chunked { .. }) {
        return Err(Error::RepackUnsupported(format!(
            "dataset {path}: chunked or filtered non-string variable-length datasets cannot be \
             repacked (their element references live inside compressed chunks before the global \
             heap addresses are known)"
        )));
    }
    if let Some(maxshape) = &ds.dataspace()?.max_dimensions
        && maxshape != dims
    {
        return Err(Error::RepackUnsupported(format!(
            "dataset {path}: resizable non-string variable-length datasets cannot be repacked"
        )));
    }

    // Read each element's exact heap bytes (preserving the null-vs-empty
    // distinction and any embedded NULs), then re-stage with the source datatype.
    let (objects, _element_size) = ds.read_vlen_sequence_bytes(VlenStringReadOptions::default())?;
    let elements: Vec<VlStringElement> = objects
        .into_iter()
        .map(|o| match o {
            VlByteObject::Null => VlStringElement::Null,
            VlByteObject::Bytes(bytes) => VlStringElement::Bytes(bytes),
        })
        .collect();

    db.with_vlen_sequence_elements(datatype.clone(), &elements)
        .map_err(Error::Format)?;
    db.with_shape(dims);
    Ok(())
}

/// A [`ChunkProvider`] that streams a dense chunked dataset's chunks from the
/// source file one at a time during the write, so repack never holds more than a
/// single chunk's bytes. Holds an `Arc<File>` (so it owns its source with no
/// borrowed lifetime) and the source [`ChunkInfo`] for each grid slot.
struct DatasetChunkProvider {
    file: Arc<File>,
    /// Source chunk descriptors in dense row-major grid order, one per slot.
    grid_order: Vec<ChunkInfo>,
}

impl ChunkProvider for DatasetChunkProvider {
    fn chunk_bytes(&self, index: usize) -> Result<Vec<u8>, FormatError> {
        // Read exactly the chunk's compressed bytes at its recorded address, with
        // no decode and no `addr_offset` adjustment — the same slice the chunked
        // reader consumes. `read_exact_at` returns exactly `chunk_size` bytes or
        // errors, and the emitter additionally checks the length against the
        // planned size, so the layout cannot silently desync from the data.
        let info = &self.grid_order[index];
        self.file
            .source()
            .read_exact_at(info.address, info.chunk_size as usize)
    }
}

/// A planned dense chunked dataset: per-chunk sizes/masks (enough to lay out the
/// destination) plus the source chunk descriptors, both in dense grid order.
struct DenseChunkPlan {
    meta: Vec<ChunkMeta>,
    grid_order: Vec<ChunkInfo>,
}

/// Plan a chunked dataset's verbatim copy without reading any chunk bytes: if
/// every chunk-grid slot is present exactly once (a dense grid), return the
/// per-chunk [`ChunkMeta`] (sizes + filter masks) and the source [`ChunkInfo`]
/// for each slot, both in dense row-major grid order. Returns `Ok(None)` when
/// the grid has holes (a sparse dataset), so the caller falls back to read-raw.
///
/// `dims` is the dataspace shape; `chunk_dims` the logical (rank-only) chunk
/// dimensions. The grid has `num_chunks_per_dim[d] = ceil(dims[d]/chunk_dims[d])`
/// slots per dimension; a chunk at N-d offset `o` maps to grid coordinate
/// `o[d]/chunk_dims[d]` and linear (row-major) index over the grid.
fn try_plan_dense_chunks(
    ds: &Dataset,
    dims: &[u64],
    chunk_dims: &[u64],
) -> Result<Option<DenseChunkPlan>, Error> {
    let rank = dims.len();
    let mut num_chunks_per_dim = Vec::with_capacity(rank);
    for d in 0..rank {
        if chunk_dims[d] == 0 {
            return Ok(None);
        }
        num_chunks_per_dim.push(dims[d].div_ceil(chunk_dims[d]));
    }
    let total: u64 = num_chunks_per_dim.iter().product();

    let infos = ds.raw_chunks()?;
    if infos.len() as u64 != total {
        // A different chunk count than the full grid means holes (or duplicates):
        // not a dense grid.
        return Ok(None);
    }

    // Map each chunk to its linear grid slot and place its descriptor; detect any
    // hole or duplicate as a non-dense grid. No chunk bytes are read here.
    let mut slots: Vec<Option<ChunkInfo>> = (0..total).map(|_| None).collect();
    for info in infos {
        if info.offsets.len() < rank {
            return Ok(None);
        }
        let mut linear: u64 = 0;
        for d in 0..rank {
            if !info.offsets[d].is_multiple_of(chunk_dims[d]) {
                return Ok(None);
            }
            let grid_coord = info.offsets[d] / chunk_dims[d];
            if grid_coord >= num_chunks_per_dim[d] {
                return Ok(None);
            }
            linear = linear * num_chunks_per_dim[d] + grid_coord;
        }
        let slot = &mut slots[linear.to_usize()?];
        if slot.is_some() {
            return Ok(None); // duplicate offset
        }
        *slot = Some(info);
    }

    // Every slot must be filled for a dense grid.
    let mut grid_order = Vec::with_capacity(slots.len());
    for slot in slots {
        match slot {
            Some(info) => grid_order.push(info),
            None => return Ok(None),
        }
    }
    let meta = grid_order
        .iter()
        .map(|info| ChunkMeta {
            compressed_size: u64::from(info.chunk_size),
            filter_mask: info.filter_mask,
        })
        .collect();
    Ok(Some(DenseChunkPlan { meta, grid_order }))
}

/// Refuse the repack if `owner` has an attribute the reader cannot represent as
/// an [`AttrValue`] and would therefore drop. `names` is every attribute on the
/// object; `decoded` is the subset that read back, keyed by name. Any name not
/// in `decoded` is an attribute that would be silently lost.
fn check_attr_completeness(
    decoded: &std::collections::HashMap<String, AttrValue>,
    names: &[String],
    owner: &str,
) -> Result<(), Error> {
    for name in names {
        if !decoded.contains_key(name) {
            return Err(Error::RepackUnsupported(format!(
                "{owner}: attribute {name:?} has a datatype that cannot be repacked faithfully yet"
            )));
        }
    }
    Ok(())
}

/// Reject datatypes whose on-disk form this crate cannot re-emit faithfully,
/// recursing into compound members, enumeration bases, and array element types so
/// a nested occurrence is caught too. Region and non-8-byte object references are
/// the remaining refusals (their stored selections/addresses are not yet
/// rewritten); 8-byte object references are handled by the reference rewrite path.
fn check_datatype(dt: &Datatype, path: &str) -> Result<(), Error> {
    let bad = |what: &str| {
        Err(Error::RepackUnsupported(format!(
            "dataset {path}: {what} datatype cannot be repacked faithfully yet"
        )))
    };
    match dt {
        // Scalar and opaque-bytes datatypes whose on-disk form `Datatype::serialize`
        // reproduces exactly (including the time type's byte order), so reading the
        // raw element bytes and re-emitting them is byte-for-byte faithful.
        Datatype::FixedPoint { .. }
        | Datatype::FloatingPoint { .. }
        | Datatype::Time { .. }
        | Datatype::String { .. }
        | Datatype::BitField { .. }
        | Datatype::Opaque { .. } => Ok(()),
        // String-shaped variable-length datatypes (`is_string: true`, or the
        // MATLAB VLEN-of-1-byte-ASCII-string shape) are reproduced by reading
        // each element's exact heap bytes and re-staging them through the
        // writer's VL-string path; the layout/filter checks gate chunked ones.
        Datatype::VariableLength { .. } if is_vlen_string_datatype(dt) => Ok(()),
        // Non-string VL (sequences of arbitrary base types) are re-staged the
        // same way, but only when the base type's bytes carry no embedded heap or
        // file addresses that a verbatim copy would leave stale.
        Datatype::VariableLength { base_type, .. } => check_vlen_base_type(base_type, path),
        // Object references (8-byte object-header addresses) are repacked by
        // rewriting each address to its target's new location. Region references
        // (which embed a dataspace selection in the global heap) and non-8-byte
        // object references are not reproduced yet.
        Datatype::Reference {
            ref_type: ReferenceType::Object,
            size: 8,
        } => Ok(()),
        Datatype::Reference {
            ref_type: ReferenceType::Object,
            ..
        } => bad("non-8-byte object reference"),
        Datatype::Reference {
            ref_type: ReferenceType::DatasetRegion,
            ..
        } => bad("dataset-region reference"),
        Datatype::Compound { members, .. } => {
            for m in members {
                check_datatype(&m.datatype, path)?;
            }
            Ok(())
        }
        Datatype::Enumeration { base_type, .. } => check_datatype(base_type, path),
        Datatype::Array { base_type, .. } => check_datatype(base_type, path),
    }
}

/// Whether `dt` is a non-string variable-length (sequence) datatype — the kind
/// re-emitted by [`emit_vlen_sequence_dataset`]. Excludes the string-shaped VL
/// datatypes, which [`emit_vlen_string_dataset`] handles.
fn is_nonstring_vlen(dt: &Datatype) -> bool {
    matches!(dt, Datatype::VariableLength { .. }) && !is_vlen_string_datatype(dt)
}

/// A non-string VL sequence is repacked by re-staging each element's exact heap
/// bytes verbatim. That is faithful only when the base type's bytes embed no
/// addresses that would go stale on rewrite: a nested variable-length type (its
/// elements are themselves global-heap references) and a reference (a stale file
/// address) are refused, recursing through compound members, array elements, and
/// enumeration bases so a nested occurrence is caught too.
fn check_vlen_base_type(dt: &Datatype, path: &str) -> Result<(), Error> {
    let bad = |what: &str| {
        Err(Error::RepackUnsupported(format!(
            "dataset {path}: variable-length sequence of {what} cannot be repacked faithfully yet"
        )))
    };
    match dt {
        Datatype::FixedPoint { .. }
        | Datatype::FloatingPoint { .. }
        | Datatype::Time { .. }
        | Datatype::String { .. }
        | Datatype::BitField { .. }
        | Datatype::Opaque { .. } => Ok(()),
        Datatype::Reference { .. } => bad("references"),
        Datatype::VariableLength { .. } => bad("variable-length elements"),
        Datatype::Compound { members, .. } => {
            for m in members {
                check_vlen_base_type(&m.datatype, path)?;
            }
            Ok(())
        }
        Datatype::Enumeration { base_type, .. } => check_vlen_base_type(base_type, path),
        Datatype::Array { base_type, .. } => check_vlen_base_type(base_type, path),
    }
}

/// Whether `dt` is an object-reference datatype handled by
/// [`emit_object_reference_dataset`].
fn is_object_reference(dt: &Datatype) -> bool {
    matches!(
        dt,
        Datatype::Reference {
            ref_type: ReferenceType::Object,
            ..
        }
    )
}

/// Whether `path` is dropped from the output: either listed in `drop`, or nested
/// under a dropped group (so its whole subtree is gone).
fn is_dropped(path: &str, drop: &BTreeSet<String>) -> bool {
    if drop.contains(path) {
        return true;
    }
    let mut p = path;
    while let Some(idx) = p.rfind('/') {
        p = &p[..idx];
        if drop.contains(p) {
            return true;
        }
    }
    false
}

/// Build a map from each source object's header address to its slash-free path,
/// for resolving object references. With a zero base address (the case object
/// references are repacked for) the stored reference value is exactly this
/// header address, so the lookup is direct.
fn build_object_address_map(file: &File) -> Result<HashMap<u64, String>, Error> {
    let mut map = HashMap::new();
    let root = file.root();
    // The root group can itself be referenced (the writer registers it under the
    // empty path).
    map.insert(root.header_address(), String::new());
    collect_addresses(&root, "", &mut map)?;
    Ok(map)
}

/// Recursively record `(header address -> path)` for every dataset and subgroup.
fn collect_addresses(
    group: &Group,
    prefix: &str,
    map: &mut HashMap<u64, String>,
) -> Result<(), Error> {
    for name in group.datasets()? {
        let ds = group.dataset(&name)?;
        map.insert(ds.header_address(), join(prefix, &name));
    }
    for name in group.groups()? {
        let child = group.group(&name)?;
        let child_path = join(prefix, &name);
        map.insert(child.header_address(), child_path.clone());
        collect_addresses(&child, &child_path, map)?;
    }
    Ok(())
}

/// Re-emit an object-reference dataset faithfully: rewrite each stored address to
/// point at its target's destination location instead of its stale source one.
///
/// Each reference is read, resolved through `addr_map` to a source path, and
/// re-staged as a path target the writer resolves once destination addresses are
/// known. Null (address 0) and undefined (`HADDR_UNDEF`) references are carried
/// verbatim. Refused by name: chunked/filtered or resizable layouts, a non-zero
/// base address, a reference to a dropped object, and a reference whose target is
/// not a hard-linked group or dataset in the source (dangling, or a named
/// datatype / region target not modelled yet).
#[allow(clippy::too_many_arguments)]
fn emit_object_reference_dataset(
    db: &mut DatasetBuilder,
    ds: &Dataset,
    path: &str,
    dims: &[u64],
    layout: &DataLayout,
    file: &Arc<File>,
    drop: &BTreeSet<String>,
    addr_map: &HashMap<u64, String>,
) -> Result<(), Error> {
    if matches!(layout, DataLayout::Chunked { .. }) {
        return Err(Error::RepackUnsupported(format!(
            "dataset {path}: chunked or filtered object-reference datasets cannot be repacked \
             (their addresses live inside compressed chunks and would need rewriting in place)"
        )));
    }
    if let Some(maxshape) = &ds.dataspace()?.max_dimensions
        && maxshape != dims
    {
        return Err(Error::RepackUnsupported(format!(
            "dataset {path}: resizable object-reference datasets cannot be repacked"
        )));
    }
    // Object references store addresses relative to the base address; the rewrite
    // path assumes a zero base (the universal case), so a userblock file is
    // refused rather than risk a mis-resolved address.
    if file.base_address() != 0 {
        return Err(Error::RepackUnsupported(format!(
            "dataset {path}: object references in a file with a non-zero base address (userblock) \
             cannot be repacked yet"
        )));
    }

    let n_elements: usize = dims.iter().product::<u64>().to_usize()?;
    let targets = if n_elements == 0 {
        Vec::new()
    } else {
        let raw = ds.read_raw()?;
        let needed = n_elements
            .checked_mul(8)
            .ok_or(FormatError::OffsetOverflow {
                offset: n_elements as u64,
                length: 8,
            })?;
        if raw.len() < needed {
            return Err(FormatError::UnexpectedEof {
                expected: needed,
                available: raw.len(),
            }
            .into());
        }
        let mut targets = Vec::with_capacity(n_elements);
        for chunk in raw[..needed].chunks_exact(8) {
            let v = u64::from_le_bytes(chunk.try_into().expect("chunks_exact(8) yields 8 bytes"));
            // 0 = null, all-ones = HADDR_UNDEF: both point at nothing, carried as-is.
            if v == 0 || v == u64::MAX {
                targets.push(ObjectRefTarget::Raw(v));
                continue;
            }
            match addr_map.get(&v) {
                Some(target_path) if is_dropped(target_path, drop) => {
                    return Err(Error::RepackUnsupported(format!(
                        "dataset {path}: object reference to dropped object {target_path:?} \
                         cannot be repacked"
                    )));
                }
                Some(target_path) => targets.push(ObjectRefTarget::Path(target_path.clone())),
                None => {
                    return Err(Error::RepackUnsupported(format!(
                        "dataset {path}: object reference to address {v:#x} resolves to no \
                         hard-linked object in the source (dangling, or a named-datatype / \
                         region target not supported yet)"
                    )));
                }
            }
        }
        targets
    };

    db.with_object_references(targets);
    db.with_shape(dims);
    Ok(())
}

/// Reject data layouts that cannot be read and re-emitted (virtual datasets;
/// contiguous/chunked with an undefined address are allowed — they are empty).
fn check_layout(layout: &DataLayout, path: &str) -> Result<(), Error> {
    match layout {
        DataLayout::Compact { .. } | DataLayout::Contiguous { .. } | DataLayout::Chunked { .. } => {
            Ok(())
        }
        DataLayout::Virtual { .. } => Err(Error::RepackUnsupported(format!(
            "dataset {path}: virtual data layout cannot be repacked"
        ))),
    }
}

/// Reject any filter that cannot be reproduced **by the re-encoding path**, so a
/// filtered dataset is never silently rewritten without its filters.
///
/// This guards only the two paths that read each dataset's *decompressed* bytes
/// and re-apply its filters from scratch: a contiguous/compact filtered dataset,
/// and the sparse-chunked fallback. A filter is safe there only when it is
/// **lossless** — then the re-encoded chunks decompress to the exact same bytes.
/// Deflate, shuffle, fletcher32, and integer scale-offset qualify. Float D-scale
/// scale-offset and ZFP are lossy: re-encoding already-decompressed values is not
/// guaranteed idempotent, so reproducing them could silently perturb the data,
/// and they are refused. SZIP this crate cannot write at all.
///
/// The dense chunked path (the common case) copies compressed chunks verbatim
/// and never calls this — there every filter is safe because nothing is decoded.
fn check_pipeline(pipeline: Option<&FilterPipeline>, path: &str) -> Result<(), Error> {
    let Some(p) = pipeline else {
        return Ok(());
    };
    for f in &p.filters {
        match f.filter_id {
            FILTER_DEFLATE | FILTER_SHUFFLE | FILTER_FLETCHER32 => {}
            FILTER_SCALEOFFSET => match scaleoffset::scale_offset_mode(&f.client_data) {
                Some(ScaleOffset::Integer(_)) => {}
                _ => {
                    return Err(Error::RepackUnsupported(format!(
                        "dataset {path}: only lossless integer scale-offset with an undefined fill value can be repacked faithfully"
                    )));
                }
            },
            other => {
                return Err(Error::RepackUnsupported(format!(
                    "dataset {path}: filter id {other} cannot be repacked yet"
                )));
            }
        }
    }
    Ok(())
}

/// Sort a name→value attribute map into a deterministic, ordered list.
fn sorted(attrs: std::collections::HashMap<String, AttrValue>) -> Vec<(String, AttrValue)> {
    attrs
        .into_iter()
        .collect::<BTreeMap<_, _>>()
        .into_iter()
        .collect()
}

/// Canonicalize a path to slash-free form: split on `/`, drop empty components,
/// rejoin. `"/a//b/"` and `"a/b"` both become `"a/b"`.
fn normalize(path: &str) -> String {
    path.split('/')
        .filter(|c| !c.is_empty())
        .collect::<Vec<_>>()
        .join("/")
}

/// Join a parent path (slash-free, possibly empty) with a child name.
fn join(parent: &str, name: &str) -> String {
    if parent.is_empty() {
        name.to_string()
    } else {
        format!("{parent}/{name}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repack_preserves_big_endian_time_dataset() {
        // The reference C library cannot create H5T_TIME, so this round-trips a
        // big-endian time dataset through our own writer and reader: repack must
        // preserve both the byte order (bf0 bit 0) and the raw element bytes.
        use crate::datatype::{Datatype, DatatypeByteOrder};
        use crate::reader::File;
        use crate::writer::FileBuilder;

        let dir = std::env::temp_dir();
        let src = dir.join("hdf5_pure_repack_time_src.h5");
        let dst = dir.join("hdf5_pure_repack_time_dst.h5");

        let dt = Datatype::Time {
            size: 4,
            byte_order: DatatypeByteOrder::BigEndian,
            bit_precision: 32,
        };
        let raw: Vec<u8> = vec![
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03,
        ];
        {
            let mut b = FileBuilder::new();
            b.create_dataset("t")
                .with_raw_data(dt.clone(), raw.clone(), 3)
                .with_shape(&[3]);
            b.write(&src).unwrap();
        }

        repack(&src, &dst, &RepackOptions::new()).unwrap();

        let f = File::open(&dst).unwrap();
        let ds = f.dataset("t").unwrap();
        assert_eq!(
            ds.datatype().unwrap(),
            dt,
            "time datatype incl. byte order must survive repack"
        );
        assert_eq!(
            ds.read_raw().unwrap(),
            raw,
            "time element bytes must be preserved"
        );

        std::fs::remove_file(&src).ok();
        std::fs::remove_file(&dst).ok();
    }

    #[test]
    fn is_dropped_matches_self_and_ancestors() {
        let drop: BTreeSet<String> = ["g/old", "lone"].iter().map(|s| s.to_string()).collect();
        // The dropped path itself.
        assert!(is_dropped("lone", &drop));
        assert!(is_dropped("g/old", &drop));
        // A descendant of a dropped group is dropped (the whole subtree goes).
        assert!(is_dropped("g/old/child", &drop));
        assert!(is_dropped("g/old/a/b", &drop));
        // Unrelated paths and partial-name collisions are not dropped.
        assert!(!is_dropped("g", &drop));
        assert!(!is_dropped("g/older", &drop));
        assert!(!is_dropped("lonely", &drop));
        assert!(!is_dropped("other/old", &drop));
    }
}
