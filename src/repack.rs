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
//! faithfully — datatype, shape, max-shape, chunking, supported filters, and
//! byte-exact element data — or the whole operation fails with
//! [`Error::RepackUnsupported`] naming the object and the reason. It refuses
//! rather than approximate. Currently reproducible:
//!
//! - Datasets with fixed-point, floating-point, fixed-length string, compound,
//!   enumeration, and array datatypes, contiguous/compact or chunked, filtered
//!   with deflate, shuffle, and/or fletcher32.
//! - Group hierarchy of arbitrary depth.
//! - Attributes representable as [`AttrValue`] (numbers, fixed and
//!   variable-length strings and their arrays), on datasets, groups, and root.
//!
//! Refused (named, never dropped silently): variable-length, time, bitfield,
//! opaque, and reference datatypes (their on-disk representation cannot yet be
//! re-emitted faithfully — references in particular would carry stale absolute
//! addresses); virtual and external data layouts; and any filter other than
//! deflate/shuffle/fletcher32 (e.g. scale-offset, szip, zfp).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::data_layout::DataLayout;
use crate::datatype::Datatype;
use crate::error::Error;
use crate::filter_pipeline::{FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_SHUFFLE, FilterPipeline};
use crate::reader::{Dataset, File, Group};
use crate::type_builders::{AttrValue, DatasetBuilder, FinishedGroup, GroupBuilder};
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
/// HDF5 file holding exactly the surviving objects with no dead space. On any
/// [`Error::RepackUnsupported`] (an object that cannot be reproduced faithfully,
/// or a drop path that does not exist) nothing is written to `dst`: the entire
/// source is validated and staged in memory before the first byte is committed.
///
/// See the [module documentation](self) for the exact fidelity contract.
pub fn repack<P: AsRef<Path>, Q: AsRef<Path>>(
    src: P,
    dst: Q,
    options: &RepackOptions,
) -> Result<(), Error> {
    let file = File::open(src)?;

    // Normalize the drop set to canonical slash-free paths and remember which
    // ones actually match, so an unmatched drop can be reported as an error.
    let drop: BTreeSet<String> = options.drop.iter().map(|p| normalize(p)).collect();
    let mut matched: BTreeSet<String> = BTreeSet::new();

    let mut builder = FileBuilder::new();
    let root = file.root();
    populate(&mut builder, &root, "", &drop, &mut matched)?;

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
) -> Result<(), Error> {
    // Attributes, in name order for a deterministic output.
    for (name, value) in sorted(src.attrs()?) {
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
        emit_dataset(sink.sink_dataset(&name), &ds, &child_path)?;
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
        populate(&mut gb, &child, &child_path, drop, matched)?;
        sink.sink_add_group(gb.finish());
    }
    Ok(())
}

/// Capture one dataset's full description and stage it on `db`, or fail with a
/// named [`Error::RepackUnsupported`] if any part cannot be reproduced.
fn emit_dataset(db: &mut DatasetBuilder, ds: &Dataset, path: &str) -> Result<(), Error> {
    let datatype = ds.datatype()?;
    let dataspace = ds.dataspace()?;
    let layout = ds.data_layout()?;
    let pipeline = ds.filter_pipeline();

    check_datatype(&datatype, path)?;
    check_layout(&layout, path)?;
    check_pipeline(pipeline.as_ref(), path)?;

    let dims = dataspace.dimensions.clone();
    let n_elements: u64 = dims.iter().product();

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
                _ => unreachable!("check_pipeline rejected unsupported filters"),
            }
        }
    }

    Ok(())
}

/// Reject datatypes whose on-disk form this crate cannot re-serialize faithfully
/// (variable-length, time, bitfield, opaque, reference), recursing into compound
/// members, enumeration bases, and array element types so a nested occurrence is
/// caught too.
fn check_datatype(dt: &Datatype, path: &str) -> Result<(), Error> {
    let bad = |what: &str| {
        Err(Error::RepackUnsupported(format!(
            "dataset {path}: {what} datatype cannot be repacked faithfully yet"
        )))
    };
    match dt {
        Datatype::FixedPoint { .. } | Datatype::FloatingPoint { .. } | Datatype::String { .. } => {
            Ok(())
        }
        Datatype::Time { .. } => bad("time"),
        Datatype::BitField { .. } => bad("bitfield"),
        Datatype::Opaque { .. } => bad("opaque"),
        Datatype::VariableLength { .. } => bad("variable-length"),
        Datatype::Reference { .. } => bad("reference"),
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

/// Reject any filter the writer cannot reproduce, so a filtered dataset is never
/// silently rewritten without its filters.
fn check_pipeline(pipeline: Option<&FilterPipeline>, path: &str) -> Result<(), Error> {
    let Some(p) = pipeline else {
        return Ok(());
    };
    for f in &p.filters {
        match f.filter_id {
            FILTER_DEFLATE | FILTER_SHUFFLE | FILTER_FLETCHER32 => {}
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
