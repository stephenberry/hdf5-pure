//! Index-table + metadata-block architecture for independent parallel dataset creation.
//!
//! Based on the approach from "Parallel Data Object Creation: Scalable Metadata Management"
//! (arxiv 2506.15114). Each creator independently builds a [`MetadataBlock`] containing
//! dataset metadata. Blocks are merged into a [`MetadataIndex`] without collective
//! synchronization, enabling concurrent dataset creation.

#[cfg(not(feature = "std"))]
use alloc::{string::String, string::ToString, vec::Vec};

#[cfg(not(feature = "std"))]
extern crate alloc;

use crate::chunked_write::ChunkOptions;
use crate::dataspace::{Dataspace, DataspaceType};
use crate::datatype::Datatype;
use crate::error::FormatError;
use crate::type_builders::AttrValue;

/// A single dataset's metadata collected independently by one creator.
#[derive(Debug, Clone)]
pub struct DatasetMetadata {
    /// Dataset name (path component, not full path).
    pub name: String,
    /// HDF5 datatype descriptor.
    pub datatype: Datatype,
    /// HDF5 dataspace (shape, rank, max dimensions).
    pub dataspace: Dataspace,
    /// Chunk layout options (chunk dims, compression, etc.).
    pub chunk_options: ChunkOptions,
    /// Optional maximum shape for resizable datasets.
    pub maxshape: Option<Vec<u64>>,
    /// User-defined attributes on this dataset.
    pub attrs: Vec<(String, AttrValue)>,
    /// Raw data bytes for this dataset.
    pub raw_data: Vec<u8>,
}

/// Metadata created independently by a single creator (e.g. one thread).
///
/// In independent mode, each creator accumulates datasets into its own block.
/// Blocks are later merged into a [`MetadataIndex`].
#[derive(Debug, Clone)]
pub struct MetadataBlock {
    /// Unique identifier for the creator (e.g. thread index).
    pub creator_id: u32,
    /// Datasets defined by this creator.
    pub datasets: Vec<DatasetMetadata>,
}

impl MetadataBlock {
    /// Create a new empty metadata block for the given creator.
    pub fn new(creator_id: u32) -> Self {
        Self {
            creator_id,
            datasets: Vec::new(),
        }
    }

    /// Add a dataset to this block.
    pub fn add_dataset(&mut self, meta: DatasetMetadata) {
        self.datasets.push(meta);
    }
}

/// Creation mode for metadata blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CreationMode {
    /// All metadata in one block (traditional collective approach).
    Collective,
    /// Per-creator blocks merged at finalization (parallel independent approach).
    Independent,
}

/// Index table that maps dataset names to their metadata block locations.
///
/// After merging, each entry records which block contributed each dataset
/// and the dataset's position within the merged sequence.
#[derive(Debug, Clone)]
pub struct IndexEntry {
    /// Dataset name.
    pub name: String,
    /// Index into the flattened dataset list.
    pub dataset_index: usize,
    /// Which creator block this came from.
    pub source_block: u32,
}

/// The merged metadata index produced from one or more [`MetadataBlock`]s.
#[derive(Debug, Clone)]
pub struct MetadataIndex {
    /// Creation mode used.
    pub mode: CreationMode,
    /// Ordered index entries (sorted by name for deterministic output).
    pub entries: Vec<IndexEntry>,
    /// Flattened list of all dataset metadata, in entry order.
    pub datasets: Vec<DatasetMetadata>,
}

impl MetadataIndex {
    /// Create a collective-mode index from a single block of datasets.
    pub fn from_collective(datasets: Vec<DatasetMetadata>) -> Result<Self, FormatError> {
        let mut entries = Vec::with_capacity(datasets.len());
        for (i, ds) in datasets.iter().enumerate() {
            entries.push(IndexEntry {
                name: ds.name.clone(),
                dataset_index: i,
                source_block: 0,
            });
        }
        // Check for duplicates
        check_duplicates(&entries)?;
        Ok(Self {
            mode: CreationMode::Collective,
            entries,
            datasets,
        })
    }

    /// Merge multiple independently created metadata blocks into a single index.
    ///
    /// Returns an error if any two blocks contain datasets with the same name.
    pub fn merge_blocks(blocks: &[MetadataBlock]) -> Result<Self, FormatError> {
        let total: usize = blocks.iter().map(|b| b.datasets.len()).sum();
        let mut entries = Vec::with_capacity(total);
        let mut datasets = Vec::with_capacity(total);
        let mut idx = 0;

        for block in blocks {
            for ds in &block.datasets {
                entries.push(IndexEntry {
                    name: ds.name.clone(),
                    dataset_index: idx,
                    source_block: block.creator_id,
                });
                datasets.push(ds.clone());
                idx += 1;
            }
        }

        // Sort entries by name for deterministic file layout
        let mut order: Vec<usize> = (0..entries.len()).collect();
        order.sort_by(|&a, &b| entries[a].name.cmp(&entries[b].name));

        let sorted_entries: Vec<IndexEntry> = order
            .iter()
            .enumerate()
            .map(|(new_idx, &old_idx)| IndexEntry {
                name: entries[old_idx].name.clone(),
                dataset_index: new_idx,
                source_block: entries[old_idx].source_block,
            })
            .collect();
        let sorted_datasets: Vec<DatasetMetadata> =
            order.iter().map(|&i| datasets[i].clone()).collect();

        check_duplicates(&sorted_entries)?;

        Ok(Self {
            mode: CreationMode::Independent,
            entries: sorted_entries,
            datasets: sorted_datasets,
        })
    }

    /// Look up a dataset by name.
    pub fn find(&self, name: &str) -> Option<&DatasetMetadata> {
        self.entries
            .iter()
            .find(|e| e.name == name)
            .map(|e| &self.datasets[e.dataset_index])
    }

    /// Number of datasets in the index.
    pub fn len(&self) -> usize {
        self.datasets.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.datasets.is_empty()
    }
}

/// Helper: build a `DatasetMetadata` from common parameters.
pub fn build_dataset_metadata(
    name: &str,
    datatype: Datatype,
    shape: Vec<u64>,
    raw_data: Vec<u8>,
    chunk_options: ChunkOptions,
    maxshape: Option<Vec<u64>>,
    attrs: Vec<(String, AttrValue)>,
) -> DatasetMetadata {
    let dataspace = Dataspace {
        space_type: if shape.is_empty() {
            DataspaceType::Scalar
        } else {
            DataspaceType::Simple
        },
        rank: shape.len() as u8,
        dimensions: shape,
        max_dimensions: maxshape.clone(),
    };
    DatasetMetadata {
        name: name.to_string(),
        datatype,
        dataspace,
        chunk_options,
        maxshape,
        attrs,
        raw_data,
    }
}

fn check_duplicates(entries: &[IndexEntry]) -> Result<(), FormatError> {
    for i in 1..entries.len() {
        if entries[i].name == entries[i - 1].name {
            return Err(FormatError::DuplicateDatasetName(entries[i].name.clone()));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::type_builders::make_f64_type;

    fn sample_meta(name: &str) -> DatasetMetadata {
        build_dataset_metadata(
            name,
            make_f64_type(),
            vec![3],
            vec![0u8; 24],
            ChunkOptions::default(),
            None,
            vec![],
        )
    }

    #[test]
    fn collective_mode_basic() {
        let ds = vec![sample_meta("a"), sample_meta("b")];
        let idx = MetadataIndex::from_collective(ds).unwrap();
        assert_eq!(idx.len(), 2);
        assert_eq!(idx.mode, CreationMode::Collective);
        assert!(idx.find("a").is_some());
        assert!(idx.find("b").is_some());
        assert!(idx.find("c").is_none());
    }

    #[test]
    fn merge_two_blocks() {
        let mut b0 = MetadataBlock::new(0);
        b0.add_dataset(sample_meta("ds_a"));
        b0.add_dataset(sample_meta("ds_c"));

        let mut b1 = MetadataBlock::new(1);
        b1.add_dataset(sample_meta("ds_b"));

        let idx = MetadataIndex::merge_blocks(&[b0, b1]).unwrap();
        assert_eq!(idx.len(), 3);
        assert_eq!(idx.entries[0].name, "ds_a");
        assert_eq!(idx.entries[1].name, "ds_b");
        assert_eq!(idx.entries[2].name, "ds_c");
    }

    #[test]
    fn merge_detects_duplicates() {
        let mut b0 = MetadataBlock::new(0);
        b0.add_dataset(sample_meta("shared_name"));

        let mut b1 = MetadataBlock::new(1);
        b1.add_dataset(sample_meta("shared_name"));

        let err = MetadataIndex::merge_blocks(&[b0, b1]).unwrap_err();
        assert!(matches!(err, FormatError::DuplicateDatasetName(ref n) if n == "shared_name"));
    }

    #[test]
    fn empty_merge() {
        let idx = MetadataIndex::merge_blocks(&[]).unwrap();
        assert!(idx.is_empty());
    }

    #[test]
    fn single_block_merge() {
        let mut b = MetadataBlock::new(0);
        b.add_dataset(sample_meta("only"));
        let idx = MetadataIndex::merge_blocks(&[b]).unwrap();
        assert_eq!(idx.len(), 1);
        assert_eq!(idx.find("only").unwrap().name, "only");
    }
}
