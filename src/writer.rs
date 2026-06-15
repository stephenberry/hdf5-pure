//! Writing API: FileBuilder and GroupBuilder for creating HDF5 files.

use crate::file_writer::FileWriter as FormatWriter;
use crate::type_builders::{
    AttrValue, DatasetBuilder as FormatDatasetBuilder, FinishedGroup,
    GroupBuilder as FormatGroupBuilder,
};

use crate::error::Error;
use crate::file_space_info::FileSpaceStrategy;
use crate::libver::LibVer;

/// Builder for creating a new HDF5 file.
///
/// # Example
///
/// ```no_run
/// use hdf5_pure::FileBuilder;
/// use hdf5_pure::AttrValue;
///
/// let mut builder = FileBuilder::new();
/// builder.create_dataset("data").with_f64_data(&[1.0, 2.0, 3.0]);
/// builder.set_attr("version", AttrValue::I64(1));
/// builder.write("output.h5").unwrap();
/// ```
pub struct FileBuilder {
    writer: FormatWriter,
}

impl FileBuilder {
    /// Create a new file builder.
    pub fn new() -> Self {
        Self {
            writer: FormatWriter::new(),
        }
    }

    /// Create a dataset at the root level. Returns a mutable reference to
    /// a `DatasetBuilder` for configuring data, shape, and attributes.
    pub fn create_dataset(&mut self, name: &str) -> &mut FormatDatasetBuilder {
        self.writer.create_dataset(name)
    }

    /// Create a group builder. Call `.finish()` on the returned builder
    /// to complete it, then pass to `add_group()`.
    pub fn create_group(&mut self, name: &str) -> FormatGroupBuilder {
        self.writer.create_group(name)
    }

    /// Add a finished group to the file.
    pub fn add_group(&mut self, group: FinishedGroup) {
        self.writer.add_group(group);
    }

    /// Set the userblock size in bytes. Must be a power of two >= 512 or 0 (no userblock).
    /// The userblock region is filled with zeros. After calling `finish()`, write your
    /// userblock data into `bytes[0..size]`.
    pub fn with_userblock(&mut self, size: u64) -> &mut Self {
        self.writer.with_userblock(size);
        self
    }

    /// Constrain the on-disk format version of the file, mirroring HDF5's
    /// `H5Pset_libver_bounds`. The produced file must fall within `[low, high]`,
    /// or [`finish`](Self::finish) / [`write`](Self::write) fails with
    /// [`Error::Format`] wrapping
    /// [`FormatError::LibverBoundsUnsatisfiable`](crate::FormatError::LibverBoundsUnsatisfiable).
    ///
    /// This crate writes exactly one format — the version 3 superblock from
    /// HDF5 1.10 ([`LibVer::WRITER_OUTPUT`]) — so this is a compatibility
    /// assertion, not a format selector: a bound that excludes 1.10 (an upper
    /// bound older than it, or a lower bound newer than it) is rejected.
    pub fn with_libver_bounds(&mut self, low: LibVer, high: LibVer) -> &mut Self {
        self.writer.with_libver_bounds(low, high);
        self
    }

    /// Set the file-space management strategy, mirroring HDF5's
    /// `H5Pset_file_space_strategy`. The strategy, persist flag, and free-space
    /// section `threshold` are recorded in the file's superblock extension, so
    /// the reference C library and a later reopen observe the choice.
    ///
    /// `persist = true` records that freed space should be tracked on disk across
    /// closes. A brand-new file has nothing to track, so this only records the
    /// intent; freeing space in a later [`EditSession`](crate::EditSession) then
    /// writes the on-disk free-space-manager blocks that survive a reopen.
    pub fn with_file_space_strategy(
        &mut self,
        strategy: FileSpaceStrategy,
        persist: bool,
        threshold: u64,
    ) -> &mut Self {
        self.writer
            .with_file_space_strategy(strategy, persist, threshold);
        self
    }

    /// Set the file-space page size, mirroring HDF5's
    /// `H5Pset_file_space_page_size`. Recorded in the superblock extension.
    pub fn with_file_space_page_size(&mut self, page_size: u64) -> &mut Self {
        self.writer.with_file_space_page_size(page_size);
        self
    }

    /// Set an attribute on the root group.
    pub fn set_attr(&mut self, name: &str, value: AttrValue) {
        self.writer.set_root_attr(name, value);
    }

    /// Serialize the file to bytes in memory.
    pub fn finish(self) -> Result<Vec<u8>, Error> {
        Ok(self.writer.finish()?)
    }

    /// Serialize and write the file to the given path.
    pub fn write<P: AsRef<std::path::Path>>(self, path: P) -> Result<(), Error> {
        let bytes = self.finish()?;
        std::fs::write(path, bytes).map_err(Error::Io)
    }
}

impl Default for FileBuilder {
    fn default() -> Self {
        Self::new()
    }
}
