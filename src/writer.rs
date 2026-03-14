//! Writing API: FileBuilder and GroupBuilder for creating HDF5 files.

use crate::file_writer::FileWriter as FormatWriter;
use crate::type_builders::{
    AttrValue, DatasetBuilder as FormatDatasetBuilder, FinishedGroup,
    GroupBuilder as FormatGroupBuilder,
};

use crate::error::Error;

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
