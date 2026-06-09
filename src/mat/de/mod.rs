//! Deserializer for MATLAB v7.3 `.mat` files.

mod reader;
mod value_de;

use serde::de::DeserializeOwned;

use crate::mat::error::MatError;
use crate::mat::value::MatValue;

use value_de::MatValueDeserializer;

/// Deserialize a MAT v7.3 file (or any MATLAB-compatible HDF5 file) into `T`.
///
/// The root is treated as a struct whose fields are the top-level datasets
/// and groups in the file.
pub fn from_bytes<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, MatError> {
    let fields = reader::read_file(bytes)?;
    let root = MatValue::Struct(fields);
    T::deserialize(MatValueDeserializer::new(root))
}
