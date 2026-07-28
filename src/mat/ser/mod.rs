//! Serializer implementation for MATLAB v7.3 `.mat` files.

mod emit;
mod emit_with_builder;
mod root;
mod value_ser;

pub use root::{to_bytes, to_bytes_with_options, to_writer, to_writer_with_options};
