//! Serializer implementation for MATLAB v7.3 `.mat` files.

mod emit;
mod root;
mod value_ser;

pub use root::to_bytes;
