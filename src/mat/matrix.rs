//! Explicit 2-D matrix newtype with column-major serialization.
//!
//! Rust uses row-major ordering; MATLAB uses column-major. When you serialize
//! a [`Matrix<T>`] of shape `rows × cols`, the serializer:
//! - transposes the data to column-major byte layout
//! - stores the HDF5 dataset with shape `[cols, rows]`
//!
//! The result reads back in MATLAB as an `rows × cols` matrix of the expected
//! numeric class. Round-trip through `from_bytes` produces a `Matrix` with
//! the same Rust-side `rows`/`cols` and row-major `data`.
//!
//! `Vec<Vec<T>>` is also recognized by the serializer as a 2-D matrix
//! (checked for ragged rows). Use `Matrix` when you want an unambiguous API.

use core::fmt;

use serde::de::{self, MapAccess, Visitor};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Sentinel struct name recognized by the MAT serializer/deserializer.
pub(crate) const MATRIX_SENTINEL: &str = "__hdf5_pure_mat_Matrix__";

/// A dense 2-D matrix stored in row-major order (Rust convention).
///
/// Serializes as a MATLAB-compatible column-major HDF5 dataset.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    /// Row-major: element `(r, c)` is at index `r * cols + c`.
    data: Vec<T>,
}

impl<T> Matrix<T> {
    /// Build a `Matrix` from a row-major data vector.
    ///
    /// Panics if `data.len() != rows * cols`.
    pub fn from_row_major(rows: usize, cols: usize, data: Vec<T>) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "Matrix::from_row_major: data length {} does not match {}×{} = {}",
            data.len(),
            rows,
            cols,
            rows * cols
        );
        Self { rows, cols, data }
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Row-major flat data.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Consume the matrix and return its row-major flat data.
    pub fn into_data(self) -> Vec<T> {
        self.data
    }
}

impl<T: Clone + Default> Matrix<T> {
    /// Build a zero-initialized matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![T::default(); rows * cols],
        }
    }
}

// ---------------------------------------------------------------------------
// Serialize / Deserialize
// ---------------------------------------------------------------------------

impl<T: Serialize> Serialize for Matrix<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // We use a struct with the sentinel name so the MAT serializer can
        // recognize this as a matrix and apply column-major transposition.
        // A generic serializer (serde_json, ...) sees an ordinary struct.
        let mut s = serializer.serialize_struct(MATRIX_SENTINEL, 3)?;
        s.serialize_field("rows", &self.rows)?;
        s.serialize_field("cols", &self.cols)?;
        s.serialize_field("data", &self.data)?;
        s.end()
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Matrix<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct MatrixVisitor<T>(core::marker::PhantomData<T>);

        impl<'de, T: Deserialize<'de>> Visitor<'de> for MatrixVisitor<T> {
            type Value = Matrix<T>;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a Matrix<T> struct with fields rows, cols, data")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Matrix<T>, A::Error> {
                let mut rows: Option<usize> = None;
                let mut cols: Option<usize> = None;
                let mut data: Option<Vec<T>> = None;
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "rows" => rows = Some(map.next_value()?),
                        "cols" => cols = Some(map.next_value()?),
                        "data" => data = Some(map.next_value()?),
                        _ => {
                            let _: serde::de::IgnoredAny = map.next_value()?;
                        }
                    }
                }
                let rows = rows.ok_or_else(|| de::Error::missing_field("rows"))?;
                let cols = cols.ok_or_else(|| de::Error::missing_field("cols"))?;
                let data = data.ok_or_else(|| de::Error::missing_field("data"))?;
                if data.len() != rows * cols {
                    return Err(de::Error::custom(format!(
                        "Matrix data length {} does not match {}×{} = {}",
                        data.len(),
                        rows,
                        cols,
                        rows * cols
                    )));
                }
                Ok(Matrix { rows, cols, data })
            }
        }

        deserializer.deserialize_struct(
            MATRIX_SENTINEL,
            &["rows", "cols", "data"],
            MatrixVisitor(core::marker::PhantomData),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn from_row_major_length_mismatch_panics() {
        let _ = Matrix::from_row_major(2, 3, vec![1.0_f64]);
    }
}
