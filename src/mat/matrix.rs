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
//!
//! # Why three sentinel constants
//!
//! [`MATRIX_SENTINEL`], [`MATRIX_COMPLEX64_SENTINEL`], and
//! [`MATRIX_COMPLEX32_SENTINEL`] are not redundant. The element class for a
//! non-empty `Matrix<T>` can always be recovered from the serialized data
//! (the inner `Vec<T>` carries enough type information through the seq
//! unification path), so a single sentinel would suffice for the common
//! case.
//!
//! Empty matrices break that. A 0×0 / 0×N / N×0 `Matrix<Complex64>`
//! produces a `Vec<Complex64>` of length zero; the seq unification observes
//! no elements, defaults to an `f64`-empty `NumVec`, and the serializer
//! would emit a numeric (non-complex) HDF5 dataset. The dedicated
//! `Matrix<Complex64>` / `Matrix<Complex32>` sentinels carry the element
//! class through the empty path so the on-disk dataset is the correct
//! compound `{real, imag}` shape. The corresponding sealed
//! [`MatElement`] trait makes the sentinel choice a compile-time property
//! of `T`; missing dispatch surfaces as a compile error rather than a
//! silent class loss. Do not collapse the three sentinels into one without
//! re-solving the empty-shape problem.

use core::fmt;

use serde::de::{self, MapAccess, Visitor};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::mat::complex::{Complex32, Complex64};

/// Sentinel struct name recognized by the MAT serializer/deserializer for a
/// numeric `Matrix<T>` whose element class is carried by the serialized data.
pub(crate) const MATRIX_SENTINEL: &str = "__hdf5_pure_mat_Matrix__";

/// Sentinel for `Matrix<Complex64>`. Distinct from `MATRIX_SENTINEL` so an
/// empty 0×0 / 0×N / N×0 matrix still writes as a complex (compound)
/// dataset on disk: with no elements, the inner `Vec<Complex64>` would
/// otherwise collapse to a default `f64`-empty NumVec and lose the class.
pub(crate) const MATRIX_COMPLEX64_SENTINEL: &str = "__hdf5_pure_mat_MatrixComplex64__";

/// Sentinel for `Matrix<Complex32>`. See [`MATRIX_COMPLEX64_SENTINEL`].
pub(crate) const MATRIX_COMPLEX32_SENTINEL: &str = "__hdf5_pure_mat_MatrixComplex32__";

mod sealed {
    pub trait Sealed {}
}

/// Element types permitted as the parameter `T` in [`Matrix<T>`] for MAT
/// (de)serialization.
///
/// This trait is sealed: it cannot be implemented outside this crate. MAT
/// v7.3 admits only a fixed set of numeric classes (IEEE float and integer
/// primitives plus the two complex compound types), so opening the trait
/// would only let downstream code construct `Matrix<T>` values that produce
/// malformed MAT files at runtime. Sealing turns that into a compile error.
///
/// Each impl supplies a [`SENTINEL`](Self::SENTINEL) string that the MAT
/// serializer/deserializer use to (a) recognize the struct as a `Matrix<T>`
/// and (b) preserve the element class on the 0-element path, where the
/// inner data vector carries no class information of its own.
///
/// Adding a new element type means: extend the impls below AND add matching
/// dispatch arms in `mat::ser::value_ser` and `mat::de::value_de`. The
/// sentinel constant is required to (de)serialize a `Matrix<T>`, so missing
/// dispatch surfaces as a compile error rather than silently writing the
/// wrong class.
pub trait MatElement: sealed::Sealed {
    /// Sentinel struct name the MAT (de)serializer matches on.
    const SENTINEL: &'static str;
}

macro_rules! impl_mat_element {
    ($($t:ty => $sentinel:ident),* $(,)?) => {
        $(
            impl sealed::Sealed for $t {}
            impl MatElement for $t {
                const SENTINEL: &'static str = $sentinel;
            }
        )*
    };
}

impl_mat_element! {
    f64 => MATRIX_SENTINEL,
    f32 => MATRIX_SENTINEL,
    i8 => MATRIX_SENTINEL,
    i16 => MATRIX_SENTINEL,
    i32 => MATRIX_SENTINEL,
    i64 => MATRIX_SENTINEL,
    u8 => MATRIX_SENTINEL,
    u16 => MATRIX_SENTINEL,
    u32 => MATRIX_SENTINEL,
    u64 => MATRIX_SENTINEL,
    bool => MATRIX_SENTINEL,
    Complex64 => MATRIX_COMPLEX64_SENTINEL,
    Complex32 => MATRIX_COMPLEX32_SENTINEL,
}

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

impl<T: MatElement + Serialize> Serialize for Matrix<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // We use a struct with the sentinel name so the MAT serializer can
        // recognize this as a matrix and apply column-major transposition.
        // A generic serializer (serde_json, ...) sees an ordinary struct
        // whose name is `T::SENTINEL` (one of the MATRIX_*_SENTINEL constants).
        let mut s = serializer.serialize_struct(T::SENTINEL, 3)?;
        s.serialize_field("rows", &self.rows)?;
        s.serialize_field("cols", &self.cols)?;
        s.serialize_field("data", &self.data)?;
        s.end()
    }
}

impl<'de, T: MatElement + Deserialize<'de>> Deserialize<'de> for Matrix<T> {
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
            T::SENTINEL,
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
