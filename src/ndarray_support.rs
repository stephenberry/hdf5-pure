//! Ergonomic N-dimensional array I/O via the [`ndarray`] crate.
//!
//! Enabled by the `ndarray` feature. This module adds:
//!
//! * [`DatasetBuilder::with_ndarray`](crate::writer::FileBuilder) — write any
//!   [`ndarray`] array (owned or a view) of a supported scalar type, inferring
//!   both the dataset shape and datatype from the array.
//! * [`Dataset::read_array`](crate::Dataset::read_array) /
//!   [`Dataset::read_array_dyn`](crate::Dataset::read_array_dyn) — read a
//!   dataset back into a typed [`ndarray::Array`] (fixed rank) or
//!   [`ndarray::ArrayD`] (runtime rank).
//!
//! # Memory order
//!
//! HDF5 stores dataset elements in row-major (C) order, which is also
//! `ndarray`'s default layout, so reads and writes are a flat copy with no
//! transpose. On write, non-standard-layout inputs (Fortran-order, transposed,
//! or strided views) are repacked into row-major order via
//! [`ndarray::ArrayBase::as_standard_layout`]; arrays that are already standard
//! layout are borrowed without copying.
//!
//! # Example
//!
//! ```
//! use hdf5_pure::{File, FileBuilder};
//! use ndarray::{array, Array2};
//!
//! let a: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
//!
//! let mut fb = FileBuilder::new();
//! fb.create_dataset("m").with_ndarray(&a);
//! let bytes = fb.finish().unwrap();
//!
//! let file = File::from_bytes(bytes).unwrap();
//! let back: Array2<f64> = file.dataset("m").unwrap().read_array().unwrap();
//! assert_eq!(a, back);
//! ```

use ndarray::{Array, ArrayBase, ArrayD, Data, Dimension, IxDyn};

use crate::error::Error;
use crate::reader::Dataset;
use crate::type_builders::DatasetBuilder;

mod sealed {
    pub trait Sealed {}
}

/// A Rust scalar type that can be stored as an HDF5 dataset element.
///
/// This trait is sealed: it is implemented for the fixed set of scalar types
/// the crate's reader and writer support (`f32`, `f64`, the signed integers
/// `i8`/`i16`/`i32`/`i64`, and the unsigned integers `u8`/`u16`/`u32`/`u64`)
/// and cannot be implemented for other types. It is the element bound for the
/// [`ndarray`] read/write helpers and dispatches to the existing per-type
/// reader/writer methods, so an `ndarray` read or write has exactly the same
/// datatype, endianness, and conversion behavior as the corresponding
/// [`Dataset::read_f64`](crate::Dataset::read_f64) /
/// [`DatasetBuilder::with_f64_data`] family.
pub trait H5Element: sealed::Sealed + Copy {
    /// Read every element of `ds` as `Self`, in row-major order.
    #[doc(hidden)]
    fn read_from(ds: &Dataset<'_>) -> Result<Vec<Self>, Error>;

    /// Set `builder`'s data and datatype from a flat row-major slice.
    #[doc(hidden)]
    fn write_into(builder: &mut DatasetBuilder, data: &[Self]);
}

macro_rules! impl_h5_element {
    ($ty:ty, $read:ident, $write:ident) => {
        impl sealed::Sealed for $ty {}
        impl H5Element for $ty {
            fn read_from(ds: &Dataset<'_>) -> Result<Vec<Self>, Error> {
                ds.$read()
            }
            fn write_into(builder: &mut DatasetBuilder, data: &[Self]) {
                builder.$write(data);
            }
        }
    };
}

impl_h5_element!(f32, read_f32, with_f32_data);
impl_h5_element!(f64, read_f64, with_f64_data);
impl_h5_element!(i8, read_i8, with_i8_data);
impl_h5_element!(i16, read_i16, with_i16_data);
impl_h5_element!(i32, read_i32, with_i32_data);
impl_h5_element!(i64, read_i64, with_i64_data);
impl_h5_element!(u8, read_u8, with_u8_data);
impl_h5_element!(u16, read_u16, with_u16_data);
impl_h5_element!(u32, read_u32, with_u32_data);
impl_h5_element!(u64, read_u64, with_u64_data);

impl Dataset<'_> {
    /// Read the dataset into a statically-ranked [`ndarray::Array`].
    ///
    /// The dimensionality `D` (e.g. [`ndarray::Ix2`]) is usually inferred from
    /// the binding, so a call site reads as `let m: Array2<f64> =
    /// ds.read_array()?;`.
    ///
    /// `T` is the type you want the elements *delivered as*, not an assertion
    /// about the dataset's stored type. The stored bytes are coerced into `T`
    /// using the same rules as [`Dataset::read_f64`](crate::Dataset::read_f64)
    /// and its siblings, so the conversion can be lossy: reading an `f64`
    /// dataset as `i32` truncates, and reading an `i32` dataset as `f64`
    /// widens. There is no check that `T` matches the on-disk datatype, so pick
    /// `T` to match the stored type when you need an exact, lossless read.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] if the dataset's runtime rank does not match
    /// `D`. Use [`read_array_dyn`](Self::read_array_dyn) when the rank is not
    /// known at compile time.
    pub fn read_array<T: H5Element, D: Dimension>(&self) -> Result<Array<T, D>, Error> {
        self.read_array_dyn::<T>()?
            .into_dimensionality::<D>()
            .map_err(|e| Error::Shape(e.to_string()))
    }

    /// Read the dataset into a dynamically-ranked [`ndarray::ArrayD`].
    ///
    /// The array's shape is taken from the dataset's dataspace, so this works
    /// for any rank without naming it at compile time. As with
    /// [`read_array`](Self::read_array), `T` is the requested element type and
    /// the stored bytes are coerced into it (possibly lossily) following the
    /// [`Dataset::read_f64`](crate::Dataset::read_f64) conversion rules, with
    /// no check that `T` matches the on-disk datatype.
    pub fn read_array_dyn<T: H5Element>(&self) -> Result<ArrayD<T>, Error> {
        let shape: Vec<usize> = self.shape()?.iter().map(|&d| d as usize).collect();
        let data = T::read_from(self)?;
        ArrayD::from_shape_vec(IxDyn(&shape), data).map_err(|e| Error::Shape(e.to_string()))
    }
}

impl DatasetBuilder {
    /// Set the dataset's data, datatype, and shape from an [`ndarray`] array.
    ///
    /// Accepts any owned array or view of a supported scalar type by reference
    /// (e.g. `&Array2<f64>`, `&arr.view()`). The dataset's shape is taken from
    /// the array's shape and its datatype from the element type, so this is the
    /// N-dimensional counterpart of the flat `with_*_data` methods.
    ///
    /// Data is written in row-major (C) order. Inputs that are not already in
    /// standard layout (transposed, Fortran-order, or strided views) are
    /// repacked once into row-major order; standard-layout inputs are used
    /// without copying. The builder is returned so chunking and compression
    /// can be chained, e.g. `b.with_ndarray(&a).with_chunks(&[64, 64])`.
    pub fn with_ndarray<T, S, D>(&mut self, arr: &ArrayBase<S, D>) -> &mut Self
    where
        T: H5Element,
        S: Data<Elem = T>,
        D: Dimension,
    {
        let shape: Vec<u64> = arr.shape().iter().map(|&d| d as u64).collect();
        // `as_standard_layout` borrows when `arr` is already row-major
        // contiguous and copies into a fresh row-major buffer otherwise, so the
        // resulting slice is always the C-order data HDF5 expects.
        let standard = arr.as_standard_layout();
        let flat = standard
            .as_slice()
            .expect("as_standard_layout yields a contiguous standard-layout array");
        T::write_into(self, flat);
        self.shape = Some(shape);
        self
    }
}
