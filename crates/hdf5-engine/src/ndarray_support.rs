//! Engine-side support for the [`ndarray`] integration (the `ndarray` feature).
//!
//! The public surface (`hdf5_pure::H5Element`, `DatasetBuilder::with_ndarray`,
//! and `Dataset::read_array`) is split across two crates because the types it
//! touches are: the writer's [`DatasetBuilder`] lives here in `hdf5-engine`,
//! while the reader's `Dataset` lives in `hdf5-api` (above the engine).
//!
//! To keep the cycle from forming, the read direction is inverted through the
//! [`ScalarSource`] trait: this crate defines it and dispatches `H5Element`
//! reads against it, and `hdf5-api` implements it for its `Dataset` and adds
//! the `Dataset::read_array` / `read_array_dyn` methods. The write direction
//! needs no inversion: `with_ndarray` lives here, next to `DatasetBuilder`.

use ndarray::{ArrayBase, Data, Dimension};

use crate::error::Error;
use crate::type_builders::DatasetBuilder;

mod sealed {
    pub trait Sealed {}
}

/// A reader that can produce a dataset's elements as each supported scalar
/// type. Implemented by `hdf5-api`'s `Dataset` so [`H5Element::read_from`] can
/// dispatch reads without `hdf5-engine` depending on the reader crate.
#[doc(hidden)]
pub trait ScalarSource {
    fn read_f32(&self) -> Result<Vec<f32>, Error>;
    fn read_f64(&self) -> Result<Vec<f64>, Error>;
    fn read_i8(&self) -> Result<Vec<i8>, Error>;
    fn read_i16(&self) -> Result<Vec<i16>, Error>;
    fn read_i32(&self) -> Result<Vec<i32>, Error>;
    fn read_i64(&self) -> Result<Vec<i64>, Error>;
    fn read_u8(&self) -> Result<Vec<u8>, Error>;
    fn read_u16(&self) -> Result<Vec<u16>, Error>;
    fn read_u32(&self) -> Result<Vec<u32>, Error>;
    fn read_u64(&self) -> Result<Vec<u64>, Error>;
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
/// `Dataset::read_f64` / `DatasetBuilder::with_f64_data` family.
pub trait H5Element: sealed::Sealed + Copy {
    /// Read every element of `src` as `Self`, in row-major order.
    #[doc(hidden)]
    fn read_from(src: &dyn ScalarSource) -> Result<Vec<Self>, Error>;

    /// Set `builder`'s data and datatype from a flat row-major slice.
    #[doc(hidden)]
    fn write_into(builder: &mut DatasetBuilder, data: &[Self]);
}

macro_rules! impl_h5_element {
    ($ty:ty, $read:ident, $write:ident) => {
        impl sealed::Sealed for $ty {}
        impl H5Element for $ty {
            fn read_from(src: &dyn ScalarSource) -> Result<Vec<Self>, Error> {
                src.$read()
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
