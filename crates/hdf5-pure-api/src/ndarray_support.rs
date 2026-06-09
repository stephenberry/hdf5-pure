//! Ergonomic N-dimensional array I/O via the [`ndarray`] crate (read side).
//!
//! Enabled by the `ndarray` feature. The element trait [`H5Element`] and the
//! writer method `DatasetBuilder::with_ndarray` live in `hdf5-pure-engine` (next to
//! the builder); this module adds the reader side:
//!
//! * [`Dataset::read_array`](crate::Dataset::read_array) /
//!   [`Dataset::read_array_dyn`](crate::Dataset::read_array_dyn) — read a
//!   dataset back into a typed [`ndarray::Array`] (fixed rank) or
//!   [`ndarray::ArrayD`] (runtime rank).
//!
//! # Memory order
//!
//! HDF5 stores dataset elements in row-major (C) order, which is also
//! `ndarray`'s default layout, so reads and writes are a flat copy with no
//! transpose.
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

use ndarray::{Array, ArrayD, Dimension, IxDyn};

use crate::convert::TryToUsize;
use crate::error::Error;
use crate::reader::Dataset;

pub use hdf5_pure_engine::ndarray_support::{H5Element, ScalarSource};

// Invert the engine -> reader dependency: the engine's `H5Element::read_from`
// dispatches through `ScalarSource`, which the reader's `Dataset` implements by
// delegating to its existing per-type `read_*` methods.
impl ScalarSource for Dataset<'_> {
    fn read_f32(&self) -> Result<Vec<f32>, Error> {
        Dataset::read_f32(self)
    }
    fn read_f64(&self) -> Result<Vec<f64>, Error> {
        Dataset::read_f64(self)
    }
    fn read_i8(&self) -> Result<Vec<i8>, Error> {
        Dataset::read_i8(self)
    }
    fn read_i16(&self) -> Result<Vec<i16>, Error> {
        Dataset::read_i16(self)
    }
    fn read_i32(&self) -> Result<Vec<i32>, Error> {
        Dataset::read_i32(self)
    }
    fn read_i64(&self) -> Result<Vec<i64>, Error> {
        Dataset::read_i64(self)
    }
    fn read_u8(&self) -> Result<Vec<u8>, Error> {
        Dataset::read_u8(self)
    }
    fn read_u16(&self) -> Result<Vec<u16>, Error> {
        Dataset::read_u16(self)
    }
    fn read_u32(&self) -> Result<Vec<u32>, Error> {
        Dataset::read_u32(self)
    }
    fn read_u64(&self) -> Result<Vec<u64>, Error> {
        Dataset::read_u64(self)
    }
}

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
        let shape: Vec<usize> = self
            .shape()?
            .iter()
            .map(|&d| d.to_usize())
            .collect::<Result<Vec<_>, _>>()?;
        let data = T::read_from(self)?;
        ArrayD::from_shape_vec(IxDyn(&shape), data).map_err(|e| Error::Shape(e.to_string()))
    }
}
