//! N-dimensional array I/O via the `ndarray` crate (the `ndarray` feature).
//!
//! Shape and datatype are taken from the array, and data is stored row-major
//! (C order), matching HDF5, so reads and writes are a flat copy with no
//! transpose.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example ndarray_io --features ndarray
//! ```

use hdf5_pure::{File, FileBuilder};
use ndarray::{Array2, ArrayD, array};

fn main() {
    let m: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

    let mut builder = FileBuilder::new();
    builder.create_dataset("m").with_ndarray(&m); // shape [2, 3], f64
    let file = File::from_bytes(builder.finish().unwrap()).unwrap();

    // Read back at a statically known rank: the dimensionality is inferred from
    // the binding's type.
    let back: Array2<f64> = file.dataset("m").unwrap().read_array().unwrap();
    println!("matrix:\n{back}");
    assert_eq!(back, m);

    // Or read at a rank only known at runtime.
    let dynamic: ArrayD<f64> = file.dataset("m").unwrap().read_array_dyn().unwrap();
    println!("runtime rank: {}", dynamic.ndim());
    assert_eq!(dynamic.shape(), &[2, 3]);

    // Non-standard layouts (here a transposed view) are repacked to row-major
    // on write, so what you read back matches the logical array you passed in.
    let mut builder = FileBuilder::new();
    builder.create_dataset("mt").with_ndarray(&m.t());
    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    let transposed: Array2<f64> = file.dataset("mt").unwrap().read_array().unwrap();
    assert_eq!(transposed, m.t());
    println!("transposed view stored as shape {:?}", transposed.dim());

    println!("\nndarray round-trips verified");
}
