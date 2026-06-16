//! Reading and writing datasets generically over the element type, without
//! reaching for the per-type `with_f64_data` / `read_f64` family.
//!
//! The [`H5Element`] bound is implemented for `f32`/`f64` and the 8/16/32/64-bit
//! signed and unsigned integers, so one function can serve every scalar type.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example generic_io
//! ```

use hdf5_pure::{Error, File, FileBuilder, H5Element};

/// Store a flat slice of any supported scalar. The datatype is inferred from
/// `T`, so the caller never names it.
fn store<T: H5Element>(builder: &mut FileBuilder, name: &str, values: &[T]) {
    builder.create_dataset(name).with_data(values);
}

/// Load a dataset back as `Vec<T>`. `T` is the type you want delivered, not an
/// assertion about the stored type (it coerces like `read_f64`), so pick `T` to
/// match the stored type for a lossless read.
fn load<T: H5Element>(file: &File, name: &str) -> Result<Vec<T>, Error> {
    file.dataset(name)?.read::<T>()
}

fn main() {
    let mut builder = FileBuilder::new();
    store(&mut builder, "u32s", &[1u32, 2, 3]);
    store(&mut builder, "i16s", &[-1i16, 0, 7]);
    store(&mut builder, "f64s", &[1.5f64, 2.5, 3.5]);

    let file = File::from_bytes(builder.finish().unwrap()).unwrap();

    // The element type is usually inferred from the binding...
    let counts: Vec<u32> = load(&file, "u32s").unwrap();
    let offsets: Vec<i16> = load(&file, "i16s").unwrap();
    // ...or named with turbofish.
    let readings = file.dataset("f64s").unwrap().read::<f64>().unwrap();

    println!("u32s: {counts:?}");
    println!("i16s: {offsets:?}");
    println!("f64s: {readings:?}");

    assert_eq!(counts, vec![1, 2, 3]);
    assert_eq!(offsets, vec![-1, 0, 7]);
    assert_eq!(readings, vec![1.5, 2.5, 3.5]);

    // Cross-type coercion: an i16 dataset requested as f64 widens, exactly like
    // `read_f64` on the same dataset would.
    let widened: Vec<f64> = load(&file, "i16s").unwrap();
    assert_eq!(widened, vec![-1.0, 0.0, 7.0]);
    println!("\ngeneric round-trip verified (including i16 -> f64 widening)");
}
