//! Compound (struct-like) datasets and complex numbers.
//!
//! A compound dataset stores a record of named fields per element, like a C
//! struct or an HDF5 `H5Tcreate(H5T_COMPOUND)` type. Complex numbers are a
//! special case stored as a `{real, imag}` compound.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example compound_types
//! ```

use hdf5_pure::{File, FileBuilder};

fn main() {
    // ---- Numeric tuples as compound records -----------------------------
    // Fields are encoded one at a time, so the on-disk layout does not depend
    // on Rust's tuple layout. The element type is `(i8, u64, f32)`.
    let records = [(1i8, 20u64, 3.5f32), (2, 30, 4.5), (3, 40, 5.5)];

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("records")
        .with_compound_values(&records)
        .expect("encode compound");

    // ---- Complex numbers ------------------------------------------------
    // Stored as a compound `{real: f64, imag: f64}`, the convention MATLAB and
    // h5py use.
    let waveform = [(1.0f64, 0.0f64), (0.0, 1.0), (-1.0, 0.0)];
    builder
        .create_dataset("waveform")
        .with_complex64_data(&waveform);

    let file = File::from_bytes(builder.finish().unwrap()).unwrap();

    // Read compound records straight back into the matching tuple type. Tuple
    // fields are matched by position-derived names ("0", "1", ...), which is how
    // `with_compound_values` writes them.
    let back = file
        .dataset("records")
        .unwrap()
        .read_compound::<(i8, u64, f32)>()
        .unwrap();
    println!("records: {back:?}");
    assert_eq!(back, records);

    // The on-disk datatype carries the field names and exact byte offsets.
    let dtype = file.dataset("records").unwrap().dtype().unwrap();
    println!("records datatype: {dtype:?}");

    // Reading complex back is a rough edge worth knowing about: the fields are
    // named `real`/`imag`, not the tuple names `read_compound::<(f64, f64)>()`
    // expects, and there is no `read_complex64` convenience yet. The portable
    // path is to read the raw record bytes and decode the little-endian pairs
    // yourself (this is exactly what the crate's MATLAB reader does).
    let dtype = file.dataset("waveform").unwrap().dtype().unwrap();
    println!("waveform datatype: {dtype:?}");
    let raw = file.dataset("waveform").unwrap().read_raw().unwrap();
    let back_wave = decode_complex64(&raw);
    println!("waveform (real, imag): {back_wave:?}");
    assert_eq!(back_wave, waveform);

    println!("\ncompound and complex round-trips verified");
}

/// Decode a `{real: f64, imag: f64}` compound dataset's raw record bytes into
/// `(real, imag)` pairs (16 little-endian bytes per element).
fn decode_complex64(raw: &[u8]) -> Vec<(f64, f64)> {
    raw.chunks_exact(16)
        .map(|rec| {
            let re = f64::from_le_bytes(rec[0..8].try_into().unwrap());
            let im = f64::from_le_bytes(rec[8..16].try_into().unwrap());
            (re, im)
        })
        .collect()
}
