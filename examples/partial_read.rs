//! Reading a leading-dimension **row window** of a dataset instead of the whole
//! thing — [`Dataset::read_raw_rows`] and the typed `read_*_rows` helpers.
//!
//! A window `[start, start + count)` reads only the storage it overlaps: a single
//! bounded sub-read for compact/contiguous layouts, and just the chunks whose
//! first-dimension span overlaps the window for chunked layouts. Peak memory
//! scales with the window (plus one chunk), not the dataset — so a large dataset
//! can be streamed a fixed number of rows at a time. A window is byte-for-byte
//! the whole-dataset read sliced to that row range.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example partial_read
//! ```

use hdf5_pure::{File, FileBuilder};

fn main() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("frames.h5");

    // ---- A chunked 2-D dataset: 1000 rows of 4 columns, 64-row chunks ----
    let rows = 1000usize;
    let cols = 4usize;
    let data: Vec<f64> = (0..rows * cols).map(|i| i as f64).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("frames")
        .with_f64_data(&data)
        .with_shape(&[rows as u64, cols as u64])
        .with_chunks(&[64, cols as u64]);
    builder.write(&path).expect("write file");

    let file = File::open(&path).expect("open");
    let ds = file.dataset("frames").expect("open dataset");

    // ---- Read only a row window ------------------------------------------
    // Rows 100..150 (each row is 4 columns), decoded straight to `f64`.
    let window = ds.read_f64_rows(100, 50).expect("windowed read");
    assert_eq!(window.len(), 50 * cols);
    println!("read rows 100..150 -> {} elements", window.len());

    // ---- Stream the whole dataset a window at a time ---------------------
    // Only the chunks each window overlaps are decoded, so peak memory stays at
    // one window plus a chunk regardless of how large the dataset is. The final
    // window is clamped to the dataset, so it needs no special-casing.
    let n0 = ds.shape().unwrap()[0];
    let step = 128u64;
    let mut total = 0usize;
    for start in (0..n0).step_by(step as usize) {
        total += ds.read_f64_rows(start, step).unwrap().len();
    }
    assert_eq!(total, rows * cols);

    // ---- A window equals the whole read sliced to the same rows ----------
    let whole = ds.read_f64().unwrap();
    let (start, count) = (250usize, 300usize);
    let w = ds.read_f64_rows(start as u64, count as u64).unwrap();
    assert_eq!(w, whole[start * cols..(start + count) * cols]);
    println!("row window matches the whole read sliced to the same rows");

    println!("windowed read verified");
}
