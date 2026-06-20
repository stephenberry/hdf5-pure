//! Microbenchmarks for the read and write hot paths.
//!
//! These exercise the code that dominates a real workload: decoding raw bytes
//! into typed `Vec`s, assembling chunked datasets into a dense buffer, the
//! shuffle filter, and building a file. They go through the public API
//! (`FileBuilder` / `File`) so they stay valid across internal refactors and
//! measure what a user actually pays for.
//!
//! Run with:
//!
//! ```bash
//! cargo bench --bench hot_paths
//! ```
//!
//! Each group fixes a representative dataset size; compare the reported times
//! before and after a change rather than reading absolute numbers.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use hdf5_pure::{File, FileBuilder};

/// Build a file image from a configuring closure.
fn build(configure: impl FnOnce(&mut FileBuilder)) -> Vec<u8> {
    let mut b = FileBuilder::new();
    configure(&mut b);
    b.finish().expect("serialize file")
}

// ---------------------------------------------------------------------------
// Contiguous typed decode (isolates the data_read byte->typed conversion)
// ---------------------------------------------------------------------------

fn bench_contiguous_decode(c: &mut Criterion) {
    let n: usize = 1 << 20; // ~1M elements

    let f64_bytes = build(|b| {
        let data: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
        b.create_dataset("d").with_f64_data(&data);
    });
    let i32_bytes = build(|b| {
        let data: Vec<i32> = (0..n)
            .map(|i| (i as i32).wrapping_mul(2654435761u32 as i32))
            .collect();
        b.create_dataset("d").with_i32_data(&data);
    });
    let u16_bytes = build(|b| {
        let data: Vec<u16> = (0..n).map(|i| (i & 0xFFFF) as u16).collect();
        b.create_dataset("d").with_u16_data(&data);
    });

    let f64_file = File::from_bytes(f64_bytes).unwrap();
    let i32_file = File::from_bytes(i32_bytes).unwrap();
    let u16_file = File::from_bytes(u16_bytes).unwrap();

    let mut g = c.benchmark_group("contiguous_decode_1M");
    g.bench_function("read_f64", |b| {
        b.iter(|| black_box(f64_file.dataset("d").unwrap().read_f64().unwrap()))
    });
    g.bench_function("read_i32", |b| {
        b.iter(|| black_box(i32_file.dataset("d").unwrap().read_i32().unwrap()))
    });
    g.bench_function("read_u16", |b| {
        b.iter(|| black_box(u16_file.dataset("d").unwrap().read_u16().unwrap()))
    });
    // Cross-type coercion: decode an i32-stored dataset to f64 (FixedPoint path).
    g.bench_function("read_i32_as_f64", |b| {
        b.iter(|| black_box(i32_file.dataset("d").unwrap().read_f64().unwrap()))
    });
    g.finish();
}

// ---------------------------------------------------------------------------
// Chunked assembly (isolates copy_chunk_to_output scatter into the output buf)
// ---------------------------------------------------------------------------

fn bench_chunked_assembly(c: &mut Criterion) {
    let rows = 1024u64;
    let cols = 1024u64;
    let data: Vec<f64> = (0..(rows * cols)).map(|i| i as f64).collect();

    // Uncompressed chunked: read time is dominated by chunk assembly + decode.
    let plain = build(|b| {
        b.create_dataset("d")
            .with_f64_data(&data)
            .with_shape(&[rows, cols])
            .with_chunks(&[128, 128]);
    });
    // Shuffle + deflate: adds the shuffle filter and inflate to the assembly.
    let shuf_def = build(|b| {
        b.create_dataset("d")
            .with_f64_data(&data)
            .with_shape(&[rows, cols])
            .with_chunks(&[128, 128])
            .with_shuffle()
            .with_deflate(6);
    });

    let plain_file = File::from_bytes(plain).unwrap();
    let shuf_def_file = File::from_bytes(shuf_def).unwrap();

    let mut g = c.benchmark_group("chunked_read_1024x1024_f64");
    g.bench_function("uncompressed", |b| {
        b.iter(|| black_box(plain_file.dataset("d").unwrap().read_f64().unwrap()))
    });
    g.bench_function("shuffle_deflate", |b| {
        b.iter(|| black_box(shuf_def_file.dataset("d").unwrap().read_f64().unwrap()))
    });
    g.finish();
}

// ---------------------------------------------------------------------------
// Write path
// ---------------------------------------------------------------------------

fn bench_write(c: &mut Criterion) {
    let n = 1 << 20;
    let data: Vec<f64> = (0..n).map(|i| i as f64 * 0.25).collect();
    let rows = 1024u64;
    let cols = 1024u64;
    let data2d: Vec<f64> = (0..(rows * cols)).map(|i| i as f64).collect();

    let mut g = c.benchmark_group("write_1M_f64");
    g.bench_function("contiguous", |b| {
        b.iter(|| {
            black_box(build(|bld| {
                bld.create_dataset("d").with_f64_data(black_box(&data));
            }))
        })
    });
    g.bench_function("chunked_shuffle_deflate", |b| {
        b.iter(|| {
            black_box(build(|bld| {
                bld.create_dataset("d")
                    .with_f64_data(black_box(&data2d))
                    .with_shape(&[rows, cols])
                    .with_chunks(&[128, 128])
                    .with_shuffle()
                    .with_deflate(6);
            }))
        })
    });
    g.finish();
}

// ---------------------------------------------------------------------------
// MATLAB v7.3 serde round-trip (write transpose + numeric seq decode)
// ---------------------------------------------------------------------------

/// Exercise the MAT serde path end to end: serializing a struct holding a large
/// 2-D matrix (column-major transpose) and large numeric vectors, then reading
/// it back (per-element sequence decode). This is the path `mat::to_bytes` /
/// `mat::from_bytes` users actually pay for; the low-level benches above bypass
/// it. No-op unless the `serde` feature is enabled.
#[cfg(feature = "serde")]
fn bench_mat_roundtrip(c: &mut Criterion) {
    use hdf5_pure::mat::{self, Matrix};
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize)]
    struct Payload {
        matrix: Matrix<f64>,
        samples: Vec<f64>,
        labels: Vec<i32>,
    }

    let rows = 512usize;
    let cols = 512usize;
    let matrix = Matrix::from_row_major(
        rows,
        cols,
        (0..rows * cols).map(|i| i as f64 * 0.5).collect(),
    );
    let samples: Vec<f64> = (0..(1usize << 20)).map(|i| i as f64).collect();
    let labels: Vec<i32> = (0..(1usize << 20)).map(|i| i as i32).collect();
    let payload = Payload {
        matrix,
        samples,
        labels,
    };
    let bytes = mat::to_bytes(&payload).expect("serialize payload");

    let mut g = c.benchmark_group("mat_roundtrip");
    g.bench_function("to_bytes", |b| {
        b.iter(|| black_box(mat::to_bytes(black_box(&payload)).unwrap()))
    });
    g.bench_function("from_bytes", |b| {
        b.iter(|| {
            let p: Payload = mat::from_bytes(black_box(&bytes)).unwrap();
            black_box(p)
        })
    });
    g.finish();
}

#[cfg(not(feature = "serde"))]
fn bench_mat_roundtrip(_c: &mut Criterion) {}

criterion_group!(
    benches,
    bench_contiguous_decode,
    bench_chunked_assembly,
    bench_write,
    bench_mat_roundtrip
);
criterion_main!(benches);
