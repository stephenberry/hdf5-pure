//! End-to-end write/read roundtrip through the full HDF5 stack with ZFP
//! compression enabled. This is the Step 4 integration test — the codec
//! itself is covered by `zfp_crosscheck.rs`; this harness checks that the
//! chunked writer, filter pipeline, and chunked reader all thread the new
//! ChunkContext through correctly.

#![cfg(feature = "zfp")]

use hdf5_pure::{File, FileBuilder};

fn max_abs_err(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .fold(0f64, f64::max)
}

#[test]
fn zfp_f32_roundtrip_rate16() {
    let vals: Vec<f32> = (0..32).map(|i| i as f32 * 0.25).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("v")
        .with_f32_data(&vals)
        .with_chunks(&[32])
        .with_zfp(16.0);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("v").unwrap();
    let back = ds.read_f32().unwrap();
    assert_eq!(back.len(), vals.len());
    // Rate 16 on f32 should give very small error for this simple ramp.
    assert!(
        max_abs_err(&vals, &back) < 0.01,
        "max_err {} > 0.01",
        max_abs_err(&vals, &back)
    );
}

#[test]
fn zfp_f64_roundtrip_rate32() {
    let vals: Vec<f64> = (0..32).map(|i| i as f64 * 0.25).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("v")
        .with_f64_data(&vals)
        .with_chunks(&[32])
        .with_zfp(32.0);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("v").unwrap();
    let back = ds.read_f64().unwrap();
    assert_eq!(back.len(), vals.len());
    let max_err = vals
        .iter()
        .zip(back.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0f64, f64::max);
    assert!(max_err < 1e-6, "max_err {max_err} > 1e-6");
}

#[test]
fn zfp_i32_roundtrip_rate32() {
    let vals: Vec<i32> = (0..16).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("v")
        .with_i32_data(&vals)
        .with_chunks(&[16])
        .with_zfp(32.0);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("v").unwrap();
    let back = ds.read_i32().unwrap();
    assert_eq!(back.len(), vals.len());
    // ZFP integer mode at max rate is not bit-identical but should be close.
    let max_err = vals
        .iter()
        .zip(back.iter())
        .map(|(&a, &b)| (a - b).abs())
        .max()
        .unwrap();
    assert!(max_err <= 2, "max_err {max_err} > 2");
}
