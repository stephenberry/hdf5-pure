//! End-to-end write/read roundtrip through the full HDF5 stack with ZFP
//! compression enabled. This is the Step 4 integration test — the codec
//! itself is covered by `zfp_crosscheck.rs`; this harness checks that the
//! chunked writer, filter pipeline, and chunked reader all thread the new
//! ChunkContext through correctly.

#![cfg(feature = "zfp")]

use hdf5_pure::{Error, File, FileBuilder, FormatError};

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

#[test]
fn zfp_f32_roundtrip_2d() {
    // Smooth 8×12 ramp (values ~0 to ~95/4).
    let vals: Vec<f32> = (0..96).map(|i| i as f32 * 0.25).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("v")
        .with_f32_data(&vals)
        .with_shape(&[8, 12])
        .with_chunks(&[8, 12])
        .with_zfp(16.0);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("v").unwrap();
    let back = ds.read_f32().unwrap();
    assert_eq!(back.len(), vals.len());
    assert!(
        max_abs_err(&vals, &back) < 0.1,
        "max_err {} > 0.1",
        max_abs_err(&vals, &back)
    );
}

#[test]
fn zfp_f64_roundtrip_3d() {
    let dims = [4usize, 4, 4];
    let n = dims.iter().product();
    let vals: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("v")
        .with_f64_data(&vals)
        .with_shape(&[dims[0] as u64, dims[1] as u64, dims[2] as u64])
        .with_chunks(&[dims[0] as u64, dims[1] as u64, dims[2] as u64])
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
    assert!(max_err < 0.01, "max_err {max_err} > 0.01");
}

#[test]
fn zfp_f32_roundtrip_4d() {
    let dims = [4usize, 4, 4, 4];
    let n = dims.iter().product();
    let vals: Vec<f32> = (0..n).map(|i| i as f32 * 0.125).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("v")
        .with_f32_data(&vals)
        .with_shape(&dims.map(|d| d as u64))
        .with_chunks(&dims.map(|d| d as u64))
        .with_zfp(16.0);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("v").unwrap();
    let back = ds.read_f32().unwrap();
    assert_eq!(back.len(), vals.len());
    assert!(
        max_abs_err(&vals, &back) < 0.1,
        "max_err {} > 0.1",
        max_abs_err(&vals, &back)
    );
}

#[test]
fn zfp_on_unsupported_scalar_errors_not_panics() {
    // u8 is not one of ZFP's supported scalar types. Finalize must surface
    // a FormatError::UnsupportedZfp, not panic.
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("v")
        .with_u8_data(&[1u8, 2, 3, 4])
        .with_chunks(&[4])
        .with_zfp(16.0);
    let err = builder
        .finish()
        .expect_err("ZFP on u8 should error, not succeed");
    assert!(
        matches!(err, Error::Format(FormatError::UnsupportedZfp(_))),
        "unexpected error variant: {err:?}",
    );
}

#[test]
fn zfp_with_5d_chunks_errors_not_panics() {
    // 5D chunks are beyond the ZFP rank limit. Finalize must surface a
    // FormatError::UnsupportedZfp, not panic inside the cd_values builder.
    let vals: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("v")
        .with_f32_data(&vals)
        .with_shape(&[2, 2, 2, 2, 2])
        .with_chunks(&[2, 2, 2, 2, 2])
        .with_zfp(16.0);
    let err = builder
        .finish()
        .expect_err("ZFP with 5D chunks should error, not succeed");
    assert!(
        matches!(err, Error::Format(FormatError::UnsupportedZfp(_))),
        "unexpected error variant: {err:?}",
    );
}
