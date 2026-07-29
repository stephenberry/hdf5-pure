//! End-to-end write/read roundtrip through the full HDF5 stack with LZF
//! compression enabled. The codec itself is covered by `src/lzf_crosscheck.rs`
//! against h5py-produced fixtures; this harness checks that the chunked
//! writer, filter pipeline, edit engine, and repack all thread LZF through
//! correctly. LZF is lossless, so every comparison is exact equality.

use hdf5_pure::{Error, File, FileBuilder, FormatError, RepackOptions, repack};

fn tmp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(name)
}

#[test]
fn lzf_i32_roundtrip() {
    let vals: Vec<i32> = (0..256).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("v")
        .with_i32_data(&vals)
        .with_chunks(&[256])
        .with_lzf();
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("v").unwrap();
    assert_eq!(ds.filters(), vec![32000]);
    assert_eq!(ds.read_i32().unwrap(), vals);
}

#[test]
fn lzf_shuffle_f64_roundtrip() {
    let vals: Vec<f64> = (0..512).map(|i| (i as f64 * 0.05).sin()).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("v")
        .with_f64_data(&vals)
        .with_chunks(&[128])
        .with_shuffle()
        .with_lzf();
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("v").unwrap();
    assert_eq!(ds.filters(), vec![2, 32000]);
    assert_eq!(ds.read_f64().unwrap(), vals);
}

#[test]
fn lzf_multi_chunk_partial_edge_roundtrip() {
    // 1000 elements over 128-element chunks: 7 full chunks + a partial edge.
    let vals: Vec<i64> = (0..1000).map(|i| i * 7 - 500).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("v")
        .with_i64_data(&vals)
        .with_chunks(&[128])
        .with_lzf();
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    assert_eq!(file.dataset("v").unwrap().read_i64().unwrap(), vals);
}

#[test]
fn lzf_incompressible_roundtrip() {
    // LZF is written as a mandatory filter, so incompressible chunks are
    // stored compressed (slightly grown), not raw — they must still decode.
    // The bytes are the u8_noise fixture, the same stream h5py's optional
    // filter refused to compress.
    let vals = std::fs::read(fixture("u8_noise.raw.bin")).unwrap();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("v")
        .with_u8_data(&vals)
        .with_chunks(&[1024])
        .with_lzf();
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    assert_eq!(file.dataset("v").unwrap().read_u8().unwrap(), vals);
}

#[test]
fn lzf_records_h5py_cd_values() {
    // h5py's lzf_set_local convention: [filter version, liblzf version,
    // chunk size in bytes]. Its reader uses [2] as a buffer hint.
    let vals: Vec<i32> = (0..64).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("v")
        .with_i32_data(&vals)
        .with_chunks(&[64])
        .with_lzf();
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let pipeline = file.dataset("v").unwrap().filter_pipeline();
    assert_eq!(pipeline.len(), 1);
    assert_eq!(pipeline[0].id, 32000);
    assert_eq!(pipeline[0].name.as_deref(), Some("lzf"));
    assert_eq!(pipeline[0].client_data, vec![4, 0x0105, 64 * 4]);
}

#[test]
fn lzf_plus_deflate_refused() {
    // Two general byte compressors on one dataset is never useful; the
    // builder refuses the combination rather than silently dropping one.
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("v")
        .with_i32_data(&[1, 2, 3, 4])
        .with_chunks(&[4])
        .with_lzf()
        .with_deflate(6);
    let err = builder
        .finish()
        .expect_err("lzf + deflate should error, not succeed");
    assert!(
        matches!(err, Error::Format(FormatError::FilterError(_))),
        "unexpected error variant: {err:?}",
    );
}

/// An LZF-compressed dataset can be added through the edit engine (which
/// requires `pipeline_reencodable` to accept the filter) and reads back exact.
#[test]
fn add_lzf_dataset_in_place() {
    let path = tmp("hdf5_pure_edit_add_lzf.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("original").with_f64_data(&[1.0, 2.0]);
        b.write(&path).unwrap();
    }

    let data: Vec<i32> = (0..256).map(|i| i % 17).collect();
    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("lzf", |b| {
                b.with_i32_data(&data).with_chunks(&[64]).with_lzf();
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("lzf").unwrap().read_i32().unwrap(), data);
    std::fs::remove_file(&path).ok();
}

/// Overwriting an LZF dataset in place re-encodes its chunks through the LZF
/// compressor, the path `pipeline_reencodable` used to refuse.
#[test]
fn overwrite_lzf_dataset_in_place() {
    let path = tmp("hdf5_pure_edit_overwrite_lzf.h5");
    let before: Vec<i32> = (0..256).collect();
    let after: Vec<i32> = (0..256).map(|i| i * 3).collect();
    {
        let mut b = FileBuilder::new();
        b.create_dataset("v")
            .with_i32_data(&before)
            .with_chunks(&[64])
            .with_lzf();
        b.write(&path).unwrap();
    }

    {
        let session = File::open_rw(&path).unwrap();
        session.dataset("v").unwrap().write(&after).unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("v").unwrap().read_i32().unwrap(), after);
    std::fs::remove_file(&path).ok();
}

fn fixture(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/lzf")
        .join(name)
}

/// h5py registers LZF as optional and stored this incompressible chunk raw
/// with its filter-mask bit set — a file our writer can never produce, and the
/// only way to exercise the mask-skip path on an LZF pipeline.
#[test]
fn reads_h5py_lzf_masked_chunk() {
    let file = File::open(fixture("u8_noise.h5")).unwrap();
    let ds = file.dataset("v").unwrap();
    assert_eq!(ds.filters(), vec![32000]);
    let expected = std::fs::read(fixture("u8_noise.raw.bin")).unwrap();
    assert_eq!(ds.read_u8().unwrap(), expected);
}

/// A shuffle+lzf chain exactly as h5py writes it.
#[test]
fn reads_h5py_shuffle_lzf() {
    let file = File::open(fixture("f64_shuffle_lzf.h5")).unwrap();
    let ds = file.dataset("v").unwrap();
    assert_eq!(ds.filters(), vec![2, 32000]);
    let raw = std::fs::read(fixture("f64_shuffle_lzf.raw.bin")).unwrap();
    let expected: Vec<f64> = raw
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(ds.read_f64().unwrap(), expected);
}

/// A multi-chunk h5py LZF dataset (with a partial edge chunk) reads through.
#[test]
fn reads_h5py_lzf_multichunk() {
    let file = File::open(fixture("i64_multichunk.h5")).unwrap();
    let ds = file.dataset("v").unwrap();
    let raw = std::fs::read(fixture("i64_multichunk.raw.bin")).unwrap();
    let expected: Vec<i64> = raw
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(ds.read_i64().unwrap(), expected);
}

/// Repack carries an LZF dataset through: the dense chunked path copies
/// chunks verbatim, and `check_pipeline` now accepts LZF as lossless for the
/// re-encode paths.
#[test]
fn repack_roundtrips_lzf() {
    let src = tmp("hdf5_pure_repack_lzf_src.h5");
    let dst = tmp("hdf5_pure_repack_lzf_dst.h5");
    let data: Vec<f64> = (0..1024).map(|i| (i as f64).cos()).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("vals")
        .with_f64_data(&data)
        .with_chunks(&[256])
        .with_shuffle()
        .with_lzf();
    b.write(&src).unwrap();

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let file = File::open(&dst).unwrap();
    let ds = file.dataset("vals").unwrap();
    assert_eq!(ds.filters(), vec![2, 32000]);
    assert_eq!(ds.read_f64().unwrap(), data);

    std::fs::remove_file(&src).ok();
    std::fs::remove_file(&dst).ok();
}
