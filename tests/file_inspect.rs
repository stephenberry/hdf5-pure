//! Tests for the file-inspection helpers added for issue #32 (Group A + B):
//! `is_hdf5` / `is_hdf5_bytes`, `File::file_size`, `File::libver_bound`, and
//! `FileBuilder::with_libver_bounds`.

use hdf5_pure::{File, FileBuilder, LibVer, is_hdf5, is_hdf5_bytes};

fn sample_file() -> Vec<u8> {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_f64_data(&[1.0, 2.0, 3.0]);
    builder.finish().unwrap()
}

#[test]
fn is_hdf5_bytes_detects_signature() {
    let bytes = sample_file();
    assert!(is_hdf5_bytes(&bytes));
    assert!(!is_hdf5_bytes(&[0u8; 64]));
    assert!(!is_hdf5_bytes(&[]));
}

#[test]
fn is_hdf5_path_roundtrip() {
    let dir = std::env::temp_dir();
    let h5 = dir.join("hdf5_pure_is_hdf5_yes.h5");
    let other = dir.join("hdf5_pure_is_hdf5_no.bin");
    std::fs::write(&h5, sample_file()).unwrap();
    std::fs::write(&other, b"not an hdf5 file at all").unwrap();

    assert!(is_hdf5(&h5).unwrap());
    assert!(!is_hdf5(&other).unwrap());

    // A missing file is an I/O error, not `Ok(false)`.
    assert!(is_hdf5(dir.join("hdf5_pure_definitely_missing.h5")).is_err());

    std::fs::remove_file(&h5).ok();
    std::fs::remove_file(&other).ok();
}

#[test]
fn file_size_matches_buffer_and_metadata() {
    let bytes = sample_file();
    let len = bytes.len() as u64;

    let file = File::from_bytes(bytes.clone()).unwrap();
    assert_eq!(file.file_size(), len);

    let path = std::env::temp_dir().join("hdf5_pure_file_size.h5");
    std::fs::write(&path, &bytes).unwrap();
    let on_disk = File::open(&path).unwrap();
    assert_eq!(on_disk.file_size(), len);
    assert_eq!(on_disk.file_size(), std::fs::metadata(&path).unwrap().len());
    std::fs::remove_file(&path).ok();
}

#[test]
fn libver_bound_reports_writer_format() {
    // This crate writes a version 3 superblock, i.e. the HDF5 1.10 format.
    let file = File::from_bytes(sample_file()).unwrap();
    assert_eq!(file.superblock().version, 3);
    assert_eq!(file.libver_bound(), LibVer::V110);
    assert_eq!(file.libver_bound(), LibVer::WRITER_OUTPUT);
}

#[test]
fn libver_bounds_accept_straddling_range() {
    let mut builder = FileBuilder::new();
    builder.with_libver_bounds(LibVer::Earliest, LibVer::LATEST);
    builder.create_dataset("data").with_i32_data(&[1, 2, 3]);
    // Earliest..=Latest straddles the produced 1.10 format → accepted.
    assert!(builder.finish().is_ok());
}

#[test]
fn libver_bounds_reject_too_old_upper_bound() {
    let mut builder = FileBuilder::new();
    builder.with_libver_bounds(LibVer::Earliest, LibVer::V18);
    builder.create_dataset("data").with_i32_data(&[1, 2, 3]);
    // Upper bound 1.8 cannot hold the 1.10 format this crate emits.
    let err = builder.finish().unwrap_err();
    assert!(
        err.to_string().contains("library-version bounds"),
        "unexpected error: {err}"
    );
}

#[test]
fn libver_bounds_reject_too_new_lower_bound() {
    let mut builder = FileBuilder::new();
    builder.with_libver_bounds(LibVer::V112, LibVer::LATEST);
    builder.create_dataset("data").with_i32_data(&[1, 2, 3]);
    // Lower bound 1.12 demands a format newer than the 1.10 this crate emits.
    assert!(builder.finish().is_err());
}
