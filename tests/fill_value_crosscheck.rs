// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit-pointer targets; skip them on 32-bit so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(not(target_pointer_width = "32"))]
//! Reference-C-library interop for configurable fill values (issue #151).
//!
//! Two directions, both make-or-break:
//!
//! * The pure writer sets a fill value and the reference C library must report
//!   the same value through its own `H5Pget_fill_value` path.
//! * The reference C library writes a fill value — under both the default format
//!   (a version-2 Fill Value message) and the latest format (a version-3
//!   message) — and the pure reader must recover it.

use hdf5::file::LibraryVersion;
use hdf5_pure::{File, FileBuilder, ScaleOffset};
use tempfile::tempdir;

/// Read a dataset's typed fill value back through the reference C library.
fn c_fill_value<T: hdf5::H5Type>(path: &std::path::Path, name: &str) -> Option<T> {
    let file = hdf5::File::open(path).unwrap();
    let ds = file.dataset(name).unwrap();
    let fv = ds.dcpl().unwrap().get_fill_value_as::<T>().unwrap();
    file.close().unwrap();
    fv
}

// ---- pure writes, C library reads ------------------------------------------

#[test]
fn c_reads_pure_written_contiguous_fill() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("pure_contig.h5");
    let mut fb = FileBuilder::new();
    fb.create_dataset("d")
        .with_i32_data(&[10, 20, 30])
        .with_fill_value(-7_i32);
    fb.write(&path).unwrap();

    assert_eq!(c_fill_value::<i32>(&path, "d"), Some(-7));
}

#[test]
fn c_reads_pure_written_chunked_fill() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("pure_chunked.h5");
    let mut fb = FileBuilder::new();
    fb.create_dataset("d")
        .with_f64_data(&[1.0, 2.0, 3.0, 4.0])
        .with_shape(&[4])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[2])
        .with_fill_value(3.5_f64);
    fb.write(&path).unwrap();

    assert_eq!(c_fill_value::<f64>(&path, "d"), Some(3.5));
}

#[test]
fn c_sees_no_fill_value_when_pure_sets_none() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("pure_none.h5");
    let mut fb = FileBuilder::new();
    fb.create_dataset("d").with_i32_data(&[1, 2, 3]);
    fb.write(&path).unwrap();

    // The crate writes HDF5's *default* fill message (neither the "defined" nor
    // the "undefined" bit set). The C library resolves that default to the type's
    // implicit zero, so it reports `Some(0)` — whereas the pure reader reports
    // `None` for the same message, since there is no *user-defined* value. Both
    // describe the same on-disk state: unwritten elements read back as zero.
    assert_eq!(c_fill_value::<i32>(&path, "d"), Some(0));
}

// ---- C library writes, pure reads ------------------------------------------

/// Create a dataset with a fill value using the reference C library under the
/// given format bounds, so the on-disk Fill Value message version is controlled:
/// the default (`Earliest`) bound writes a version-2 message, `V110`/latest a
/// version-3 message.
fn c_write_i32_fill(path: &std::path::Path, low: LibraryVersion, high: LibraryVersion) {
    let file = hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(low, high))
        .create(path)
        .unwrap();
    let ds = file
        .new_dataset::<i32>()
        .fill_value(-7_i32)
        .shape((3,))
        .create("d")
        .unwrap();
    ds.write(&[10_i32, 20, 30]).unwrap();
    file.close().unwrap();
}

#[test]
fn pure_reads_c_written_v2_fill() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_v2.h5");
    // Earliest..=V18 keeps the classic format, whose Fill Value message is v2.
    c_write_i32_fill(&path, LibraryVersion::Earliest, LibraryVersion::V18);

    let file = File::open(&path).unwrap();
    let ds = file.dataset("d").unwrap();
    assert_eq!(ds.fill_value::<i32>().unwrap(), Some(-7));
    assert_eq!(ds.read_i32().unwrap(), vec![10, 20, 30]);
}

#[test]
fn pure_reads_c_written_v3_fill() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_v3.h5");
    // The latest format writes a version-3 Fill Value message.
    c_write_i32_fill(&path, LibraryVersion::V110, LibraryVersion::latest());

    let file = File::open(&path).unwrap();
    let ds = file.dataset("d").unwrap();
    assert_eq!(ds.fill_value::<i32>().unwrap(), Some(-7));
    assert_eq!(ds.read_i32().unwrap(), vec![10, 20, 30]);
}

/// Create a chunked f64 dataset with a fill value using the reference C library.
/// The handles are dropped when this returns, so the file is fully flushed before
/// the caller reopens it.
fn c_write_chunked_f64_fill(path: &std::path::Path) {
    let file = hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
        .create(path)
        .unwrap();
    let ds = file
        .new_dataset::<f64>()
        .fill_value(2.5_f64)
        .chunk((2,))
        .shape((4,))
        .create("d")
        .unwrap();
    ds.write(&[1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    file.close().unwrap();
}

#[test]
fn pure_reads_c_written_chunked_f64_fill() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_chunked.h5");
    c_write_chunked_f64_fill(&path);

    let file = File::open(&path).unwrap();
    let ds = file.dataset("d").unwrap();
    assert_eq!(ds.fill_value::<f64>().unwrap(), Some(2.5));
    assert_eq!(ds.read_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

/// Create an i32 dataset whose fill value is explicitly *undefined* (a version-3
/// message with the "undefined" bit set), with the reference C library.
fn c_write_i32_no_fill(path: &std::path::Path) {
    let file = hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
        .create(path)
        .unwrap();
    let ds = file
        .new_dataset::<i32>()
        .no_fill_value()
        .shape((3,))
        .create("d")
        .unwrap();
    ds.write(&[1_i32, 2, 3]).unwrap();
    file.close().unwrap();
}

#[test]
fn pure_reads_none_when_c_sets_no_fill() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_no_fill.h5");
    c_write_i32_no_fill(&path);

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().fill_value::<i32>().unwrap(),
        None
    );
}

// ---- fill value alongside a filter -----------------------------------------

#[test]
fn c_reads_pure_scaleoffset_dataset_data_and_fill() {
    // A dataset fill value coexists with the scale-offset filter: the pure writer
    // emits the fill-undefined form of the filter (its own on-disk contract) plus
    // a defined dataset Fill Value message. The C library decodes the data through
    // the filter's own fill config (so the data is intact) and reports the dataset
    // fill value separately — the two do not collide.
    let dir = tempdir().unwrap();
    let path = dir.path().join("pure_so_fill.h5");
    let data: Vec<i32> = (0..16).collect();
    let mut fb = FileBuilder::new();
    fb.create_dataset("d")
        .with_i32_data(&data)
        .with_shape(&[16])
        .with_chunks(&[8])
        .with_scale_offset(ScaleOffset::Integer(0))
        .with_fill_value(-1_i32);
    fb.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("d").unwrap();
    assert_eq!(ds.read_raw::<i32>().unwrap(), data);
    assert_eq!(
        ds.dcpl().unwrap().get_fill_value_as::<i32>().unwrap(),
        Some(-1)
    );
    file.close().unwrap();
}
