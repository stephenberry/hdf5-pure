// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// gated to 64-bit-pointer targets; skip on 32-bit so the pure-Rust suite still
// runs under `cross test --target i686-...`.
#![cfg(not(target_pointer_width = "32"))]
//! Reference-C-library interop for chunked, filtered, and resizable
//! variable-length string datasets (issue #109).
//!
//! These datasets are the ones the writer used to refuse. Their element
//! references live inside chunks that are split (and compressed) before the file
//! layout would normally fix the global-heap addresses, so their collections are
//! placed *first* — immediately after the superblock — and patched in before any
//! chunk is encoded.
//!
//! That is a layout this crate had never emitted. The pure reader catches an
//! address that lands on no collection at all (it checks the `GCOL` signature),
//! but not one that lands on the *wrong* collection, and it accepts any file
//! whose placement and emission are consistently wrong in the same way. The C
//! library independently validates that the addresses name real collections at
//! offsets it computes itself, in a file it did not write.

use hdf5::types::VarLenUnicode;
use hdf5_pure::{File, FileBuilder, FileSpaceStrategy};
use std::sync::{Mutex, MutexGuard, OnceLock};
use tempfile::tempdir;

/// The C library is not thread-safe across concurrent file handles in this
/// harness; serialize every test that touches it.
static C_LIB: OnceLock<Mutex<()>> = OnceLock::new();

fn c_lib_guard() -> MutexGuard<'static, ()> {
    C_LIB
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Read a VL-string dataset with the reference C library.
fn read_c(path: &std::path::Path, name: &str) -> Vec<String> {
    let f = hdf5::File::open(path).unwrap();
    let ds = f.dataset(name).unwrap();
    ds.read_raw::<VarLenUnicode>()
        .unwrap()
        .into_iter()
        .map(|s| s.as_str().to_owned())
        .collect()
}

/// Read the same dataset back through this crate.
fn read_pure(path: &std::path::Path, name: &str) -> Vec<String> {
    File::open(path)
        .unwrap()
        .dataset(name)
        .unwrap()
        .read_string()
        .unwrap()
}

/// Assert both libraries agree with `expected`, and that the C library also
/// reports the layout we intended (chunked, not silently contiguous).
fn assert_both_read(path: &std::path::Path, name: &str, expected: &[String]) {
    assert_eq!(read_pure(path, name), expected, "pure reader disagreed");
    assert_eq!(read_c(path, name), expected, "C library disagreed");
}

fn words(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("value-{i}")).collect()
}

/// Plain chunked: no filter, so the references are merely split across chunks.
#[test]
fn c_library_reads_chunked_vlen_strings() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("chunked.h5");
    let data = words(10);

    let mut b = FileBuilder::new();
    b.create_dataset("labels")
        .with_vlen_strings(&data.iter().map(String::as_str).collect::<Vec<_>>())
        .with_chunks(&[3]);
    b.write(&path).unwrap();

    assert_both_read(&path, "labels", &data);

    // The layout really is chunked — otherwise this test would pass trivially by
    // falling back to the long-supported contiguous path.
    let f = hdf5::File::open(&path).unwrap();
    assert!(
        f.dataset("labels").unwrap().is_chunked(),
        "expected a chunked layout"
    );
}

/// Filtered: deflate compresses the reference bytes, so a placeholder address
/// left in them could not be patched afterwards. This is the case the early
/// placement exists for.
#[test]
fn c_library_reads_filtered_chunked_vlen_strings() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("filtered.h5");
    let data = words(64);

    let mut b = FileBuilder::new();
    b.create_dataset("labels")
        .with_vlen_strings(&data.iter().map(String::as_str).collect::<Vec<_>>())
        .with_chunks(&[8])
        .with_deflate(6);
    b.write(&path).unwrap();

    assert_both_read(&path, "labels", &data);

    let f = hdf5::File::open(&path).unwrap();
    assert!(
        !f.dataset("labels").unwrap().filters().is_empty(),
        "expected a filter pipeline on the dataset"
    );
}

/// Resizable: `maxshape` alone makes the dataset chunked, and the C library must
/// see a genuine unlimited dimension.
#[test]
fn c_library_reads_resizable_vlen_strings() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("resizable.h5");
    let data = words(7);

    let mut b = FileBuilder::new();
    b.create_dataset("labels")
        .with_vlen_strings(&data.iter().map(String::as_str).collect::<Vec<_>>())
        .with_shape(&[data.len() as u64])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[4]);
    b.write(&path).unwrap();

    assert_both_read(&path, "labels", &data);

    let f = hdf5::File::open(&path).unwrap();
    let ds = f.dataset("labels").unwrap();
    assert!(ds.is_resizable(), "expected an unlimited dimension");
}

/// A multi-dimensional chunked VL dataset: the chunk splitter walks rows, so the
/// per-element reference offsets differ from the flat 1-D case.
#[test]
fn c_library_reads_2d_chunked_vlen_strings() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("grid.h5");
    let data = words(12); // 4 x 3

    let mut b = FileBuilder::new();
    b.create_dataset("grid")
        .with_vlen_strings(&data.iter().map(String::as_str).collect::<Vec<_>>())
        .with_shape(&[4, 3])
        .with_chunks(&[2, 3]);
    b.write(&path).unwrap();

    assert_both_read(&path, "grid", &data);
}

/// More heap objects than one collection holds (65,535 — see issue #189), so the
/// early placement must assign an address per collection and the references must
/// select the right one. A single wrong collection address here reads back as
/// garbage in the C library.
#[test]
fn c_library_reads_chunked_vlen_strings_across_heap_collections() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("many.h5");
    let data = words(70_000);

    let mut b = FileBuilder::new();
    b.create_dataset("labels")
        .with_vlen_strings(&data.iter().map(String::as_str).collect::<Vec<_>>())
        .with_chunks(&[4096]);
    b.write(&path).unwrap();

    // Spot-check the boundaries rather than comparing 70k strings twice.
    let pure = read_pure(&path, "labels");
    let c = read_c(&path, "labels");
    assert_eq!(pure.len(), data.len());
    assert_eq!(c.len(), data.len());
    for i in [0usize, 1, 65_534, 65_535, 65_536, 69_999] {
        assert_eq!(pure[i], data[i], "pure reader differs at {i}");
        assert_eq!(c[i], data[i], "C library differs at {i}");
    }
}

/// The paged layout is a separate code path in the writer, with the collections
/// living in the metadata region. Cover it too, since the early placement moved
/// where that region starts.
#[test]
fn c_library_reads_paged_chunked_vlen_strings() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("paged.h5");
    let data = words(20);

    let mut b = FileBuilder::new();
    b.create_dataset("labels")
        .with_vlen_strings(&data.iter().map(String::as_str).collect::<Vec<_>>())
        .with_chunks(&[5]);
    b.with_file_space_strategy(FileSpaceStrategy::Page, true, 1);
    b.write(&path).unwrap();

    assert_both_read(&path, "labels", &data);
}

/// A file mixing a chunked VL dataset (placed early) with a contiguous one
/// (placed late) exercises both placements in one layout, which is where an
/// address/emission-order mismatch would show up.
#[test]
fn c_library_reads_mixed_chunked_and_contiguous_vlen_strings() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("mixed.h5");
    let chunked = words(9);
    let contiguous: Vec<String> = ["alpha", "beta", "gamma"]
        .iter()
        .map(|s| (*s).to_owned())
        .collect();

    let mut b = FileBuilder::new();
    b.create_dataset("chunked")
        .with_vlen_strings(&chunked.iter().map(String::as_str).collect::<Vec<_>>())
        .with_chunks(&[4]);
    b.create_dataset("plain")
        .with_vlen_strings(&contiguous.iter().map(String::as_str).collect::<Vec<_>>());
    b.write(&path).unwrap();

    assert_both_read(&path, "chunked", &chunked);
    assert_both_read(&path, "plain", &contiguous);
}

/// Two chunked VL datasets whose collections are both placed early, with
/// **distinct** content.
///
/// This is what pins the placement/emission ordering invariant: `early_gcol` is
/// walked once to assign addresses and again to write the bytes, and the two
/// walks must agree. With one early dataset no ordering is expressible, and with
/// two identical ones a swap is a no-op — so a reversed emission order would be
/// invisible. Here it makes each dataset read the other's strings, and the C
/// library rejects the file outright.
#[test]
fn c_library_reads_two_distinct_chunked_vlen_string_datasets() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("two_early.h5");

    // Different lengths as well as different text, so a swap cannot survive by
    // the two collections happening to be the same size.
    let first: Vec<String> = (0..12).map(|i| format!("first-{i}")).collect();
    let second: Vec<String> = (0..30)
        .map(|i| format!("second-{i}-with-a-longer-body"))
        .collect();

    let mut b = FileBuilder::new();
    b.create_dataset("a")
        .with_vlen_strings(&first.iter().map(String::as_str).collect::<Vec<_>>())
        .with_chunks(&[4]);
    b.create_dataset("b")
        .with_vlen_strings(&second.iter().map(String::as_str).collect::<Vec<_>>())
        .with_chunks(&[7])
        .with_deflate(6);
    b.write(&path).unwrap();

    assert_both_read(&path, "a", &first);
    assert_both_read(&path, "b", &second);
}

/// Five early-placed datasets, mixed with a contiguous VL dataset that keeps the
/// late placement. Exercises the two cursors in one file and makes an ordering
/// slip anywhere in the middle of `early_gcol` detectable, not just a full
/// reversal of two.
#[test]
fn c_library_reads_many_chunked_vlen_string_datasets() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("many_early.h5");

    let sets: Vec<Vec<String>> = (0..5)
        .map(|d| {
            (0..(6 + d * 5))
                .map(|i| format!("ds{d}-item{i}{}", "x".repeat(d)))
                .collect()
        })
        .collect();
    let contiguous: Vec<String> = (0..4).map(|i| format!("late-{i}")).collect();

    let mut b = FileBuilder::new();
    for (d, data) in sets.iter().enumerate() {
        b.create_dataset(&format!("chunked{d}"))
            .with_vlen_strings(&data.iter().map(String::as_str).collect::<Vec<_>>())
            .with_chunks(&[3]);
    }
    b.create_dataset("contiguous")
        .with_vlen_strings(&contiguous.iter().map(String::as_str).collect::<Vec<_>>());
    b.write(&path).unwrap();

    for (d, data) in sets.iter().enumerate() {
        assert_both_read(&path, &format!("chunked{d}"), data);
    }
    assert_both_read(&path, "contiguous", &contiguous);
}
