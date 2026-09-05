// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit little-endian targets; skip them elsewhere so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
//! The reference C library as the oracle for a regional read.
//!
//! It writes the fixtures — the compact layout among them, which this crate's
//! writer never produces — and reads every region back through
//! `H5Sselect_hyperslab`, so a region here is checked against the selection the
//! format defines rather than against this crate's own whole read cut in Rust,
//! which is what `tests/region_read.rs` does.

use hdf5::dataset::Layout as CLayout;
use hdf5::{Hyperslab, Selection, SliceOrIndex};
use hdf5_pure::{File, Layout};
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

const ROWS: usize = 15;
const COLS: usize = 11;

fn values() -> Vec<f64> {
    (0..ROWS * COLS).map(|i| i as f64 * 0.5 - 7.0).collect()
}

/// The C library's read of the box `start .. start + count` of a 2-D dataset,
/// through a hyperslab selection at unit stride. An axis one element wide is
/// selected by index, which drops it from the result, so the typed slice
/// readers see a rank they can name: a vector for one remaining axis, a matrix
/// for two. A box one element wide on both axes leaves no axis and is not among
/// the boxes checked here.
fn c_region(ds: &hdf5::Dataset, start: [u64; 2], count: [u64; 2]) -> Vec<f64> {
    let axes: Vec<SliceOrIndex> = (0..2)
        .map(|i| {
            if count[i] == 1 {
                SliceOrIndex::Index(start[i] as usize)
            } else {
                SliceOrIndex::SliceCount {
                    start: start[i] as usize,
                    step: 1,
                    count: count[i] as usize,
                    block: 1,
                }
            }
        })
        .collect();
    let remaining = axes
        .iter()
        .filter(|a| !matches!(a, SliceOrIndex::Index(_)))
        .count();
    let selection = Selection::from(Hyperslab::from(axes));
    match remaining {
        1 => ds.read_slice_1d::<f64, _>(selection).unwrap().to_vec(),
        2 => ds
            .read_slice_2d::<f64, _>(selection)
            .unwrap()
            .iter()
            .copied()
            .collect(),
        _ => unreachable!("BOXES leave one or two axes"),
    }
}

/// Boxes of a `[15, 11]` dataset: the whole, one straddling chunk boundaries on
/// both axes of a `[4, 3]` grid, the last column, exactly one chunk, most of
/// the dataset with edges inside chunks, one whole row, a two-wide column, and
/// the first four columns — an inner axis cut from its start.
const BOXES: [([u64; 2], [u64; 2]); 8] = [
    ([0, 0], [15, 11]),
    ([0, 0], [15, 4]),
    ([3, 2], [5, 5]),
    ([0, 10], [15, 1]),
    ([4, 3], [4, 3]),
    ([2, 1], [12, 9]),
    ([7, 0], [1, 11]),
    ([5, 4], [8, 2]),
];

fn check(path: &std::path::Path, name: &str) {
    let c = hdf5::File::open(path).unwrap();
    let c_ds = c.dataset(name).unwrap();
    for file in [
        File::open(path).unwrap(),
        File::open_streaming(path).unwrap(),
    ] {
        let ds = file.dataset(name).unwrap();
        for (start, count) in BOXES {
            assert_eq!(
                ds.read_f64_region(&start, &count).unwrap(),
                c_region(&c_ds, start, count),
                "{name}: region {start:?} + {count:?}"
            );
        }
    }
}

#[test]
fn c_regions_of_every_layout_match_the_c_librarys_hyperslab_read() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("regions.h5");
    let data = values();
    {
        let f = hdf5::File::create(&path).unwrap();
        f.new_dataset::<f64>()
            .layout(CLayout::Compact)
            .shape((ROWS, COLS))
            .create("compact")
            .unwrap()
            .write_raw(&data)
            .unwrap();
        f.new_dataset::<f64>()
            .layout(CLayout::Contiguous)
            .shape((ROWS, COLS))
            .create("contiguous")
            .unwrap()
            .write_raw(&data)
            .unwrap();
        f.new_dataset::<f64>()
            .chunk((4, 3))
            .deflate(4)
            .shape((ROWS, COLS))
            .create("chunked")
            .unwrap()
            .write_raw(&data)
            .unwrap();
        f.close().unwrap();
    }
    // The compact source really is compact, or this test proves nothing about
    // that arm of the reader.
    {
        let f = File::open(&path).unwrap();
        assert!(
            matches!(
                f.dataset("compact").unwrap().layout().unwrap(),
                Layout::Compact { .. }
            ),
            "expected a compact source layout"
        );
    }
    for name in ["compact", "contiguous", "chunked"] {
        check(&path, name);
    }
}

/// A dataset the C library created and never wrote has no storage behind it;
/// a region of it is the fill value, the same answer the C library gives for
/// the hyperslab.
#[test]
fn c_regions_of_unwritten_storage_read_as_the_fill_the_c_library_reads() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("blank.h5");
    {
        let f = hdf5::File::create(&path).unwrap();
        f.new_dataset::<f64>()
            .chunk((4, 3))
            .fill_value(-1.5f64)
            .shape((ROWS, COLS))
            .create("chunked")
            .unwrap();
        f.new_dataset::<f64>()
            .layout(CLayout::Contiguous)
            .fill_value(2.25f64)
            .shape((ROWS, COLS))
            .create("contiguous")
            .unwrap();
        f.close().unwrap();
    }
    for name in ["chunked", "contiguous"] {
        check(&path, name);
    }
}
