//! Cross-validation for Extensible-Array-indexed chunked datasets (one unlimited
//! dimension), in both directions against the reference C HDF5 library.
//!
//! These guard the EA block-size geometry, super blocks, and paged data blocks.
//! Sizes are chosen to span every structural range:
//!   - 20      inline (4) + first direct data block (16)
//!   - 300     all 6 direct data blocks + the first super block
//!   - 2000    several super blocks
//!   - 50000   deeper super-block nesting
//!   - 140000  past 131060 chunks: paged data blocks
//!
//! Before the geometry fix, anything past 20 chunks silently corrupted on read.

use hdf5::Extent;
use hdf5::file::LibraryVersion;
use hdf5_pure::{File, FileBuilder};
use tempfile::tempdir;

const SIZES: &[usize] = &[20, 300, 2000, 50000, 140000];

/// Create a 1-D unlimited, chunked i32 dataset with the reference C library,
/// using the latest format so the chunk index is an Extensible Array.
fn write_with_c(path: &std::path::Path, n: usize) {
    let file = hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
        .create(path)
        .unwrap();
    let ds = file
        .new_dataset::<i32>()
        .chunk((1,))
        .shape((Extent::resizable(n),))
        .create("d")
        .unwrap();
    let data: Vec<i32> = (0..n as i32).collect();
    ds.write(&data).unwrap();
    file.close().unwrap();
}

/// Create the same dataset with hdf5-pure.
fn write_with_pure(path: &std::path::Path, n: usize) {
    let data: Vec<i32> = (0..n as i32).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&data)
        .with_shape(&[n as u64])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[1]);
    b.write(path).unwrap();
}

/// Direction 1: hdf5-pure writes, the reference C library reads.
#[test]
fn pure_writes_c_reads() {
    for &n in SIZES {
        let dir = tempdir().unwrap();
        let path = dir.path().join("ea.h5");
        write_with_pure(&path, n);

        let file = hdf5::File::open(&path).unwrap();
        let ds = file.dataset("d").unwrap();
        let values = ds.read_raw::<i32>().unwrap();
        let expected: Vec<i32> = (0..n as i32).collect();
        assert_eq!(values.len(), n, "C read wrong length for n={n}");
        assert_eq!(values, expected, "C read wrong data for n={n}");
    }
}

/// Direction 2: the reference C library writes, hdf5-pure reads.
#[test]
fn c_writes_pure_reads() {
    for &n in SIZES {
        let dir = tempdir().unwrap();
        let path = dir.path().join("ea.h5");
        write_with_c(&path, n);

        let bytes = std::fs::read(&path).unwrap();
        let file = File::from_bytes(bytes).unwrap();
        let ds = file.dataset("d").unwrap();
        let values = ds.read_i32().unwrap();
        let expected: Vec<i32> = (0..n as i32).collect();
        assert_eq!(values.len(), n, "hdf5-pure read wrong length for n={n}");
        assert_eq!(values, expected, "hdf5-pure read wrong data for n={n}");
    }
}

/// The EA header's six statistics fields must match the C library byte-for-byte
/// (the C library recomputes and would reject inconsistent computed stats; the
/// stored max_idx_set / nelmts must also agree). Compares the headers of the two
/// files for identical datasets.
#[test]
fn eahd_stats_match_c() {
    fn stats_of(path: &std::path::Path) -> Vec<u64> {
        let b = std::fs::read(path).unwrap();
        let h = (0..b.len() - 4).find(|&i| &b[i..i + 4] == b"EAHD").unwrap();
        (0..6)
            .map(|k| {
                let p = h + 12 + k * 8;
                u64::from_le_bytes(b[p..p + 8].try_into().unwrap())
            })
            .collect()
    }

    for &n in SIZES {
        let dir = tempdir().unwrap();
        let c_path = dir.path().join("c.h5");
        let pure_path = dir.path().join("pure.h5");
        write_with_c(&c_path, n);
        write_with_pure(&pure_path, n);
        assert_eq!(
            stats_of(&pure_path),
            stats_of(&c_path),
            "EAHD stats differ from the C library at n={n}"
        );
    }
}
