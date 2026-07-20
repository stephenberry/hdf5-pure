// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// gated to 64-bit-pointer targets; skip on 32-bit so the pure-Rust suite still
// runs under `cross test --target i686-...`.
#![cfg(not(target_pointer_width = "32"))]
//! Interop for the owned SWMR writer (issue #148, PR B): after a clean
//! `File::close`, the SWMR-write flag is cleared and the reference C library
//! reads the streamed appends back exactly.

use hdf5_pure::{File, FileBuilder};
use tempfile::tempdir;

fn build_swmr(path: &std::path::Path, n: i32, chunk: u64) {
    let data: Vec<i32> = (0..n).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&data)
        .with_shape(&[n as u64])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[chunk]);
    b.write(path).unwrap();
}

#[test]
fn c_library_reads_swmr_appends_after_close() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("s.h5");
    build_swmr(&path, 50, 1); // chunk length 1: the common streaming layout

    {
        let file = File::open_swmr_writer(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        // Several chunk-aligned appends across the inline -> direct -> super-block
        // boundaries of the Extensible-Array index.
        ds.append(&(50..120).collect::<Vec<i32>>()).unwrap();
        ds.append(&(120..250).collect::<Vec<i32>>()).unwrap();
        file.close().unwrap();
    }

    let f = hdf5::File::open(&path).unwrap();
    let v = f.dataset("d").unwrap().read_raw::<i32>().unwrap();
    assert_eq!(v, (0..250).collect::<Vec<_>>());
    f.close().unwrap();
}
