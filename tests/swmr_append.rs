//! SWMR append-writer tests: hdf5-pure appends in place to an unlimited
//! Extensible-Array dataset, and the result is read back by hdf5-pure and by the
//! reference C library. Appends cross the inline -> direct-block -> super-block
//! boundaries so the in-place index growth is exercised.

use hdf5::Extent;
use hdf5::file::LibraryVersion;
use hdf5_pure::{File, FileBuilder, SwmrWriter};
use tempfile::tempdir;

fn pure_create(path: &std::path::Path, n: usize) {
    let data: Vec<i32> = (0..n as i32).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&data)
        .with_shape(&[n as u64])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[1]);
    b.write(path).unwrap();
}

fn c_create(path: &std::path::Path, n: usize) {
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

fn read_pure(path: &std::path::Path) -> Vec<i32> {
    let f = File::from_bytes(std::fs::read(path).unwrap()).unwrap();
    f.dataset("d").unwrap().read_i32().unwrap()
}

fn read_c(path: &std::path::Path) -> Vec<i32> {
    let f = hdf5::File::open(path).unwrap();
    f.dataset("d").unwrap().read_raw::<i32>().unwrap()
}

/// Append to an hdf5-pure-created file, crossing every structural boundary, and
/// confirm both hdf5-pure and the C library read the full result.
#[test]
fn append_to_pure_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    pure_create(&path, 10);

    {
        let mut w = SwmrWriter::open(&path).unwrap();
        // 10 -> 100 (fills direct data blocks)
        w.append_i32("d", &(10..100).collect::<Vec<_>>()).unwrap();
        // 100 -> 300 (crosses into the first super block)
        w.append_i32("d", &(100..300).collect::<Vec<_>>()).unwrap();
        // 300 -> 5000 (deeper super-block nesting)
        w.append_i32("d", &(300..5000).collect::<Vec<_>>()).unwrap();
    }

    let expected: Vec<i32> = (0..5000).collect();
    assert_eq!(read_pure(&path), expected, "hdf5-pure read mismatch");
    assert_eq!(read_c(&path), expected, "C-library read mismatch");
}

/// Append to a C-library-created file and confirm both readers agree.
#[test]
fn append_to_c_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    c_create(&path, 10);

    {
        let mut w = SwmrWriter::open(&path).unwrap();
        w.append_i32("d", &(10..1000).collect::<Vec<_>>()).unwrap();
    }

    let expected: Vec<i32> = (0..1000).collect();
    assert_eq!(read_c(&path), expected, "C-library read mismatch");
    assert_eq!(read_pure(&path), expected, "hdf5-pure read mismatch");
}

/// End-to-end SWMR loop within hdf5-pure: a refreshing reader follows the
/// append writer's in-place appends (separate file handles, same file).
#[test]
fn refreshing_reader_follows_pure_appends() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    pure_create(&path, 10);

    let mut reader = File::open_swmr(&path).unwrap();
    assert_eq!(reader.dataset("d").unwrap().read_i32().unwrap(), (0..10).collect::<Vec<_>>());

    let mut w = SwmrWriter::open(&path).unwrap();
    w.append_i32("d", &(10..300).collect::<Vec<_>>()).unwrap();

    reader.refresh().unwrap();
    {
        let ds = reader.dataset("d").unwrap();
        assert_eq!(ds.shape().unwrap(), vec![300]);
        assert_eq!(ds.read_i32().unwrap(), (0..300).collect::<Vec<_>>());
    }

    // Another round of appends, then refresh again.
    w.append_i32("d", &(300..900).collect::<Vec<_>>()).unwrap();
    reader.refresh().unwrap();
    assert_eq!(
        reader.dataset("d").unwrap().read_i32().unwrap(),
        (0..900).collect::<Vec<_>>()
    );
}

/// f64 dataset append, just to exercise a non-4-byte element size.
#[test]
fn append_f64_pure_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("f.h5");
    {
        let data: Vec<f64> = (0..5).map(|i| i as f64).collect();
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_f64_data(&data)
            .with_shape(&[5])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[1]);
        b.write(&path).unwrap();
    }
    {
        let mut w = SwmrWriter::open(&path).unwrap();
        let more: Vec<f64> = (5..400).map(|i| i as f64).collect();
        w.append_f64("d", &more).unwrap();
    }
    let f = hdf5::File::open(&path).unwrap();
    let v = f.dataset("d").unwrap().read_raw::<f64>().unwrap();
    let expected: Vec<f64> = (0..400).map(|i| i as f64).collect();
    assert_eq!(v, expected);
}
