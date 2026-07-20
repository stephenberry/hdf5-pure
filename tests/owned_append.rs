//! Write-through append on an owned handle (issue #148, phase 2).
//!
//! A file opened with `File::open_rw` hands out `Dataset` handles that both read
//! and append in place: `Dataset::append` grows the dataset, and reads through
//! the same handle observe the new length and data. A read-only file refuses the
//! append.

use hdf5_pure::{Error, File, FileBuilder};
use tempfile::tempdir;

/// Create a rank-1, unlimited i32 dataset with chunk length `chunk`, seeded 0..n.
fn create_i32(path: &std::path::Path, n: i32, chunk: u64) {
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
fn append_and_read_through_one_handle() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    create_i32(&path, 5, 4); // [0,1,2,3,4]
    {
        let file = File::open_rw(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        assert_eq!(ds.shape().unwrap(), vec![5]);
        ds.append(&[5i32, 6, 7]).unwrap();
        // Read through the SAME handle: it sees the new length and data.
        assert_eq!(ds.shape().unwrap(), vec![8]);
        assert_eq!(ds.read_i32().unwrap(), (0..8).collect::<Vec<_>>());
    } // drop the writer -> releases the exclusive lock before we reopen
    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("d").unwrap().read_i32().unwrap(),
        (0..8).collect::<Vec<_>>()
    );
}

#[test]
fn many_appends_across_calls() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    create_i32(&path, 0, 4);
    {
        let file = File::open_rw(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        for v in 0..20i32 {
            ds.append(&[v]).unwrap();
        }
        assert_eq!(ds.read_i32().unwrap(), (0..20).collect::<Vec<_>>());
    }
    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("d").unwrap().read_i32().unwrap(),
        (0..20).collect::<Vec<_>>()
    );
}

#[test]
fn append_to_readonly_file_is_refused() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    create_i32(&path, 3, 4);
    let bytes = std::fs::read(&path).unwrap();
    let file = File::from_bytes(bytes).unwrap();
    let mut ds = file.dataset("d").unwrap();
    assert!(matches!(ds.append(&[9i32]), Err(Error::ReadOnly)));
}

#[test]
fn open_rw_reads_like_open() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    create_i32(&path, 6, 3);
    let file = File::open_rw(&path).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        (0..6).collect::<Vec<_>>()
    );
    assert_eq!(file.dataset("d").unwrap().shape().unwrap(), vec![6]);
}
