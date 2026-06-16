//! OS advisory file-locking behavior for the write/edit/read open paths
//! (issue #73). These tests use only hdf5-pure (no reference C library), so they
//! run on every target. Same-process opens contend because the std lock uses
//! `flock` (Unix) / `LockFileEx` (Windows), which are per-open-file-description.

use hdf5_pure::{
    EditSession, Error, File, FileAccessOptions, FileBuilder, FileLocking, SwmrWriter,
};
use tempfile::tempdir;

/// A plain, in-place-editable starter file (consistency_flags = 0).
fn write_starter(path: &std::path::Path) {
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_i32_data(&[1, 2, 3]);
    b.write(path).unwrap();
}

/// An unlimited Extensible-Array dataset the SWMR append writer can open.
fn write_appendable(path: &std::path::Path) {
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&[0, 1, 2, 3])
        .with_shape(&[4])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[1]);
    b.write(path).unwrap();
}

#[test]
fn exclusive_writer_lock_blocks_second_writer_and_plain_reader() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("edit.h5");
    write_starter(&path);

    let session = EditSession::open(&path).unwrap(); // holds the exclusive lock

    // A second writer is refused while the first is alive.
    assert!(
        matches!(EditSession::open(&path), Err(Error::FileLocked(_))),
        "second EditSession::open should be FileLocked"
    );
    // A plain reader (shared lock) is refused against the active exclusive lock.
    assert!(
        matches!(File::open(&path), Err(Error::FileLocked(_))),
        "File::open should be FileLocked while a writer is active"
    );

    drop(session); // releases the lock

    // Once the writer is gone, a plain read succeeds again.
    File::open(&path).expect("read should succeed after the writer is dropped");
}

#[test]
fn disabled_locking_bypasses_the_lock() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("edit.h5");
    write_starter(&path);

    let _session = EditSession::open(&path).unwrap(); // Enabled: exclusive lock held

    // An explicit FileLocking::Disabled reader ignores the held lock.
    let opts = FileAccessOptions::new().with_file_locking(FileLocking::Disabled);
    File::open_with_options(&path, opts)
        .expect("Disabled locking should bypass the held exclusive lock");

    // And a Disabled writer can open alongside it too.
    EditSession::open_with_locking(&path, FileLocking::Disabled)
        .expect("Disabled locking should bypass the held exclusive lock for writers");
}

#[test]
fn swmr_writer_lock_blocks_second_writer_but_not_swmr_reader() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("swmr.h5");
    write_appendable(&path);

    let writer = SwmrWriter::open(&path).unwrap(); // exclusive lock held

    // A second SWMR writer is refused.
    assert!(
        matches!(SwmrWriter::open(&path), Err(Error::FileLocked(_))),
        "second SwmrWriter::open should be FileLocked"
    );

    // A SWMR reader takes no lock by design and coexists with the live writer.
    File::open_swmr(&path).expect("open_swmr should coexist with an active SWMR writer");

    drop(writer); // releases the lock and clears the SWMR flag

    SwmrWriter::open(&path).expect("a new SWMR writer should open after the first is dropped");
}
