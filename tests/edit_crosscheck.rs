// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit-pointer targets; skip them on 32-bit so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(not(target_pointer_width = "32"))]
//! Cross-validation for in-place editing against the reference C library
//! (PR #38 / issue #32): files the C library *writes* are edited in place by
//! `EditSession`, and the result is read back by both `hdf5-pure` and the C
//! library. This proves the editor works on files it did not itself produce, in
//! both the HDF5 1.8 (v2 superblock) and 1.10+ (v3 superblock) formats.
//!
//! Boundary recorded here too: the default earliest format (version 0
//! superblock with symbol-table groups) is refused cleanly, leaving the file
//! untouched. (A separate, content-dependent limit not asserted here: the C
//! library sometimes lays a group header out across multiple chunks, which the
//! editor does not yet rebuild — see `src/edit.rs`.)

use hdf5::file::LibraryVersion;
use hdf5_pure::{EditSession, File, FileBuilder};
use tempfile::tempdir;

/// Stage an add, an add-into-a-group, a delete, and a copy — the full op set.
fn stage_edits(session: &mut EditSession) {
    session
        .create_dataset("added")
        .with_f64_data(&[100.0, 200.0]);
    session
        .create_dataset("grp/gamma")
        .with_i32_data(&[1, 2, 3]);
    session.delete("doomed");
    session.copy("alpha", "alpha_copy");
}

/// Read the edited file back through both readers and assert every object is
/// correct (and the deleted one is gone).
fn assert_edits_applied(path: &std::path::Path) {
    // hdf5-pure reader.
    let f = File::open(path).unwrap();
    assert_eq!(
        f.dataset("alpha").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
    assert_eq!(
        f.dataset("alpha_copy").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
    assert_eq!(
        f.dataset("added").unwrap().read_f64().unwrap(),
        vec![100.0, 200.0]
    );
    assert_eq!(
        f.dataset("grp/beta").unwrap().read_i32().unwrap(),
        vec![10, 20, 30, 40]
    );
    assert_eq!(
        f.dataset("grp/gamma").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    assert!(
        f.dataset("doomed").is_err(),
        "deleted dataset still present (pure)"
    );

    // Reference C library reader — the real interop proof.
    let c = hdf5::File::open(path).unwrap();
    assert_eq!(
        c.dataset("alpha_copy").unwrap().read_raw::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
    assert_eq!(
        c.dataset("added").unwrap().read_raw::<f64>().unwrap(),
        vec![100.0, 200.0]
    );
    assert_eq!(
        c.dataset("grp/beta").unwrap().read_raw::<i32>().unwrap(),
        vec![10, 20, 30, 40]
    );
    assert_eq!(
        c.dataset("grp/gamma").unwrap().read_raw::<i32>().unwrap(),
        vec![1, 2, 3]
    );
    assert!(
        c.dataset("doomed").is_err(),
        "deleted dataset still present (C library)"
    );
}

/// Write the starter file (two root datasets + a group with a dataset) with the
/// C library at the given library-version bounds.
fn write_c_starter(path: &std::path::Path, low: LibraryVersion, high: LibraryVersion) {
    let file = hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(low, high))
        .create(path)
        .unwrap();
    file.new_dataset::<f64>()
        .shape((3,))
        .create("alpha")
        .unwrap()
        .write(&[1.0f64, 2.0, 3.0])
        .unwrap();
    file.new_dataset::<i32>()
        .shape((2,))
        .create("doomed")
        .unwrap()
        .write(&[7i32, 8])
        .unwrap();
    let grp = file.create_group("grp").unwrap();
    grp.new_dataset::<i32>()
        .shape((4,))
        .create("beta")
        .unwrap()
        .write(&[10i32, 20, 30, 40])
        .unwrap();
    file.close().unwrap();
}

#[test]
fn pure_written_file_edited_then_read_by_c_library() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("pure.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("alpha").with_f64_data(&[1.0, 2.0, 3.0]);
    b.create_dataset("doomed").with_i32_data(&[7, 8]);
    let mut g = b.create_group("grp");
    g.create_dataset("beta").with_i32_data(&[10, 20, 30, 40]);
    b.add_group(g.finish());
    b.write(&path).unwrap();

    let mut session = EditSession::open(&path).unwrap();
    stage_edits(&mut session);
    session.commit().unwrap();

    assert_edits_applied(&path);
}

#[test]
fn c_written_v2_file_edited_in_place() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_v2.h5");
    write_c_starter(&path, LibraryVersion::V18, LibraryVersion::V18);
    assert_eq!(File::open(&path).unwrap().superblock().version, 2);

    let mut session = EditSession::open(&path).unwrap();
    stage_edits(&mut session);
    session.commit().unwrap();

    assert_edits_applied(&path);
}

#[test]
fn c_written_v3_file_edited_in_place() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_v3.h5");
    write_c_starter(&path, LibraryVersion::V110, LibraryVersion::latest());
    assert_eq!(File::open(&path).unwrap().superblock().version, 3);

    let mut session = EditSession::open(&path).unwrap();
    stage_edits(&mut session);
    session.commit().unwrap();

    assert_edits_applied(&path);
}

#[test]
fn c_earliest_symboltable_file_is_refused_cleanly() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_v0.h5");
    // Earliest low bound yields a version 0 superblock with symbol-table groups.
    write_c_starter(&path, LibraryVersion::Earliest, LibraryVersion::V18);
    assert!(
        File::open(&path).unwrap().superblock().version <= 1,
        "expected a v0/v1 superblock from the earliest libver bound"
    );
    let before = std::fs::read(&path).unwrap();

    let err = match EditSession::open(&path) {
        Ok(_) => panic!("expected a version 0 / symbol-table file to be refused at open"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("symbol-table") || err.to_string().contains("version 0"),
        "unexpected reason: {err}"
    );
    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "rejected open modified the file"
    );

    // hdf5-pure can still *read* the default-format file.
    assert_eq!(
        File::open(&path)
            .unwrap()
            .dataset("alpha")
            .unwrap()
            .read_f64()
            .unwrap(),
        vec![1.0, 2.0, 3.0]
    );
}
