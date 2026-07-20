// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// gated to 64-bit-pointer targets.
#![cfg(not(target_pointer_width = "32"))]
//! The reference HDF5 C library must be able to *modify* a file hdf5-pure wrote,
//! not just read it. Inserting a link into a group makes the C library read the
//! group's Group Info message (`H5G_obj_insert` -> `H5O_msg_read(H5O_GINFO_ID)`);
//! a group that has a Link Info message but no Group Info message fails that read
//! with "message type not found". hdf5-pure used to write groups that way, so the
//! C library could read its files but not add objects to them. These tests prove
//! the C library can now add datasets and groups to hdf5-pure-created files —
//! including into an hdf5-pure-created subgroup and after an in-place edit — and
//! that hdf5-pure reads those C-added objects back.
//!
//! Read-back here covers compact groups and small (single-direct-block) dense
//! groups. Reading a *large* multi-block dense fractal-heap group — which the C
//! library produces only when it grows one of our groups far past the compact
//! limit — is a separate, pre-existing limitation of the fractal-heap reader and
//! is not exercised here.
//!
//! Every call here goes through the safe `hdf5-metno` API, which serializes its
//! own C calls through an internal lock, so these tests need no extra guard.

#![allow(deprecated)] // exercises the deprecated EditSession/SwmrWriter shims (issue #148)
use hdf5_pure::{EditSession, File, FileBuilder};
use tempfile::tempdir;

#[test]
fn c_library_adds_objects_to_our_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("ours.h5");

    // hdf5-pure writes a root dataset and a subgroup holding a dataset.
    let mut b = FileBuilder::new();
    b.create_dataset("root_ds").with_i32_data(&[1, 2, 3]);
    let mut grp = b.create_group("grp");
    grp.create_dataset("inner").with_i32_data(&[4, 5, 6]);
    b.add_group(grp.finish());
    b.write(&path).unwrap();

    // The C library opens it read-write and inserts links into both the root
    // group and the hdf5-pure-created subgroup, then creates a fresh subgroup.
    {
        let f = hdf5::File::open_rw(&path).unwrap();
        f.new_dataset::<i32>()
            .shape((3,))
            .create("added_to_root")
            .unwrap()
            .write(&[7i32, 8, 9])
            .unwrap();
        f.new_dataset::<i32>()
            .shape((2,))
            .create("grp/added_to_subgroup") // insert into OUR subgroup
            .unwrap()
            .write(&[10i32, 11])
            .unwrap();
        f.create_group("new_group").unwrap();
        f.close().unwrap();
    }

    // hdf5-pure reads back every object: the originals and the C-added ones.
    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("root_ds").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    assert_eq!(
        f.dataset("grp/inner").unwrap().read_i32().unwrap(),
        vec![4, 5, 6]
    );
    assert_eq!(
        f.dataset("added_to_root").unwrap().read_i32().unwrap(),
        vec![7, 8, 9]
    );
    assert_eq!(
        f.dataset("grp/added_to_subgroup")
            .unwrap()
            .read_i32()
            .unwrap(),
        vec![10, 11]
    );
}

#[test]
fn c_library_writes_after_an_in_place_edit() {
    // A file hdf5-pure wrote and then edited in place must still be writable by
    // the C library: the EditSession rewrites the affected group headers, and
    // those rewrites must keep (or add) the Group Info message.
    let dir = tempdir().unwrap();
    let path = dir.path().join("edited.h5");

    let mut b = FileBuilder::new();
    b.create_dataset("a").with_i32_data(&[1, 2, 3]);
    b.write(&path).unwrap();

    // Edit in place: adding a group rewrites the root group's object header (via
    // `inspect_group`) and writes a fresh subgroup header (via
    // `fresh_group_region`); both must carry a Group Info message.
    {
        let mut s = EditSession::open(&path).unwrap();
        s.create_group("edited_grp");
        s.commit().unwrap();
    }

    // The C library adds objects to both the rewritten root and the
    // EditSession-created subgroup.
    {
        let f = hdf5::File::open_rw(&path).unwrap();
        f.new_dataset::<i32>()
            .shape((1,))
            .create("c_after_edit")
            .unwrap()
            .write(&[42i32])
            .unwrap();
        f.new_dataset::<i32>()
            .shape((1,))
            .create("edited_grp/c_in_edited")
            .unwrap()
            .write(&[99i32])
            .unwrap();
        f.close().unwrap();
    }

    let f = File::open(&path).unwrap();
    assert_eq!(f.dataset("a").unwrap().read_i32().unwrap(), vec![1, 2, 3]);
    assert_eq!(
        f.dataset("c_after_edit").unwrap().read_i32().unwrap(),
        vec![42]
    );
    assert_eq!(
        f.dataset("edited_grp/c_in_edited")
            .unwrap()
            .read_i32()
            .unwrap(),
        vec![99]
    );
}

#[test]
fn c_library_converts_our_large_group_to_dense() {
    // The Group Info message we write omits the optional max-compact value, so
    // the C library uses its default of 8. When it inserts past that into an
    // hdf5-pure group, it must convert our compact link messages to dense
    // (fractal-heap) link storage. This exercises that conversion path end to
    // end and confirms hdf5-pure reads the dense result the C library produced.
    //
    // Kept at 16 links on purpose: that dense heap fits a single direct block,
    // which hdf5-pure reads. Growing far past this (~125+ links) makes the C
    // library build a multi-direct-block heap with indirect blocks, which the
    // fractal-heap reader cannot yet parse — a separate, pre-existing reader
    // limitation, independent of this write-side fix.
    let dir = tempdir().unwrap();
    let path = dir.path().join("large.h5");

    // hdf5-pure writes a group with 8 compact link messages (== max-compact).
    let mut b = FileBuilder::new();
    let mut g = b.create_group("g");
    for i in 0..8 {
        g.create_dataset(format!("d{i}").as_str())
            .with_i32_data(&[i]);
    }
    b.add_group(g.finish());
    b.write(&path).unwrap();

    // The C library inserts links 9..16, crossing into dense storage.
    {
        let f = hdf5::File::open_rw(&path).unwrap();
        for i in 8..16 {
            f.new_dataset::<i32>()
                .shape((1,))
                .create(format!("g/d{i}").as_str())
                .unwrap()
                .write(&[i])
                .unwrap();
        }
        f.close().unwrap();
    }

    // hdf5-pure reads every link back, now from the dense (fractal-heap) storage
    // the C library wrote.
    let f = File::open(&path).unwrap();
    for i in 0..16 {
        assert_eq!(
            f.dataset(format!("g/d{i}").as_str())
                .unwrap()
                .read_i32()
                .unwrap(),
            vec![i],
            "link d{i} after C dense conversion"
        );
    }
}
