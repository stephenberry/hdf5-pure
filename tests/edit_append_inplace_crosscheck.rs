// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit-pointer targets; skip them on 32-bit so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(not(target_pointer_width = "32"))]
//! Reference-C-library interop for issue #146: the unified append + edit session.
//!
//! Covers the fast, immediate `EditSession::append_inplace` and the staged
//! `set_dataset_attr` / `remove_dataset_attr` against files the C library *wrote*,
//! reading the result back with both the C library and this crate. The
//! make-or-break case is `set_dataset_attr` / `set_group_attr` on a C-written
//! object carrying an *undefined-address* Attribute Info message (which nearly
//! every real-world object has, for attribute creation-order metadata): the shared
//! compact-attribute walkers must accept it rather than mistake it for dense
//! storage.

use hdf5::Extent;
use hdf5::file::LibraryVersion;
use hdf5_pure::{AttrValue, EditSession, Error, File};
use tempfile::tempdir;

/// Create a rank-1 unlimited (Extensible-Array indexed) i32 dataset `name` with the
/// C library under the latest format, seeded with `0..n`, chunk length `chunk`.
fn c_create_unlimited(path: &std::path::Path, name: &str, n: i32, chunk: usize) {
    let file = hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
        .create(path)
        .unwrap();
    let ds = file
        .new_dataset::<i32>()
        .chunk((chunk,))
        .shape((Extent::resizable(n as usize),))
        .create(name)
        .unwrap();
    ds.write(&(0..n).collect::<Vec<_>>()).unwrap();
    file.close().unwrap();
}

fn read_c(path: &std::path::Path, name: &str) -> Vec<i32> {
    let f = hdf5::File::open(path).unwrap();
    let v = f.dataset(name).unwrap().read_raw::<i32>().unwrap();
    f.close().unwrap();
    v
}

fn read_pure(path: &std::path::Path, name: &str) -> Vec<i32> {
    File::open(path)
        .unwrap()
        .dataset(name)
        .unwrap()
        .read_i32()
        .unwrap()
}

// ---- in-place append against C-written files --------------------------------

#[test]
fn append_inplace_to_c_dataset_both_read() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    c_create_unlimited(&path, "d", 8, 4);

    {
        let mut s = EditSession::open(&path).unwrap();
        s.append_inplace_i32("d", &[8, 9, 10, 11, 12]).unwrap(); // any length (unfiltered)
    }

    let expected: Vec<i32> = (0..13).collect();
    assert_eq!(read_pure(&path, "d"), expected);
    assert_eq!(read_c(&path, "d"), expected);
}

#[test]
fn hard_link_aliasing_append_inplace_stays_coherent() {
    // Two hard links to one dataset: appending in place via either path must stay
    // coherent, because the geometry cache is keyed by object-header address, not
    // by path (both links share the one header).
    let dir = tempdir().unwrap();
    let path = dir.path().join("alias.h5");
    c_create_unlimited(&path, "d", 4, 4);
    {
        let file = hdf5::File::open_rw(&path).unwrap();
        file.link_hard("d", "alias").unwrap();
        file.close().unwrap();
    }

    {
        let mut s = EditSession::open(&path).unwrap();
        s.append_inplace_i32("d", &[4, 5]).unwrap(); // via "d"    -> 0..6
        s.append_inplace_i32("alias", &[6, 7]).unwrap(); // via alias -> 0..8
        s.append_inplace_i32("d", &[8]).unwrap(); // via "d"    -> 0..9
    }

    let expected: Vec<i32> = (0..9).collect();
    assert_eq!(read_pure(&path, "d"), expected);
    assert_eq!(read_c(&path, "alias"), expected); // same object, both names
}

// ---- dataset attributes on C-written objects (undefined-AttributeInfo) -------

#[test]
fn c_dataset_with_attribute_info_accepts_set_dataset_attr() {
    // A C-written dataset carrying attributes has an *undefined-address* Attribute
    // Info message (creation-order metadata, not dense storage). `set_dataset_attr`
    // must accept it, preserve the existing attribute, and add the new one — read
    // back by both libraries.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_ds_attr.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&path)
            .unwrap();
        let ds = file.new_dataset::<i32>().shape((4,)).create("d").unwrap();
        ds.write(&[1i32, 2, 3, 4]).unwrap();
        let a = ds.new_attr::<i64>().shape(()).create("orig").unwrap();
        a.write_scalar(&7i64).unwrap();
        file.close().unwrap();
    }

    {
        let mut s = EditSession::open(&path).unwrap();
        s.set_dataset_attr("d", "added", AttrValue::I64(3));
        s.commit().unwrap(); // must NOT be refused as "dense attribute storage"
    }

    // Reference C library sees the data and both attributes.
    let c = hdf5::File::open(&path).unwrap();
    let d = c.dataset("d").unwrap();
    assert_eq!(d.read_raw::<i32>().unwrap(), vec![1, 2, 3, 4]);
    let orig: i64 = d.attr("orig").unwrap().read_scalar().unwrap();
    let added: i64 = d.attr("added").unwrap().read_scalar().unwrap();
    assert_eq!((orig, added), (7, 3));
    c.close().unwrap();

    // Pure reader agrees.
    let f = File::open(&path).unwrap();
    let attrs = f.dataset("d").unwrap().attrs().unwrap();
    assert_eq!(attrs.get("orig"), Some(&AttrValue::I64(7)));
    assert_eq!(attrs.get("added"), Some(&AttrValue::I64(3)));
}

#[test]
fn c_group_with_attribute_info_accepts_set_group_attr() {
    // Regression for the shared-walker fix on the *group* path: a C-written group
    // carrying an attribute also has an undefined-address Attribute Info message,
    // and `set_group_attr` (which pre-dates this fix) must no longer spuriously
    // refuse it.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_grp_attr.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&path)
            .unwrap();
        let g = file.create_group("grp").unwrap();
        let a = g.new_attr::<i64>().shape(()).create("orig").unwrap();
        a.write_scalar(&11i64).unwrap();
        // A dataset so the file has content besides the group.
        file.new_dataset::<i32>()
            .shape((2,))
            .create("d")
            .unwrap()
            .write(&[5i32, 6])
            .unwrap();
        file.close().unwrap();
    }

    {
        let mut s = EditSession::open(&path).unwrap();
        s.set_group_attr("grp", "added", AttrValue::I64(22));
        s.commit().unwrap();
    }

    let c = hdf5::File::open(&path).unwrap();
    let g = c.group("grp").unwrap();
    let orig: i64 = g.attr("orig").unwrap().read_scalar().unwrap();
    let added: i64 = g.attr("added").unwrap().read_scalar().unwrap();
    assert_eq!((orig, added), (11, 22));
}

#[test]
fn set_dataset_attr_on_chunked_dataset_c_reads() {
    // An attribute edit on a *chunked* (Extensible-Array) dataset relocates the
    // header while preserving the data-layout message verbatim, so the chunk data
    // and index stay in place. Both libraries must read the grown-attribute dataset
    // and its data correctly, and a subsequent in-place append must still work
    // (the header address changed, so the geometry cache re-locates).
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_chunked_attr.h5");
    c_create_unlimited(&path, "d", 8, 4);

    {
        let mut s = EditSession::open(&path).unwrap();
        s.set_dataset_attr("d", "checked", AttrValue::I64(1));
        s.commit().unwrap();
        s.append_inplace_i32("d", &[8, 9, 10, 11]).unwrap(); // re-locates, then grows
    }

    let expected: Vec<i32> = (0..12).collect();
    assert_eq!(read_pure(&path, "d"), expected);
    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(c.dataset("d").unwrap().read_raw::<i32>().unwrap(), expected);
    let checked: i64 = c
        .dataset("d")
        .unwrap()
        .attr("checked")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(checked, 1);
}

#[test]
fn set_dataset_attr_multi_hard_link_refused() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_ds_multilink.h5");
    c_create_unlimited(&path, "d", 4, 4);
    {
        let file = hdf5::File::open_rw(&path).unwrap();
        file.link_hard("d", "alias").unwrap();
        file.close().unwrap();
    }
    let before = std::fs::read(&path).unwrap();

    {
        let mut s = EditSession::open(&path).unwrap();
        s.set_dataset_attr("d", "x", AttrValue::I64(1));
        let err = s.commit().unwrap_err();
        assert!(
            matches!(err, Error::EditUnsupported(_))
                && err.to_string().contains("single hard link"),
            "expected single-hard-link refusal, got: {err}"
        );
    }
    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "file modified on refusal"
    );
}

// ---- combined mixed edits ---------------------------------------------------

#[test]
fn combined_mixed_edits_c_readable() {
    // One long-lived session mixes an immediate in-place append, a staged group
    // creation, a staged dataset-attribute edit, and a staged recursive delete —
    // then more in-place appends after the commit. The reference C library reads
    // every result correctly.
    let dir = tempdir().unwrap();
    let path = dir.path().join("combined.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .chunk((4,))
            .shape((Extent::resizable(4),))
            .create("log")
            .unwrap()
            .write(&[0i32, 1, 2, 3])
            .unwrap();
        file.new_dataset::<i32>()
            .shape((3,))
            .create("keep")
            .unwrap()
            .write(&[7i32, 8, 9])
            .unwrap();
        let g = file.create_group("old").unwrap();
        g.new_dataset::<i32>()
            .shape((2,))
            .create("inner")
            .unwrap()
            .write(&[1i32, 2])
            .unwrap();
        file.close().unwrap();
    }

    {
        let mut s = EditSession::open(&path).unwrap();
        s.append_inplace_i32("log", &[4, 5, 6, 7]).unwrap(); // immediate -> 0..8
        s.create_group("run"); // staged
        s.set_dataset_attr("keep", "checked", AttrValue::I64(1)); // staged
        s.delete("old"); // staged recursive delete
        s.commit().unwrap();
        s.append_inplace_i32("log", &[8, 9]).unwrap(); // immediate -> 0..10
    }

    // Reference C library.
    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(
        c.dataset("log").unwrap().read_raw::<i32>().unwrap(),
        (0..10).collect::<Vec<_>>()
    );
    assert_eq!(
        c.dataset("keep").unwrap().read_raw::<i32>().unwrap(),
        vec![7, 8, 9]
    );
    let checked: i64 = c
        .dataset("keep")
        .unwrap()
        .attr("checked")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(checked, 1);
    assert!(c.group("run").is_ok(), "created group missing");
    assert!(c.group("old").is_err(), "deleted group still present");
    c.close().unwrap();

    // Pure reader agrees on the grown dataset.
    assert_eq!(read_pure(&path, "log"), (0..10).collect::<Vec<_>>());
}
