// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit-pointer targets; skip them on 32-bit so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(not(target_pointer_width = "32"))]
//! Cross-validation for in-place editing against the reference C library
//! (issue #32): files the C library *writes* are edited in place by
//! `EditSession`, and the result is read back by both `hdf5-pure` and the C
//! library. This proves the editor works on files it did not itself produce, in
//! both the HDF5 1.8 (v2 superblock) and 1.10+ (v3 superblock) formats, and
//! including headers the C library splits across multiple chunks (which the
//! editor collapses into a single chunk on rewrite).
//!
//! The default earliest format (version 0 superblock with symbol-table groups)
//! is also editable: each group on the edited path is converted to the latest
//! compact-link format on rewrite, the superblock's root symbol-table entry is
//! repointed, and the result is read back correctly by the C library.

use hdf5::file::LibraryVersion;
use hdf5_pure::{AttrValue, EditSession, File, FileBuilder, ScaleOffset};
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

    {
        let mut session = EditSession::open(&path).unwrap();
        stage_edits(&mut session);
        session.commit().unwrap();
    } // drop the editor (release its exclusive lock) before reading back

    assert_edits_applied(&path);
}

#[test]
fn c_written_v2_file_edited_in_place() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_v2.h5");
    write_c_starter(&path, LibraryVersion::V18, LibraryVersion::V18);
    assert_eq!(File::open(&path).unwrap().superblock().version, 2);

    {
        let mut session = EditSession::open(&path).unwrap();
        stage_edits(&mut session);
        session.commit().unwrap();
    } // drop the editor (release its exclusive lock) before reading back

    assert_edits_applied(&path);
}

#[test]
fn c_written_v3_file_edited_in_place() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_v3.h5");
    write_c_starter(&path, LibraryVersion::V110, LibraryVersion::latest());
    assert_eq!(File::open(&path).unwrap().superblock().version, 3);

    {
        let mut session = EditSession::open(&path).unwrap();
        stage_edits(&mut session);
        session.commit().unwrap();
    } // drop the editor (release its exclusive lock) before reading back

    assert_edits_applied(&path);
}

#[test]
fn c_multichunk_group_header_is_collapsed_and_edited() {
    // The C library lays a group header out across multiple chunks once it holds
    // enough messages; several root attributes reliably force that. The editor
    // must collapse the continuation chunks into one header on rewrite, preserve
    // the existing messages (the attributes), and apply the edit.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_multichunk.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&path)
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
        // Several root-group attributes push the root header past one chunk.
        for i in 0..6 {
            let a = file
                .new_attr::<i64>()
                .shape(())
                .create(format!("meta{i}").as_str())
                .unwrap();
            a.write_scalar(&(i as i64 * 100)).unwrap();
        }
        file.close().unwrap();
    }

    {
        let mut session = EditSession::open(&path).unwrap();
        stage_edits(&mut session);
        session.commit().unwrap();
    }

    assert_edits_applied(&path);

    // The C-written root attributes survived the multi-chunk -> single-chunk
    // rewrite of the root header (verified by the C library).
    let c = hdf5::File::open(&path).unwrap();
    for i in 0..6 {
        let v: i64 = c.attr(&format!("meta{i}")).unwrap().read_scalar().unwrap();
        assert_eq!(
            v,
            i as i64 * 100,
            "root attribute meta{i} lost or corrupted"
        );
    }
}

#[test]
fn c_v0_symboltable_file_edited_then_read_by_c_library() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_v0.h5");
    // Earliest low bound yields a version 0 superblock with symbol-table groups.
    write_c_starter(&path, LibraryVersion::Earliest, LibraryVersion::V18);
    assert!(
        File::open(&path).unwrap().superblock().version <= 1,
        "expected a v0/v1 superblock from the earliest libver bound"
    );

    // Add at root, add into the (symbol-table) group, and delete a root dataset.
    // (Copy of the existing v1 objects is not supported, so it is not staged.)
    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .create_dataset("added")
            .with_f64_data(&[100.0, 200.0]);
        session
            .create_dataset("grp/gamma")
            .with_i32_data(&[1, 2, 3]);
        session.delete("doomed");
        session.commit().unwrap();
    }

    // The superblock stays version 0; the edited groups were converted to v2.
    let f = File::open(&path).unwrap();
    assert!(f.superblock().version <= 1);
    assert_eq!(
        f.dataset("alpha").unwrap().read_f64().unwrap(),
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
    assert!(f.dataset("doomed").is_err());

    // The reference C library reads the edited version-0 file and agrees.
    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(
        c.dataset("alpha").unwrap().read_raw::<f64>().unwrap(),
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
    assert!(c.dataset("doomed").is_err());
}

#[test]
fn c_v0_root_attributes_survive_conversion() {
    // A version-0 root is a symbol-table (v1) group; editing converts it to v2.
    // Its existing attributes must survive that conversion, verified by the C
    // library reading them back.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_v0_attrs.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::Earliest, LibraryVersion::V18))
            .create(&path)
            .unwrap();
        file.new_dataset::<f64>()
            .shape((2,))
            .create("d")
            .unwrap()
            .write(&[1.0f64, 2.0])
            .unwrap();
        file.new_attr::<i64>()
            .shape(())
            .create("tag")
            .unwrap()
            .write_scalar(&77i64)
            .unwrap();
        file.close().unwrap();
    }
    assert!(File::open(&path).unwrap().superblock().version <= 1);

    {
        let mut session = EditSession::open(&path).unwrap();
        session.create_dataset("extra").with_i32_data(&[9]);
        session.commit().unwrap();
    }

    let c = hdf5::File::open(&path).unwrap();
    let tag: i64 = c.attr("tag").unwrap().read_scalar().unwrap();
    assert_eq!(tag, 77, "root attribute lost converting the v0 group to v2");
    assert_eq!(
        c.dataset("d").unwrap().read_raw::<f64>().unwrap(),
        vec![1.0, 2.0]
    );
    assert_eq!(
        c.dataset("extra").unwrap().read_raw::<i32>().unwrap(),
        vec![9]
    );
}

#[test]
fn c_library_reads_group_attributes_edited_in_place() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_group_attrs.h5");
    let mut b = FileBuilder::new();
    let mut g = b.create_group("grp");
    g.set_attr("count", AttrValue::I64(1));
    g.set_attr("drop", AttrValue::I64(9));
    b.add_group(g.finish());
    b.write(&path).unwrap();

    {
        let mut session = EditSession::open(&path).unwrap();
        session.set_group_attr("grp", "count", AttrValue::I64(2));
        session.set_group_attr("grp", "added", AttrValue::I64(3));
        session.remove_group_attr("grp", "drop");
        session.create_group("new_grp");
        session.set_group_attr("new_grp", "tag", AttrValue::I64(55));
        session.commit().unwrap();
    }

    let c = hdf5::File::open(&path).unwrap();
    let grp = c.group("grp").unwrap();
    let count: i64 = grp.attr("count").unwrap().read_scalar().unwrap();
    let added: i64 = grp.attr("added").unwrap().read_scalar().unwrap();
    assert_eq!(count, 2);
    assert_eq!(added, 3);
    assert!(
        grp.attr("drop").is_err(),
        "removed group attribute still present"
    );
    let tag: i64 = c
        .group("new_grp")
        .unwrap()
        .attr("tag")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(tag, 55);
}

#[test]
fn free_space_reuse_and_truncation_stay_c_readable() {
    // Free-space management (issue #21): within one session, add a large dataset
    // then delete it. The freed blocks and superseded headers are reclaimed and
    // the file is truncated. The reference C library must still read the survivor
    // correctly from the shrunken file, and its end-of-file must be consistent.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_freespace.h5");
    write_c_starter(&path, LibraryVersion::V110, LibraryVersion::latest());
    let size_start = std::fs::metadata(&path).unwrap().len();

    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .create_dataset("bulk")
            .with_f64_data(&vec![9.0; 2048]);
        session.commit().unwrap();
        let size_with_bulk = std::fs::metadata(&path).unwrap().len();
        assert!(size_with_bulk > size_start);

        session.delete("bulk");
        session.commit().unwrap();
        let size_after = std::fs::metadata(&path).unwrap().len();
        assert!(
            size_after < size_with_bulk,
            "deleting the bulk dataset should shrink the file (was {size_with_bulk}, now {size_after})"
        );
    }

    // hdf5-pure: end-of-file matches the truncated physical size.
    let f = File::open(&path).unwrap();
    assert_eq!(f.file_size(), std::fs::metadata(&path).unwrap().len());
    assert!(f.dataset("bulk").is_err());

    // The reference C library reads the shrunken file and the survivors intact.
    let c = hdf5::File::open(&path).unwrap();
    assert!(
        c.dataset("bulk").is_err(),
        "deleted dataset still present (C)"
    );
    assert_eq!(
        c.dataset("alpha").unwrap().read_raw::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
    assert_eq!(
        c.dataset("grp/beta").unwrap().read_raw::<i32>().unwrap(),
        vec![10, 20, 30, 40]
    );
}

#[test]
fn chunked_and_filtered_datasets_added_in_place_are_c_readable() {
    // Issue #76: chunked / filtered / extensible datasets added in place to a
    // file the editor did not write must be read back faithfully by the
    // reference C library — the real interop proof that their chunk data, index,
    // and filter pipeline are emitted correctly.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_chunked_edit.h5");
    write_c_starter(&path, LibraryVersion::V110, LibraryVersion::latest());

    let f64_data: Vec<f64> = (0..400).map(|i| i as f64 * 0.25).collect();
    let i32_data: Vec<i32> = (0..256).map(|i| 1000 + (i % 11)).collect();
    let ext_data: Vec<i32> = (0..128).collect();

    {
        let mut session = EditSession::open(&path).unwrap();
        session
            .create_dataset("deflated")
            .with_f64_data(&f64_data)
            .with_chunks(&[100])
            .with_deflate(6);
        session
            .create_dataset("shuffled")
            .with_f64_data(&f64_data)
            .with_chunks(&[64])
            .with_shuffle()
            .with_deflate(4);
        session
            .create_dataset("checked")
            .with_i32_data(&i32_data)
            .with_chunks(&[80])
            .with_fletcher32();
        session
            .create_dataset("scaled")
            .with_i32_data(&i32_data)
            .with_chunks(&[80])
            .with_scale_offset(ScaleOffset::Integer(0));
        // Into a group the C library wrote, exercising header relocation.
        session
            .create_dataset("grp/grid")
            .with_i32_data(&(0..(6 * 4)).collect::<Vec<i32>>())
            .with_shape(&[6, 4])
            .with_chunks(&[4, 3]);
        // Extensible (unlimited) dataset → Extensible-Array chunk index.
        session
            .create_dataset("stream")
            .with_i32_data(&ext_data)
            .with_shape(&[128])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[32]);
        session.commit().unwrap();
    }

    // hdf5-pure reads everything back.
    {
        let f = File::open(&path).unwrap();
        assert_eq!(f.dataset("deflated").unwrap().read_f64().unwrap(), f64_data);
        assert_eq!(f.dataset("shuffled").unwrap().read_f64().unwrap(), f64_data);
        assert_eq!(f.dataset("checked").unwrap().read_i32().unwrap(), i32_data);
        assert_eq!(f.dataset("scaled").unwrap().read_i32().unwrap(), i32_data);
        assert_eq!(
            f.dataset("grp/grid").unwrap().read_i32().unwrap(),
            (0..(6 * 4)).collect::<Vec<i32>>()
        );
        assert_eq!(f.dataset("stream").unwrap().read_i32().unwrap(), ext_data);
        // The C-written survivors are intact.
        assert_eq!(
            f.dataset("alpha").unwrap().read_f64().unwrap(),
            vec![1.0, 2.0, 3.0]
        );
    }

    // The reference C library reads the filtered/chunked/extensible additions and
    // sees them as chunked storage.
    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(
        c.dataset("deflated").unwrap().read_raw::<f64>().unwrap(),
        f64_data
    );
    assert_eq!(
        c.dataset("shuffled").unwrap().read_raw::<f64>().unwrap(),
        f64_data
    );
    assert_eq!(
        c.dataset("checked").unwrap().read_raw::<i32>().unwrap(),
        i32_data
    );
    assert_eq!(
        c.dataset("scaled").unwrap().read_raw::<i32>().unwrap(),
        i32_data
    );
    assert_eq!(
        c.dataset("grp/grid").unwrap().read_raw::<i32>().unwrap(),
        (0..(6 * 4)).collect::<Vec<i32>>()
    );
    assert_eq!(
        c.dataset("stream").unwrap().read_raw::<i32>().unwrap(),
        ext_data
    );
    for name in [
        "deflated", "shuffled", "checked", "scaled", "grp/grid", "stream",
    ] {
        assert!(
            c.dataset(name).unwrap().chunk().is_some(),
            "C library does not see {name} as chunked"
        );
    }
    // The C-written original is untouched.
    assert_eq!(
        c.dataset("alpha").unwrap().read_raw::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
}
