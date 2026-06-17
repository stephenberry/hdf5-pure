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

#[test]
fn deleting_chunked_datasets_in_place_stays_c_readable() {
    // Reclaiming chunked storage on delete (issue #77) must leave a file the
    // reference C library still reads. The starter is written by the C library
    // with HDF5 1.8 bounds, so its chunked dataset uses a *B-tree v1* index — the
    // foreign layout the editor's own writer never emits. The editor deletes it
    // in place, then churns an editor-written (Fixed Array) chunked dataset to
    // prove its storage is reclaimed and reused rather than leaked. Finally both
    // readers see only the contiguous survivor in the shrunken, valid file.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_chunked_delete.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V18, LibraryVersion::V18))
            .create(&path)
            .unwrap();
        file.new_dataset::<f64>()
            .shape((3,))
            .create("keep")
            .unwrap()
            .write(&[1.0f64, 2.0, 3.0])
            .unwrap();
        let chunked: Vec<i32> = (0..2048).collect();
        file.new_dataset::<i32>()
            .shape((2048,))
            .chunk((256,)) // 8 chunks of 256 i32 = 1024 bytes each, B-tree v1
            .create("c_chunked")
            .unwrap()
            .write(&chunked)
            .unwrap();
        file.close().unwrap();
    }

    let with_c_chunked = std::fs::metadata(&path).unwrap().len();

    {
        let mut session = EditSession::open(&path).unwrap();
        // Reclaim the C-written B-tree-v1 chunked dataset.
        session.delete("c_chunked");
        session.commit().unwrap();
        // Re-add *contiguous* datasets sized to the reclaimed 1024-byte chunk
        // holes. (Chunked adds always append their blob, so contiguous adds are
        // what exercise reuse of a freed interior hole.) Six 1024-byte blocks fit
        // within the ~8 KB the delete freed, so the file barely grows; if the
        // B-tree-v1 storage had leaked instead, they would all append and the
        // file would grow by ~6 KB, failing the bound below.
        for i in 0..6 {
            session
                .create_dataset(&format!("r{i}"))
                .with_i32_data(&vec![i; 256]); // 256 i32 = 1024 bytes, one chunk hole
            session.commit().unwrap();
        }
    }
    let after = std::fs::metadata(&path).unwrap().len();
    assert!(
        after < with_c_chunked + 4096,
        "deleting the C-written B-tree-v1 chunked dataset must reclaim its storage \
         for the contiguous re-adds to reuse, not leak it (was {with_c_chunked}, \
         after six 1 KiB re-adds {after})"
    );

    // hdf5-pure: the deleted dataset is gone, the survivor and re-adds read back,
    // and the recorded end-of-file matches the physical size.
    let f = File::open(&path).unwrap();
    assert_eq!(f.file_size(), std::fs::metadata(&path).unwrap().len());
    assert!(f.dataset("c_chunked").is_err());
    assert_eq!(
        f.dataset("keep").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
    for i in 0..6 {
        assert_eq!(
            f.dataset(&format!("r{i}")).unwrap().read_i32().unwrap(),
            vec![i; 256]
        );
    }

    // The reference C library reads the reclaimed file too.
    let c = hdf5::File::open(&path).unwrap();
    assert!(
        c.dataset("c_chunked").is_err(),
        "deleted chunked dataset still present (C)"
    );
    assert_eq!(
        c.dataset("keep").unwrap().read_raw::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
    for i in 0..6 {
        assert_eq!(
            c.dataset(&format!("r{i}"))
                .unwrap()
                .read_raw::<i32>()
                .unwrap(),
            vec![i; 256]
        );
    }
}

#[test]
fn deleting_one_of_several_hard_links_keeps_the_survivor() {
    // Issue #77 review (finding #1): an HDF5 object can have several hard links.
    // Deleting ONE link must not reclaim the object's storage while another link
    // still references it — freeing it would corrupt the survivor once the bytes
    // are reused. Covers a chunked (Fixed Array) object and a contiguous one,
    // each linked twice; one link of each is deleted, then churn forces the
    // allocator to reuse anything wrongly freed.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_hardlink_survivor.h5");
    let chunked: Vec<i32> = (0..512).collect();
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .shape((512,))
            .chunk((64,))
            .create("chunked_orig")
            .unwrap()
            .write(&chunked)
            .unwrap();
        file.link_hard("/chunked_orig", "chunked_alias").unwrap();
        file.new_dataset::<f64>()
            .shape((4,))
            .create("contig_orig")
            .unwrap()
            .write(&[1.0f64, 2.0, 3.0, 4.0])
            .unwrap();
        file.link_hard("/contig_orig", "contig_alias").unwrap();
        file.close().unwrap();
    }

    {
        let mut session = EditSession::open(&path).unwrap();
        session.delete("chunked_orig");
        session.delete("contig_orig");
        session.commit().unwrap();
        for i in 0..6 {
            session
                .create_dataset(&format!("f{i}"))
                .with_i32_data(&vec![i; 300]);
            session.commit().unwrap();
            session.delete(&format!("f{i}"));
            session.commit().unwrap();
        }
    }

    // The surviving aliases must still read their data through both readers; the
    // deleted link names are gone.
    let f = File::open(&path).unwrap();
    assert!(f.dataset("chunked_orig").is_err());
    assert!(f.dataset("contig_orig").is_err());
    assert_eq!(
        f.dataset("chunked_alias").unwrap().read_i32().unwrap(),
        chunked
    );
    assert_eq!(
        f.dataset("contig_alias").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );

    let c = hdf5::File::open(&path).unwrap();
    assert!(c.dataset("chunked_orig").is_err());
    assert_eq!(
        c.dataset("chunked_alias")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        chunked
    );
    assert_eq!(
        c.dataset("contig_alias")
            .unwrap()
            .read_raw::<f64>()
            .unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn deleting_all_hard_links_to_an_object_in_one_commit_is_safe() {
    // Issue #77 review (finding #2): deleting every hard link to a chunked object
    // in a single commit must not double-free its storage (a debug-build panic in
    // the free list) nor corrupt the file. The storage is conservatively left as
    // dead bytes; the file stays valid and the unrelated survivor reads back.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_hardlink_all.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .shape((512,))
            .chunk((64,))
            .create("orig")
            .unwrap()
            .write(&(0..512).collect::<Vec<i32>>())
            .unwrap();
        file.link_hard("/orig", "alias").unwrap();
        file.new_dataset::<f64>()
            .shape((2,))
            .create("keep")
            .unwrap()
            .write(&[9.0f64, 8.0])
            .unwrap();
        file.close().unwrap();
    }
    {
        let mut session = EditSession::open(&path).unwrap();
        session.delete("orig");
        session.delete("alias"); // both links to the same object, one commit
        session.commit().unwrap();
    }
    let f = File::open(&path).unwrap();
    assert!(f.dataset("orig").is_err());
    assert!(f.dataset("alias").is_err());
    assert_eq!(
        f.dataset("keep").unwrap().read_f64().unwrap(),
        vec![9.0, 8.0]
    );
    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(
        c.dataset("keep").unwrap().read_raw::<f64>().unwrap(),
        vec![9.0, 8.0]
    );
}

#[test]
fn cross_file_copy_from_read_by_c_library() {
    // A cross-file `copy_from` must produce a destination the reference C library
    // can read: the copied object's verbatim headers, data, and attributes have to
    // be valid in the new file, not just resolvable by this crate's own reader.
    let dir = tempdir().unwrap();
    let src_path = dir.path().join("xsrc.h5");
    let dst_path = dir.path().join("xdst.h5");

    // Source: an attributed dataset and a group subtree. Written with this crate
    // (compact attributes, which the verbatim copy reproduces — the C library
    // stores even a single latest-format attribute densely, which the copy path
    // does not yet handle for either same- or cross-file copies).
    {
        let mut b = FileBuilder::new();
        let ds = b.create_dataset("calibration");
        ds.with_f64_data(&[0.99, 1.0, 1.01]);
        ds.set_attr("revision", AttrValue::I64(7));
        let mut bundle = b.create_group("bundle");
        bundle.create_dataset("inner").with_i32_data(&[5, 6]);
        b.add_group(bundle.finish());
        b.write(&src_path).unwrap();
    }

    // Destination: a C-written starter (alpha, doomed, grp/beta).
    write_c_starter(&dst_path, LibraryVersion::V110, LibraryVersion::latest());

    {
        let source = File::open(&src_path).unwrap();
        let mut session = EditSession::open(&dst_path).unwrap();
        session
            .copy_from(&source, "calibration", "calibration")
            .unwrap();
        session.copy_from(&source, "bundle", "bundle").unwrap();
        session.commit().unwrap();
    } // drop the editor (release its exclusive lock) before reading back

    // hdf5-pure reader.
    let f = File::open(&dst_path).unwrap();
    assert_eq!(
        f.dataset("calibration").unwrap().read_f64().unwrap(),
        vec![0.99, 1.0, 1.01]
    );
    assert_eq!(
        f.dataset("bundle/inner").unwrap().read_i32().unwrap(),
        vec![5, 6]
    );

    // Reference C library reader — the interop proof.
    let c = hdf5::File::open(&dst_path).unwrap();
    assert_eq!(
        c.dataset("calibration").unwrap().read_raw::<f64>().unwrap(),
        vec![0.99, 1.0, 1.01]
    );
    let revision: i64 = c
        .dataset("calibration")
        .unwrap()
        .attr("revision")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(revision, 7);
    assert_eq!(
        c.dataset("bundle/inner")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        vec![5, 6]
    );
    // The destination's pre-existing objects survive.
    assert_eq!(
        c.dataset("alpha").unwrap().read_raw::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
}

#[test]
fn cross_file_copy_from_c_written_attributed_dataset() {
    // A C-library-written object with a few attributes stores them compactly but
    // also carries an Attribute Info message with an *undefined* heap address; the
    // copy path must treat that as compact (not dense), so `copy_from` succeeds and
    // the C library reads the copy back — attributes included.
    let dir = tempdir().unwrap();
    let src_path = dir.path().join("c_attr_src.h5");
    let dst_path = dir.path().join("c_attr_dst.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&src_path)
            .unwrap();
        let ds = file
            .new_dataset::<f64>()
            .shape((3,))
            .create("calibration")
            .unwrap();
        ds.write(&[0.99f64, 1.0, 1.01]).unwrap();
        ds.new_attr::<i64>()
            .shape(())
            .create("revision")
            .unwrap()
            .write_scalar(&7i64)
            .unwrap();
        ds.new_attr::<f64>()
            .shape(())
            .create("gain")
            .unwrap()
            .write_scalar(&2.5f64)
            .unwrap();
        file.close().unwrap();
    }
    write_c_starter(&dst_path, LibraryVersion::V110, LibraryVersion::latest());

    {
        let source = File::open(&src_path).unwrap();
        let mut session = EditSession::open(&dst_path).unwrap();
        session
            .copy_from(&source, "calibration", "calibration")
            .unwrap();
        session.commit().unwrap();
    }

    let c = hdf5::File::open(&dst_path).unwrap();
    assert_eq!(
        c.dataset("calibration").unwrap().read_raw::<f64>().unwrap(),
        vec![0.99, 1.0, 1.01]
    );
    let revision: i64 = c
        .dataset("calibration")
        .unwrap()
        .attr("revision")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(revision, 7);
    let gain: f64 = c
        .dataset("calibration")
        .unwrap()
        .attr("gain")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(gain, 2.5);
}

#[test]
fn same_file_copy_of_c_written_attributed_object() {
    // The same compact-attribute fix applies to the in-file `copy`: a C-written
    // attributed object is now copyable in place and read back by the C library.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_attr_infile.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&path)
            .unwrap();
        let ds = file.new_dataset::<i32>().shape((2,)).create("src").unwrap();
        ds.write(&[10i32, 20]).unwrap();
        ds.new_attr::<i64>()
            .shape(())
            .create("tag")
            .unwrap()
            .write_scalar(&99i64)
            .unwrap();
        file.close().unwrap();
    }

    {
        let mut session = EditSession::open(&path).unwrap();
        session.copy("src", "dup");
        session.commit().unwrap();
    }

    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(
        c.dataset("dup").unwrap().read_raw::<i32>().unwrap(),
        vec![10, 20]
    );
    let tag: i64 = c
        .dataset("dup")
        .unwrap()
        .attr("tag")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(tag, 99);
}

#[test]
fn cross_file_copy_from_rejects_dense_attributes() {
    // Above the compact threshold (8 attributes) the C library stores attributes
    // densely — a real fractal heap, whose address would dangle in another file —
    // so a verbatim cross-file copy is refused, leaving the destination untouched.
    let dir = tempdir().unwrap();
    let src_path = dir.path().join("dense_src.h5");
    let dst_path = dir.path().join("dense_dst.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&src_path)
            .unwrap();
        let ds = file.new_dataset::<i32>().shape((1,)).create("ds").unwrap();
        ds.write(&[1i32]).unwrap();
        for i in 0..12 {
            ds.new_attr::<i64>()
                .shape(())
                .create(format!("a{i}").as_str())
                .unwrap()
                .write_scalar(&(i as i64))
                .unwrap();
        }
        file.close().unwrap();
    }
    write_c_starter(&dst_path, LibraryVersion::V110, LibraryVersion::latest());
    let dst_before = std::fs::read(&dst_path).unwrap();

    {
        let source = File::open(&src_path).unwrap();
        let mut session = EditSession::open(&dst_path).unwrap();
        let err = session.copy_from(&source, "ds", "dup").unwrap_err();
        assert!(err.to_string().contains("dense"), "got: {err}");
    }
    assert_eq!(std::fs::read(&dst_path).unwrap(), dst_before);
}
