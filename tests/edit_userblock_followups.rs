//! Delete and copy editing on files that carry a userblock (non-zero base
//! address) — the follow-up parity work to the userblock slice of issue #104.
//!
//! Both operations rewrite or reclaim on-disk addresses, all of which are stored
//! relative to the userblock base on such a file. Each test edits a synthetic
//! userblock file and reads the result back through the pure-Rust reader, and
//! every test asserts the 512-byte userblock survives the edit byte-for-byte.
//!
//! (The resizing-overwrite parity for the contiguous and compact layouts is
//! exercised in `edit_userblock_crosscheck.rs`: only the reference C library can
//! create the never-written-contiguous and compact-layout fixtures those paths
//! need.)

use hdf5_pure::{AttrValue, EditSession, File, FileBuilder};

const UB: usize = 512;
const MARKER: &[u8] = b"USERBLOCK-FOLLOWUP-104";

/// Stamp a recognizable marker across the userblock region of `bytes` and return
/// the 512-byte userblock as written, for later byte-for-byte comparison.
fn stamp_userblock(bytes: &mut [u8]) -> Vec<u8> {
    bytes[..MARKER.len()].copy_from_slice(MARKER);
    bytes[UB - 1] = 0xAB;
    bytes[..UB].to_vec()
}

fn assert_userblock_unchanged(path: &std::path::Path, original: &[u8]) {
    let after = std::fs::read(path).unwrap();
    assert_eq!(
        &after[..UB],
        original,
        "userblock bytes changed across the edit"
    );
}

/// Build a userblock file with two root datasets and a nested group+dataset.
fn build_userblock_file(path: &std::path::Path) -> Vec<u8> {
    let mut b = FileBuilder::new();
    b.with_userblock(UB as u64);
    b.create_dataset("alpha")
        .with_f64_data(&[1.0, 2.0, 3.0, 4.0]);
    b.create_dataset("beta").with_i32_data(&[10, 20, 30]);
    let mut g = b.create_group("grp");
    g.create_dataset("inner").with_f64_data(&[7.5, 8.5]);
    b.add_group(g.finish());
    let mut bytes = b.finish().unwrap();
    let userblock = stamp_userblock(&mut bytes);
    std::fs::write(path, &bytes).unwrap();
    userblock
}

// ---- delete ----

#[test]
fn userblock_delete_dataset_roundtrip() {
    let path = std::env::temp_dir().join("hdf5_pure_ub_fu_delete_ds.h5");
    let userblock = build_userblock_file(&path);

    {
        let mut s = EditSession::open(&path).unwrap();
        s.delete("alpha");
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert!(file.dataset("alpha").is_err(), "alpha should be deleted");
    // Neighbours survive.
    assert_eq!(
        file.dataset("beta").unwrap().read_i32().unwrap(),
        vec![10, 20, 30]
    );
    assert_eq!(
        file.dataset("grp/inner").unwrap().read_f64().unwrap(),
        vec![7.5, 8.5]
    );
    let mut datasets = file.root().datasets().unwrap();
    datasets.sort();
    assert_eq!(datasets, vec!["beta".to_string()]);
    assert_userblock_unchanged(&path, &userblock);

    std::fs::remove_file(&path).ok();
}

#[test]
fn userblock_delete_group_subtree_roundtrip() {
    // Deleting a group reclaims its whole subtree (its header plus the nested
    // dataset's header and data). On a userblock file every child link and data
    // address is base-relative, so the subtree walk must re-absolutize them.
    let path = std::env::temp_dir().join("hdf5_pure_ub_fu_delete_grp.h5");
    let userblock = build_userblock_file(&path);

    {
        let mut s = EditSession::open(&path).unwrap();
        s.delete("grp");
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert!(file.group("grp").is_err(), "grp should be deleted");
    assert!(file.dataset("grp/inner").is_err());
    assert!(file.root().groups().unwrap().is_empty());
    assert_eq!(
        file.dataset("alpha").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_userblock_unchanged(&path, &userblock);

    std::fs::remove_file(&path).ok();
}

#[test]
fn userblock_delete_chunked_dataset_roundtrip() {
    // Deleting a chunked/filtered dataset reclaims its chunk index and chunk data
    // blocks via the base-aware `chunked_storage_spans`.
    let path = std::env::temp_dir().join("hdf5_pure_ub_fu_delete_chunk.h5");
    let mut b = FileBuilder::new();
    b.with_userblock(UB as u64);
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    let chunked: Vec<f64> = (0..800).map(|i| (i % 7) as f64 * 0.5).collect();
    b.create_dataset("c")
        .with_f64_data(&chunked)
        .with_shape(&[800])
        .with_chunks(&[50])
        .with_deflate(6);
    let mut bytes = b.finish().unwrap();
    let userblock = stamp_userblock(&mut bytes);
    std::fs::write(&path, &bytes).unwrap();

    {
        let mut s = EditSession::open(&path).unwrap();
        s.delete("c");
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert!(file.dataset("c").is_err(), "c should be deleted");
    assert_eq!(
        file.dataset("keep").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    assert_userblock_unchanged(&path, &userblock);

    std::fs::remove_file(&path).ok();
}

#[test]
fn userblock_delete_then_reuse_freed_space() {
    // Deleting a sizeable contiguous dataset frees its data extent; a later commit
    // in the same session should reuse that freed hole for a contiguous add rather
    // than only appending. The reused write lands at the freed *absolute* offset,
    // so a base mistake in `collect_free_spans` would either leak (no reuse) or, far
    // worse, free a still-live region the reuse then corrupts.
    let path = std::env::temp_dir().join("hdf5_pure_ub_fu_delete_reuse.h5");
    let mut b = FileBuilder::new();
    b.with_userblock(UB as u64);
    let big: Vec<f64> = (0..256).map(|i| i as f64).collect();
    b.create_dataset("big").with_f64_data(&big);
    b.create_dataset("keep").with_i32_data(&[7, 8, 9]);
    let mut bytes = b.finish().unwrap();
    let userblock = stamp_userblock(&mut bytes);
    std::fs::write(&path, &bytes).unwrap();

    let reuse: Vec<f64> = (0..64).map(|i| (i as f64) * -1.5).collect();
    {
        let mut s = EditSession::open(&path).unwrap();
        s.delete("big");
        s.commit().unwrap();
        let len_after_delete = std::fs::metadata(&path).unwrap().len();
        s.create_dataset("reuse").with_f64_data(&reuse);
        s.commit().unwrap();
        let len_after_reuse = std::fs::metadata(&path).unwrap().len();
        assert!(
            len_after_reuse < len_after_delete + (reuse.len() * 8) as u64,
            "reuse commit did not reclaim freed delete space \
             (delete={len_after_delete}, reuse={len_after_reuse})"
        );
    }

    let file = File::open(&path).unwrap();
    assert!(file.dataset("big").is_err());
    assert_eq!(file.dataset("reuse").unwrap().read_f64().unwrap(), reuse);
    assert_eq!(
        file.dataset("keep").unwrap().read_i32().unwrap(),
        vec![7, 8, 9]
    );
    assert_userblock_unchanged(&path, &userblock);

    std::fs::remove_file(&path).ok();
}

#[test]
fn userblock_delete_one_of_several_then_read_attr() {
    // A delete rewrites the parent group's header (relinking survivors) and frees
    // the removed object. A sibling group's compact attribute must remain readable.
    let path = std::env::temp_dir().join("hdf5_pure_ub_fu_delete_attr.h5");
    let mut b = FileBuilder::new();
    b.with_userblock(UB as u64);
    b.create_dataset("doomed").with_f64_data(&[1.0, 2.0]);
    b.create_dataset("survivor").with_i32_data(&[5, 6]);
    let mut g = b.create_group("grp");
    g.set_attr("tag", AttrValue::AsciiString("kept".into()));
    b.add_group(g.finish());
    let mut bytes = b.finish().unwrap();
    let userblock = stamp_userblock(&mut bytes);
    std::fs::write(&path, &bytes).unwrap();

    {
        let mut s = EditSession::open(&path).unwrap();
        s.delete("doomed");
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert!(file.dataset("doomed").is_err());
    assert_eq!(
        file.dataset("survivor").unwrap().read_i32().unwrap(),
        vec![5, 6]
    );
    let attrs = file.group("grp").unwrap().attrs().unwrap();
    assert_eq!(attrs.get("tag"), Some(&AttrValue::String("kept".into())));
    assert_userblock_unchanged(&path, &userblock);

    std::fs::remove_file(&path).ok();
}

// ---- copy (in-file) ----

#[test]
fn userblock_copy_dataset_roundtrip() {
    // Copying a contiguous dataset writes a fresh data block and header; on a
    // userblock file the new data address and the parent link to the copy must both
    // be stored base-relative.
    let path = std::env::temp_dir().join("hdf5_pure_ub_fu_copy_ds.h5");
    let userblock = build_userblock_file(&path);

    {
        let mut s = EditSession::open(&path).unwrap();
        s.copy("alpha", "alpha_copy");
        s.copy("grp/inner", "grp/inner_copy");
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    // Original untouched, copy reads identical values.
    assert_eq!(
        file.dataset("alpha").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        file.dataset("alpha_copy").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        file.dataset("grp/inner_copy").unwrap().read_f64().unwrap(),
        vec![7.5, 8.5]
    );
    assert_userblock_unchanged(&path, &userblock);

    std::fs::remove_file(&path).ok();
}

#[test]
fn userblock_copy_group_subtree_roundtrip() {
    // Copying a whole group deep-copies its nested dataset too; every child link
    // and data address in the copy is written base-relative.
    let path = std::env::temp_dir().join("hdf5_pure_ub_fu_copy_grp.h5");
    let userblock = build_userblock_file(&path);

    {
        let mut s = EditSession::open(&path).unwrap();
        s.copy("grp", "grp_copy");
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    // Both the original subtree and the copy are present and identical.
    assert_eq!(
        file.dataset("grp/inner").unwrap().read_f64().unwrap(),
        vec![7.5, 8.5]
    );
    assert_eq!(
        file.dataset("grp_copy/inner").unwrap().read_f64().unwrap(),
        vec![7.5, 8.5]
    );
    let mut groups = file.root().groups().unwrap();
    groups.sort();
    assert_eq!(groups, vec!["grp".to_string(), "grp_copy".to_string()]);
    assert_userblock_unchanged(&path, &userblock);

    std::fs::remove_file(&path).ok();
}

#[test]
fn userblock_copy_chunked_dataset_roundtrip() {
    // Copying a chunked/filtered dataset enumerates the source chunks (on a
    // base-relative view of the file) and rebuilds the index at the new location.
    let path = std::env::temp_dir().join("hdf5_pure_ub_fu_copy_chunk.h5");
    let mut b = FileBuilder::new();
    b.with_userblock(UB as u64);
    let data: Vec<f64> = (0..600).map(|i| (i % 9) as f64 * 0.25).collect();
    b.create_dataset("c")
        .with_f64_data(&data)
        .with_shape(&[600])
        .with_chunks(&[40])
        .with_deflate(6);
    let mut bytes = b.finish().unwrap();
    let userblock = stamp_userblock(&mut bytes);
    std::fs::write(&path, &bytes).unwrap();

    {
        let mut s = EditSession::open(&path).unwrap();
        s.copy("c", "c_copy");
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("c").unwrap().read_f64().unwrap(), data);
    assert_eq!(file.dataset("c_copy").unwrap().read_f64().unwrap(), data);
    assert_userblock_unchanged(&path, &userblock);

    std::fs::remove_file(&path).ok();
}

// ---- copy (cross-file, into a userblock destination) ----

#[test]
fn userblock_cross_file_copy_into_userblock_dest() {
    // A base-0 source file is copied into a userblock destination. The destination
    // writes the copy base-relative even though the source was read base-0.
    let dst_path = std::env::temp_dir().join("hdf5_pure_ub_fu_xcopy_dst.h5");
    let src_path = std::env::temp_dir().join("hdf5_pure_ub_fu_xcopy_src.h5");
    let userblock = build_userblock_file(&dst_path);

    // A plain (no-userblock) source file.
    {
        let mut b = FileBuilder::new();
        b.create_dataset("payload").with_i32_data(&[100, 200, 300]);
        let mut g = b.create_group("sub");
        g.create_dataset("leaf").with_f64_data(&[1.25, 2.5]);
        b.add_group(g.finish());
        b.write(&src_path).unwrap();
    }

    {
        let source = File::open(&src_path).unwrap();
        let mut s = EditSession::open(&dst_path).unwrap();
        s.copy_from(&source, "payload", "imported").unwrap();
        s.copy_from(&source, "sub", "imported_grp").unwrap();
        s.commit().unwrap();
    }

    let file = File::open(&dst_path).unwrap();
    assert_eq!(
        file.dataset("imported").unwrap().read_i32().unwrap(),
        vec![100, 200, 300]
    );
    assert_eq!(
        file.dataset("imported_grp/leaf")
            .unwrap()
            .read_f64()
            .unwrap(),
        vec![1.25, 2.5]
    );
    // Destination originals untouched.
    assert_eq!(
        file.dataset("alpha").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_userblock_unchanged(&dst_path, &userblock);

    std::fs::remove_file(&dst_path).ok();
    std::fs::remove_file(&src_path).ok();
}
