//! Free-space reuse and truncation during in-place editing (issue #21).
//!
//! `File::open_rw` records the regions a commit vacates — deleted objects' blocks
//! and superseded group headers — and, within the same session, reuses them for
//! later writes instead of growing the file, truncating the file when a freed
//! run reaches end-of-file. These tests pin down both the size behavior and that
//! survivors stay byte-exact and the file stays valid.

use hdf5_pure::{
    AttrValue, EditBacking, File, FileAccessProperties, FileBuilder, FileSpaceStrategy,
    MemoryStrategy, SyncPolicy,
};

#[path = "common/temp_fixture.rs"]
mod temp_fixture;
use temp_fixture::temp_path;

/// The superblock's end-of-file must equal the actual file length after every
/// commit, including ones that truncate.
fn assert_eof_matches_file(path: &std::path::Path) {
    let file = File::open(path).unwrap();
    let eof = file.file_size();
    let actual = std::fs::metadata(path).unwrap().len();
    assert_eq!(
        eof, actual,
        "superblock EOF must match the physical file size"
    );
}

#[test]
fn delete_then_truncate_shrinks_within_session() {
    let path = temp_path("hdf5_pure_fs_shrink.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep")
        .with_f64_data(&[1.0, 2.0, 3.0, 4.0]);
    b.write(&path).unwrap();
    let size_start = std::fs::metadata(&path).unwrap().len();

    // A single session: add a large dataset, then delete it. The deleted blocks
    // and the superseded root header form a run reaching end-of-file, so the
    // file is truncated back down rather than left bloated.
    {
        let s = File::open_rw(&path).unwrap();
        s.root()
            .create_dataset("big", |b| {
                b.with_f64_data(&vec![7.0; 1024]);
            })
            .unwrap();
        s.commit().unwrap();
        let size_after_add = std::fs::metadata(&path).unwrap().len();
        assert!(size_after_add > size_start, "adding should grow the file");

        s.root().delete("big").unwrap();
        s.commit().unwrap();
        let size_after_delete = std::fs::metadata(&path).unwrap().len();
        assert!(
            size_after_delete < size_after_add,
            "deleting the just-added dataset should shrink the file (was {size_after_add}, now {size_after_delete})"
        );
    }

    assert_eof_matches_file(&path);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), vec!["keep".to_string()]);
    assert_eq!(
        file.dataset("keep").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert!(file.dataset("big").is_err());
}

#[test]
fn churn_within_session_stays_bounded() {
    let path = temp_path("hdf5_pure_fs_churn.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    b.write(&path).unwrap();

    let mut high_water = 0u64;
    {
        let s = File::open_rw(&path).unwrap();
        // Repeatedly add then delete a sizable dataset in the same session. With
        // reuse + truncation the file must not grow without bound across cycles.
        for i in 0..8 {
            s.root()
                .create_dataset("scratch", |b| {
                    b.with_f64_data(&vec![i as f64; 512]);
                })
                .unwrap();
            s.commit().unwrap();
            high_water = high_water.max(std::fs::metadata(&path).unwrap().len());
            s.root().delete("scratch").unwrap();
            s.commit().unwrap();
        }
    }

    let final_size = std::fs::metadata(&path).unwrap().len();
    // After the last delete the scratch space is reclaimed, so the file is far
    // smaller than the running peak — proof the freed space was reused, not
    // leaked on every cycle.
    assert!(
        final_size < high_water,
        "churn should reclaim space (peak {high_water}, final {final_size})"
    );

    assert_eof_matches_file(&path);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), vec!["keep".to_string()]);
    assert_eq!(
        file.dataset("keep").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
}

#[test]
fn reuse_keeps_survivors_byte_exact() {
    let path = temp_path("hdf5_pure_fs_reuse_exact.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("a").with_i32_data(&[10, 20, 30]);
    b.create_dataset("b").with_f64_data(&[1.5, 2.5]);
    b.write(&path).unwrap();

    {
        let s = File::open_rw(&path).unwrap();
        // Delete b, then add c in a later commit: c's bytes should land in the
        // region b vacated. a and the newly written c must both read back exact.
        s.root().delete("b").unwrap();
        s.commit().unwrap();
        s.root()
            .create_dataset("c", |b| {
                b.with_i32_data(&[7, 8, 9, 10]);
            })
            .unwrap();
        s.commit().unwrap();
    }

    assert_eof_matches_file(&path);
    let file = File::open(&path).unwrap();
    let mut names = file.root().datasets().unwrap();
    names.sort();
    assert_eq!(names, vec!["a".to_string(), "c".to_string()]);
    assert_eq!(
        file.dataset("a").unwrap().read_i32().unwrap(),
        vec![10, 20, 30]
    );
    assert_eq!(
        file.dataset("c").unwrap().read_i32().unwrap(),
        vec![7, 8, 9, 10]
    );
    assert!(file.dataset("b").is_err());
}

#[test]
fn delete_subtree_reclaims_all_members() {
    let path = temp_path("hdf5_pure_fs_subtree.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[1]);
    b.write(&path).unwrap();

    {
        let s = File::open_rw(&path).unwrap();
        s.root().create_group("grp").unwrap();
        s.root()
            .create_dataset("grp/x", |b| {
                b.with_f64_data(&vec![1.0; 256]);
            })
            .unwrap();
        s.root()
            .create_dataset("grp/y", |b| {
                b.with_f64_data(&vec![2.0; 256]);
            })
            .unwrap();
        s.commit().unwrap();
        let with_group = std::fs::metadata(&path).unwrap().len();

        s.root().delete("grp").unwrap();
        s.commit().unwrap();
        let after = std::fs::metadata(&path).unwrap().len();
        // The whole subtree (group header + both datasets' headers and data) is
        // reclaimed, shrinking the file well below its size with the group.
        assert!(
            after < with_group,
            "deleting a subtree should reclaim its members (was {with_group}, now {after})"
        );
    }

    assert_eof_matches_file(&path);
    let file = File::open(&path).unwrap();
    assert!(file.group("grp").is_err());
    assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), vec![1]);
}

#[test]
fn trailing_slack_past_recorded_eof_stays_readable() {
    // `commit` makes the superblock recording the smaller end-of-file durable
    // *before* it physically `set_len`s the file, so a crash in that window leaves
    // a durable, smaller superblock EOF plus the not-yet-removed trailing bytes.
    // This pins down the reader-side property that makes such a crash harmless:
    // the reader navigates by the superblock's end-of-file address and never reads
    // the slack past it. It reproduces that on-disk state by re-appending leftover
    // bytes to a cleanly committed file and confirms the file still reads exactly.
    // (It exercises the *outcome* of the ordering, not the ordering itself —
    // fault-injecting between the superblock sync and `set_len` would need a seam
    // File::open_rw does not yet expose, and remains future work.)
    let path = temp_path("hdf5_pure_fs_trailing_slack.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[11, 22, 33]);
    b.write(&path).unwrap();

    // Add then delete a large dataset so the second commit truncates the file
    // back down, leaving the recorded end-of-file equal to the physical size.
    {
        let s = File::open_rw(&path).unwrap();
        s.root()
            .create_dataset("scratch", |b| {
                b.with_f64_data(&vec![5.0; 2048]);
            })
            .unwrap();
        s.commit().unwrap();
        s.root().delete("scratch").unwrap();
        s.commit().unwrap();
    }

    let (logical_eof, physical) = {
        let f = File::open(&path).unwrap();
        (f.superblock().eof_address, f.file_size())
    };
    assert_eq!(
        logical_eof, physical,
        "a clean truncating commit leaves no slack past the recorded end-of-file"
    );

    // Simulate the crash: the smaller-EOF superblock is already durable, but the
    // process died before `set_len`, so the freed tail is still on disk. Re-append
    // leftover bytes to reproduce that physical state.
    const SLACK: u64 = 4096;
    {
        use std::io::Write;
        let mut handle = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        handle.write_all(&vec![0xAB; SLACK as usize]).unwrap();
        handle.flush().unwrap();
    }

    // The trailing slack is invisible to the reader: survivors read byte-exact and
    // the deleted object stays gone, even though the physical file now exceeds the
    // recorded end-of-file.
    let f = File::open(&path).unwrap();
    assert_eq!(
        f.superblock().eof_address,
        logical_eof,
        "the durable end-of-file address is unaffected by trailing slack"
    );
    assert_eq!(
        f.file_size(),
        physical + SLACK,
        "the physical file carries the leftover bytes the crash left behind"
    );
    assert_eq!(f.root().datasets().unwrap(), vec!["keep".to_string()]);
    assert_eq!(
        f.dataset("keep").unwrap().read_i32().unwrap(),
        vec![11, 22, 33]
    );
    assert!(f.dataset("scratch").is_err());
}

#[test]
fn deleting_filtered_chunked_dataset_reclaims_storage() {
    // A chunked, filtered (Fixed Array index) dataset's storage — chunk data
    // blocks plus the FAHD/FADB index — is now reclaimed on delete. Deleting it
    // shrinks the file below its size with the dataset present, the survivor
    // stays byte-exact, and the file stays valid.
    let path = temp_path("hdf5_pure_fs_chunked_filtered.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[42, 43, 44]);
    b.write(&path).unwrap();
    let size_keep_only = std::fs::metadata(&path).unwrap().len();

    {
        let s = File::open_rw(&path).unwrap();
        s.root()
            .create_dataset("comp", |b| {
                b.with_f64_data(&vec![3.0; 4096])
                    .with_chunks(&[512])
                    .with_deflate(6);
            })
            .unwrap();
        s.commit().unwrap();
        let size_with_comp = std::fs::metadata(&path).unwrap().len();
        assert!(
            size_with_comp > size_keep_only,
            "adding a chunked dataset should grow the file"
        );

        s.root().delete("comp").unwrap();
        s.commit().unwrap();
        let size_after_delete = std::fs::metadata(&path).unwrap().len();
        assert!(
            size_after_delete < size_with_comp,
            "deleting the chunked dataset should reclaim its storage \
             (was {size_with_comp}, now {size_after_delete})"
        );
    }

    assert_eof_matches_file(&path);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), vec!["keep".to_string()]);
    assert_eq!(
        file.dataset("keep").unwrap().read_i32().unwrap(),
        vec![42, 43, 44]
    );
    assert!(file.dataset("comp").is_err());
}

#[test]
fn deleting_unfiltered_chunked_dataset_truncates_fully() {
    // An unfiltered chunked dataset is laid out as one contiguous blob (chunk
    // data followed by the Fixed Array index, nothing between). Adding it at
    // end-of-file then deleting it in the same session reclaims the whole blob as
    // a trailing run, truncating the file back to essentially its prior size.
    let path = temp_path("hdf5_pure_fs_chunked_unfiltered.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    b.write(&path).unwrap();
    let size_start = std::fs::metadata(&path).unwrap().len();

    {
        let s = File::open_rw(&path).unwrap();
        // 8 chunks of 512 f64 = 4096 bytes each.
        s.root()
            .create_dataset("big", |b| {
                b.with_f64_data(&vec![7.0; 4096]).with_chunks(&[512]);
            })
            .unwrap();
        s.commit().unwrap();
        let size_with_big = std::fs::metadata(&path).unwrap().len();
        assert!(size_with_big > size_start + 4096 * 8);

        s.root().delete("big").unwrap();
        s.commit().unwrap();
        let size_after_delete = std::fs::metadata(&path).unwrap().len();
        // The reclaimed blob formed a trailing run, so the file is truncated back
        // close to its starting size (a small, bounded amount of reused/rewritten
        // header churn aside).
        assert!(
            size_after_delete < size_start + 4096,
            "deleting an unfiltered chunked dataset should truncate the file back \
             near its start (start {size_start}, after delete {size_after_delete})"
        );
    }

    assert_eof_matches_file(&path);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), vec!["keep".to_string()]);
    assert_eq!(
        file.dataset("keep").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    assert!(file.dataset("big").is_err());
}

#[test]
fn deleting_paged_fixed_array_dataset_reclaims_storage() {
    // More than 1024 chunks puts the Fixed Array data block into its *paged*
    // layout (a page-init bitmap and per-page checksums). 1100 chunks of 16 f64
    // exercise that index-sizing path end to end: the whole index plus chunk
    // data is reclaimed on delete.
    let path = temp_path("hdf5_pure_fs_paged_fa.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[7]);
    b.write(&path).unwrap();
    let size_start = std::fs::metadata(&path).unwrap().len();

    {
        let s = File::open_rw(&path).unwrap();
        s.root()
            .create_dataset("paged", |b| {
                b.with_f64_data(&vec![1.0; 1100 * 16]).with_chunks(&[16]);
            })
            .unwrap(); // 1100 chunks > 1024 page size
        s.commit().unwrap();
        let size_with_paged = std::fs::metadata(&path).unwrap().len();

        s.root().delete("paged").unwrap();
        s.commit().unwrap();
        let size_after_delete = std::fs::metadata(&path).unwrap().len();
        assert!(
            size_after_delete < size_with_paged,
            "deleting a paged fixed-array dataset should reclaim its storage \
             (was {size_with_paged}, now {size_after_delete})"
        );
        // The chunks and the index form one trailing run, so the file truncates
        // back near its start.
        assert!(size_after_delete < size_start + 4096);
    }

    assert_eof_matches_file(&path);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), vec!["keep".to_string()]);
    assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), vec![7]);
    assert!(file.dataset("paged").is_err());
}

#[test]
fn deleting_extensible_dataset_reclaims_storage() {
    // A dataset with an unlimited maximum dimension uses an Extensible Array chunk
    // index (EAHD/EAIB and, past the inline slots, data blocks). Deleting it
    // reclaims the whole index plus its chunk data.
    let path = temp_path("hdf5_pure_fs_extensible.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[9]);
    b.write(&path).unwrap();
    let size_start = std::fs::metadata(&path).unwrap().len();

    {
        let s = File::open_rw(&path).unwrap();
        // 64 chunks of 64 f64 — enough to spill past the index block's inline
        // slots into separate data blocks, exercising the data-block walk.
        s.root()
            .create_dataset("ext", |b| {
                b.with_f64_data(&vec![2.5; 4096])
                    .with_chunks(&[64])
                    .with_maxshape(&[u64::MAX]);
            })
            .unwrap();
        s.commit().unwrap();
        let size_with_ext = std::fs::metadata(&path).unwrap().len();
        assert!(size_with_ext > size_start);

        s.root().delete("ext").unwrap();
        s.commit().unwrap();
        let size_after_delete = std::fs::metadata(&path).unwrap().len();
        assert!(
            size_after_delete < size_with_ext,
            "deleting an extensible-array dataset should reclaim its storage \
             (was {size_with_ext}, now {size_after_delete})"
        );
    }

    assert_eof_matches_file(&path);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), vec!["keep".to_string()]);
    assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), vec![9]);
    assert!(file.dataset("ext").is_err());
}

#[test]
fn deleting_single_chunk_dataset_reclaims_storage() {
    // A dataset whose single chunk covers the whole shape uses the single-chunk
    // index (the chunk address lives in the layout message, no separate index
    // structure). Deleting it reclaims that one chunk block.
    let path = temp_path("hdf5_pure_fs_single_chunk.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[5, 6]);
    b.write(&path).unwrap();
    let size_start = std::fs::metadata(&path).unwrap().len();

    {
        let s = File::open_rw(&path).unwrap();
        // 1024 f64 = 8192 bytes in a single chunk, whose blob reclaims as one
        // trailing run.
        s.root()
            .create_dataset("one", |b| {
                b.with_f64_data(&vec![1.25; 1024]).with_chunks(&[1024]);
            })
            .unwrap(); // one chunk covers the whole dataset
        s.commit().unwrap();
        let size_with_one = std::fs::metadata(&path).unwrap().len();
        assert!(size_with_one > size_start + 8192);

        s.root().delete("one").unwrap();
        s.commit().unwrap();
        let size_after_delete = std::fs::metadata(&path).unwrap().len();
        assert!(
            size_after_delete < size_with_one,
            "deleting a single-chunk dataset should reclaim its chunk \
             (was {size_with_one}, now {size_after_delete})"
        );
        // The 8192-byte chunk dominates; reclaim truncates the file back near
        // its starting size.
        assert!(size_after_delete < size_start + 4096);
    }

    assert_eof_matches_file(&path);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), vec!["keep".to_string()]);
    assert_eq!(
        file.dataset("keep").unwrap().read_i32().unwrap(),
        vec![5, 6]
    );
    assert!(file.dataset("one").is_err());
}

#[test]
fn chunked_churn_within_session_stays_bounded() {
    // Repeatedly add then delete a sizable chunked dataset in one session. With
    // chunk + index reclaim and reuse the file must not grow without bound.
    let path = temp_path("hdf5_pure_fs_chunked_churn.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    b.write(&path).unwrap();

    let mut high_water = 0u64;
    {
        let s = File::open_rw(&path).unwrap();
        for i in 0..8 {
            s.root()
                .create_dataset("scratch", |b| {
                    b.with_f64_data(&vec![i as f64; 2048])
                        .with_chunks(&[256])
                        .with_deflate(4);
                })
                .unwrap();
            s.commit().unwrap();
            high_water = high_water.max(std::fs::metadata(&path).unwrap().len());
            s.root().delete("scratch").unwrap();
            s.commit().unwrap();
        }
    }

    let final_size = std::fs::metadata(&path).unwrap().len();
    assert!(
        final_size < high_water,
        "chunked churn should reclaim space (peak {high_water}, final {final_size})"
    );

    assert_eof_matches_file(&path);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), vec!["keep".to_string()]);
    assert_eq!(
        file.dataset("keep").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
}

#[test]
fn deleting_subtree_with_chunked_members_reclaims() {
    // Deleting a group reclaims its chunked-dataset members' storage too (the
    // free walk descends the subtree).
    let path = temp_path("hdf5_pure_fs_chunked_subtree.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[1]);
    b.write(&path).unwrap();

    {
        let s = File::open_rw(&path).unwrap();
        s.root().create_group("grp").unwrap();
        s.root()
            .create_dataset("grp/a", |b| {
                b.with_f64_data(&vec![1.0; 4096]).with_chunks(&[512]);
            })
            .unwrap();
        s.root()
            .create_dataset("grp/b", |b| {
                b.with_f64_data(&vec![2.0; 2048])
                    .with_chunks(&[256])
                    .with_deflate(6);
            })
            .unwrap();
        s.commit().unwrap();
        let with_group = std::fs::metadata(&path).unwrap().len();

        s.root().delete("grp").unwrap();
        s.commit().unwrap();
        let after = std::fs::metadata(&path).unwrap().len();
        assert!(
            after < with_group,
            "deleting a subtree with chunked members should reclaim them \
             (was {with_group}, now {after})"
        );
    }

    assert_eof_matches_file(&path);
    let file = File::open(&path).unwrap();
    assert!(file.group("grp").is_err());
    assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), vec![1]);
}

#[test]
fn persisted_chunked_reclaim_is_disjoint_and_reusable() {
    // With persistence on, deleting a chunked dataset records its storage as
    // free sections that stay disjoint and are reused across reopen.
    let path = temp_path("hdf5_pure_fs_chunked_persist.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[1; 50]);
    b.create_dataset("comp")
        .with_f64_data(&vec![3.0; 4096])
        .with_chunks(&[512])
        .with_deflate(6);
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    b.write(&path).unwrap();

    {
        let s = File::open_rw(&path).unwrap();
        s.root().delete("comp").unwrap();
        s.commit().unwrap();
    }
    assert_eof_matches_file(&path);

    let f = File::open(&path).unwrap();
    assert!(f.file_space_info().unwrap().persist);
    let mut free = f.persisted_free_space();
    free.sort();
    for w in free.windows(2) {
        assert!(
            w[0].0 + w[0].1 <= w[1].0,
            "persisted free regions must be disjoint after a chunked delete: {free:?}"
        );
    }
    assert_eq!(f.dataset("keep").unwrap().read_i32().unwrap(), vec![1; 50]);
    assert!(f.dataset("comp").is_err());
}

#[test]
fn persisted_free_space_survives_reopen_and_is_reused() {
    // A file created with persist = true keeps its freed space recorded on disk
    // (the FSHD/FSSE managers), so a freed region survives close/reopen and a
    // later session reuses it instead of growing the file. This is the cross-
    // session counterpart to the within-session reuse above.
    let path = temp_path("hdf5_pure_fs_persist_roundtrip.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("a").with_i32_data(&[1; 100]);
    b.create_dataset("big").with_i32_data(&[7; 400]); // 1600 bytes of raw data
    b.create_dataset("c").with_i32_data(&[3; 100]);
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    b.write(&path).unwrap();

    // Session 1: delete "big"; its storage is persisted as free space.
    {
        let s = File::open_rw(&path).unwrap();
        s.root().delete("big").unwrap();
        s.commit().unwrap();
    }
    assert_eof_matches_file(&path);

    // A fresh reader recovers the persisted free space across the reopen, and the
    // surviving datasets stay byte-exact alongside the on-disk managers.
    let f = File::open(&path).unwrap();
    assert_eq!(f.file_space_strategy(), Some(FileSpaceStrategy::FsmAggr));
    assert!(f.file_space_info().unwrap().persist);
    let free = f.persisted_free_space();
    let total: u64 = free.iter().map(|(_, l)| l).sum();
    assert!(
        total >= 1600 && free.iter().any(|&(_, l)| l >= 1600),
        "the deleted dataset's storage is persisted as a free section: {free:?}"
    );
    assert_eq!(f.dataset("a").unwrap().read_i32().unwrap(), vec![1; 100]);
    assert_eq!(f.dataset("c").unwrap().read_i32().unwrap(), vec![3; 100]);
    assert!(f.dataset("big").is_err());
    drop(f);

    // Session 2: opening seeds the free list from the persisted managers, so a
    // new dataset reuses the freed hole rather than appending its data at EOF.
    let eof_before = std::fs::metadata(&path).unwrap().len();
    {
        let s = File::open_rw(&path).unwrap();
        s.root()
            .create_dataset("d", |b| {
                b.with_i32_data(&[9; 300]);
            })
            .unwrap(); // 1200 bytes, fits the hole
        s.commit().unwrap();
    }
    assert_eof_matches_file(&path);
    let eof_after = std::fs::metadata(&path).unwrap().len();
    assert!(
        eof_after < eof_before + 1200,
        "the new dataset should reuse the persisted free hole, not grow the file \
         by its full size (before={eof_before} after={eof_after})"
    );

    let f = File::open(&path).unwrap();
    assert_eq!(f.dataset("d").unwrap().read_i32().unwrap(), vec![9; 300]);
    assert_eq!(f.dataset("a").unwrap().read_i32().unwrap(), vec![1; 100]);
    assert_eq!(f.dataset("c").unwrap().read_i32().unwrap(), vec![3; 100]);
    // Persistence remains armed: the file still records its free space on disk.
    assert!(f.file_space_info().unwrap().persist);
}

#[test]
fn persisted_managers_stay_consistent_across_many_commits() {
    // Several persisting commits in a row: each supersedes the previous on-disk
    // managers and extension, recording them as free, so the file stays valid and
    // its tracked free space never double-counts or loses a region. Both this
    // crate and a fresh reader must agree on the result after every step.
    let path = temp_path("hdf5_pure_fs_persist_multi.h5");
    let mut b = FileBuilder::new();
    for i in 0..6 {
        b.create_dataset(&format!("d{i}"))
            .with_i32_data(&vec![i; 200]);
    }
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    b.write(&path).unwrap();

    // Churn across multiple sessions: delete some, add some, each its own commit.
    for round in 0..4 {
        {
            let s = File::open_rw(&path).unwrap();
            s.root().delete(&format!("d{round}")).unwrap();
            s.root()
                .create_dataset(&format!("n{round}"), |b| {
                    b.with_i32_data(&vec![100 + round; 150]);
                })
                .unwrap();
            s.commit().unwrap();
        } // release the editor's lock before reading the file back
        // Free regions never overlap and the file remains valid after each round.
        assert_eof_matches_file(&path);
        let f = File::open(&path).unwrap();
        assert!(f.file_space_info().unwrap().persist);
        let mut free = f.persisted_free_space();
        free.sort();
        for w in free.windows(2) {
            assert!(
                w[0].0 + w[0].1 <= w[1].0,
                "persisted free regions must be disjoint: {free:?}"
            );
        }
    }

    // Every surviving dataset reads back byte-exact.
    let f = File::open(&path).unwrap();
    for i in 4..6 {
        assert_eq!(
            f.dataset(&format!("d{i}")).unwrap().read_i32().unwrap(),
            vec![i; 200]
        );
    }
    for round in 0..4 {
        assert_eq!(
            f.dataset(&format!("n{round}")).unwrap().read_i32().unwrap(),
            vec![100 + round; 150]
        );
    }
    assert!(f.dataset("d0").is_err());
}

/// The bytes a chunked dataset of `elems` `f64` elements occupies, near enough:
/// the chunk data itself, which dominates its index and header.
fn f64_data_bytes(elems: usize) -> u64 {
    (elems * 8) as u64
}

#[test]
fn chunked_dataset_reuses_the_hole_a_chunked_delete_left() {
    // The headline of issue #261. A chunked dataset's data region — chunk bytes
    // plus the index that addresses them — used to be *appended* unconditionally,
    // because its embedded addresses were computed from the end-of-file it was
    // about to land at. Sizing it before placing it lets it go into a freed region
    // instead, so delete-then-write of the same shape costs the file nothing.
    //
    // The hole is interior on purpose: `tail` sits above the deleted dataset, so
    // truncation cannot be what keeps the file small.
    let path = temp_path("hdf5_pure_fs_chunked_reuse.h5");
    const ELEMS: usize = 32768; // 256 KiB of raw data
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    b.create_dataset("big")
        .with_f64_data(&vec![1.0; ELEMS])
        .with_chunks(&[4096]);
    b.create_dataset("tail").with_i32_data(&[9; 16]);
    b.write(&path).unwrap();
    let start = std::fs::metadata(&path).unwrap().len();

    {
        let s = File::open_rw(&path).unwrap();
        s.root().delete("big").unwrap();
        s.commit().unwrap();
        s.root()
            .create_dataset("big2", |b| {
                b.with_f64_data(&vec![2.0; ELEMS]).with_chunks(&[4096]);
            })
            .unwrap();
        s.commit().unwrap();
    }

    let end = std::fs::metadata(&path).unwrap().len();
    assert!(
        end < start + f64_data_bytes(ELEMS) / 10,
        "the replacement dataset should land in the hole the deleted one left, \
         not past it (start={start}, end={end})"
    );

    assert_eof_matches_file(&path);
    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("big2").unwrap().read_f64().unwrap(),
        vec![2.0; ELEMS]
    );
    assert_eq!(
        f.dataset("keep").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    assert_eq!(f.dataset("tail").unwrap().read_i32().unwrap(), vec![9; 16]);
    assert!(f.dataset("big").is_err());
}

#[test]
fn filtered_chunked_dataset_reuses_freed_space() {
    // Same as above for a *filtered* dataset, whose compressed size is not known
    // until the pipeline has run: the placement is chosen from the compressed
    // set's size, so the filter pass still happens exactly once.
    let path = temp_path("hdf5_pure_fs_chunked_reuse_filtered.h5");
    const ELEMS: usize = 32768;
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    b.create_dataset("big")
        .with_f64_data(&(0..ELEMS).map(|i| i as f64).collect::<Vec<f64>>())
        .with_chunks(&[4096])
        .with_deflate(4);
    b.create_dataset("tail").with_i32_data(&[9; 16]);
    b.write(&path).unwrap();
    let start = std::fs::metadata(&path).unwrap().len();
    let hole = start; // an upper bound on what the deleted dataset can free

    {
        let s = File::open_rw(&path).unwrap();
        s.root().delete("big").unwrap();
        s.commit().unwrap();
        let after_delete = std::fs::metadata(&path).unwrap().len();
        s.root()
            .create_dataset("big2", |b| {
                b.with_f64_data(&(0..ELEMS).map(|i| i as f64).collect::<Vec<f64>>())
                    .with_chunks(&[4096])
                    .with_deflate(4);
            })
            .unwrap();
        s.commit().unwrap();
        let end = std::fs::metadata(&path).unwrap().len();
        assert!(
            end < after_delete + hole / 10,
            "an identically filtered replacement should reuse the freed region \
             (after_delete={after_delete}, end={end})"
        );
    }

    assert_eof_matches_file(&path);
    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("big2").unwrap().read_f64().unwrap(),
        (0..ELEMS).map(|i| i as f64).collect::<Vec<f64>>()
    );
    assert_eq!(f.dataset("tail").unwrap().read_i32().unwrap(), vec![9; 16]);
}

#[test]
fn reusing_a_chunked_hole_keeps_its_neighbors_byte_exact() {
    // Reuse writes over a dead interior region, so the test that matters is what
    // happens to the *live* bytes on either side of it. Both neighbors — one
    // below the hole, one above — must read back exactly, and every chunk of the
    // dataset written into the hole must too.
    let path = temp_path("hdf5_pure_fs_chunked_reuse_neighbors.h5");
    const ELEMS: usize = 16384;
    let below: Vec<f64> = (0..2048).map(|i| i as f64 * 0.5).collect();
    let above: Vec<i32> = (0..2048).map(|i| i * 3).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("below").with_f64_data(&below);
    b.create_dataset("victim")
        .with_f64_data(&vec![7.0; ELEMS])
        .with_chunks(&[2048]);
    b.create_dataset("above").with_i32_data(&above);
    b.write(&path).unwrap();

    let replacement: Vec<f64> = (0..ELEMS).map(|i| (i % 97) as f64).collect();
    {
        let s = File::open_rw(&path).unwrap();
        s.root().delete("victim").unwrap();
        s.commit().unwrap();
        s.root()
            .create_dataset("fresh", |b| {
                b.with_f64_data(&replacement).with_chunks(&[2048]);
            })
            .unwrap();
        s.commit().unwrap();
    }

    assert_eof_matches_file(&path);
    let f = File::open(&path).unwrap();
    assert_eq!(f.dataset("below").unwrap().read_f64().unwrap(), below);
    assert_eq!(f.dataset("above").unwrap().read_i32().unwrap(), above);
    assert_eq!(f.dataset("fresh").unwrap().read_f64().unwrap(), replacement);
}

#[test]
fn paged_commit_reuses_freed_space_within_its_page_type() {
    // A paged file segregates metadata and raw data by page, so it tracks free
    // space per page type and a commit may only draw from the list matching what
    // it is placing. It used to draw from none of them and always append.
    //
    // The file stays valid and every page stays homogeneous — the crosscheck
    // suite reads these files with the reference C library, which is where a
    // mixed page would show up.
    let path = temp_path("hdf5_pure_fs_paged_reuse.h5");
    const ELEMS: usize = 32768;
    let mut b = FileBuilder::new();
    b.with_file_space_strategy(FileSpaceStrategy::Page, true, 1);
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    b.create_dataset("big")
        .with_f64_data(&vec![1.0; ELEMS])
        .with_chunks(&[4096]);
    b.create_dataset("tail").with_i32_data(&[9; 16]);
    b.write(&path).unwrap();
    let start = std::fs::metadata(&path).unwrap().len();

    {
        let s = File::open_rw(&path).unwrap();
        s.root().delete("big").unwrap();
        s.commit().unwrap();
        s.root()
            .create_dataset("big2", |b| {
                b.with_f64_data(&vec![2.0; ELEMS]).with_chunks(&[4096]);
            })
            .unwrap();
        s.commit().unwrap();
    }

    let end = std::fs::metadata(&path).unwrap().len();
    assert!(
        end < start + f64_data_bytes(ELEMS) / 4,
        "a paged file should reuse its freed raw pages rather than append past \
         them (start={start}, end={end})"
    );

    assert_eof_matches_file(&path);
    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("big2").unwrap().read_f64().unwrap(),
        vec![2.0; ELEMS]
    );
    assert_eq!(
        f.dataset("keep").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    assert_eq!(f.dataset("tail").unwrap().read_i32().unwrap(), vec![9; 16]);
}

#[test]
fn persisted_chunked_free_space_is_reused_after_a_reopen() {
    // The cross-session case: one session deletes a chunked dataset, a *later*
    // one writes a fresh one. The second session only knows about the hole from
    // the on-disk managers it seeds its free list from, so this pins the seeding
    // and the chunked placement together.
    let path = temp_path("hdf5_pure_fs_chunked_persist_reuse.h5");
    const ELEMS: usize = 32768;
    let mut b = FileBuilder::new();
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    b.create_dataset("big")
        .with_f64_data(&vec![1.0; ELEMS])
        .with_chunks(&[4096]);
    b.create_dataset("tail").with_i32_data(&[9; 16]);
    b.write(&path).unwrap();
    let start = std::fs::metadata(&path).unwrap().len();

    {
        let s = File::open_rw(&path).unwrap();
        s.root().delete("big").unwrap();
        s.commit().unwrap();
    }
    {
        let s = File::open_rw(&path).unwrap();
        s.root()
            .create_dataset("big2", |b| {
                b.with_f64_data(&vec![2.0; ELEMS]).with_chunks(&[4096]);
            })
            .unwrap();
        s.commit().unwrap();
    }

    let end = std::fs::metadata(&path).unwrap().len();
    assert!(
        end < start + f64_data_bytes(ELEMS) / 10,
        "the second session should reuse the hole the first one persisted \
         (start={start}, end={end})"
    );

    assert_eof_matches_file(&path);
    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("big2").unwrap().read_f64().unwrap(),
        vec![2.0; ELEMS]
    );
    assert_eq!(
        f.dataset("keep").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
}

#[test]
fn both_read_write_backings_reuse_a_freed_hole_alike() {
    // One engine drives two backings (issue #198): the bounded one writes through
    // a handle, the mirrored one through a whole-file buffer. `File::open_rw`
    // prefers bounded, so every test above exercises that half; this one pins the
    // other, since a hole is reused by writing into the middle of the image and
    // that is precisely where the two backings differ.
    const ELEMS: usize = 16384;
    for (name, strategy) in [
        ("hdf5_pure_fs_backing_bounded.h5", MemoryStrategy::Bounded),
        ("hdf5_pure_fs_backing_mirrored.h5", MemoryStrategy::Mirrored),
    ] {
        let path = temp_path(name);
        let mut b = FileBuilder::new();
        b.create_dataset("big")
            .with_f64_data(&vec![1.0; ELEMS])
            .with_chunks(&[2048]);
        b.create_dataset("tail").with_i32_data(&[9; 16]);
        b.write(&path).unwrap();
        let start = std::fs::metadata(&path).unwrap().len();

        {
            let s = File::open_rw_with_options(
                &path,
                FileAccessProperties::new().with_memory_strategy(strategy),
            )
            .unwrap();
            assert_eq!(
                s.edit_backing(),
                Some(match strategy {
                    MemoryStrategy::Bounded => EditBacking::Bounded,
                    _ => EditBacking::Mirrored,
                }),
                "{name}: the requested backing is the one under test"
            );
            s.root().delete("big").unwrap();
            s.commit().unwrap();
            s.root()
                .create_dataset("big2", |b| {
                    b.with_f64_data(&vec![2.0; ELEMS]).with_chunks(&[2048]);
                })
                .unwrap();
            s.commit().unwrap();
        }

        let end = std::fs::metadata(&path).unwrap().len();
        assert!(
            end < start + f64_data_bytes(ELEMS) / 10,
            "{name}: the replacement should reuse the freed hole \
             (start={start}, end={end})"
        );
        assert_eof_matches_file(&path);
        let f = File::open(&path).unwrap();
        assert_eq!(
            f.dataset("big2").unwrap().read_f64().unwrap(),
            vec![2.0; ELEMS]
        );
        assert_eq!(f.dataset("tail").unwrap().read_i32().unwrap(), vec![9; 16]);
    }
}

#[test]
fn churn_of_groups_attributes_and_datasets_stays_bounded() {
    // The whole free-space story on one file: groups carrying attributes, plain
    // and filtered chunked datasets, deleted by subtree and rewritten identically,
    // round after round. Every round frees exactly what the next one needs, so a
    // file that reuses its free space returns to the same size each time while one
    // that appends grows by a full round's worth per cycle.
    //
    // A "ceiling" dataset written above the first round is what makes this a test
    // of *reuse*. Without it every freed round would reach end-of-file and be
    // truncated away, which keeps the file just as small while reusing nothing.
    let path = temp_path("hdf5_pure_fs_full_churn.h5");
    const ROUNDS: usize = 5;
    const GROUPS: usize = 4;
    const ELEMS: usize = 8192; // 64 KiB per dataset

    let write_round = |s: &File, round: usize| {
        for g in 0..GROUPS {
            let name = format!("g{g}");
            s.root()
                .create_group_with(&name, |grp| {
                    for a in 0..8 {
                        grp.set_attr(
                            &format!("attr{a}"),
                            AttrValue::F64Array(vec![round as f64; 32]),
                        );
                    }
                })
                .unwrap();
            s.root()
                .create_dataset(&format!("{name}/plain"), |b| {
                    b.with_f64_data(&vec![round as f64; ELEMS])
                        .with_chunks(&[1024]);
                })
                .unwrap();
            s.root()
                .create_dataset(&format!("{name}/filtered"), |b| {
                    b.with_f64_data(&(0..ELEMS).map(|i| (i + round) as f64).collect::<Vec<f64>>())
                        .with_chunks(&[1024])
                        .with_deflate(4);
                })
                .unwrap();
        }
    };

    let mut b = FileBuilder::new();
    b.create_dataset("anchor").with_i32_data(&[7; 64]);
    b.write(&path).unwrap();

    let baseline;
    let mut peak;
    {
        let s = File::open_rw(&path).unwrap();
        write_round(&s, 0);
        s.commit().unwrap();
        // Everything churned from here on sits *below* this dataset, so a freed
        // round is an interior hole rather than a trailing run.
        s.root()
            .create_dataset("ceiling", |b| {
                b.with_f64_data(&vec![1.25; ELEMS]);
            })
            .unwrap();
        s.commit().unwrap();
        baseline = std::fs::metadata(&path).unwrap().len();
        peak = baseline;

        for round in 1..=ROUNDS {
            for g in 0..GROUPS {
                s.root().delete(&format!("g{g}")).unwrap();
            }
            s.commit().unwrap();
            write_round(&s, round);
            s.commit().unwrap();
            peak = peak.max(std::fs::metadata(&path).unwrap().len());
        }
    }

    let end = std::fs::metadata(&path).unwrap().len();
    assert!(
        end < baseline + baseline / 4,
        "churning {ROUNDS} identical rounds should reuse the space each round \
         frees, not accumulate it (baseline={baseline}, peak={peak}, final={end})"
    );

    assert_eof_matches_file(&path);
    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("anchor").unwrap().read_i32().unwrap(),
        vec![7; 64]
    );
    for g in 0..GROUPS {
        assert_eq!(
            f.dataset(&format!("g{g}/plain"))
                .unwrap()
                .read_f64()
                .unwrap(),
            vec![ROUNDS as f64; ELEMS]
        );
        assert_eq!(
            f.dataset(&format!("g{g}/filtered"))
                .unwrap()
                .read_f64()
                .unwrap(),
            (0..ELEMS)
                .map(|i| (i + ROUNDS) as f64)
                .collect::<Vec<f64>>()
        );
        let attrs = f.group(&format!("g{g}")).unwrap().attrs().unwrap();
        for a in 0..8 {
            assert_eq!(
                attrs.get(&format!("attr{a}")),
                Some(&AttrValue::F64Array(vec![ROUNDS as f64; 32])),
            );
        }
    }
}

#[test]
fn corrupt_persisted_section_is_skipped_not_fatal() {
    // A malformed free-space manager must not be trusted blindly: if a persisted
    // section claims a region past end-of-file, seeding it and later handing it
    // out would write out of bounds. The editor skips such a section instead, so
    // the open + commit still succeeds and the live data stays intact.
    let path = temp_path("hdf5_pure_fs_persist_corrupt.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[5; 50]);
    b.create_dataset("victim").with_i32_data(&[6; 300]); // 1200-byte data block
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    b.write(&path).unwrap();
    {
        let s = File::open_rw(&path).unwrap();
        s.root().delete("victim").unwrap(); // frees the largest tracked section
        s.commit().unwrap();
    }

    // Find the largest persisted free section, then corrupt its offset to an
    // out-of-bounds address. FSSE groups sections by size, each group being a
    // `count`, a `size`, then `count` `(offset, class)` pairs; locate the largest
    // section's `size` field and rewrite the offset that follows it. (Counts and
    // sizes are 1 and 8 bytes here.)
    let big_len = File::open(&path)
        .unwrap()
        .persisted_free_space()
        .into_iter()
        .map(|(_, len)| len)
        .max()
        .expect("the deletion left a persisted free section");
    let mut bytes = std::fs::read(&path).unwrap();
    let file_len = bytes.len() as u64;
    let fsse = bytes
        .windows(4)
        .position(|w| w == b"FSSE")
        .expect("a persisted file with a freed section has an FSSE block");
    let size_le = big_len.to_le_bytes();
    let size_pos = fsse
        + 13
        + bytes[fsse + 13..]
            .windows(8)
            .position(|w| w == size_le)
            .expect("the largest section's size field is in the FSSE block");
    let off = size_pos + 8; // the offset field immediately follows the size
    bytes[off..off + 8].copy_from_slice(&(file_len + 4096).to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();

    // Add a dataset large enough that only the (corrupt) largest region could
    // satisfy it: without the bounds guard, the allocator would pick that region
    // and `write_at` an out-of-bounds address, panicking. The guard skips it, so
    // the data is appended instead and the commit succeeds.
    {
        let s = File::open_rw(&path).unwrap();
        s.root()
            .create_dataset("added", |b| {
                b.with_i32_data(&[9; 250]);
            })
            .unwrap(); // 1000 bytes
        s.commit().unwrap();
    }
    assert_eof_matches_file(&path);
    let f = File::open(&path).unwrap();
    assert_eq!(f.dataset("keep").unwrap().read_i32().unwrap(), vec![5; 50]);
    assert_eq!(
        f.dataset("added").unwrap().read_i32().unwrap(),
        vec![9; 250]
    );
}

/// A paged file under delete-and-recreate churn stops growing rather than
/// spending a page per commit (issue #286).
///
/// Every commit on a persisting paged file rewrites the superblock extension and
/// the free-space manager blocks. Those used to be appended at a fresh
/// page-aligned end-of-file and padded out to a page boundary, with the padding
/// recorded nowhere: a workload that stayed within a fixed budget by deleting its
/// oldest objects still grew without bound, at roughly one page per commit, and
/// `reusable_free_bytes` accounted for a fraction of what the file had spent.
///
/// The assertion is on the *shape* of the growth rather than a byte figure. The
/// file is allowed to reach whatever size the live data and its own layout need
/// over the first few rounds; what it may not do is keep climbing once the
/// workload is in its steady state, which is what an unreclaimed page per commit
/// looks like from the outside.
#[test]
fn paged_churn_reaches_a_steady_size() {
    const PAGE: u64 = 16384;
    const ROUNDS: usize = 12;
    const LIVE: usize = 2;

    let path = temp_path("hdf5_pure_fs_paged_churn.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("seed").with_i32_data(&[0i32; 4]);
    b.with_file_space_strategy(FileSpaceStrategy::Page, true, 1)
        .with_file_space_page_size(PAGE);
    b.write(&path).unwrap();

    let rows: Vec<i32> = (0..500).map(|i| i % 97).collect();
    let mut sizes = Vec::new();
    for round in 0..ROUNDS {
        // A fresh session per round, since the defect's worst half only shows
        // across a close: free space is handed back to disk in a manager that
        // records no page type, and until this fix a reopened session could never
        // spend it on metadata again.
        {
            let f = File::open_rw(&path).unwrap();
            f.root().create_group(&format!("g{round}")).unwrap();
            f.commit().unwrap();
            let g = f.group(&format!("g{round}")).unwrap();
            for name in ["a", "b"] {
                g.create_dataset(name, |b| {
                    b.with_i32_data(&rows)
                        .with_shape(&[rows.len() as u64])
                        .with_maxshape(&[u64::MAX])
                        .with_chunks(&[128]);
                })
                .unwrap();
            }
            f.commit().unwrap();
        }
        if round >= LIVE {
            let f = File::open_rw(&path).unwrap();
            f.root().delete(&format!("g{}", round - LIVE)).unwrap();
            f.commit().unwrap();
            // Every byte the file has spent and released is offered back, rather
            // than a fraction of it: the padding around the rewritten managers
            // used to be recorded nowhere at all. This also pins that the tail
            // reused rather than appended this round — an appended tail leaves the
            // remainder of the page it opened in the session's list only, where the
            // managers on disk cannot yet name it.
            let acct = f.space_accounting().unwrap();
            drop(f);
            let persisted: u64 = File::open(&path)
                .unwrap()
                .persisted_free_space()
                .iter()
                .map(|(_, len)| len)
                .sum();
            assert_eq!(
                acct.reusable_free_bytes, persisted,
                "round {round}: free space the session can spend must be the free \
                 space the file records"
            );
            sizes.push(std::fs::metadata(&path).unwrap().len());
        }
    }

    // The steady state: the last third of the run adds nothing. Deleting one
    // round's group makes room for the next round's, and the commit tail lands in
    // the space the previous tail vacated.
    let settled = sizes[sizes.len() * 2 / 3];
    let last = *sizes.last().unwrap();
    assert_eq!(
        last, settled,
        "a paged file under steady churn must stop growing (sizes by round: {sizes:?})"
    );

    let f = File::open(&path).unwrap();
    for round in ROUNDS - LIVE..ROUNDS {
        let g = f.group(&format!("g{round}")).unwrap();
        for name in ["a", "b"] {
            assert_eq!(g.dataset(name).unwrap().read_i32().unwrap(), rows);
        }
    }
    drop(f);
    assert_eof_matches_file(&path);
}

/// A paged commit's tail — the rewritten extension and manager blocks — is
/// placed in space the file already had free, instead of opening a page for
/// itself at end-of-file.
///
/// The size test above states the consequence; this states the mechanism, which a
/// size assertion alone would not distinguish from the file simply having room to
/// spare. The tail does not settle at one fixed address — best fit picks whatever
/// hole suits as the free list shuffles — so what is asserted is that whatever
/// address it takes was free *before* the commit that wrote it, which is exactly
/// what it never was while the tail could only append.
#[test]
fn a_paged_commit_tail_is_placed_in_free_space() {
    const PAGE: u64 = 16384;
    let path = temp_path("hdf5_pure_fs_paged_tail_reuse.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("seed").with_i32_data(&[0i32; 4]);
    b.with_file_space_strategy(FileSpaceStrategy::Page, true, 1)
        .with_file_space_page_size(PAGE);
    b.write(&path).unwrap();
    let start_len = std::fs::metadata(&path).unwrap().len();

    // Three commits that change nothing but one attribute and the tail the commit
    // has to rewrite regardless.
    for i in 0..3 {
        let before: Vec<(u64, u64)> = File::open(&path).unwrap().persisted_free_space();
        let f = File::open_rw(&path).unwrap();
        f.root().set_attr("n", AttrValue::I64(i)).unwrap();
        f.commit().unwrap();
        drop(f);

        let f = File::open(&path).unwrap();
        let managers: Vec<u64> = f
            .file_space_info()
            .expect("a persisting file records its managers")
            .manager_addrs
            .iter()
            .copied()
            .filter(|&a| a != u64::MAX)
            .collect();
        assert!(
            !managers.is_empty(),
            "commit {i}: the file has free space, so it has managers to place"
        );
        for addr in managers {
            assert!(
                before.iter().any(|&(a, len)| addr >= a && addr < a + len),
                "commit {i}: a manager block landed at {addr}, which was not free \
                 before the commit ({before:?}) — the tail appended instead of \
                 reusing"
            );
        }
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            start_len,
            "commit {i}: rewriting the tail must not grow the file"
        );
        // Once the tail is reusing rather than appending, a commit that rewrites
        // only the tail neither gains nor loses free space: the new tail takes
        // exactly what the old one gives back. A commit that took more room than
        // it filled would show up here as free space quietly draining away, which
        // is the same leak in miniature.
        if i > 0 {
            let before_total: u64 = before.iter().map(|(_, len)| len).sum();
            let after_total: u64 = f.persisted_free_space().iter().map(|(_, len)| len).sum();
            assert_eq!(
                after_total, before_total,
                "commit {i}: rewriting the tail must conserve free space"
            );
        }
    }
    assert_eof_matches_file(&path);
}

/// Across a sweep of layouts, a paged commit that rewrites only its tail
/// conserves free space exactly.
///
/// The tail is sized against the free lists it then draws from, so its length and
/// the hole it lands in determine each other, and the arithmetic settles
/// differently depending on how the file happens to be laid out — whether the
/// chosen hole is consumed whole or merely shrunk, whether the remainder changes
/// which manager records it. A single fixture exercises one of those settlements.
/// Sweeping the filler size walks the tail through many, and the invariant is the
/// same in all of them: what the new tail takes is what the old one gave back,
/// to the byte. A tail that reserved more room than it filled would leave the
/// difference recorded by nobody, and the total would drop.
#[test]
fn a_paged_tail_conserves_free_space_across_layouts() {
    const PAGE: u64 = 16384;
    for filler in 0..64usize {
        let path = temp_path(&format!("hdf5_pure_fs_paged_sweep_{filler}.h5"));
        let mut b = FileBuilder::new();
        b.create_dataset("seed")
            .with_i32_data(&(0..100 + filler as i32).collect::<Vec<i32>>());
        b.with_file_space_strategy(FileSpaceStrategy::Page, true, 1)
            .with_file_space_page_size(PAGE);
        b.write(&path).unwrap();

        // The first commit settles the tail into the file's own free space; from
        // the second on, each tail replaces the previous one and nothing else
        // moves.
        //
        // `OnClose` for the same reason as the flat sweep below: what these
        // sessions are read for is the free space each one leaves recorded, and
        // the ordering `Always` buys with two or three `fsync`s a commit is not
        // something any assertion here looks at.
        let mut totals = Vec::new();
        for i in 0..3 {
            let f = File::open_rw_with_options(
                &path,
                FileAccessProperties::new().with_sync_policy(SyncPolicy::OnClose),
            )
            .unwrap();
            f.root().set_attr("n", AttrValue::I64(i)).unwrap();
            f.commit().unwrap();
            drop(f);
            totals.push(
                File::open(&path)
                    .unwrap()
                    .persisted_free_space()
                    .iter()
                    .map(|(_, len)| len)
                    .sum::<u64>(),
            );
        }
        assert_eq!(
            totals[1], totals[2],
            "filler {filler}: rewriting the tail must conserve free space, not \
             quietly retire some of it (totals {totals:?})"
        );
    }
}

/// A **non-paged** persisting file's commit tail — the rewritten extension and
/// its one `FSHD`/`FSSE` pair — is placed in space an earlier commit freed, the
/// same way the paged tail above is (issue #358).
///
/// The paged tail learned this first (issue #286); the flat one kept appending,
/// so every commit grew the file by a tail and freed its predecessor's, and a
/// `FileSpaceStrategy::FsmAggr` file under churn climbed forever while reporting
/// all of it as reusable. As above, the address is not fixed — best fit picks
/// whatever hole suits — so what is asserted is that the manager landed where the
/// file was already free, which is what it never was while the tail could only
/// append.
#[test]
fn a_persisting_commit_tail_is_placed_in_free_space() {
    let path = temp_path("hdf5_pure_fs_flat_tail_reuse.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("seed").with_i32_data(&[0i32; 4]);
    b.create_dataset("scratch").with_i32_data(&[7i32; 256]);
    // A live object above the hole, so the deletion below leaves an interior hole
    // rather than a run reaching end-of-file.
    b.create_dataset("ceiling").with_i32_data(&[9i32; 4]);
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    b.write(&path).unwrap();

    // The hole the tail will move into: a fresh file has none, so the first tail
    // has nowhere to go but end-of-file.
    {
        let f = File::open_rw(&path).unwrap();
        f.root().delete("scratch").unwrap();
        f.commit().unwrap();
    }
    let start_len = std::fs::metadata(&path).unwrap().len();

    // Three commits that change nothing but one attribute and the tail the commit
    // has to rewrite regardless.
    for i in 0..3 {
        let before: Vec<(u64, u64)> = File::open(&path).unwrap().persisted_free_space();
        let f = File::open_rw(&path).unwrap();
        f.root().set_attr("n", AttrValue::I64(i)).unwrap();
        f.commit().unwrap();
        drop(f);

        let f = File::open(&path).unwrap();
        let managers: Vec<u64> = f
            .file_space_info()
            .expect("a persisting file records its managers")
            .manager_addrs
            .iter()
            .copied()
            .filter(|&a| a != u64::MAX)
            .collect();
        assert!(
            !managers.is_empty(),
            "commit {i}: the file has free space, so it has a manager to place"
        );
        for addr in managers {
            assert!(
                before.iter().any(|&(a, len)| addr >= a && addr < a + len),
                "commit {i}: a manager block landed at {addr}, which was not free \
                 before the commit ({before:?}) — the tail appended instead of \
                 reusing"
            );
        }
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            start_len,
            "commit {i}: rewriting the tail must not grow the file"
        );
    }

    assert_eof_matches_file(&path);
    let f = File::open(&path).unwrap();
    assert_eq!(f.dataset("seed").unwrap().read_i32().unwrap(), [0; 4]);
    assert_eq!(f.dataset("ceiling").unwrap().read_i32().unwrap(), [9; 4]);
}

/// The consequence of the test above, on the workload the issue describes: a
/// non-paged persisting file under delete-and-recreate churn stops growing
/// (issue #358).
///
/// Shaped like [`paged_churn_reaches_a_steady_size`]: the file may reach whatever
/// size its live data and layout need over the first rounds, but once the
/// workload is in its steady state it may not keep climbing. A tail appended per
/// commit is what climbing looks like from the outside — about 590 bytes a round
/// here, forever, with every one of them recorded as free.
#[test]
fn persisting_churn_reaches_a_steady_size() {
    const ROUNDS: usize = 16;
    const LIVE: usize = 2;

    let path = temp_path("hdf5_pure_fs_flat_churn.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("seed").with_i32_data(&[0i32; 4]);
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    b.write(&path).unwrap();

    let rows: Vec<i32> = (0..500).map(|i| i % 97).collect();
    let mut sizes = Vec::new();
    for round in 0..ROUNDS {
        // A fresh session per round, so each round's tail is placed from the free
        // list a reopen seeds out of the on-disk managers rather than from one the
        // session has been carrying.
        {
            let f = File::open_rw(&path).unwrap();
            f.root().create_group(&format!("g{round}")).unwrap();
            f.commit().unwrap();
            let g = f.group(&format!("g{round}")).unwrap();
            for name in ["a", "b"] {
                g.create_dataset(name, |b| {
                    b.with_i32_data(&rows);
                })
                .unwrap();
            }
            f.commit().unwrap();
        }
        if round >= LIVE {
            let f = File::open_rw(&path).unwrap();
            f.root().delete(&format!("g{}", round - LIVE)).unwrap();
            f.commit().unwrap();
            drop(f);
            sizes.push(std::fs::metadata(&path).unwrap().len());
        }
    }

    let settled = sizes[sizes.len() * 2 / 3];
    let last = *sizes.last().unwrap();
    assert_eq!(
        last, settled,
        "a persisting file under steady churn must stop growing (sizes by round: {sizes:?})"
    );

    let f = File::open(&path).unwrap();
    for round in ROUNDS - LIVE..ROUNDS {
        let g = f.group(&format!("g{round}")).unwrap();
        for name in ["a", "b"] {
            assert_eq!(g.dataset(name).unwrap().read_i32().unwrap(), rows);
        }
    }
    drop(f);
    assert_eof_matches_file(&path);
}

/// Across a sweep of layouts, a non-paged persisting file whose commits rewrite
/// only the tail holds its size (issue #358).
///
/// The flat counterpart of [`a_paged_tail_conserves_free_space_across_layouts`],
/// asserting size rather than the free-space total, because the two say different
/// things here: the flat tail may take a few more bytes than its blocks fill, so
/// consecutive commits legitimately record totals a section record apart, while
/// the file's *size* may not move at all once a tail is landing in free space.
///
/// A single fixture reaches one settlement of an arithmetic that depends on the
/// whole free list — the tail's length and the hole it lands in determine each
/// other — so this sweeps the filler size and the file carries three holes rather
/// than one, which is what makes the settlement turn over as the tail moves. The
/// difference that buys is measured: requiring the tail to fill its reservation
/// exactly leaves `a_persisting_commit_tail_is_placed_in_free_space` passing, and
/// fails here from filler 18 on, where the file gains a tail per commit again.
#[test]
fn a_persisting_tail_holds_its_size_across_layouts() {
    for filler in 0..64usize {
        let path = temp_path(&format!("hdf5_pure_fs_flat_sweep_{filler}.h5"));
        let mut b = FileBuilder::new();
        b.create_dataset("seed")
            .with_i32_data(&(0..100 + filler as i32).collect::<Vec<i32>>());
        // Three holes of unrelated sizes rather than one, so the free list the
        // tail is sized against has several sections and its length turns over as
        // the tail moves between them.
        for (i, len) in [256usize, 37, 91].into_iter().enumerate() {
            b.create_dataset(&format!("scratch{i}"))
                .with_i32_data(&vec![7i32; len + filler]);
            b.create_dataset(&format!("keep{i}"))
                .with_i32_data(&[9i32; 4]);
        }
        b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
        b.write(&path).unwrap();

        // Nine sessions per filler, 576 across the sweep, and what each one is
        // read for is the size of the file it leaves behind. `SyncPolicy::Always`
        // would spend two or three `fsync`s per commit establishing an ordering
        // no assertion here looks at; `OnClose` keeps the one barrier that is
        // load-bearing, at the close before each size is read, and writes the
        // same bytes either way (`tests/sync_policy.rs`).
        let open = |path: &std::path::Path| {
            File::open_rw_with_options(
                path,
                FileAccessProperties::new().with_sync_policy(SyncPolicy::OnClose),
            )
            .unwrap()
        };

        for i in 0..3 {
            let f = open(&path);
            f.root().delete(&format!("scratch{i}")).unwrap();
            f.commit().unwrap();
        }

        // The first of these settles the tail into a hole; from the second on,
        // each commit replaces an earlier tail and nothing else moves.
        let mut sizes = Vec::new();
        for i in 0..6 {
            let f = open(&path);
            f.root().set_attr("n", AttrValue::I64(i)).unwrap();
            f.commit().unwrap();
            drop(f);
            sizes.push(std::fs::metadata(&path).unwrap().len());
        }
        assert_eq!(
            sizes[1],
            *sizes.last().unwrap(),
            "filler {filler}: rewriting the tail must not grow the file (sizes {sizes:?})"
        );
        assert_eof_matches_file(&path);
    }
}

/// A paged file reuses a freed hole across a close, not only within one session
/// (issue #358, which reported the opposite).
///
/// `paged_commit_reuses_freed_space_within_its_page_type` deletes and re-adds in
/// one session, where the hole is in the list that session has been carrying. Here
/// the delete and the re-add are separate sessions, so the second one knows the
/// hole only from the per-page-type managers it seeds its free list from on open —
/// the paged counterpart of `persisted_chunked_free_space_is_reused_after_a_reopen`.
#[test]
fn paged_free_space_is_reused_after_a_reopen() {
    const ELEMS: usize = 32768;
    let path = temp_path("hdf5_pure_fs_paged_persist_reuse.h5");
    let mut b = FileBuilder::new();
    b.with_file_space_strategy(FileSpaceStrategy::Page, true, 1);
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    b.create_dataset("big")
        .with_f64_data(&vec![1.0; ELEMS])
        .with_chunks(&[4096]);
    b.create_dataset("tail").with_i32_data(&[9; 16]);
    b.write(&path).unwrap();
    let start = std::fs::metadata(&path).unwrap().len();

    {
        let s = File::open_rw(&path).unwrap();
        s.root().delete("big").unwrap();
        s.commit().unwrap();
    }
    {
        let s = File::open_rw(&path).unwrap();
        assert!(
            s.space_accounting().unwrap().reusable_free_bytes >= f64_data_bytes(ELEMS),
            "the reopened session must recover the freed dataset's pages"
        );
        s.root()
            .create_dataset("big2", |b| {
                b.with_f64_data(&vec![2.0; ELEMS]).with_chunks(&[4096]);
            })
            .unwrap();
        s.commit().unwrap();
    }

    let end = std::fs::metadata(&path).unwrap().len();
    assert!(
        end < start + f64_data_bytes(ELEMS) / 4,
        "a reopened paged session should write into the pages the delete freed \
         (start={start}, end={end})"
    );

    assert_eof_matches_file(&path);
    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("big2").unwrap().read_f64().unwrap(),
        vec![2.0; ELEMS]
    );
    assert_eq!(
        f.dataset("keep").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    assert_eq!(f.dataset("tail").unwrap().read_i32().unwrap(), vec![9; 16]);
}

/// Build `name` as a rank-1, unlimited, chunked i32 dataset of length zero.
fn create_log(session: &File, name: &str) {
    session
        .root()
        .create_dataset(name, |b| {
            b.with_i32_data(&[])
                .with_shape(&[0])
                .with_chunks(&[1024])
                .with_maxshape(&[u64::MAX]);
        })
        .unwrap();
    session.commit().unwrap();
}

/// An immediate [`Dataset::append`] must allocate its chunks out of the free
/// space a prior commit left, exactly as `create_dataset` + `commit` does
/// (issue #349). Before the fix the append always extended end-of-file, so the
/// replacement dataset doubled the file rather than moving into the hole its
/// predecessor vacated.
#[test]
fn immediate_append_reuses_a_freed_hole() {
    let payload: Vec<i32> = (0..16384).collect();
    let path = temp_path("hdf5_pure_fs_immediate_append_reuse.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    b.write(&path).unwrap();

    let session = File::open_rw(&path).unwrap();
    create_log(&session, "log");
    session.dataset("log").unwrap().append(&payload).unwrap();
    let after_first = std::fs::metadata(&path).unwrap().len();

    // A ceiling above the churned region: without a live object after it, the
    // deletion below leaves a free run reaching end-of-file that the commit
    // truncates away, and the test would pass on truncation rather than on reuse.
    session
        .root()
        .create_dataset("ceiling", |b| {
            b.with_i32_data(&[9, 9, 9]);
        })
        .unwrap();
    session.commit().unwrap();

    session.root().delete("log").unwrap();
    session.commit().unwrap();
    let freed = session.space_accounting().unwrap().reusable_free_bytes;
    assert!(
        freed >= payload.len() as u64 * 4,
        "the deleted dataset's chunks should be reusable (only {freed} bytes are)"
    );

    create_log(&session, "log2");
    session.dataset("log2").unwrap().append(&payload).unwrap();
    session.close().unwrap();
    let after_second = std::fs::metadata(&path).unwrap().len();

    assert!(
        after_second < after_first + freed / 2,
        "the second append should have moved into the {freed}-byte hole, not \
         extended the file (was {after_first}, now {after_second})"
    );
    assert_eof_matches_file(&path);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("log2").unwrap().read_i32().unwrap(), payload);
    // Reuse must not have written over either survivor.
    assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), [1, 2, 3]);
    assert_eq!(
        file.dataset("ceiling").unwrap().read_i32().unwrap(),
        [9, 9, 9]
    );
}

/// A file that **persists** its free-space managers is the one case an immediate
/// append must keep extending end-of-file (issue #349).
///
/// Its holes are recorded on disk, and only a commit or the close-time rewrite
/// updates that record. An append has neither: it publishes through its own four
/// phases and never repoints the superblock, so an append that consumed a
/// persisted hole and then lost its process would leave a manager offering space
/// a live chunk occupies, and the next session would allocate over the top of it.
/// The staged `append_staged` + `commit` remains the way to reuse here.
///
/// Asserted as "the persisted free space is still there afterwards" rather than
/// as a file-size comparison, because it is the *manager* staying truthful that
/// the carve-out is about.
#[test]
fn a_persisting_file_appends_at_end_of_file() {
    let payload: Vec<i32> = (0..16384).collect();
    let path = temp_path("hdf5_pure_fs_immediate_append_persisting.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    // Comfortably larger than the payload, so the hole still holds the whole
    // append after the commits below have placed their own tails inside it.
    b.create_dataset("scratch")
        .with_i32_data(&vec![7; payload.len() + 1024]);
    b.create_dataset("ceiling").with_i32_data(&[9, 9, 9]);
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    b.write(&path).unwrap();

    let session = File::open_rw(&path).unwrap();
    session.root().delete("scratch").unwrap();
    session.commit().unwrap();
    // After the commit that creates the dataset, so the staged path's own
    // (legitimate) reuse is not counted against the append.
    create_log(&session, "log");
    let freed = session.space_accounting().unwrap().reusable_free_bytes;
    assert!(freed >= payload.len() as u64 * 4, "expected a sizable hole");

    session.dataset("log").unwrap().append(&payload).unwrap();
    assert_eq!(
        session.space_accounting().unwrap().reusable_free_bytes,
        freed,
        "an immediate append must not spend free space the on-disk managers \
         still advertise"
    );
    session.close().unwrap();

    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("log").unwrap().read_i32().unwrap(), payload);
    let persisted: u64 = file.persisted_free_space().iter().map(|(_, l)| l).sum();
    assert!(
        persisted >= freed,
        "the managers must still describe the untouched hole ({persisted} of {freed})"
    );
}
