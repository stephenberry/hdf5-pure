//! Free-space reuse and truncation during in-place editing (issue #21).
//!
//! `EditSession` records the regions a commit vacates — deleted objects' blocks
//! and superseded group headers — and, within the same session, reuses them for
//! later writes instead of growing the file, truncating the file when a freed
//! run reaches end-of-file. These tests pin down both the size behavior and that
//! survivors stay byte-exact and the file stays valid.

use hdf5_pure::{EditSession, File, FileBuilder, FileSpaceStrategy};

fn tmp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(name)
}

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
    let path = tmp("hdf5_pure_fs_shrink.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep")
        .with_f64_data(&[1.0, 2.0, 3.0, 4.0]);
    b.write(&path).unwrap();
    let size_start = std::fs::metadata(&path).unwrap().len();

    // A single session: add a large dataset, then delete it. The deleted blocks
    // and the superseded root header form a run reaching end-of-file, so the
    // file is truncated back down rather than left bloated.
    {
        let mut s = EditSession::open(&path).unwrap();
        s.create_dataset("big").with_f64_data(&vec![7.0; 1024]);
        s.commit().unwrap();
        let size_after_add = std::fs::metadata(&path).unwrap().len();
        assert!(size_after_add > size_start, "adding should grow the file");

        s.delete("big");
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
    std::fs::remove_file(&path).ok();
}

#[test]
fn churn_within_session_stays_bounded() {
    let path = tmp("hdf5_pure_fs_churn.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    b.write(&path).unwrap();

    let mut high_water = 0u64;
    {
        let mut s = EditSession::open(&path).unwrap();
        // Repeatedly add then delete a sizable dataset in the same session. With
        // reuse + truncation the file must not grow without bound across cycles.
        for i in 0..8 {
            s.create_dataset("scratch")
                .with_f64_data(&vec![i as f64; 512]);
            s.commit().unwrap();
            high_water = high_water.max(std::fs::metadata(&path).unwrap().len());
            s.delete("scratch");
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
    std::fs::remove_file(&path).ok();
}

#[test]
fn reuse_keeps_survivors_byte_exact() {
    let path = tmp("hdf5_pure_fs_reuse_exact.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("a").with_i32_data(&[10, 20, 30]);
    b.create_dataset("b").with_f64_data(&[1.5, 2.5]);
    b.write(&path).unwrap();

    {
        let mut s = EditSession::open(&path).unwrap();
        // Delete b, then add c in a later commit: c's bytes should land in the
        // region b vacated. a and the newly written c must both read back exact.
        s.delete("b");
        s.commit().unwrap();
        s.create_dataset("c").with_i32_data(&[7, 8, 9, 10]);
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
    std::fs::remove_file(&path).ok();
}

#[test]
fn delete_subtree_reclaims_all_members() {
    let path = tmp("hdf5_pure_fs_subtree.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[1]);
    b.write(&path).unwrap();

    {
        let mut s = EditSession::open(&path).unwrap();
        s.create_group("grp");
        s.create_dataset("grp/x").with_f64_data(&vec![1.0; 256]);
        s.create_dataset("grp/y").with_f64_data(&vec![2.0; 256]);
        s.commit().unwrap();
        let with_group = std::fs::metadata(&path).unwrap().len();

        s.delete("grp");
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
    std::fs::remove_file(&path).ok();
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
    // EditSession does not yet expose, and remains future work.)
    let path = tmp("hdf5_pure_fs_trailing_slack.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[11, 22, 33]);
    b.write(&path).unwrap();

    // Add then delete a large dataset so the second commit truncates the file
    // back down, leaving the recorded end-of-file equal to the physical size.
    {
        let mut s = EditSession::open(&path).unwrap();
        s.create_dataset("scratch").with_f64_data(&vec![5.0; 2048]);
        s.commit().unwrap();
        s.delete("scratch");
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
    std::fs::remove_file(&path).ok();
}

#[test]
fn deleting_chunked_dataset_leaks_but_does_not_corrupt() {
    // A chunked/compressed dataset's storage (B-tree index + chunks) is not one
    // of the block classes the free-walk enumerates, so its bytes are left
    // behind rather than risk freeing a region still in use. The delete must
    // still succeed and leave a valid file with the survivor intact.
    let path = tmp("hdf5_pure_fs_chunked_leak.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[42, 43, 44]);
    b.create_dataset("comp")
        .with_f64_data(&vec![3.0; 4096])
        .with_chunks(&[512])
        .with_deflate(6);
    b.write(&path).unwrap();

    {
        let mut s = EditSession::open(&path).unwrap();
        s.delete("comp");
        s.commit().unwrap();
    }

    assert_eof_matches_file(&path);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), vec!["keep".to_string()]);
    assert_eq!(
        file.dataset("keep").unwrap().read_i32().unwrap(),
        vec![42, 43, 44]
    );
    assert!(file.dataset("comp").is_err());
    std::fs::remove_file(&path).ok();
}

#[test]
fn persisted_free_space_survives_reopen_and_is_reused() {
    // A file created with persist = true keeps its freed space recorded on disk
    // (the FSHD/FSSE managers), so a freed region survives close/reopen and a
    // later session reuses it instead of growing the file. This is the cross-
    // session counterpart to the within-session reuse above.
    let path = tmp("hdf5_pure_fs_persist_roundtrip.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("a").with_i32_data(&[1; 100]);
    b.create_dataset("big").with_i32_data(&[7; 400]); // 1600 bytes of raw data
    b.create_dataset("c").with_i32_data(&[3; 100]);
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    b.write(&path).unwrap();

    // Session 1: delete "big"; its storage is persisted as free space.
    {
        let mut s = EditSession::open(&path).unwrap();
        s.delete("big");
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
        let mut s = EditSession::open(&path).unwrap();
        s.create_dataset("d").with_i32_data(&[9; 300]); // 1200 bytes, fits the hole
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
    std::fs::remove_file(&path).ok();
}

#[test]
fn persisted_managers_stay_consistent_across_many_commits() {
    // Several persisting commits in a row: each supersedes the previous on-disk
    // managers and extension, recording them as free, so the file stays valid and
    // its tracked free space never double-counts or loses a region. Both this
    // crate and a fresh reader must agree on the result after every step.
    let path = tmp("hdf5_pure_fs_persist_multi.h5");
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
            let mut s = EditSession::open(&path).unwrap();
            s.delete(&format!("d{round}"));
            s.create_dataset(&format!("n{round}"))
                .with_i32_data(&vec![100 + round; 150]);
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
    std::fs::remove_file(&path).ok();
}

#[test]
fn corrupt_persisted_section_is_skipped_not_fatal() {
    // A malformed free-space manager must not be trusted blindly: if a persisted
    // section claims a region past end-of-file, seeding it and later handing it
    // out would write out of bounds. The editor skips such a section instead, so
    // the open + commit still succeeds and the live data stays intact.
    let path = tmp("hdf5_pure_fs_persist_corrupt.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[5; 50]);
    b.create_dataset("victim").with_i32_data(&[6; 300]); // 1200-byte data block
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    b.write(&path).unwrap();
    {
        let mut s = EditSession::open(&path).unwrap();
        s.delete("victim"); // frees the largest tracked section
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
        let mut s = EditSession::open(&path).unwrap();
        s.create_dataset("added").with_i32_data(&[9; 250]); // 1000 bytes
        s.commit().unwrap();
    }
    assert_eof_matches_file(&path);
    let f = File::open(&path).unwrap();
    assert_eq!(f.dataset("keep").unwrap().read_i32().unwrap(), vec![5; 50]);
    assert_eq!(
        f.dataset("added").unwrap().read_i32().unwrap(),
        vec![9; 250]
    );
    std::fs::remove_file(&path).ok();
}
