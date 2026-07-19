//! Live space accounting on [`EditSession`] (issue #150).
//!
//! [`EditSession::space_accounting`] reports the file *as the session currently
//! holds it*: its live logical size and the free space it can reuse. These tests
//! pin the semantics the docs promise — the size tracks immediate in-place
//! appends, reusable free reflects only committed frees (not staged edits nor a
//! fresh non-persisting session's pre-existing holes), a persisting session seeds
//! its reusable free from disk on open, and the total always equals the summed
//! region lengths.

use hdf5_pure::{EditSession, File, FileBuilder, FileSpaceStrategy};
use tempfile::tempdir;

/// The scalar total must always equal the summed lengths of the reported regions,
/// which must be sorted, disjoint, and contained within the logical size.
fn assert_internally_consistent(acct: &hdf5_pure::SpaceAccounting) {
    let summed: u64 = acct.reusable_free_space.iter().map(|(_, len)| len).sum();
    assert_eq!(
        summed, acct.reusable_free_bytes,
        "reusable_free_bytes must equal the summed region lengths: {:?}",
        acct.reusable_free_space
    );
    for w in acct.reusable_free_space.windows(2) {
        assert!(
            w[0].0 + w[0].1 <= w[1].0,
            "reusable free regions must be sorted and disjoint: {:?}",
            acct.reusable_free_space
        );
    }
    for &(addr, len) in &acct.reusable_free_space {
        assert!(
            addr + len <= acct.logical_size,
            "a free region [{addr}, {}) must lie within logical_size {}",
            addr + len,
            acct.logical_size
        );
    }
}

/// Build a plain (non-persisting) file with three i32 datasets `a`, `big`, `c` in
/// that order, so deleting `big` leaves an interior hole (not a trailing run that
/// would be truncated away).
fn build_a_big_c(path: &std::path::Path, persist: bool) {
    let mut b = FileBuilder::new();
    b.create_dataset("a").with_i32_data(&[1; 100]);
    b.create_dataset("big").with_i32_data(&[7; 400]); // 1600 bytes of raw data
    b.create_dataset("c").with_i32_data(&[3; 100]);
    if persist {
        b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    }
    b.write(path).unwrap();
}

#[test]
fn fresh_session_reports_file_size_and_no_reusable_free() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("plain.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_f64_data(&[1.0, 2.0, 3.0, 4.0]);
    b.write(&p).unwrap();

    let on_disk = std::fs::metadata(&p).unwrap().len();
    let s = EditSession::open(&p).unwrap();
    let acct = s.space_accounting();

    assert_eq!(
        acct.logical_size, on_disk,
        "logical_size must equal the physical file length"
    );
    assert_eq!(
        acct.reusable_free_bytes, 0,
        "a freshly opened non-persisting session tracks no reusable free space"
    );
    assert!(acct.reusable_free_space.is_empty());
    assert_internally_consistent(&acct);
}

#[test]
fn logical_size_grows_with_immediate_append() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("append.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&(0..8).collect::<Vec<_>>())
        .with_shape(&[8])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[4]);
    b.write(&p).unwrap();

    let mut s = EditSession::open(&p).unwrap();
    let before = s.space_accounting().logical_size;

    // An immediate, durable in-place append: the mirror grows and stays in lockstep
    // with the on-disk file, so logical_size reflects it at once.
    s.append_inplace_i32("d", &[8, 9, 10, 11, 12, 13, 14, 15])
        .unwrap();
    let acct = s.space_accounting();

    assert!(
        acct.logical_size > before,
        "an in-place append must grow the live logical size ({before} -> {})",
        acct.logical_size
    );
    assert_eq!(
        acct.logical_size,
        std::fs::metadata(&p).unwrap().len(),
        "logical_size must match the on-disk file length after a durable append"
    );
    assert_eq!(
        acct.reusable_free_bytes, 0,
        "an in-place append feeds no reusable free space (any abandoned index block \
         is untracked, never reused)"
    );
    assert_internally_consistent(&acct);
}

#[test]
fn staged_delete_is_not_counted_until_commit() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("staged.h5");
    build_a_big_c(&p, false);

    let mut s = EditSession::open(&p).unwrap();
    assert!(!s.has_staged_edits());
    assert_eq!(s.space_accounting().reusable_free_bytes, 0);

    // Staging the delete must not change the accounting: bytes are freed at commit.
    s.delete("big");
    assert!(s.has_staged_edits());
    assert_eq!(
        s.space_accounting().reusable_free_bytes,
        0,
        "a staged (uncommitted) delete frees nothing yet"
    );

    // Committing frees `big`'s interior storage, which becomes reusable.
    s.commit().unwrap();
    assert!(!s.has_staged_edits());
    let acct = s.space_accounting();
    assert!(
        acct.reusable_free_bytes >= 1600,
        "committing the delete of an interior dataset makes its ~1600 bytes reusable \
         (got {})",
        acct.reusable_free_bytes
    );
    assert_internally_consistent(&acct);
}

#[test]
fn reused_free_shrinks_the_reusable_total() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("reuse.h5");
    build_a_big_c(&p, false);

    {
        let mut s = EditSession::open(&p).unwrap();
        s.delete("big");
        s.commit().unwrap();
        let free_before = s.space_accounting().reusable_free_bytes;
        assert!(free_before >= 1600);

        // A new dataset that fits the freed hole reuses it rather than growing the
        // file, so the reusable total drops.
        s.create_dataset("d").with_i32_data(&[9; 300]); // 1200 bytes, fits the hole
        s.commit().unwrap();
        let acct = s.space_accounting();
        assert!(
            acct.reusable_free_bytes < free_before,
            "reusing the hole must shrink the reusable total ({free_before} -> {})",
            acct.reusable_free_bytes
        );
        assert_internally_consistent(&acct);
    }

    // Reopen only after the session (and its exclusive file lock) is dropped —
    // Windows enforces the lock against a concurrent `File::open`.
    let f = File::open(&p).unwrap();
    assert_eq!(f.dataset("d").unwrap().read_i32().unwrap(), vec![9; 300]);
    assert_eq!(f.dataset("a").unwrap().read_i32().unwrap(), vec![1; 100]);
}

#[test]
fn fresh_nonpersisting_session_ignores_existing_holes() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("holes.h5");
    build_a_big_c(&p, false);

    // Session 1 leaves an interior hole on disk (untracked, since not persisting).
    {
        let mut s = EditSession::open(&p).unwrap();
        s.delete("big");
        s.commit().unwrap();
    }

    // Session 2, freshly opened, does not scan for or track those holes.
    let s = EditSession::open(&p).unwrap();
    let acct = s.space_accounting();
    assert_eq!(
        acct.reusable_free_bytes, 0,
        "a non-persisting reopen ignores holes left by a prior session"
    );
    assert!(acct.reusable_free_space.is_empty());
    assert_eq!(acct.logical_size, std::fs::metadata(&p).unwrap().len());
    assert_internally_consistent(&acct);
}

#[test]
fn persisting_session_seeds_reusable_free_on_open() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("persist.h5");
    build_a_big_c(&p, true);

    // Session 1: delete `big`; with persistence on, its storage is recorded on disk.
    {
        let mut s = EditSession::open(&p).unwrap();
        s.delete("big");
        s.commit().unwrap();
    }

    // What a plain reader recovers from the on-disk managers is the yardstick.
    let persisted: u64 = File::open(&p)
        .unwrap()
        .persisted_free_space()
        .iter()
        .map(|(_, len)| len)
        .sum();
    assert!(
        persisted >= 1600,
        "the delete must persist as free space on disk"
    );

    // Session 2: opening seeds the free list from those managers, so the accounting
    // reports the reusable space immediately, before any edit in this session.
    let s = EditSession::open(&p).unwrap();
    let acct = s.space_accounting();
    assert_eq!(
        acct.reusable_free_bytes, persisted,
        "a persisting session seeds its reusable free from disk to match the reader"
    );
    assert!(!acct.reusable_free_space.is_empty());
    assert_internally_consistent(&acct);
}
