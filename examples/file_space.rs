//! File-space strategy and persistent free-space *reuse*.
//!
//! Mirroring `H5Pset_file_space_strategy` / `H5Pset_file_space_page_size`, a
//! written file can record how it manages free space. With `persist = true`, the
//! regions an `EditSession` frees are written to on-disk free-space-manager
//! blocks, so a *later* session (this crate's or the reference C library's) seeds
//! its free list from them and writes new objects into the holes instead of
//! growing the file.
//!
//! This example proves that reuse actually happens rather than just that free
//! space is tracked: it deletes a dataset, then adds a same-sized one in a fresh
//! session and shows the file barely grows. A non-persisting file is run through
//! the identical churn as a control, where the freed space is forgotten on close
//! and the file grows by the full size of the new data.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example file_space
//! ```

#![allow(deprecated)] // exercises the deprecated EditSession/SwmrWriter shims (issue #148)
use hdf5_pure::{EditSession, File, FileBuilder, FileSpaceStrategy};
use std::path::Path;

/// 4096 f64 values = 32 KiB of raw data — large enough that whether the file
/// reuses the freed hole or grows is unmistakable in the byte counts.
const SCRATCH_LEN: usize = 4096;
const SCRATCH_BYTES: u64 = (SCRATCH_LEN * size_of::<f64>()) as u64;

fn main() {
    let dir = tempfile::tempdir().expect("temp dir");

    // ---- Persisting file: free space is reused across sessions ----------
    let persisting = dir.path().join("managed.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    // A free-space-manager strategy that persists across sessions. (The paged
    // strategy also persists, but a paged file is grown only through the
    // append-only `File::open_rw_bounded`, which cannot reuse a freed hole the
    // way this delete-then-re-add demo does.)
    builder.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1); // strategy, persist, threshold
    builder.write(&persisting).expect("write file");

    // The strategy is stored in a superblock-extension message and survives a
    // reopen (the reference C library observes it too).
    let strategy = File::open(&persisting).unwrap().file_space_strategy();
    assert_eq!(strategy, Some(FileSpaceStrategy::FsmAggr));
    println!("recorded strategy: {strategy:?}\n");

    println!("persisting file (strategy = FsmAggr, persist = true):");
    let reuse_growth = churn(&persisting);

    // Because the file persists free space, deleting `scratch` recorded its
    // region on disk; the next session seeded its free list from that record and
    // wrote `scratch2` into the hole, so the file grew by far less than the new
    // data. (Reused bytes never reach end-of-file, so nothing is truncated.)
    assert!(
        reuse_growth * 4 < SCRATCH_BYTES,
        "expected reuse: re-adding {SCRATCH_BYTES} bytes grew the file by only \
         {reuse_growth}, a small fraction of the data — the freed hole was reused"
    );

    // ---- Control: a non-persisting file forgets its free list -----------
    // Identical churn on a default-strategy file. Here the free list lives only
    // for the open session and is discarded on close, so the fresh session that
    // adds `scratch2` has no record of the hole and must grow the file.
    let default = dir.path().join("default.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    builder.write(&default).expect("write file");

    println!("\ncontrol — non-persisting file (default strategy):");
    let control_growth = churn(&default);

    assert!(
        control_growth >= SCRATCH_BYTES,
        "expected no cross-session reuse: re-adding {SCRATCH_BYTES} bytes should \
         grow the file by at least that, got {control_growth}"
    );

    // ---- Both files still hold the original and the re-added data -------
    for path in [&persisting, &default] {
        let file = File::open(path).unwrap();
        assert_eq!(
            file.dataset("keep").unwrap().read_i32().unwrap(),
            vec![1, 2, 3]
        );
        assert_eq!(
            file.dataset("scratch2").unwrap().read_f64().unwrap(),
            vec![7.0; SCRATCH_LEN]
        );
    }

    println!(
        "\nverified: persisting reuse grew the file by {reuse_growth} bytes vs \
         {control_growth} for the non-persisting control ({SCRATCH_BYTES} bytes of new data)"
    );
}

/// Add a dataset, delete it, then add a same-sized one in a *fresh* session.
/// Returns how many bytes the file grew on that last add — the figure that
/// reveals whether the freed region was reused (small) or not (≈ the data size).
fn churn(path: &Path) -> u64 {
    // Add `scratch`, then delete it in its own session so the region is freed.
    let mut session = EditSession::open(path).expect("open for editing");
    session
        .create_dataset("scratch")
        .with_f64_data(&vec![0.0; SCRATCH_LEN]);
    session.commit().unwrap();
    drop(session); // release the editor's exclusive lock before the next session

    let mut session = EditSession::open(path).expect("reopen for editing");
    session.delete("scratch");
    session.commit().unwrap();
    drop(session); // release the lock before reading the file back
    let after_delete = len(path);

    // For a persisting file these regions are on disk and a later session reuses
    // them; for a non-persisting file the list is empty here.
    let free = File::open(path).unwrap().persisted_free_space();
    let total_free: u64 = free.iter().map(|&(_, len)| len).sum();
    println!(
        "  after delete: {after_delete} bytes, {} persisted free region(s) ({total_free} bytes)",
        free.len()
    );

    // Add a same-sized dataset in a fresh session and measure the growth.
    let mut session = EditSession::open(path).expect("reopen for editing");
    session
        .create_dataset("scratch2")
        .with_f64_data(&vec![7.0; SCRATCH_LEN]);
    session.commit().unwrap();
    drop(session); // release the lock before reading the file back
    let after_readd = len(path);
    let growth = after_readd - after_delete;
    println!(
        "  after re-adding {SCRATCH_BYTES} bytes of new data: {after_readd} bytes (+{growth})"
    );
    growth
}

fn len(path: &Path) -> u64 {
    std::fs::metadata(path).unwrap().len()
}
