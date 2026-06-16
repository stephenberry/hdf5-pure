//! File-space strategy and persistent free-space tracking.
//!
//! Mirroring `H5Pset_file_space_strategy` / `H5Pset_file_space_page_size`, a
//! written file can record how it manages free space. With `persist = true`,
//! the regions an `EditSession` frees are recorded in on-disk free-space-manager
//! blocks so that later sessions (this crate's and the reference C library's)
//! recover and reuse them rather than only ever growing the file.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example file_space
//! ```

use hdf5_pure::{EditSession, File, FileBuilder, FileSpaceStrategy};

fn main() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("managed.h5");

    // Record a paged file-space strategy and persist freed space across reopen.
    let mut builder = FileBuilder::new();
    builder.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    builder
        .with_file_space_strategy(FileSpaceStrategy::Page, true, 1) // strategy, persist, threshold
        .with_file_space_page_size(4096);
    builder.write(&path).expect("write file");

    // The strategy is stored in a superblock-extension message and survives a
    // reopen (the reference C library observes it too).
    let strategy = File::open(&path).unwrap().file_space_strategy();
    println!("recorded strategy: {strategy:?}");
    assert_eq!(strategy, Some(FileSpaceStrategy::Page));

    // Create then delete a dataset. Because the file persists free space, the
    // freed region is recorded on disk rather than discarded.
    let mut session = EditSession::open(&path).expect("open for editing");
    session
        .create_dataset("scratch")
        .with_f64_data(&vec![0.0; 4096]);
    session.commit().unwrap();

    let mut session = EditSession::open(&path).expect("reopen for editing");
    session.delete("scratch");
    session.commit().unwrap();

    // Later sessions seed their free list from these regions, so add/delete
    // churn reuses space instead of only growing the file.
    let file = File::open(&path).unwrap();
    let free = file.persisted_free_space();
    let total_free: u64 = free.iter().map(|&(_, len)| len).sum();
    println!("persisted free regions: {}", free.len());
    println!("total persisted free bytes: {total_free}");

    // The kept data is unaffected by the free-space bookkeeping.
    assert_eq!(
        file.dataset("keep").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    println!("\nfile-space strategy persisted and free space tracked");
}
