//! Streaming in-place appends with `AppendWriter` — open the file once and
//! append many times at amortized `O(1)` index cost.
//!
//! `AppendWriter` is the throughput-oriented append path: it keeps the file open
//! across appends and grows the Extensible-Array index in place, so it never
//! re-reads the file or rebuilds the index the way `EditSession::append_dataset`
//! does. Every append is crash-atomic (the dataspace dimension is published
//! last), and the result reads back in the reference C library and h5py.
//!
//! An unfiltered dataset (this example) accepts **any-length** appends; a
//! filtered dataset must be appended in whole chunks (see the guide).
//!
//! Run with:
//!
//! ```bash
//! cargo run --example append_writer
//! ```

use hdf5_pure::{AppendWriter, File, FileBuilder};

fn main() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("stream.h5");

    // ---- Create an unlimited, chunked dataset ---------------------------
    // One unlimited dimension plus `with_chunks` makes this an
    // Extensible-Array-indexed dataset (this crate allocates the index eagerly,
    // so it can be grown from the very first append).
    let initial: Vec<i32> = (0..6).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("samples")
        .with_i32_data(&initial)
        .with_shape(&[initial.len() as u64])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[4]);
    builder.write(&path).expect("write initial file");

    // ---- Stream appends through a single open writer --------------------
    // The dataset starts at length 6 with a chunk length of 4, so its trailing
    // chunk is partial (2 of 4). Each append below is an arbitrary length: the
    // partial chunk is rewritten, its index element (a single chunk address) is
    // repointed with one atomic write, then any further whole chunks are
    // inserted. The writer holds an exclusive lock for its lifetime.
    {
        let mut writer = AppendWriter::open(&path).expect("open for appending");
        writer.append_i32("samples", &[6, 7, 8]).expect("append 1"); // -> 9
        writer.append_i32("samples", &[9, 10]).expect("append 2"); // -> 11
        writer
            .append_i32("samples", &(11..20).collect::<Vec<_>>())
            .expect("append 3"); // -> 20
        writer.close().expect("flush and release the lock");
    } // the lock is released here; drop() would release it too

    // ---- Verify ---------------------------------------------------------
    let file = File::open(&path).expect("reopen");
    let all = file.dataset("samples").unwrap().read_i32().unwrap();
    println!("dataset now holds {} samples: {all:?}", all.len());
    assert_eq!(all, (0..20).collect::<Vec<_>>());
    println!("streaming in-place append verified");
}
