//! What the read paths are allowed to allocate, as rules rather than as numbers
//! (issue #228).
//!
//! Every bound here is a statement about how the work *scales* — a windowed read
//! allocates on the order of its window, a chunked read allocates a fixed number
//! of blocks per chunk — so it holds on every platform and survives a toolchain
//! that changes a `Vec`'s growth by one step. The exact figures are pinned
//! separately, on one platform, in `tests/allocation_baseline.rs`.
//!
//! Measurement is per region and per thread (`tests/common/allocation.rs`), so
//! these tests share a process without serializing against each other: what a
//! region records is what the calling thread allocated inside it, and nothing
//! else.

use hdf5_pure::{File, FileBuilder};

#[global_allocator]
static ALLOC: heapscope::Alloc = heapscope::Alloc::system();

#[path = "common/allocation.rs"]
mod allocation;

use allocation::measure;

#[test]
fn windowed_row_read_allocates_on_the_order_of_the_window() {
    // 4 MiB of f64 rows with inner-split storage chunks: [2048, 32, 8] in
    // [64, 16, 4] chunks — a 2x2 inner chunk grid per 64-row band, deflated so
    // chunks really decode (32 KiB decompressed each).
    const N0: usize = 2048;
    const ROW_ELEMS: usize = 32 * 8;
    const DATASET_BYTES: usize = N0 * ROW_ELEMS * 8;

    let data: Vec<f64> = (0..N0 * ROW_ELEMS).map(|i| i as f64).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("t")
        .with_f64_data(&data)
        .with_shape(&[N0 as u64, 32, 8])
        .with_chunks(&[64, 16, 4])
        .with_deflate(3);
    builder.write(&path).unwrap();
    drop(data);

    // The streaming backend reads from disk on demand; the buffered backend
    // would hold the whole file in memory by design and drown the signal.
    let file = File::open_streaming(&path).unwrap();
    let ds = file.dataset("t").unwrap();

    // A 64-row window straddling a chunk-band boundary mid-file, so it decodes
    // chunks from two leading bands.
    let (window, measured) = measure("windowed_row_read", || ds.read_f64_rows(992, 64).unwrap());

    // Window + raw/typed conversion + a few decompressed 32 KiB chunks lands
    // well under 1 MiB; a whole-read fallback peaks above the full dataset
    // (the assembled whole read plus cached chunks and the sliced window).
    assert!(
        measured.peak_bytes < (DATASET_BYTES / 4) as u64,
        "peak allocation during the windowed read must be bounded by the window, \
         not the {DATASET_BYTES}-byte dataset; measured {measured}"
    );

    // The window must still be the right bytes.
    let expected: Vec<f64> = (992 * ROW_ELEMS..(992 + 64) * ROW_ELEMS)
        .map(|i| i as f64)
        .collect();
    assert_eq!(window, expected);
}

#[test]
fn windowed_vlen_string_read_allocates_on_the_order_of_the_window() {
    // ~4 MiB of variable-length string payload: 32k rows of 128-byte strings,
    // plus ~512 KiB of heap references in the dataset itself. The writer packs
    // the strings into one giant heap collection — the degenerate case for a
    // windowed read, whose directory alone rivals the window if parsed whole.
    const N0: usize = 32 * 1024;
    const STR_LEN: usize = 128;
    const PAYLOAD_BYTES: usize = N0 * STR_LEN;

    let strings: Vec<String> = (0..N0)
        .map(|i| format!("{i:0>width$}", width = STR_LEN))
        .collect();
    let refs: Vec<&str> = strings.iter().map(String::as_str).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("labels").with_vlen_strings(&refs);
    builder.write(&path).unwrap();
    drop(refs);
    drop(strings);

    let file = File::open_streaming(&path).unwrap();
    let ds = file.dataset("labels").unwrap();

    // 256 mid-file rows — ~32 KiB of text plus ~4 KiB of references, resolved
    // against a window-filtered slice of the collection's directory.
    let (window, measured) = measure("windowed_vlen_read", || {
        ds.read_string_rows(15_000, 256).unwrap()
    });

    // Window references + text + touched heap-collection directories land far
    // under 1 MiB; resolving every reference first peaks above the full payload
    // (all references plus a Vec<String> of every row). This also catches a
    // collection parse that starts buffering whole collections instead of
    // walking their metadata.
    assert!(
        measured.peak_bytes < (PAYLOAD_BYTES / 4) as u64,
        "peak allocation during the windowed vlen read must be bounded by the window, \
         not the {PAYLOAD_BYTES}-byte payload; measured {measured}"
    );

    // The other half of the same story, which the peak cannot see. A global heap
    // collection chains each object's position in the previous object's header,
    // so reaching the window's objects means passing every object before them —
    // the traversal is the format's, not this crate's, and no bound here can
    // remove it. What it must not cost is an *allocation* apiece: reading this
    // window once took 65,536 of them, two per object of a collection it kept 256
    // entries from (issue #228). One per eight objects is far above what a
    // windowed walk needs and far below what a per-object one takes, which is the
    // gap this bound is placed in.
    assert!(
        measured.blocks < (N0 / 8) as u64,
        "a windowed vlen read must not allocate per object of the collection it \
         walks: measured {measured} over a {N0}-object collection"
    );

    // The window must still be the right strings.
    assert_eq!(window.len(), 256);
    let expected: Vec<String> = (15_000..15_000 + 256)
        .map(|i| format!("{i:0>width$}", width = STR_LEN))
        .collect();
    assert_eq!(window, expected);
}

/// A dataset whose chunks lie end to end for its whole length is what the
/// coalescing reader (`src/chunk_span.rs`) exists for, and the worst case for
/// its memory bound: a coalescer with no budget would merge all 2,048 chunks
/// into a single 8 MiB span and hold the dataset a second time to serve them.
/// The 256 KiB budget is what keeps the span a constant beside the read rather
/// than a second copy of it.
///
/// The read here is [`Dataset::read_raw`] rather than a typed one on purpose:
/// a typed read allocates its own copy of the dataset, and a term that large
/// hides the one being measured.
///
/// The three bounds are one story told three ways — peak, bytes, blocks — and
/// each catches a defect the others do not: a second buffer of dataset size, a
/// throwaway copy of every chunk, and a per-chunk `Vec` in the scatter loop.
#[test]
fn whole_read_of_adjacent_chunks_costs_a_constant_per_chunk() {
    // 8 MiB of f64 in 4 KiB chunks: 2,048 of them, unfiltered so a chunk's
    // stored bytes are its data and a span holds no less than the file does.
    const N0: usize = 1024 * 1024;
    const DATASET_BYTES: usize = N0 * 8;
    const CHUNK_ELEMS: usize = 512;
    const CHUNKS: u64 = (N0 / CHUNK_ELEMS) as u64;

    let data: Vec<f64> = (0..N0).map(|i| i as f64).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("t")
        .with_f64_data(&data)
        .with_shape(&[N0 as u64])
        .with_chunks(&[CHUNK_ELEMS as u64]);
    builder.write(&path).unwrap();
    drop(data);

    let file = File::open_streaming(&path).unwrap();
    let ds = file.dataset("t").unwrap();

    let (all, measured) = measure("whole_read", || ds.read_raw().unwrap());

    // The read's own output is the dataset, and the parsed chunk index is most
    // of what is left; an unbudgeted span would add the dataset again.
    assert!(
        measured.peak_bytes < (DATASET_BYTES + DATASET_BYTES / 2) as u64,
        "peak allocation during the whole read must not grow with the run of \
         adjacent chunks; measured {measured} against a {DATASET_BYTES}-byte dataset"
    );

    // Bytes *ever* allocated, which the peak cannot see: a copy of each chunk
    // that is made and dropped costs the dataset over again without moving the
    // high-water mark at all. The read's own output is one dataset; the chunk
    // cache is entitled to its configured budget on top, and nothing else here
    // scales with the data.
    assert!(
        measured.bytes < (DATASET_BYTES + DATASET_BYTES / 4) as u64,
        "a whole read must allocate the dataset about once — its output — not a \
         second time in throwaway per-chunk copies; measured {measured} against a \
         {DATASET_BYTES}-byte dataset"
    );

    // Per-chunk *count*, which neither byte figure can see: a `Vec` of the
    // chunk's coordinates costs 32 bytes and is invisible next to 8 MiB, but it
    // is an allocator round trip per chunk of every dataset this crate reads.
    const BLOCKS_PER_CHUNK: u64 = 3;
    assert!(
        measured.blocks <= BLOCKS_PER_CHUNK * CHUNKS + 256,
        "a whole read must cost a small constant number of allocations per chunk, \
         not a growing one: measured {measured} over {CHUNKS} chunks, above the \
         {BLOCKS_PER_CHUNK} per chunk this bound allows"
    );

    assert_eq!(all.len(), DATASET_BYTES);
    assert_eq!(
        &all[DATASET_BYTES - 8..],
        &((N0 - 1) as f64).to_le_bytes()[..]
    );
}
