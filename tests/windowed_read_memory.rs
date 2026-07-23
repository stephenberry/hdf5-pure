//! Peak-allocation regression test for the windowed row-read API: a small row
//! window of a large inner-chunked dataset must allocate on the order of the
//! window (plus a few decompressed chunks), not the dataset. A whole-read
//! fallback peaks above the full dataset size and fails the assertion.
//!
//! The counting allocator applies to this whole test binary, so this file must
//! hold exactly one `#[test]`: a second test running in parallel would pollute
//! the counter.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use hdf5_pure::{File, FileBuilder};

/// Bytes currently allocated and the high-water mark, maintained by the
/// counting allocator below. Relaxed ordering is fine: the test is effectively
/// single-threaded during the measured section, and the assertion bound has a
/// 4x margin — exactness is not required, only the order of magnitude.
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

struct CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            if new_size >= layout.size() {
                let grow = new_size - layout.size();
                let live = LIVE.fetch_add(grow, Ordering::Relaxed) + grow;
                PEAK.fetch_max(live, Ordering::Relaxed);
            } else {
                LIVE.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        new_ptr
    }
}

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

#[test]
fn inner_chunked_window_read_is_memory_bounded() {
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

    // Measure only the windowed read: a 64-row window straddling a chunk-band
    // boundary mid-file, so it decodes chunks from two leading bands.
    let base = LIVE.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);

    let window = ds.read_f64_rows(992, 64).unwrap();

    let peak = PEAK.load(Ordering::Relaxed) - base;
    eprintln!("peak allocation during the windowed read: {peak} bytes (dataset: {DATASET_BYTES})");

    // Window + raw/typed conversion + a few decompressed 32 KiB chunks lands
    // well under 1 MiB; a whole-read fallback peaks above the full dataset
    // (the assembled whole read plus cached chunks and the sliced window).
    assert!(
        peak < DATASET_BYTES / 4,
        "peak allocation during the windowed read must be bounded by the window, \
         not the {DATASET_BYTES}-byte dataset; measured {peak} bytes"
    );

    // The window must still be the right bytes.
    let expected: Vec<f64> = (992 * ROW_ELEMS..(992 + 64) * ROW_ELEMS)
        .map(|i| i as f64)
        .collect();
    assert_eq!(window, expected);
}
