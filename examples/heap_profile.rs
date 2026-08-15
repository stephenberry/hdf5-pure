//! Profile what this crate allocates, and where (issue #228).
//!
//! The tests under `tests/allocation_bounds.rs` say whether a bound still holds.
//! This says *which call site* spent the bytes, which is the question you have
//! next when one of them fails, or when you are looking for something to make
//! cheaper in the first place.
//!
//! ```sh
//! cargo run --release --example heap_profile
//! ```
//!
//! It writes `target/heap-profile.html` — one self-contained page, open it in a
//! browser — and prints the heaviest call sites to the terminal. Each phase of
//! the workload is a named region, so the page can answer "what did the *write*
//! allocate" separately from "what did the read allocate", which no stack trace
//! can: the same `Vec::with_capacity` in the same helper is a different cost in
//! each.
//!
//! Release rather than debug is deliberate. A debug build's allocation *counts*
//! are the same, but its call stacks are full of the inlining that did not
//! happen, so the profile reads as a list of iterator adapters.

use hdf5_pure::{File, FileBuilder};

#[global_allocator]
static ALLOC: heapscope::Alloc = heapscope::Alloc::system();

/// 8 MiB of f64 in 4 KiB chunks — enough chunks (2,048) that a per-chunk cost
/// stands out against the constants around it.
const ELEMENTS: usize = 1024 * 1024;
const CHUNK_ELEMS: u64 = 512;
/// A vlen dataset packed into one global heap collection, which is the layout
/// that makes a windowed read walk far more than it keeps.
const LABELS: usize = 32 * 1024;
const LABEL_LEN: usize = 128;

const PROFILE_PATH: &str = "target/heap-profile.html";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let profiler = heapscope::Profiler::builder()
        .output(heapscope::Output::html(PROFILE_PATH))
        .build()
        .map_err(|e| {
            format!(
                "heapscope could not start: {e}\n\
                 On x86_64 this needs the frame pointers `.cargo/config.toml` \
                 sets for that target; a build that overrides RUSTFLAGS drops them."
            )
        })?;

    let dir = tempfile::tempdir()?;
    let numeric = dir.path().join("numeric.h5");
    let labels = dir.path().join("labels.h5");

    {
        let _region = heapscope::region("write chunked");
        let data: Vec<f64> = (0..ELEMENTS).map(|i| i as f64).collect();
        let mut builder = FileBuilder::new();
        builder
            .create_dataset("t")
            .with_f64_data(&data)
            .with_shape(&[ELEMENTS as u64])
            .with_chunks(&[CHUNK_ELEMS]);
        builder.write(&numeric)?;
    }

    {
        let _region = heapscope::region("write vlen strings");
        let strings: Vec<String> = (0..LABELS)
            .map(|i| format!("{i:0>width$}", width = LABEL_LEN))
            .collect();
        let refs: Vec<&str> = strings.iter().map(String::as_str).collect();
        let mut builder = FileBuilder::new();
        builder.create_dataset("labels").with_vlen_strings(&refs);
        builder.write(&labels)?;
    }

    let numeric_file = File::open_streaming(&numeric)?;
    let numeric_ds = numeric_file.dataset("t")?;

    let whole = {
        let _region = heapscope::region("read chunked whole");
        numeric_ds.read_raw()?
    };
    assert_eq!(
        whole.len(),
        ELEMENTS * 8,
        "whole read returned wrong length"
    );
    drop(whole);

    let window = {
        let _region = heapscope::region("read chunked window");
        numeric_ds.read_f64_rows(ELEMENTS as u64 / 2, 4096)?
    };
    assert_eq!(window.len(), 4096, "row window returned wrong length");
    assert_eq!(
        window[0],
        (ELEMENTS / 2) as f64,
        "row window started at the wrong row"
    );
    drop(window);

    let labels_file = File::open_streaming(&labels)?;
    let labels_ds = labels_file.dataset("labels")?;

    let strings = {
        let _region = heapscope::region("read vlen window");
        labels_ds.read_string_rows(LABELS as u64 / 2, 256)?
    };
    assert_eq!(strings.len(), 256, "string window returned wrong length");
    assert_eq!(
        strings[0].trim_start_matches('0'),
        (LABELS / 2).to_string(),
        "string window started at the wrong row"
    );
    drop(strings);

    profiler.print_summary(10)?;
    // Dropping the profiler is what writes the page, so the check below has to
    // come after it rather than at the end of `main`.
    drop(profiler);

    let written = std::fs::metadata(PROFILE_PATH)
        .map_err(|e| format!("{PROFILE_PATH} was not written: {e}"))?
        .len();
    assert!(written > 0, "{PROFILE_PATH} was written empty");
    println!("\nwrote {PROFILE_PATH} ({written} bytes) — open it in a browser");

    Ok(())
}
