//! The exact allocation figures for one write-then-read cycle, recorded rather
//! than chosen (issue #228).
//!
//! `tests/allocation_bounds.rs` states the rules — a windowed read allocates on
//! the order of its window, a chunked read costs a constant per chunk — and those
//! run everywhere. This file is the other half: the numbers themselves, committed
//! under `tests/baselines/` so a change to them arrives in a pull request as a
//! line a reviewer reads rather than as nothing at all. A 5% creep passes every
//! rule in this suite and shows up here.
//!
//! # Why this is behind a feature, one platform, and one feature set
//!
//! An exact allocation count is a property of a target, a toolchain *and* a
//! feature set, not of this crate alone: a `Vec` growth policy, an `OsString` on
//! one platform and a `String` on another, or a new `std` release all move it
//! without anything here changing. The feature set is not hypothetical —
//! `--all-features` measured 416 bytes above the default set on the same machine,
//! in the same commit. So the figures are recorded where they can be checked (see
//! the `heap-baseline` job in `.github/workflows/ci.yml`) and the default suite
//! runs the rules instead. Widening this to a second platform is a matter of
//! measuring it there, not of loosening anything.
//!
//! Hence the `not(any(...))` below: this compiles only under the crate's default
//! features, which is the set the figures were recorded under and the set a
//! dependent's build most resembles. Under `--all-features` — the pre-push gate —
//! it compiles to nothing rather than failing against numbers from another
//! configuration. A feature added to the crate and not to that list will make the
//! `--all-features` gate start running this and fail; the fix is the one-line
//! addition, and the loud failure is the point.
//!
//! # When it fails
//!
//! Read the named figures. A number that went *up* is the point of the gate: some
//! path started allocating more. A number that went *down* is usually good news
//! that still has to be recorded, because the file says what today's behaviour is
//! and a stale one gates nothing.
//!
//! Re-record with:
//!
//! ```sh
//! HEAPSCOPE_UPDATE_BASELINE=1 cargo test --features heap-baseline --test allocation_baseline
//! ```
//!
//! and commit the diff with the change that caused it.
//!
//! # What is measured
//!
//! The whole process, which is why this binary holds exactly one `#[test]`: the
//! profiler is process-wide, and `HeapStats` cannot be built from a
//! `heapscope::Region`'s figures (it is `#[non_exhaustive]`), so a baseline is a
//! reading of the run rather than of a phase inside it. The consequence worth
//! knowing is that the fixture below is part of the measurement: changing what it
//! writes changes the numbers legitimately. The upside is that the write path is
//! pinned here too, and it allocates more than the read does.
#![cfg(all(
    feature = "heap-baseline",
    not(any(
        feature = "fast-deflate",
        feature = "provenance",
        feature = "parallel",
        feature = "zfp",
        feature = "ndarray",
        feature = "serde",
        feature = "num-complex",
        feature = "matio-crosscheck",
    ))
))]

use hdf5_pure::{File, FileBuilder};

#[global_allocator]
static ALLOC: heapscope::Alloc = heapscope::Alloc::system();

/// Write an 8 MiB chunked dataset and read it back whole: 2,048 chunks through
/// the writer's chunk assembly and the reader's coalescing span planner, which is
/// where all but a constant of the allocations in either path live.
#[test]
fn writing_and_reading_a_chunked_dataset_matches_its_recorded_figures() {
    let _profiler = heapscope::Profiler::builder()
        .no_output()
        .build()
        .unwrap_or_else(|e| {
            panic!(
                "heapscope could not start: {e}\n\
                 On x86_64 this is usually the missing frame pointers that \
                 `.cargo/config.toml` sets for that target; a build that \
                 overrides RUSTFLAGS drops them."
            )
        });

    const N0: usize = 1024 * 1024;
    const CHUNK_ELEMS: u64 = 512;

    let data: Vec<f64> = (0..N0).map(|i| i as f64).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("baseline.h5");

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("t")
        .with_f64_data(&data)
        .with_shape(&[N0 as u64])
        .with_chunks(&[CHUNK_ELEMS]);
    builder.write(&path).unwrap();
    drop(data);

    let file = File::open_streaming(&path).unwrap();
    let all = file.dataset("t").unwrap().read_raw().unwrap();
    assert_eq!(all.len(), N0 * 8);
    drop(all);
    drop(file);

    heapscope::assert_baseline!("tests/baselines/chunked_write_read.txt");
}
