//! Measuring what one piece of work allocates, from inside a test binary.
//!
//! Included with `#[path = "common/allocation.rs"] mod allocation;` rather than
//! through `common/mod.rs`, deliberately: that module also declares the
//! reference-C-library helpers, and a binary that names it links a static
//! libhdf5 it has no use for. These helpers are pure Rust and pull in
//! `heapscope` alone.
//!
//! # What the including binary has to declare
//!
//! ```ignore
//! #[global_allocator]
//! static ALLOC: heapscope::Alloc = heapscope::Alloc::system();
//! ```
//!
//! A `#[global_allocator]` is a property of the binary, so it cannot live here.
//! Leaving it out is not a silent miss: [`measure`] refuses to start a profiler
//! that would record nothing, naming the line above.
#![allow(dead_code)]

use std::sync::OnceLock;

/// What one measured piece of work allocated.
///
/// Every figure is that work's own: allocations made on another thread, or on
/// this one outside the measured call, are not in it. That is what lets several
/// measured tests share a process — which they must, since the profiler is
/// process-wide and `cargo test` runs a binary's tests concurrently.
#[derive(Clone, Copy, Debug)]
pub struct Measured {
    /// Allocations made, counting a reallocation as one.
    pub blocks: u64,
    /// Bytes ever allocated, freed or not.
    pub bytes: u64,
    /// The most this work held at once, in bytes.
    pub peak_bytes: u64,
    /// Bytes it allocated that were still live when it returned.
    pub live_bytes: u64,
}

impl std::fmt::Display for Measured {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} allocations, {} bytes allocated, {} bytes at peak, {} still live",
            self.blocks, self.bytes, self.peak_bytes, self.live_bytes
        )
    }
}

/// The process's one profiler, started on first use and kept for the run.
static PROFILER: OnceLock<()> = OnceLock::new();

fn start_profiling() {
    PROFILER.get_or_init(|| {
        let profiler = heapscope::Profiler::builder()
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
        // The profiler must outlive every test in this binary, and there is
        // nowhere to put it: dropping it ends the run, and a `static` would need
        // it to be `Sync`. Nothing is lost by forgetting it — the builder above
        // asks for no output, so its drop had nothing left to do.
        std::mem::forget(profiler);
    });
}

/// Runs `work` inside a named region and returns what that region alone recorded.
///
/// `name` must be unique within the binary: heapscope interns region names, so
/// two measurements sharing one would be added together.
#[track_caller]
pub fn measure<R>(name: &'static str, work: impl FnOnce() -> R) -> (R, Measured) {
    start_profiling();

    let result = {
        let _region = heapscope::region(name);
        work()
    };

    let stats = heapscope::Snapshot::capture()
        .regions
        .into_iter()
        .find(|r| r.name.as_deref() == Some(name))
        .unwrap_or_else(|| {
            panic!("no region named {name:?} in the profile: did `work` allocate nothing at all?")
        });

    let measured = Measured {
        blocks: stats.counts.total_blocks,
        bytes: stats.counts.total_bytes,
        peak_bytes: stats.counts.max_bytes,
        live_bytes: stats.counts.curr_bytes,
    };
    eprintln!("{name}: {measured}");
    (result, measured)
}
