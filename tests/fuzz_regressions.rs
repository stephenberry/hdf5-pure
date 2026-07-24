//! Regression tests for inputs found by the `cargo fuzz` targets.
//!
//! Each fixture under `tests/fixtures/fuzz/` is a minimized crash input; the
//! test drives it through the same public reader entry points the fuzz target
//! exercises and asserts the library refuses it with an error instead of
//! aborting the process.

use hdf5_pure::{Error, File, FormatError, Group};

/// A chunked fixed-length **string** dataset whose datatype declares a per-element
/// size of ~2.86 GB (`0xAAAAAAAA` bytes). The shape is only `[50]`, so the element
/// count stays tiny, but `num_elements * elem_size` is ~143 GB. Before the fix the
/// chunked reader eagerly allocated that whole logical extent (`vec![0u8; …]`) from
/// a 1.6 KB file, tripping an out-of-memory abort under the fuzzer's RSS limit
/// (issue #185). The read must now fail cleanly with an `InvalidChunkGeometry`
/// error, which the per-chunk logical-size guard raises before any allocation.
#[test]
fn oom_chunked_string_huge_elem_is_refused() {
    let bytes =
        std::fs::read("tests/fixtures/fuzz/oom_chunked_string_huge_elem.h5").expect("read fixture");

    let file = File::from_bytes(bytes).expect("file parses; only the data read is malformed");

    // Walking metadata must not allocate or panic.
    let _ = file.superblock();
    let root = file.root();

    let dataset = find_first_dataset(&file, &root).expect("fixture has a dataset");
    assert_eq!(dataset.shape().unwrap(), vec![50]);

    // The read is refused with a chunk-geometry error rather than attempting the
    // ~143 GB allocation.
    match dataset.read_string() {
        Err(Error::Format(FormatError::InvalidChunkGeometry(_))) => {}
        other => panic!("expected InvalidChunkGeometry, got {other:?}"),
    }
}

/// Return the first dataset reachable from `group`, mirroring the fuzz target's
/// shallow group/dataset walk.
fn find_first_dataset(file: &File, group: &Group) -> Option<hdf5_pure::Dataset> {
    if let Ok(names) = group.datasets() {
        for name in names {
            if let Ok(dataset) = group.dataset(&name) {
                let _ = file.dataset(&name);
                return Some(dataset);
            }
        }
    }
    if let Ok(names) = group.groups() {
        for name in names {
            if let Ok(child) = group.group(&name) {
                if let Some(found) = find_first_dataset(file, &child) {
                    return Some(found);
                }
            }
        }
    }
    None
}
