// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit-pointer targets; skip them on 32-bit so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(not(target_pointer_width = "32"))]
//! Reference-C-library interop for the bounded read-write backend (issue #147):
//! files grown through `File::open_rw_bounded` read back byte-correct in the
//! reference C library, both when the C library wrote the original file and
//! when this crate did (unfiltered any-length and filtered whole-chunk).

use hdf5::Extent;
use hdf5::file::LibraryVersion;
use hdf5::plist::file_create::FileSpaceStrategy as CStrategy;
use hdf5_pure::{File, FileBuilder, FileSpaceStrategy};
use std::sync::{Mutex, MutexGuard};
use tempfile::tempdir;

// The reference free-space query, resolved at link time from the statically
// linked libhdf5: a positive result proves the C library loaded and parsed the
// on-disk free-space managers this crate's bounded finalize wrote.
unsafe extern "C" {
    fn H5Fget_freespace(file_id: i64) -> i64;
}

// libhdf5 is not built thread-safe here. `hdf5-metno` serializes its own calls
// through a private global lock, but the raw `H5Fget_freespace` FFI above bypasses
// it, so that raw call can race a `hdf5-metno` operation running on another test
// thread and crash the C library. To prevent it, EVERY test that touches the C
// library takes this guard as its first line and holds it for the whole body, so
// no two tests ever run C-library code at once (matching `file_space_crosscheck`).
// A new test that calls into `hdf5::…` without the guard reintroduces the race —
// it surfaces as an intermittent SIGSEGV, typically only on CI. Poisoning is
// ignored so one test's panic does not cascade.
static C_LIB: Mutex<()> = Mutex::new(());

fn c_lib_guard() -> MutexGuard<'static, ()> {
    C_LIB.lock().unwrap_or_else(|e| e.into_inner())
}

/// Create a rank-1 unlimited (Extensible-Array indexed) i32 dataset `name` with the
/// C library under the latest format, seeded with `0..n`, chunk length `chunk`.
fn c_create_unlimited(path: &std::path::Path, name: &str, n: i32, chunk: usize) {
    let file = hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
        .create(path)
        .unwrap();
    let ds = file
        .new_dataset::<i32>()
        .chunk((chunk,))
        .shape((Extent::resizable(n as usize),))
        .create(name)
        .unwrap();
    ds.write(&(0..n).collect::<Vec<_>>()).unwrap();
    file.close().unwrap();
}

fn read_c(path: &std::path::Path, name: &str) -> Vec<i32> {
    let f = hdf5::File::open(path).unwrap();
    let v = f.dataset(name).unwrap().read_raw::<i32>().unwrap();
    f.close().unwrap();
    v
}

fn read_pure(path: &std::path::Path, name: &str) -> Vec<i32> {
    File::open(path)
        .unwrap()
        .dataset(name)
        .unwrap()
        .read_i32()
        .unwrap()
}

/// Create a rank-1 unlimited chunked i32 dataset with this crate's writer.
fn pure_create(path: &std::path::Path, n: i32, chunk: u64, deflate: bool) {
    let data: Vec<i32> = (0..n).collect();
    let mut b = FileBuilder::new();
    let ds = b
        .create_dataset("d")
        .with_i32_data(&data)
        .with_shape(&[n as u64])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[chunk]);
    if deflate {
        ds.with_deflate(6);
    }
    b.write(path).unwrap();
}

#[test]
fn bounded_append_to_c_dataset_both_read() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("c.h5");
    c_create_unlimited(&path, "d", 8, 4);

    {
        let file = File::open_rw_bounded(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append(&[8i32, 9, 10, 11, 12]).unwrap(); // any length (unfiltered)
        ds.append(&[13i32]).unwrap();
    }

    let expected: Vec<i32> = (0..14).collect();
    assert_eq!(read_pure(&path, "d"), expected);
    assert_eq!(read_c(&path, "d"), expected);
}

#[test]
fn bounded_filtered_append_reads_back_in_c() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("filtered.h5");
    pure_create(&path, 8, 4, true);

    {
        let file = File::open_rw_bounded(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append(&[8i32, 9, 10, 11]).unwrap(); // whole chunks only when filtered
        ds.append(&[12i32, 13, 14, 15]).unwrap();
    }

    let expected: Vec<i32> = (0..16).collect();
    assert_eq!(read_pure(&path, "d"), expected);
    assert_eq!(read_c(&path, "d"), expected);
}

#[test]
fn bounded_batched_large_append_reads_back_in_c() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("large.h5");
    pure_create(&path, 3, 256, false);

    // One call far past the internal 1 MiB batch budget, from an unaligned
    // start, so the C library reads back a file grown through several
    // crash-atomic batches.
    let total = 400_000i32;
    {
        let file = File::open_rw_bounded(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append(&(3..total).collect::<Vec<i32>>()).unwrap();
    }

    let got = read_c(&path, "d");
    assert_eq!(got.len(), total as usize);
    assert!(got.iter().enumerate().all(|(i, &v)| v == i as i32));
    assert_eq!(read_pure(&path, "d").len(), total as usize);
}

/// A persisting file (this crate's writer, `persist = true`, non-paged) grown
/// through the bounded backend and finalized at `close` reads back byte-correct
/// in the C library, which also loads the free-space managers the finalize wrote
/// (issue #173).
#[test]
fn bounded_persist_finalize_reads_back_in_c() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("pure_persist.h5");

    let mut b = FileBuilder::new();
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    b.create_dataset("d")
        .with_i32_data(&(0..10).collect::<Vec<i32>>())
        .with_shape(&[10])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[4]);
    b.write(&path).unwrap();

    {
        let file = File::open_rw_bounded(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append(&(10..30).collect::<Vec<i32>>()).unwrap();
        file.close().unwrap();
    }

    let expected: Vec<i32> = (0..30).collect();
    assert_eq!(read_pure(&path, "d"), expected);

    // The C library reads the data, recovers the persisting strategy, and its
    // free-space query parses the managers the finalize wrote.
    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(f.dataset("d").unwrap().read_raw::<i32>().unwrap(), expected);
    let strat = f.create_plist().unwrap().get_file_space_strategy().unwrap();
    assert!(
        matches!(
            strat,
            CStrategy::FreeSpaceManager {
                paged: false,
                persist: true,
                ..
            }
        ),
        "C recovers our persisting FSM strategy, got {strat:?}"
    );
    let free = unsafe { H5Fget_freespace(f.id()) };
    assert!(
        free >= 0,
        "C loads our free-space managers without error (got {free})"
    );
}

/// The mirror of the above: the C library creates the persisting file, the
/// bounded backend grows it and finalizes at `close`, and both libraries read
/// the full sequence back.
#[test]
fn bounded_persist_on_c_created_file_reads_back() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_persist.h5");

    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .with_fcpl(|fcpl| {
                fcpl.file_space_strategy(CStrategy::FreeSpaceManager {
                    paged: false,
                    persist: true,
                    threshold: 1,
                })
            })
            .create(&path)
            .unwrap();
        let ds = file
            .new_dataset::<i32>()
            .chunk((4,))
            .shape((Extent::resizable(8),))
            .create("d")
            .unwrap();
        ds.write(&(0..8).collect::<Vec<_>>()).unwrap();
        file.close().unwrap();
    }

    {
        let file = File::open_rw_bounded(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append(&(8..20).collect::<Vec<i32>>()).unwrap();
        file.close().unwrap();
    }

    let expected: Vec<i32> = (0..20).collect();
    assert_eq!(read_pure(&path, "d"), expected);
    assert_eq!(read_c(&path, "d"), expected);
}

/// Variable-length string reads route through the file source; on read-write
/// backends (bounded AND mirror) that must reach the real bytes through the
/// engine, not the empty borrowed view (which made every heap-backed VL read
/// fail with UnexpectedEof).
#[test]
fn vlen_strings_read_on_bounded_and_mirror_files() {
    use hdf5::types::VarLenUnicode;
    use std::str::FromStr;

    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("vlen.h5");
    let words = ["alpha", "beta", "", "δelta"];
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&path)
            .unwrap();
        let vals: Vec<VarLenUnicode> = words
            .iter()
            .map(|s| VarLenUnicode::from_str(s).unwrap())
            .collect();
        file.new_dataset::<VarLenUnicode>()
            .shape((words.len(),))
            .create("labels")
            .unwrap()
            .write(&vals)
            .unwrap();
        let ds = file
            .new_dataset::<i32>()
            .chunk((4,))
            .shape((Extent::resizable(4),))
            .create("samples")
            .unwrap();
        ds.write(&[0i32, 1, 2, 3]).unwrap();
        file.close().unwrap();
    }
    let expected: Vec<String> = words.iter().map(|s| s.to_string()).collect();

    {
        let file = File::open_rw_bounded(&path).unwrap();
        let labels = file.dataset("labels").unwrap();
        assert_eq!(labels.read_string().unwrap(), expected);
        // Interleave an append, then read the heap-backed strings again.
        let mut samples = file.dataset("samples").unwrap();
        samples.append(&[4i32, 5]).unwrap();
        assert_eq!(labels.read_string().unwrap(), expected);
    }
    {
        let file = File::open_rw(&path).unwrap();
        assert_eq!(
            file.dataset("labels").unwrap().read_string().unwrap(),
            expected
        );
    }
    assert_eq!(read_c(&path, "samples"), (0..6).collect::<Vec<_>>());
}
