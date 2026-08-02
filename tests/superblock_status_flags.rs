//! The superblock's status-flags byte gates every open (issue #245).
//!
//! The byte is durable: a SWMR writer raises it and clears it on a clean close,
//! so a writer that exits without one leaves it set. These tests pin which opens
//! that byte turns away — a plain read, a streaming read, an editor, a second
//! SWMR writer — which follow it — a SWMR reader — and that
//! `File::clear_swmr_flag` restores all of them.

use hdf5_pure::{Error, File, FileBuilder, MemoryStrategy};
use tempfile::tempdir;

/// An appendable file: rank-1, unlimited, Extensible-Array indexed, unfiltered —
/// what the SWMR writer accepts.
fn build_swmr(path: &std::path::Path) {
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&[0i32, 1, 2, 3])
        .with_shape(&[4])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[4]);
    b.write(path).unwrap();
}

/// The status-flags byte as it stands on disk, read as bytes because the opens
/// under test refuse a flagged file.
fn flags(path: &std::path::Path) -> u8 {
    let bytes = std::fs::read(path).unwrap();
    let sig = b"\x89HDF\r\n\x1a\n";
    let off = bytes.windows(sig.len()).position(|w| w == sig).unwrap();
    bytes[off + 11]
}

/// Leave the file flagged exactly as a crashed writer would: leak the writer so
/// neither `close` nor `Drop` clears the byte.
fn flag_as_if_crashed(path: &std::path::Path) {
    std::mem::forget(File::open_swmr_writer(path).unwrap());
    assert_eq!(flags(path), 0x05, "the leaked writer left the file flagged");
}

#[track_caller]
fn assert_marked_in_use(err: Error, what: &str) {
    let msg = err.to_string();
    assert!(
        matches!(err, Error::FileMarkedInUse(_)),
        "{what} must report the status flag, got {err:?}"
    );
    assert!(
        msg.contains("clear_swmr_flag"),
        "{what} must name the recovery, got: {msg}"
    );
}

/// The issue's repro: a file left flagged by a crashed SWMR writer is refused by
/// every open that would read a snapshot of it or write to it, where all of them
/// used to succeed — `open_rw` going on to edit the file in place under the
/// writer that (as far as the file records) still holds it.
#[test]
fn a_flagged_file_is_refused_by_the_reading_and_writing_opens() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("flagged.h5");
    build_swmr(&path);
    flag_as_if_crashed(&path);

    assert_marked_in_use(File::open(&path).unwrap_err(), "File::open");
    assert_marked_in_use(
        File::open_streaming(&path).unwrap_err(),
        "File::open_streaming",
    );
    assert_marked_in_use(File::open_rw(&path).unwrap_err(), "File::open_rw");
    assert_marked_in_use(
        File::open_swmr_writer(&path).unwrap_err(),
        "a second File::open_swmr_writer",
    );
}

/// The bounded editor reaches the same refusal as the mirrored one: both open the
/// same engine, and a strategy that picks a backing must not pick a rule.
#[test]
fn both_edit_backings_refuse_a_flagged_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("flagged.h5");
    build_swmr(&path);
    flag_as_if_crashed(&path);

    for strategy in [MemoryStrategy::Bounded, MemoryStrategy::Mirrored] {
        let props = hdf5_pure::FileAccessProperties::new().with_memory_strategy(strategy);
        assert_marked_in_use(
            File::open_rw_with_options(&path, props).unwrap_err(),
            &format!("open_rw under {strategy:?}"),
        );
    }
}

/// A SWMR reader is the one open that follows the flag rather than being turned
/// away by it — that pairing is the whole point of the flag — and it reads the
/// data a live writer has already appended.
#[test]
fn a_swmr_reader_follows_a_flagged_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("live.h5");
    build_swmr(&path);

    let writer = File::open_swmr_writer(&path).unwrap();
    writer
        .dataset("d")
        .unwrap()
        .append(&[4i32, 5, 6, 7])
        .unwrap();

    let reader = File::open_swmr(&path).expect("a SWMR reader attaches to a flagged file");
    assert_eq!(
        reader.dataset("d").unwrap().read_i32().unwrap(),
        (0..8).collect::<Vec<_>>()
    );
    drop(reader);
    writer.close().unwrap();
}

/// `clear_swmr_flag` is the documented recovery, so every open it was blocking
/// has to work afterwards — including the SWMR writer, whose own flag it cleared.
#[test]
fn clearing_the_flag_restores_every_open() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("recovered.h5");
    build_swmr(&path);
    flag_as_if_crashed(&path);

    File::clear_swmr_flag(&path).unwrap();
    assert_eq!(flags(&path), 0x00);

    assert_eq!(
        File::open(&path)
            .unwrap()
            .dataset("d")
            .unwrap()
            .read_i32()
            .unwrap(),
        (0..4).collect::<Vec<_>>()
    );
    File::open_streaming(&path).unwrap();
    File::open_swmr(&path).unwrap();
    drop(File::open_rw(&path).unwrap());
    File::open_swmr_writer(&path).unwrap().close().unwrap();
}

/// `File::from_bytes` does not check the flag: the caller already holds a
/// snapshot, there is no live file to coordinate over, and it is the way out for
/// a flagged file on a read-only mount, where the recovery `clear_swmr_flag`
/// needs write access it cannot get. The C library *does* check under its
/// in-memory driver, so this is a deliberate divergence, not parity.
#[test]
fn from_bytes_reads_a_flagged_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("flagged.h5");
    build_swmr(&path);
    flag_as_if_crashed(&path);

    let file = File::from_bytes(std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(file.superblock().consistency_flags, 0x05);
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        (0..4).collect::<Vec<_>>()
    );
}
