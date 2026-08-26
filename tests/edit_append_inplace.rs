//! Pure-Rust tests for [`Dataset::append`] (issue #146): immediate,
//! crash-atomic O(1) in-place appends driven from the same long-lived
//! `File::open_rw` that stages group/dataset/attribute/delete edits — interleaved
//! with those staged edits, without reopening the file. Crash-consistency phasing
//! lives in the in-crate `edit::tests` module; C-library interop (including
//! hard-link aliasing and a combined mixed-edit file) lives in
//! `edit_crosscheck.rs`.

use hdf5_pure::{AttrValue, Error, File, FileBuilder, FileSpaceStrategy, FormatError};
use tempfile::tempdir;

/// Build a rank-1, unlimited i32 dataset at `name` with the given chunk length and
/// optional deflate, seeded with `0..n`.
fn build(path: &std::path::Path, name: &str, n: i32, chunk: u64, deflate: bool) {
    let data: Vec<i32> = (0..n).collect();
    let mut b = FileBuilder::new();
    let ds = b
        .create_dataset(name)
        .with_i32_data(&data)
        .with_shape(&[n as u64])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[chunk]);
    if deflate {
        ds.with_deflate(6);
    }
    b.write(path).unwrap();
}

fn read_i32(path: &std::path::Path, name: &str) -> Vec<i32> {
    let f = File::open(path).unwrap();
    f.dataset(name).unwrap().read_i32().unwrap()
}

// ---- functional -------------------------------------------------------------

#[test]
fn unfiltered_any_length_across_calls() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    build(&p, "d", 6, 4, false); // partial tail (6 % 4 != 0)

    {
        let s = File::open_rw(&p).unwrap();
        s.dataset("d").unwrap().append(&[6, 7]).unwrap(); // grows the partial chunk
        s.dataset("d").unwrap().append(&[8, 9, 10, 11, 12]).unwrap(); // any length
        s.dataset("d").unwrap().append(&[13i32]).unwrap(); // generic entry point
    }

    assert_eq!(read_i32(&p, "d"), (0..14).collect::<Vec<_>>());
}

#[test]
fn unfiltered_raw_append() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    build(&p, "d", 4, 4, false);

    {
        let s = File::open_rw(&p).unwrap();
        let bytes: Vec<u8> = [4i32, 5, 6].iter().flat_map(|v| v.to_le_bytes()).collect();
        s.dataset("d").unwrap().append_raw(&bytes).unwrap();
    }

    assert_eq!(read_i32(&p, "d"), (0..7).collect::<Vec<_>>());
}

#[test]
fn filtered_whole_chunk() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    build(&p, "d", 8, 4, true); // 2 full chunks, deflate

    {
        let s = File::open_rw(&p).unwrap();
        s.dataset("d").unwrap().append(&[8, 9, 10, 11]).unwrap(); // one whole chunk
    }

    assert_eq!(read_i32(&p, "d"), (0..12).collect::<Vec<_>>());
}

#[test]
fn filtered_onto_a_partial_trailing_chunk_refused() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    build(&p, "d", 8, 4, true);

    let s = File::open_rw(&p).unwrap();
    // Not a whole chunk (2 of 4), but the dataset starts on a boundary, so this
    // only inserts a new partial chunk no reader can see yet: allowed.
    s.dataset("d").unwrap().append(&[8, 9]).unwrap();
    // Now the trailing chunk is partial and visible, and growing it would
    // repoint a filtered index element, which is not power-loss atomic.
    let err = s.dataset("d").unwrap().append(&[10, 11]).unwrap_err();
    assert!(matches!(err, Error::AppendInPlaceUnsupported(_)));
    drop(s);
    assert_eq!(read_i32(&p, "d"), (0..10).collect::<Vec<_>>());
}

// ---- interleave with staged tree edits --------------------------------------

#[test]
fn interleave_append_stage_commit_append() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    // Two datasets: "d" grows in place, "doomed" is deleted.
    {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&(0..6).collect::<Vec<_>>())
            .with_shape(&[6])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[4]);
        b.create_dataset("doomed").with_i32_data(&[1, 2, 3]);
        b.write(&p).unwrap();
    }

    {
        let s = File::open_rw(&p).unwrap();
        // Immediate append.
        s.dataset("d").unwrap().append(&[6, 7, 8]).unwrap(); // d -> 0..9
        assert!(!s.has_staged_edits());

        // Stage a batch of tree edits.
        s.root()
            .create_group_with("run2", |g| {
                g.set_attr("count", AttrValue::I64(7));
            })
            .unwrap();
        s.root()
            .create_dataset("created", |b| {
                b.with_i32_data(&(0..4).collect::<Vec<_>>())
                    .with_shape(&[4])
                    .with_maxshape(&[u64::MAX])
                    .with_chunks(&[4]);
            })
            .unwrap();
        s.root().delete("doomed").unwrap();
        assert!(s.has_staged_edits());
        s.commit().unwrap();
        assert!(!s.has_staged_edits());

        // Append again after the header-relocating + deleting commit: the cache was
        // invalidated at commit entry, so this re-locates against the fresh mirror.
        s.dataset("d").unwrap().append(&[9, 10]).unwrap(); // d -> 0..11
        // And append to the dataset created by the staged commit.
        s.dataset("created").unwrap().append(&[4, 5, 6, 7]).unwrap(); // created -> 0..8
    }

    assert_eq!(read_i32(&p, "d"), (0..11).collect::<Vec<_>>());
    assert_eq!(read_i32(&p, "created"), (0..8).collect::<Vec<_>>());

    let f = File::open(&p).unwrap();
    let attrs = f.group("run2").unwrap().attrs().unwrap();
    assert_eq!(attrs.get("count"), Some(&AttrValue::I64(7)));
    assert!(f.dataset("doomed").is_err(), "doomed should be gone");
}

// ---- pending-conflict guard -------------------------------------------------

#[test]
fn guard_refuses_pending_delete() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    build(&p, "d", 4, 4, false);

    let s = File::open_rw(&p).unwrap();
    s.root().delete("d").unwrap();
    let err = s.dataset("d").unwrap().append(&[4]).unwrap_err();
    assert!(matches!(err, Error::AppendInPlaceUnsupported(_)));
}

#[test]
fn guard_refuses_append_after_staged_write() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    build(&p, "d", 4, 4, false);

    let s = File::open_rw(&p).unwrap();
    // First append is fine (nothing staged yet).
    s.dataset("d").unwrap().append(&[4, 5]).unwrap();
    // Now stage an overwrite of the same dataset; a second in-place append must be
    // refused (commit would relocate the header the append planned against).
    s.dataset("d")
        .unwrap()
        .write_staged(|b| {
            b.with_i32_data(&(0..6).collect::<Vec<_>>())
                .with_shape(&[6]);
        })
        .unwrap();
    let err = s.dataset("d").unwrap().append(&[6]).unwrap_err();
    assert!(matches!(err, Error::AppendInPlaceUnsupported(_)));
}

#[test]
fn guard_refuses_ancestor_delete() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("grp/d")
            .with_i32_data(&(0..4).collect::<Vec<_>>())
            .with_shape(&[4])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[4]);
        b.write(&p).unwrap();
    }

    let s = File::open_rw(&p).unwrap();
    s.root().delete("grp").unwrap(); // deletes the ancestor group of grp/d
    // The staged delete takes the subtree out of reach: the dataset no longer
    // resolves, so there is no handle to append through. Either way the append
    // cannot land — which is the guarantee this pins.
    let err = match s.dataset("grp/d") {
        Ok(mut ds) => ds.append(&[4]).unwrap_err(),
        Err(e) => e,
    };
    assert!(
        matches!(
            err,
            Error::AppendInPlaceUnsupported(_) | Error::Format(FormatError::PathNotFound(_))
        ),
        "expected the append to be unreachable, got {err:?}"
    );
}

// ---- fast-path refusals + fallback ------------------------------------------

#[test]
fn fallback_non_extensible_array_refused() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    // A plain contiguous dataset (no maxshape / chunks) is not Extensible-Array
    // indexed.
    {
        let mut b = FileBuilder::new();
        b.create_dataset("d").with_i32_data(&[1, 2, 3, 4]);
        b.write(&p).unwrap();
    }

    let s = File::open_rw(&p).unwrap();
    let err = s.dataset("d").unwrap().append(&[5]).unwrap_err();
    assert!(matches!(err, Error::AppendInPlaceUnsupported(_)));
}

#[test]
fn userblock_refuses_inplace_but_staged_append_dataset_works() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("ub.h5");
    {
        let mut b = FileBuilder::new();
        b.with_userblock(512);
        b.create_dataset("d")
            .with_i32_data(&(0..8).collect::<Vec<_>>())
            .with_shape(&[8])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[4]);
        b.write(&p).unwrap();
    }

    {
        let s = File::open_rw(&p).unwrap();
        // Fast path refuses a userblock (base address != 0), with the distinct,
        // catchable error.
        let err = s.dataset("d").unwrap().append(&[8, 9, 10, 11]).unwrap_err();
        assert!(matches!(err, Error::AppendInPlaceUnsupported(_)));
        // The staged fallback handles it: rebuild the index, repoint last.
        s.dataset("d")
            .unwrap()
            .append_staged(|b| {
                b.append_i32(&[8, 9, 10, 11]);
            })
            .unwrap();
        s.commit().unwrap();
    }

    assert_eq!(read_i32(&p, "d"), (0..12).collect::<Vec<_>>());
}

#[test]
fn refusal_leaves_session_usable() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    build(&p, "d", 4, 4, false);

    {
        let s = File::open_rw(&p).unwrap();
        // A datatype mismatch is refused...
        let err = s.dataset("d").unwrap().append(&[1.0]).unwrap_err();
        assert!(matches!(err, Error::AppendInPlaceUnsupported(_)));
        // ...and the session keeps working for a correct append afterward.
        s.dataset("d").unwrap().append(&[4, 5, 6]).unwrap();
    }

    assert_eq!(read_i32(&p, "d"), (0..7).collect::<Vec<_>>());
}

/// A file that persists its free space accepts an immediate in-place append as
/// well as a staged one (issue #198). The in-place path leaves the on-disk
/// free-space managers describing a file shorter than it has become, which is what
/// `File::close` rewrites; both paths must leave every row readable.
#[test]
fn persisting_file_takes_both_inplace_and_staged_appends() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("persist.h5");
    {
        let mut b = FileBuilder::new();
        b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
        b.create_dataset("d")
            .with_i32_data(&(0..8).collect::<Vec<_>>())
            .with_shape(&[8])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[4]);
        b.write(&p).unwrap();
    }

    {
        let s = File::open_rw(&p).unwrap();
        s.dataset("d").unwrap().append(&[8, 9, 10, 11]).unwrap();
        s.dataset("d")
            .unwrap()
            .append_staged(|b| {
                b.append_i32(&[12, 13, 14, 15]);
            })
            .unwrap();
        s.commit().unwrap();
        s.close().unwrap();
    }

    assert_eq!(read_i32(&p, "d"), (0..16).collect::<Vec<_>>());
    let before = File::open(&p).unwrap().persisted_free_space();

    // A second session appends in place again and closes without staging
    // anything, so the manager re-homing is `close`'s doing rather than a
    // commit's. It re-homes the extension and manager blocks, freeing the ones
    // it supersedes, so the recorded free space must differ afterwards — which
    // is what distinguishes a real rewrite from a skipped one.
    {
        let s = File::open_rw(&p).unwrap();
        s.dataset("d").unwrap().append(&[16, 17]).unwrap();
        s.close().unwrap();
    }
    assert_eq!(read_i32(&p, "d"), (0..18).collect::<Vec<_>>());
    assert_ne!(
        File::open(&p).unwrap().persisted_free_space(),
        before,
        "closing the session did not rewrite the free-space managers"
    );
    // The re-homed managers must still parse: every region they describe has to
    // lie inside the file, which is what a stale (mid-file) manager set would
    // not survive.
    let reopened = File::open(&p).unwrap();
    let size = reopened.file_size();
    for (addr, len) in reopened.persisted_free_space() {
        assert!(
            addr + len <= size,
            "free region {addr}+{len} runs past the {size}-byte file"
        );
    }
}

#[test]
fn many_small_appends_one_session() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    build(&p, "d", 0, 8, false); // start empty (but index allocated by the writer)

    {
        let s = File::open_rw(&p).unwrap();
        for i in 0..100i32 {
            s.dataset("d").unwrap().append(&[i]).unwrap();
        }
    }

    assert_eq!(read_i32(&p, "d"), (0..100).collect::<Vec<_>>());
}
