//! Pure-Rust tests for `EditSession::append_dataset`: append new elements to an
//! existing chunked, unlimited, Extensible-Array-indexed dataset in place —
//! filtered and unfiltered, chunk-aligned and not — and read the result back
//! with this crate. C-library interop lives in `append_crosscheck.rs`.

#![allow(deprecated)] // exercises the deprecated EditSession/SwmrWriter shims (issue #148)
use hdf5_pure::{AppendBuilder, EditSession, Error, File, FileBuilder, ScaleOffset};
use tempfile::tempdir;

/// Create a rank-1, unlimited i32 dataset with the given chunk length and
/// (optional) filters, seeded with `0..n`.
fn create_i32(
    path: &std::path::Path,
    n: i32,
    chunk: u64,
    deflate: bool,
    shuffle: bool,
    fletcher32: bool,
) {
    let data: Vec<i32> = (0..n).collect();
    let mut b = FileBuilder::new();
    let ds = b
        .create_dataset("d")
        .with_i32_data(&data)
        .with_shape(&[n as u64])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[chunk]);
    if shuffle {
        ds.with_shuffle();
    }
    if deflate {
        ds.with_deflate(6);
    }
    if fletcher32 {
        ds.with_fletcher32();
    }
    b.write(path).unwrap();
}

fn read_i32(path: &std::path::Path) -> Vec<i32> {
    let f = File::open(path).unwrap();
    f.dataset("d").unwrap().read_i32().unwrap()
}

fn append_i32(path: &std::path::Path, values: &[i32]) {
    let mut s = EditSession::open(path).unwrap();
    s.append_dataset("d").append_i32(values);
    s.commit().unwrap();
}

#[test]
fn append_deflate_shuffle_chunk_aligned() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    // 8 elements, chunk 4 => 2 full chunks. Append 8 more (2 chunks): aligned.
    create_i32(&path, 8, 4, true, true, false);
    append_i32(&path, &(8..16).collect::<Vec<_>>());
    assert_eq!(read_i32(&path), (0..16).collect::<Vec<_>>());
}

#[test]
fn append_unaligned_crosses_chunk_boundary() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    // 6 elements, chunk 4 => 1 full chunk + a partial (2 of 4). Append 5 => 11:
    // the partial chunk is rewritten and the tail grows to 3 chunks.
    create_i32(&path, 6, 4, true, true, false);
    append_i32(&path, &(6..11).collect::<Vec<_>>());
    assert_eq!(read_i32(&path), (0..11).collect::<Vec<_>>());
}

#[test]
fn append_repeated_partial_then_partial() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    create_i32(&path, 3, 4, true, false, false); // 1 partial chunk (3 of 4)
    append_i32(&path, &[3, 4]); // -> 5 (crosses into chunk 1, partial)
    append_i32(&path, &[5, 6, 7]); // -> 8 (fills chunk 1)
    append_i32(&path, &[8]); // -> 9 (new partial chunk 2)
    assert_eq!(read_i32(&path), (0..9).collect::<Vec<_>>());
}

#[test]
fn append_unfiltered() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    create_i32(&path, 6, 3, false, false, false);
    append_i32(&path, &(6..13).collect::<Vec<_>>()); // unaligned (6->13)
    assert_eq!(read_i32(&path), (0..13).collect::<Vec<_>>());
}

#[test]
fn append_fletcher32() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    create_i32(&path, 5, 4, true, false, true);
    append_i32(&path, &(5..12).collect::<Vec<_>>());
    assert_eq!(read_i32(&path), (0..12).collect::<Vec<_>>());
}

#[test]
fn append_scale_offset_f64() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    let init: Vec<f64> = (0..6).map(|i| i as f64 * 0.25).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_f64_data(&init)
        .with_shape(&[6])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[4])
        .with_scale_offset(ScaleOffset::FloatDScale(2));
    b.write(&path).unwrap();

    let more: Vec<f64> = (6..14).map(|i| i as f64 * 0.25).collect();
    {
        let mut s = EditSession::open(&path).unwrap();
        s.append_dataset("d").append_f64(&more);
        s.commit().unwrap();
    }
    let back = File::open(&path)
        .unwrap()
        .dataset("d")
        .unwrap()
        .read_f64()
        .unwrap();
    let expected: Vec<f64> = (0..14).map(|i| i as f64 * 0.25).collect();
    assert_eq!(back.len(), expected.len());
    for (a, b) in back.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-9, "{a} != {b}");
    }
}

#[test]
fn append_generic_and_raw() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    create_i32(&path, 4, 4, true, false, false);
    // Generic append<T>.
    {
        let mut s = EditSession::open(&path).unwrap();
        s.append_dataset("d").append(&[4i32, 5, 6, 7]);
        s.commit().unwrap();
    }
    // Raw append (little-endian i32 bytes).
    {
        let mut s = EditSession::open(&path).unwrap();
        let mut bytes = Vec::new();
        for v in [8i32, 9, 10] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        s.append_dataset("d").append_raw(&bytes);
        s.commit().unwrap();
    }
    assert_eq!(read_i32(&path), (0..11).collect::<Vec<_>>());
}

#[test]
fn append_to_userblock_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("ub.h5");
    // A 512-byte userblock (as a .mat file has) makes the superblock base nonzero,
    // so every stored address is base-relative. The append must place and index
    // its new chunks base-relative too.
    let mut b = FileBuilder::new();
    b.with_userblock(512);
    b.create_dataset("d")
        .with_i32_data(&(0..8).collect::<Vec<_>>())
        .with_shape(&[8])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[4])
        .with_shuffle()
        .with_deflate(6);
    b.write(&path).unwrap();
    append_i32(&path, &(8..17).collect::<Vec<_>>()); // unaligned 8 -> 17
    assert_eq!(read_i32(&path), (0..17).collect::<Vec<_>>());
}

#[test]
fn zero_length_append_is_noop() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    create_i32(&path, 5, 4, true, false, false);
    {
        let mut s = EditSession::open(&path).unwrap();
        s.append_dataset("d").append_i32(&[]);
        s.commit().unwrap();
    }
    assert_eq!(read_i32(&path), (0..5).collect::<Vec<_>>());
}

#[test]
fn many_appends_do_not_rewrite_existing_data() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    // Incompressible base data so its on-disk size dominates the per-append index
    // overhead: if each append rewrote the whole dataset the file would grow by
    // ~one base per append (10x+); keeping existing chunk data in place bounds it.
    let lcg = |mut x: u32| {
        move || {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            x as i32
        }
    };
    let base_n = 20_000usize;
    let mut rng = lcg(0x1234_5678);
    let base: Vec<i32> = (0..base_n).map(|_| rng()).collect();
    {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&base)
            .with_shape(&[base_n as u64])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[200])
            .with_deflate(6);
        b.write(&path).unwrap();
    }
    let size_after_create = std::fs::metadata(&path).unwrap().len();

    let mut expected = base;
    {
        let mut s = EditSession::open(&path).unwrap();
        for _ in 0..10 {
            let batch: Vec<i32> = (0..200).map(|_| rng()).collect();
            expected.extend_from_slice(&batch);
            s.append_dataset("d").append_i32(&batch);
            s.commit().unwrap();
        }
    }
    assert_eq!(read_i32(&path), expected);

    let final_size = std::fs::metadata(&path).unwrap().len();
    // 10 appends of 200 to a 20 000-element incompressible base. Rewriting the
    // dataset each append would roughly multiply the file size; keeping data in
    // place keeps growth to index/header churn, well under a single extra base.
    assert!(
        final_size < size_after_create * 2,
        "file grew from {size_after_create} to {final_size}; appends must not rewrite existing data"
    );
}

#[test]
fn introspection_reports_eligibility() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    create_i32(&path, 8, 4, true, true, false);
    let f = File::open(&path).unwrap();
    let ds = f.dataset("d").unwrap();
    assert!(ds.is_chunked());
    assert_eq!(ds.maxshape().unwrap(), Some(vec![u64::MAX]));
    assert_eq!(ds.chunk_shape().unwrap(), Some(vec![4]));
    // shuffle (2) then deflate (1), in pipeline order.
    assert_eq!(ds.filters(), vec![2, 1]);
}

#[test]
fn introspection_on_contiguous_dataset() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&[1, 2, 3])
        .with_shape(&[3]);
    b.write(&path).unwrap();
    let f = File::open(&path).unwrap();
    let ds = f.dataset("d").unwrap();
    assert!(!ds.is_chunked());
    assert_eq!(ds.maxshape().unwrap(), None);
    assert_eq!(ds.chunk_shape().unwrap(), None);
    assert!(ds.filters().is_empty());
}

// --- Refusal matrix: every case leaves the file unchanged and returns
// Error::AppendUnsupported. ---

/// Stage an append on `dataset` via `build`, commit, and return the result —
/// dropping the session (and its exclusive file lock) before returning so a
/// subsequent readback can reopen the file. On Windows the write lock is
/// mandatory, so a still-open session makes `File::open` fail with a lock
/// violation; keeping the session scoped here is what lets each case verify the
/// file is unchanged after a refused commit.
fn commit_append(
    path: &std::path::Path,
    dataset: &str,
    build: impl FnOnce(&mut AppendBuilder),
) -> Result<(), Error> {
    let mut s = EditSession::open(path).unwrap();
    build(s.append_dataset(dataset));
    s.commit()
}

fn assert_append_unsupported(res: Result<(), Error>) {
    match res {
        Err(Error::AppendUnsupported(_)) => {}
        other => panic!("expected AppendUnsupported, got {other:?}"),
    }
}

#[test]
fn refuse_contiguous() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&[1, 2, 3])
        .with_shape(&[3]);
    b.write(&path).unwrap();
    assert_append_unsupported(commit_append(&path, "d", |a| {
        a.append_i32(&[4, 5]);
    }));
    assert_eq!(read_i32(&path), vec![1, 2, 3]);
}

#[test]
fn refuse_fixed_chunked_not_unlimited() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    // Chunked but a finite maximum (not unlimited) => fixed-array index, refused.
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&(0..6).collect::<Vec<_>>())
        .with_shape(&[6])
        .with_maxshape(&[100])
        .with_chunks(&[3]);
    b.write(&path).unwrap();
    assert_append_unsupported(commit_append(&path, "d", |a| {
        a.append_i32(&[6, 7, 8]);
    }));
    assert_eq!(read_i32(&path), (0..6).collect::<Vec<_>>());
}

#[test]
fn refuse_rank2() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    // 2-D dataset with one unlimited axis (EA index) — refused as rank > 1.
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&(0..12).collect::<Vec<_>>())
        .with_shape(&[3, 4])
        .with_maxshape(&[u64::MAX, 4])
        .with_chunks(&[1, 4]);
    b.write(&path).unwrap();
    assert_append_unsupported(commit_append(&path, "d", |a| {
        a.append_i32(&[0, 0, 0, 0]);
    }));
}

#[test]
fn refuse_datatype_mismatch() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    create_i32(&path, 4, 4, true, false, false);
    assert_append_unsupported(commit_append(&path, "d", |a| {
        a.append_f64(&[1.0, 2.0]); // f64 onto i32
    }));
    assert_eq!(read_i32(&path), (0..4).collect::<Vec<_>>());
}

#[test]
fn refuse_raw_wrong_length() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    create_i32(&path, 4, 4, true, false, false);
    assert_append_unsupported(commit_append(&path, "d", |a| {
        a.append_raw(&[1, 2, 3]); // 3 bytes, elem size 4
    }));
    assert_eq!(read_i32(&path), (0..4).collect::<Vec<_>>());
}

#[test]
fn refuse_mixed_element_types() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    create_i32(&path, 4, 4, true, false, false);
    assert_append_unsupported(commit_append(&path, "d", |a| {
        a.append_i32(&[4, 5]).append_i64(&[6]);
    }));
    assert_eq!(read_i32(&path), (0..4).collect::<Vec<_>>());
}

#[test]
fn refuse_nonexistent_dataset() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    create_i32(&path, 4, 4, true, false, false);
    assert_append_unsupported(commit_append(&path, "missing", |a| {
        a.append_i32(&[1]);
    }));
}
