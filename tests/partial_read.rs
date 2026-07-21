//! Tests for the windowed (leading-dimension row-range) read API:
//! `Dataset::read_rows_raw` and the typed `read_*_rows` helpers (issue: partial
//! reads).
//!
//! The invariant under test everywhere: a row window must be byte-for-byte the
//! same as the whole-dataset read sliced to that row range — for every layout
//! (contiguous, chunked, inner-chunked fallback), on both the in-memory and the
//! streaming backend, and across edge windows (empty, past-the-end, straddling
//! chunk boundaries).

use hdf5_pure::{File, FileBuilder};

/// Open the bytes produced by `build` on both backends: buffered (in-memory) and
/// streaming (from a temp file). The windowed read dispatches per backend, so
/// every case is checked on both. The `TempDir` is returned so it outlives the
/// streaming file handle.
fn on_both_backends(
    build: impl FnOnce(&mut FileBuilder),
) -> (File, File, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.h5");
    let mut builder = FileBuilder::new();
    build(&mut builder);
    builder.write(&path).unwrap();

    let buffered = File::open(&path).unwrap();
    let streaming = File::open_streaming(&path).unwrap();
    (buffered, streaming, dir)
}

/// Row windows worth checking for a dataset whose leading dimension is `n0`:
/// whole, first row, a middle band, the last row, empty, past-the-end (clamps to
/// empty), and an over-long window (clamps to the tail).
fn windows(n0: u64) -> Vec<(u64, u64)> {
    let mut w = vec![
        (0, n0),
        (0, 1.min(n0)),
        (n0 / 2, n0 - n0 / 2),
        (n0.saturating_sub(1), 1.min(n0)),
        (n0 / 3, n0 / 3),
        (0, 0),
        (n0, 5),         // start past the end -> empty
        (n0.saturating_sub(2), 100), // over-long -> clamps to the tail
    ];
    w.dedup();
    w
}

fn check_f64(file: &File, name: &str, row_elems: usize, extra: &[(u64, u64)]) {
    let ds = file.dataset(name).unwrap();
    let full = ds.read_f64().unwrap();
    let n0 = (full.len() / row_elems.max(1)) as u64;

    for (start, count) in windows(n0).into_iter().chain(extra.iter().copied()) {
        let clamped_start = start.min(n0);
        let clamped_count = count.min(n0 - clamped_start);
        let lo = clamped_start as usize * row_elems;
        let hi = lo + clamped_count as usize * row_elems;

        let window = ds.read_f64_rows(start, count).unwrap();
        assert_eq!(
            window,
            &full[lo..hi],
            "{name}: read_f64_rows({start}, {count}) != full[{lo}..{hi}]"
        );
    }
}

fn check_u8(file: &File, name: &str, row_elems: usize, extra: &[(u64, u64)]) {
    let ds = file.dataset(name).unwrap();
    let full = ds.read_u8().unwrap();
    let n0 = (full.len() / row_elems.max(1)) as u64;

    for (start, count) in windows(n0).into_iter().chain(extra.iter().copied()) {
        let clamped_start = start.min(n0);
        let clamped_count = count.min(n0 - clamped_start);
        let lo = clamped_start as usize * row_elems;
        let hi = lo + clamped_count as usize * row_elems;

        let window = ds.read_u8_rows(start, count).unwrap();
        assert_eq!(
            window,
            &full[lo..hi],
            "{name}: read_u8_rows({start}, {count}) != full[{lo}..{hi}]"
        );
    }
}

#[test]
fn contiguous_1d_f64() {
    let data: Vec<f64> = (0..100).map(|i| i as f64 * 1.5).collect();
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("x").with_f64_data(&data).with_shape(&[100]);
    });
    check_f64(&buffered, "x", 1, &[]);
    check_f64(&streaming, "x", 1, &[]);
}

#[test]
fn contiguous_2d_f64_preserves_inner_shape() {
    // shape [20, 3]: each row is 3 elements; the window must keep that stride.
    let data: Vec<f64> = (0..60).map(f64::from).collect();
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("m").with_f64_data(&data).with_shape(&[20, 3]);
    });
    check_f64(&buffered, "m", 3, &[(5, 7), (0, 20)]);
    check_f64(&streaming, "m", 3, &[(5, 7), (0, 20)]);
}

#[test]
fn chunked_1d_deflate_multi_chunk() {
    // 1000 rows in 64-row chunks -> many chunks; windows straddle boundaries.
    let data: Vec<f64> = (0..1000).map(|i| i as f64 * 0.25).collect();
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("c")
            .with_f64_data(&data)
            .with_shape(&[1000])
            .with_chunks(&[64])
            .with_deflate(4);
    });
    // Windows that start/end inside a chunk, cross exactly one boundary, and span
    // several chunks.
    let extra = &[(60, 10), (64, 64), (100, 300), (63, 2), (999, 1)];
    check_f64(&buffered, "c", 1, extra);
    check_f64(&streaming, "c", 1, extra);
}

#[test]
fn chunked_frame_per_chunk_4d_u8() {
    // The re_hdf5 image case: [10, 2, 2, 3] uint8, one frame per chunk, deflated.
    let data: Vec<u8> = (0..10 * 2 * 2 * 3).map(|i| (i % 251) as u8).collect();
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("frames")
            .with_u8_data(&data)
            .with_shape(&[10, 2, 2, 3])
            .with_chunks(&[1, 2, 2, 3])
            .with_deflate(4);
    });
    let extra = &[(3, 4), (0, 1), (9, 1)];
    check_u8(&buffered, "frames", 2 * 2 * 3, extra);
    check_u8(&streaming, "frames", 2 * 2 * 3, extra);
}

#[test]
fn chunked_multi_row_chunk_partial_edges() {
    // shape [50, 4], chunks [8, 4]: a chunk spans 8 rows, so a window straddling a
    // chunk exercises the top/bottom row-band trim inside a single chunk.
    let data: Vec<f64> = (0..200).map(f64::from).collect();
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("d")
            .with_f64_data(&data)
            .with_shape(&[50, 4])
            .with_chunks(&[8, 4])
            .with_deflate(2);
    });
    let extra = &[(3, 2), (6, 5), (7, 1), (8, 8), (0, 9), (49, 1)];
    check_f64(&buffered, "d", 4, extra);
    check_f64(&streaming, "d", 4, extra);
}

#[test]
fn inner_chunked_falls_back_to_whole_read() {
    // shape [20, 6], chunks [4, 2]: the inner dim is chunked, so the windowed chunk
    // reader returns None and the caller falls back to a sliced whole-read. The
    // result must still be correct.
    let data: Vec<f64> = (0..120).map(f64::from).collect();
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("t")
            .with_f64_data(&data)
            .with_shape(&[20, 6])
            .with_chunks(&[4, 2]);
    });
    let extra = &[(3, 5), (0, 20), (10, 1)];
    check_f64(&buffered, "t", 6, extra);
    check_f64(&streaming, "t", 6, extra);
}

#[test]
fn scalar_0d_is_a_single_row() {
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("s").with_i64_data(&[42]).with_shape(&[]);
    });
    for file in [&buffered, &streaming] {
        let ds = file.dataset("s").unwrap();
        assert_eq!(ds.read_i64_rows(0, 1).unwrap(), vec![42]);
        // Past the single row -> empty.
        assert_eq!(ds.read_i64_rows(1, 1).unwrap(), Vec::<i64>::new());
    }
}

#[test]
fn fixed_length_strings_window() {
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("labels")
            .with_vlen_strings(&["idle", "reach", "grasp", "lift", "place"]);
    });
    for file in [&buffered, &streaming] {
        let ds = file.dataset("labels").unwrap();
        let full = ds.read_string().unwrap();
        assert_eq!(ds.read_string_rows(0, 5).unwrap(), full);
        assert_eq!(ds.read_string_rows(1, 2).unwrap(), &full[1..3]);
        assert_eq!(ds.read_string_rows(4, 10).unwrap(), &full[4..5]);
        assert_eq!(ds.read_string_rows(5, 3).unwrap(), Vec::<String>::new());
    }
}

#[test]
fn typed_readers_match_whole_dataset() {
    // Spot-check several dtypes go through the same window path and decode right.
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("i32").with_i32_data(&[10, 20, 30, 40, 50]);
        b.create_dataset("u64")
            .with_u64_data(&[1, 2, 3, 4])
            .with_chunks(&[2])
            .with_shape(&[4]);
        b.create_dataset("f32").with_f32_data(&[1.5, 2.5, 3.5]);
    });
    for file in [&buffered, &streaming] {
        let i32s = file.dataset("i32").unwrap();
        assert_eq!(i32s.read_i32_rows(1, 3).unwrap(), &i32s.read_i32().unwrap()[1..4]);

        let u64s = file.dataset("u64").unwrap();
        assert_eq!(u64s.read_u64_rows(1, 2).unwrap(), &u64s.read_u64().unwrap()[1..3]);

        let f32s = file.dataset("f32").unwrap();
        assert_eq!(f32s.read_f32_rows(0, 2).unwrap(), &f32s.read_f32().unwrap()[0..2]);
    }
}

#[test]
fn read_rows_raw_matches_read_raw() {
    // The raw-byte contract, independent of dtype decoding, on a compressed chunked
    // dataset large enough to span chunks.
    let data: Vec<f64> = (0..500).map(|i| i as f64 * -0.5).collect();
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("r")
            .with_f64_data(&data)
            .with_shape(&[500])
            .with_chunks(&[100])
            .with_deflate(6);
    });
    for file in [&buffered, &streaming] {
        let ds = file.dataset("r").unwrap();
        let full = ds.read_raw().unwrap();
        let elem = 8; // f64
        for (start, count) in [(0u64, 500u64), (150, 200), (99, 3), (450, 100)] {
            let s = start as usize * elem;
            let e = (start + count).min(500) as usize * elem;
            assert_eq!(ds.read_rows_raw(start, count).unwrap(), &full[s..e]);
        }
    }
}
