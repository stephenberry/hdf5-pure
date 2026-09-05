//! Tests for the regional read API: `Dataset::read_raw_region` and the typed
//! `read_*_region` helpers.
//!
//! The invariant under test everywhere: a region must be byte-for-byte the
//! whole-dataset read cut to that box — for every layout (contiguous, chunked,
//! inner-chunked grids, filtered), on both the in-memory and the streaming
//! backend, and across edge regions (empty, one element, boxes straddling chunk
//! boundaries on inner axes, the whole dataset). What separates a region from a
//! row window is the axis it cuts, so every fixture here cuts an inner one.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use hdf5_pure::{Error, File, FileBuilder, FormatError, Source};

/// Open the bytes produced by `build` on both backends: buffered (in-memory) and
/// streaming (from a temp file). The regional read dispatches per backend, so
/// every case is checked on both. The `TempDir` is returned so it outlives the
/// streaming file handle.
fn on_both_backends(build: impl FnOnce(&mut FileBuilder)) -> (File, File, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.h5");
    let mut builder = FileBuilder::new();
    build(&mut builder);
    builder.write(&path).unwrap();

    let buffered = File::open(&path).unwrap();
    let streaming = File::open_streaming(&path).unwrap();
    (buffered, streaming, dir)
}

/// `full` cut to the box `start .. start + count` of a row-major array of
/// `dims`, the slow way — one element at a time — so the reader's run and chunk
/// arithmetic is checked against something that has none.
fn cut<T: Copy>(full: &[T], dims: &[u64], start: &[u64], count: &[u64]) -> Vec<T> {
    let rank = dims.len();
    let n = count.iter().product::<u64>() as usize;
    let mut out = Vec::with_capacity(n);
    let mut at = vec![0u64; rank];
    for _ in 0..n {
        let mut flat = 0u64;
        for i in 0..rank {
            flat = flat * dims[i] + start[i] + at[i];
        }
        out.push(full[flat as usize]);
        for i in (0..rank).rev() {
            at[i] += 1;
            if at[i] < count[i] {
                break;
            }
            at[i] = 0;
        }
    }
    out
}

/// Regions worth checking for a dataset of `dims`: the whole, the first
/// element, a box in the middle, the last element, an empty box at the far
/// corner, and the last column — a band that spans every axis but the last.
fn regions(dims: &[u64]) -> Vec<(Vec<u64>, Vec<u64>)> {
    let rank = dims.len();
    let zeros = vec![0u64; rank];
    let ones = vec![1u64; rank];
    let last: Vec<u64> = dims.iter().map(|&d| d - 1).collect();
    let quarter: Vec<u64> = dims.iter().map(|&d| d / 4).collect();
    let half: Vec<u64> = dims.iter().map(|&d| (d / 2).max(1)).collect();
    let mut column_start = zeros.clone();
    let mut column_count = dims.to_vec();
    if rank > 0 {
        column_start[rank - 1] = last[rank - 1];
        column_count[rank - 1] = 1;
    }
    vec![
        (zeros.clone(), dims.to_vec()),
        (zeros, ones.clone()),
        (quarter, half),
        (last, ones),
        (dims.to_vec(), vec![0; rank]),
        (column_start, column_count),
    ]
}

fn boxes(extra: &[(&[u64], &[u64])]) -> Vec<(Vec<u64>, Vec<u64>)> {
    extra
        .iter()
        .map(|(s, c)| (s.to_vec(), c.to_vec()))
        .collect()
}

fn check_f64(file: &File, name: &str, dims: &[u64], extra: &[(&[u64], &[u64])]) {
    let ds = file.dataset(name).unwrap();
    let full = ds.read_f64().unwrap();
    for (start, count) in regions(dims).into_iter().chain(boxes(extra)) {
        let region = ds
            .read_f64_region(&start, &count)
            .unwrap_or_else(|e| panic!("region {start:?} + {count:?}: {e}"));
        assert_eq!(
            region,
            cut(&full, dims, &start, &count),
            "region {start:?} + {count:?} of {dims:?}"
        );
    }
}

fn check_u8(file: &File, name: &str, dims: &[u64], extra: &[(&[u64], &[u64])]) {
    let ds = file.dataset(name).unwrap();
    let full = ds.read_u8().unwrap();
    for (start, count) in regions(dims).into_iter().chain(boxes(extra)) {
        let region = ds
            .read_u8_region(&start, &count)
            .unwrap_or_else(|e| panic!("region {start:?} + {count:?}: {e}"));
        assert_eq!(
            region,
            cut(&full, dims, &start, &count),
            "region {start:?} + {count:?} of {dims:?}"
        );
    }
}

fn check_u16(file: &File, name: &str, dims: &[u64], extra: &[(&[u64], &[u64])]) {
    let ds = file.dataset(name).unwrap();
    let full = ds.read_u16().unwrap();
    for (start, count) in regions(dims).into_iter().chain(boxes(extra)) {
        let region = ds
            .read_u16_region(&start, &count)
            .unwrap_or_else(|e| panic!("region {start:?} + {count:?}: {e}"));
        assert_eq!(
            region,
            cut(&full, dims, &start, &count),
            "region {start:?} + {count:?} of {dims:?}"
        );
    }
}

fn check_f32(file: &File, name: &str, dims: &[u64], extra: &[(&[u64], &[u64])]) {
    let ds = file.dataset(name).unwrap();
    let full = ds.read_f32().unwrap();
    for (start, count) in regions(dims).into_iter().chain(boxes(extra)) {
        let region = ds
            .read_f32_region(&start, &count)
            .unwrap_or_else(|e| panic!("region {start:?} + {count:?}: {e}"));
        assert_eq!(
            region,
            cut(&full, dims, &start, &count),
            "region {start:?} + {count:?} of {dims:?}"
        );
    }
}

#[test]
fn contiguous_2d_cuts_columns_into_one_run_per_row() {
    // [12, 10] contiguous: a box that cuts the inner axis is one storage run
    // per row of the box; a box spanning the inner axis is a single run.
    let dims = [12u64, 10];
    let data: Vec<f64> = (0..120).map(|i| f64::from(i) * 0.5).collect();
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("c").with_f64_data(&data).with_shape(&dims);
    });
    let extra: &[(&[u64], &[u64])] = &[
        (&[1, 2], &[3, 4]),
        (&[0, 5], &[12, 5]),
        (&[0, 0], &[12, 5]), // the first columns: cut from its start
        (&[5, 0], &[1, 10]),
        (&[11, 9], &[1, 1]),
    ];
    check_f64(&buffered, "c", &dims, extra);
    check_f64(&streaming, "c", &dims, extra);
}

#[test]
fn contiguous_3d_folds_the_axes_a_region_spans() {
    // [4, 5, 6] u8: a box spanning the innermost axis folds it into the run of
    // the axis above; a box cutting it does not.
    let dims = [4u64, 5, 6];
    let data: Vec<u8> = (0..120).map(|i| (i * 7 % 251) as u8).collect();
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("v").with_u8_data(&data).with_shape(&dims);
    });
    let extra: &[(&[u64], &[u64])] = &[
        (&[1, 1, 0], &[2, 3, 6]),
        (&[0, 0, 2], &[4, 5, 2]),
        (&[2, 0, 0], &[1, 5, 6]),
    ];
    check_u8(&buffered, "v", &dims, extra);
    check_u8(&streaming, "v", &dims, extra);
}

#[test]
fn inner_chunked_2d_grid_straddles_chunks_on_both_axes() {
    // [23, 10] in [4, 3] chunks, deflated: neither dimension is a multiple of
    // its chunk extent, so the last chunk row and column overhang the dataset,
    // and a box can start and end inside a chunk on either axis.
    let dims = [23u64, 10];
    let data: Vec<f64> = (0..230).map(|i| f64::from(i) * 0.5 - 3.0).collect();
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("e")
            .with_f64_data(&data)
            .with_shape(&dims)
            .with_chunks(&[4, 3])
            .with_deflate(4);
    });
    let extra: &[(&[u64], &[u64])] = &[
        (&[3, 2], &[5, 5]),  // straddles a chunk boundary on both axes
        (&[0, 9], &[23, 1]), // the last column, through every chunk row
        (&[22, 9], &[1, 1]), // the corner element of the overhanging chunk
        (&[4, 3], &[4, 3]),  // exactly one chunk
        (&[2, 1], &[20, 8]), // most of the dataset, edges inside chunks
    ];
    check_f64(&buffered, "e", &dims, extra);
    check_f64(&streaming, "e", &dims, extra);
}

#[test]
fn a_band_of_whole_rows_is_the_row_window() {
    // A region spanning every inner axis is a row window, and the two must
    // agree element for element — the window is read through the region.
    let data: Vec<f64> = (0..200).map(f64::from).collect();
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("d")
            .with_f64_data(&data)
            .with_shape(&[50, 4])
            .with_chunks(&[8, 4])
            .with_deflate(2);
    });
    for file in [&buffered, &streaming] {
        let ds = file.dataset("d").unwrap();
        for (start, count) in [(6u64, 5u64), (0, 8), (7, 1), (47, 3)] {
            assert_eq!(
                ds.read_f64_region(&[start, 0], &[count, 4]).unwrap(),
                ds.read_f64_rows(start, count).unwrap(),
                "rows {start}..{}",
                start + count
            );
        }
    }
}

#[test]
fn frames_of_a_4d_stack_with_shuffle_and_deflate() {
    // [10, 2, 6, 3] u16, one half-frame per chunk, shuffled and deflated: a
    // frame is one region, and a box across frames meets a chunk in each.
    let dims = [10u64, 2, 6, 3];
    let data: Vec<u16> = (0..360).map(|i| (i * 37 % 65521) as u16).collect();
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("stack")
            .with_u16_data(&data)
            .with_shape(&dims)
            .with_chunks(&[1, 2, 3, 3])
            .with_shuffle()
            .with_deflate(4);
    });
    let extra: &[(&[u64], &[u64])] = &[
        (&[4, 0, 0, 0], &[1, 2, 6, 3]), // one frame
        (&[2, 1, 2, 1], &[3, 1, 3, 2]), // a box across three frames
        (&[9, 1, 5, 2], &[1, 1, 1, 1]), // the last element
    ];
    check_u16(&buffered, "stack", &dims, extra);
    check_u16(&streaming, "stack", &dims, extra);
}

#[test]
fn a_unit_leading_axis_windows_along_the_next_one() {
    // The shape netCDF writers give a single time step — `[1, rows, columns]`
    // — where a row window is the whole plane: a region cuts the second axis.
    let dims = [1u64, 40, 9];
    let data: Vec<f32> = (0..360).map(|i| i as f32 * 0.25).collect();
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("plane")
            .with_f32_data(&data)
            .with_shape(&dims)
            .with_chunks(&[1, 8, 9])
            .with_deflate(1);
    });
    let extra: &[(&[u64], &[u64])] = &[
        (&[0, 13, 0], &[1, 10, 9]), // a band of rows, straddling two chunks
        (&[0, 0, 4], &[1, 40, 2]),  // two columns through every chunk
        (&[0, 32, 0], &[1, 8, 9]),  // exactly the last chunk
    ];
    check_f32(&buffered, "plane", &dims, extra);
    check_f32(&streaming, "plane", &dims, extra);
}

/// A source that counts the bytes it serves. That counter is what makes the
/// test below load-bearing rather than decorative: a read that quietly
/// materialized the plane and cut the region out of it would still return the
/// right numbers, and only the byte count separates that from reading the
/// chunks the region meets.
struct Counting {
    bytes: Vec<u8>,
    served: Arc<AtomicU64>,
}

impl Source for Counting {
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        let at = usize::try_from(offset).expect("the fixture fits this platform");
        let end = at + buf.len();
        if end > self.bytes.len() {
            return Err(FormatError::UnexpectedEof {
                expected: end,
                available: self.bytes.len(),
            });
        }
        buf.copy_from_slice(&self.bytes[at..end]);
        self.served.fetch_add(buf.len() as u64, Ordering::Relaxed);
        Ok(())
    }
}

#[test]
fn a_region_reads_only_the_chunks_it_meets() {
    // [1, 400, 64] f32 in [1, 8, 64] chunks — 50 chunks of 2 KiB, unfiltered
    // so bytes served are bytes stored. A 16-row band at row 100 meets three
    // chunks (96..104, 104..112, 112..120); a whole-plane read would serve all
    // fifty.
    const ROWS: u64 = 400;
    const COLS: u64 = 64;
    const CHUNK_ROWS: u64 = 8;
    const CHUNK_BYTES: u64 = CHUNK_ROWS * COLS * 4;
    let data: Vec<f32> = (0..ROWS * COLS).map(|i| i as f32).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plane.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("plane")
            .with_f32_data(&data)
            .with_shape(&[1, ROWS, COLS])
            .with_chunks(&[1, CHUNK_ROWS, COLS]);
        b.write(&path).unwrap();
    }
    let whole = std::fs::read(&path).unwrap();
    let served = Arc::new(AtomicU64::new(0));
    let file = File::from_source(Counting {
        bytes: whole,
        served: Arc::clone(&served),
    })
    .unwrap();
    let ds = file.dataset("plane").unwrap();

    let before = served.load(Ordering::Relaxed);
    let band = ds.read_f32_region(&[0, 100, 0], &[1, 16, COLS]).unwrap();
    let cost = served.load(Ordering::Relaxed) - before;
    assert_eq!(
        band,
        cut(&data, &[1, ROWS, COLS], &[0, 100, 0], &[1, 16, COLS]),
        "a region read through a source differs from what was written"
    );
    // Three chunks of data plus the chunk index, walked once. The index of
    // fifty chunks is a few KiB; a plane materialized whole is 100 KiB.
    let bound = 3 * CHUNK_BYTES + 16 * 1024;
    assert!(
        cost <= bound,
        "a 16-row region served {cost} bytes, above its three chunks plus index ({bound})"
    );

    // The adjacent band starts in the chunk the first one finished on, which
    // the cache retained; its other chunk is new.
    let before = served.load(Ordering::Relaxed);
    let next = ds.read_f32_region(&[0, 116, 0], &[1, 8, COLS]).unwrap();
    let cost = served.load(Ordering::Relaxed) - before;
    assert_eq!(
        next,
        cut(&data, &[1, ROWS, COLS], &[0, 116, 0], &[1, 8, COLS])
    );
    assert!(
        cost <= CHUNK_BYTES,
        "the adjacent band served {cost} bytes; the chunk it shares with its \
         predecessor should come from the cache and only the new one from the source"
    );
}

#[test]
fn a_region_past_the_edge_or_of_another_rank_is_refused() {
    let data: Vec<f64> = (0..48).map(f64::from).collect();
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("g")
            .with_f64_data(&data)
            .with_shape(&[6, 8])
            .with_chunks(&[2, 4]);
    });
    for file in [&buffered, &streaming] {
        let ds = file.dataset("g").unwrap();

        // Past the edge on the inner axis: refused, naming the axis, rather
        // than clamped to a 3×2 box the caller did not ask for.
        let err = ds.read_raw_region(&[2, 6], &[3, 3]).unwrap_err();
        match &err {
            Error::InvalidRegion(msg) => {
                assert!(msg.contains("axis 1") && msg.contains("6..9"), "{msg}");
            }
            other => panic!("expected InvalidRegion, got {other:?}"),
        }
        assert!(err.to_string().contains("region does not fit"), "{err}");

        // The wrong number of axes.
        assert!(matches!(
            ds.read_f64_region(&[2], &[3, 3]),
            Err(Error::InvalidRegion(_))
        ));
        assert!(matches!(
            ds.read_f64_region(&[0, 0, 0], &[6, 8, 1]),
            Err(Error::InvalidRegion(_))
        ));

        // One past the edge is past the edge.
        assert!(matches!(
            ds.read_raw_region(&[0, 7], &[6, 2]),
            Err(Error::InvalidRegion(_))
        ));

        // A zero count anywhere is an empty read, at the far corner included.
        assert!(ds.read_f64_region(&[2, 3], &[0, 3]).unwrap().is_empty());
        assert!(ds.read_f64_region(&[6, 8], &[0, 0]).unwrap().is_empty());
        assert!(ds.read_raw_region(&[0, 0], &[6, 0]).unwrap().is_empty());
    }
}

#[test]
fn typed_regions_match_their_whole_reads() {
    let dims = [6u64, 5, 4];
    let n = 120usize;
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("i16")
            .with_i16_data(&(0..n).map(|i| i as i16 - 60).collect::<Vec<_>>())
            .with_shape(&dims)
            .with_chunks(&[2, 5, 2]);
        b.create_dataset("u16")
            .with_u16_data(&(0..n).map(|i| i as u16 * 3).collect::<Vec<_>>())
            .with_shape(&dims)
            .with_chunks(&[3, 2, 4])
            .with_deflate(1);
        b.create_dataset("i32")
            .with_i32_data(&(0..n).map(|i| i as i32 * -7).collect::<Vec<_>>())
            .with_shape(&dims);
        b.create_dataset("u8")
            .with_u8_data(&(0..n).map(|i| i as u8).collect::<Vec<_>>())
            .with_shape(&dims)
            .with_chunks(&[6, 1, 4]);
        b.create_dataset("f32")
            .with_f32_data(&(0..n).map(|i| i as f32 / 3.0).collect::<Vec<_>>())
            .with_shape(&dims)
            .with_chunks(&[1, 5, 4])
            .with_shuffle()
            .with_deflate(3);
    });
    let (start, count) = (&[1u64, 1, 1], &[4u64, 3, 2]);
    for file in [&buffered, &streaming] {
        let ds = file.dataset("i16").unwrap();
        assert_eq!(
            ds.read_i16_region(start, count).unwrap(),
            cut(&ds.read_i16().unwrap(), &dims, start, count)
        );
        let ds = file.dataset("u16").unwrap();
        assert_eq!(
            ds.read_u16_region(start, count).unwrap(),
            cut(&ds.read_u16().unwrap(), &dims, start, count)
        );
        let ds = file.dataset("i32").unwrap();
        assert_eq!(
            ds.read_i32_region(start, count).unwrap(),
            cut(&ds.read_i32().unwrap(), &dims, start, count)
        );
        // The widening decoders read the same file.
        assert_eq!(
            ds.read_i64_region(start, count).unwrap(),
            cut(&ds.read_i64().unwrap(), &dims, start, count)
        );
        assert_eq!(
            ds.read_f64_region(start, count).unwrap(),
            cut(&ds.read_f64().unwrap(), &dims, start, count)
        );
        let ds = file.dataset("u8").unwrap();
        assert_eq!(
            ds.read_u8_region(start, count).unwrap(),
            cut(&ds.read_u8().unwrap(), &dims, start, count)
        );
        assert_eq!(
            ds.read_i8_region(start, count).unwrap(),
            cut(&ds.read_i8().unwrap(), &dims, start, count)
        );
        assert_eq!(
            ds.read_u32_region(start, count).unwrap(),
            cut(&ds.read_u32().unwrap(), &dims, start, count)
        );
        assert_eq!(
            ds.read_u64_region(start, count).unwrap(),
            cut(&ds.read_u64().unwrap(), &dims, start, count)
        );
        let ds = file.dataset("f32").unwrap();
        assert_eq!(
            ds.read_f32_region(start, count).unwrap(),
            cut(&ds.read_f32().unwrap(), &dims, start, count)
        );
    }
}

#[test]
fn read_raw_region_matches_read_raw() {
    // The raw path is what every typed helper decodes from; its bytes must be
    // the whole read's bytes cut element by element.
    let dims = [9u64, 7];
    let data: Vec<f64> = (0..63).map(|i| f64::from(i) * 1.5).collect();
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("r")
            .with_f64_data(&data)
            .with_shape(&dims)
            .with_chunks(&[4, 3])
            .with_deflate(2);
    });
    for file in [&buffered, &streaming] {
        let ds = file.dataset("r").unwrap();
        let raw = ds.read_raw().unwrap();
        let (elems, rest) = raw.as_chunks::<8>();
        assert!(rest.is_empty(), "f64 elements are whole");
        for (start, count) in regions(&dims)
            .into_iter()
            .chain(boxes(&[(&[2, 2], &[5, 3])]))
        {
            let expected: Vec<u8> = cut(elems, &dims, &start, &count).concat();
            assert_eq!(
                ds.read_raw_region(&start, &count).unwrap(),
                expected,
                "region {start:?} + {count:?}"
            );
        }
        // The whole dataset as a region is the whole read.
        assert_eq!(ds.read_raw_region(&[0, 0], &dims).unwrap(), raw);
    }
}

#[test]
fn vlen_strings_region_resolves_only_its_own_cells() {
    // Variable-length strings are heap-backed; `read_string_region` resolves
    // the references of the region's cells, in the region's row-major order.
    let words = [
        "a", "bb", "ccc", "dddd", "e", "ff", "ggg", "hhhh", "i", "jj", "kkk", "llll",
    ];
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("grid")
            .with_vlen_strings(&words)
            .with_shape(&[3, 4]);
    });
    for file in [&buffered, &streaming] {
        let ds = file.dataset("grid").unwrap();
        let all = ds.read_string().unwrap();
        assert_eq!(all.len(), 12);
        assert_eq!(
            ds.read_string_region(&[1, 1], &[2, 2]).unwrap(),
            vec!["ff", "ggg", "jj", "kkk"]
        );
        assert_eq!(
            ds.read_string_region(&[0, 3], &[3, 1]).unwrap(),
            vec!["dddd", "hhhh", "llll"]
        );
        assert_eq!(ds.read_string_region(&[0, 0], &[3, 4]).unwrap(), all);
        assert!(ds.read_string_region(&[1, 1], &[0, 2]).unwrap().is_empty());

        // Lockstep with the raw path: the string count of a region equals the
        // number of elements `read_raw_region` reads for the same region.
        let elem = ds.datatype().unwrap().type_size() as usize;
        for (start, count) in [([1u64, 1], [2u64, 2]), ([0, 3], [3, 1]), ([2, 0], [1, 4])] {
            let n_strings = ds.read_string_region(&start, &count).unwrap().len();
            let n_raw = ds.read_raw_region(&start, &count).unwrap().len() / elem;
            assert_eq!(n_strings, n_raw, "region {start:?} + {count:?}");
        }
    }
}

#[test]
fn a_region_skips_the_chunk_columns_it_misses() {
    // [256, 256] f32 in [32, 32] chunks: an 8×8 grid of 4 KiB chunks, 256 KiB
    // in all, unfiltered so bytes served are bytes stored. What a row window
    // cannot do is skip a chunk column; the byte count is what shows a region
    // doing it.
    const SIDE: u64 = 256;
    const CHUNK: u64 = 32;
    const CHUNK_BYTES: u64 = CHUNK * CHUNK * 4;
    let data: Vec<f32> = (0..SIDE * SIDE).map(|i| i as f32).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("grid.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("grid")
            .with_f32_data(&data)
            .with_shape(&[SIDE, SIDE])
            .with_chunks(&[CHUNK, CHUNK]);
        b.write(&path).unwrap();
    }
    let whole = std::fs::read(&path).unwrap();
    let served = Arc::new(AtomicU64::new(0));
    let file = File::from_source(Counting {
        bytes: whole,
        served: Arc::clone(&served),
    })
    .unwrap();
    let ds = file.dataset("grid").unwrap();

    // A box inside one chunk column and two chunk rows: two chunks. The rows
    // it spans hold sixteen chunks, the dataset sixty-four.
    let before = served.load(Ordering::Relaxed);
    let inner = ds.read_f32_region(&[100, 100], &[50, 20]).unwrap();
    let cost = served.load(Ordering::Relaxed) - before;
    assert_eq!(inner, cut(&data, &[SIDE, SIDE], &[100, 100], &[50, 20]));
    let bound = 2 * CHUNK_BYTES + 16 * 1024;
    assert!(
        cost <= bound,
        "a box inside one chunk column served {cost} bytes, above its two chunks plus index ({bound}); \
         the rows it spans would be {} bytes",
        16 * CHUNK_BYTES
    );

    // A whole chunk column: eight chunks, two of them the cache still holds
    // from the box above, and the index already walked.
    let before = served.load(Ordering::Relaxed);
    let column = ds.read_f32_region(&[0, 96], &[SIDE, 32]).unwrap();
    let cost = served.load(Ordering::Relaxed) - before;
    assert_eq!(column, cut(&data, &[SIDE, SIDE], &[0, 96], &[SIDE, 32]));
    assert!(
        cost <= 6 * CHUNK_BYTES,
        "a chunk column served {cost} bytes; at most six new chunks were due"
    );
}
