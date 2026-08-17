// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit little-endian targets; skip them elsewhere so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
//! A chunk index numbers its slots over the dataset's *maximum* chunk grid, and
//! an Extensible Array rotates its unlimited dimension to the front first
//! (issue #299). Both facts are invisible to a round trip through this crate
//! alone — the writer and the reader agreed with each other while disagreeing
//! with every other HDF5 reader — so the oracle here is the reference C library
//! on the other side of each file.
//!
//! The combinations are swept rather than sampled. There are few enough of them
//! to enumerate, and the defect this file exists for was invisible in exactly
//! the sample the suite already had: every chunked crosscheck was either rank 1
//! or `maxshape == shape`, and growing the *first* dimension changes no
//! multiplier, so three of the four obvious fixtures pass over the bug.

use hdf5::file::LibraryVersion;
use hdf5_pure::{ChunkIndex, File, FileBuilder};
use tempfile::tempdir;

/// `H5S_UNLIMITED`.
const U: u64 = u64::MAX;

/// One swept combination: a label for the assertion messages, then the shape,
/// the chunk dimensions and the maximum shape.
type Case = (String, Vec<u64>, Vec<u64>, Vec<u64>);

/// One family of cases: a label, a shape, chunk dimensions, and the maximum
/// shapes to try against them.
type Family<'a> = (&'a str, &'a [u64], &'a [u64], &'a [&'a [u64]]);

/// Every shape/chunk/maximum-shape combination the sweep covers, with a label
/// for the assertion messages.
///
/// Ranks 1 to 3, and for each the maximum shape equal to the shape, larger in
/// each dimension separately and in all of them, and unlimited in each
/// dimension separately. Two chunk geometries per rank: one that yields several
/// chunks, and one whose single chunk covers the shape — that second family is
/// what distinguishes "one chunk stored" from "one chunk possible", and writing
/// a single-chunk layout for the first made the reference library abort.
fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    let families: &[Family<'_>] = &[
        ("r1", &[7], &[3], &[&[7][..], &[16][..], &[U][..]]),
        ("r1 one chunk", &[3], &[4], &[&[3][..], &[8][..], &[U][..]]),
        (
            "r2",
            &[3, 3],
            &[2, 2],
            &[
                &[3, 3][..],
                &[8, 3][..],
                &[3, 8][..],
                &[8, 8][..],
                &[U, 3][..],
                &[3, U][..],
                &[U, 8][..],
                &[8, U][..],
            ],
        ),
        (
            "r2 one chunk",
            &[3, 3],
            &[4, 4],
            &[&[3, 3][..], &[8, 8][..], &[U, 3][..], &[3, U][..]],
        ),
        (
            "r3",
            &[2, 3, 4],
            &[1, 2, 2],
            &[
                &[2, 3, 4][..],
                &[4, 3, 4][..],
                &[2, 6, 4][..],
                &[2, 3, 8][..],
                &[4, 6, 8][..],
                &[U, 3, 4][..],
                &[2, U, 4][..],
                &[2, 3, U][..],
                &[U, 6, 8][..],
                &[4, 6, U][..],
            ],
        ),
    ];
    for (family, shape, chunks, maxshapes) in families {
        for ms in *maxshapes {
            out.push((
                format!("{family} shape={shape:?} chunks={chunks:?} max={ms:?}"),
                shape.to_vec(),
                chunks.to_vec(),
                ms.to_vec(),
            ));
        }
    }
    out
}

/// `1..=n` for a shape of `n` elements: every value distinct, so a chunk landing
/// at the wrong offset shows up as a wrong value rather than a coincidence.
fn values(shape: &[u64]) -> Vec<u32> {
    (1..=shape.iter().product::<u64>() as u32).collect()
}

fn pure_write(path: &std::path::Path, shape: &[u64], chunks: &[u64], maxshape: &[u64]) {
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_u32_data(&values(shape))
        .with_shape(shape)
        .with_maxshape(maxshape)
        .with_chunks(chunks);
    b.write(path).unwrap();
}

/// Write through the reference C library, pinned to the latest format so it
/// chooses a Fixed or Extensible Array.
///
/// Without the bound it writes a version-1 B-tree, whose keys carry each chunk's
/// coordinates explicitly — so it exercises none of the positional numbering
/// this file is about, and a read crosscheck against it passes no matter what.
fn c_write(path: &std::path::Path, shape: &[u64], chunks: &[u64], maxshape: &[u64]) {
    let file = hdf5::FileBuilder::new()
        .with_fapl(|fp| fp.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
        .create(path)
        .unwrap();
    let ds = file
        .new_dataset::<u32>()
        .chunk(chunks.iter().map(|&d| d as usize).collect::<Vec<_>>())
        .shape(
            shape
                .iter()
                .zip(maxshape)
                .map(|(&s, &m)| hdf5::Extent::new(s as usize, (m != U).then_some(m as usize)))
                .collect::<Vec<_>>(),
        )
        .create("d")
        .unwrap();
    ds.write_raw(&values(shape)).unwrap();
    // The dataset handle keeps the file open; a close with it still alive
    // leaves the object header unflushed and the file unreadable.
    drop(ds);
    file.close().unwrap();
}

/// Whether the reference library will create this geometry at all.
///
/// It refuses a chunk larger than the dataset's current extent ("Chunk
/// dimensions exceed data shape"), where this crate accepts one and stores a
/// single partly-empty chunk. That drops both one-chunk families from the two
/// directions that need the C library to *write*; the case they exist for —
/// one chunk stored where the maximum shape allows more — is asserted against a
/// C *read* by
/// `one_stored_chunk_with_room_for_more_is_not_the_single_chunk_layout`, which
/// is the direction where getting it wrong aborts the process.
fn c_library_accepts(shape: &[u64], chunks: &[u64]) -> bool {
    chunks.iter().zip(shape).all(|(&c, &s)| c <= s)
}

fn c_read(path: &std::path::Path) -> Vec<u32> {
    let file = hdf5::File::open(path).unwrap();
    let v = file.dataset("d").unwrap().read_raw::<u32>().unwrap();
    file.close().unwrap();
    v
}

fn pure_read(path: &std::path::Path) -> Vec<u32> {
    let file = File::open(path).unwrap();
    file.dataset("d").unwrap().read_u32().unwrap()
}

/// The pure writer's files must read back correctly *through the C library*.
///
/// The pure reader is asserted too, but it is the weaker half: it agreed with
/// the broken writer, which is exactly how the defect survived 2,000 tests.
#[test]
fn c_reads_every_maxshape_this_crate_writes() {
    let dir = tempdir().unwrap();
    for (label, shape, chunks, maxshape) in cases() {
        let path = dir.path().join("pure.h5");
        pure_write(&path, &shape, &chunks, &maxshape);
        assert_eq!(pure_read(&path), values(&shape), "pure read: {label}");
        assert_eq!(c_read(&path), values(&shape), "C read: {label}");
        std::fs::remove_file(&path).unwrap();
    }
}

/// The other direction: what the C library writes with a positional index, this
/// crate must read.
#[test]
fn pure_reads_every_maxshape_the_c_library_writes() {
    let dir = tempdir().unwrap();
    let mut skipped = 0;
    for (label, shape, chunks, maxshape) in cases() {
        if !c_library_accepts(&shape, &chunks) {
            skipped += 1;
            continue;
        }
        let path = dir.path().join("c.h5");
        c_write(&path, &shape, &chunks, &maxshape);
        assert_eq!(pure_read(&path), values(&shape), "{label}");
        std::fs::remove_file(&path).unwrap();
    }
    // A filter that quietly matched everything would leave this test passing
    // over an empty sweep, so the number it drops is asserted rather than
    // assumed.
    assert_eq!(skipped, 7, "only the one-chunk families are skipped");
}

/// The index *kind* has to match the reference library's choice too, and it is
/// chosen from the maximum grid rather than from the chunks stored.
///
/// Values alone do not pin this down: a single-chunk layout resolves chunk 0 to
/// the right address, so a dataset of one chunk that could hold four reads
/// correctly and still trips `H5D__single_idx_get_addr`'s assertion — the
/// reference library aborts the reading process rather than returning anything
/// at all. Comparing the kind against a C-written file of the same geometry is
/// what states the rule.
#[test]
fn the_index_kind_matches_the_reference_librarys_choice() {
    let dir = tempdir().unwrap();
    let mut skipped = 0;
    for (label, shape, chunks, maxshape) in cases() {
        if !c_library_accepts(&shape, &chunks) {
            skipped += 1;
            continue;
        }
        let pure_path = dir.path().join("pure.h5");
        let c_path = dir.path().join("c.h5");
        pure_write(&pure_path, &shape, &chunks, &maxshape);
        c_write(&c_path, &shape, &chunks, &maxshape);

        let kind = |p: &std::path::Path| {
            File::open(p)
                .unwrap()
                .dataset("d")
                .unwrap()
                .chunk_index()
                .unwrap()
        };
        assert_eq!(kind(&pure_path), kind(&c_path), "{label}");

        std::fs::remove_file(&pure_path).unwrap();
        std::fs::remove_file(&c_path).unwrap();
    }
    assert_eq!(skipped, 7, "only the one-chunk families are skipped");
}

/// A single chunk that the maximum shape lets grow into a second is a Fixed
/// Array, not the single-chunk layout its one chunk suggests.
///
/// Spelled out on its own because it is the case with a *crash* on the other
/// side rather than a wrong value, and because it is the one an "is it one
/// chunk?" test would get wrong in the obvious way.
#[test]
fn one_stored_chunk_with_room_for_more_is_not_the_single_chunk_layout() {
    let dir = tempdir().unwrap();
    let fixed = dir.path().join("fixed.h5");
    let growable = dir.path().join("growable.h5");
    pure_write(&fixed, &[3, 3], &[4, 4], &[3, 3]);
    pure_write(&growable, &[3, 3], &[4, 4], &[8, 8]);

    let kind = |p: &std::path::Path| {
        File::open(p)
            .unwrap()
            .dataset("d")
            .unwrap()
            .chunk_index()
            .unwrap()
    };
    assert_eq!(kind(&fixed), Some(ChunkIndex::SingleChunk));
    assert_eq!(kind(&growable), Some(ChunkIndex::FixedArray));
    assert_eq!(c_read(&growable), values(&[3, 3]));
}

/// Two unlimited dimensions are refused at write time.
///
/// The reference library indexes that dataspace with a version-2 B-tree, which
/// this crate does not write; producing an Extensible Array instead gave a file
/// `H5Dopen2` rejects outright ("already found unlimited dimension"), readable
/// by nothing but this crate.
#[test]
fn two_unlimited_dimensions_are_refused() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("two.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_u32_data(&values(&[3, 3]))
        .with_shape(&[3, 3])
        .with_maxshape(&[U, U])
        .with_chunks(&[2, 2]);
    let err = b.write(&path).unwrap_err();
    assert!(format!("{err}").contains("at most one dimension"), "{err}");
}

/// A maximum shape enormously wider than the shape is refused rather than
/// written.
///
/// A Fixed Array declares a slot for every chunk the maximum allows, so this
/// dataset of four chunks would otherwise ask for an index of 10^12 elements.
/// The refusal names the unlimited maximum as the thing to use instead.
#[test]
fn a_maximum_shape_needing_an_enormous_index_is_refused() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("huge.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_u32_data(&values(&[3, 3]))
        .with_shape(&[3, 3])
        .with_maxshape(&[2_000_000, 2_000_000])
        .with_chunks(&[2, 2]);
    let err = b.write(&path).unwrap_err();
    let text = format!("{err}");
    assert!(text.contains("chunk index"), "{text}");
    assert!(text.contains("unlimited"), "{text}");

    // The same growth expressed as an unlimited dimension is accepted, and is
    // what the refusal points the caller at.
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_u32_data(&values(&[3, 3]))
        .with_shape(&[3, 3])
        .with_maxshape(&[U, 2_000_000])
        .with_chunks(&[2, 2]);
    b.write(&path).unwrap();
    assert_eq!(c_read(&path), values(&[3, 3]));
}

/// A resizable multi-dimensional dataset still reads correctly after it is
/// actually grown, which is the point of declaring a maximum shape at all.
///
/// The sweep above never resizes, so it would pass over a numbering that is
/// right for the original extent and wrong for the extended one.
#[test]
fn a_grown_multidimensional_dataset_reads_back_through_both_libraries() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("grown.h5");

    // Written by the C library, since this crate's own resize is rank-1 only.
    let file = hdf5::FileBuilder::new()
        .with_fapl(|fp| fp.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
        .create(&path)
        .unwrap();
    let ds = file
        .new_dataset::<u32>()
        .chunk([2usize, 2])
        .shape([hdf5::Extent::new(3, None), hdf5::Extent::new(3, Some(8))])
        .create("d")
        .unwrap();
    ds.write_raw(&values(&[3, 3])).unwrap();
    ds.resize([6usize, 3]).unwrap();
    ds.write_raw(&(1u32..=18).collect::<Vec<_>>()).unwrap();
    drop(ds);
    file.close().unwrap();

    assert_eq!(pure_read(&path), (1u32..=18).collect::<Vec<_>>());
}

/// `repack` rewrites a dataset's chunks verbatim and rebuilds its index from
/// chunk sizes alone, through a second planner that has to number the slots the
/// same way the encoding path does.
///
/// The sweep above never reaches that planner: it plans from a shape it is
/// handed rather than from a dataset it encodes, and it was the one call site
/// that had no shape to number over at all.
#[test]
fn repack_preserves_a_resizable_multidimensional_dataset() {
    let dir = tempdir().unwrap();
    let src = dir.path().join("src.h5");
    let dst = dir.path().join("dst.h5");

    // Maximum shape wider in the *trailing* dimension, so the slot numbering
    // differs from dense grid order: chunks land in slots 0, 1, 4 and 5.
    let data: Vec<u32> = (1..=64).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_u32_data(&data)
        .with_shape(&[8, 8])
        .with_maxshape(&[8, 16])
        .with_chunks(&[4, 4])
        .with_deflate(6);
    b.write(&src).unwrap();

    hdf5_pure::repack(&src, &dst, &hdf5_pure::RepackOptions::new()).unwrap();

    assert_eq!(pure_read(&dst), data, "pure read of the repacked file");
    assert_eq!(c_read(&dst), data, "C read of the repacked file");
}

/// An in-place overwrite whose re-encoded chunks shrink rebuilds the index where
/// it sits, to record the new stored sizes. That rebuild has to put each chunk
/// back in the slot it came from.
///
/// The chunks are enumerated in dense grid order, so a rebuild that numbered
/// them densely would move every chunk past the first row — silently, since the
/// rebuilt index is the same length either way.
#[test]
fn a_shrinking_inplace_overwrite_keeps_each_chunk_in_its_slot() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("shrink.h5");

    // Incompressible first, so the chunk slots are large.
    let noisy: Vec<u32> = (0..64u32)
        .map(|i| i.wrapping_mul(2_654_435_761) ^ (i << 7))
        .collect();
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_u32_data(&noisy)
        .with_shape(&[8, 8])
        .with_maxshape(&[8, 16])
        .with_chunks(&[4, 4])
        .with_deflate(6);
    b.write(&path).unwrap();
    let size_before = std::fs::metadata(&path).unwrap().len();

    // Distinct values that still deflate to less than the noise did, so every
    // chunk fits its slot with room to spare and the index is rebuilt in place.
    let tidy: Vec<u32> = (1..=64).collect();
    {
        let session = hdf5_pure::File::open_rw(&path).unwrap();
        session
            .dataset("d")
            .unwrap()
            .write_staged(|b| {
                b.with_u32_data(&tidy).with_shape(&[8, 8]);
            })
            .unwrap();
        session.commit().unwrap();
    }
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        size_before,
        "the overwrite must fit its slots, or it relocates and tests a different path"
    );

    assert_eq!(pure_read(&path), tidy);
    assert_eq!(c_read(&path), tidy);
}
