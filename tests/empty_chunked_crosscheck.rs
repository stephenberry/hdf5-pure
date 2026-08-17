// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit little-endian targets; skip them elsewhere so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
//! An empty (zero-element) chunked dataset, checked against the reference C
//! library.
//!
//! This crate allocates a chunk index eagerly, where the C library allocates
//! lazily and leaves the layout message's address undefined until the first
//! chunk is written. Both are valid encodings of "nothing stored yet", and the
//! two writer defects these tests pin are both cases where the eager encoding
//! was built from information a chunk-less dataset does not have: an index
//! element sized from a chunk that was never written, and a Fixed Array
//! declaring zero entries where the C library writes no index at all.
//!
//! The in-place edit engine's empty chunked datasets are held to the same
//! standard here: they have to be indistinguishable from the whole-file
//! writer's, which means the C library reads, opens the index of, and grows
//! them.

use hdf5::Extent;
use hdf5::file::LibraryVersion;
use hdf5_pure::{File, FileBuilder};
use tempfile::tempdir;

/// The whole-file writer's empty chunked datasets, which the edit engine's are
/// meant to be indistinguishable from. (Only the 1.10+ output is exercised:
/// chunked storage needs the version 4 layout message, so the 1.8 format refuses
/// it outright.)
#[test]
fn c_reads_the_whole_file_writers_empty_chunked_datasets() {
    for unlimited in [false, true] {
        let dir = tempdir().unwrap();
        let path = dir.path().join("whole.h5");
        let mut b = FileBuilder::new();
        {
            let ds = b
                .create_dataset("col")
                .with_i64_data(&[])
                .with_shape(&[0])
                .with_chunks(&[512]);
            if unlimited {
                ds.with_maxshape(&[u64::MAX]);
            }
        }
        b.write(&path).unwrap();

        let file = hdf5::File::open(&path).unwrap();
        let ds = file.dataset("col").unwrap();
        assert_eq!(ds.shape(), vec![0], "unlimited={unlimited}");
        assert!(
            ds.read_raw::<i64>().unwrap().is_empty(),
            "unlimited={unlimited}"
        );
    }
}

/// The same property for the whole-file writer, at every rank and both ways of
/// arriving at zero elements. `num_chunks` is the assertion that opens the index.
#[test]
fn c_opens_the_chunk_index_of_every_empty_chunked_dataset_we_write() {
    let cases: &[(&str, &[u64], &[u64], bool)] = &[
        ("1-D fixed", &[0], &[64], false),
        ("1-D unlimited", &[0], &[64], true),
        ("2-D fixed, leading zero", &[0, 3], &[4, 3], false),
        ("2-D unlimited, leading zero", &[0, 3], &[4, 3], true),
        ("2-D fixed, inner zero", &[4, 0], &[2, 8], false),
        ("3-D fixed", &[0, 2, 3], &[4, 2, 3], false),
    ];
    for &(label, shape, chunks, unlimited) in cases {
        let dir = tempdir().unwrap();
        let path = dir.path().join("w.h5");
        let mut b = FileBuilder::new();
        {
            let ds = b
                .create_dataset("col")
                .with_i64_data(&[])
                .with_shape(shape)
                .with_chunks(chunks);
            if unlimited {
                // Only the leading dimension: an Extensible Array indexes
                // exactly one unlimited dimension, and the C library refuses to
                // open a dataset that declares two.
                let mut ms = shape.to_vec();
                ms[0] = u64::MAX;
                ds.with_maxshape(&ms);
            }
        }
        b.write(&path).unwrap();

        let file = hdf5::File::open(&path).unwrap();
        let ds = file.dataset("col").unwrap();
        assert!(ds.is_chunked(), "[{label}] chunked storage");
        let expected_shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        assert_eq!(ds.shape(), expected_shape, "[{label}] shape");
        assert_eq!(
            ds.num_chunks(),
            Some(0),
            "[{label}] the C library must be able to open the chunk index"
        );
        assert!(ds.read_raw::<i64>().unwrap().is_empty(), "[{label}] read");
    }
}

/// A filtered dataset created empty has no chunk to size its index element from,
/// and the width it declares has to be the one its *first* chunk will need — the
/// reference C library encodes to whatever the header says.
///
/// Sized deliberately past 64 KiB of incompressible data per chunk, which is
/// exactly what the old 2-byte field could not hold: the C library wrote chunks
/// the field could not address, and the result read back as a truncated deflate
/// stream. The chunk geometry is the same in both files, so the widths must
/// match; asserted as equality rather than as a literal, since the rule is that
/// the width tracks the geometry, not that it takes a particular value.
#[test]
fn an_empty_filtered_dataset_declares_the_width_its_first_chunk_needs() {
    /// The `EAHD` element-size byte: signature (4), version (1), client id (1).
    fn ea_element_size(path: &std::path::Path) -> u8 {
        let b = std::fs::read(path).unwrap();
        let h = (0..b.len() - 4)
            .find(|&i| &b[i..i + 4] == b"EAHD")
            .expect("an extensible array header");
        b[h + 6]
    }

    // xorshift, so deflate cannot shrink a chunk below the old field's ceiling.
    let mut state = 0x1234_5678u32;
    let chunk_elems = 32_768usize; // 128 KiB per chunk
    let data: Vec<i32> = (0..chunk_elems)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as i32
        })
        .collect();

    let dir = tempdir().unwrap();
    let empty = dir.path().join("empty.h5");
    let seeded = dir.path().join("seeded.h5");
    let write = |path: &std::path::Path, values: &[i32]| {
        let mut b = FileBuilder::new();
        b.create_dataset("col")
            .with_i32_data(values)
            .with_shape(&[values.len() as u64])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[chunk_elems as u64])
            .with_deflate(6);
        b.write(path).unwrap();
    };
    write(&empty, &[]);
    write(&seeded, &data);
    assert_eq!(
        ea_element_size(&empty),
        ea_element_size(&seeded),
        "an empty filtered dataset must declare the same element width as the \
         identical one that already holds a chunk"
    );

    // The consequence, end to end: the C library fills the dataset the edit
    // engine declared, and this crate reads back what it wrote.
    {
        let file = hdf5::File::open_rw(&empty).unwrap();
        let ds = file.dataset("col").unwrap();
        ds.resize((chunk_elems,)).unwrap();
        ds.write(&data).unwrap();
        drop(ds);
        file.close().unwrap();
    }
    let file = File::open(&empty).unwrap();
    assert_eq!(file.dataset("col").unwrap().read_i32().unwrap(), data);
}

/// And the same dataset grown by *this* crate rather than the C library. The
/// in-place append writes into the index the empty dataset declared, so a width
/// that does not fit the chunk is refused outright — a dataset that should have
/// been appendable, refused because of how it was created.
#[test]
fn an_empty_filtered_dataset_accepts_an_append_that_fills_its_chunk() {
    let mut state = 0x9E37_79B9u32;
    let chunk_elems = 32_768usize;
    let data: Vec<i32> = (0..chunk_elems)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as i32
        })
        .collect();

    let dir = tempdir().unwrap();
    let path = dir.path().join("append.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("col")
        .with_i32_data(&[])
        .with_shape(&[0])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[chunk_elems as u64])
        .with_deflate(6);
    b.write(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session.dataset("col").unwrap().append(&data).unwrap();
    }
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("col").unwrap().read_i32().unwrap(), data);
    drop(file);
    let file = hdf5::File::open(&path).unwrap();
    assert_eq!(
        file.dataset("col").unwrap().read_raw::<i32>().unwrap(),
        data
    );
}

/// Write a starter file and add one empty chunked dataset to it through the
/// in-place edit engine, the path issue #284 refused.
fn edit_in_an_empty_chunked_dataset(path: &std::path::Path, unlimited: bool) {
    let mut b = FileBuilder::new();
    b.create_dataset("existing").with_f64_data(&[1.0, 2.0]);
    b.write(path).unwrap();

    let session = File::open_rw(path).unwrap();
    session
        .root()
        .create_dataset("col", |b| {
            b.with_i64_data(&[]).with_shape(&[0]).with_chunks(&[512]);
            if unlimited {
                b.with_maxshape(&[u64::MAX]);
            }
        })
        .unwrap();
    session.commit().unwrap();
}

/// The issue's round trip: the edit engine declares the column empty, then the C
/// library reads it, extends it, and writes into it — and this crate reads back
/// what the C library wrote. An index built over zero chunks that the C library
/// could read but not grow would pass a read-only check and fail here.
#[test]
fn c_grows_an_empty_chunked_dataset_added_in_place() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("edited.h5");
    edit_in_an_empty_chunked_dataset(&path, true);

    {
        let file = hdf5::File::open_rw(&path).unwrap();
        let ds = file.dataset("col").unwrap();
        assert_eq!(ds.shape(), vec![0]);
        assert!(ds.read_raw::<i64>().unwrap().is_empty());
        ds.resize((4,)).unwrap();
        ds.write(&[10i64, 20, 30, 40]).unwrap();
        drop(ds);
        file.close().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("col").unwrap().read_i64().unwrap(),
        vec![10, 20, 30, 40]
    );
    // The pre-existing dataset the edit appended past is untouched.
    assert_eq!(
        file.dataset("existing").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0]
    );
}

/// The same dataset without `maxshape`: a fixed-shape empty chunked dataset,
/// whose index is a fixed array over zero chunks rather than an extensible one.
/// The C library has to accept that too, or the edit engine is emitting a
/// structure only its own reader understands.
#[test]
fn c_reads_a_fixed_shape_empty_chunked_dataset_added_in_place() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("edited_fixed.h5");
    edit_in_an_empty_chunked_dataset(&path, false);

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("col").unwrap();
    assert_eq!(ds.shape(), vec![0]);
    assert!(ds.read_raw::<i64>().unwrap().is_empty());
    // Shape and an empty read are equally true of a *contiguous* dataset, so
    // they cannot tell whether the edit engine emitted chunked storage at all.
    // These reach the C library's own view of the chunk index, which is the
    // thing under test.
    assert!(ds.is_chunked(), "the C library must see chunked storage");
    assert_eq!(ds.chunk(), Some(vec![512]));
    // `num_chunks` opens the index. A Fixed Array declaring zero entries — what
    // this crate wrote before it adopted the C library's convention of leaving a
    // chunk-less fixed-shape dataset unallocated — makes this call *fail*, while
    // reads and `shape` keep working. It is the only assertion here that can see
    // the difference.
    assert_eq!(
        ds.num_chunks(),
        Some(0),
        "the C library must be able to open the chunk index"
    );
}

/// The other direction, and the one this crate used to fail: the C library
/// allocates a chunked dataset's index lazily, so an empty one carries the
/// undefined address. Reading it is the empty buffer — `read_raw` errored with
/// "no address for chunked layout" until the reader learned to tell "nothing
/// stored because there is nothing to store" from "nothing stored where data
/// must be".
#[test]
fn pure_reads_the_c_librarys_unallocated_empty_chunked_datasets() {
    for unlimited in [false, true] {
        for latest in [false, true] {
            let dir = tempdir().unwrap();
            let path = dir.path().join("c.h5");
            {
                let mut opts = hdf5::File::with_options();
                if latest {
                    opts.with_fapl(|p| {
                        p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest())
                    });
                }
                let file = opts.create(&path).unwrap();
                let builder = file.new_dataset::<i64>().chunk((512,));
                let ds = if unlimited {
                    builder.shape((Extent::resizable(0),)).create("col")
                } else {
                    builder.shape((0,)).create("col")
                }
                .unwrap();
                drop(ds);
                file.close().unwrap();
            }

            let label = format!("unlimited={unlimited} latest={latest}");
            // Both readers: the buffered one and the streaming one reach
            // different functions, and only one of them was covered when this
            // was written.
            let file = File::open(&path).unwrap();
            let ds = file.dataset("col").unwrap();
            assert_eq!(ds.shape().unwrap(), vec![0], "[{label}] buffered");
            assert_eq!(
                ds.read_i64().unwrap(),
                Vec::<i64>::new(),
                "[{label}] buffered"
            );
            drop(file);

            let file = File::open_streaming(&path).unwrap();
            let ds = file.dataset("col").unwrap();
            assert_eq!(ds.shape().unwrap(), vec![0], "[{label}] streaming");
            assert_eq!(
                ds.read_i64().unwrap(),
                Vec::<i64>::new(),
                "[{label}] streaming"
            );
        }
    }
}
