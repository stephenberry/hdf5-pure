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
