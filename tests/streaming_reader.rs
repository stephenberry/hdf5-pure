//! End-to-end tests for `File::open_streaming` (issue #27).
//!
//! Writes a real HDF5 file, then reads every dataset both via the buffered
//! `File::open` and the lazy `File::open_streaming`, asserting identical
//! results. This exercises the whole streaming stack end to end: superblock
//! detection, v2 group path resolution, object-header parsing, and contiguous /
//! Fixed-Array / Extensible-Array chunked data reads — all from a `Read + Seek`
//! source that never buffers the whole file.

use hdf5_pure::{File, FileBuilder};

#[test]
fn open_streaming_matches_buffered() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("streaming.h5");

    let contig: Vec<f64> = (0..100).map(|i| i as f64 * 1.5).collect();
    let fixed_chunked: Vec<i32> = (0..1000).collect();
    let unlimited_chunked: Vec<i32> = (0..500).map(|i| i * 3).collect();
    let inner: Vec<f64> = vec![10.0, 20.0, 30.0];

    {
        let mut b = FileBuilder::new();
        b.create_dataset("contig")
            .with_f64_data(&contig)
            .with_shape(&[100]);
        // Fixed shape + chunks -> Fixed Array (or implicit) chunk index.
        b.create_dataset("fixed_chunked")
            .with_i32_data(&fixed_chunked)
            .with_shape(&[1000])
            .with_chunks(&[64]);
        // Unlimited dimension -> Extensible Array chunk index.
        b.create_dataset("unlimited_chunked")
            .with_i32_data(&unlimited_chunked)
            .with_shape(&[500])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[64]);
        // Nested group + dataset exercises v2 group path resolution.
        let mut g = b.create_group("grp");
        g.create_dataset("inner")
            .with_f64_data(&inner)
            .with_shape(&[3]);
        b.add_group(g.finish());
        b.write(&path).unwrap();
    }

    let buffered = File::open(&path).unwrap();
    let streaming = File::open_streaming(&path).unwrap();

    for name in ["contig", "fixed_chunked", "unlimited_chunked", "grp/inner"] {
        let b_shape = buffered.dataset(name).unwrap().shape().unwrap();
        let s_shape = streaming.dataset(name).unwrap().shape().unwrap();
        assert_eq!(b_shape, s_shape, "shape mismatch for {name}");
    }

    // Contiguous f64.
    assert_eq!(
        streaming.dataset("contig").unwrap().read_f64().unwrap(),
        contig
    );
    // Fixed-Array-indexed chunked i32.
    assert_eq!(
        streaming
            .dataset("fixed_chunked")
            .unwrap()
            .read_i32()
            .unwrap(),
        fixed_chunked
    );
    // Extensible-Array-indexed chunked i32.
    assert_eq!(
        streaming
            .dataset("unlimited_chunked")
            .unwrap()
            .read_i32()
            .unwrap(),
        unlimited_chunked
    );
    // Dataset reached through a nested group.
    assert_eq!(
        streaming.dataset("grp/inner").unwrap().read_f64().unwrap(),
        inner
    );

    // And the streaming reads match the buffered reads byte-for-byte.
    assert_eq!(
        buffered.dataset("contig").unwrap().read_f64().unwrap(),
        streaming.dataset("contig").unwrap().read_f64().unwrap()
    );
    assert_eq!(
        buffered
            .dataset("unlimited_chunked")
            .unwrap()
            .read_i32()
            .unwrap(),
        streaming
            .dataset("unlimited_chunked")
            .unwrap()
            .read_i32()
            .unwrap()
    );
}
