//! End-to-end tests for `File::open_streaming` (issue #27).
//!
//! Writes a real HDF5 file, then reads every dataset both via the buffered
//! `File::open` and the lazy `File::open_streaming`, asserting identical
//! results. This exercises the whole streaming stack end to end: superblock
//! detection, v2 group path resolution, object-header parsing, and contiguous /
//! Fixed-Array / Extensible-Array chunked data reads — all from a `Read + Seek`
//! source that never buffers the whole file.

use hdf5_pure::{ChunkCacheConfig, File, FileAccessOptions, FileBuilder, MetadataCacheConfig};

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

#[test]
fn open_streaming_with_access_options_reads_chunked_data() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("streaming_options.h5");
    let data: Vec<i32> = (0..256).map(|i| i * 2).collect();

    {
        let mut b = FileBuilder::new();
        b.create_dataset("chunked")
            .with_i32_data(&data)
            .with_shape(&[256])
            .with_chunks(&[32]);
        b.write(&path).unwrap();
    }

    let options = FileAccessOptions::new()
        .with_metadata_cache(MetadataCacheConfig::new(4096).with_max_entry_bytes(512))
        .with_chunk_cache(ChunkCacheConfig::disabled());
    let file = File::open_streaming_with_options(&path, options).unwrap();
    assert_eq!(file.access_options(), options);

    let dataset = file.dataset("chunked").unwrap();
    assert_eq!(dataset.read_i32().unwrap(), data);
    assert_eq!(dataset.read_i32().unwrap(), data);
}

/// Recursively assert the streaming backend reports the identical groups,
/// datasets, shapes, and attributes as the buffered backend for the group at
/// `path`. Returns the number of attributes compared, so a caller can confirm
/// the walk actually exercised attribute reads rather than passing trivially on
/// empty maps.
fn assert_group_parity(buffered: &File, streaming: &File, path: &str) -> usize {
    let display = if path.is_empty() { "/" } else { path };
    let bg = if path.is_empty() {
        buffered.root()
    } else {
        buffered.group(path).unwrap()
    };
    let sg = if path.is_empty() {
        streaming.root()
    } else {
        streaming.group(path).unwrap()
    };

    let b_attrs = bg.attrs().unwrap();
    let s_attrs = sg.attrs().unwrap();
    assert_eq!(b_attrs, s_attrs, "group attrs mismatch at '{display}'");
    let mut count = b_attrs.len();

    let mut b_ds = bg.datasets().unwrap();
    b_ds.sort();
    let mut s_ds = sg.datasets().unwrap();
    s_ds.sort();
    assert_eq!(b_ds, s_ds, "datasets mismatch at '{display}'");
    for name in &b_ds {
        let full = child_path(path, name);
        let bd = buffered.dataset(&full).unwrap();
        let sd = streaming.dataset(&full).unwrap();
        assert_eq!(
            bd.shape().unwrap(),
            sd.shape().unwrap(),
            "shape mismatch for '{full}'"
        );
        let bda = bd.attrs().unwrap();
        let sda = sd.attrs().unwrap();
        assert_eq!(bda, sda, "dataset attrs mismatch for '{full}'");
        count += bda.len();
    }

    let mut b_g = bg.groups().unwrap();
    b_g.sort();
    let mut s_g = sg.groups().unwrap();
    s_g.sort();
    assert_eq!(b_g, s_g, "subgroups mismatch at '{display}'");
    for name in &b_g {
        count += assert_group_parity(buffered, streaming, &child_path(path, name));
    }

    count
}

fn child_path(parent: &str, name: &str) -> String {
    if parent.is_empty() {
        name.to_string()
    } else {
        format!("{parent}/{name}")
    }
}

/// The streaming backend must resolve v1 (symbol-table) groups and read
/// compact, dense (fractal-heap), and variable-length attributes identically to
/// the buffered backend. Each fixture's expected attribute count guards against
/// a trivially-empty parity pass.
#[test]
fn streaming_matches_buffered_groups_and_attributes_across_fixtures() {
    // (fixture, expected total attributes across the whole walk)
    let cases = [
        ("two_groups.h5", 0),        // v1 groups, no attributes
        ("nested_groups.h5", 0),     // nested v1 groups
        ("simple_dataset.h5", 0),    // v1 root group + dataset
        ("attrs.h5", 4),             // v1 groups + compact attrs (root + dataset)
        ("mixed_attrs.h5", 3),       // v1 subgroup + scalar/array compact attrs
        ("vl_strings.h5", 1),        // v1 root + VL-string attr (global heap)
        ("dense_attrs.h5", 50),      // v2 dataset + dense (fractal-heap) attrs
        ("dense_attrs_root.h5", 20), // v2 root group + dense attrs
        ("v2_groups.h5", 0),         // v2 groups, no attributes
    ];

    for (fixture, expected_attrs) in cases {
        let path = format!("tests/fixtures/{fixture}");
        let buffered = File::open(&path).unwrap();
        let streaming = File::open_streaming(&path).unwrap();
        let counted = assert_group_parity(&buffered, &streaming, "");
        assert_eq!(
            counted, expected_attrs,
            "attribute count mismatch for {fixture}"
        );
    }
}
