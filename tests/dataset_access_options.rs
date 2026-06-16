//! Per-dataset access options (DAPL): the `H5Pset_chunk_cache` analogue, which
//! overrides the file-wide `H5Pset_cache`-style chunk-cache default for a single
//! dataset. See issue #48.

use hdf5_pure::{ChunkCacheConfig, DatasetAccessOptions, File, FileAccessOptions, FileBuilder};

/// Write a small chunked dataset, plus one nested in a group, and return the
/// file path (kept alive by the returned `TempDir`).
fn write_fixture() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dapl.h5");
    let data: Vec<i32> = (0..256).collect();
    {
        let mut b = FileBuilder::new();
        b.create_dataset("chunked")
            .with_i32_data(&data)
            .with_shape(&[256])
            .with_chunks(&[32]);
        let mut g = b.create_group("grp");
        g.create_dataset("inner")
            .with_i32_data(&data)
            .with_shape(&[256])
            .with_chunks(&[32]);
        b.add_group(g.finish());
        b.write(&path).unwrap();
    }
    (dir, path)
}

#[test]
fn dataset_without_options_inherits_file_wide_default() {
    let (_dir, path) = write_fixture();
    let file_default = ChunkCacheConfig::from_h5p_cache(64, 128 * 1024);
    let file = File::open_with_options(
        &path,
        FileAccessOptions::new().with_chunk_cache(file_default),
    )
    .unwrap();

    // `dataset` and `dataset_with_options(new())` both inherit the file default.
    assert_eq!(
        file.dataset("chunked").unwrap().chunk_cache_config(),
        file_default
    );
    assert_eq!(
        file.dataset_with_options("chunked", DatasetAccessOptions::new())
            .unwrap()
            .chunk_cache_config(),
        file_default
    );
}

#[test]
fn per_dataset_override_takes_precedence_over_file_default() {
    let (_dir, path) = write_fixture();
    let file_default = ChunkCacheConfig::new();
    let override_cfg = ChunkCacheConfig::from_h5p_cache(8, 16 * 1024).with_index_cache(false);
    assert_ne!(file_default, override_cfg);

    let file = File::open_with_options(
        &path,
        FileAccessOptions::new().with_chunk_cache(file_default),
    )
    .unwrap();

    let options = DatasetAccessOptions::new().with_chunk_cache(override_cfg);
    assert_eq!(options.chunk_cache(), Some(override_cfg));

    let ds = file.dataset_with_options("chunked", options).unwrap();
    assert_eq!(ds.chunk_cache_config(), override_cfg);

    // A sibling handle opened without the override still sees the file default,
    // proving the override is scoped to the one dataset handle.
    let sibling = file.dataset("chunked").unwrap();
    assert_eq!(sibling.chunk_cache_config(), file_default);
}

#[test]
fn override_with_disabled_cache_still_reads_correct_data() {
    let (_dir, path) = write_fixture();
    let expected: Vec<i32> = (0..256).collect();
    let file = File::open(&path).unwrap();

    let ds = file
        .dataset_with_options(
            "chunked",
            DatasetAccessOptions::new().with_chunk_cache(ChunkCacheConfig::disabled()),
        )
        .unwrap();
    assert_eq!(ds.chunk_cache_config(), ChunkCacheConfig::disabled());
    // Read twice: with the cache disabled every read re-fetches, and both must
    // still produce identical, correct data.
    assert_eq!(ds.read_i32().unwrap(), expected);
    assert_eq!(ds.read_i32().unwrap(), expected);
}

#[test]
fn group_dataset_with_options_overrides_chunk_cache() {
    let (_dir, path) = write_fixture();
    let expected: Vec<i32> = (0..256).collect();
    let override_cfg = ChunkCacheConfig::from_h5p_cache(4, 8 * 1024);
    let file = File::open(&path).unwrap();
    let group = file.group("grp").unwrap();

    let inherited = group.dataset("inner").unwrap();
    assert_eq!(
        inherited.chunk_cache_config(),
        file.access_options().chunk_cache()
    );

    let overridden = group
        .dataset_with_options("inner", DatasetAccessOptions::new().with_chunk_cache(override_cfg))
        .unwrap();
    assert_eq!(overridden.chunk_cache_config(), override_cfg);
    assert_eq!(overridden.read_i32().unwrap(), expected);
}

#[test]
fn streaming_backend_honors_per_dataset_override() {
    let (_dir, path) = write_fixture();
    let expected: Vec<i32> = (0..256).collect();
    let override_cfg = ChunkCacheConfig::disabled();
    let file = File::open_streaming(&path).unwrap();

    let ds = file
        .dataset_with_options(
            "chunked",
            DatasetAccessOptions::new().with_chunk_cache(override_cfg),
        )
        .unwrap();
    assert_eq!(ds.chunk_cache_config(), override_cfg);
    assert_eq!(ds.read_i32().unwrap(), expected);
    assert_eq!(ds.read_i32().unwrap(), expected);
}

#[test]
fn default_options_equal_new() {
    assert_eq!(DatasetAccessOptions::default(), DatasetAccessOptions::new());
    assert_eq!(DatasetAccessOptions::new().chunk_cache(), None);
}
