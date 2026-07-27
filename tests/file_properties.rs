//! One `FileAccessProperties` (`fapl`) for every open, and `FileCreateProperties`
//! (`fcpl`) as a reusable creation-property value (issues #204, #205).

use hdf5_pure::{
    ChunkCacheConfig, File, FileAccessProperties, FileBuilder, FileCreateProperties, FileLocking,
    FileSpaceStrategy,
};
use tempfile::tempdir;

fn write_chunked(path: &std::path::Path) {
    let mut b = FileBuilder::new();
    b.create_dataset("data")
        .with_f64_data(&(0..4096).map(|i| i as f64).collect::<Vec<_>>())
        .with_shape(&[4096])
        .with_chunks(&[256]);
    b.write(path).unwrap();
}

/// The whole point of #204: one options value configured once, then handed to
/// both a read path and a read-write path.
#[test]
fn one_options_value_serves_read_and_read_write_opens() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("shared.h5");
    write_chunked(&path);

    let options = FileAccessProperties::new()
        .with_chunk_cache(ChunkCacheConfig::new().with_max_bytes(1 << 20))
        .with_locking(FileLocking::Disabled);

    // Read path.
    {
        let file = File::open_with_options(&path, options).unwrap();
        assert_eq!(
            file.dataset("data").unwrap().read_f64().unwrap().len(),
            4096
        );
    }
    // Read-write path — the same value, which had no spelling before #204.
    {
        let file = File::open_rw_with_options(&path, options).unwrap();
        assert_eq!(
            file.dataset("data").unwrap().read_f64().unwrap().len(),
            4096
        );
    }
    // Bounded read-write path.
    {
        let file = File::open_rw_bounded_with_options(&path, options).unwrap();
        assert_eq!(
            file.dataset("data").unwrap().read_f64().unwrap().len(),
            4096
        );
    }
}

/// The mirror backend previously discarded access options entirely
/// (`from_rw_session` hardcoded the defaults), so a chunk cache configured for
/// `open_rw` did nothing. It now applies.
#[test]
fn open_rw_honors_the_configured_chunk_cache() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("cache.h5");
    write_chunked(&path);

    // Each handle holds an exclusive lock for its life, so scope them.
    {
        let enabled = FileAccessProperties::new()
            .with_chunk_cache(ChunkCacheConfig::new().with_max_bytes(1 << 20));
        let file = File::open_rw_with_options(&path, enabled).unwrap();
        let ds = file.dataset("data").unwrap();
        ds.read_f64().unwrap();
        let stats = ds.chunk_cache_stats();
        assert!(
            stats.cached_chunks() > 0,
            "an enabled chunk cache should retain chunks on the mirror backend, got {stats:?}"
        );
    }
    {
        let disabled = FileAccessProperties::new().with_chunk_cache(ChunkCacheConfig::disabled());
        let file = File::open_rw_with_options(&path, disabled).unwrap();
        let ds = file.dataset("data").unwrap();
        ds.read_f64().unwrap();
        assert_eq!(
            ds.chunk_cache_stats().cached_chunks(),
            0,
            "a disabled chunk cache must retain nothing"
        );
    }
}

/// The bounded backend hardcoded `FileLocking::Enabled`, so it could not be used
/// where OS locking is unavailable. The policy now comes from the options.
#[test]
fn open_rw_bounded_honors_the_locking_policy() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bounded_lock.h5");
    write_chunked(&path);

    let unlocked = FileAccessProperties::new().with_locking(FileLocking::Disabled);
    let _first = File::open_rw_bounded_with_options(&path, unlocked).unwrap();
    // Nothing is locked, so a second bounded open and a plain read both succeed.
    File::open_rw_bounded_with_options(&path, unlocked)
        .expect("a second Disabled bounded open should succeed: neither took a lock");
    File::open(&path).expect("a reader should succeed against a Disabled bounded writer");
}

#[test]
fn open_rw_bounded_still_locks_by_default() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bounded_default.h5");
    write_chunked(&path);

    let _held = File::open_rw_bounded(&path).unwrap();
    assert!(
        File::open_rw_bounded(&path).is_err(),
        "the default policy must still take an exclusive lock"
    );
}

/// #205: the same creation properties, applied as a value or one at a time, must
/// produce the identical file.
#[test]
fn create_options_match_the_individual_setters() {
    let options = FileCreateProperties::new()
        .with_userblock(512)
        .with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 8)
        .with_file_space_page_size(4096);

    let from_value = {
        let mut b = FileBuilder::new();
        b.with_create_properties(options);
        b.create_dataset("d").with_f64_data(&[1.0, 2.0]);
        b.finish().unwrap()
    };
    let from_setters = {
        let mut b = FileBuilder::new();
        b.with_userblock(512);
        b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 8);
        b.with_file_space_page_size(4096);
        b.create_dataset("d").with_f64_data(&[1.0, 2.0]);
        b.finish().unwrap()
    };
    assert_eq!(
        from_value, from_setters,
        "applying an fcpl value must equal setting each property individually"
    );
}

/// A layout defined once in a helper and reused, which is the composition #205
/// exists to enable.
#[test]
fn create_options_are_reusable_across_files() {
    fn shared_layout() -> FileCreateProperties {
        FileCreateProperties::new().with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 4)
    }

    let dir = tempdir().unwrap();
    for name in ["a.h5", "b.h5"] {
        let path = dir.path().join(name);
        let mut b = FileBuilder::new();
        b.with_create_properties(shared_layout());
        b.create_dataset("d").with_f64_data(&[1.0]);
        b.write(&path).unwrap();

        let file = File::open(&path).unwrap();
        let info = file
            .file_space_info()
            .expect("the strategy should be recorded in the superblock extension");
        assert_eq!(info.strategy, FileSpaceStrategy::FsmAggr);
        assert!(info.persist);
        assert_eq!(info.threshold, 4);
    }
}

/// `File::create` gained creation properties: they were previously unreachable
/// through the owned-handle path, which always got the defaults.
#[test]
fn create_with_options_records_creation_properties() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("created.h5");

    let file = File::create_with_options(
        &path,
        FileCreateProperties::new().with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 16),
        FileAccessProperties::new().with_locking(FileLocking::Disabled),
    )
    .unwrap();
    file.root().create_group("g").unwrap();
    file.commit().unwrap();
    drop(file);

    let file = File::open(&path).unwrap();
    assert!(file.group("g").is_ok());
    let info = file.file_space_info().expect("strategy should be recorded");
    assert_eq!(info.strategy, FileSpaceStrategy::FsmAggr);
    assert_eq!(info.threshold, 16);
}

#[test]
fn create_with_default_options_matches_create() {
    let dir = tempdir().unwrap();
    let a = dir.path().join("a.h5");
    let b = dir.path().join("b.h5");

    File::create(&a).unwrap().close().unwrap();
    File::create_with_options(&b, FileCreateProperties::new(), FileAccessProperties::new())
        .unwrap()
        .close()
        .unwrap();

    assert_eq!(
        std::fs::read(&a).unwrap(),
        std::fs::read(&b).unwrap(),
        "default options must reproduce the plain `create` file"
    );
}

/// An invalid creation property is reported when the file is written, not when
/// the options value is built — the value is inert data.
#[test]
fn invalid_creation_property_is_reported_at_write_time() {
    // A page size must be a power of two >= 512.
    let options = FileCreateProperties::new()
        .with_file_space_strategy(FileSpaceStrategy::Page, true, 1)
        .with_file_space_page_size(700);
    assert_eq!(
        options.file_space_page_size(),
        Some(700),
        "the value records what it was given, without validating it"
    );

    let mut b = FileBuilder::new();
    b.with_create_properties(options);
    b.create_dataset("d").with_f64_data(&[1.0]);
    assert!(
        b.finish().is_err(),
        "an illegal page size must be refused when the file is written"
    );
}

/// Pins the documented gap that a non-paged userblock size is **not** validated
/// against HDF5's power-of-two rule (see `docs/reference/property-support.md`).
/// Recorded so tightening it later is a deliberate change with a failing test,
/// not a silent one.
#[test]
fn non_paged_userblock_size_is_currently_unvalidated() {
    let mut b = FileBuilder::new();
    b.with_create_properties(FileCreateProperties::new().with_userblock(700));
    b.create_dataset("d").with_f64_data(&[1.0]);
    assert!(
        b.finish().is_ok(),
        "700 is not a power of two; if this now fails, the writer gained the \
         validation and the property-support reference needs updating"
    );
}

/// The 0.25.0 spellings still resolve, to the very same types, for one
/// deprecation cycle (#198). If an alias is dropped or repointed, this stops
/// compiling — which is the whole guarantee the rename promised.
#[test]
#[allow(deprecated)]
fn deprecated_aliases_resolve_to_the_renamed_types() {
    use hdf5_pure::{DatasetAccessOptions, FileAccessOptions, FileCreateOptions};

    // Same type, not merely a convertible one: assignment across the two
    // spellings only compiles if the alias is that exact type.
    let access: FileAccessProperties = FileAccessOptions::new().with_locking(FileLocking::Disabled);
    assert_eq!(access.locking(), FileLocking::Disabled);

    let create: FileCreateProperties = FileCreateOptions::new().with_userblock(512);
    assert_eq!(create.userblock(), 512);

    let dapl: hdf5_pure::DatasetAccessProperties =
        DatasetAccessOptions::new().with_chunk_cache(ChunkCacheConfig::new());
    assert!(dapl.chunk_cache().is_some());

    // The renamed builder method and its deprecated shim agree.
    let mut old = FileBuilder::new();
    old.with_create_options(create);
    let mut new = FileBuilder::new();
    new.with_create_properties(create);
}
