//! `MemoryStrategy`: what a read-write open is allowed to spend on holding the
//! file, and how a caller finds out what it got (issue #198, step 3).
//!
//! The bounded engine cannot edit every file — a userblock still needs the
//! whole-file mirror — and the question these tests pin is what happens then.
//! The default refuses; `Auto` falls back; and either way the resulting `File`
//! says which one it is, because "bounded" is a memory guarantee and a caller
//! who was quietly given the mirror instead has no other way to notice.

use hdf5_pure::{File, FileAccessProperties, FileBuilder, FileSpaceStrategy, MemoryStrategy};

/// A latest-format file the bounded engine handles: no userblock, v2+ superblock.
fn plain_file(path: &std::path::Path) {
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&[1, 2, 3])
        .with_shape(&[3])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[3]);
    b.write(path).unwrap();
}

/// The same file behind a 512-byte userblock, which the bounded engine refuses.
fn userblock_file(path: &std::path::Path) {
    let mut b = FileBuilder::new();
    b.with_userblock(512);
    b.create_dataset("d")
        .with_i32_data(&[1, 2, 3])
        .with_shape(&[3])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[3]);
    b.write(path).unwrap();
}

#[test]
fn a_userblock_file_is_refused_under_the_default_strategy() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ub.h5");
    userblock_file(&path);

    // The default is `Bounded`, so this is also the behavior of a caller who
    // never mentions a strategy at all.
    let err = File::open_rw_bounded(&path).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("userblock"),
        "the refusal should name the userblock, got: {msg}"
    );

    let explicit = File::open_rw_bounded_with_options(
        &path,
        FileAccessProperties::new().with_memory_strategy(MemoryStrategy::Bounded),
    );
    assert!(
        explicit.is_err(),
        "spelling out the default must not change it"
    );
}

#[test]
fn auto_falls_back_to_the_mirror_and_says_so() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ub_auto.h5");
    userblock_file(&path);

    let file = File::open_rw_bounded_with_options(
        &path,
        FileAccessProperties::new().with_memory_strategy(MemoryStrategy::Auto),
    )
    .expect("Auto falls back to the mirror rather than refusing");

    // The whole point of the observable: `Auto` is a request, and this is how a
    // caller learns the fallback was taken and memory now scales with the file.
    assert_eq!(file.memory_strategy(), Some(MemoryStrategy::Mirrored));

    // And it is a real editing session, not just an open that succeeded.
    //
    // Through the *staged* path, because a userblock defeats the immediate
    // in-place append on either engine — the fallback buys the staged edit
    // surface, which is what the mirror has over the bounded engine here, not
    // in-place append. `Dataset::append` still refuses, naming `append_staged`.
    let mut ds = file.dataset("d").unwrap();
    assert!(
        ds.append(&[4i32]).is_err(),
        "in-place append needs no userblock"
    );
    ds.append_staged(|b| {
        b.append_i32(&[4, 5]);
    })
    .unwrap();
    file.close().unwrap();

    let reopened = File::open(&path).unwrap();
    assert_eq!(
        reopened.dataset("d").unwrap().read_i32().unwrap(),
        vec![1, 2, 3, 4, 5]
    );
}

#[test]
fn auto_keeps_the_bounded_engine_when_the_file_allows_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plain_auto.h5");
    plain_file(&path);

    let file = File::open_rw_bounded_with_options(
        &path,
        FileAccessProperties::new().with_memory_strategy(MemoryStrategy::Auto),
    )
    .unwrap();

    // `Auto` is "prefer bounded", not "mirror when in doubt". A file the bounded
    // engine can edit must not be silently upgraded to the mirror.
    assert_eq!(file.memory_strategy(), Some(MemoryStrategy::Bounded));
    file.dataset("d").unwrap().append(&[4i32]).unwrap();
    file.close().unwrap();
}

#[test]
fn mirrored_takes_the_mirror_even_on_a_bounded_capable_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plain_mirrored.h5");
    plain_file(&path);

    let file = File::open_rw_bounded_with_options(
        &path,
        FileAccessProperties::new().with_memory_strategy(MemoryStrategy::Mirrored),
    )
    .unwrap();
    assert_eq!(file.memory_strategy(), Some(MemoryStrategy::Mirrored));
    file.dataset("d").unwrap().append(&[4i32]).unwrap();
    file.close().unwrap();
}

#[test]
fn a_paged_file_without_persisted_free_space_is_refused_under_every_strategy() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("paged.h5");
    let mut b = FileBuilder::new();
    // Paged, but *not* persisting its free-space managers.
    b.with_file_space_strategy(FileSpaceStrategy::Page, false, 0);
    b.create_dataset("d")
        .with_i32_data(&[1, 2, 3])
        .with_shape(&[3])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[3]);
    b.write(&path).unwrap();

    // The staged commit refuses this too, so falling back to the mirror would
    // trade a clear error at open for the same error later with work already
    // staged. `Auto` must not "rescue" it.
    for strategy in [MemoryStrategy::Bounded, MemoryStrategy::Auto] {
        let err = File::open_rw_bounded_with_options(
            &path,
            FileAccessProperties::new().with_memory_strategy(strategy),
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("persisted free space"),
            "{strategy:?} should refuse naming the persisted free space, got: {msg}"
        );
    }
}

#[test]
fn a_file_with_no_editing_session_has_no_strategy_to_report() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ro.h5");
    plain_file(&path);

    assert_eq!(File::open(&path).unwrap().memory_strategy(), None);
    assert_eq!(
        File::open_streaming(&path).unwrap().memory_strategy(),
        None,
        "a streaming open builds no editing session either"
    );
}

#[test]
fn the_requested_strategy_survives_on_the_properties() {
    // The request and the outcome are different questions; the properties keep
    // answering the first one even after an open resolved it to the second.
    let props = FileAccessProperties::new().with_memory_strategy(MemoryStrategy::Auto);
    assert_eq!(props.memory_strategy(), MemoryStrategy::Auto);
    assert_eq!(
        FileAccessProperties::new().memory_strategy(),
        MemoryStrategy::Bounded,
        "the default must stay the strict one"
    );

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("props.h5");
    userblock_file(&path);
    let file = File::open_rw_bounded_with_options(&path, props).unwrap();
    assert_eq!(
        file.access_properties().memory_strategy(),
        MemoryStrategy::Auto
    );
    assert_eq!(file.memory_strategy(), Some(MemoryStrategy::Mirrored));
}
