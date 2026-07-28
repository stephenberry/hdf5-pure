//! `MemoryStrategy`: what a read-write open is allowed to spend on holding the
//! file, who decides, and how a caller finds out what it got (issue #198,
//! steps 3 and 4).
//!
//! `File::open_rw` picks its backing from the file rather than making the caller
//! pick a function. The bounded engine cannot edit every file — a userblock still
//! needs the whole-file mirror — and these tests pin what happens then: `open_rw`
//! falls back, an explicit `Bounded` refuses, and either way the resulting `File`
//! says which one it is, because "bounded" is a memory guarantee and a caller who
//! was quietly given the mirror instead has no other way to notice.

use hdf5_pure::{File, FileAccessProperties, FileBuilder, FileSpaceStrategy, MemoryStrategy};

fn with_strategy(strategy: MemoryStrategy) -> FileAccessProperties {
    FileAccessProperties::new().with_memory_strategy(strategy)
}

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

/// Paged, but *not* persisting its free-space managers: neither engine can edit
/// it, so no strategy makes it editable.
fn paged_nonpersist_file(path: &std::path::Path) {
    let mut b = FileBuilder::new();
    b.with_file_space_strategy(FileSpaceStrategy::Page, false, 0);
    b.create_dataset("d")
        .with_i32_data(&[1, 2, 3])
        .with_shape(&[3])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[3]);
    b.write(path).unwrap();
}

/// The headline of step 4: the plain `open_rw` a caller reaches for first now
/// edits an ordinary file with bounded memory, without being asked and without a
/// second function to know about.
#[test]
fn open_rw_edits_an_ordinary_file_bounded() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plain.h5");
    plain_file(&path);

    let file = File::open_rw(&path).unwrap();
    assert_eq!(file.memory_strategy(), Some(MemoryStrategy::Bounded));
    file.dataset("d").unwrap().append(&[4i32]).unwrap();
    file.close().unwrap();

    assert_eq!(
        File::open(&path)
            .unwrap()
            .dataset("d")
            .unwrap()
            .read_i32()
            .unwrap(),
        vec![1, 2, 3, 4]
    );
}

/// And it keeps working on a file the bounded engine cannot edit, which is what
/// makes the dispatch safe to put behind the unqualified entry point.
#[test]
fn open_rw_falls_back_to_the_mirror_and_says_so() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ub.h5");
    userblock_file(&path);

    let file = File::open_rw(&path).unwrap();

    // The whole point of the observable: a caller learns the fallback was taken
    // and memory now scales with the file.
    assert_eq!(file.memory_strategy(), Some(MemoryStrategy::Mirrored));

    // And it is a real editing session, not just an open that succeeded.
    //
    // Through the *staged* path, because a userblock defeats the immediate
    // in-place append on either engine — the mirror buys the staged edit surface
    // here, not in-place append. `Dataset::append` still refuses, naming
    // `append_staged`.
    let mut ds = file.dataset("d").unwrap();
    let err = ds
        .append(&[4i32])
        .expect_err("in-place append needs no userblock");
    assert!(
        matches!(&err, hdf5_pure::Error::AppendInPlaceUnsupported(m) if m.contains("append_staged")),
        "the refusal should name append_staged as the fallback, got: {err:?}"
    );
    ds.append_staged(|b| {
        b.append_i32(&[4, 5]);
    })
    .unwrap();
    // Drop the dataset handle before closing: it holds a clone of the same inner
    // file, so the exclusive lock outlives `close` while it is alive, and the
    // read below fails on Windows (os error 33) where OS locks are mandatory.
    drop(ds);
    file.close().unwrap();

    let reopened = File::open(&path).unwrap();
    assert_eq!(
        reopened.dataset("d").unwrap().read_i32().unwrap(),
        vec![1, 2, 3, 4, 5]
    );
}

#[test]
fn bounded_refuses_the_file_open_rw_would_mirror() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ub_strict.h5");
    userblock_file(&path);

    let err = File::open_rw_with_options(&path, with_strategy(MemoryStrategy::Bounded))
        .expect_err("Bounded must not silently spend O(file size) memory");
    let msg = err.to_string();
    assert!(
        msg.contains("userblock"),
        "the refusal should name the userblock, got: {msg}"
    );

    // The deprecated entry point is the same request spelled the old way.
    #[allow(deprecated)]
    let old = File::open_rw_bounded(&path);
    assert_eq!(
        old.unwrap_err().to_string(),
        msg,
        "open_rw_bounded must stay an alias for the strict strategy"
    );
}

/// `Mirrored` is the escape hatch back to what `open_rw` did before it learned to
/// dispatch: whole-file mirror, whatever the file looks like.
#[test]
fn mirrored_takes_the_mirror_even_on_a_bounded_capable_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plain_mirrored.h5");
    plain_file(&path);

    let file = File::open_rw_with_options(&path, with_strategy(MemoryStrategy::Mirrored)).unwrap();
    assert_eq!(file.memory_strategy(), Some(MemoryStrategy::Mirrored));
    file.dataset("d").unwrap().append(&[4i32]).unwrap();
    file.close().unwrap();
}

/// An explicit strategy overrides the entry point's own default in *both*
/// directions, which is what makes the deprecated function a pure default and not
/// a separate capability.
#[test]
fn an_explicit_strategy_overrides_either_entry_point() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ub_override.h5");
    userblock_file(&path);

    // open_rw defaults to falling back; asking for Bounded refuses instead.
    let err = File::open_rw_with_options(&path, with_strategy(MemoryStrategy::Bounded))
        .expect_err("an explicit Bounded must override open_rw's fallback");
    assert!(
        err.to_string().contains("userblock"),
        "the refusal should name what the bounded engine cannot edit, got: {err}"
    );

    // open_rw_bounded defaults to refusing; asking for Auto falls back instead.
    #[allow(deprecated)]
    let file =
        File::open_rw_bounded_with_options(&path, with_strategy(MemoryStrategy::Auto)).unwrap();
    assert_eq!(file.memory_strategy(), Some(MemoryStrategy::Mirrored));
}

/// A file neither engine can edit is refused whatever is preferred — falling back
/// would only trade a clear error at open for the same error at commit, with work
/// already staged. `Mirrored` still opens it, because that is a request for a
/// backing rather than a preference, and reading such a file is legitimate.
#[test]
fn a_paged_file_without_persisted_free_space_is_refused_unless_the_mirror_is_demanded() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("paged.h5");
    paged_nonpersist_file(&path);

    for properties in [
        FileAccessProperties::new(),
        with_strategy(MemoryStrategy::Bounded),
        with_strategy(MemoryStrategy::Auto),
    ] {
        let err = File::open_rw_with_options(&path, properties).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("persisted free space"),
            "{:?} should refuse naming the persisted free space, got: {msg}",
            properties.memory_strategy()
        );
    }

    let file = File::open_rw_with_options(&path, with_strategy(MemoryStrategy::Mirrored))
        .expect("Mirrored is a request for a backing, and the mirror can read this file");
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
}

/// The shared refusals must be checked *before* the bounded-only ones, or falling
/// back for a userblock skips them and hands back a session that cannot commit.
///
/// A userblock is the case that makes the ordering observable: persisted free space
/// is declined for a non-zero base address, so this file reaches the engine looking
/// exactly like a paged non-persisting one, while also being bounded-ineligible. If
/// the fallback ran first it would mirror the file, accept staged work, and only
/// refuse at `commit` — the late refusal this dispatch exists to eliminate.
#[test]
fn a_shared_refusal_beats_a_fallback_the_same_file_qualifies_for() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ub_paged.h5");
    let mut b = FileBuilder::new();
    b.with_userblock(4096);
    b.with_file_space_strategy(FileSpaceStrategy::Page, false, 0);
    b.create_dataset("d")
        .with_i32_data(&[1, 2, 3])
        .with_shape(&[3]);
    b.write(&path).unwrap();

    let err = File::open_rw(&path).expect_err("neither backing can edit this file");
    let msg = err.to_string();
    assert!(
        msg.contains("persisted free space"),
        "the shared paged refusal must win over the userblock fallback, got: {msg}"
    );
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
    let props = with_strategy(MemoryStrategy::Auto);
    assert_eq!(props.memory_strategy(), Some(MemoryStrategy::Auto));
    assert_eq!(
        FileAccessProperties::new().memory_strategy(),
        None,
        "unset must stay unset, so each entry point can supply its own default"
    );

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("props.h5");
    userblock_file(&path);
    let file = File::open_rw_with_options(&path, props).unwrap();
    assert_eq!(
        file.access_properties().memory_strategy(),
        Some(MemoryStrategy::Auto)
    );
    assert_eq!(file.memory_strategy(), Some(MemoryStrategy::Mirrored));
}
