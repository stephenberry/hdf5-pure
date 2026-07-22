//! Bounded-memory read-write backend (`File::open_rw_bounded`, issue #147):
//! reads, immediate appends, internal batching, cache coherence, and the typed
//! refusals for everything mirror-only.

use hdf5_pure::{
    AttrValue, Error, File, FileAccessOptions, FileBuilder, FileSpaceStrategy, MetadataCacheConfig,
};
use tempfile::tempdir;

/// Build a rank-1 unlimited chunked i32 dataset `d` seeded with `0..n`, with
/// optional deflate.
fn build(path: &std::path::Path, n: i32, chunk: u64, deflate: bool) {
    let data: Vec<i32> = (0..n).collect();
    let mut b = FileBuilder::new();
    let ds = b
        .create_dataset("d")
        .with_i32_data(&data)
        .with_shape(&[n as u64])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[chunk]);
    if deflate {
        ds.with_deflate(6);
    }
    b.write(path).unwrap();
}

fn read_i32(path: &std::path::Path) -> Vec<i32> {
    File::open(path)
        .unwrap()
        .dataset("d")
        .unwrap()
        .read_i32()
        .unwrap()
}

#[test]
fn append_and_read_through_one_handle() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("a.h5");
    build(&p, 6, 4, false);
    {
        let file = File::open_rw_bounded(&p).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append(&[6i32, 7, 8]).unwrap();
        // The appending handle observes the new length immediately.
        assert_eq!(ds.shape().unwrap(), vec![9]);
        assert_eq!(ds.read_i32().unwrap(), (0..9).collect::<Vec<_>>());
        ds.append(&[9i32]).unwrap();
        assert_eq!(ds.read_i32().unwrap(), (0..10).collect::<Vec<_>>());
    }
    // Scope the writer before re-opening: Windows file locks are mandatory.
    assert_eq!(read_i32(&p), (0..10).collect::<Vec<_>>());
}

#[test]
fn many_appends_across_calls_stay_o1() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("many.h5");
    build(&p, 0, 8, false);
    {
        let file = File::open_rw_bounded(&p).unwrap();
        let mut ds = file.dataset("d").unwrap();
        let mut next = 0i32;
        for _ in 0..200 {
            let batch: Vec<i32> = (next..next + 5).collect();
            ds.append(&batch).unwrap();
            next += 5;
        }
        assert_eq!(ds.shape().unwrap(), vec![1000]);
    }
    assert_eq!(read_i32(&p), (0..1000).collect::<Vec<_>>());
}

#[test]
fn refetched_handle_observes_appends() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("refetch.h5");
    build(&p, 4, 4, false);
    let file = File::open_rw_bounded(&p).unwrap();
    let mut ds = file.dataset("d").unwrap();
    ds.append(&[4i32, 5, 6, 7]).unwrap();
    let fresh = file.dataset("d").unwrap();
    assert_eq!(fresh.shape().unwrap(), vec![8]);
    assert_eq!(fresh.read_i32().unwrap(), (0..8).collect::<Vec<_>>());
}

#[test]
fn filtered_appends_whole_chunks_only() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("filtered.h5");
    build(&p, 8, 4, true);
    {
        let file = File::open_rw_bounded(&p).unwrap();
        let mut ds = file.dataset("d").unwrap();
        // Chunk-aligned filtered append is accepted.
        ds.append(&[8i32, 9, 10, 11]).unwrap();
        assert_eq!(ds.read_i32().unwrap(), (0..12).collect::<Vec<_>>());
        // Non-chunk-aligned filtered append is refused (same engine rule as
        // open_rw).
        let err = ds.append(&[12i32]).unwrap_err();
        assert!(
            matches!(err, Error::AppendInPlaceUnsupported(_)),
            "unexpected error: {err:?}"
        );
    }
    assert_eq!(read_i32(&p), (0..12).collect::<Vec<_>>());
}

#[test]
fn large_append_batches_internally() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("large.h5");
    build(&p, 3, 256, false);
    // ~2.5 MiB of i32 in ONE call: far past the 1 MiB batch budget, unaligned
    // start (3), so the run exercises partial-tail fill + several whole-chunk
    // batches + a trailing remainder.
    let total = 655_360i32 + 7;
    {
        let file = File::open_rw_bounded(&p).unwrap();
        let mut ds = file.dataset("d").unwrap();
        let batch: Vec<i32> = (3..total).collect();
        ds.append(&batch).unwrap();
        assert_eq!(ds.shape().unwrap(), vec![total as u64]);
    }
    let got = read_i32(&p);
    assert_eq!(got.len(), total as usize);
    assert!(got.iter().enumerate().all(|(i, &v)| v == i as i32));
}

#[test]
fn staged_surface_returns_typed_error() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("staged.h5");
    build(&p, 4, 4, false);
    let file = File::open_rw_bounded(&p).unwrap();
    let mut ds = file.dataset("d").unwrap();
    let root = file.root();

    assert!(matches!(
        ds.write(&[9i32, 9, 9, 9]),
        Err(Error::BoundedStagedUnsupported)
    ));
    assert!(matches!(
        ds.set_attr("units", AttrValue::String("m".into())),
        Err(Error::BoundedStagedUnsupported)
    ));
    assert!(matches!(
        ds.append_staged(|b| {
            b.append_i32(&[1]);
        }),
        Err(Error::BoundedStagedUnsupported)
    ));
    assert!(matches!(
        root.create_group("g"),
        Err(Error::BoundedStagedUnsupported)
    ));
    assert!(matches!(
        root.delete("d"),
        Err(Error::BoundedStagedUnsupported)
    ));
    assert!(matches!(
        file.commit(),
        Err(Error::BoundedStagedUnsupported)
    ));
    assert!(matches!(
        file.copy("d", "d2"),
        Err(Error::BoundedStagedUnsupported)
    ));
    assert!(matches!(
        file.space_accounting(),
        Err(Error::BoundedStagedUnsupported)
    ));
}

#[test]
fn close_seals_writes_but_not_reads() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("close.h5");
    build(&p, 4, 4, false);
    let file = File::open_rw_bounded(&p).unwrap();
    let mut ds = file.dataset("d").unwrap();
    ds.append(&[4i32]).unwrap();
    file.clone().close().unwrap();
    assert!(matches!(ds.append(&[5i32]), Err(Error::FileClosed)));
    // Reads through the surviving handle still work.
    assert_eq!(ds.read_i32().unwrap(), (0..5).collect::<Vec<_>>());
}

#[test]
fn bounded_open_takes_the_exclusive_lock() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("lock.h5");
    build(&p, 4, 4, false);
    let bounded = File::open_rw_bounded(&p).unwrap();
    let err = File::open_rw(&p).unwrap_err();
    assert!(matches!(err, Error::FileLocked(_)), "got: {err:?}");
    drop(bounded);
    File::open_rw(&p).unwrap();
}

#[test]
fn userblock_file_is_refused_at_open() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("userblock.h5");
    let mut b = FileBuilder::new();
    b.with_userblock(512);
    b.create_dataset("d")
        .with_i32_data(&[1, 2, 3])
        .with_shape(&[3])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[2]);
    b.write(&p).unwrap();
    let err = File::open_rw_bounded(&p).unwrap_err();
    assert!(matches!(err, Error::EditUnsupported(_)), "got: {err:?}");
}

#[test]
fn persisted_free_space_file_is_refused_at_open() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("persist.h5");
    let mut b = FileBuilder::new();
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    b.create_dataset("d")
        .with_i32_data(&[1, 2, 3])
        .with_shape(&[3])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[2]);
    b.write(&p).unwrap();
    let err = File::open_rw_bounded(&p).unwrap_err();
    assert!(matches!(err, Error::EditUnsupported(_)), "got: {err:?}");
}

#[test]
fn metadata_cache_stays_coherent_across_appends() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("cache.h5");
    build(&p, 4, 4, false);
    let options =
        FileAccessOptions::new().with_metadata_cache(MetadataCacheConfig::new(256 * 1024));
    let file = File::open_rw_bounded_with_options(&p, options).unwrap();
    let mut ds = file.dataset("d").unwrap();
    // Prime the metadata cache with the object-header windows.
    assert_eq!(ds.shape().unwrap(), vec![4]);
    assert_eq!(ds.read_i32().unwrap(), vec![0, 1, 2, 3]);
    // The append patches the dataspace dimension in place; overlapping cached
    // windows must be invalidated so re-reads observe the new length.
    ds.append(&[4i32, 5]).unwrap();
    assert_eq!(ds.shape().unwrap(), vec![6]);
    assert_eq!(ds.read_i32().unwrap(), (0..6).collect::<Vec<_>>());
    let fresh = file.dataset("d").unwrap();
    assert_eq!(fresh.shape().unwrap(), vec![6]);
}

#[test]
fn reads_match_streaming_capabilities() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("reads.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&[1, 2, 3])
        .with_shape(&[3])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[2]);
    let mut grp = b.create_group("grp");
    grp.create_dataset("nested")
        .with_f64_data(&[1.5, 2.5])
        .with_shape(&[2]);
    b.add_group(grp.finish());
    b.write(&p).unwrap();

    let file = File::open_rw_bounded(&p).unwrap();
    // Groups, nested paths, and non-append datasets all read.
    assert_eq!(
        file.dataset("grp/nested").unwrap().read_f64().unwrap(),
        vec![1.5, 2.5]
    );
    let grp = file.group("grp").unwrap();
    assert_eq!(grp.datasets().unwrap(), vec!["nested"]);
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
}

#[test]
fn unaligned_filtered_multi_batch_append_is_refused_atomically() {
    // A filtered append that is (a) not chunk-aligned and (b) larger than the
    // internal batch budget must be refused up front with NO batch applied —
    // the same atomic refusal as open_rw — not partially committed before the
    // final short batch errors.
    let dir = tempdir().unwrap();
    let p = dir.path().join("atomic.h5");
    build(&p, 256, 256, true);
    let before = std::fs::read(&p).unwrap();
    {
        let file = File::open_rw_bounded(&p).unwrap();
        let mut ds = file.dataset("d").unwrap();
        // ~2 MiB of i32, one element past chunk alignment.
        let unaligned: Vec<i32> = (0..524_289).collect();
        let err = ds.append(&unaligned).unwrap_err();
        assert!(
            matches!(err, Error::AppendInPlaceUnsupported(_)),
            "got: {err:?}"
        );
        assert_eq!(ds.shape().unwrap(), vec![256]);
    }
    assert_eq!(
        std::fs::read(&p).unwrap(),
        before,
        "a refused append modified the file"
    );
    // Same atomicity for a raw append whose byte length is not a whole number
    // of elements.
    let before = std::fs::read(&p).unwrap();
    {
        let file = File::open_rw_bounded(&p).unwrap();
        let mut ds = file.dataset("d").unwrap();
        let mut bytes = vec![0u8; 2 * 1024 * 1024];
        bytes.push(0); // not a whole i32
        let err = ds.append_raw(&bytes).unwrap_err();
        assert!(
            matches!(err, Error::AppendInPlaceUnsupported(_)),
            "got: {err:?}"
        );
    }
    assert_eq!(std::fs::read(&p).unwrap(), before);
}

#[test]
fn chunk_introspection_works_on_bounded_files() {
    // chunks() walks the chunk index through the file source; on a bounded
    // (and mirror) file that must go through the engine's store, not the
    // empty borrowed view.
    let dir = tempdir().unwrap();
    let p = dir.path().join("chunks.h5");
    build(&p, 8, 4, false);
    let file = File::open_rw_bounded(&p).unwrap();
    let mut ds = file.dataset("d").unwrap();
    let chunks = ds.chunks().unwrap();
    assert_eq!(chunks.len(), 2);
    ds.append(&[8i32, 9]).unwrap();
    let chunks = file.dataset("d").unwrap().chunks().unwrap();
    assert_eq!(chunks.len(), 3);
}
