// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// gated to 64-bit-pointer targets.
#![cfg(not(target_pointer_width = "32"))]
//! Cross-validation of the file-space strategy (`H5Pset_file_space_strategy`)
//! against the reference HDF5 C library: a strategy hdf5-pure writes is read back
//! by the C library, and a strategy the C library writes is read back by
//! hdf5-pure — including a persisted (persist=true) file whose File Space Info
//! message carries free-space-manager addresses.

#![allow(deprecated)] // exercises the deprecated EditSession/SwmrWriter shims (issue #148)
use hdf5::plist::file_create::FileSpaceStrategy as CStrategy;
use hdf5_pure::{EditSession, File, FileBuilder, FileSpaceStrategy};
use std::sync::{Mutex, MutexGuard};
use tempfile::tempdir;

// The reference free-space query, resolved at link time from the statically
// linked libhdf5. Returns the total free space the C library tracks for the open
// file, which it can only report by loading and parsing the on-disk
// free-space-manager (`FSHD`/`FSSE`) blocks — so a positive result proves the C
// library accepts the managers hdf5-pure wrote.
unsafe extern "C" {
    fn H5Fget_freespace(file_id: i64) -> i64;
}

// `hdf5-metno` serializes its own C calls through an internal lock, but the raw
// `H5Fget_freespace` FFI above bypasses it. Serialize every C-library call in this
// file through one mutex so the raw call never races a concurrent libhdf5 call in
// another test (the C library is not built thread-safe here). Poisoning is
// ignored: a panic in one test must not cascade into the others.
static C_LIB: Mutex<()> = Mutex::new(());

fn c_lib_guard() -> MutexGuard<'static, ()> {
    C_LIB.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
fn c_library_reads_our_strategy() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    for (i, (ours, expected)) in [
        (
            FileSpaceStrategy::FsmAggr,
            CStrategy::FreeSpaceManager {
                paged: false,
                persist: false,
                threshold: 1,
            },
        ),
        (
            FileSpaceStrategy::Page,
            CStrategy::FreeSpaceManager {
                paged: true,
                persist: false,
                threshold: 1,
            },
        ),
        // The C library maps the AGGR strategy onto its `PageAggregation` name.
        (FileSpaceStrategy::Aggr, CStrategy::PageAggregation),
        (FileSpaceStrategy::None, CStrategy::None),
    ]
    .into_iter()
    .enumerate()
    {
        let path = dir.path().join(format!("ours_{i}.h5"));
        let mut b = FileBuilder::new();
        b.create_dataset("d").with_i32_data(&[10, 20, 30]);
        b.with_file_space_strategy(ours, false, 1)
            .with_file_space_page_size(4096);
        b.write(&path).unwrap();

        // The C library opens our file, reads the data, and recovers the strategy
        // from the superblock-extension File Space Info message.
        let f = hdf5::File::open(&path).unwrap();
        assert_eq!(
            f.dataset("d").unwrap().read_raw::<i32>().unwrap(),
            vec![10, 20, 30]
        );
        let strat = f.create_plist().unwrap().get_file_space_strategy().unwrap();
        assert_eq!(
            strat, expected,
            "strategy {ours:?} read back by the C library"
        );
    }
}

#[test]
fn we_read_c_library_persisted_free_space() {
    let _c = c_lib_guard();
    // The C library writes a persisted FSM file with real free space (a deleted
    // dataset). hdf5-pure must follow the File Space Info manager addresses to the
    // on-disk FSHD/FSSE blocks and recover the freed sections.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_persisted_fsm.h5");
    {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_v110())
            .with_fcpl(|fcpl| {
                fcpl.file_space_strategy(CStrategy::FreeSpaceManager {
                    paged: false,
                    persist: true,
                    threshold: 1,
                })
            })
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .shape((100,))
            .create("a")
            .unwrap()
            .write(&vec![1i32; 100])
            .unwrap();
        file.new_dataset::<i32>()
            .shape((400,))
            .create("b")
            .unwrap()
            .write(&vec![2i32; 400])
            .unwrap();
        file.new_dataset::<i32>()
            .shape((100,))
            .create("c")
            .unwrap()
            .write(&vec![3i32; 100])
            .unwrap();
        file.unlink("b").unwrap(); // frees b's storage into the persisted FSM
        file.close().unwrap();
    }

    let f = File::open(&path).unwrap();
    assert_eq!(f.file_space_strategy(), Some(FileSpaceStrategy::FsmAggr));
    assert!(f.file_space_info().unwrap().persist);

    let free = f.persisted_free_space();
    assert!(
        !free.is_empty(),
        "expected persisted free sections from the deleted dataset"
    );
    // The freed regions are non-overlapping and account for real space (b held
    // 400 * 4 = 1600 bytes of raw data, freed as at least one large section).
    let total: u64 = free.iter().map(|(_, len)| *len).sum();
    assert!(
        total >= 1600,
        "freed space {total} should cover the dataset"
    );
    assert!(free.iter().any(|&(_, len)| len >= 1600));

    // Surviving datasets still read correctly alongside the persisted managers.
    assert_eq!(f.dataset("a").unwrap().read_i32().unwrap(), vec![1; 100]);
    assert_eq!(f.dataset("c").unwrap().read_i32().unwrap(), vec![3; 100]);
}

#[test]
fn c_library_reads_our_persisted_free_space() {
    let _c = c_lib_guard();
    // The mirror of `we_read_c_library_persisted_free_space`: hdf5-pure writes a
    // persisted file with real free space (a deleted dataset), and the reference
    // C library opens it, recovers the strategy, reads the survivors, and loads
    // the on-disk free-space managers we wrote.
    let dir = tempdir().unwrap();
    let path = dir.path().join("ours_persisted.h5");

    // Create a persisted file, then free a dataset's storage in place.
    let mut b = FileBuilder::new();
    b.create_dataset("a").with_i32_data(&[1; 100]);
    b.create_dataset("big").with_i32_data(&[7; 400]); // 1600 bytes of raw data
    b.create_dataset("c").with_i32_data(&[3; 100]);
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    b.write(&path).unwrap();
    {
        let mut s = EditSession::open(&path).unwrap();
        s.delete("big");
        s.commit().unwrap();
    }

    // hdf5-pure's own reader recovers the persisted sections (covering "big").
    let ours = File::open(&path).unwrap();
    let total_ours: u64 = ours.persisted_free_space().iter().map(|(_, l)| l).sum();
    assert!(
        total_ours >= 1600,
        "we persist the freed storage: {total_ours}"
    );
    drop(ours);

    // The C library opens the same file: strategy and persist flag round-trip,
    // the survivors read byte-exact, and `H5Fget_freespace` parses our managers.
    let f = hdf5::File::open(&path).unwrap();
    let strat = f.create_plist().unwrap().get_file_space_strategy().unwrap();
    assert_eq!(
        strat,
        CStrategy::FreeSpaceManager {
            paged: false,
            persist: true,
            threshold: 1,
        },
        "C library recovers our persisted FSM strategy"
    );
    assert_eq!(
        f.dataset("a").unwrap().read_raw::<i32>().unwrap(),
        vec![1; 100]
    );
    assert_eq!(
        f.dataset("c").unwrap().read_raw::<i32>().unwrap(),
        vec![3; 100]
    );
    assert!(f.dataset("big").is_err(), "the deleted dataset is gone");

    // Loading the managers requires parsing our FSHD/FSSE blocks; the C library
    // reports at least the freed dataset's storage as free space.
    let free_c = unsafe { H5Fget_freespace(f.id()) };
    assert!(
        free_c >= 1600,
        "C library loads our free-space managers and reports the freed space (got {free_c})"
    );
}

#[test]
fn we_read_c_library_strategy() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();

    // Each case: a C-written strategy and what hdf5-pure should report. The C
    // library only writes a File Space Info message for non-default settings, so
    // these all differ from the default (FSM_AGGR, non-persistent).
    let write_c = |path: &std::path::Path, strategy: CStrategy| {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_v110())
            .with_fcpl(move |fcpl| fcpl.file_space_strategy(strategy))
            .create(path)
            .unwrap();
        file.new_dataset::<i32>()
            .shape((3,))
            .create("d")
            .unwrap()
            .write(&[1i32, 2, 3])
            .unwrap();
        file.close().unwrap();
    };

    // NONE and AGGR: non-persistent, no manager addresses.
    let p_none = dir.path().join("c_none.h5");
    write_c(&p_none, CStrategy::None);
    let f = File::open(&p_none).unwrap();
    assert_eq!(f.file_space_strategy(), Some(FileSpaceStrategy::None));
    assert!(!f.file_space_info().unwrap().persist);
    assert_eq!(f.dataset("d").unwrap().read_i32().unwrap(), vec![1, 2, 3]);

    let p_aggr = dir.path().join("c_aggr.h5");
    write_c(&p_aggr, CStrategy::PageAggregation);
    assert_eq!(
        File::open(&p_aggr).unwrap().file_space_strategy(),
        Some(FileSpaceStrategy::Aggr)
    );

    // FSM_AGGR with persist=true: the message carries free-space-manager
    // addresses, which we parse (without following them to their on-disk blocks).
    let p_persist = dir.path().join("c_persist.h5");
    write_c(
        &p_persist,
        CStrategy::FreeSpaceManager {
            paged: false,
            persist: true,
            threshold: 5,
        },
    );
    let f = File::open(&p_persist).unwrap();
    let info = f
        .file_space_info()
        .expect("persisted file records a strategy");
    assert_eq!(info.strategy, FileSpaceStrategy::FsmAggr);
    assert!(info.persist, "persist flag is read");
    assert_eq!(info.threshold, 5);
    assert!(
        !info.manager_addrs.is_empty(),
        "a persisted file carries free-space-manager addresses"
    );
    // The data still reads correctly.
    assert_eq!(f.dataset("d").unwrap().read_i32().unwrap(), vec![1, 2, 3]);
}
