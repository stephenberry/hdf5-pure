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

#[test]
fn c_library_reads_our_fresh_persisting_file() {
    let _c = c_lib_guard();
    // Regression for issue #178: a fresh `persist = true` non-paged file must record
    // a defined `eoa_fsm_fsalloc` in its File Space Info message. hdf5-pure once
    // wrote the UNDEF sentinel, which an assertion-enabled libhdf5 rejects on open
    // (H5Fsuper.c: `fs_persist => eoa_fsm_fsalloc != UNDEF`). No dataset is deleted,
    // so the file genuinely tracks no free space — the corner the strategy
    // crosschecks (persist = false, or persist = true only after a delete) missed.
    let dir = tempdir().unwrap();
    let path = dir.path().join("ours_fresh_persist.h5");

    let mut b = FileBuilder::new();
    b.create_dataset("d").with_i32_data(&[1, 2, 3]);
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    b.write(&path).unwrap();

    // Nothing is freed, so our own tracked free space is zero.
    let total_ours: u64 = File::open(&path)
        .unwrap()
        .persisted_free_space()
        .iter()
        .map(|(_, l)| l)
        .sum();
    assert_eq!(total_ours, 0);

    // The C library opens the persisting file (which would abort before the fix),
    // recovers the persist flag, reads the data, and reports a matching (zero)
    // free-space total.
    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(
        f.create_plist().unwrap().get_file_space_strategy().unwrap(),
        CStrategy::FreeSpaceManager {
            paged: false,
            persist: true,
            threshold: 1,
        }
    );
    assert_eq!(
        f.dataset("d").unwrap().read_raw::<i32>().unwrap(),
        vec![1, 2, 3]
    );
    let free_c = unsafe { H5Fget_freespace(f.id()) };
    assert_eq!(free_c as u64, total_ours);
}

#[test]
fn c_library_reads_our_paged_file() {
    let _c = c_lib_guard();
    // hdf5-pure writes a genuine paged (H5F_FSPACE_STRATEGY_PAGE) persisting file
    // with small and large datasets. The reference C library must recover the
    // paged strategy, read every dataset, load the per-page-type free-space
    // managers (H5Fget_freespace parses our FSHD/FSSE blocks), and reopen the
    // file read-write to mutate it (proving the paged layout is consistent).
    let dir = tempdir().unwrap();
    let path = dir.path().join("ours_paged.h5");

    let small_a: Vec<i32> = (0..100).collect();
    let small_b: Vec<i32> = (0..400).collect();
    let big: Vec<i32> = (0..5000).collect(); // 20000 bytes >= page: its own run

    let mut b = FileBuilder::new();
    b.create_dataset("a").with_i32_data(&small_a);
    b.create_dataset("b").with_i32_data(&small_b);
    b.create_dataset("big").with_i32_data(&big);
    b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
        .with_file_space_page_size(16384);
    b.write(&path).unwrap();

    // hdf5-pure's own view of the tracked free space (SUPER + DRAW + LARGE tails).
    let ours = File::open(&path).unwrap();
    let total_ours: u64 = ours.persisted_free_space().iter().map(|(_, l)| l).sum();
    assert!(total_ours > 0, "paged persist tracks page-tail free space");
    drop(ours);

    // The C library recovers the paged strategy and reads every dataset.
    let f = hdf5::File::open(&path).unwrap();
    let strat = f.create_plist().unwrap().get_file_space_strategy().unwrap();
    assert_eq!(
        strat,
        CStrategy::FreeSpaceManager {
            paged: true,
            persist: true,
            threshold: 0,
        },
        "C library recovers our paged strategy"
    );
    assert_eq!(f.dataset("a").unwrap().read_raw::<i32>().unwrap(), small_a);
    assert_eq!(f.dataset("b").unwrap().read_raw::<i32>().unwrap(), small_b);
    assert_eq!(f.dataset("big").unwrap().read_raw::<i32>().unwrap(), big);

    // Loading the managers requires parsing our FSHD/FSSE blocks; the C library's
    // free-space total equals the sum of the sections we wrote.
    let free_c = unsafe { H5Fget_freespace(f.id()) };
    assert_eq!(
        free_c as u64, total_ours,
        "C library free-space total matches our paged managers"
    );
    drop(f);

    // The C library reopens the paged file read-write and adds a dataset, then
    // reads everything back: the paged layout survives a C round-trip.
    {
        let f = hdf5::File::open_rw(&path).unwrap();
        f.new_dataset::<i32>()
            .shape((30,))
            .create("added")
            .unwrap()
            .write(&(0..30).collect::<Vec<i32>>())
            .unwrap();
        f.close().unwrap();
    }
    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(f.dataset("a").unwrap().read_raw::<i32>().unwrap(), small_a);
    assert_eq!(f.dataset("big").unwrap().read_raw::<i32>().unwrap(), big);
    assert_eq!(
        f.dataset("added").unwrap().read_raw::<i32>().unwrap(),
        (0..30).collect::<Vec<i32>>()
    );
}

#[test]
fn c_library_reads_our_paged_chunked_file() {
    let _c = c_lib_guard();
    // Paged interop for chunked datasets: hdf5-pure writes a paged persisting
    // file with a small chunked dataset and a large compressed (shuffle+deflate)
    // chunked dataset whose data begins a page-aligned run. The C library must
    // read both, parse the managers, and reopen read-write.
    let dir = tempdir().unwrap();
    let path = dir.path().join("ours_paged_chunked.h5");

    let small: Vec<f64> = (0..64).map(|i| i as f64).collect();
    let big: Vec<f64> = (0..8000).map(|i| i as f64 * 0.5).collect();

    let mut b = FileBuilder::new();
    b.create_dataset("s")
        .with_f64_data(&small)
        .with_shape(&[64])
        .with_chunks(&[16]);
    {
        let ds = b
            .create_dataset("big")
            .with_f64_data(&big)
            .with_shape(&[8000])
            .with_chunks(&[1000]);
        ds.with_shuffle().with_deflate(6);
    }
    b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
        .with_file_space_page_size(16384);
    b.write(&path).unwrap();

    let ours = File::open(&path).unwrap();
    let total_ours: u64 = ours.persisted_free_space().iter().map(|(_, l)| l).sum();
    drop(ours);

    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(
        f.create_plist().unwrap().get_file_space_strategy().unwrap(),
        CStrategy::FreeSpaceManager {
            paged: true,
            persist: true,
            threshold: 0,
        },
    );
    assert_eq!(f.dataset("s").unwrap().read_raw::<f64>().unwrap(), small);
    assert_eq!(f.dataset("big").unwrap().read_raw::<f64>().unwrap(), big);
    let free_c = unsafe { H5Fget_freespace(f.id()) };
    assert_eq!(
        free_c as u64, total_ours,
        "C free-space matches our managers"
    );
    drop(f);

    // C reopens read-write and appends a dataset without corrupting the file.
    {
        let f = hdf5::File::open_rw(&path).unwrap();
        f.new_dataset::<f64>()
            .shape((4,))
            .create("extra")
            .unwrap()
            .write(&[1.0f64, 2.0, 3.0, 4.0])
            .unwrap();
        f.close().unwrap();
    }
    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(f.dataset("big").unwrap().read_raw::<f64>().unwrap(), big);
    assert_eq!(
        f.dataset("extra").unwrap().read_raw::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn c_library_reads_our_bounded_mutated_paged_file() {
    let _c = c_lib_guard();
    // hdf5-pure creates a genuine paged persisting file, then grows it through the
    // bounded backend (`File::open_rw_bounded`), appending enough rows to force
    // extensible-array index growth so the append allocates metadata as well as
    // raw chunks (exercising page segregation). The reference C library must then
    // recover the paged strategy, read every row, load the per-page-type managers
    // (H5Fget_freespace parses our FSHD/FSSE blocks), and reopen it read-write.
    let dir = tempdir().unwrap();
    let path = dir.path().join("ours_paged_mutated.h5");

    // Create with one chunk, then bounded-append to 5000 rows.
    {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&(0..64).collect::<Vec<i32>>())
            .with_shape(&[64])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[64]);
        b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
            .with_file_space_page_size(4096);
        b.write(&path).unwrap();
    }
    {
        let file = File::open_rw_bounded(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append(&(64..5000).collect::<Vec<i32>>()).unwrap();
        file.close().unwrap();
    }

    // hdf5-pure's own view of the tracked free space after the mutation.
    let want: Vec<i32> = (0..5000).collect();
    let ours = File::open(&path).unwrap();
    assert_eq!(ours.dataset("d").unwrap().read_i32().unwrap(), want);
    let total_ours: u64 = ours.persisted_free_space().iter().map(|(_, l)| l).sum();
    drop(ours);

    // The C library recovers the paged strategy and reads every row.
    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(
        f.create_plist().unwrap().get_file_space_strategy().unwrap(),
        CStrategy::FreeSpaceManager {
            paged: true,
            persist: true,
            threshold: 0,
        },
        "C library recovers our paged strategy after a bounded mutation"
    );
    assert_eq!(f.dataset("d").unwrap().read_raw::<i32>().unwrap(), want);
    // Loading the managers parses our rewritten FSHD/FSSE blocks; the C library's
    // free-space total equals the sum of the sections we wrote.
    let free_c = unsafe { H5Fget_freespace(f.id()) };
    assert_eq!(
        free_c as u64, total_ours,
        "C free-space total matches our rewritten paged managers"
    );
    drop(f);

    // The C library reopens the mutated paged file read-write and appends more
    // rows, then reads everything back: the paged layout survives a C round-trip.
    {
        let f = hdf5::File::open_rw(&path).unwrap();
        let ds = f.dataset("d").unwrap();
        ds.resize((5100,)).unwrap();
        ds.write_slice(&(5000..5100).collect::<Vec<i32>>(), 5000..5100)
            .unwrap();
        f.close().unwrap();
    }
    let f = hdf5::File::open(&path).unwrap();
    let after = f.dataset("d").unwrap().read_raw::<i32>().unwrap();
    assert_eq!(after.len(), 5100);
    assert_eq!(after[..5000], want[..]);
    assert_eq!(after[5099], 5099);
}

#[test]
fn pure_bounded_mutates_c_created_paged_file() {
    let _c = c_lib_guard();
    // The reverse direction: the reference C library creates a genuine paged
    // persisting file with an unlimited chunked dataset; hdf5-pure grows it through
    // the bounded backend; then both the C library and hdf5-pure read every row
    // back and the C library re-parses the rewritten managers.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_paged_pure_mutated.h5");

    {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_v110())
            .with_fcpl(|fcpl| {
                fcpl.file_space_strategy(CStrategy::FreeSpaceManager {
                    paged: true,
                    persist: true,
                    threshold: 1,
                })
                .file_space_page_size(4096)
            })
            .create(&path)
            .unwrap();
        let ds = file
            .new_dataset::<i32>()
            .chunk((64,))
            .shape((hdf5::Extent::resizable(64),))
            .create("d")
            .unwrap();
        ds.write(&(0..64).collect::<Vec<i32>>()).unwrap();
        file.close().unwrap();
    }

    // hdf5-pure grows the C-created paged file.
    {
        let file = File::open_rw_bounded(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append(&(64..4000).collect::<Vec<i32>>()).unwrap();
        file.close().unwrap();
    }

    let want: Vec<i32> = (0..4000).collect();
    // hdf5-pure reads its own mutation back.
    let ours = File::open(&path).unwrap();
    assert_eq!(ours.dataset("d").unwrap().read_i32().unwrap(), want);
    let total_ours: u64 = ours.persisted_free_space().iter().map(|(_, l)| l).sum();
    drop(ours);

    // The C library reads it back and re-parses the managers.
    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(
        f.create_plist().unwrap().get_file_space_strategy().unwrap(),
        CStrategy::FreeSpaceManager {
            paged: true,
            persist: true,
            threshold: 1,
        },
    );
    assert_eq!(f.dataset("d").unwrap().read_raw::<i32>().unwrap(), want);
    let free_c = unsafe { H5Fget_freespace(f.id()) };
    assert_eq!(
        free_c as u64, total_ours,
        "C free-space matches our managers"
    );
}
