//! Staged edits on genuine paged files through `File::open_rw` (issue #198,
//! step 1). The whole-file editor used to refuse every paged file and send the
//! caller to `File::open_rw_bounded`; it now commits a persisting paged file
//! through a page-aware tail that keeps the per-page-type managers intact.
//! A paged file *without* persisted managers is still refused, because nothing
//! on disk records which pages hold metadata and which hold raw data.

use hdf5_pure::{Error, File, FileBuilder, FileSpaceStrategy};

const PAGE: u64 = 4096;

fn tmp(name: &str) -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tmp");
    let _ = std::fs::create_dir_all(&p);
    p.push(name);
    p
}

/// Build a paged file with one contiguous i32 dataset `d` seeded with `0..n`.
fn build_paged(path: &std::path::Path, n: i32, persist: bool) {
    let data: Vec<i32> = (0..n).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&data)
        .with_shape(&[n as u64]);
    b.with_file_space_strategy(FileSpaceStrategy::Page, persist, 0)
        .with_file_space_page_size(PAGE);
    b.write(path).unwrap();
}

/// The on-disk paged invariants: a whole number of pages, a page-aligned EOA
/// that matches the file length, and free sections that neither overlap nor run
/// past end-of-file.
fn assert_paged_ok(path: &std::path::Path) {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(
        bytes.len() as u64 % PAGE,
        0,
        "file is a whole number of pages"
    );
    let f = File::open(path).unwrap();
    assert_eq!(f.file_space_strategy(), Some(FileSpaceStrategy::Page));
    let info = f.file_space_info().expect("records a strategy");
    assert!(info.persist, "still persisting");
    assert_eq!(info.page_size, PAGE);
    assert_eq!(info.eoa_pre_fsm % PAGE, 0, "EOA page-aligned");
    assert_eq!(info.eoa_pre_fsm, bytes.len() as u64, "EOA == file size");
    let mut free = f.persisted_free_space();
    free.sort_by_key(|&(a, _)| a);
    let mut prev_end = 0u64;
    for (addr, len) in &free {
        assert!(*addr >= prev_end, "sections do not overlap");
        assert!(addr + len <= bytes.len() as u64, "section within the file");
        prev_end = addr + len;
    }
}

/// A persisting paged file accepts a staged dataset addition through
/// `File::open_rw`: both the old and the new dataset read back, and the file
/// still satisfies every paged invariant.
#[test]
fn paged_persist_staged_create_dataset() {
    let path = tmp("pure_paged_staged_create.h5");
    build_paged(&path, 64, true);

    {
        let s = File::open_rw(&path).unwrap();
        s.root()
            .create_dataset("added", |b| {
                b.with_i32_data(&(1000..1100).collect::<Vec<i32>>());
            })
            .unwrap();
        s.commit().unwrap();
    }

    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("d").unwrap().read_i32().unwrap(),
        (0..64).collect::<Vec<i32>>()
    );
    assert_eq!(
        f.dataset("added").unwrap().read_i32().unwrap(),
        (1000..1100).collect::<Vec<i32>>()
    );
    drop(f);
    assert_paged_ok(&path);
}

/// Every page holding raw dataset bytes must hold *only* raw bytes.
///
/// A paged file never mixes metadata and raw data within one page — that is the
/// invariant the commit's page-typed allocation exists to maintain, and the one
/// thing the C library cannot report on, since it reads such a file happily
/// either way. Raw extents come from the public layout introspection; a page that
/// overlaps one must contain no object-header signature (`OHDR`, or a
/// continuation block's `OCHK`), which is what a metadata allocation landing in a
/// raw page would leave behind.
fn assert_pages_homogeneous(path: &std::path::Path, datasets: &[&str]) {
    let bytes = std::fs::read(path).unwrap();
    let f = File::open(path).unwrap();
    let mut raw: Vec<(u64, u64)> = Vec::new();
    for name in datasets {
        let ds = f.dataset(name).unwrap();
        match ds.layout().unwrap() {
            hdf5_pure::Layout::Contiguous {
                address: Some(addr),
                size,
            } => raw.push((addr, size)),
            hdf5_pure::Layout::Chunked { .. } => {
                for c in ds.chunks().unwrap() {
                    raw.push((c.address, c.storage_size));
                }
            }
            // Compact data lives inside the object header (metadata), and an
            // unallocated contiguous dataset owns no bytes at all.
            _ => {}
        }
    }
    assert!(!raw.is_empty(), "expected at least one raw extent to check");
    drop(f);

    let mut raw_pages: Vec<u64> = Vec::new();
    for (addr, size) in raw {
        if size == 0 {
            continue;
        }
        let first = addr / PAGE;
        let last = (addr + size - 1) / PAGE;
        for p in first..=last {
            raw_pages.push(p);
        }
    }
    raw_pages.sort_unstable();
    raw_pages.dedup();

    for p in raw_pages {
        let start = (p * PAGE) as usize;
        let end = ((p + 1) * PAGE).min(bytes.len() as u64) as usize;
        let page = &bytes[start..end];
        for sig in [b"OHDR", b"OCHK"] {
            assert!(
                !page.windows(4).any(|w| w == sig),
                "page {p} holds raw data and the {} signature: a metadata \
                 allocation landed in a raw page",
                std::str::from_utf8(sig).unwrap()
            );
        }
    }
}

/// A staged commit that allocates both page types keeps them in separate pages.
/// The append writes raw chunks and rebuilds the (metadata) chunk index, and the
/// new dataset adds raw values and a metadata object header, so the commit has to
/// switch page type several times.
#[test]
fn paged_staged_commit_keeps_pages_homogeneous() {
    let path = tmp("pure_paged_staged_homogeneous.h5");
    // A chunked, unlimited dataset so the staged append rebuilds an index.
    {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&(0..64).collect::<Vec<i32>>())
            .with_shape(&[64])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[64]);
        b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
            .with_file_space_page_size(PAGE);
        b.write(&path).unwrap();
    }
    {
        let s = File::open_rw(&path).unwrap();
        let mut ds = s.dataset("d").unwrap();
        ds.append_staged(|b| {
            b.append_i32(&(64..4000).collect::<Vec<i32>>());
        })
        .unwrap();
        s.root()
            .create_dataset("added", |b| {
                b.with_f64_data(&vec![2.5f64; 1024]);
            })
            .unwrap();
        s.commit().unwrap();
    }

    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("d").unwrap().read_i32().unwrap(),
        (0..4000).collect::<Vec<i32>>()
    );
    drop(f);
    assert_paged_ok(&path);
    assert_pages_homogeneous(&path, &["d", "added"]);
}

/// A paged file that does not persist its free space is still refused: without
/// on-disk managers there is no record of which pages are metadata and which are
/// raw, so a commit could not keep the two segregated.
#[test]
fn paged_without_persist_is_refused() {
    let path = tmp("pure_paged_staged_nopersist.h5");
    build_paged(&path, 64, false);

    let s = File::open_rw(&path).unwrap();
    s.root()
        .create_dataset("added", |b| {
            b.with_i32_data(&[1i32, 2, 3]);
        })
        .unwrap();
    let err = s.commit().unwrap_err();
    assert!(
        matches!(&err, Error::EditUnsupported(m) if m.contains("persisted free space")),
        "expected a persisted-free-space refusal, got {err:?}"
    );
}
