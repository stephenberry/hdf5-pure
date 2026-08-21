// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit little-endian targets.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
//! A dataset whose element bytes live in *external* files (`H5Pset_external`).
//!
//! Its data-layout message is contiguous with the data address **undefined** —
//! byte-for-byte the encoding a never-written dataset uses — and an External Data
//! Files message (header type 7) beside it names where the bytes actually are.
//! This crate does not follow those files, so "undefined address" must not be
//! read as "stores nothing": a rewrite that did would emit a schema-only dataset
//! in place of one holding data, and report success.

use std::ffi::{CString, c_char, c_int};

use hdf5_pure::{File, RepackOptions, repack};
use tempfile::tempdir;

const DEFAULT: i64 = 0;

unsafe extern "C" {
    fn H5Screate_simple(rank: c_int, dims: *const u64, maxdims: *const u64) -> i64;
    fn H5Sclose(space_id: i64) -> c_int;
    fn H5Pcreate(cls_id: i64) -> i64;
    fn H5Pclose(plist_id: i64) -> c_int;
    fn H5Pset_external(plist_id: i64, name: *const c_char, offset: i64, size: u64) -> c_int;
    fn H5Dcreate2(
        loc_id: i64,
        name: *const c_char,
        type_id: i64,
        space_id: i64,
        lcpl_id: i64,
        dcpl_id: i64,
        dapl_id: i64,
    ) -> i64;
    fn H5Dwrite(
        dset: i64,
        mem_type: i64,
        mem_space: i64,
        file_space: i64,
        dxpl: i64,
        buf: *const core::ffi::c_void,
    ) -> c_int;
    fn H5Dclose(dset_id: i64) -> c_int;
    static H5P_CLS_DATASET_CREATE_ID_g: i64;
    static H5T_NATIVE_INT_g: i64;
}

const N: usize = 16;

/// Write `path` holding `/d`, an `N`-element i32 dataset stored in `payload`.
fn write_external_fixture(path: &std::path::Path, payload: &str) {
    let file = hdf5::File::create(path).unwrap();
    let dcpl = unsafe { H5Pcreate(H5P_CLS_DATASET_CREATE_ID_g) };
    assert!(dcpl >= 0, "H5Pcreate");
    let ename = CString::new(payload).unwrap();
    assert!(
        unsafe { H5Pset_external(dcpl, ename.as_ptr(), 0, (N * 4) as u64) } >= 0,
        "H5Pset_external"
    );
    let dims = [N as u64];
    let space = unsafe { H5Screate_simple(1, dims.as_ptr(), std::ptr::null()) };
    let dname = CString::new("d").unwrap();
    let ds = unsafe {
        H5Dcreate2(
            file.id(),
            dname.as_ptr(),
            H5T_NATIVE_INT_g,
            space,
            DEFAULT,
            dcpl,
            DEFAULT,
        )
    };
    assert!(ds >= 0, "H5Dcreate2 with external storage");
    let vals: Vec<i32> = (1..=N as i32).collect();
    assert!(
        unsafe {
            H5Dwrite(
                ds,
                H5T_NATIVE_INT_g,
                DEFAULT,
                DEFAULT,
                DEFAULT,
                vals.as_ptr().cast(),
            )
        } >= 0,
        "H5Dwrite"
    );
    unsafe { H5Dclose(ds) };
    unsafe { H5Sclose(space) };
    unsafe { H5Pclose(dcpl) };
    file.close().unwrap();
}

/// `repack` refuses an externally stored dataset rather than reproducing it
/// without its data.
///
/// The C library reads 1..=16 from the source. Every byte of that lives outside
/// the file, so a rewrite has nothing to copy; the only honest answer is to
/// refuse. Before this was checked, `repack` returned `Ok` and the destination
/// answered the fill value for every element — the failure mode a
/// values-round-trip assertion cannot see, since the source's *reader* here
/// returns the same zeros.
#[test]
fn repack_refuses_external_data_storage() {
    let dir = tempdir().unwrap();
    let src = dir.path().join("external.h5");
    let dst = dir.path().join("external_repacked.h5");
    write_external_fixture(&src, "external_payload.bin");

    // Ground truth: the C library follows the external file and finds the data.
    {
        let c = hdf5::File::open(&src).unwrap();
        let d = c.dataset("d").unwrap();
        assert_eq!(
            d.read_raw::<i32>().unwrap(),
            (1..=N as i32).collect::<Vec<i32>>(),
            "the source holds real data"
        );
        assert_eq!(d.storage_size(), (N * 4) as u64);
        c.close().unwrap();
    }

    // And the layout message is indistinguishable from a never-written one.
    {
        let f = File::open(&src).unwrap();
        assert!(matches!(
            f.dataset("d").unwrap().layout().unwrap(),
            hdf5_pure::Layout::Contiguous {
                address: None,
                size: 64
            }
        ));
    }

    let err = repack(&src, &dst, &RepackOptions::new()).unwrap_err();
    match err {
        hdf5_pure::Error::RepackUnsupported(msg) => assert!(
            msg.contains('d') && msg.contains("external"),
            "the refusal must name the dataset and the reason: {msg}"
        ),
        other => panic!("expected RepackUnsupported, got {other:?}"),
    }
    assert!(!dst.exists(), "dst must not be created when repack refuses");

    std::fs::remove_file("external_payload.bin").ok();
}
