// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit little-endian targets; skip them elsewhere so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
//! The optional prefix blocks of a version 2 object header survive an in-place
//! edit (PR #422).
//!
//! A version 2 object header may carry two blocks between its flags byte and its
//! chunk-0 size field: four timestamps (access, modification, change, birth) and
//! the attribute phase-change thresholds. The reference C library stores the
//! timestamps on **every** v2 header it writes — `H5O_CRT_OHDR_FLAGS_DEF` is
//! `H5O_HDR_STORE_TIMES` — so this reaches every file libhdf5, h5py or netCDF-4
//! produces in the latest format, and `EditSession` used to drop both blocks on
//! any rewrite, leaving `H5Oget_info` reporting four zeroed times.
//!
//! The fixture has to come from the C library: nothing in this crate's whole-file
//! writer emits either block, so a fixture written here would only test the
//! editor against itself. And the assertion has to come from the C library too —
//! this crate exposes no timestamp reader at all, so `H5Oget_info_by_name3` is
//! the only thing that can say the file still means what it meant.

use std::ffi::{CString, c_char, c_int, c_uint};
use std::path::Path;
use std::sync::{Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use hdf5_pure::{AttrValue, File};
use tempfile::tempdir;

// The entry points these fixtures need, resolved at link time from the statically
// linked libhdf5. `hdf5-metno` exposes neither the attribute phase-change
// property nor the object-info timestamps, and both are what this file is about.
unsafe extern "C" {
    fn H5Pcreate(cls_id: i64) -> i64;
    fn H5Pclose(plist_id: i64) -> c_int;
    fn H5Pset_attr_phase_change(plist_id: i64, max_compact: c_uint, min_dense: c_uint) -> c_int;
    fn H5Pget_attr_phase_change(
        plist_id: i64,
        max_compact: *mut c_uint,
        min_dense: *mut c_uint,
    ) -> c_int;
    fn H5Fcreate(name: *const c_char, flags: c_uint, fcpl: i64, fapl: i64) -> i64;
    fn H5Fopen(name: *const c_char, flags: c_uint, fapl: i64) -> i64;
    fn H5Fclose(id: i64) -> c_int;
    fn H5Gcreate2(loc: i64, name: *const c_char, lcpl: i64, gcpl: i64, gapl: i64) -> i64;
    fn H5Gopen2(loc: i64, name: *const c_char, gapl: i64) -> i64;
    fn H5Gget_create_plist(group_id: i64) -> i64;
    fn H5Gclose(id: i64) -> c_int;
    fn H5Dcreate2(
        loc: i64,
        name: *const c_char,
        type_id: i64,
        space: i64,
        lcpl: i64,
        dcpl: i64,
        dapl: i64,
    ) -> i64;
    fn H5Dopen2(loc: i64, name: *const c_char, dapl: i64) -> i64;
    fn H5Dget_create_plist(dataset_id: i64) -> i64;
    fn H5Dclose(id: i64) -> c_int;
    fn H5Screate(class: c_int) -> i64;
    fn H5Screate_simple(rank: c_int, dims: *const u64, maxdims: *const u64) -> i64;
    fn H5Sclose(id: i64) -> c_int;
    fn H5Acreate2(
        loc: i64,
        name: *const c_char,
        type_id: i64,
        space: i64,
        acpl: i64,
        aapl: i64,
    ) -> i64;
    fn H5Awrite(attr: i64, mem_type: i64, buf: *const std::ffi::c_void) -> c_int;
    fn H5Aclose(id: i64) -> c_int;
    fn H5Oget_info_by_name3(
        loc_id: i64,
        name: *const c_char,
        oinfo: *mut ObjInfo,
        fields: c_uint,
        lapl_id: i64,
    ) -> c_int;
}

unsafe extern "C" {
    static H5P_CLS_GROUP_CREATE_ID_g: i64;
    static H5P_CLS_DATASET_CREATE_ID_g: i64;
    static H5T_NATIVE_INT_g: i64;
}

/// `H5O_info2_t`: file number, object token, type, reference count, the four
/// timestamps, and the attribute count.
///
/// `c_ulong` rather than `u64` for `fileno` on purpose — it is `unsigned long` in
/// C, which is 32 bits on Windows and 64 elsewhere, and every field after it
/// would shift by four bytes if this said `u64`. The timestamp assertions below
/// are what check the layout: a field read at the wrong offset lands in the token
/// or the attribute count and cannot pass a "this is within the last few seconds"
/// test.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
struct ObjInfo {
    fileno: std::ffi::c_ulong,
    token: [u8; 16],
    otype: c_int,
    rc: c_uint,
    atime: i64,
    mtime: i64,
    ctime: i64,
    btime: i64,
    num_attrs: u64,
}

const H5P_DEFAULT: i64 = 0;
const H5F_ACC_TRUNC: c_uint = 0x0002;
const H5F_ACC_RDONLY: c_uint = 0x0000;
const H5S_SCALAR: c_int = 0;
/// `H5O_INFO_BASIC | H5O_INFO_TIME`.
const H5O_INFO_BASIC_AND_TIME: c_uint = 0x0001 | 0x0002;

/// A phase-change pair the C library would never write by default (its defaults
/// are 8 and 6), and one it therefore stores in the header prefix rather than
/// leaving implied.
const MAX_COMPACT: c_uint = 32;
const MIN_DENSE: c_uint = 24;

fn cstr(s: &str) -> CString {
    CString::new(s).expect("a test name holds no NUL")
}

// `hdf5-metno` serializes its own C calls through an internal lock, and the raw
// FFI above bypasses it. Serialize every C-library call in this file through one
// mutex so a raw call never races a concurrent libhdf5 call in another test (the
// C library is not built thread-safe here). Poisoning is ignored: a panic in one
// test must not cascade into the others.
static C_LIB: Mutex<()> = Mutex::new(());

/// Hold the C-library lock, and force the library's global property-list class
/// ids to be initialized: reading them before `H5open` yields zero, and every
/// `H5Pcreate` then fails.
fn c_lib_guard() -> MutexGuard<'static, ()> {
    let guard = C_LIB.lock().unwrap_or_else(|e| e.into_inner());
    hdf5::library_version();
    guard
}

/// Both objects of the fixture, so every assertion covers a group and a dataset —
/// the two headers the editor rebuilds by different routes.
const OBJECTS: [&str; 2] = ["/g", "/d"];

/// Write a file whose group `/g` and dataset `/d` each carry one integer
/// attribute and a non-default attribute phase-change pair.
///
/// Nothing here asks for timestamps: the C library stores them on every version 2
/// header it writes, which is the whole point.
fn write_fixture(path: &Path) {
    let _c = c_lib_guard();
    // Safety: every id below is produced by the matching C call and closed here.
    unsafe {
        let name = cstr(path.to_str().expect("a temp path is UTF-8"));
        let file = H5Fcreate(name.as_ptr(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        assert!(file > 0, "create {}", path.display());

        let gcpl = H5Pcreate(H5P_CLS_GROUP_CREATE_ID_g);
        assert_eq!(H5Pset_attr_phase_change(gcpl, MAX_COMPACT, MIN_DENSE), 0);
        let gname = cstr("g");
        let group = H5Gcreate2(file, gname.as_ptr(), H5P_DEFAULT, gcpl, H5P_DEFAULT);
        assert!(group > 0, "create group");

        let dcpl = H5Pcreate(H5P_CLS_DATASET_CREATE_ID_g);
        assert_eq!(H5Pset_attr_phase_change(dcpl, MAX_COMPACT, MIN_DENSE), 0);
        let dims = [4u64];
        let dspace = H5Screate_simple(1, dims.as_ptr(), std::ptr::null());
        let dname = cstr("d");
        let dataset = H5Dcreate2(
            file,
            dname.as_ptr(),
            H5T_NATIVE_INT_g,
            dspace,
            H5P_DEFAULT,
            dcpl,
            H5P_DEFAULT,
        );
        assert!(dataset > 0, "create dataset");

        let scalar = H5Screate(H5S_SCALAR);
        for owner in [group, dataset] {
            let an = cstr("kept");
            let attr = H5Acreate2(
                owner,
                an.as_ptr(),
                H5T_NATIVE_INT_g,
                scalar,
                H5P_DEFAULT,
                H5P_DEFAULT,
            );
            assert!(attr > 0, "create attribute");
            let value = 1i32;
            assert_eq!(
                H5Awrite(attr, H5T_NATIVE_INT_g, (&raw const value).cast()),
                0
            );
            H5Aclose(attr);
        }
        H5Sclose(scalar);
        H5Sclose(dspace);
        H5Dclose(dataset);
        H5Gclose(group);
        H5Pclose(dcpl);
        H5Pclose(gcpl);
        H5Fclose(file);
    }
}

/// The object info the C library reports for `object` in `path`.
fn object_info(path: &Path, object: &str) -> ObjInfo {
    let _c = c_lib_guard();
    let name = cstr(path.to_str().expect("a temp path is UTF-8"));
    let oname = cstr(object);
    // Safety: the file id is closed below, and `info` is written by the library.
    unsafe {
        let file = H5Fopen(name.as_ptr(), H5F_ACC_RDONLY, H5P_DEFAULT);
        assert!(file > 0, "the C library opens {}", path.display());
        let mut info = ObjInfo::default();
        let rc = H5Oget_info_by_name3(
            file,
            oname.as_ptr(),
            &raw mut info,
            H5O_INFO_BASIC_AND_TIME,
            H5P_DEFAULT,
        );
        assert_eq!(rc, 0, "the C library reads info for {object}");
        H5Fclose(file);
        info
    }
}

/// The attribute phase-change thresholds the C library reports for `object`'s
/// creation property list.
fn phase_change(path: &Path, object: &str) -> (c_uint, c_uint) {
    let _c = c_lib_guard();
    let name = cstr(path.to_str().expect("a temp path is UTF-8"));
    let oname = cstr(object);
    // Safety: every id below is produced by the matching C call and closed here.
    unsafe {
        let file = H5Fopen(name.as_ptr(), H5F_ACC_RDONLY, H5P_DEFAULT);
        assert!(file > 0, "the C library opens {}", path.display());
        let is_group = object == "/g";
        let id = if is_group {
            H5Gopen2(file, oname.as_ptr(), H5P_DEFAULT)
        } else {
            H5Dopen2(file, oname.as_ptr(), H5P_DEFAULT)
        };
        assert!(id > 0, "the C library opens {object}");
        let plist = if is_group {
            H5Gget_create_plist(id)
        } else {
            H5Dget_create_plist(id)
        };
        assert!(plist > 0, "the C library reads {object}'s creation plist");
        let mut max_compact = 0;
        let mut min_dense = 0;
        assert_eq!(
            H5Pget_attr_phase_change(plist, &raw mut max_compact, &raw mut min_dense),
            0,
            "the C library reads {object}'s phase-change thresholds",
        );
        H5Pclose(plist);
        if is_group {
            H5Gclose(id);
        } else {
            H5Dclose(id);
        }
        H5Fclose(file);
        (max_compact, min_dense)
    }
}

fn now_secs() -> i64 {
    i64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("the test host's clock is after 1970")
            .as_secs(),
    )
    .expect("seconds since the epoch fit an i64")
}

/// Editing an attribute in place rebuilds both object headers. Their four
/// timestamps and their attribute phase-change thresholds have to come out the
/// other side, with the modification and change times moved to the edit.
#[test]
fn an_in_place_edit_keeps_a_headers_times_and_phase_change_thresholds() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("t.h5");
    write_fixture(&p);

    let before: Vec<ObjInfo> = OBJECTS.iter().map(|o| object_info(&p, o)).collect();
    for (object, info) in OBJECTS.iter().zip(&before) {
        assert!(
            info.btime > 0,
            "{object} was written without timestamps, so this fixture proves nothing: {info:?}",
        );
        assert_eq!(
            phase_change(&p, object),
            (MAX_COMPACT, MIN_DENSE),
            "{object} was written without the phase-change block",
        );
    }

    // The stored timestamps are whole seconds, so the fixture's own times have to
    // land in a strictly earlier second than the edit for "the edit moved this
    // one" and "the edit left that one alone" to be different claims. One second
    // of wall clock buys both; there is no other way to age a time the C library
    // stamps itself.
    std::thread::sleep(std::time::Duration::from_millis(1100));
    let edit_started = now_secs();

    {
        let s = File::open_rw(&p).expect("the editor opens the fixture");
        s.group("g")
            .expect("the group is reachable")
            .set_attr("added", AttrValue::I64(7))
            .expect("the group takes an attribute");
        s.dataset("d")
            .expect("the dataset is reachable")
            .set_attr("added", AttrValue::I64(7))
            .expect("the dataset takes an attribute");
        s.commit().expect("the commit lands");
    }
    let edit_ended = now_secs();

    for (object, was) in OBJECTS.iter().zip(&before) {
        let now = object_info(&p, object);
        assert_eq!(
            now.btime, was.btime,
            "{object} lost or moved its birth time; the header prefix was dropped",
        );
        assert_eq!(
            now.atime, was.atime,
            "{object} moved its access time, which a rewrite is not",
        );
        assert!(
            (edit_started..=edit_ended).contains(&now.mtime),
            "{object} reports modification time {} outside the edit's [{edit_started}, {edit_ended}]",
            now.mtime,
        );
        assert!(
            (edit_started..=edit_ended).contains(&now.ctime),
            "{object} reports change time {} outside the edit's [{edit_started}, {edit_ended}]",
            now.ctime,
        );
        assert_eq!(
            phase_change(&p, object),
            (MAX_COMPACT, MIN_DENSE),
            "{object} lost the attribute phase-change thresholds the header stored",
        );
    }

    // The edit itself did what it said, so none of the above is a verdict on a
    // file that quietly lost its attributes.
    let f = File::open(&p).unwrap();
    for attrs in [
        f.group("g").unwrap().attrs().unwrap(),
        f.dataset("d").unwrap().attrs().unwrap(),
    ] {
        assert_eq!(attrs.get("added"), Some(&AttrValue::I64(7)));
        assert_eq!(attrs.get("kept"), Some(&AttrValue::I32(1)));
    }
}
