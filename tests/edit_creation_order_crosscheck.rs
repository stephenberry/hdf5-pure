// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit little-endian targets; skip them elsewhere so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
//! In-place edits of objects that track attribute creation order (issue #416).
//!
//! `H5Pset_attr_creation_order` — h5py's `track_order=True`, and what netCDF-4
//! sets on every object it writes — makes every object-header message record six
//! bytes wide instead of four, carrying a creation index. `EditSession` used to
//! refuse such an object outright.
//!
//! The fixtures have to come from the reference C library: nothing in this
//! crate's whole-file writer emits creation order, so a fixture written here
//! would only test the editor against itself. And the interesting assertion is
//! not "the attributes are still there" — this crate's own reader ignores
//! creation order entirely, and would say yes to a file whose indexes are
//! nonsense. It is what the C library makes of the order afterwards:
//! `H5Aiterate2` over `H5_INDEX_CRT_ORDER` (h5py's `track_order` iteration) and
//! `H5Aget_info_by_name`'s `corder`, which name the exact index each attribute
//! carries.

use std::ffi::{CString, c_char, c_int, c_uint, c_void};
use std::path::Path;
use std::sync::{Mutex, MutexGuard};

use hdf5_pure::{AttrValue, Error, File};
use tempfile::tempdir;

// The creation-order entry points, resolved at link time from the statically
// linked libhdf5. `hdf5-metno` exposes no way to set a creation-order property
// or to iterate by creation order, and both are what these fixtures are about.
unsafe extern "C" {
    fn H5Pcreate(cls_id: i64) -> i64;
    fn H5Pclose(plist_id: i64) -> c_int;
    fn H5Pset_attr_creation_order(plist_id: i64, crt_order_flags: c_uint) -> c_int;
    fn H5Pset_link_creation_order(plist_id: i64, crt_order_flags: c_uint) -> c_int;
    fn H5Fcreate(name: *const c_char, flags: c_uint, fcpl: i64, fapl: i64) -> i64;
    fn H5Fopen(name: *const c_char, flags: c_uint, fapl: i64) -> i64;
    fn H5Fclose(id: i64) -> c_int;
    fn H5Gcreate2(loc: i64, name: *const c_char, lcpl: i64, gcpl: i64, gapl: i64) -> i64;
    fn H5Gopen2(loc: i64, name: *const c_char, gapl: i64) -> i64;
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
    fn H5Awrite(attr: i64, mem_type: i64, buf: *const c_void) -> c_int;
    fn H5Aread(attr: i64, mem_type: i64, buf: *mut c_void) -> c_int;
    fn H5Aopen(loc: i64, name: *const c_char, aapl: i64) -> i64;
    fn H5Aopen_by_idx(
        loc: i64,
        obj_name: *const c_char,
        idx_type: c_int,
        order: c_int,
        n: u64,
        aapl: i64,
        lapl: i64,
    ) -> i64;
    fn H5Aget_name(attr: i64, buf_size: usize, buf: *mut c_char) -> isize;
    fn H5Aclose(id: i64) -> c_int;
    fn H5Aiterate2(
        loc: i64,
        idx_type: c_int,
        order: c_int,
        idx: *mut u64,
        op: extern "C" fn(i64, *const c_char, *const AttrInfo, *mut c_void) -> c_int,
        op_data: *mut c_void,
    ) -> c_int;
    fn H5Aget_info_by_name(
        loc: i64,
        obj_name: *const c_char,
        attr_name: *const c_char,
        info: *mut AttrInfo,
        lapl: i64,
    ) -> c_int;
}

unsafe extern "C" {
    static H5P_CLS_FILE_CREATE_ID_g: i64;
    static H5P_CLS_GROUP_CREATE_ID_g: i64;
    static H5P_CLS_DATASET_CREATE_ID_g: i64;
    static H5T_NATIVE_INT_g: i64;
}

/// `H5A_info_t`: `corder_valid` (`hbool_t`, one byte, padded), `corder`, `cset`,
/// `data_size`. Only the first two fields are read here, and the layout is
/// checked by the assertions that read `corder` back — a wrong offset would
/// report indexes that do not match the order the attributes were created in.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct AttrInfo {
    corder_valid: u8,
    corder: u32,
    cset: c_int,
    data_size: u64,
}

const H5P_DEFAULT: i64 = 0;
const H5F_ACC_TRUNC: c_uint = 0x0002;
const H5F_ACC_RDONLY: c_uint = 0x0000;
const H5P_CRT_ORDER_TRACKED: c_uint = 0x0001;
const H5P_CRT_ORDER_INDEXED: c_uint = 0x0002;
const H5S_SCALAR: c_int = 0;
const H5_INDEX_NAME: c_int = 0;
const H5_INDEX_CRT_ORDER: c_int = 1;
const H5_ITER_INC: c_int = 0;

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

/// Whether the object creation property list should also index creation order,
/// which is what adds the creation-order B-tree once attributes go dense.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Indexed {
    No,
    Yes,
}

impl Indexed {
    fn flags(self) -> c_uint {
        match self {
            Self::No => H5P_CRT_ORDER_TRACKED,
            Self::Yes => H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED,
        }
    }
}

/// Write a file whose group `/g` and dataset `/d` both track attribute creation
/// order, each carrying one integer attribute per name in `names`, created in
/// that order and valued by its position.
///
/// The file creation property list carries the same setting so the root group
/// tracks it too, which is what h5py's `File(..., track_order=True)` does.
fn write_tracked(path: &Path, names: &[String], indexed: Indexed) {
    let _c = c_lib_guard();
    unsafe {
        let fcpl = H5Pcreate(H5P_CLS_FILE_CREATE_ID_g);
        assert!(fcpl > 0, "file creation property list");
        assert_eq!(H5Pset_attr_creation_order(fcpl, indexed.flags()), 0);
        let name = cstr(path.to_str().expect("a temp path is UTF-8"));
        let file = H5Fcreate(name.as_ptr(), H5F_ACC_TRUNC, fcpl, H5P_DEFAULT);
        assert!(file > 0, "create {}", path.display());

        let gcpl = H5Pcreate(H5P_CLS_GROUP_CREATE_ID_g);
        assert_eq!(H5Pset_attr_creation_order(gcpl, indexed.flags()), 0);
        let gname = cstr("g");
        let group = H5Gcreate2(file, gname.as_ptr(), H5P_DEFAULT, gcpl, H5P_DEFAULT);
        assert!(group > 0, "create group");

        let dcpl = H5Pcreate(H5P_CLS_DATASET_CREATE_ID_g);
        assert_eq!(H5Pset_attr_creation_order(dcpl, indexed.flags()), 0);
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
            for (i, name) in names.iter().enumerate() {
                let an = cstr(name);
                let attr = H5Acreate2(
                    owner,
                    an.as_ptr(),
                    H5T_NATIVE_INT_g,
                    scalar,
                    H5P_DEFAULT,
                    H5P_DEFAULT,
                );
                assert!(attr > 0, "create attribute {name}");
                let value = i as i32;
                assert_eq!(
                    H5Awrite(attr, H5T_NATIVE_INT_g, (&raw const value).cast()),
                    0
                );
                H5Aclose(attr);
            }
        }
        H5Sclose(scalar);
        H5Sclose(dspace);
        H5Dclose(dataset);
        H5Gclose(group);
        H5Pclose(dcpl);
        H5Pclose(gcpl);
        H5Fclose(file);
        H5Pclose(fcpl);
    }
}

/// A file whose group `/g` tracks *link* creation order, as netCDF-4 writes.
fn write_link_tracked(path: &Path) {
    let _c = c_lib_guard();
    unsafe {
        let fcpl = H5Pcreate(H5P_CLS_FILE_CREATE_ID_g);
        let flags = H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED;
        assert_eq!(H5Pset_link_creation_order(fcpl, flags), 0);
        assert_eq!(H5Pset_attr_creation_order(fcpl, flags), 0);
        let name = cstr(path.to_str().expect("a temp path is UTF-8"));
        let file = H5Fcreate(name.as_ptr(), H5F_ACC_TRUNC, fcpl, H5P_DEFAULT);
        assert!(file > 0, "create {}", path.display());

        let gcpl = H5Pcreate(H5P_CLS_GROUP_CREATE_ID_g);
        assert_eq!(H5Pset_link_creation_order(gcpl, flags), 0);
        assert_eq!(H5Pset_attr_creation_order(gcpl, flags), 0);
        let gname = cstr("g");
        let group = H5Gcreate2(file, gname.as_ptr(), H5P_DEFAULT, gcpl, H5P_DEFAULT);
        assert!(group > 0, "create group");

        let dims = [4u64];
        let dspace = H5Screate_simple(1, dims.as_ptr(), std::ptr::null());
        let dname = cstr("d");
        let dataset = H5Dcreate2(
            group,
            dname.as_ptr(),
            H5T_NATIVE_INT_g,
            dspace,
            H5P_DEFAULT,
            H5P_DEFAULT,
            H5P_DEFAULT,
        );
        assert!(dataset > 0, "create dataset");
        H5Dclose(dataset);
        H5Sclose(dspace);
        H5Gclose(group);
        H5Pclose(gcpl);
        H5Fclose(file);
        H5Pclose(fcpl);
    }
}

/// Collect attribute names into the `Vec<String>` `op_data` points at.
extern "C" fn collect(
    _loc: i64,
    name: *const c_char,
    _info: *const AttrInfo,
    op_data: *mut c_void,
) -> c_int {
    // Safety: every caller passes a `&mut Vec<String>` as `op_data`, and the C
    // library hands back a NUL-terminated attribute name.
    unsafe {
        let names = &mut *op_data.cast::<Vec<String>>();
        names.push(
            std::ffi::CStr::from_ptr(name)
                .to_string_lossy()
                .into_owned(),
        );
    }
    0
}

/// An open handle to `/g` or `/d`, whichever `path` names, closed on drop.
struct Owner {
    file: i64,
    id: i64,
    is_group: bool,
}

impl Owner {
    fn open(path: &Path, object: &str) -> Self {
        let _c = c_lib_guard();
        let name = cstr(path.to_str().expect("a temp path is UTF-8"));
        // Safety: the ids come from the C library and are closed in `drop`.
        unsafe {
            let file = H5Fopen(name.as_ptr(), H5F_ACC_RDONLY, H5P_DEFAULT);
            assert!(file > 0, "the C library opens {}", path.display());
            let oname = cstr(object);
            let is_group = object == "g";
            let id = if is_group {
                H5Gopen2(file, oname.as_ptr(), H5P_DEFAULT)
            } else {
                H5Dopen2(file, oname.as_ptr(), H5P_DEFAULT)
            };
            assert!(id > 0, "the C library opens /{object}");
            Self { file, id, is_group }
        }
    }

    /// Attribute names in the order `idx_type` orders them.
    fn names(&self, idx_type: c_int) -> Vec<String> {
        let _c = c_lib_guard();
        let mut names: Vec<String> = Vec::new();
        let mut idx = 0u64;
        // Safety: `collect` interprets `op_data` as the `Vec<String>` passed here.
        let rc = unsafe {
            H5Aiterate2(
                self.id,
                idx_type,
                H5_ITER_INC,
                &raw mut idx,
                collect,
                (&raw mut names).cast(),
            )
        };
        assert_eq!(rc, 0, "the C library iterates attributes");
        names
    }

    /// The name of the `n`th attribute in creation order, read through
    /// `H5Aopen_by_idx`.
    ///
    /// Distinct from [`names`](Self::names) on purpose: `H5Aiterate2` with an
    /// increasing order builds its table from the *name* index whatever the
    /// requested index type, so it reads the creation index out of the name
    /// index's records and never touches the creation-order B-tree. This walks
    /// `H5A__dense_open_by_idx`, which indexes straight into that B-tree
    /// whenever the Attribute Info message names one — so a dense object whose
    /// message declares the index has to actually carry it.
    fn name_by_creation_index(&self, n: u64) -> String {
        let _c = c_lib_guard();
        let here = cstr(".");
        // Safety: the ids come from the C library and are closed below; the
        // buffer is sized by the length the library reports for the name.
        unsafe {
            let attr = H5Aopen_by_idx(
                self.id,
                here.as_ptr(),
                H5_INDEX_CRT_ORDER,
                H5_ITER_INC,
                n,
                H5P_DEFAULT,
                H5P_DEFAULT,
            );
            assert!(
                attr > 0,
                "the C library opens attribute {n} by creation order"
            );
            let len = H5Aget_name(attr, 0, std::ptr::null_mut());
            assert!(len > 0, "the C library reports a name length");
            let mut buf = vec![0u8; len as usize + 1];
            let got = H5Aget_name(attr, buf.len(), buf.as_mut_ptr().cast());
            assert_eq!(got, len, "the C library reads the name");
            H5Aclose(attr);
            buf.truncate(len as usize);
            String::from_utf8(buf).expect("a fixture name is UTF-8")
        }
    }

    /// The value of the integer attribute `name`.
    fn value(&self, name: &str) -> i32 {
        let _c = c_lib_guard();
        let an = cstr(name);
        // Safety: the attribute is a scalar native int, matching the read type.
        unsafe {
            let attr = H5Aopen(self.id, an.as_ptr(), H5P_DEFAULT);
            assert!(attr > 0, "the C library opens attribute {name}");
            let mut value = 0i32;
            assert_eq!(H5Aread(attr, H5T_NATIVE_INT_g, (&raw mut value).cast()), 0);
            H5Aclose(attr);
            value
        }
    }
}

impl Drop for Owner {
    fn drop(&mut self) {
        let _c = c_lib_guard();
        // Safety: both ids were produced by the matching open calls above.
        unsafe {
            if self.is_group {
                H5Gclose(self.id);
            } else {
                H5Dclose(self.id);
            }
            H5Fclose(self.file);
        }
    }
}

/// The creation index the C library reports for attribute `attr` of `/object`.
fn creation_index(path: &Path, object: &str, attr: &str) -> u32 {
    let _c = c_lib_guard();
    let name = cstr(path.to_str().expect("a temp path is UTF-8"));
    let oname = cstr(object);
    let aname = cstr(attr);
    // Safety: the file id is closed below, and `info` is written by the library.
    unsafe {
        let file = H5Fopen(name.as_ptr(), H5F_ACC_RDONLY, H5P_DEFAULT);
        assert!(file > 0, "the C library opens {}", path.display());
        let mut info = AttrInfo::default();
        let rc = H5Aget_info_by_name(
            file,
            oname.as_ptr(),
            aname.as_ptr(),
            &raw mut info,
            H5P_DEFAULT,
        );
        assert_eq!(rc, 0, "the C library reads info for {object}/{attr}");
        H5Fclose(file);
        assert!(
            info.corder_valid != 0,
            "{object}/{attr} carries no creation index, so the object stopped tracking the order",
        );
        info.corder
    }
}

fn names(count: usize) -> Vec<String> {
    (0..count).map(|i| format!("a{i:02}")).collect()
}

/// Both objects of a fixture, so every test covers a group and a dataset — the
/// two headers the editor rebuilds by different routes.
const OBJECTS: [&str; 2] = ["g", "d"];

/// Set `name` on both `/g` and `/d` through one edit session.
fn set_on_both(path: &Path, name: &str, value: i64) {
    let s = File::open_rw(path).expect("the editor opens a tracked file");
    s.group("g")
        .expect("the group is reachable")
        .set_attr(name, AttrValue::I64(value))
        .expect("the group takes an attribute");
    s.dataset("d")
        .expect("the dataset is reachable")
        .set_attr(name, AttrValue::I64(value))
        .expect("the dataset takes an attribute");
    s.commit().expect("the commit lands");
}

#[test]
fn an_object_tracking_creation_order_can_be_edited_at_all() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("t.h5");
    write_tracked(&p, &names(3), Indexed::Yes);

    set_on_both(&p, "added", 99);

    for object in OBJECTS {
        let owner = Owner::open(&p, object);
        assert_eq!(
            owner.names(H5_INDEX_NAME),
            ["a00", "a01", "a02", "added"],
            "/{object} lost an attribute by name",
        );
        // The new attribute takes the next creation index, so it iterates last —
        // which is exactly what h5py's `track_order` iteration shows.
        assert_eq!(
            owner.names(H5_INDEX_CRT_ORDER),
            ["a00", "a01", "a02", "added"],
            "/{object} did not put the new attribute last in creation order",
        );
        assert_eq!(owner.value("added"), 99);
        assert_eq!(owner.value("a01"), 1, "/{object} lost an existing value");
    }
    assert_eq!(creation_index(&p, "g", "added"), 3);
    assert_eq!(creation_index(&p, "d", "added"), 3);
}

#[test]
fn the_editor_reads_a_tracked_object_back_itself() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("t.h5");
    write_tracked(&p, &names(3), Indexed::Yes);
    set_on_both(&p, "added", 7);

    let f = File::open(&p).unwrap();
    let attrs = f.dataset("d").unwrap().attrs().unwrap();
    assert_eq!(attrs.len(), 4);
    assert_eq!(attrs.get("added"), Some(&AttrValue::I64(7)));
    assert_eq!(f.dataset("d").unwrap().read_i32().unwrap().len(), 4);
}

#[test]
fn overwriting_an_attribute_keeps_the_creation_index_it_had() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("t.h5");
    write_tracked(&p, &names(3), Indexed::Yes);

    set_on_both(&p, "a00", -1);

    for object in OBJECTS {
        let owner = Owner::open(&p, object);
        assert_eq!(
            owner.names(H5_INDEX_CRT_ORDER),
            ["a00", "a01", "a02"],
            "/{object} moved an overwritten attribute in the creation order",
        );
        assert_eq!(owner.value("a00"), -1);
    }
    assert_eq!(creation_index(&p, "g", "a00"), 0);
    assert_eq!(creation_index(&p, "d", "a00"), 0);
}

#[test]
fn deleting_an_attribute_leaves_a_gap_rather_than_renumbering() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("t.h5");
    write_tracked(&p, &names(4), Indexed::Yes);

    // One from the middle and one from the end: the middle deletion is what
    // leaves a gap in the surviving indexes, and the end one is what separates
    // the counter the Attribute Info message records from the highest index
    // still in use.
    {
        let s = File::open_rw(&p).unwrap();
        let g = s.group("g").unwrap();
        let mut d = s.dataset("d").unwrap();
        for name in ["a01", "a03"] {
            g.remove_attr(name).unwrap();
            d.remove_attr(name).unwrap();
        }
        s.commit().unwrap();
    }
    // A fresh attribute takes index 4, neither a freed index nor one past the
    // highest survivor: the reference C library hands them out from a counter
    // that only rises.
    set_on_both(&p, "added", 5);

    for object in OBJECTS {
        let owner = Owner::open(&p, object);
        assert_eq!(owner.names(H5_INDEX_CRT_ORDER), ["a00", "a02", "added"]);
        assert_eq!(creation_index(&p, object, "a00"), 0);
        assert_eq!(
            creation_index(&p, object, "a02"),
            2,
            "/{object} renumbered the attributes that survived",
        );
        assert_eq!(
            creation_index(&p, object, "added"),
            4,
            "/{object} lowered the creation-index counter a deletion must leave alone",
        );
    }
}

#[test]
fn a_compact_set_crossing_the_threshold_carries_its_creation_order_into_the_heap() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("t.h5");
    // Eight compact attributes; one is deleted and six more added in one
    // session. That is past the writer's compact threshold, so the whole set is
    // rebuilt into a fractal heap — and the deletion is what makes each
    // attribute's creation index differ from its position in the rebuilt set,
    // so an index taken from the position would show up here.
    write_tracked(&p, &names(8), Indexed::Yes);

    {
        let s = File::open_rw(&p).unwrap();
        let g = s.group("g").unwrap();
        let mut d = s.dataset("d").unwrap();
        g.remove_attr("a01").unwrap();
        d.remove_attr("a01").unwrap();
        for i in 8..14 {
            g.set_attr(&format!("a{i:02}"), AttrValue::I64(i as i64))
                .unwrap();
            d.set_attr(&format!("a{i:02}"), AttrValue::I64(i as i64))
                .unwrap();
        }
        s.commit().unwrap();
    }

    let expected: Vec<String> = names(14).into_iter().filter(|n| n != "a01").collect();
    for object in OBJECTS {
        let owner = Owner::open(&p, object);
        assert_eq!(
            owner.names(H5_INDEX_CRT_ORDER),
            expected,
            "/{object} lost the creation order on the way into the heap",
        );
        assert_eq!(owner.names(H5_INDEX_NAME), expected);
        // Straight through the creation-order B-tree, which this object's
        // Attribute Info message declares.
        for (n, name) in expected.iter().enumerate() {
            assert_eq!(
                &owner.name_by_creation_index(n as u64),
                name,
                "/{object} does not carry the creation-order index it declares",
            );
        }
        for name in &expected {
            let created: u32 = name[1..].parse().expect("a fixture name is a index");
            assert_eq!(owner.value(name), created as i32, "/{object} {name}");
            assert_eq!(
                creation_index(&p, object, name),
                created,
                "/{object} gave {name} an index that is not the order it was created in",
            );
        }
    }
}

#[test]
fn a_tracked_set_goes_dense_without_a_creation_order_index_when_the_object_has_none() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("t.h5");
    // Tracked but not *indexed*: the heap gets a name index only, and the
    // creation index each attribute carries lives in that index's records.
    write_tracked(&p, &names(8), Indexed::No);

    {
        let s = File::open_rw(&p).unwrap();
        let mut d = s.dataset("d").unwrap();
        for i in 8..13 {
            d.set_attr(&format!("a{i:02}"), AttrValue::I64(i as i64))
                .unwrap();
        }
        s.commit().unwrap();
    }

    let owner = Owner::open(&p, "d");
    assert_eq!(owner.names(H5_INDEX_NAME), names(13));
    for (i, name) in names(13).iter().enumerate() {
        assert_eq!(creation_index(&p, "d", name), i as u32);
    }
}

#[test]
fn an_object_already_dense_keeps_its_creation_order_across_an_edit() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("t.h5");
    // Twelve attributes is past the C library's own compact threshold, so both
    // objects already store their attributes in a fractal heap.
    write_tracked(&p, &names(12), Indexed::Yes);

    set_on_both(&p, "added", 42);

    let mut expected: Vec<String> = names(12);
    expected.push("added".to_string());
    for object in OBJECTS {
        let owner = Owner::open(&p, object);
        assert_eq!(
            owner.names(H5_INDEX_CRT_ORDER),
            expected,
            "/{object} reordered a dense set that was already tracked",
        );
        assert_eq!(owner.value("added"), 42);
        for (n, name) in expected.iter().enumerate() {
            assert_eq!(
                &owner.name_by_creation_index(n as u64),
                name,
                "/{object} does not carry the creation-order index it declares",
            );
        }
        for (i, name) in names(12).iter().enumerate() {
            assert_eq!(creation_index(&p, object, name), i as u32);
        }
        assert_eq!(creation_index(&p, object, "added"), 12);
    }
}

#[test]
fn a_group_tracking_link_creation_order_refuses_a_new_child() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("t.h5");
    write_link_tracked(&p);

    // Its attributes are editable — the object header's creation-order tracking
    // is what this change lifted.
    {
        let s = File::open_rw(&p).unwrap();
        s.group("g")
            .unwrap()
            .set_attr("note", AttrValue::I64(1))
            .unwrap();
        s.commit().unwrap();
    }
    assert_eq!(Owner::open(&p, "g").value("note"), 1);

    // Its membership is not: a Link message this crate emits carries no creation
    // index, so adding one would leave an iteration by link creation order
    // walking an unnumbered link.
    let s = File::open_rw(&p).unwrap();
    let g = s.group("g").unwrap();
    g.create_dataset("extra", |d| {
        d.with_i32_data(&[1, 2]);
    })
    .unwrap();
    let err = s.commit().unwrap_err();
    assert!(
        matches!(&err, Error::EditUnsupported(m) if m.contains("link creation order")),
        "got: {err}",
    );
}
