// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// gated to 64-bit-pointer targets.
#![cfg(not(target_pointer_width = "32"))]
//! Committed (`H5Tcommit`) datatypes, written by the reference C library and read
//! back by hdf5-pure (issue #254).
//!
//! A committed datatype is stored in its own object header, and every object that
//! uses it stores a *reference* to that header in place of the type. Nothing in
//! the referring bytes says so — a version 2 reference to address `0x320` decodes
//! as a perfectly well-formed zero-width time datatype — so the only thing that
//! separates a reference from an encoding is a flag bit, and reading the bytes
//! without it returns the wrong type with no error anywhere.
//!
//! This crate's writer cannot emit a committed type, so the fixtures here are
//! built through the C library's own `H5Tcommit2`, and the type each object
//! *should* report is the one the C library reads back from the same file.

use std::ffi::{CString, c_char, c_int, c_void};
use std::path::Path;
use std::sync::{Mutex, MutexGuard};

use hdf5_pure::{AttrValue, Datatype, File, RepackOptions};
use tempfile::tempdir;

// The committed-datatype entry points, resolved at link time from the statically
// linked libhdf5. `hdf5-metno` exposes no way to commit a datatype, and a
// committed type is precisely what these fixtures need.
unsafe extern "C" {
    fn H5Tcommit2(
        loc_id: i64,
        name: *const c_char,
        type_id: i64,
        lcpl_id: i64,
        tcpl_id: i64,
        tapl_id: i64,
    ) -> c_int;
    fn H5Screate_simple(rank: c_int, dims: *const u64, maxdims: *const u64) -> i64;
    fn H5Sclose(space_id: i64) -> c_int;
    fn H5Acreate2(
        loc_id: i64,
        attr_name: *const c_char,
        type_id: i64,
        space_id: i64,
        acpl_id: i64,
        aapl_id: i64,
    ) -> i64;
    fn H5Awrite(attr_id: i64, mem_type_id: i64, buf: *const c_void) -> c_int;
    fn H5Aclose(attr_id: i64) -> c_int;
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
        dset_id: i64,
        mem_type_id: i64,
        mem_space_id: i64,
        file_space_id: i64,
        dxpl_id: i64,
        buf: *const c_void,
    ) -> c_int;
    fn H5Dclose(dset_id: i64) -> c_int;
}

/// `H5P_DEFAULT` and `H5S_ALL` are both the zero id.
const DEFAULT: i64 = 0;

// `hdf5-metno` serializes its own C calls through an internal lock, and the raw
// FFI above bypasses it. Serialize every C-library call in this file through one
// mutex so a raw call never races a concurrent libhdf5 call in another test (the
// C library is not built thread-safe here). Poisoning is ignored: a panic in one
// test must not cascade into the others.
static C_LIB: Mutex<()> = Mutex::new(());

fn c_lib_guard() -> MutexGuard<'static, ()> {
    C_LIB.lock().unwrap_or_else(|e| e.into_inner())
}

/// What a fixture puts in the file, so every assertion below names one constant
/// rather than repeating a literal the fixture could drift away from.
const ATTR_VALUE: i32 = 7;
const DATASET_VALUES: [i32; 3] = [10, 20, 30];

/// The i32 type every fixture commits, as this crate decodes it. Committing does
/// not change the encoding — only where it is stored — so the answer a resolved
/// reference must produce is exactly the answer an inline `H5T_STD_I32LE` gives.
fn committed_i32() -> Datatype {
    Datatype::FixedPoint {
        size: 4,
        byte_order: hdf5_pure::DatatypeByteOrder::LittleEndian,
        signed: true,
        bit_offset: 0,
        bit_precision: 32,
    }
}

/// What a fixture puts the committed `/mytype` to use for.
///
/// The two uses are separable on purpose. Repack refuses each of them in its own
/// place, and a fixture carrying both lets either refusal stand in for the other:
/// remove one and the file is still refused, by the other, with every test still
/// green.
#[derive(Clone, Copy, Default)]
struct Fixture {
    /// A `shared_attr` attribute on the root group and on `/data` whose datatype
    /// is the committed one.
    committed_attrs: bool,
    /// A `/typed` dataset whose *element* type is the committed one.
    committed_dataset: bool,
    /// Further committed attributes on the root, enough of them to push its
    /// attribute storage out of the object header and into a fractal heap — a
    /// separate decode path with the same defect.
    dense_attrs: usize,
}

/// Everything the committed type can be used for, which is what the read-side
/// tests want: one file exercising every path.
const EVERYTHING: Fixture = Fixture {
    committed_attrs: true,
    committed_dataset: true,
    dense_attrs: 0,
};

/// Write a file whose `/mytype` is a committed i32, put to the uses `fixture`
/// names. `/data` and the ordinary `plain` attribute beside each committed one
/// are always present.
fn write_committed_fixture(path: &Path, fixture: Fixture) {
    let file = hdf5::File::create(path).expect("create fixture");
    // Dropped before `file`, as HDF5 requires every id in a file to be released
    // before the file itself.
    let dtype = hdf5::Datatype::from_type::<i32>().expect("transient i32 type");

    let name = CString::new("mytype").unwrap();
    let rc = unsafe {
        H5Tcommit2(
            file.id(),
            name.as_ptr(),
            dtype.id(),
            DEFAULT,
            DEFAULT,
            DEFAULT,
        )
    };
    assert!(rc >= 0, "H5Tcommit2 failed");

    let one = [1u64];
    let scalar_space = unsafe { H5Screate_simple(1, one.as_ptr(), std::ptr::null()) };
    assert!(scalar_space >= 0, "H5Screate_simple failed");

    // An ordinary dataset to hang attributes on, written through the safe API.
    file.new_dataset::<f64>()
        .shape([1])
        .create("data")
        .expect("create /data")
        .write(&[1.0f64])
        .expect("write /data");
    let data = file.dataset("data").expect("open /data");

    for owner in [file.id(), data.id()] {
        if fixture.committed_attrs {
            write_attr(
                owner,
                "shared_attr",
                Some(dtype.id()),
                scalar_space,
                ATTR_VALUE,
            );
        }
        // An ordinary attribute beside it: the C library abandons an object's
        // whole attribute list when one attribute fails to decode, so a healthy
        // neighbour is what makes that collateral damage visible.
        write_attr(owner, "plain", None, scalar_space, -ATTR_VALUE);
    }

    for i in 0..fixture.dense_attrs {
        write_attr(
            file.id(),
            &format!("dense{i:02}"),
            Some(dtype.id()),
            scalar_space,
            i as i32,
        );
    }

    if fixture.committed_dataset {
        // A dataset whose *element* type is the committed one.
        let three = [3u64];
        let vector_space = unsafe { H5Screate_simple(1, three.as_ptr(), std::ptr::null()) };
        assert!(vector_space >= 0, "H5Screate_simple failed");
        let typed_name = CString::new("typed").unwrap();
        let dset = unsafe {
            H5Dcreate2(
                file.id(),
                typed_name.as_ptr(),
                dtype.id(),
                vector_space,
                DEFAULT,
                DEFAULT,
                DEFAULT,
            )
        };
        assert!(dset >= 0, "H5Dcreate2 failed");
        let rc = unsafe {
            H5Dwrite(
                dset,
                dtype.id(),
                DEFAULT,
                DEFAULT,
                DEFAULT,
                DATASET_VALUES.as_ptr().cast::<c_void>(),
            )
        };
        assert!(rc >= 0, "H5Dwrite failed");
        unsafe {
            H5Dclose(dset);
            H5Sclose(vector_space);
        }
    }

    unsafe { H5Sclose(scalar_space) };
}

/// Create an i32 attribute on `owner`. `committed` names the committed type when
/// the attribute is to reference one; `None` uses a fresh transient copy, whose
/// encoding the C library stores inline in the message.
fn write_attr(owner: i64, name: &str, committed: Option<i64>, space: i64, value: i32) {
    let transient = hdf5::Datatype::from_type::<i32>().expect("transient i32 type");
    let type_id = committed.unwrap_or_else(|| transient.id());
    let cname = CString::new(name).unwrap();
    let attr = unsafe { H5Acreate2(owner, cname.as_ptr(), type_id, space, DEFAULT, DEFAULT) };
    assert!(attr >= 0, "H5Acreate2 failed for {name}");
    let rc = unsafe {
        H5Awrite(
            attr,
            transient.id(),
            std::ptr::from_ref(&value).cast::<c_void>(),
        )
    };
    assert!(rc >= 0, "H5Awrite failed for {name}");
    unsafe { H5Aclose(attr) };
}

/// An attribute whose datatype is a committed one reports the type it names, on
/// the root group and on a dataset alike — and reports its value, which the
/// zero-width type the reference used to decode as made unreadable.
#[test]
fn a_committed_attribute_datatype_resolves_to_the_type_it_names() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("committed.h5");
    write_committed_fixture(&path, EVERYTHING);

    let file = File::open(&path).unwrap();
    for (owner, datatypes, attrs) in [
        (
            "root group",
            file.root().attr_datatypes().unwrap(),
            file.root().attrs().unwrap(),
        ),
        (
            "/data",
            file.dataset("data").unwrap().attr_datatypes().unwrap(),
            file.dataset("data").unwrap().attrs().unwrap(),
        ),
    ] {
        assert_eq!(
            datatypes.get("shared_attr"),
            Some(&committed_i32()),
            "{owner}: a committed attribute datatype was not resolved"
        );
        assert_eq!(
            attrs.get("shared_attr"),
            Some(&AttrValue::I64Array(vec![i64::from(ATTR_VALUE)])),
            "{owner}: a committed attribute's value was dropped"
        );
        // The ordinary attribute beside it is unaffected. Both fixtures' attributes
        // share one rank-1 dataspace, so both arrive as arrays.
        assert_eq!(
            attrs.get("plain"),
            Some(&AttrValue::I64Array(vec![i64::from(-ATTR_VALUE)])),
            "{owner}: an inline attribute changed"
        );
    }
}

/// A dataset whose element type is committed reads its elements. The old decode
/// produced a zero-width type, which is the one thing that cannot hold any data
/// at all.
#[test]
fn a_committed_dataset_datatype_resolves_and_its_data_reads() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("committed.h5");
    write_committed_fixture(&path, EVERYTHING);

    let file = File::open(&path).unwrap();
    let typed = file.dataset("typed").unwrap();
    assert_eq!(typed.datatype().unwrap(), committed_i32());
    assert_eq!(typed.read_i32().unwrap(), DATASET_VALUES);
}

/// Dense attribute storage decodes committed datatypes the same way. The bytes
/// come out of a fractal heap rather than the object header, which is a second
/// decode path — and each reader backend walks that heap with its own code, so
/// both are checked here rather than one standing in for the other.
#[test]
fn committed_datatypes_resolve_in_dense_attribute_storage() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("dense.h5");
    // Comfortably past the eight-attribute threshold that moves storage to a heap.
    const DENSE: usize = 40;
    write_committed_fixture(
        &path,
        Fixture {
            dense_attrs: DENSE,
            ..EVERYTHING
        },
    );

    for (backend, file) in [
        ("buffered", File::open(&path).unwrap()),
        ("streaming", File::open_streaming(&path).unwrap()),
    ] {
        let datatypes = file.root().attr_datatypes().unwrap();
        let attrs = file.root().attrs().unwrap();
        assert_eq!(
            datatypes.len(),
            DENSE + 2,
            "{backend}: every attribute must still be reported"
        );
        for i in 0..DENSE {
            let name = format!("dense{i:02}");
            assert_eq!(
                datatypes.get(&name),
                Some(&committed_i32()),
                "{backend}/{name}: a committed datatype was not resolved in dense storage"
            );
            assert_eq!(
                attrs.get(&name),
                Some(&AttrValue::I64Array(vec![i as i64])),
                "{backend}/{name}: a dense committed attribute's value was dropped"
            );
        }
    }
}

/// An in-place edit of an object that carries a committed attribute leaves that
/// attribute alone and still succeeds.
///
/// The editor walks an object-header region with no file context, so it cannot
/// decode the committed attribute — but it does not need to. The reference stays
/// valid inside the same file, and identifying an attribute by name never needs
/// its datatype, so refusing the whole object because one of its neighbours is
/// committed would be a limit with no cause.
#[test]
fn an_edit_passes_over_a_committed_attribute_it_does_not_touch() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("committed.h5");
    write_committed_fixture(&path, EVERYTHING);

    {
        let file = File::open_rw(&path).unwrap();
        file.root().set_attr("added", AttrValue::I64(5)).unwrap();
        file.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let attrs = file.root().attrs().unwrap();
    assert_eq!(attrs.get("added"), Some(&AttrValue::I64(5)));
    assert_eq!(
        attrs.get("shared_attr"),
        Some(&AttrValue::I64Array(vec![i64::from(ATTR_VALUE)])),
        "the committed attribute did not survive an edit beside it"
    );
    assert_eq!(
        file.root().attr_datatypes().unwrap().get("shared_attr"),
        Some(&committed_i32())
    );
    drop(file);

    // And the C library reads the result. This crate accepting its own output
    // proves nothing on its own: one bad attribute makes libhdf5 abandon an
    // object's whole attribute list, which is exactly the damage a rewrite of a
    // committed reference does and exactly what a pure-Rust round trip cannot see.
    let c_file = hdf5::File::open(&path).expect("C library must open the edited file");
    let mut names = c_file.attr_names().expect("C library must list attributes");
    names.sort();
    assert_eq!(
        names,
        vec![
            "added".to_string(),
            "plain".to_string(),
            "shared_attr".to_string()
        ]
    );
    assert_eq!(
        c_file
            .attr("shared_attr")
            .unwrap()
            .read_1d::<i32>()
            .unwrap()
            .to_vec(),
        vec![ATTR_VALUE]
    );
}

/// A cross-file copy of an object carrying a committed attribute is refused.
///
/// The copy moves the attribute's bytes verbatim, and those bytes are an address
/// in the *source* file — nothing in the destination is at it. The object-header
/// record's own shared flag does not report this: it describes the attribute
/// message, not the datatype field inside it, so the screen has to read the
/// attribute's flags byte to see it at all.
#[test]
fn a_cross_file_copy_refuses_a_committed_attribute() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let src_path = dir.path().join("committed.h5");
    let dst_path = dir.path().join("dest.h5");
    write_committed_fixture(&src_path, EVERYTHING);
    hdf5_pure::FileBuilder::new().write(&dst_path).unwrap();

    let source = File::open(&src_path).unwrap();
    let dest = File::open_rw(&dst_path).unwrap();
    let err = dest
        .copy_from(&source, "data", "data")
        .expect_err("a committed attribute must not cross files");
    let message = err.to_string();
    assert!(
        message.contains("committed"),
        "the refusal must name the committed datatype, got: {message}"
    );
}

/// An in-place append to a dataset whose element type is committed is refused.
///
/// The append engine walks the object header itself and sizes each element from
/// the datatype message's bytes. Those bytes are a reference, and they decode as
/// a zero-width type — so without reading the record's shared flag the engine
/// takes its geometry from a type the dataset does not have. A typed append then
/// fails with a misleading complaint that the element type does not match, and
/// [`Dataset::append_raw`], which has no type to compare against, would stride
/// through the data one byte per element. The C library writes this shape for
/// `maxshape=(None,)` on a latest-format file.
#[test]
fn an_in_place_append_refuses_a_committed_element_type() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("appendable.h5");
    write_appendable_committed_fixture(&path);

    let file = File::open_rw(&path).unwrap();
    let mut typed = file.dataset("typed").unwrap();
    let err = typed
        .append(&[40i32])
        .expect_err("an append must not size its elements from a reference");
    let message = err.to_string();
    assert!(
        message.contains("committed (shared) datatype"),
        "the refusal must name the committed datatype, got: {message}"
    );
}

/// A latest-format file whose `/typed` is chunked, unlimited along its one
/// dimension, and typed by the committed `/mytype` — the extensible-array shape
/// this crate's in-place append engine maintains.
fn write_appendable_committed_fixture(path: &Path) {
    let mut fb = hdf5::FileBuilder::new();
    fb.with_fapl(|fapl| fapl.libver_latest());
    let file = fb.create(path).expect("create fixture");
    let dtype = hdf5::Datatype::from_type::<i32>().expect("transient i32 type");

    let name = CString::new("mytype").unwrap();
    let rc = unsafe {
        H5Tcommit2(
            file.id(),
            name.as_ptr(),
            dtype.id(),
            DEFAULT,
            DEFAULT,
            DEFAULT,
        )
    };
    assert!(rc >= 0, "H5Tcommit2 failed");

    let dims = [DATASET_VALUES.len() as u64];
    let maxdims = [u64::MAX]; // H5S_UNLIMITED
    let space = unsafe { H5Screate_simple(1, dims.as_ptr(), maxdims.as_ptr()) };
    assert!(space >= 0, "H5Screate_simple failed");

    let dcpl = hdf5::plist::dataset_create::DatasetCreateBuilder::new()
        .chunk([4usize])
        .finish()
        .expect("chunked dcpl");
    let typed_name = CString::new("typed").unwrap();
    let dset = unsafe {
        H5Dcreate2(
            file.id(),
            typed_name.as_ptr(),
            dtype.id(),
            space,
            DEFAULT,
            dcpl.id(),
            DEFAULT,
        )
    };
    assert!(dset >= 0, "H5Dcreate2 failed");
    let rc = unsafe {
        H5Dwrite(
            dset,
            dtype.id(),
            DEFAULT,
            DEFAULT,
            DEFAULT,
            DATASET_VALUES.as_ptr().cast::<c_void>(),
        )
    };
    assert!(rc >= 0, "H5Dwrite failed");

    unsafe {
        H5Dclose(dset);
        H5Sclose(space);
    }
}

/// The streaming backend reads a committed datatype through the same reference,
/// resolved by reading the target object header on demand rather than indexing a
/// slice. Two backends, one answer.
#[test]
fn the_streaming_backend_resolves_a_committed_datatype_too() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("committed.h5");
    write_committed_fixture(&path, EVERYTHING);

    let file = File::open_streaming(&path).unwrap();
    assert_eq!(
        file.root().attr_datatypes().unwrap().get("shared_attr"),
        Some(&committed_i32())
    );
    assert_eq!(
        file.dataset("typed").unwrap().datatype().unwrap(),
        committed_i32()
    );
    assert_eq!(
        file.dataset("typed").unwrap().read_i32().unwrap(),
        DATASET_VALUES
    );
}

/// Repack every use of a committed type separately, and require each refusal to
/// name what it refused.
///
/// Reproducing one would mean inlining the resolved type and dropping the named
/// type object, so every reader that reports the type by name would stop — an
/// approximation, and this module refuses those. Before the fix the same calls
/// returned `Ok` and produced a file libhdf5 could not read *any* attributes
/// from, which is why the assertion is on the message and not merely on `is_err`.
///
/// One fixture per use, because a file carrying two of them only proves that
/// *some* refusal fired: drop the attribute check on a file that also has a
/// committed dataset type and the dataset check catches it, with nothing to show
/// that the attribute path stopped guarding anything. The `/mytype` object is in
/// every fixture by construction — a committed type is an object — so the two
/// referencing cases drop it explicitly to get its own refusal out of the way.
#[test]
fn repack_refuses_each_use_of_a_committed_datatype() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    for (what, fixture, drop_type_object, expected) in [
        (
            "an attribute datatype",
            Fixture {
                committed_attrs: true,
                ..Fixture::default()
            },
            true,
            "attribute \"shared_attr\" has a committed",
        ),
        (
            "a dataset element type",
            Fixture {
                committed_dataset: true,
                ..Fixture::default()
            },
            true,
            "dataset typed: a committed",
        ),
        (
            // Nothing references the type, so neither check above sees it — and
            // the object is linked into the root group all the same.
            "an unreferenced named datatype object",
            Fixture::default(),
            false,
            "mytype: a committed (named) datatype object",
        ),
    ] {
        let src = dir.path().join("committed.h5");
        let dst = dir.path().join("repacked.h5");
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&dst);
        write_committed_fixture(&src, fixture);

        let mut options = RepackOptions::new();
        if drop_type_object {
            options = options.drop_path("mytype");
        }
        let err = hdf5_pure::repack(&src, &dst, &options).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains(expected),
            "repack must refuse {what} naming it; expected {expected:?}, got: {message}"
        );
        assert!(
            !dst.exists() || std::fs::metadata(&dst).unwrap().len() == 0,
            "a refused repack must not leave a readable output behind ({what})"
        );
    }
}
