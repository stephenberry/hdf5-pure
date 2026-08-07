// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// gated to 64-bit-pointer targets.
#![cfg(not(target_pointer_width = "32"))]
//! An enumeration *attribute* written by the reference C library must reach the
//! caller, decoded through the enum's integer base type.
//!
//! This is the read direction of `enum_base_crosscheck.rs`, and it matters because
//! the C library is what writes the files in the wild: `H5T_NATIVE_HBOOL` — what
//! h5py gives every `np.bool_` and what `hdf5-metno` gives Rust's `bool` — is an
//! `enum[FALSE, TRUE]` over an 8-bit integer, so a boolean attribute is an enum
//! attribute. `attrs()` used to drop those silently, taking with it flags like
//! DROID's per-episode `success`.
//!
//! Every call here goes through the safe `hdf5-metno` API, which serializes its
//! own C calls through an internal lock, so these tests need no extra guard.

use std::collections::HashMap;

use hdf5_pure::{AttrValue, File};
use tempfile::tempdir;

/// Write attributes with the C library, then read them all back with this crate.
fn c_writes_then_pure_reads(write: impl FnOnce(&hdf5::File)) -> HashMap<String, AttrValue> {
    let dir = tempdir().unwrap();
    let path = dir.path().join("enum_attrs.h5");

    {
        let file = hdf5::File::create(&path).expect("the C library should create the file");
        write(&file);
        file.close().expect("close");
    }

    let bytes = std::fs::read(&path).expect("read back the written file");
    File::from_bytes(bytes)
        .expect("from_bytes")
        .root()
        .attrs()
        .expect("attrs")
}

#[test]
fn a_c_written_bool_attribute_reaches_the_caller() {
    let attrs = c_writes_then_pure_reads(|file| {
        file.new_attr::<bool>()
            .create("flag")
            .expect("create attribute")
            .write_scalar(&true)
            .expect("write attribute");
    });

    // The codes survive; `FALSE`/`TRUE` do not, since no `AttrValue` carries them.
    assert_eq!(attrs.get("flag"), Some(&AttrValue::I64(1)));
}

#[test]
fn a_c_written_bool_array_attribute_reaches_the_caller() {
    let attrs = c_writes_then_pure_reads(|file| {
        file.new_attr::<bool>()
            .shape([3])
            .create("flags")
            .expect("create attribute")
            .write(&[true, false, true])
            .expect("write attribute");
    });

    assert_eq!(
        attrs.get("flags"),
        Some(&AttrValue::I64Array(vec![1, 0, 1]))
    );
}

/// A boolean attribute must not cost the object its other attributes: the decoder
/// skips what it cannot represent, so a regression here would be silent.
#[test]
fn a_bool_attribute_sits_alongside_the_rest() {
    let attrs = c_writes_then_pure_reads(|file| {
        file.new_attr::<bool>()
            .create("success")
            .expect("create attribute")
            .write_scalar(&false)
            .expect("write attribute");
        file.new_attr::<f64>()
            .create("frequency")
            .expect("create attribute")
            .write_scalar(&30.0)
            .expect("write attribute");
    });

    assert_eq!(attrs.get("success"), Some(&AttrValue::I64(0)));
    assert_eq!(attrs.get("frequency"), Some(&AttrValue::F64(30.0)));
}
