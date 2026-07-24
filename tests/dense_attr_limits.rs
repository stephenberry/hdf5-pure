//! The dense (fractal-heap) attribute emitter builds a single direct block, a
//! single-leaf B-tree, and stores every attribute as a managed object. The
//! whole-file writer used to choose dense storage on attribute count alone and
//! never check any of that, so an attribute past the heap's managed-object limit
//! was written into a heap that reads back empty and aborts an assertion-enabled
//! reference C library (issue #191).
//!
//! Dense storage kicks in above eight attributes, so every case here uses nine
//! or more; below that the compact path and its own size limit apply.

use hdf5_pure::{AttrValue, Error, File, FileBuilder, FormatError};

/// Nine attributes, the first sized to `payload` bytes of text and the rest
/// small, which is enough to select dense storage.
fn nine_attrs(payload: usize) -> FileBuilder {
    let mut builder = FileBuilder::new();
    builder.set_attr("big", AttrValue::AsciiString("y".repeat(payload)));
    for i in 0..8 {
        builder.set_attr(&format!("a{i}"), AttrValue::I64(i));
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);
    builder
}

/// The largest text payload the dense path accepts, found by probing down. The
/// limit is on the *serialized* attribute, whose overhead is not visible from
/// outside the crate, so the boundary is measured rather than assumed.
fn largest_accepted_payload() -> usize {
    for payload in (1..=70_000).rev() {
        if nine_attrs(payload).finish().is_ok() {
            return payload;
        }
    }
    panic!("no payload was accepted");
}

/// The attribute name and size a dense refusal reports.
fn too_large(err: &Error) -> (String, usize, usize) {
    match err {
        Error::Format(FormatError::DenseAttributeTooLarge { name, size, limit }) => {
            (name.clone(), *size, *limit)
        }
        other => panic!("expected DenseAttributeTooLarge, got {other:?}"),
    }
}

#[test]
fn attribute_past_the_managed_object_limit_is_refused() {
    let payload = largest_accepted_payload();

    let (name, size, limit) = too_large(&nine_attrs(payload + 1).finish().unwrap_err());
    assert_eq!(name, "big");
    assert!(
        size > limit,
        "reported size {size} should exceed limit {limit}"
    );
}

/// The reported shape from the issue: nine attributes of 100,000 bytes each was
/// accepted and produced a file that read back with no attributes at all.
#[test]
fn the_reported_shape_is_refused_rather_than_silently_lost() {
    let mut builder = FileBuilder::new();
    for i in 0..9 {
        builder.set_attr(
            &format!("a{i}"),
            AttrValue::AsciiString("y".repeat(100_000)),
        );
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reported.h5");
    assert_eq!(too_large(&builder.write(&path).unwrap_err()).0, "a0");
}

/// The boundary must bite only past the limit, so the largest accepted
/// attribute still round-trips through this crate's own reader.
#[test]
fn attribute_at_the_managed_object_limit_round_trips() {
    let payload = largest_accepted_payload();
    let bytes = nine_attrs(payload).finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let attrs = file.root().attrs().unwrap();
    assert_eq!(attrs.len(), 9);
}

/// The old bound rejected any dense set whose *total* passed 64 KiB, but the
/// emitter sizes its root direct block to the content and such heaps read back
/// correctly. A multi-megabyte set of individually small attributes must be
/// written, not refused.
#[test]
fn a_multi_megabyte_set_of_small_attributes_is_written() {
    let mut builder = FileBuilder::new();
    for i in 0..40 {
        builder.set_attr(&format!("a{i}"), AttrValue::AsciiString("y".repeat(60_000)));
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);

    let bytes = builder.finish().unwrap();
    assert!(bytes.len() > 2_000_000, "expected a multi-megabyte heap");

    let file = File::from_bytes(bytes).unwrap();
    assert_eq!(file.root().attrs().unwrap().len(), 40);
}

/// Dense storage applies to group and dataset attributes too, and both are
/// checked.
#[test]
fn group_and_dataset_dense_attrs_are_checked() {
    let payload = largest_accepted_payload() + 1;

    let mut builder = FileBuilder::new();
    let mut group = builder.create_group("g");
    group.set_attr("big", AttrValue::AsciiString("y".repeat(payload)));
    for i in 0..8 {
        group.set_attr(&format!("a{i}"), AttrValue::I64(i));
    }
    group.create_dataset("y").with_f64_data(&[1.0]);
    builder.add_group(group.finish());
    assert_eq!(too_large(&builder.finish().unwrap_err()).0, "big");

    let mut builder = FileBuilder::new();
    let ds = builder.create_dataset("x").with_f64_data(&[1.0]);
    ds.set_attr("big", AttrValue::AsciiString("y".repeat(payload)));
    for i in 0..8 {
        ds.set_attr(&format!("a{i}"), AttrValue::I64(i));
    }
    assert_eq!(too_large(&builder.finish().unwrap_err()).0, "big");
}
