//! Dense (fractal-heap) attribute storage and the bounds that remain on it.
//!
//! An attribute too large to be a managed heap object is written as a fractal-heap
//! *huge* object: its bytes sit outside the managed blocks and a huge-objects v2
//! B-tree maps a generated ID to their address and length (issue #195). Before
//! that it was refused outright, which was itself a fix for writing a heap that
//! read back empty and aborted an assertion-enabled reference C library (#191).
//!
//! Dense storage is selected by either of two things: more than eight
//! attributes, or one attribute too large for an object-header message. Most
//! cases here use nine or more to reach it by count; the ones that deliberately
//! use fewer reach it by size, and say so.

use hdf5_pure::{AttrValue, Error, File, FileBuilder, FormatError};

mod common;
use common::heap::{has_fractal_heap, huge_object_bytes, huge_object_count, managed_object_count};

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

/// The text of a string attribute, whichever string variant the reader chose for
/// it. Which one it picks is a datatype question and not what these tests are
/// about; the content is.
#[track_caller]
fn text<'a>(attrs: &'a std::collections::HashMap<String, AttrValue>, name: &str) -> &'a str {
    match attrs.get(name) {
        Some(AttrValue::AsciiString(s) | AttrValue::String(s)) => s,
        other => panic!("expected string attribute {name:?}, got {other:?}"),
    }
}

/// The largest text payload that still fits a managed heap object, found by
/// probing down. The threshold is on the *serialized* attribute, whose overhead
/// is not visible from outside the crate, so it is measured rather than assumed.
fn largest_managed_payload() -> usize {
    for payload in (1..=70_000).rev() {
        let bytes = nine_attrs(payload)
            .finish()
            .expect("every size is writable");
        if huge_object_count(&bytes) == 0 {
            return payload;
        }
    }
    panic!("no payload was stored as a managed object");
}

#[test]
fn an_attribute_past_the_managed_object_limit_becomes_a_huge_object() {
    let payload = largest_managed_payload();

    let at_limit = nine_attrs(payload).finish().unwrap();
    assert_eq!(huge_object_count(&at_limit), 0);
    assert_eq!(
        File::from_bytes(at_limit)
            .unwrap()
            .root()
            .attrs()
            .unwrap()
            .len(),
        9
    );

    // One byte more changes the storage class, and nothing else.
    let past = nine_attrs(payload + 1).finish().unwrap();
    assert_eq!(huge_object_count(&past), 1);

    let file = File::from_bytes(past).unwrap();
    let attrs = file.root().attrs().unwrap();
    assert_eq!(attrs.len(), 9);
    let big = text(&attrs, "big");
    assert_eq!(big.len(), payload + 1);
    assert!(big.bytes().all(|b| b == b'y'), "huge object bytes differ");
}

/// Dense storage is not chosen by attribute count alone. An attribute too large
/// for an object-header message forces it on its own, which is the only way a
/// *single* large attribute can be written at all — the compact path's answer to
/// one is `AttributeMessageTooLarge`, no matter that dense storage could hold it.
///
/// This mirrors the reference C library's own disjunction, and it is what makes
/// the ordinary case work: one big attribute on an object with no others.
#[test]
fn one_large_attribute_forces_dense_storage_on_its_own() {
    for others in [0usize, 2, 7] {
        let mut builder = FileBuilder::new();
        builder.set_attr("big", AttrValue::AsciiString("y".repeat(100_000)));
        for i in 0..others {
            builder.set_attr(&format!("a{i}"), AttrValue::I64(i as i64));
        }
        builder.create_dataset("x").with_f64_data(&[1.0]);

        let bytes = builder
            .finish()
            .unwrap_or_else(|e| panic!("{} attributes refused: {e}", others + 1));
        assert_eq!(
            huge_object_count(&bytes),
            1,
            "with {} attributes the large one should be a huge object",
            others + 1
        );

        let file = File::from_bytes(bytes).unwrap();
        let attrs = file.root().attrs().unwrap();
        assert_eq!(attrs.len(), others + 1);
        assert_eq!(text(&attrs, "big").len(), 100_000);
    }
}

/// The other half of that disjunction still holds: a small set of small
/// attributes stays in the object header, so the size rule has not quietly
/// promoted every file to dense storage.
#[test]
fn small_attributes_below_the_count_threshold_stay_compact() {
    let mut builder = FileBuilder::new();
    for i in 0..8 {
        builder.set_attr(&format!("a{i}"), AttrValue::AsciiString("y".repeat(1_000)));
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);

    let bytes = builder.finish().unwrap();
    assert!(
        !has_fractal_heap(&bytes),
        "eight small attributes must stay compact"
    );

    let file = File::from_bytes(bytes).unwrap();
    assert_eq!(file.root().attrs().unwrap().len(), 8);
}

/// The shape reported in #191: nine attributes of 100,000 bytes each. It was
/// silently lost, then refused, and is now written — every one of them as a huge
/// object, since each is far past the managed limit.
#[test]
fn the_reported_shape_round_trips_through_huge_storage() {
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
    builder.write(&path).unwrap();

    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(huge_object_count(&bytes), 9);

    let file = File::open(&path).unwrap();
    let attrs = file.root().attrs().unwrap();
    assert_eq!(attrs.len(), 9);
    for i in 0..9 {
        assert_eq!(text(&attrs, &format!("a{i}")).len(), 100_000);
    }
}

/// Managed and huge objects share one heap, and an attribute set that mixes them
/// has to keep both indexes straight: the direct block is sized by the managed
/// objects alone, while the name index spans all of them.
#[test]
fn a_mixed_managed_and_huge_set_round_trips() {
    let mut builder = FileBuilder::new();
    for i in 0..6 {
        builder.set_attr(
            &format!("small{i}"),
            AttrValue::AsciiString("y".repeat(100)),
        );
    }
    for i in 0..3 {
        builder.set_attr(
            &format!("big{i}"),
            AttrValue::AsciiString("z".repeat(80_000)),
        );
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);

    let bytes = builder.finish().unwrap();
    // The heap counts the two classes separately and every attribute is in
    // exactly one of them, so a miscount here is a heap that describes itself
    // wrongly even if the data survives.
    assert_eq!(huge_object_count(&bytes), 3);
    assert_eq!(managed_object_count(&bytes), 6);
    let declared = huge_object_bytes(&bytes);
    assert!(
        (3 * 80_000..3 * 80_000 + 1_000).contains(&declared),
        "declared huge size {declared} does not account for three 80,000-byte payloads"
    );

    let file = File::from_bytes(bytes).unwrap();
    let attrs = file.root().attrs().unwrap();
    assert_eq!(attrs.len(), 9);
    for i in 0..6 {
        let s = text(&attrs, &format!("small{i}"));
        assert_eq!(s.len(), 100);
        assert!(s.bytes().all(|b| b == b'y'), "managed object bytes differ");
    }
    for i in 0..3 {
        let s = text(&attrs, &format!("big{i}"));
        assert_eq!(s.len(), 80_000);
        assert!(s.bytes().all(|b| b == b'z'), "huge object bytes differ");
    }
}

/// The emitter sizes its root direct block to the content, so a multi-megabyte
/// set of individually small attributes stays entirely managed.
#[test]
fn a_multi_megabyte_set_of_small_attributes_stays_managed() {
    let mut builder = FileBuilder::new();
    for i in 0..40 {
        builder.set_attr(&format!("a{i}"), AttrValue::AsciiString("y".repeat(60_000)));
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);

    let bytes = builder.finish().unwrap();
    assert!(bytes.len() > 2_000_000, "expected a multi-megabyte heap");
    assert_eq!(huge_object_count(&bytes), 0);

    let file = File::from_bytes(bytes).unwrap();
    assert_eq!(file.root().attrs().unwrap().len(), 40);
}

/// The attribute *count* still has a bound, and it is not the 65,535 the B-tree
/// leaf's 2-byte record-count field suggests: the reference C library derives the
/// width it needs from the leaf's capacity, which follows the power-of-two node
/// size this writer declares, so the real limit is 61,680. One more silently
/// produced a file that aborted libhdf5.
#[test]
fn the_attribute_count_boundary_is_enforced() {
    let build = |n: usize| {
        let mut builder = FileBuilder::new();
        for i in 0..n {
            builder.set_attr(&format!("a{i:06}"), AttrValue::I64(i as i64));
        }
        builder.create_dataset("x").with_f64_data(&[1.0]);
        builder
    };

    let bytes = build(61_680)
        .finish()
        .expect("the limit itself is writable");
    let file = File::from_bytes(bytes).unwrap();
    assert_eq!(file.root().attrs().unwrap().len(), 61_680);

    match build(61_681).finish().unwrap_err() {
        Error::Format(FormatError::TooManyDenseAttributes { count, limit }) => {
            assert_eq!(count, 61_681);
            assert_eq!(limit, 61_680);
        }
        other => panic!("expected TooManyDenseAttributes, got {other:?}"),
    }
}

/// Huge storage lifts the limit on an attribute's *data*, not on the fields that
/// describe it: the message encodes its name, datatype and dataspace lengths in
/// 2-byte fields, so a name past that must still be refused rather than truncated
/// into a message that decodes as something else.
#[test]
fn an_attribute_name_too_long_for_its_length_field_is_refused() {
    let mut builder = FileBuilder::new();
    builder.set_attr(&"n".repeat(70_000), AttrValue::I64(1));
    for i in 0..8 {
        builder.set_attr(&format!("a{i}"), AttrValue::I64(i));
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);

    match builder.finish().unwrap_err() {
        Error::Format(FormatError::AttributeFieldTooLong {
            field, size, limit, ..
        }) => {
            assert_eq!(field, "name");
            assert_eq!(size, 70_001, "the null terminator counts toward the field");
            assert_eq!(limit, u16::MAX as usize);
        }
        other => panic!("expected AttributeFieldTooLong, got {other:?}"),
    }

    // A name one byte inside the field is written, and comes back intact.
    let name = "n".repeat(u16::MAX as usize - 1);
    let mut builder = FileBuilder::new();
    builder.set_attr(&name, AttrValue::I64(1));
    for i in 0..8 {
        builder.set_attr(&format!("a{i}"), AttrValue::I64(i));
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);

    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    let attrs = file.root().attrs().unwrap();
    assert_eq!(attrs.len(), 9);
    assert!(attrs.contains_key(&name));
}

/// Dense storage applies to group and dataset attributes too, and both reach the
/// same emitter.
#[test]
fn group_and_dataset_attributes_use_huge_storage_too() {
    let payload = largest_managed_payload() + 1;

    let mut builder = FileBuilder::new();
    let mut group = builder.create_group("g");
    group.set_attr("big", AttrValue::AsciiString("y".repeat(payload)));
    for i in 0..8 {
        group.set_attr(&format!("a{i}"), AttrValue::I64(i));
    }
    group.create_dataset("y").with_f64_data(&[1.0]);
    builder.add_group(group.finish());

    let bytes = builder.finish().unwrap();
    assert_eq!(huge_object_count(&bytes), 1);
    let file = File::from_bytes(bytes).unwrap();
    assert_eq!(file.group("g").unwrap().attrs().unwrap().len(), 9);

    let mut builder = FileBuilder::new();
    let ds = builder.create_dataset("x").with_f64_data(&[1.0]);
    ds.set_attr("big", AttrValue::AsciiString("y".repeat(payload)));
    for i in 0..8 {
        ds.set_attr(&format!("a{i}"), AttrValue::I64(i));
    }

    let bytes = builder.finish().unwrap();
    assert_eq!(huge_object_count(&bytes), 1);
    let file = File::from_bytes(bytes).unwrap();
    assert_eq!(file.dataset("x").unwrap().attrs().unwrap().len(), 9);
}
