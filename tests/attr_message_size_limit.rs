//! An attribute stored compactly lives in its object header, whose per-message
//! size field is 2 bytes wide. The whole-file writer used to cast the message
//! length into that field unchecked, so an oversized attribute either vanished
//! from the file it was written to or left the header unparseable (issue #190).
//! These tests pin the refusal on every object that carries attributes.

use hdf5_pure::{AttrValue, Error, File, FileBuilder, FormatError, OBJECT_HEADER_MESSAGE_MAX};

/// A `VarLenAsciiArray` past the message-size limit: each element contributes a
/// 16-byte global-heap reference, so ~4,100 elements cross it. This is the shape
/// from the issue report, where the attribute silently disappeared.
fn oversized_vlen() -> AttrValue {
    AttrValue::VarLenAsciiArray((0..4_200).map(|i| format!("s{i}")).collect())
}

/// A fixed-width string array past the message-size limit. No variable-length
/// data is involved, so this is not specific to the heap-backed shape above; in
/// the issue report the truncated length left readers parsing a message body as
/// a message header.
fn oversized_fixed() -> AttrValue {
    AttrValue::StringArray(vec!["x".repeat(200); 400])
}

/// The name and size an oversized-attribute refusal reports, or a panic
/// describing what came back instead.
fn too_large(err: &Error) -> (String, usize) {
    match err {
        Error::Format(FormatError::AttributeMessageTooLarge { name, size }) => {
            (name.clone(), *size)
        }
        other => panic!("expected AttributeMessageTooLarge, got {other:?}"),
    }
}

#[test]
fn root_vlen_attr_past_the_limit_is_refused() {
    let mut builder = FileBuilder::new();
    builder.set_attr("labels", oversized_vlen());
    builder.create_dataset("x").with_f64_data(&[1.0]);

    let (name, size) = too_large(&builder.finish().unwrap_err());
    assert_eq!(name, "labels");
    assert!(size > OBJECT_HEADER_MESSAGE_MAX);
}

#[test]
fn root_fixed_width_attr_past_the_limit_is_refused() {
    let mut builder = FileBuilder::new();
    builder.set_attr("big", oversized_fixed());
    builder.create_dataset("x").with_f64_data(&[1.0]);

    let (name, size) = too_large(&builder.finish().unwrap_err());
    assert_eq!(name, "big");
    assert!(size > OBJECT_HEADER_MESSAGE_MAX);
}

#[test]
fn group_attr_past_the_limit_is_refused() {
    let mut builder = FileBuilder::new();
    let mut group = builder.create_group("g");
    group.set_attr("big", oversized_fixed());
    group.create_dataset("x").with_f64_data(&[1.0]);
    builder.add_group(group.finish());

    assert_eq!(too_large(&builder.finish().unwrap_err()).0, "big");
}

#[test]
fn dataset_attr_past_the_limit_is_refused() {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("x")
        .with_f64_data(&[1.0])
        .set_attr("big", oversized_fixed());

    assert_eq!(too_large(&builder.finish().unwrap_err()).0, "big");
}

/// `write` refuses too, rather than only `finish`. It creates the destination
/// before serializing, so a refusal leaves an empty file behind — no HDF5
/// content, but the path exists. Asserted so the behaviour is on the record.
#[test]
fn write_to_disk_refuses_and_leaves_an_empty_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("oversized.h5");

    let mut builder = FileBuilder::new();
    builder.set_attr("big", oversized_fixed());
    builder.create_dataset("x").with_f64_data(&[1.0]);

    assert_eq!(too_large(&builder.write(&path).unwrap_err()).0, "big");
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 0);
}

/// An oversized attribute is not the only way to overflow the message-size
/// field: a long enough dataset name does it through the Link message. That has
/// no name-carrying check in front of it, so it lands on the writer's backstop —
/// which must still refuse rather than truncate.
#[test]
fn oversized_link_message_hits_the_writer_backstop() {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset(&"a".repeat(70_000))
        .with_f64_data(&[1.0]);

    match builder.finish().unwrap_err() {
        Error::Format(FormatError::ObjectHeaderMessageTooLarge { size, .. }) => {
            assert!(size > OBJECT_HEADER_MESSAGE_MAX);
        }
        other => panic!("expected ObjectHeaderMessageTooLarge, got {other:?}"),
    }
}

/// The refusal must bite only past the limit: a large attribute that still fits
/// the size field round-trips, on every object that carries attributes.
#[test]
fn large_but_fitting_attrs_still_round_trip() {
    // 4,000 variable-length elements sit just under the limit; the issue reports
    // this exact count as the last one that worked.
    let labels: Vec<String> = (0..4_000).map(|i| format!("s{i}")).collect();

    let mut builder = FileBuilder::new();
    builder.set_attr("labels", AttrValue::VarLenAsciiArray(labels.clone()));
    builder
        .create_dataset("x")
        .with_f64_data(&[1.0])
        .set_attr("labels", AttrValue::VarLenAsciiArray(labels.clone()));
    let mut group = builder.create_group("g");
    group.set_attr("labels", AttrValue::VarLenAsciiArray(labels));
    group.create_dataset("y").with_f64_data(&[2.0]);
    builder.add_group(group.finish());

    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    assert_eq!(file.root().attrs().unwrap().len(), 1);
    assert_eq!(file.dataset("x").unwrap().attrs().unwrap().len(), 1);
    assert_eq!(file.group("g").unwrap().attrs().unwrap().len(), 1);
}
