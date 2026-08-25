//! Pure-Rust tests for the fixed-width string dataset writers
//! (`DatasetBuilder::with_ascii_strings` and its three siblings, issue #355).
//!
//! The unit tests beside the builder assert what it stages; these assert what a
//! reader gets back out of a file, which is the other half of the same claim —
//! a datatype message and a run of element bytes that agree on a width are only
//! correct if `read_string` recovers the values from them.
//!
//! C-library interop lives in `fixed_string_crosscheck.rs`.

use hdf5_pure::{
    CharacterSet, Datatype, Error, File, FileBuilder, FormatError, StringPadding,
    VlenStringReadOptions,
};
use tempfile::tempdir;

/// Every fixed-width string entry point, through a file and back.
#[test]
fn fixed_width_strings_round_trip_through_a_file() {
    let values = ["north", "s", "", "east"];
    let mut b = FileBuilder::new();
    b.create_dataset("derived_ascii")
        .with_ascii_strings(&values)
        .unwrap();
    b.create_dataset("sized_ascii")
        .with_ascii_strings_sized(&values, 16)
        .unwrap();
    b.create_dataset("derived_utf8")
        .with_strings(&values)
        .unwrap();
    b.create_dataset("sized_utf8")
        .with_strings_sized(&values, 16)
        .unwrap();
    let file = File::from_bytes(b.finish().unwrap()).unwrap();

    for (name, width, charset) in [
        ("derived_ascii", 5, CharacterSet::Ascii),
        ("sized_ascii", 16, CharacterSet::Ascii),
        ("derived_utf8", 5, CharacterSet::Utf8),
        ("sized_utf8", 16, CharacterSet::Utf8),
    ] {
        let ds = file.dataset(name).unwrap();
        assert_eq!(
            ds.datatype().unwrap(),
            Datatype::String {
                size: width,
                padding: StringPadding::NullPad,
                charset,
            },
            "{name}"
        );
        assert_eq!(ds.shape().unwrap(), vec![4], "{name}");
        assert_eq!(ds.read_string().unwrap(), values, "{name}");
        // The windowed reader decodes straight from the raw window, a separate
        // path from the whole-dataset one.
        assert_eq!(ds.read_string_rows(1, 2).unwrap(), ["s", ""], "{name}");
    }
}

/// The width counts bytes, and a value whose characters are wider than a byte
/// still reads back whole. Written through the UTF-8 entry point because the
/// charset bit is what tells a reader the bytes are not one character each.
#[test]
fn a_utf8_value_is_measured_in_bytes_and_read_back_whole() {
    let values = ["mètre", "K", "°C"];
    let mut b = FileBuilder::new();
    b.create_dataset("units").with_strings(&values).unwrap();
    let file = File::from_bytes(b.finish().unwrap()).unwrap();

    let ds = file.dataset("units").unwrap();
    // "mètre" is 6 bytes over 5 characters, and it sets the width.
    assert_eq!(
        ds.datatype().unwrap(),
        Datatype::String {
            size: 6,
            padding: StringPadding::NullPad,
            charset: CharacterSet::Utf8,
        }
    );
    assert_eq!(ds.read_string().unwrap(), values);
}

/// The reason [`DatasetBuilder::with_ascii_strings_sized`] exists: a dataset can
/// be extended, and a width derived from the values in hand leaves a later,
/// longer value with nowhere to go. Declaring 16 up front makes the append fit.
///
/// The append itself pads by hand through `append_raw`, which is what a caller
/// does today — the typed append counterpart is not written yet.
#[test]
fn a_declared_width_leaves_room_for_a_later_longer_value() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("stations.h5");

    let mut b = FileBuilder::new();
    b.create_dataset("station")
        .with_ascii_strings_sized(&["north", "s"], 16)
        .unwrap()
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[4]);
    b.write(&path).unwrap();

    // Longer than the derived width of 5 would have been, and it fits.
    let appended = ["north-northeast", "e"];
    {
        let session = File::open_rw(&path).unwrap();
        let mut raw = Vec::new();
        for value in appended {
            let mut element = value.as_bytes().to_vec();
            element.resize(16, 0);
            raw.extend_from_slice(&element);
        }
        session
            .dataset("station")
            .unwrap()
            .append_staged(|a| {
                a.append_raw(&raw);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let ds = file.dataset("station").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![4]);
    assert_eq!(
        ds.read_string().unwrap(),
        ["north", "s", "north-northeast", "e"]
    );
}

/// A refusal leaves the file writable: the builder staged nothing, so the same
/// dataset name is still free for a call that fits.
#[test]
fn a_refused_width_leaves_the_builder_usable() {
    let mut b = FileBuilder::new();
    let ds = b.create_dataset("station");
    assert!(matches!(
        ds.with_ascii_strings_sized(&["north"], 2),
        Err(FormatError::FixedStringTooLong { index: 0, .. })
    ));
    assert!(matches!(
        ds.with_ascii_strings_sized(&["north"], 0),
        Err(FormatError::ZeroFixedStringWidth)
    ));
    ds.with_ascii_strings_sized(&["north"], 8).unwrap();

    let file = File::from_bytes(b.finish().unwrap()).unwrap();
    assert_eq!(
        file.dataset("station").unwrap().read_string().unwrap(),
        ["north"]
    );
}

/// A fixed-width dataset is not a variable-length one, and the reader that
/// bounds variable-length allocation says so rather than decoding the padding
/// as heap references.
#[test]
fn a_fixed_width_dataset_is_refused_by_the_vlen_reader() {
    let mut b = FileBuilder::new();
    b.create_dataset("station")
        .with_ascii_strings(&["north"])
        .unwrap();
    let file = File::from_bytes(b.finish().unwrap()).unwrap();

    let ds = file.dataset("station").unwrap();
    assert!(
        matches!(
            ds.read_vlen_strings(VlenStringReadOptions::default()),
            Err(Error::Format(FormatError::TypeMismatch {
                expected: "VariableLength string",
                ..
            }))
        ),
        "expected a type mismatch naming the type it wanted"
    );
    // The charset-agnostic reader still works on it, which is the point of
    // `read_string` dispatching on the datatype.
    assert_eq!(ds.read_string().unwrap(), ["north"]);
}

/// Every value empty is the case a zero-width datatype would be the natural
/// answer to, and the one the C library cannot read. One padding byte per
/// element is what goes out instead, and it reads back as the empty string.
#[test]
fn all_empty_values_take_a_one_byte_datatype() {
    let mut b = FileBuilder::new();
    b.create_dataset("blank")
        .with_ascii_strings(&["", "", ""])
        .unwrap();
    let file = File::from_bytes(b.finish().unwrap()).unwrap();

    let ds = file.dataset("blank").unwrap();
    assert_eq!(
        ds.datatype().unwrap(),
        Datatype::String {
            size: 1,
            padding: StringPadding::NullPad,
            charset: CharacterSet::Ascii,
        }
    );
    assert_eq!(ds.read_string().unwrap(), ["", "", ""]);
}
