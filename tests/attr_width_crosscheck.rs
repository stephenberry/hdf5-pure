// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit little-endian targets; skip them elsewhere so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
//! A numeric attribute keeps the width it is stored at, in both directions
//! across the reference C library (issues #350 and #354).
//!
//! `AttrValue` used to carry integers only at 64 bits and floats only at 64,
//! so an attribute written from a value was eight bytes wide whatever the
//! caller asked for, and one read back arrived as `I64`/`U64`/`F64` whatever
//! the file held. The C library preserves `H5T_NATIVE_INT16`, `H5T_NATIVE_FLOAT`
//! and their siblings, and it wrote most of the files this crate is pointed at,
//! so it is the authority on both halves: what a width means on disk, and what
//! the files in the wild carry.
//!
//! The values are the extremes of each width — `i8::MIN`, `u16::MAX`, `f32::MAX`
//! and the rest — because a value that only fits at its own width is what
//! catches a narrowing that wrapped or rounded, or a widening that lost a sign,
//! which `-7` in every slot would not.

use hdf5::types::{FloatSize, IntSize, TypeDescriptor};
use hdf5_pure::{AttrValue, File, FileBuilder};
use tempfile::tempdir;

/// One attribute per width: the name, the value this crate writes, and the type
/// the C library must report for it.
fn scalar_widths() -> Vec<(&'static str, AttrValue, TypeDescriptor)> {
    vec![
        (
            "f32",
            AttrValue::F32(f32::MAX),
            TypeDescriptor::Float(FloatSize::U4),
        ),
        (
            "f64",
            AttrValue::F64(f64::MAX),
            TypeDescriptor::Float(FloatSize::U8),
        ),
        (
            "i8",
            AttrValue::I8(i8::MIN),
            TypeDescriptor::Integer(IntSize::U1),
        ),
        (
            "i16",
            AttrValue::I16(i16::MIN),
            TypeDescriptor::Integer(IntSize::U2),
        ),
        (
            "i32",
            AttrValue::I32(i32::MIN),
            TypeDescriptor::Integer(IntSize::U4),
        ),
        (
            "i64",
            AttrValue::I64(i64::MIN),
            TypeDescriptor::Integer(IntSize::U8),
        ),
        (
            "u8",
            AttrValue::U8(u8::MAX),
            TypeDescriptor::Unsigned(IntSize::U1),
        ),
        (
            "u16",
            AttrValue::U16(u16::MAX),
            TypeDescriptor::Unsigned(IntSize::U2),
        ),
        (
            "u32",
            AttrValue::U32(u32::MAX),
            TypeDescriptor::Unsigned(IntSize::U4),
        ),
        (
            "u64",
            AttrValue::U64(u64::MAX),
            TypeDescriptor::Unsigned(IntSize::U8),
        ),
    ]
}

/// The same widths as two-element arrays, which reach the writer by a different
/// arm than the scalars and are what a caller with a vector of measurements
/// writes.
fn array_widths() -> Vec<(&'static str, AttrValue, TypeDescriptor)> {
    vec![
        (
            "f32s",
            AttrValue::F32Array(vec![f32::MIN, f32::MAX]),
            TypeDescriptor::Float(FloatSize::U4),
        ),
        (
            "f64s",
            AttrValue::F64Array(vec![f64::MIN, f64::MAX]),
            TypeDescriptor::Float(FloatSize::U8),
        ),
        (
            "i8s",
            AttrValue::I8Array(vec![i8::MIN, i8::MAX]),
            TypeDescriptor::Integer(IntSize::U1),
        ),
        (
            "i16s",
            AttrValue::I16Array(vec![i16::MIN, i16::MAX]),
            TypeDescriptor::Integer(IntSize::U2),
        ),
        (
            "i32s",
            AttrValue::I32Array(vec![i32::MIN, i32::MAX]),
            TypeDescriptor::Integer(IntSize::U4),
        ),
        (
            "u8s",
            AttrValue::U8Array(vec![0, u8::MAX]),
            TypeDescriptor::Unsigned(IntSize::U1),
        ),
        (
            "u16s",
            AttrValue::U16Array(vec![0, u16::MAX]),
            TypeDescriptor::Unsigned(IntSize::U2),
        ),
        (
            "u32s",
            AttrValue::U32Array(vec![0, u32::MAX]),
            TypeDescriptor::Unsigned(IntSize::U4),
        ),
        (
            "i64s",
            AttrValue::I64Array(vec![i64::MIN, i64::MAX]),
            TypeDescriptor::Integer(IntSize::U8),
        ),
        (
            "u64s",
            AttrValue::U64Array(vec![0, u64::MAX]),
            TypeDescriptor::Unsigned(IntSize::U8),
        ),
    ]
}

/// The C library reports each attribute this crate wrote at the width the value
/// named, and reads its value back.
///
/// The datatype is the claim under test: a value written eight bytes wide reads
/// back through this crate as whatever it stored, so only a second library can
/// say what the file really holds.
#[test]
fn c_reads_every_width_this_crate_writes() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("widths.h5");

    let cases: Vec<_> = scalar_widths().into_iter().chain(array_widths()).collect();
    let mut builder = FileBuilder::new();
    for (name, value, _) in &cases {
        builder.set_attr(name, value.clone());
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    for (name, value, expected) in &cases {
        let attr = file.attr(name).unwrap();
        assert_eq!(
            attr.dtype().unwrap().to_descriptor().unwrap(),
            *expected,
            "attribute {name} reached the file at the wrong width"
        );
        // The values too, so a file that declares the right width and stores the
        // wrong bytes in it does not pass. Each lane reads through the widest
        // type of its own kind, which holds every value on that side exactly.
        match expected {
            TypeDescriptor::Unsigned(_) => assert_eq!(
                attr.read_raw::<u64>().unwrap(),
                value.to_u64s().unwrap(),
                "attribute {name} holds wrong values"
            ),
            TypeDescriptor::Integer(_) => assert_eq!(
                attr.read_raw::<i64>().unwrap(),
                value.to_i64s().unwrap(),
                "attribute {name} holds wrong values"
            ),
            TypeDescriptor::Float(_) => assert_eq!(
                attr.read_raw::<f64>().unwrap(),
                value.to_f64s().unwrap(),
                "attribute {name} holds wrong values"
            ),
            other => panic!("attribute {name} expects an unhandled descriptor {other:?}"),
        }
    }
    file.close().unwrap();
}

/// The read direction: every width the C library writes arrives as the variant
/// of that width, rather than every one of them arriving as 64-bit.
///
/// This is the half a caller sees on somebody else's file — an instrument
/// writing `int16` counts, an h5py `np.uint8` flag, a `np.float32` calibration —
/// and the half issues #350 and #354 reported.
#[test]
fn every_width_the_c_library_writes_reads_back_as_itself() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_widths.h5");

    {
        let file = hdf5::File::create(&path).unwrap();
        macro_rules! c_attr {
            ($name:literal, $ty:ty, $scalar:expr, $array:expr) => {{
                file.new_attr::<$ty>()
                    .shape(())
                    .create(concat!($name, "_scalar"))
                    .unwrap()
                    .write_scalar(&$scalar)
                    .unwrap();
                file.new_attr::<$ty>()
                    .shape([2])
                    .create(concat!($name, "_array"))
                    .unwrap()
                    .write(&$array)
                    .unwrap();
            }};
        }
        c_attr!("i8", i8, i8::MIN, [i8::MIN, i8::MAX]);
        c_attr!("i16", i16, i16::MIN, [i16::MIN, i16::MAX]);
        c_attr!("i32", i32, i32::MIN, [i32::MIN, i32::MAX]);
        c_attr!("i64", i64, i64::MIN, [i64::MIN, i64::MAX]);
        c_attr!("u8", u8, u8::MAX, [0u8, u8::MAX]);
        c_attr!("u16", u16, u16::MAX, [0u16, u16::MAX]);
        c_attr!("u32", u32, u32::MAX, [0u32, u32::MAX]);
        c_attr!("u64", u64, u64::MAX, [0u64, u64::MAX]);
        c_attr!("f32", f32, f32::MAX, [f32::MIN, f32::MAX]);
        c_attr!("f64", f64, f64::MAX, [f64::MIN, f64::MAX]);
        file.close().unwrap();
    }

    let bytes = std::fs::read(&path).unwrap();
    let attrs = File::from_bytes(bytes).unwrap().root().attrs().unwrap();

    let expected = [
        ("i8_scalar", AttrValue::I8(i8::MIN)),
        ("i8_array", AttrValue::I8Array(vec![i8::MIN, i8::MAX])),
        ("i16_scalar", AttrValue::I16(i16::MIN)),
        ("i16_array", AttrValue::I16Array(vec![i16::MIN, i16::MAX])),
        ("i32_scalar", AttrValue::I32(i32::MIN)),
        ("i32_array", AttrValue::I32Array(vec![i32::MIN, i32::MAX])),
        ("i64_scalar", AttrValue::I64(i64::MIN)),
        ("i64_array", AttrValue::I64Array(vec![i64::MIN, i64::MAX])),
        ("u8_scalar", AttrValue::U8(u8::MAX)),
        ("u8_array", AttrValue::U8Array(vec![0, u8::MAX])),
        ("u16_scalar", AttrValue::U16(u16::MAX)),
        ("u16_array", AttrValue::U16Array(vec![0, u16::MAX])),
        ("u32_scalar", AttrValue::U32(u32::MAX)),
        ("u32_array", AttrValue::U32Array(vec![0, u32::MAX])),
        ("u64_scalar", AttrValue::U64(u64::MAX)),
        ("u64_array", AttrValue::U64Array(vec![0, u64::MAX])),
        ("f32_scalar", AttrValue::F32(f32::MAX)),
        ("f32_array", AttrValue::F32Array(vec![f32::MIN, f32::MAX])),
        ("f64_scalar", AttrValue::F64(f64::MAX)),
        ("f64_array", AttrValue::F64Array(vec![f64::MIN, f64::MAX])),
    ];
    for (name, value) in &expected {
        assert_eq!(attrs.get(*name), Some(value), "attribute {name}");
    }
    assert_eq!(
        attrs.len(),
        expected.len(),
        "every attribute the C library wrote must decode, got {:?}",
        attrs.keys().collect::<Vec<_>>()
    );
}
