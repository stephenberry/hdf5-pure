//! Simplified type representations for the high-level API.

use std::collections::HashMap;
use std::fmt;

pub use crate::file_writer::AttrValue;

/// Simplified datatype enum for the high-level API.
///
/// Maps from the detailed `crate::datatype::Datatype` to a
/// user-friendly representation.
///
/// Non-exhaustive: variants are added as this crate supports more datatypes, so
/// match with a `_` arm.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum DType {
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    String,
    Compound(Vec<(std::string::String, DType)>),
    Enum(Vec<std::string::String>),
    Array(Box<DType>, Vec<u32>),
    VariableLengthString,
    /// HDF5 object reference (8-byte address).
    ObjectReference,
    Other(std::string::String),
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::F64 => write!(f, "f64"),
            DType::I8 => write!(f, "i8"),
            DType::I16 => write!(f, "i16"),
            DType::I32 => write!(f, "i32"),
            DType::I64 => write!(f, "i64"),
            DType::U8 => write!(f, "u8"),
            DType::U16 => write!(f, "u16"),
            DType::U32 => write!(f, "u32"),
            DType::U64 => write!(f, "u64"),
            DType::String => write!(f, "string"),
            DType::VariableLengthString => write!(f, "vlen_string"),
            DType::ObjectReference => write!(f, "object_ref"),
            DType::Compound(fields) => {
                write!(f, "compound{{")?;
                for (i, (name, dt)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{name}: {dt}")?;
                }
                write!(f, "}}")
            }
            DType::Enum(names) => write!(f, "enum[{}]", names.join(", ")),
            DType::Array(base, dims) => write!(f, "array<{base}, {dims:?}>"),
            DType::Other(desc) => write!(f, "other({desc})"),
        }
    }
}

/// Convert a low-level `Datatype` to a simplified `DType`.
pub(crate) fn classify_datatype(dt: &crate::datatype::Datatype) -> DType {
    use crate::datatype::Datatype;

    match dt {
        Datatype::FloatingPoint { size: 4, .. } => DType::F32,
        Datatype::FloatingPoint { size: 8, .. } => DType::F64,
        // Widen before `* 8`: `size` is an on-disk `u32`, so a crafted odd size
        // near `u32::MAX` would overflow a `u32` bit-width computation (issue #140).
        Datatype::FloatingPoint { size, .. } => {
            DType::Other(format!("float{}", u64::from(*size) * 8))
        }
        Datatype::FixedPoint {
            size: 1,
            signed: true,
            ..
        } => DType::I8,
        Datatype::FixedPoint {
            size: 2,
            signed: true,
            ..
        } => DType::I16,
        Datatype::FixedPoint {
            size: 4,
            signed: true,
            ..
        } => DType::I32,
        Datatype::FixedPoint {
            size: 8,
            signed: true,
            ..
        } => DType::I64,
        Datatype::FixedPoint {
            size: 1,
            signed: false,
            ..
        } => DType::U8,
        Datatype::FixedPoint {
            size: 2,
            signed: false,
            ..
        } => DType::U16,
        Datatype::FixedPoint {
            size: 4,
            signed: false,
            ..
        } => DType::U32,
        Datatype::FixedPoint {
            size: 8,
            signed: false,
            ..
        } => DType::U64,
        Datatype::FixedPoint { size, signed, .. } => {
            let prefix = if *signed { "i" } else { "u" };
            DType::Other(format!("{prefix}{}", u64::from(*size) * 8))
        }
        Datatype::String { .. } => DType::String,
        Datatype::VariableLength {
            is_string: true, ..
        } => DType::VariableLengthString,
        Datatype::Compound { members, .. } => {
            let fields = members
                .iter()
                .map(|m| (m.name.clone(), classify_datatype(&m.datatype)))
                .collect();
            DType::Compound(fields)
        }
        Datatype::Enumeration { members, .. } => {
            let names = members.iter().map(|m| m.name.clone()).collect();
            DType::Enum(names)
        }
        Datatype::Array {
            base_type,
            dimensions,
        } => DType::Array(Box::new(classify_datatype(base_type)), dimensions.clone()),
        Datatype::Reference {
            ref_type: crate::datatype::ReferenceType::Object,
            ..
        } => DType::ObjectReference,
        _ => DType::Other(format!("{dt:?}")),
    }
}

/// Read attribute messages into a `HashMap<String, AttrValue>`.
///
/// Best-effort: attributes that can't be decoded are silently skipped.
/// `base_address` is the file-level userblock offset — needed so that
/// variable-length attribute data (stored in global heap collections with
/// addresses relative to the base) can be located correctly.
pub(crate) fn attrs_to_map<S: crate::source::Source + ?Sized>(
    attrs: &[crate::attribute::AttributeMessage],
    source: &S,
    offset_size: u8,
    length_size: u8,
    base_address: u64,
) -> HashMap<std::string::String, AttrValue> {
    let mut map = HashMap::new();
    for attr in attrs {
        if let Some(val) = decode_attr_value(attr, source, offset_size, length_size, base_address) {
            map.insert(attr.name.clone(), val);
        }
    }
    map
}

/// Decode one attribute message into the [`AttrValue`] variant that describes
/// what the file holds.
///
/// The datatype and the dataspace together determine the variant, so a value
/// this crate wrote reads back as the variant it was written from. The
/// dataspace kind — not the element count — decides scalar against array, so a
/// one-element array stays an array. Charset selects the `Ascii*` variants.
///
/// Two things are still not recoverable. Integer and float widths are widened
/// to `i64`/`u64`/`f64`, since there are no narrower array variants; and a true
/// variable-length string (`H5T_STRING` with `STRSIZE = VAR`, which this
/// crate's writer never emits) has no variant of its own, so it reads as the
/// fixed-width variant of the same charset and arity and would be rewritten
/// fixed-width.
fn decode_attr_value<S: crate::source::Source + ?Sized>(
    attr: &crate::attribute::AttributeMessage,
    source: &S,
    offset_size: u8,
    length_size: u8,
    base_address: u64,
) -> Option<AttrValue> {
    use crate::dataspace::DataspaceType;
    use crate::datatype::{CharacterSet, Datatype};

    // A scalar dataspace and a 1-element simple dataspace are different on
    // disk (v1 carries rank 0, v2 a type byte), and the write side picks
    // between them per variant, so this is what makes the round trip faithful
    // at length one.
    let scalar = attr.dataspace.space_type == DataspaceType::Scalar;

    match &attr.datatype {
        Datatype::FloatingPoint { .. } => {
            let vals = attr.read_as_f64().ok()?;
            if scalar {
                Some(AttrValue::F64(*vals.first()?))
            } else {
                Some(AttrValue::F64Array(vals))
            }
        }
        Datatype::FixedPoint { signed: true, .. } => {
            let vals = attr.read_as_i64().ok()?;
            if scalar {
                Some(AttrValue::I64(*vals.first()?))
            } else {
                Some(AttrValue::I64Array(vals))
            }
        }
        Datatype::FixedPoint { signed: false, .. } => {
            let vals = attr.read_as_u64().ok()?;
            if scalar {
                Some(AttrValue::U64(*vals.first()?))
            } else {
                // No U64Array variant, store as I64Array
                #[expect(
                    clippy::cast_possible_wrap,
                    reason = "no U64Array AttrValue variant; values above i64::MAX are \
                              reinterpreted as i64 by design (bit pattern preserved)"
                )]
                let i64_vals: Vec<i64> = vals.iter().map(|&v| v as i64).collect();
                Some(AttrValue::I64Array(i64_vals))
            }
        }
        Datatype::String { charset, .. } => {
            let strings = attr.read_as_strings().ok()?;
            let ascii = *charset == CharacterSet::Ascii;
            match (ascii, scalar) {
                (true, true) => Some(AttrValue::AsciiString(strings.into_iter().next()?)),
                (true, false) => Some(AttrValue::AsciiStringArray(strings)),
                (false, true) => Some(AttrValue::String(strings.into_iter().next()?)),
                (false, false) => Some(AttrValue::StringArray(strings)),
            }
        }
        Datatype::VariableLength {
            is_string,
            base_type,
            charset,
            ..
        } if *is_string || is_ascii_char_vlen_base(base_type) => {
            // Two MATLAB-relevant encodings share the same on-disk byte
            // layout (length + heap ref + object index per element; heap
            // object holds raw bytes without terminator):
            //   - is_string: true             — H5T_STRING{STRSIZE=VAR}
            //   - VLEN of H5T_STRING{SIZE=1}  — what matio / MATLAB emit
            //
            // The reader resolves each element from the global heap, adding
            // `base_address` to the (relative) collection addresses.
            let strings = crate::vl_data::read_vl_strings_from_source(
                source,
                &attr.raw_data,
                attr.dataspace.num_elements(),
                offset_size,
                length_size,
                base_address,
                crate::vl_data::VlenStringReadOptions::default(),
            )
            .ok()?;
            // The VLEN-of-1-byte-ASCII encoding is the one this crate writes,
            // for `VarLenAsciiArray`, so it reads back as that variant. A true
            // variable-length string has no variant of its own; keep its
            // charset and arity and let it read as the fixed-width variant.
            let ascii_char_vlen = is_ascii_char_vlen_base(base_type);
            if ascii_char_vlen && !scalar {
                return Some(AttrValue::VarLenAsciiArray(strings));
            }
            let ascii = ascii_char_vlen || *charset == Some(CharacterSet::Ascii);
            match (ascii, scalar) {
                (true, true) => Some(AttrValue::AsciiString(strings.into_iter().next()?)),
                (true, false) => Some(AttrValue::AsciiStringArray(strings)),
                (false, true) => Some(AttrValue::String(strings.into_iter().next()?)),
                (false, false) => Some(AttrValue::StringArray(strings)),
            }
        }
        _ => None,
    }
}

/// Recognize the MATLAB-style VLEN encoding where the base type is a 1-byte
/// ASCII string (`H5T_VLEN { H5T_STRING { STRSIZE 1, ..., CSET ASCII } }`).
/// Other VLEN sequences of strings may exist but we only auto-decode this
/// specific shape as a string array.
fn is_ascii_char_vlen_base(base: &crate::datatype::Datatype) -> bool {
    use crate::datatype::{CharacterSet, Datatype};
    matches!(
        base,
        Datatype::String {
            size: 1,
            charset: CharacterSet::Ascii,
            ..
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatype::{Datatype, DatatypeByteOrder};

    #[test]
    fn classify_unusual_size_does_not_overflow() {
        // A crafted datatype `size` near `u32::MAX` must not overflow the
        // `size * 8` bit-width computation used for the `Other(..)` label
        // (issue #140); the value is widened to `u64` first.
        let int = Datatype::FixedPoint {
            size: u32::MAX,
            byte_order: DatatypeByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 0,
        };
        assert_eq!(
            classify_datatype(&int),
            DType::Other(format!("i{}", u64::from(u32::MAX) * 8))
        );

        let float = Datatype::FloatingPoint {
            size: u32::MAX,
            byte_order: DatatypeByteOrder::LittleEndian,
            bit_offset: 0,
            bit_precision: 0,
            exponent_location: 0,
            exponent_size: 0,
            mantissa_location: 0,
            mantissa_size: 0,
            exponent_bias: 0,
        };
        assert_eq!(
            classify_datatype(&float),
            DType::Other(format!("float{}", u64::from(u32::MAX) * 8))
        );
    }

    /// Write one attribute per case, read every attribute back, and return the
    /// values keyed by name. Goes through the real writer and reader, in
    /// memory, so what it measures is the file rather than a constructed
    /// message.
    fn round_trip(cases: &[(&str, AttrValue)]) -> HashMap<std::string::String, AttrValue> {
        let mut builder = crate::writer::FileBuilder::new();
        for (name, value) in cases {
            builder.set_attr(*name, value.clone());
        }
        // A dataset gives the global heap a reason to exist for the
        // variable-length cases, matching how these files are really built.
        builder.create_dataset("x").with_f64_data(&[1.0]);
        let bytes = builder.finish().unwrap();
        crate::File::from_bytes(bytes)
            .unwrap()
            .root()
            .attrs()
            .unwrap()
    }

    /// The variant a string attribute was written from is the variant it reads
    /// back as. The one-element arrays are the point: charset and dataspace
    /// kind are both on disk, so neither collapses into the scalar form.
    #[test]
    fn every_string_variant_round_trips_to_itself() {
        let cases = vec![
            ("utf8_scalar", AttrValue::String("m/s".into())),
            ("utf8_one", AttrValue::StringArray(vec!["m/s".into()])),
            (
                "utf8_two",
                AttrValue::StringArray(vec!["m/s".into(), "kg".into()]),
            ),
            ("ascii_scalar", AttrValue::AsciiString("double".into())),
            (
                "ascii_one",
                AttrValue::AsciiStringArray(vec!["double".into()]),
            ),
            (
                "ascii_two",
                AttrValue::AsciiStringArray(vec!["double".into(), "int16".into()]),
            ),
            ("vlen_one", AttrValue::VarLenAsciiArray(vec!["x".into()])),
            (
                "vlen_three",
                AttrValue::VarLenAsciiArray(vec!["x".into(), "y".into(), "velocity".into()]),
            ),
        ];
        let read = round_trip(&cases);
        for (name, written) in &cases {
            assert_eq!(read.get(*name), Some(written), "attribute {name}");
        }
    }

    /// Array-ness survives at length one for numbers too.
    #[test]
    fn every_numeric_variant_round_trips_to_itself() {
        let cases = vec![
            ("f64_scalar", AttrValue::F64(1.5)),
            ("f64_one", AttrValue::F64Array(vec![1.5])),
            ("f64_two", AttrValue::F64Array(vec![1.5, 2.5])),
            ("i64_scalar", AttrValue::I64(-7)),
            ("i64_one", AttrValue::I64Array(vec![-7])),
            ("i64_two", AttrValue::I64Array(vec![-7, 8])),
            ("u64_scalar", AttrValue::U64(7)),
        ];
        let read = round_trip(&cases);
        for (name, written) in &cases {
            assert_eq!(read.get(*name), Some(written), "attribute {name}");
        }
    }

    /// Width is the one thing the read side still cannot recover: there are no
    /// narrower array variants, so a 32-bit attribute widens. This pins the
    /// documented limitation rather than endorsing it — if narrower variants
    /// are ever added, this test is the one that should fail.
    #[test]
    fn integer_width_is_not_recovered() {
        let read = round_trip(&[("i32", AttrValue::I32(-7)), ("u32", AttrValue::U32(7))]);
        assert_eq!(read.get("i32"), Some(&AttrValue::I64(-7)));
        assert_eq!(read.get("u32"), Some(&AttrValue::U64(7)));
    }

    /// The accessors are what a consumer should use, and they read every shape
    /// above as the same logical value — which is the reason the reader is free
    /// to be faithful about the variant.
    #[test]
    fn accessors_span_the_shapes_the_reader_now_distinguishes() {
        let read = round_trip(&[
            ("scalar", AttrValue::AsciiString("double".into())),
            ("one", AttrValue::StringArray(vec!["double".into()])),
            ("vlen", AttrValue::VarLenAsciiArray(vec!["double".into()])),
        ]);
        for name in ["scalar", "one", "vlen"] {
            assert_eq!(
                read.get(name).and_then(AttrValue::as_str),
                Some("double"),
                "attribute {name}"
            );
        }
    }
}
