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
/// What is still not recoverable, because [`AttrValue`] has no way to express
/// it — each of these reads correctly but would be rewritten differently:
///
/// - **Width.** Integers and floats widen to `i64`/`u64`/`f64`; there are no
///   narrower array variants.
/// - **Variable-length strings.** A true `H5T_STRING` with `STRSIZE = VAR`,
///   which this crate's writer never emits, has no variant of its own and reads
///   as the fixed-width variant of the same charset and arity.
/// - **Rank.** Every array variant is one-dimensional, so a rank-2 attribute
///   reads as its elements flattened.
/// - **Padding and declared width.** A fixed-width string reports its content,
///   not its `STRSIZE` or whether it was null-terminated, null-padded or
///   space-padded.
/// - **Null dataspaces.** These read as an empty array variant.
///
/// A numeric attribute whose message holds fewer bytes than its dataspace
/// promises is reported undecodable (`None`) rather than defaulted, since no
/// value would be truthful. An empty *string* is different: its zero-size
/// datatype legitimately decodes to no elements, and the empty string is the
/// value, so it is kept.
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
                Some(AttrValue::U64Array(vals))
            }
        }
        Datatype::String { charset, .. } => {
            let strings = attr.read_as_strings().ok()?;
            let ascii = *charset == CharacterSet::Ascii;
            // A zero-size string datatype decodes to no elements at all, so a
            // scalar takes the empty string rather than reporting the whole
            // attribute undecodable — `attrs_to_map` drops what this returns
            // `None` for, and an empty string attribute must not disappear.
            match (ascii, scalar) {
                (true, true) => Some(AttrValue::AsciiString(one_or_empty(strings))),
                (true, false) => Some(AttrValue::AsciiStringArray(strings)),
                (false, true) => Some(AttrValue::String(one_or_empty(strings))),
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
            match (
                vlen_string_shape(*is_string, base_type, charset.as_ref()),
                scalar,
            ) {
                (VlenStringShape::AsciiCharSequence, false) => {
                    Some(AttrValue::VarLenAsciiArray(strings))
                }
                (VlenStringShape::AsciiCharSequence | VlenStringShape::Ascii, true) => {
                    Some(AttrValue::AsciiString(one_or_empty(strings)))
                }
                (VlenStringShape::Ascii, false) => Some(AttrValue::AsciiStringArray(strings)),
                (VlenStringShape::Utf8, true) => Some(AttrValue::String(one_or_empty(strings))),
                (VlenStringShape::Utf8, false) => Some(AttrValue::StringArray(strings)),
            }
        }
        _ => None,
    }
}

/// Which family of `AttrValue` variants a variable-length string attribute
/// belongs to, decided from its datatype alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VlenStringShape {
    /// A VLEN *sequence* of 1-byte ASCII strings — the encoding MATLAB and matio
    /// use, and the one this crate writes for
    /// [`AttrValue::VarLenAsciiArray`]. Only this shape has a variant that
    /// preserves it.
    AsciiCharSequence,
    /// A true variable-length ASCII string (`H5T_STRING`, `STRSIZE = VAR`).
    Ascii,
    /// A true variable-length UTF-8 string, or one whose charset is unstated.
    Utf8,
}

/// Classify a variable-length string datatype.
///
/// `is_string` is what separates a true variable-length string from a sequence
/// of 1-byte strings, and it has to be consulted: libhdf5 writes a VL string's
/// base type as a 1-byte *integer*, so the base type alone happens to be enough
/// for the files it produces — but nothing in the format stops a writer from
/// giving a VL string a 1-byte *string* base, which is byte-identical to the
/// MATLAB sequence's base. Trusting the base type alone would then report the
/// MATLAB encoding for a value that is not in it, and rewrite it as a different
/// datatype class.
fn vlen_string_shape(
    is_string: bool,
    base_type: &crate::datatype::Datatype,
    charset: Option<&crate::datatype::CharacterSet>,
) -> VlenStringShape {
    use crate::datatype::CharacterSet;
    if !is_string && is_ascii_char_vlen_base(base_type) {
        return VlenStringShape::AsciiCharSequence;
    }
    // A VL string states its own charset; a sequence of ASCII chars carries it
    // on the base type instead.
    if charset == Some(&CharacterSet::Ascii) || is_ascii_char_vlen_base(base_type) {
        VlenStringShape::Ascii
    } else {
        VlenStringShape::Utf8
    }
}

/// The single string a scalar attribute holds, or the empty string when the
/// datatype decoded to no elements at all.
///
/// A zero-size string datatype yields no elements (`read_as_strings` returns an
/// empty vec), which is how an empty string attribute is stored. Reporting the
/// attribute undecodable there would drop it from `attrs()` entirely, since
/// `attrs_to_map` keeps only what decodes.
fn one_or_empty(strings: Vec<std::string::String>) -> std::string::String {
    strings.into_iter().next().unwrap_or_default()
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
            builder.set_attr(name, value.clone());
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
            ("u64_one", AttrValue::U64Array(vec![7])),
            ("u64_two", AttrValue::U64Array(vec![7, 8])),
        ];
        let read = round_trip(&cases);
        for (name, written) in &cases {
            assert_eq!(read.get(*name), Some(written), "attribute {name}");
        }
    }

    /// An unsigned value above `i64::MAX` reads back as itself. It used to be
    /// reinterpreted into a negative `i64` for want of a `U64Array` variant, so
    /// this is the case that variant exists for — and the accessors report the
    /// value rather than a wrapped one at every length.
    #[test]
    fn a_full_range_unsigned_value_survives_at_every_length() {
        let read = round_trip(&[
            ("scalar", AttrValue::U64(u64::MAX)),
            ("one", AttrValue::U64Array(vec![u64::MAX])),
            ("two", AttrValue::U64Array(vec![u64::MAX, 1])),
        ]);
        assert_eq!(read.get("scalar"), Some(&AttrValue::U64(u64::MAX)));
        assert_eq!(read.get("one"), Some(&AttrValue::U64Array(vec![u64::MAX])));
        assert_eq!(
            read.get("two"),
            Some(&AttrValue::U64Array(vec![u64::MAX, 1]))
        );
        // Read through the accessors: the full-range value is reported as
        // unsigned, and asking for it as `i64` refuses rather than wrapping.
        for (name, expected) in [
            ("scalar", vec![u64::MAX]),
            ("one", vec![u64::MAX]),
            ("two", vec![u64::MAX, 1]),
        ] {
            let value = read.get(name).expect("present");
            assert_eq!(value.to_u64s(), Some(expected), "{name} must read unsigned");
            assert_eq!(
                value.to_i64s(),
                None,
                "{name} does not fit an i64 and must not wrap"
            );
        }
        assert_eq!(read["scalar"].as_u64(), Some(u64::MAX));
        assert_eq!(read["one"].as_u64(), Some(u64::MAX));
        assert_eq!(
            read["two"].as_u64(),
            None,
            "two elements are not a single value"
        );
    }

    /// The MATLAB sequence-of-ASCII-chars encoding and a true variable-length
    /// ASCII string have the same base type once a writer chooses a 1-byte
    /// string base for the latter; `is_string` is the only thing separating
    /// them. libhdf5 writes an integer base, so no file it produces reaches the
    /// ambiguous case — which is exactly why this is asserted here rather than
    /// left to the C crosscheck, where the branch cannot be reached.
    #[test]
    fn only_a_sequence_of_ascii_chars_claims_the_varlen_variant() {
        use crate::datatype::{CharacterSet, Datatype, DatatypeByteOrder, StringPadding};

        let char_base = Datatype::String {
            size: 1,
            padding: StringPadding::NullTerminate,
            charset: CharacterSet::Ascii,
        };
        let int_base = Datatype::FixedPoint {
            size: 1,
            byte_order: DatatypeByteOrder::LittleEndian,
            signed: false,
            bit_offset: 0,
            bit_precision: 8,
        };

        // What MATLAB and matio write, and what this crate writes.
        assert_eq!(
            vlen_string_shape(false, &char_base, None),
            VlenStringShape::AsciiCharSequence
        );
        // The same base type, but flagged a string: a true variable-length
        // string, which must not claim the sequence variant.
        assert_eq!(
            vlen_string_shape(true, &char_base, Some(&CharacterSet::Ascii)),
            VlenStringShape::Ascii
        );
        // What libhdf5 actually writes for a variable-length string.
        assert_eq!(
            vlen_string_shape(true, &int_base, Some(&CharacterSet::Ascii)),
            VlenStringShape::Ascii
        );
        assert_eq!(
            vlen_string_shape(true, &int_base, Some(&CharacterSet::Utf8)),
            VlenStringShape::Utf8
        );
        // An unstated charset reads as UTF-8, which is the lossless assumption:
        // every ASCII string is valid UTF-8, so nothing is corrupted by it.
        assert_eq!(
            vlen_string_shape(true, &int_base, None),
            VlenStringShape::Utf8
        );
    }

    /// An empty string attribute stays present and stays empty.
    ///
    /// A zero-length string is stored with a zero-size datatype, which decodes
    /// to no elements; treating that as undecodable dropped the attribute out of
    /// `attrs()` entirely, because `attrs_to_map` keeps only what decodes.
    #[test]
    fn an_empty_string_attribute_is_not_dropped() {
        let cases = vec![
            ("utf8", AttrValue::String(std::string::String::new())),
            ("ascii", AttrValue::AsciiString(std::string::String::new())),
        ];
        let read = round_trip(&cases);
        for (name, written) in &cases {
            assert_eq!(
                read.get(*name),
                Some(written),
                "attribute {name} must survive with its empty value"
            );
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
