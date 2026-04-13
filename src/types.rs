//! Simplified type representations for the high-level API.

use std::collections::HashMap;
use std::fmt;

pub use crate::file_writer::AttrValue;

/// Simplified datatype enum for the high-level API.
///
/// Maps from the detailed `crate::datatype::Datatype` to a
/// user-friendly representation.
#[derive(Debug, Clone, PartialEq)]
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
        Datatype::FloatingPoint { size, .. } => DType::Other(format!("float{}", size * 8)),
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
            DType::Other(format!("{prefix}{}", size * 8))
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
pub(crate) fn attrs_to_map(
    attrs: &[crate::attribute::AttributeMessage],
    file_data: &[u8],
    offset_size: u8,
    length_size: u8,
) -> HashMap<std::string::String, AttrValue> {
    let mut map = HashMap::new();
    for attr in attrs {
        if let Some(val) = decode_attr_value(attr, file_data, offset_size, length_size) {
            map.insert(attr.name.clone(), val);
        }
    }
    map
}

fn decode_attr_value(
    attr: &crate::attribute::AttributeMessage,
    file_data: &[u8],
    offset_size: u8,
    length_size: u8,
) -> Option<AttrValue> {
    use crate::datatype::Datatype;

    match &attr.datatype {
        Datatype::FloatingPoint { .. } => {
            let vals = attr.read_as_f64().ok()?;
            if vals.len() == 1 {
                Some(AttrValue::F64(vals[0]))
            } else {
                Some(AttrValue::F64Array(vals))
            }
        }
        Datatype::FixedPoint { signed: true, .. } => {
            let vals = attr.read_as_i64().ok()?;
            if vals.len() == 1 {
                Some(AttrValue::I64(vals[0]))
            } else {
                Some(AttrValue::I64Array(vals))
            }
        }
        Datatype::FixedPoint { signed: false, .. } => {
            let vals = attr.read_as_u64().ok()?;
            if vals.len() == 1 {
                Some(AttrValue::U64(vals[0]))
            } else {
                // No U64Array variant, store as I64Array
                let i64_vals: Vec<i64> = vals.iter().map(|&v| v as i64).collect();
                Some(AttrValue::I64Array(i64_vals))
            }
        }
        Datatype::String { .. } => {
            let strings = attr.read_as_strings().ok()?;
            if strings.len() == 1 {
                Some(AttrValue::String(strings[0].clone()))
            } else {
                Some(AttrValue::StringArray(strings))
            }
        }
        Datatype::VariableLength {
            is_string: true, ..
        } => {
            let strings = attr
                .read_vl_strings(file_data, offset_size, length_size)
                .ok()?;
            if strings.len() == 1 {
                Some(AttrValue::String(strings[0].clone()))
            } else {
                Some(AttrValue::StringArray(strings))
            }
        }
        _ => None,
    }
}
