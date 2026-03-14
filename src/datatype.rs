//! HDF5 Datatype message parsing (message type 0x0003).
//!
//! Supports all 12 HDF5 type classes (0–11) with recursive parsing
//! for compound, enumeration, variable-length, and array types.

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec, vec::Vec};

use byteorder::{ByteOrder, LittleEndian};

use crate::error::FormatError;

/// Byte order of numeric data.
#[derive(Debug, Clone, PartialEq)]
pub enum DatatypeByteOrder {
    LittleEndian,
    BigEndian,
    Vax,
}

/// String padding type.
#[derive(Debug, Clone, PartialEq)]
pub enum StringPadding {
    NullTerminate,
    NullPad,
    SpacePad,
}

/// Character set encoding.
#[derive(Debug, Clone, PartialEq)]
pub enum CharacterSet {
    Ascii,
    Utf8,
}

/// Reference type.
#[derive(Debug, Clone, PartialEq)]
pub enum ReferenceType {
    Object,
    DatasetRegion,
}

/// A member of a compound datatype.
#[derive(Debug, Clone, PartialEq)]
pub struct CompoundMember {
    /// Member name.
    pub name: String,
    /// Byte offset within the compound.
    pub byte_offset: u64,
    /// Member datatype.
    pub datatype: Datatype,
}

/// A member of an enumeration datatype.
#[derive(Debug, Clone, PartialEq)]
pub struct EnumMember {
    /// Member name.
    pub name: String,
    /// Raw value bytes (length = base type size).
    pub value: Vec<u8>,
}

/// Parsed HDF5 datatype.
#[derive(Debug, Clone, PartialEq)]
pub enum Datatype {
    /// Class 0: Fixed-point (integer) types.
    FixedPoint {
        size: u32,
        byte_order: DatatypeByteOrder,
        signed: bool,
        bit_offset: u16,
        bit_precision: u16,
    },
    /// Class 1: Floating-point types.
    FloatingPoint {
        size: u32,
        byte_order: DatatypeByteOrder,
        bit_offset: u16,
        bit_precision: u16,
        exponent_location: u8,
        exponent_size: u8,
        mantissa_location: u8,
        mantissa_size: u8,
        exponent_bias: u32,
    },
    /// Class 2: Time type (rarely used).
    Time {
        size: u32,
        bit_precision: u16,
    },
    /// Class 3: Fixed-length string.
    String {
        size: u32,
        padding: StringPadding,
        charset: CharacterSet,
    },
    /// Class 4: Bit field.
    BitField {
        size: u32,
        byte_order: DatatypeByteOrder,
        bit_offset: u16,
        bit_precision: u16,
    },
    /// Class 5: Opaque data.
    Opaque {
        size: u32,
        tag: Vec<u8>,
    },
    /// Class 6: Compound type.
    Compound {
        size: u32,
        members: Vec<CompoundMember>,
    },
    /// Class 7: Reference type.
    Reference {
        size: u32,
        ref_type: ReferenceType,
    },
    /// Class 8: Enumeration type.
    Enumeration {
        size: u32,
        base_type: Box<Datatype>,
        members: Vec<EnumMember>,
    },
    /// Class 9: Variable-length type.
    VariableLength {
        is_string: bool,
        padding: Option<StringPadding>,
        charset: Option<CharacterSet>,
        base_type: Box<Datatype>,
    },
    /// Class 10: Array type.
    Array {
        base_type: Box<Datatype>,
        dimensions: Vec<u32>,
    },
}

fn ensure_len(data: &[u8], offset: usize, needed: usize) -> Result<(), FormatError> {
    match offset.checked_add(needed) {
        Some(end) if end <= data.len() => Ok(()),
        _ => Err(FormatError::UnexpectedEof {
            expected: offset.saturating_add(needed),
            available: data.len(),
        }),
    }
}

fn parse_string_padding(val: u8) -> Result<StringPadding, FormatError> {
    match val {
        0 => Ok(StringPadding::NullTerminate),
        1 => Ok(StringPadding::NullPad),
        2 => Ok(StringPadding::SpacePad),
        _ => Err(FormatError::InvalidStringPadding(val)),
    }
}

fn parse_charset(val: u8) -> Result<CharacterSet, FormatError> {
    match val {
        0 => Ok(CharacterSet::Ascii),
        1 => Ok(CharacterSet::Utf8),
        _ => Err(FormatError::InvalidCharacterSet(val)),
    }
}

/// Read a null-terminated string from `data` starting at `offset`.
/// Returns (string, bytes_consumed including the null terminator).
fn read_null_terminated_string(data: &[u8], offset: usize) -> Result<(String, usize), FormatError> {
    if offset >= data.len() {
        return Err(FormatError::UnexpectedEof {
            expected: offset + 1,
            available: data.len(),
        });
    }
    let remaining = &data[offset..];
    let null_pos = remaining
        .iter()
        .position(|&b| b == 0)
        .ok_or(FormatError::UnexpectedEof {
            expected: offset + 1,
            available: data.len(),
        })?;
    let name = String::from_utf8_lossy(&remaining[..null_pos]).into_owned();
    Ok((name, null_pos + 1))
}

/// Determine how many bytes are needed to encode `compound_size` as a byte offset (v3).
fn offset_bytes_for_size(compound_size: u32) -> usize {
    if compound_size <= 0xFF {
        1
    } else if compound_size <= 0xFFFF {
        2
    } else {
        4
    }
}

/// Read an unsigned integer of 1, 2, 4, or 8 bytes (LE).
fn read_uint(data: &[u8], offset: usize, nbytes: usize) -> Result<u64, FormatError> {
    ensure_len(data, offset, nbytes)?;
    let slice = &data[offset..offset + nbytes];
    Ok(match nbytes {
        1 => slice[0] as u64,
        2 => LittleEndian::read_u16(slice) as u64,
        4 => LittleEndian::read_u32(slice) as u64,
        8 => LittleEndian::read_u64(slice),
        _ => {
            return Err(FormatError::UnexpectedEof {
                expected: offset + nbytes,
                available: data.len(),
            });
        }
    })
}

impl Datatype {
    /// Parse a datatype message from raw bytes.
    ///
    /// Returns `(Datatype, bytes_consumed)` for recursive parsing.
    pub fn parse(data: &[u8]) -> Result<(Datatype, usize), FormatError> {
        // Minimum header: 4 bytes (class_and_version + 3 bytes bit field) + 4 bytes size = 8
        ensure_len(data, 0, 8)?;

        let class_and_version = data[0];
        let class_id = class_and_version & 0x0F;
        let version = (class_and_version >> 4) & 0x0F;

        // 24-bit class bit field (little-endian)
        let bf0 = data[1];
        let bf1 = data[2];
        let bf2 = data[3];
        let _bit_field_24 = (bf0 as u32) | ((bf1 as u32) << 8) | ((bf2 as u32) << 16);

        let size = LittleEndian::read_u32(&data[4..8]);
        let mut pos = 8;

        match class_id {
            0 => {
                // Fixed-Point
                ensure_len(data, pos, 4)?;
                let byte_order = if bf0 & 0x01 == 0 {
                    DatatypeByteOrder::LittleEndian
                } else {
                    DatatypeByteOrder::BigEndian
                };
                let signed = (bf0 >> 3) & 0x01 == 1;
                let bit_offset = LittleEndian::read_u16(&data[pos..pos + 2]);
                let bit_precision = LittleEndian::read_u16(&data[pos + 2..pos + 4]);
                pos += 4;
                Ok((
                    Datatype::FixedPoint {
                        size,
                        byte_order,
                        signed,
                        bit_offset,
                        bit_precision,
                    },
                    pos,
                ))
            }
            1 => {
                // Floating-Point
                ensure_len(data, pos, 12)?;
                let bo_low = bf0 & 0x01;
                let bo_high = (bf0 >> 6) & 0x01;
                let byte_order = match (bo_high, bo_low) {
                    (0, 0) => DatatypeByteOrder::LittleEndian,
                    (0, 1) => DatatypeByteOrder::BigEndian,
                    (1, 0) => DatatypeByteOrder::Vax,
                    (1, 1) => DatatypeByteOrder::Vax,
                    _ => unreachable!(),
                };
                let bit_offset = LittleEndian::read_u16(&data[pos..pos + 2]);
                let bit_precision = LittleEndian::read_u16(&data[pos + 2..pos + 4]);
                let exponent_location = data[pos + 4];
                let exponent_size = data[pos + 5];
                let mantissa_location = data[pos + 6];
                let mantissa_size = data[pos + 7];
                let exponent_bias = LittleEndian::read_u32(&data[pos + 8..pos + 12]);
                pos += 12;
                Ok((
                    Datatype::FloatingPoint {
                        size,
                        byte_order,
                        bit_offset,
                        bit_precision,
                        exponent_location,
                        exponent_size,
                        mantissa_location,
                        mantissa_size,
                        exponent_bias,
                    },
                    pos,
                ))
            }
            2 => {
                // Time
                ensure_len(data, pos, 2)?;
                let bit_precision = LittleEndian::read_u16(&data[pos..pos + 2]);
                pos += 2;
                Ok((Datatype::Time { size, bit_precision }, pos))
            }
            3 => {
                // String
                let padding_val = bf0 & 0x0F;
                let charset_val = (bf0 >> 4) & 0x0F;
                let padding = parse_string_padding(padding_val)?;
                let charset = parse_charset(charset_val)?;
                Ok((Datatype::String { size, padding, charset }, pos))
            }
            4 => {
                // Bit Field
                ensure_len(data, pos, 4)?;
                let byte_order = if bf0 & 0x01 == 0 {
                    DatatypeByteOrder::LittleEndian
                } else {
                    DatatypeByteOrder::BigEndian
                };
                let bit_offset = LittleEndian::read_u16(&data[pos..pos + 2]);
                let bit_precision = LittleEndian::read_u16(&data[pos + 2..pos + 4]);
                pos += 4;
                Ok((
                    Datatype::BitField {
                        size,
                        byte_order,
                        bit_offset,
                        bit_precision,
                    },
                    pos,
                ))
            }
            5 => {
                // Opaque
                let tag_len = bf0 as usize;
                ensure_len(data, pos, tag_len)?;
                let tag = data[pos..pos + tag_len].to_vec();
                // Tags are padded to multiple of 8 bytes
                let padded = (tag_len + 7) & !7;
                let pos = 8 + padded; // from start of properties
                Ok((Datatype::Opaque { size, tag }, pos))
            }
            6 => {
                // Compound
                let num_members = (bf0 as u16) | ((bf1 as u16) << 8);
                let mut members = Vec::with_capacity(num_members as usize);

                if version == 3 || version == 4 {
                    let ob = offset_bytes_for_size(size);
                    for _ in 0..num_members {
                        let (name, name_len) = read_null_terminated_string(data, pos)?;
                        pos += name_len;
                        let byte_offset = read_uint(data, pos, ob)?;
                        pos += ob;
                        let (member_dt, consumed) = Datatype::parse(&data[pos..])?;
                        pos += consumed;
                        members.push(CompoundMember {
                            name,
                            byte_offset,
                            datatype: member_dt,
                        });
                    }
                } else if version == 1 || version == 2 {
                    // v1/v2: name, offset(4), dimensionality(1), reserved(3), dim_perm(4),
                    //         reserved_dims(up to 4*4=16), member datatype
                    for _ in 0..num_members {
                        let (name, name_len) = read_null_terminated_string(data, pos)?;
                        pos += name_len;
                        // v1: names padded to 8-byte boundary
                        if version == 1 {
                            let total_name_bytes = name_len;
                            let padded = (total_name_bytes + 7) & !7;
                            pos = pos - name_len + padded;
                        }
                        ensure_len(data, pos, 4)?;
                        let byte_offset = LittleEndian::read_u32(&data[pos..pos + 4]) as u64;
                        pos += 4;
                        // dimensionality(1) + reserved(3) + dim_perm(4) + 4 dim slots(16) = 24
                        ensure_len(data, pos, 24)?;
                        pos += 24;
                        let (member_dt, consumed) = Datatype::parse(&data[pos..])?;
                        pos += consumed;
                        members.push(CompoundMember {
                            name,
                            byte_offset,
                            datatype: member_dt,
                        });
                    }
                } else {
                    return Err(FormatError::InvalidDatatypeVersion {
                        class: class_id,
                        version,
                    });
                }

                Ok((Datatype::Compound { size, members }, pos))
            }
            7 => {
                // Reference
                let ref_type_val = bf0 & 0x0F;
                let ref_type = match ref_type_val {
                    0 => ReferenceType::Object,
                    1 => ReferenceType::DatasetRegion,
                    _ => return Err(FormatError::InvalidReferenceType(ref_type_val)),
                };
                Ok((Datatype::Reference { size, ref_type }, pos))
            }
            8 => {
                // Enumeration
                let num_members = (bf0 as u16) | ((bf1 as u16) << 8);
                // Parse base type
                let (base_type, base_consumed) = Datatype::parse(&data[pos..])?;
                pos += base_consumed;
                let base_size = base_type.type_size();
                let mut members = Vec::with_capacity(num_members as usize);
                // Enum layout: base_type, then all names (null-terminated), then all values
                // v1/v2: names are padded to 8-byte boundaries
                // v3: names are just null-terminated
                let mut member_names = Vec::with_capacity(num_members as usize);
                for _ in 0..num_members {
                    let (name, name_len) = read_null_terminated_string(data, pos)?;
                    if version < 3 {
                        let padded = (name_len + 7) & !7;
                        pos += padded;
                    } else {
                        pos += name_len;
                    }
                    member_names.push(name);
                }
                // Now values
                for name in &member_names {
                    ensure_len(data, pos, base_size as usize)?;
                    let value = data[pos..pos + base_size as usize].to_vec();
                    pos += base_size as usize;
                    members.push(EnumMember {
                        name: name.clone(),
                        value,
                    });
                }
                Ok((
                    Datatype::Enumeration {
                        size,
                        base_type: Box::new(base_type),
                        members,
                    },
                    pos,
                ))
            }
            9 => {
                // Variable-Length
                let vl_type = bf0 & 0x0F;
                let is_string = vl_type == 1;
                let padding = if is_string {
                    let pad_val = (bf0 >> 4) & 0x0F;
                    Some(parse_string_padding(pad_val)?)
                } else {
                    None
                };
                let charset = if is_string {
                    let cs_val = bf1 & 0x0F;
                    Some(parse_charset(cs_val)?)
                } else {
                    None
                };
                let (base_type, consumed) = Datatype::parse(&data[pos..])?;
                pos += consumed;
                Ok((
                    Datatype::VariableLength {
                        is_string,
                        padding,
                        charset,
                        base_type: Box::new(base_type),
                    },
                    pos,
                ))
            }
            10 => {
                // Array
                if version == 2 {
                    ensure_len(data, pos, 4)?;
                    let ndims = data[pos] as usize;
                    pos += 4; // ndims(1) + reserved(3)
                    ensure_len(data, pos, ndims * 4 + ndims * 4)?;
                    let mut dimensions = Vec::with_capacity(ndims);
                    for _ in 0..ndims {
                        dimensions.push(LittleEndian::read_u32(&data[pos..pos + 4]));
                        pos += 4;
                    }
                    // skip permutation indices
                    pos += ndims * 4;
                    let (base_type, consumed) = Datatype::parse(&data[pos..])?;
                    pos += consumed;
                    Ok((
                        Datatype::Array {
                            base_type: Box::new(base_type),
                            dimensions,
                        },
                        pos,
                    ))
                } else if version == 3 {
                    ensure_len(data, pos, 1)?;
                    let ndims = data[pos] as usize;
                    pos += 1;
                    ensure_len(data, pos, ndims * 4)?;
                    let mut dimensions = Vec::with_capacity(ndims);
                    for _ in 0..ndims {
                        dimensions.push(LittleEndian::read_u32(&data[pos..pos + 4]));
                        pos += 4;
                    }
                    let (base_type, consumed) = Datatype::parse(&data[pos..])?;
                    pos += consumed;
                    Ok((
                        Datatype::Array {
                            base_type: Box::new(base_type),
                            dimensions,
                        },
                        pos,
                    ))
                } else {
                    Err(FormatError::InvalidDatatypeVersion {
                        class: class_id,
                        version,
                    })
                }
            }
            11 => {
                // Complex number — store as compound of two floats internally
                // Parse like compound with version 3 and 2 members
                // But actually class 11 has no special properties beyond class 6 compound.
                // It's just recognized as a separate class. For now parse the 2 members
                // as compound.
                let num_members = (bf0 as u16) | ((bf1 as u16) << 8);
                let mut members = Vec::with_capacity(num_members as usize);
                let ob = offset_bytes_for_size(size);
                for _ in 0..num_members {
                    let (name, name_len) = read_null_terminated_string(data, pos)?;
                    pos += name_len;
                    let byte_offset = read_uint(data, pos, ob)?;
                    pos += ob;
                    let (member_dt, consumed) = Datatype::parse(&data[pos..])?;
                    pos += consumed;
                    members.push(CompoundMember {
                        name,
                        byte_offset,
                        datatype: member_dt,
                    });
                }
                Ok((Datatype::Compound { size, members }, pos))
            }
            _ => Err(FormatError::InvalidDatatypeClass(class_id)),
        }
    }

    /// Serialize datatype to HDF5 message bytes.
    pub fn serialize(&self) -> Vec<u8> {
        match self {
            Datatype::FixedPoint { size, byte_order, signed, bit_offset, bit_precision } => {
                let mut bf0 = 0u8;
                if matches!(byte_order, DatatypeByteOrder::BigEndian) { bf0 |= 0x01; }
                if *signed { bf0 |= 0x08; }
                let mut buf = Self::build_header(0, 1, [bf0, 0, 0], *size);
                buf.extend_from_slice(&bit_offset.to_le_bytes());
                buf.extend_from_slice(&bit_precision.to_le_bytes());
                buf
            }
            Datatype::FloatingPoint { size, byte_order, bit_offset, bit_precision,
                exponent_location, exponent_size, mantissa_location, mantissa_size, exponent_bias } => {
                let mut bf0 = 0x20u8; // bit 5: sign location bit (standard IEEE 754)
                match byte_order {
                    DatatypeByteOrder::BigEndian => { bf0 |= 0x01; }
                    DatatypeByteOrder::Vax => { bf0 |= 0x40; }
                    _ => {}
                }
                // bf[1] = sign bit location (bit position of sign in the value)
                let bf1 = (*size * 8 - 1) as u8;
                let mut buf = Self::build_header(1, 1, [bf0, bf1, 0], *size);
                buf.extend_from_slice(&bit_offset.to_le_bytes());
                buf.extend_from_slice(&bit_precision.to_le_bytes());
                buf.push(*exponent_location);
                buf.push(*exponent_size);
                buf.push(*mantissa_location);
                buf.push(*mantissa_size);
                buf.extend_from_slice(&exponent_bias.to_le_bytes());
                buf
            }
            Datatype::String { size, padding, charset } => {
                let pad_val = match padding {
                    StringPadding::NullTerminate => 0,
                    StringPadding::NullPad => 1,
                    StringPadding::SpacePad => 2,
                };
                let cs_val = match charset {
                    CharacterSet::Ascii => 0,
                    CharacterSet::Utf8 => 1,
                };
                let bf0 = pad_val | (cs_val << 4);
                Self::build_header(3, 1, [bf0, 0, 0], *size)
            }
            Datatype::VariableLength { is_string, padding, charset, base_type } => {
                let mut bf0 = if *is_string { 0x01u8 } else { 0x00 };
                if *is_string {
                    if let Some(p) = padding {
                        let pv = match p {
                            StringPadding::NullTerminate => 0,
                            StringPadding::NullPad => 1,
                            StringPadding::SpacePad => 2,
                        };
                        bf0 |= pv << 4;
                    }
                }
                let bf1 = if *is_string {
                    charset.as_ref().map_or(0, |c| match c {
                        CharacterSet::Ascii => 0,
                        CharacterSet::Utf8 => 1,
                    })
                } else { 0 };
                let mut buf = Self::build_header(9, 1, [bf0, bf1, 0], 16);
                buf.extend_from_slice(&base_type.serialize());
                buf
            }
            Datatype::Compound { size, members } => {
                let num = members.len() as u16;
                let bf0 = (num & 0xFF) as u8;
                let bf1 = ((num >> 8) & 0xFF) as u8;
                let mut buf = Self::build_header(6, 3, [bf0, bf1, 0], *size);
                let ob = offset_bytes_for_size(*size);
                for m in members {
                    // Null-terminated name
                    buf.extend_from_slice(m.name.as_bytes());
                    buf.push(0);
                    // Byte offset (variable-width)
                    match ob {
                        1 => buf.push(m.byte_offset as u8),
                        2 => buf.extend_from_slice(&(m.byte_offset as u16).to_le_bytes()),
                        _ => buf.extend_from_slice(&(m.byte_offset as u32).to_le_bytes()),
                    }
                    // Recursively serialize member datatype
                    buf.extend_from_slice(&m.datatype.serialize());
                }
                buf
            }
            Datatype::Enumeration { size, base_type, members } => {
                let num = members.len() as u16;
                let bf0 = (num & 0xFF) as u8;
                let bf1 = ((num >> 8) & 0xFF) as u8;
                let mut buf = Self::build_header(8, 3, [bf0, bf1, 0], *size);
                // Base type
                buf.extend_from_slice(&base_type.serialize());
                // All names (null-terminated)
                for m in members {
                    buf.extend_from_slice(m.name.as_bytes());
                    buf.push(0);
                }
                // All values
                for m in members {
                    buf.extend_from_slice(&m.value);
                }
                buf
            }
            Datatype::Array { base_type, dimensions } => {
                let mut buf = Self::build_header(10, 3, [0, 0, 0], self.type_size());
                buf.push(dimensions.len() as u8);
                for &d in dimensions {
                    buf.extend_from_slice(&d.to_le_bytes());
                }
                buf.extend_from_slice(&base_type.serialize());
                buf
            }
            Datatype::Reference { size, ref_type } => {
                let bf0 = match ref_type {
                    ReferenceType::Object => 0,
                    ReferenceType::DatasetRegion => 1,
                };
                Self::build_header(7, 1, [bf0, 0, 0], *size)
            }
            _ => {
                Vec::new()
            }
        }
    }

    fn build_header(class: u8, version: u8, bf: [u8; 3], size: u32) -> Vec<u8> {
        let mut buf = vec![0u8; 8];
        buf[0] = (class & 0x0F) | ((version & 0x0F) << 4);
        buf[1] = bf[0];
        buf[2] = bf[1];
        buf[3] = bf[2];
        buf[4..8].copy_from_slice(&size.to_le_bytes());
        buf
    }

    /// Return the size in bytes of one element of this type.
    pub fn type_size(&self) -> u32 {
        match self {
            Datatype::FixedPoint { size, .. } => *size,
            Datatype::FloatingPoint { size, .. } => *size,
            Datatype::Time { size, .. } => *size,
            Datatype::String { size, .. } => *size,
            Datatype::BitField { size, .. } => *size,
            Datatype::Opaque { size, .. } => *size,
            Datatype::Compound { size, .. } => *size,
            Datatype::Reference { size, .. } => *size,
            Datatype::Enumeration { size, .. } => *size,
            Datatype::VariableLength { .. } => 16, // typically pointer + length
            Datatype::Array { base_type, dimensions } => {
                let elem_count: u32 = dimensions.iter().copied()
                    .fold(1u32, |a, b| a.saturating_mul(b));
                base_type.type_size().saturating_mul(elem_count)
            }
        }
    }
}

/// Build a datatype header (8 bytes) for testing.
#[cfg(test)]
fn build_dt_header(class: u8, version: u8, bf: [u8; 3], size: u32) -> Vec<u8> {
    let mut buf = vec![0u8; 8];
    buf[0] = (class & 0x0F) | ((version & 0x0F) << 4);
    buf[1] = bf[0];
    buf[2] = bf[1];
    buf[3] = bf[2];
    LittleEndian::write_u32(&mut buf[4..8], size);
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to build a fixed-point datatype message
    fn build_fixed_point(size: u32, be: bool, signed: bool, bit_offset: u16, bit_precision: u16) -> Vec<u8> {
        let bf0 = if be { 0x01 } else { 0x00 } | if signed { 0x08 } else { 0x00 };
        let mut buf = build_dt_header(0, 1, [bf0, 0, 0], size);
        let mut props = [0u8; 4];
        LittleEndian::write_u16(&mut props[0..2], bit_offset);
        LittleEndian::write_u16(&mut props[2..4], bit_precision);
        buf.extend_from_slice(&props);
        buf
    }

    // Helper to build a floating-point datatype message
    fn build_float(size: u32, exp_loc: u8, exp_size: u8, mant_loc: u8, mant_size: u8, exp_bias: u32) -> Vec<u8> {
        // LE byte order: bo_low=0, bo_high=0
        let bf0 = 0x00u8;
        let bf1 = 0x00u8;
        // mantissa norm = 2 (MSB not stored) in bits 24-31... wait, that's bf2
        let bf2 = 0x02u8; // norm = 2
        let mut buf = build_dt_header(1, 1, [bf0, bf1, bf2], size);
        let mut props = [0u8; 12];
        LittleEndian::write_u16(&mut props[0..2], 0); // bit_offset
        LittleEndian::write_u16(&mut props[2..4], (size * 8) as u16); // bit_precision
        props[4] = exp_loc;
        props[5] = exp_size;
        props[6] = mant_loc;
        props[7] = mant_size;
        LittleEndian::write_u32(&mut props[8..12], exp_bias);
        buf.extend_from_slice(&props);
        buf
    }

    #[test]
    fn test_fixed_point_u8() {
        let data = build_fixed_point(1, false, false, 0, 8);
        let (dt, consumed) = Datatype::parse(&data).unwrap();
        assert_eq!(consumed, 12);
        assert_eq!(dt, Datatype::FixedPoint {
            size: 1,
            byte_order: DatatypeByteOrder::LittleEndian,
            signed: false,
            bit_offset: 0,
            bit_precision: 8,
        });
    }

    #[test]
    fn test_fixed_point_i16_le() {
        let data = build_fixed_point(2, false, true, 0, 16);
        let (dt, _) = Datatype::parse(&data).unwrap();
        assert_eq!(dt, Datatype::FixedPoint {
            size: 2,
            byte_order: DatatypeByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 16,
        });
    }

    #[test]
    fn test_fixed_point_u32_be() {
        let data = build_fixed_point(4, true, false, 0, 32);
        let (dt, _) = Datatype::parse(&data).unwrap();
        match &dt {
            Datatype::FixedPoint { byte_order, signed, size, .. } => {
                assert_eq!(*byte_order, DatatypeByteOrder::BigEndian);
                assert!(!signed);
                assert_eq!(*size, 4);
            }
            _ => panic!("expected FixedPoint"),
        }
    }

    #[test]
    fn test_fixed_point_i64_le() {
        let data = build_fixed_point(8, false, true, 0, 64);
        let (dt, _) = Datatype::parse(&data).unwrap();
        assert_eq!(dt, Datatype::FixedPoint {
            size: 8,
            byte_order: DatatypeByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 64,
        });
    }

    #[test]
    fn test_float_f32_le() {
        // IEEE 754 f32: exp=8 bits at bit 23, mant=23 bits at bit 0, bias=127
        let data = build_float(4, 23, 8, 0, 23, 127);
        let (dt, consumed) = Datatype::parse(&data).unwrap();
        assert_eq!(consumed, 20);
        assert_eq!(dt, Datatype::FloatingPoint {
            size: 4,
            byte_order: DatatypeByteOrder::LittleEndian,
            bit_offset: 0,
            bit_precision: 32,
            exponent_location: 23,
            exponent_size: 8,
            mantissa_location: 0,
            mantissa_size: 23,
            exponent_bias: 127,
        });
    }

    #[test]
    fn test_float_f64_le() {
        let data = build_float(8, 52, 11, 0, 52, 1023);
        let (dt, _) = Datatype::parse(&data).unwrap();
        assert_eq!(dt, Datatype::FloatingPoint {
            size: 8,
            byte_order: DatatypeByteOrder::LittleEndian,
            bit_offset: 0,
            bit_precision: 64,
            exponent_location: 52,
            exponent_size: 11,
            mantissa_location: 0,
            mantissa_size: 52,
            exponent_bias: 1023,
        });
    }

    #[test]
    fn test_string_null_terminated_ascii() {
        let buf = build_dt_header(3, 1, [0x00, 0, 0], 10); // padding=0(nullterm), charset=0(ascii)
        let (dt, consumed) = Datatype::parse(&buf).unwrap();
        assert_eq!(consumed, 8);
        assert_eq!(dt, Datatype::String {
            size: 10,
            padding: StringPadding::NullTerminate,
            charset: CharacterSet::Ascii,
        });
    }

    #[test]
    fn test_string_space_padded_utf8() {
        // padding=2(space pad), charset=1(utf8) → bf0 = 0x12
        let buf = build_dt_header(3, 1, [0x12, 0, 0], 32);
        let (dt, _) = Datatype::parse(&buf).unwrap();
        assert_eq!(dt, Datatype::String {
            size: 32,
            padding: StringPadding::SpacePad,
            charset: CharacterSet::Utf8,
        });
    }

    #[test]
    fn test_opaque() {
        // tag_len = 4, tag = "BLOB"
        let mut buf = build_dt_header(5, 1, [4, 0, 0], 64);
        buf.extend_from_slice(b"BLOB");
        // Pad to 8 bytes
        buf.extend_from_slice(&[0, 0, 0, 0]);
        let (dt, consumed) = Datatype::parse(&buf).unwrap();
        assert_eq!(consumed, 16); // 8 header + 8 padded tag
        assert_eq!(dt, Datatype::Opaque {
            size: 64,
            tag: b"BLOB".to_vec(),
        });
    }

    #[test]
    fn test_compound_v3_two_members() {
        // Compound with size=12, 2 members: "x" u32 at offset 0, "y" f64 at offset 4
        // Size=12, so offset_bytes=1
        let mut buf = build_dt_header(6, 3, [2, 0, 0], 12); // 2 members
        // Member "x": name "x\0", offset=0, then u32 LE datatype
        buf.extend_from_slice(b"x\0");
        buf.push(0); // byte_offset = 0
        buf.extend_from_slice(&build_fixed_point(4, false, false, 0, 32));
        // Member "y": name "y\0", offset=4, then f64 LE datatype
        buf.extend_from_slice(b"y\0");
        buf.push(4); // byte_offset = 4
        buf.extend_from_slice(&build_float(8, 52, 11, 0, 52, 1023));

        let (dt, _) = Datatype::parse(&buf).unwrap();
        match dt {
            Datatype::Compound { size, members } => {
                assert_eq!(size, 12);
                assert_eq!(members.len(), 2);
                assert_eq!(members[0].name, "x");
                assert_eq!(members[0].byte_offset, 0);
                assert_eq!(members[1].name, "y");
                assert_eq!(members[1].byte_offset, 4);
                match &members[0].datatype {
                    Datatype::FixedPoint { size: 4, signed: false, .. } => {}
                    other => panic!("expected u32, got {other:?}"),
                }
                match &members[1].datatype {
                    Datatype::FloatingPoint { size: 8, .. } => {}
                    other => panic!("expected f64, got {other:?}"),
                }
            }
            _ => panic!("expected Compound"),
        }
    }

    #[test]
    fn test_reference_object() {
        let buf = build_dt_header(7, 1, [0, 0, 0], 8);
        let (dt, _) = Datatype::parse(&buf).unwrap();
        assert_eq!(dt, Datatype::Reference {
            size: 8,
            ref_type: ReferenceType::Object,
        });
    }

    #[test]
    fn test_reference_region() {
        let buf = build_dt_header(7, 1, [1, 0, 0], 12);
        let (dt, _) = Datatype::parse(&buf).unwrap();
        assert_eq!(dt, Datatype::Reference {
            size: 12,
            ref_type: ReferenceType::DatasetRegion,
        });
    }

    #[test]
    fn test_enumeration() {
        // Enum with base type i32 LE, 3 members
        let mut buf = build_dt_header(8, 3, [3, 0, 0], 4); // 3 members
        // Base type: i32 LE
        buf.extend_from_slice(&build_fixed_point(4, false, true, 0, 32));
        // Names: "RED\0", "GREEN\0", "BLUE\0"
        buf.extend_from_slice(b"RED\0");
        buf.extend_from_slice(b"GREEN\0");
        buf.extend_from_slice(b"BLUE\0");
        // Values: 0, 1, 2 (as i32 LE)
        buf.extend_from_slice(&0i32.to_le_bytes());
        buf.extend_from_slice(&1i32.to_le_bytes());
        buf.extend_from_slice(&2i32.to_le_bytes());

        let (dt, _) = Datatype::parse(&buf).unwrap();
        match dt {
            Datatype::Enumeration { size, base_type, members } => {
                assert_eq!(size, 4);
                assert_eq!(members.len(), 3);
                assert_eq!(members[0].name, "RED");
                assert_eq!(members[0].value, 0i32.to_le_bytes().to_vec());
                assert_eq!(members[1].name, "GREEN");
                assert_eq!(members[1].value, 1i32.to_le_bytes().to_vec());
                assert_eq!(members[2].name, "BLUE");
                assert_eq!(members[2].value, 2i32.to_le_bytes().to_vec());
                match *base_type {
                    Datatype::FixedPoint { signed: true, size: 4, .. } => {}
                    other => panic!("expected i32, got {other:?}"),
                }
            }
            _ => panic!("expected Enumeration"),
        }
    }

    #[test]
    fn test_variable_length_string_utf8() {
        // VL string: type=1, padding=0(null term), charset=1(utf8)
        // bf0: bits 0-3 = 1 (string), bits 4-7 = 0 (null term) → 0x01
        // bf1: bits 0-3 = 1 (utf8) → 0x01
        let mut buf = build_dt_header(9, 1, [0x01, 0x01, 0], 16);
        // Base type: u8 (class 0, unsigned, size 1)
        buf.extend_from_slice(&build_fixed_point(1, false, false, 0, 8));

        let (dt, _) = Datatype::parse(&buf).unwrap();
        match dt {
            Datatype::VariableLength { is_string, padding, charset, base_type } => {
                assert!(is_string);
                assert_eq!(padding, Some(StringPadding::NullTerminate));
                assert_eq!(charset, Some(CharacterSet::Utf8));
                assert_eq!(base_type.type_size(), 1);
            }
            _ => panic!("expected VariableLength"),
        }
    }

    #[test]
    fn test_variable_length_sequence_f32() {
        // VL sequence: type=0
        // bf0 = 0x00
        let mut buf = build_dt_header(9, 1, [0x00, 0x00, 0], 16);
        // Base type: f32 LE
        buf.extend_from_slice(&build_float(4, 23, 8, 0, 23, 127));

        let (dt, _) = Datatype::parse(&buf).unwrap();
        match dt {
            Datatype::VariableLength { is_string, padding, charset, base_type } => {
                assert!(!is_string);
                assert_eq!(padding, None);
                assert_eq!(charset, None);
                assert_eq!(base_type.type_size(), 4);
            }
            _ => panic!("expected VariableLength"),
        }
    }

    #[test]
    fn test_array_2d() {
        // Array [3][4] of i32 LE, version 3
        let mut buf = build_dt_header(10, 3, [0, 0, 0], 48); // 3*4*4=48
        buf.push(2); // ndims=2
        buf.extend_from_slice(&3u32.to_le_bytes()); // dim 0
        buf.extend_from_slice(&4u32.to_le_bytes()); // dim 1
        // Base type: i32 LE
        buf.extend_from_slice(&build_fixed_point(4, false, true, 0, 32));

        let (dt, _) = Datatype::parse(&buf).unwrap();
        match dt {
            Datatype::Array { base_type, dimensions } => {
                assert_eq!(dimensions, vec![3, 4]);
                match *base_type {
                    Datatype::FixedPoint { size: 4, signed: true, .. } => {}
                    other => panic!("expected i32, got {other:?}"),
                }
            }
            _ => panic!("expected Array"),
        }
    }

    #[test]
    fn test_bitfield() {
        let mut buf = build_dt_header(4, 1, [0, 0, 0], 2); // 16-bit LE bitfield
        let mut props = [0u8; 4];
        LittleEndian::write_u16(&mut props[0..2], 0);
        LittleEndian::write_u16(&mut props[2..4], 16);
        buf.extend_from_slice(&props);

        let (dt, _) = Datatype::parse(&buf).unwrap();
        assert_eq!(dt, Datatype::BitField {
            size: 2,
            byte_order: DatatypeByteOrder::LittleEndian,
            bit_offset: 0,
            bit_precision: 16,
        });
    }

    #[test]
    fn test_time() {
        let mut buf = build_dt_header(2, 1, [0, 0, 0], 8);
        let mut props = [0u8; 2];
        LittleEndian::write_u16(&mut props[0..2], 64);
        buf.extend_from_slice(&props);

        let (dt, consumed) = Datatype::parse(&buf).unwrap();
        assert_eq!(consumed, 10);
        assert_eq!(dt, Datatype::Time {
            size: 8,
            bit_precision: 64,
        });
    }

    #[test]
    fn test_nested_compound_array_enum() {
        // Compound containing a single member "data" which is an Array[2] of Enum(i32, 2 values)
        // Build the enum first
        let mut enum_bytes = build_dt_header(8, 3, [2, 0, 0], 4); // 2 members
        enum_bytes.extend_from_slice(&build_fixed_point(4, false, true, 0, 32)); // base i32
        enum_bytes.extend_from_slice(b"A\0");
        enum_bytes.extend_from_slice(b"B\0");
        enum_bytes.extend_from_slice(&0i32.to_le_bytes());
        enum_bytes.extend_from_slice(&1i32.to_le_bytes());

        // Build array[2] of that enum, version 3
        let mut array_bytes = build_dt_header(10, 3, [0, 0, 0], 8); // 2*4=8
        array_bytes.push(1); // ndims=1
        array_bytes.extend_from_slice(&2u32.to_le_bytes()); // dim[0]=2
        array_bytes.extend_from_slice(&enum_bytes);

        // Build compound with 1 member, size=8
        let mut buf = build_dt_header(6, 3, [1, 0, 0], 8); // 1 member
        buf.extend_from_slice(b"data\0");
        buf.push(0); // byte_offset = 0 (size=8, so 1 byte offsets)
        buf.extend_from_slice(&array_bytes);

        let (dt, _) = Datatype::parse(&buf).unwrap();
        match dt {
            Datatype::Compound { members, .. } => {
                assert_eq!(members.len(), 1);
                assert_eq!(members[0].name, "data");
                match &members[0].datatype {
                    Datatype::Array { dimensions, base_type } => {
                        assert_eq!(dimensions, &[2]);
                        match base_type.as_ref() {
                            Datatype::Enumeration { members, .. } => {
                                assert_eq!(members.len(), 2);
                                assert_eq!(members[0].name, "A");
                                assert_eq!(members[1].name, "B");
                            }
                            other => panic!("expected Enum, got {other:?}"),
                        }
                    }
                    other => panic!("expected Array, got {other:?}"),
                }
            }
            _ => panic!("expected Compound"),
        }
    }

    #[test]
    fn test_error_invalid_class() {
        let buf = build_dt_header(13, 1, [0, 0, 0], 4);
        let err = Datatype::parse(&buf).unwrap_err();
        assert_eq!(err, FormatError::InvalidDatatypeClass(13));
    }

    #[test]
    fn test_error_truncated_data() {
        let buf = [0u8; 4]; // too short for header
        let err = Datatype::parse(&buf).unwrap_err();
        match err {
            FormatError::UnexpectedEof { .. } => {}
            other => panic!("expected UnexpectedEof, got {other:?}"),
        }
    }

    #[test]
    fn test_error_invalid_string_padding() {
        let buf = build_dt_header(3, 1, [0x03, 0, 0], 10); // padding=3 invalid
        let err = Datatype::parse(&buf).unwrap_err();
        assert_eq!(err, FormatError::InvalidStringPadding(3));
    }

    #[test]
    fn test_error_invalid_charset() {
        let buf = build_dt_header(3, 1, [0x20, 0, 0], 10); // charset=2 invalid
        let err = Datatype::parse(&buf).unwrap_err();
        assert_eq!(err, FormatError::InvalidCharacterSet(2));
    }

    #[test]
    fn test_error_invalid_reference_type() {
        let buf = build_dt_header(7, 1, [5, 0, 0], 8);
        let err = Datatype::parse(&buf).unwrap_err();
        assert_eq!(err, FormatError::InvalidReferenceType(5));
    }

    #[test]
    fn serialize_parse_compound_roundtrip() {
        let dt = Datatype::Compound {
            size: 20,
            members: vec![
                CompoundMember {
                    name: "x".to_string(),
                    byte_offset: 0,
                    datatype: Datatype::FloatingPoint {
                        size: 8, byte_order: DatatypeByteOrder::LittleEndian,
                        bit_offset: 0, bit_precision: 64,
                        exponent_location: 52, exponent_size: 11,
                        mantissa_location: 0, mantissa_size: 52, exponent_bias: 1023,
                    },
                },
                CompoundMember {
                    name: "y".to_string(),
                    byte_offset: 8,
                    datatype: Datatype::FloatingPoint {
                        size: 8, byte_order: DatatypeByteOrder::LittleEndian,
                        bit_offset: 0, bit_precision: 64,
                        exponent_location: 52, exponent_size: 11,
                        mantissa_location: 0, mantissa_size: 52, exponent_bias: 1023,
                    },
                },
                CompoundMember {
                    name: "id".to_string(),
                    byte_offset: 16,
                    datatype: Datatype::FixedPoint {
                        size: 4, byte_order: DatatypeByteOrder::LittleEndian,
                        signed: true, bit_offset: 0, bit_precision: 32,
                    },
                },
            ],
        };
        let bytes = dt.serialize();
        let (parsed, _) = Datatype::parse(&bytes).unwrap();
        assert_eq!(parsed, dt);
    }

    #[test]
    fn serialize_parse_enum_roundtrip() {
        let dt = Datatype::Enumeration {
            size: 4,
            base_type: Box::new(Datatype::FixedPoint {
                size: 4, byte_order: DatatypeByteOrder::LittleEndian,
                signed: true, bit_offset: 0, bit_precision: 32,
            }),
            members: vec![
                EnumMember { name: "RED".to_string(), value: 0i32.to_le_bytes().to_vec() },
                EnumMember { name: "GREEN".to_string(), value: 1i32.to_le_bytes().to_vec() },
                EnumMember { name: "BLUE".to_string(), value: 2i32.to_le_bytes().to_vec() },
            ],
        };
        let bytes = dt.serialize();
        let (parsed, _) = Datatype::parse(&bytes).unwrap();
        assert_eq!(parsed, dt);
    }

    #[test]
    fn serialize_parse_array_roundtrip() {
        let dt = Datatype::Array {
            base_type: Box::new(Datatype::FloatingPoint {
                size: 8, byte_order: DatatypeByteOrder::LittleEndian,
                bit_offset: 0, bit_precision: 64,
                exponent_location: 52, exponent_size: 11,
                mantissa_location: 0, mantissa_size: 52, exponent_bias: 1023,
            }),
            dimensions: vec![3],
        };
        let bytes = dt.serialize();
        let (parsed, _) = Datatype::parse(&bytes).unwrap();
        assert_eq!(parsed, dt);
    }

    #[test]
    fn test_type_size() {
        let dt = Datatype::FixedPoint {
            size: 4, byte_order: DatatypeByteOrder::LittleEndian,
            signed: true, bit_offset: 0, bit_precision: 32,
        };
        assert_eq!(dt.type_size(), 4);

        let dt = Datatype::Array {
            base_type: Box::new(Datatype::FixedPoint {
                size: 4, byte_order: DatatypeByteOrder::LittleEndian,
                signed: true, bit_offset: 0, bit_precision: 32,
            }),
            dimensions: vec![3, 4],
        };
        assert_eq!(dt.type_size(), 48);
    }
}
