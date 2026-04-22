//! Builder types for HDF5 datatypes, attributes, datasets, and groups.
//!
//! Extracted from `file_writer.rs` to keep modules under the line limit.

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, string::ToString, vec, vec::Vec};

use crate::attribute::AttributeMessage;
use crate::chunked_write::ChunkOptions;
use crate::dataspace::{Dataspace, DataspaceType};
use crate::datatype::{
    CharacterSet, CompoundMember, Datatype, DatatypeByteOrder, EnumMember, StringPadding,
};

// ---- Datatype constructors ----

pub fn make_f64_type() -> Datatype {
    Datatype::FloatingPoint {
        size: 8,
        byte_order: DatatypeByteOrder::LittleEndian,
        bit_offset: 0,
        bit_precision: 64,
        exponent_location: 52,
        exponent_size: 11,
        mantissa_location: 0,
        mantissa_size: 52,
        exponent_bias: 1023,
    }
}

pub fn make_f32_type() -> Datatype {
    Datatype::FloatingPoint {
        size: 4,
        byte_order: DatatypeByteOrder::LittleEndian,
        bit_offset: 0,
        bit_precision: 32,
        exponent_location: 23,
        exponent_size: 8,
        mantissa_location: 0,
        mantissa_size: 23,
        exponent_bias: 127,
    }
}

pub fn make_i32_type() -> Datatype {
    Datatype::FixedPoint {
        size: 4,
        byte_order: DatatypeByteOrder::LittleEndian,
        signed: true,
        bit_offset: 0,
        bit_precision: 32,
    }
}

pub fn make_i64_type() -> Datatype {
    Datatype::FixedPoint {
        size: 8,
        byte_order: DatatypeByteOrder::LittleEndian,
        signed: true,
        bit_offset: 0,
        bit_precision: 64,
    }
}

pub fn make_u8_type() -> Datatype {
    Datatype::FixedPoint {
        size: 1,
        byte_order: DatatypeByteOrder::LittleEndian,
        signed: false,
        bit_offset: 0,
        bit_precision: 8,
    }
}

pub fn make_i8_type() -> Datatype {
    Datatype::FixedPoint {
        size: 1,
        byte_order: DatatypeByteOrder::LittleEndian,
        signed: true,
        bit_offset: 0,
        bit_precision: 8,
    }
}

pub fn make_i16_type() -> Datatype {
    Datatype::FixedPoint {
        size: 2,
        byte_order: DatatypeByteOrder::LittleEndian,
        signed: true,
        bit_offset: 0,
        bit_precision: 16,
    }
}

pub fn make_u16_type() -> Datatype {
    Datatype::FixedPoint {
        size: 2,
        byte_order: DatatypeByteOrder::LittleEndian,
        signed: false,
        bit_offset: 0,
        bit_precision: 16,
    }
}

pub fn make_u32_type() -> Datatype {
    Datatype::FixedPoint {
        size: 4,
        byte_order: DatatypeByteOrder::LittleEndian,
        signed: false,
        bit_offset: 0,
        bit_precision: 32,
    }
}

pub fn make_u64_type() -> Datatype {
    Datatype::FixedPoint {
        size: 8,
        byte_order: DatatypeByteOrder::LittleEndian,
        signed: false,
        bit_offset: 0,
        bit_precision: 64,
    }
}

pub fn make_object_reference_type() -> Datatype {
    Datatype::Reference {
        size: 8,
        ref_type: crate::datatype::ReferenceType::Object,
    }
}

// ---- Compound / Enum type builders ----

/// Builder for constructing HDF5 compound (struct) datatypes.
pub struct CompoundTypeBuilder {
    fields: Vec<(String, Datatype)>,
}

impl CompoundTypeBuilder {
    pub fn new() -> Self {
        Self { fields: Vec::new() }
    }

    /// Add a named field with the given datatype.
    pub fn field(mut self, name: &str, datatype: Datatype) -> Self {
        self.fields.push((name.to_string(), datatype));
        self
    }

    /// Add an f64 field.
    pub fn f64_field(self, name: &str) -> Self {
        self.field(name, make_f64_type())
    }
    /// Add an f32 field.
    pub fn f32_field(self, name: &str) -> Self {
        self.field(name, make_f32_type())
    }
    /// Add an i32 field.
    pub fn i32_field(self, name: &str) -> Self {
        self.field(name, make_i32_type())
    }
    /// Add an i64 field.
    pub fn i64_field(self, name: &str) -> Self {
        self.field(name, make_i64_type())
    }
    /// Add a u8 field.
    pub fn u8_field(self, name: &str) -> Self {
        self.field(name, make_u8_type())
    }
    /// Add an i8 field.
    pub fn i8_field(self, name: &str) -> Self {
        self.field(name, make_i8_type())
    }
    /// Add an i16 field.
    pub fn i16_field(self, name: &str) -> Self {
        self.field(name, make_i16_type())
    }
    /// Add a u16 field.
    pub fn u16_field(self, name: &str) -> Self {
        self.field(name, make_u16_type())
    }
    /// Add a u32 field.
    pub fn u32_field(self, name: &str) -> Self {
        self.field(name, make_u32_type())
    }
    /// Add a u64 field.
    pub fn u64_field(self, name: &str) -> Self {
        self.field(name, make_u64_type())
    }

    /// Build the compound datatype.
    pub fn build(self) -> Datatype {
        let mut offset = 0u64;
        let mut members = Vec::with_capacity(self.fields.len());
        for (name, dt) in self.fields {
            let sz = dt.type_size();
            members.push(CompoundMember {
                name,
                byte_offset: offset,
                datatype: dt,
            });
            offset += sz as u64;
        }
        Datatype::Compound {
            size: offset as u32,
            members,
        }
    }
}

impl Default for CompoundTypeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for constructing HDF5 enumeration datatypes.
pub struct EnumTypeBuilder {
    base_type: Datatype,
    members: Vec<EnumMember>,
}

impl EnumTypeBuilder {
    /// Create a new enum builder with i32 base type.
    pub fn i32_based() -> Self {
        Self {
            base_type: make_i32_type(),
            members: Vec::new(),
        }
    }

    /// Create a new enum builder with u8 base type.
    pub fn u8_based() -> Self {
        Self {
            base_type: make_u8_type(),
            members: Vec::new(),
        }
    }

    /// Add a named value.
    pub fn value(mut self, name: &str, val: i32) -> Self {
        self.members.push(EnumMember {
            name: name.to_string(),
            value: val.to_le_bytes().to_vec(),
        });
        self
    }

    /// Add a named u8 value.
    pub fn u8_value(mut self, name: &str, val: u8) -> Self {
        self.members.push(EnumMember {
            name: name.to_string(),
            value: vec![val],
        });
        self
    }

    /// Build the enumeration datatype.
    pub fn build(self) -> Datatype {
        let size = self.base_type.type_size();
        Datatype::Enumeration {
            size,
            base_type: Box::new(self.base_type),
            members: self.members,
        }
    }
}

// ---- Attribute helper ----

pub(crate) fn build_attr_message(name: &str, value: &AttrValue) -> AttributeMessage {
    match value {
        AttrValue::F64(v) => AttributeMessage {
            name: name.to_string(),
            datatype: make_f64_type(),
            dataspace: scalar_ds(),
            raw_data: v.to_le_bytes().to_vec(),
        },
        AttrValue::F64Array(arr) => {
            let mut raw = Vec::with_capacity(arr.len() * 8);
            for v in arr {
                raw.extend_from_slice(&v.to_le_bytes());
            }
            AttributeMessage {
                name: name.to_string(),
                datatype: make_f64_type(),
                dataspace: simple_1d(arr.len() as u64),
                raw_data: raw,
            }
        }
        AttrValue::I64(v) => AttributeMessage {
            name: name.to_string(),
            datatype: make_i64_type(),
            dataspace: scalar_ds(),
            raw_data: v.to_le_bytes().to_vec(),
        },
        AttrValue::I64Array(arr) => {
            let mut raw = Vec::with_capacity(arr.len() * 8);
            for v in arr {
                raw.extend_from_slice(&v.to_le_bytes());
            }
            AttributeMessage {
                name: name.to_string(),
                datatype: make_i64_type(),
                dataspace: simple_1d(arr.len() as u64),
                raw_data: raw,
            }
        }
        AttrValue::I32(v) => AttributeMessage {
            name: name.to_string(),
            datatype: make_i32_type(),
            dataspace: scalar_ds(),
            raw_data: v.to_le_bytes().to_vec(),
        },
        AttrValue::U32(v) => AttributeMessage {
            name: name.to_string(),
            datatype: make_u32_type(),
            dataspace: scalar_ds(),
            raw_data: v.to_le_bytes().to_vec(),
        },
        AttrValue::U64(v) => AttributeMessage {
            name: name.to_string(),
            datatype: make_u64_type(),
            dataspace: scalar_ds(),
            raw_data: v.to_le_bytes().to_vec(),
        },
        AttrValue::String(s) => {
            let bytes = s.as_bytes();
            AttributeMessage {
                name: name.to_string(),
                datatype: Datatype::String {
                    size: bytes.len() as u32,
                    padding: StringPadding::NullPad,
                    charset: CharacterSet::Utf8,
                },
                dataspace: scalar_ds(),
                raw_data: bytes.to_vec(),
            }
        }
        AttrValue::StringArray(arr) => {
            let max_len = arr.iter().map(|s| s.len()).max().unwrap_or(0);
            let mut raw = Vec::new();
            for s in arr {
                let mut b = s.as_bytes().to_vec();
                b.resize(max_len, 0);
                raw.extend_from_slice(&b);
            }
            AttributeMessage {
                name: name.to_string(),
                datatype: Datatype::String {
                    size: max_len as u32,
                    padding: StringPadding::NullPad,
                    charset: CharacterSet::Utf8,
                },
                dataspace: simple_1d(arr.len() as u64),
                raw_data: raw,
            }
        }
        AttrValue::AsciiString(s) => {
            let bytes = s.as_bytes();
            AttributeMessage {
                name: name.to_string(),
                datatype: Datatype::String {
                    size: bytes.len() as u32,
                    padding: StringPadding::NullPad,
                    charset: CharacterSet::Ascii,
                },
                dataspace: scalar_ds(),
                raw_data: bytes.to_vec(),
            }
        }
        AttrValue::AsciiStringArray(arr) => {
            let max_len = arr.iter().map(|s| s.len()).max().unwrap_or(0);
            let mut raw = Vec::new();
            for s in arr {
                let mut b = s.as_bytes().to_vec();
                b.resize(max_len, 0);
                raw.extend_from_slice(&b);
            }
            AttributeMessage {
                name: name.to_string(),
                datatype: Datatype::String {
                    size: max_len as u32,
                    padding: StringPadding::NullPad,
                    charset: CharacterSet::Ascii,
                },
                dataspace: simple_1d(arr.len() as u64),
                raw_data: raw,
            }
        }
        AttrValue::VarLenAsciiArray(strings) => {
            // MATLAB v7.3 (and matio) expect MATLAB_fields and similar
            // variable-length ASCII arrays encoded as:
            //   H5T_VLEN { H5T_STRING { STRSIZE=1, NULLTERM, ASCII } }
            // — a VLEN sequence of 1-byte fixed strings. The on-disk byte
            // layout is identical to H5T_STRING{STRSIZE=VAR} (length + heap
            // address + object index per element; heap object holds raw
            // bytes without null terminator), so only the datatype
            // descriptor changes.
            let vl_ref_size = 16usize; // 4 + 8 + 4 for offset_size=8
            let mut raw = Vec::with_capacity(strings.len() * vl_ref_size);
            for (i, s) in strings.iter().enumerate() {
                raw.extend_from_slice(&(s.len() as u32).to_le_bytes());
                raw.extend_from_slice(&0u64.to_le_bytes()); // patched later
                raw.extend_from_slice(&((i + 1) as u32).to_le_bytes());
            }
            AttributeMessage {
                name: name.to_string(),
                datatype: Datatype::VariableLength {
                    is_string: false,
                    padding: None,
                    charset: None,
                    base_type: Box::new(Datatype::String {
                        size: 1,
                        padding: StringPadding::NullTerminate,
                        charset: CharacterSet::Ascii,
                    }),
                },
                dataspace: simple_1d(strings.len() as u64),
                raw_data: raw,
            }
        }
    }
}

/// Build a global heap collection containing the given byte sequences.
/// Returns the serialized collection bytes.
pub(crate) fn build_global_heap_collection(strings: &[&str]) -> Vec<u8> {
    let length_size = 8usize;
    let header_size = 8 + length_size; // sig(4) + ver(1) + reserved(3) + collection_size

    // Calculate total size
    let mut obj_size_total = 0usize;
    for s in strings {
        let obj_header = 8 + length_size; // index(2) + refcount(2) + reserved(4) + size
        let padded_data_len = (s.len() + 7) & !7; // pad to 8 bytes
        obj_size_total += obj_header + padded_data_len;
    }
    obj_size_total += 8 + length_size; // free space marker (full object header size)
    let collection_size = header_size + obj_size_total;
    // The C HDF5 library enforces a minimum collection size of 4096 bytes.
    let min_collection_size = 4096;
    let padded_collection = ((collection_size.max(min_collection_size)) + 7) & !7;

    let mut buf = Vec::with_capacity(padded_collection);
    // Header
    buf.extend_from_slice(b"GCOL");
    buf.push(1); // version
    buf.extend_from_slice(&[0u8; 3]); // reserved
    buf.extend_from_slice(&(padded_collection as u64).to_le_bytes());

    // Objects (1-based indices)
    for (i, s) in strings.iter().enumerate() {
        let index = (i + 1) as u16;
        buf.extend_from_slice(&index.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes()); // ref_count
        buf.extend_from_slice(&[0u8; 4]); // reserved
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
        // Pad to 8-byte boundary
        let padded = (s.len() + 7) & !7;
        for _ in s.len()..padded {
            buf.push(0);
        }
    }

    // Free space marker (index 0): the C library uses this size as the total
    // skip distance from the start of the object (including its header), so
    // it must equal the remaining bytes in the collection from this point.
    let free_total_size = padded_collection - buf.len();
    buf.extend_from_slice(&0u16.to_le_bytes()); // index 0
    buf.extend_from_slice(&0u16.to_le_bytes()); // ref_count
    buf.extend_from_slice(&[0u8; 4]); // reserved
    buf.extend_from_slice(&(free_total_size as u64).to_le_bytes()); // size

    // Pad collection to full size
    buf.resize(padded_collection, 0);

    buf
}

/// Patch VL attribute references with the actual global heap collection address.
/// The raw_data contains VL references with placeholder addresses (0).
pub(crate) fn patch_vl_refs(raw_data: &mut [u8], collection_address: u64) {
    let vl_ref_size = 16; // 4 + 8 + 4
    let count = raw_data.len() / vl_ref_size;
    for i in 0..count {
        let addr_offset = i * vl_ref_size + 4; // skip sequence_length
        raw_data[addr_offset..addr_offset + 8].copy_from_slice(&collection_address.to_le_bytes());
    }
}

pub(crate) fn scalar_ds() -> Dataspace {
    Dataspace {
        space_type: DataspaceType::Scalar,
        rank: 0,
        dimensions: vec![],
        max_dimensions: None,
    }
}

pub(crate) fn simple_1d(n: u64) -> Dataspace {
    Dataspace {
        space_type: DataspaceType::Simple,
        rank: 1,
        dimensions: vec![n],
        max_dimensions: None,
    }
}

// ---- Attribute values ----

/// Convenient attribute values for the write API.
#[derive(Debug, Clone, PartialEq)]
pub enum AttrValue {
    F64(f64),
    F64Array(Vec<f64>),
    I32(i32),
    I64(i64),
    I64Array(Vec<i64>),
    U32(u32),
    U64(u64),
    /// UTF-8 string attribute (null-padded).
    String(String),
    StringArray(Vec<String>),
    /// Fixed-width ASCII string attribute (charset = ASCII).
    AsciiString(String),
    /// Array of fixed-width ASCII strings (null-padded to the longest element).
    /// Compatible with MATLAB `MATLAB_fields` and matio.
    AsciiStringArray(Vec<String>),
    /// Array of variable-length ASCII strings (MATLAB_fields pattern).
    /// Each element is a variable-length sequence of ASCII bytes.
    /// Requires a global heap collection in the file.
    VarLenAsciiArray(Vec<String>),
}

// ---- Dataset builder ----

/// Configuration for SHINES provenance metadata.
#[cfg(feature = "provenance")]
#[derive(Debug, Clone)]
pub struct ProvenanceConfig {
    pub creator: String,
    pub timestamp: String,
    pub source: Option<String>,
}

/// Builder for datasets.
pub struct DatasetBuilder {
    pub(crate) name: String,
    pub(crate) datatype: Option<Datatype>,
    pub(crate) shape: Option<Vec<u64>>,
    pub(crate) maxshape: Option<Vec<u64>>,
    pub(crate) data: Option<Vec<u8>>,
    pub(crate) attrs: Vec<(String, AttrValue)>,
    pub(crate) chunk_options: ChunkOptions,
    /// When set, this dataset contains object references that should be
    /// resolved by path during file serialization.
    pub(crate) reference_targets: Option<Vec<String>>,
    #[cfg(feature = "provenance")]
    pub(crate) provenance: Option<ProvenanceConfig>,
}

impl DatasetBuilder {
    pub(crate) fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            datatype: None,
            shape: None,
            maxshape: None,
            data: None,
            attrs: Vec::new(),
            chunk_options: ChunkOptions::default(),
            reference_targets: None,
            #[cfg(feature = "provenance")]
            provenance: None,
        }
    }

    pub fn with_f64_data(&mut self, data: &[f64]) -> &mut Self {
        self.datatype = Some(make_f64_type());
        let mut b = Vec::with_capacity(data.len() * 8);
        for &v in data {
            b.extend_from_slice(&v.to_le_bytes());
        }
        self.data = Some(b);
        if self.shape.is_none() {
            self.shape = Some(vec![data.len() as u64]);
        }
        self
    }

    pub fn with_f32_data(&mut self, data: &[f32]) -> &mut Self {
        self.datatype = Some(make_f32_type());
        let mut b = Vec::with_capacity(data.len() * 4);
        for &v in data {
            b.extend_from_slice(&v.to_le_bytes());
        }
        self.data = Some(b);
        if self.shape.is_none() {
            self.shape = Some(vec![data.len() as u64]);
        }
        self
    }

    pub fn with_i32_data(&mut self, data: &[i32]) -> &mut Self {
        self.datatype = Some(make_i32_type());
        let mut b = Vec::with_capacity(data.len() * 4);
        for &v in data {
            b.extend_from_slice(&v.to_le_bytes());
        }
        self.data = Some(b);
        if self.shape.is_none() {
            self.shape = Some(vec![data.len() as u64]);
        }
        self
    }

    pub fn with_i64_data(&mut self, data: &[i64]) -> &mut Self {
        self.datatype = Some(make_i64_type());
        let mut b = Vec::with_capacity(data.len() * 8);
        for &v in data {
            b.extend_from_slice(&v.to_le_bytes());
        }
        self.data = Some(b);
        if self.shape.is_none() {
            self.shape = Some(vec![data.len() as u64]);
        }
        self
    }

    pub fn with_u8_data(&mut self, data: &[u8]) -> &mut Self {
        self.datatype = Some(make_u8_type());
        self.data = Some(data.to_vec());
        if self.shape.is_none() {
            self.shape = Some(vec![data.len() as u64]);
        }
        self
    }

    pub fn with_i8_data(&mut self, data: &[i8]) -> &mut Self {
        self.datatype = Some(make_i8_type());
        let mut b = Vec::with_capacity(data.len());
        for &v in data {
            b.push(v as u8);
        }
        self.data = Some(b);
        if self.shape.is_none() {
            self.shape = Some(vec![data.len() as u64]);
        }
        self
    }

    pub fn with_i16_data(&mut self, data: &[i16]) -> &mut Self {
        self.datatype = Some(make_i16_type());
        let mut b = Vec::with_capacity(data.len() * 2);
        for &v in data {
            b.extend_from_slice(&v.to_le_bytes());
        }
        self.data = Some(b);
        if self.shape.is_none() {
            self.shape = Some(vec![data.len() as u64]);
        }
        self
    }

    pub fn with_u16_data(&mut self, data: &[u16]) -> &mut Self {
        self.datatype = Some(make_u16_type());
        let mut b = Vec::with_capacity(data.len() * 2);
        for &v in data {
            b.extend_from_slice(&v.to_le_bytes());
        }
        self.data = Some(b);
        if self.shape.is_none() {
            self.shape = Some(vec![data.len() as u64]);
        }
        self
    }

    pub fn with_u32_data(&mut self, data: &[u32]) -> &mut Self {
        self.datatype = Some(make_u32_type());
        let mut b = Vec::with_capacity(data.len() * 4);
        for &v in data {
            b.extend_from_slice(&v.to_le_bytes());
        }
        self.data = Some(b);
        if self.shape.is_none() {
            self.shape = Some(vec![data.len() as u64]);
        }
        self
    }

    pub fn with_u64_data(&mut self, data: &[u64]) -> &mut Self {
        self.datatype = Some(make_u64_type());
        let mut b = Vec::with_capacity(data.len() * 8);
        for &v in data {
            b.extend_from_slice(&v.to_le_bytes());
        }
        self.data = Some(b);
        if self.shape.is_none() {
            self.shape = Some(vec![data.len() as u64]);
        }
        self
    }

    /// Write an object reference dataset. Each address is an 8-byte absolute
    /// address pointing to an object header in the file.
    pub fn with_reference_data(&mut self, addresses: &[u64]) -> &mut Self {
        self.datatype = Some(make_object_reference_type());
        let mut b = Vec::with_capacity(addresses.len() * 8);
        for &addr in addresses {
            b.extend_from_slice(&addr.to_le_bytes());
        }
        self.data = Some(b);
        if self.shape.is_none() {
            self.shape = Some(vec![addresses.len() as u64]);
        }
        self
    }

    /// Write an object reference dataset by path. During file serialization,
    /// each path is resolved to the absolute address of the named object.
    /// Paths use `/` separators (e.g., `"#refs#/child1"`).
    pub fn with_path_references(&mut self, paths: &[&str]) -> &mut Self {
        self.datatype = Some(make_object_reference_type());
        // Placeholder zeros — will be patched during finish()
        self.data = Some(vec![0u8; paths.len() * 8]);
        self.reference_targets = Some(paths.iter().map(|s| s.to_string()).collect());
        if self.shape.is_none() {
            self.shape = Some(vec![paths.len() as u64]);
        }
        self
    }

    /// Write a complex32 (f32 real/imag pair) dataset.
    pub fn with_complex32_data(&mut self, data: &[(f32, f32)]) -> &mut Self {
        let ct = CompoundTypeBuilder::new()
            .f32_field("real")
            .f32_field("imag")
            .build();
        let mut raw = Vec::with_capacity(data.len() * 8);
        for &(r, i) in data {
            raw.extend_from_slice(&r.to_le_bytes());
            raw.extend_from_slice(&i.to_le_bytes());
        }
        self.with_compound_data(ct, raw, data.len() as u64)
    }

    /// Write a complex64 (f64 real/imag pair) dataset.
    pub fn with_complex64_data(&mut self, data: &[(f64, f64)]) -> &mut Self {
        let ct = CompoundTypeBuilder::new()
            .f64_field("real")
            .f64_field("imag")
            .build();
        let mut raw = Vec::with_capacity(data.len() * 16);
        for &(r, i) in data {
            raw.extend_from_slice(&r.to_le_bytes());
            raw.extend_from_slice(&i.to_le_bytes());
        }
        self.with_compound_data(ct, raw, data.len() as u64)
    }

    /// Write a compound (struct) dataset.
    pub fn with_compound_data(
        &mut self,
        datatype: Datatype,
        raw_data: Vec<u8>,
        num_elements: u64,
    ) -> &mut Self {
        self.datatype = Some(datatype);
        self.data = Some(raw_data);
        if self.shape.is_none() {
            self.shape = Some(vec![num_elements]);
        }
        self
    }

    /// Write an enum dataset with i32 values.
    pub fn with_enum_i32_data(&mut self, datatype: Datatype, values: &[i32]) -> &mut Self {
        self.datatype = Some(datatype);
        let mut raw = Vec::with_capacity(values.len() * 4);
        for &v in values {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        self.data = Some(raw);
        if self.shape.is_none() {
            self.shape = Some(vec![values.len() as u64]);
        }
        self
    }

    /// Write an enum dataset with u8 values.
    pub fn with_enum_u8_data(&mut self, datatype: Datatype, values: &[u8]) -> &mut Self {
        self.datatype = Some(datatype);
        self.data = Some(values.to_vec());
        if self.shape.is_none() {
            self.shape = Some(vec![values.len() as u64]);
        }
        self
    }

    /// Write an array-typed dataset.
    pub fn with_array_data(
        &mut self,
        base_type: Datatype,
        array_dims: &[u32],
        raw_data: Vec<u8>,
        num_elements: u64,
    ) -> &mut Self {
        self.datatype = Some(Datatype::Array {
            base_type: Box::new(base_type),
            dimensions: array_dims.to_vec(),
        });
        self.data = Some(raw_data);
        if self.shape.is_none() {
            self.shape = Some(vec![num_elements]);
        }
        self
    }

    pub fn with_shape(&mut self, shape: &[u64]) -> &mut Self {
        self.shape = Some(shape.to_vec());
        self
    }

    /// Set the datatype without providing data.
    /// Use with `with_shape` for empty/zero-dimension datasets.
    pub fn with_dtype(&mut self, dt: Datatype) -> &mut Self {
        self.datatype = Some(dt);
        self
    }

    /// Set maximum dimensions for a resizable dataset.
    /// Use `u64::MAX` for unlimited dimensions.
    pub fn with_maxshape(&mut self, maxshape: &[u64]) -> &mut Self {
        self.maxshape = Some(maxshape.to_vec());
        self
    }

    pub fn set_attr(&mut self, name: &str, value: AttrValue) -> &mut Self {
        self.attrs.push((name.to_string(), value));
        self
    }

    /// Enable chunked storage with given chunk dimensions.
    pub fn with_chunks(&mut self, chunk_dims: &[u64]) -> &mut Self {
        self.chunk_options.chunk_dims = Some(chunk_dims.to_vec());
        self
    }

    /// Enable deflate compression (implies chunked if not already set).
    pub fn with_deflate(&mut self, level: u32) -> &mut Self {
        self.chunk_options.deflate_level = Some(level);
        self
    }

    /// Enable shuffle filter (usually combined with deflate).
    pub fn with_shuffle(&mut self) -> &mut Self {
        self.chunk_options.shuffle = true;
        self
    }

    /// Enable fletcher32 checksum.
    pub fn with_fletcher32(&mut self) -> &mut Self {
        self.chunk_options.fletcher32 = true;
        self
    }

    /// Enable ZFP fixed-rate compression (implies chunked if not already set).
    ///
    /// `rate` is the number of compressed bits per value (e.g. 16.0 gives
    /// 2:1 compression for f32 data). Only valid for f32 datasets.
    pub fn with_zfp(&mut self, rate: f64) -> &mut Self {
        self.chunk_options.zfp_rate = Some(rate);
        self
    }

    /// Attach SHINES provenance metadata (SHA-256, creator, timestamp).
    ///
    /// The SHA-256 hash of the raw dataset bytes is computed automatically
    /// during file serialization and stored as `_provenance_sha256`.
    #[cfg(feature = "provenance")]
    pub fn with_provenance(
        &mut self,
        creator: &str,
        timestamp: &str,
        source: Option<&str>,
    ) -> &mut Self {
        self.provenance = Some(ProvenanceConfig {
            creator: creator.to_string(),
            timestamp: timestamp.to_string(),
            source: source.map(|s| s.to_string()),
        });
        self
    }
}

// ---- Group builder ----

/// Builder for HDF5 groups.
///
/// Datasets, sub-groups, and attributes can be added in any order before
/// calling [`finish()`](GroupBuilder::finish). This is useful when the full
/// set of attributes is not known up front — for example, building a
/// MATLAB struct where `MATLAB_fields` lists every child dataset name:
///
/// ```rust
/// # use hdf5_pure::{FileBuilder, AttrValue};
/// let mut builder = FileBuilder::new();
/// let mut grp = builder.create_group("my_struct");
///
/// let mut fields = Vec::new();
/// for name in &["x", "y", "z"] {
///     fields.push(name.to_string());
///     grp.create_dataset(name).with_f64_data(&[0.0]);
/// }
///
/// // Attribute set after all children are created
/// grp.set_attr("MATLAB_fields", AttrValue::VarLenAsciiArray(fields));
/// builder.add_group(grp.finish());
/// ```
pub struct GroupBuilder {
    pub(crate) name: String,
    pub(crate) datasets: Vec<DatasetBuilder>,
    pub(crate) sub_groups: Vec<FinishedGroup>,
    pub(crate) attrs: Vec<(String, AttrValue)>,
}

impl GroupBuilder {
    pub(crate) fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            datasets: Vec::new(),
            sub_groups: Vec::new(),
            attrs: Vec::new(),
        }
    }

    pub fn create_dataset(&mut self, name: &str) -> &mut DatasetBuilder {
        self.datasets.push(DatasetBuilder::new(name));
        self.datasets.last_mut().unwrap()
    }

    /// Create a nested group builder. Call `.finish()` on it and then
    /// `add_group()` to add it to this group.
    pub fn create_group(&mut self, name: &str) -> GroupBuilder {
        GroupBuilder::new(name)
    }

    /// Add a finished sub-group to this group.
    pub fn add_group(&mut self, group: FinishedGroup) {
        self.sub_groups.push(group);
    }

    pub fn set_attr(&mut self, name: &str, value: AttrValue) {
        self.attrs.push((name.to_string(), value));
    }

    /// Consume the builder, returning a FinishedGroup to add to FileWriter.
    pub fn finish(self) -> FinishedGroup {
        FinishedGroup {
            name: self.name,
            datasets: self.datasets,
            sub_groups: self.sub_groups,
            attrs: self.attrs,
        }
    }
}

/// A finished group ready for the file writer.
pub struct FinishedGroup {
    pub(crate) name: String,
    pub(crate) datasets: Vec<DatasetBuilder>,
    pub(crate) sub_groups: Vec<FinishedGroup>,
    pub(crate) attrs: Vec<(String, AttrValue)>,
}
