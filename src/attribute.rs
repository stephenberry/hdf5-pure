//! HDF5 Attribute message parsing (message type 0x000C).

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

use crate::attribute_info::AttributeInfoMessage;
use crate::btree_v2::{BTreeV2Header, collect_btree_v2_records};
use crate::data_read;
use crate::dataspace::Dataspace;
use crate::datatype::Datatype;
use crate::error::FormatError;
use crate::fractal_heap::FractalHeapHeader;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::shared_message;
use crate::vl_data;

/// A parsed HDF5 attribute message.
#[derive(Debug, Clone)]
pub struct AttributeMessage {
    /// Attribute name.
    pub name: String,
    /// Attribute datatype.
    pub datatype: Datatype,
    /// Attribute dataspace.
    pub dataspace: Dataspace,
    /// Raw attribute value data.
    pub raw_data: Vec<u8>,
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

/// Round up to the next multiple of 8.
fn pad8(x: usize) -> usize {
    (x + 7) & !7
}

impl AttributeMessage {
    /// Parse an attribute message from raw message bytes.
    ///
    /// `length_size` is needed for dataspace dimension parsing.
    pub fn parse(data: &[u8], length_size: u8) -> Result<AttributeMessage, FormatError> {
        ensure_len(data, 0, 2)?;
        let version = data[0];

        match version {
            1 => Self::parse_v1(data, length_size),
            2 => Self::parse_v2(data, length_size),
            3 => Self::parse_v3(data, length_size),
            _ => Err(FormatError::InvalidAttributeVersion(version)),
        }
    }

    fn parse_v1(data: &[u8], length_size: u8) -> Result<AttributeMessage, FormatError> {
        // version(1) + reserved(1) + name_size(2) + datatype_size(2) + dataspace_size(2) = 8
        ensure_len(data, 0, 8)?;
        let name_size = u16::from_le_bytes([data[2], data[3]]) as usize;
        let datatype_size = u16::from_le_bytes([data[4], data[5]]) as usize;
        let dataspace_size = u16::from_le_bytes([data[6], data[7]]) as usize;

        let mut pos = 8;

        // Name (padded to 8-byte boundary)
        ensure_len(data, pos, name_size)?;
        let name = extract_name(&data[pos..pos + name_size]);
        pos += pad8(name_size);

        // Datatype (padded to 8-byte boundary)
        ensure_len(data, pos, datatype_size)?;
        let (datatype, _) = Datatype::parse(&data[pos..pos + datatype_size])?;
        pos += pad8(datatype_size);

        // Dataspace (padded to 8-byte boundary)
        ensure_len(data, pos, dataspace_size)?;
        let dataspace = Dataspace::parse(&data[pos..pos + dataspace_size], length_size)?;
        pos += pad8(dataspace_size);

        // Raw data: num_elements × type_size bytes
        let raw_data = compute_raw_data(data, pos, &dataspace, &datatype);

        Ok(AttributeMessage {
            name,
            datatype,
            dataspace,
            raw_data,
        })
    }

    fn parse_v2(data: &[u8], length_size: u8) -> Result<AttributeMessage, FormatError> {
        // version(1) + flags(1) + name_size(2) + datatype_size(2) + dataspace_size(2) = 8
        ensure_len(data, 0, 8)?;
        let name_size = u16::from_le_bytes([data[2], data[3]]) as usize;
        let datatype_size = u16::from_le_bytes([data[4], data[5]]) as usize;
        let dataspace_size = u16::from_le_bytes([data[6], data[7]]) as usize;

        let mut pos = 8;

        // Name (NO padding)
        ensure_len(data, pos, name_size)?;
        let name = extract_name(&data[pos..pos + name_size]);
        pos += name_size;

        // Datatype (NO padding)
        ensure_len(data, pos, datatype_size)?;
        let (datatype, _) = Datatype::parse(&data[pos..pos + datatype_size])?;
        pos += datatype_size;

        // Dataspace (NO padding)
        ensure_len(data, pos, dataspace_size)?;
        let dataspace = Dataspace::parse(&data[pos..pos + dataspace_size], length_size)?;
        pos += dataspace_size;

        let raw_data = compute_raw_data(data, pos, &dataspace, &datatype);

        Ok(AttributeMessage {
            name,
            datatype,
            dataspace,
            raw_data,
        })
    }

    fn parse_v3(data: &[u8], length_size: u8) -> Result<AttributeMessage, FormatError> {
        // version(1) + flags(1) + name_size(2) + datatype_size(2) + dataspace_size(2) + encoding(1) = 9
        ensure_len(data, 0, 9)?;
        let name_size = u16::from_le_bytes([data[2], data[3]]) as usize;
        let datatype_size = u16::from_le_bytes([data[4], data[5]]) as usize;
        let dataspace_size = u16::from_le_bytes([data[6], data[7]]) as usize;
        let _encoding = data[8]; // 0=ASCII, 1=UTF-8

        let mut pos = 9;

        // Name (NO padding)
        ensure_len(data, pos, name_size)?;
        let name = extract_name(&data[pos..pos + name_size]);
        pos += name_size;

        // Datatype (NO padding)
        ensure_len(data, pos, datatype_size)?;
        let (datatype, _) = Datatype::parse(&data[pos..pos + datatype_size])?;
        pos += datatype_size;

        // Dataspace (NO padding)
        ensure_len(data, pos, dataspace_size)?;
        let dataspace = Dataspace::parse(&data[pos..pos + dataspace_size], length_size)?;
        pos += dataspace_size;

        let raw_data = compute_raw_data(data, pos, &dataspace, &datatype);

        Ok(AttributeMessage {
            name,
            datatype,
            dataspace,
            raw_data,
        })
    }

    /// Serialize attribute message (v2 format, no padding).
    pub fn serialize(&self, length_size: u8) -> Vec<u8> {
        self.serialize_version(2, length_size)
    }

    /// Serialize attribute message as v3 (adds character set encoding byte).
    pub fn serialize_v3(&self, length_size: u8) -> Vec<u8> {
        self.serialize_version(3, length_size)
    }

    fn serialize_version(&self, version: u8, length_size: u8) -> Vec<u8> {
        let name_bytes = {
            let mut n = self.name.as_bytes().to_vec();
            n.push(0); // null terminator
            n
        };
        let dt_bytes = self.datatype.serialize();
        let ds_bytes = self.dataspace.serialize(length_size);

        let mut buf = Vec::new();
        buf.push(version);
        buf.push(0); // flags
        buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        buf.extend_from_slice(&(dt_bytes.len() as u16).to_le_bytes());
        buf.extend_from_slice(&(ds_bytes.len() as u16).to_le_bytes());
        if version >= 3 {
            buf.push(0x00); // character set encoding: ASCII
        }
        buf.extend_from_slice(&name_bytes);
        buf.extend_from_slice(&dt_bytes);
        buf.extend_from_slice(&ds_bytes);
        buf.extend_from_slice(&self.raw_data);
        buf
    }

    /// Read attribute value as f64 values.
    pub fn read_as_f64(&self) -> Result<Vec<f64>, FormatError> {
        data_read::read_as_f64(&self.raw_data, &self.datatype)
    }

    /// Read attribute value as i64 values.
    pub fn read_as_i64(&self) -> Result<Vec<i64>, FormatError> {
        data_read::read_as_i64(&self.raw_data, &self.datatype)
    }

    /// Read attribute value as u64 values.
    pub fn read_as_u64(&self) -> Result<Vec<u64>, FormatError> {
        data_read::read_as_u64(&self.raw_data, &self.datatype)
    }

    /// Read attribute value as a single string (first element).
    pub fn read_as_string(&self) -> Result<String, FormatError> {
        let strings = data_read::read_as_strings(&self.raw_data, &self.datatype)?;
        Ok(strings.into_iter().next().unwrap_or_default())
    }

    /// Read attribute value as a vector of fixed-length strings.
    pub fn read_as_strings(&self) -> Result<Vec<String>, FormatError> {
        data_read::read_as_strings(&self.raw_data, &self.datatype)
    }

    /// Read variable-length string attribute values.
    ///
    /// Needs the full file data and offset/length sizes from the superblock
    /// because VL strings store their data in the global heap.
    pub fn read_vl_strings(
        &self,
        file_data: &[u8],
        offset_size: u8,
        length_size: u8,
    ) -> Result<Vec<String>, FormatError> {
        let num_elements = self.dataspace.num_elements();
        vl_data::read_vl_strings(
            file_data,
            &self.raw_data,
            num_elements,
            offset_size,
            length_size,
        )
    }
}

/// Compute raw data size based on dataspace and datatype, then extract from message bytes.
fn compute_raw_data(
    data: &[u8],
    pos: usize,
    dataspace: &Dataspace,
    datatype: &Datatype,
) -> Vec<u8> {
    let num_elements = dataspace.num_elements() as usize;
    let elem_size = datatype.type_size() as usize;
    let expected_size = num_elements * elem_size;
    let available = data.len().saturating_sub(pos);
    let take = expected_size.min(available);
    if take > 0 {
        data[pos..pos + take].to_vec()
    } else if available > 0 {
        // Fallback: take whatever is available (e.g., for VL types where type_size may not match)
        data[pos..].to_vec()
    } else {
        Vec::new()
    }
}

/// Extract a name from raw bytes, stripping null terminator.
fn extract_name(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// Extract all attribute messages from an object header.
pub fn extract_attributes(
    header: &ObjectHeader,
    length_size: u8,
) -> Result<Vec<AttributeMessage>, FormatError> {
    let mut attrs = Vec::new();
    for msg in &header.messages {
        if msg.msg_type == MessageType::Attribute {
            let attr = AttributeMessage::parse(&msg.data, length_size)?;
            attrs.push(attr);
        }
    }
    Ok(attrs)
}

/// Find a specific attribute by name.
pub fn find_attribute<'a>(
    attrs: &'a [AttributeMessage],
    name: &str,
) -> Option<&'a AttributeMessage> {
    attrs.iter().find(|a| a.name == name)
}

/// Extract all attributes from an object header, supporting both compact and dense storage.
///
/// This function handles:
/// - Compact attributes: inline Attribute messages (0x000C) in the object header
/// - Dense attributes: AttributeInfo message (0x0015) pointing to fractal heap + B-tree v2
/// - Shared messages: resolves shared datatype references for attribute messages
///
/// Use this instead of `extract_attributes` when reading files that may use dense storage
/// (e.g., objects with many attributes, typically >8).
pub fn extract_attributes_full(
    file_data: &[u8],
    header: &ObjectHeader,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<AttributeMessage>, FormatError> {
    let mut attrs = Vec::new();

    // Collect compact attributes (inline in OH)
    for msg in &header.messages {
        if msg.msg_type == MessageType::Attribute {
            if shared_message::is_shared(msg.flags) {
                // Shared attribute: resolve the reference to get actual attribute data
                let shared_ref = shared_message::parse_shared_ref(&msg.data, offset_size)?;
                let resolved_data = shared_message::resolve_shared_message(
                    file_data,
                    &shared_ref,
                    MessageType::Attribute,
                    offset_size,
                    length_size,
                )?;
                let attr = AttributeMessage::parse(&resolved_data, length_size)?;
                attrs.push(attr);
            } else {
                let attr = AttributeMessage::parse(&msg.data, length_size)?;
                attrs.push(attr);
            }
        }
    }

    // Check for dense attributes via AttributeInfo message
    let attr_info = find_attribute_info(header, offset_size)?;
    if let Some(info) = attr_info
        && let Some(fh_addr) = info.fractal_heap_address
    {
        let dense_attrs =
            extract_dense_attributes(file_data, &info, fh_addr, offset_size, length_size)?;
        attrs.extend(dense_attrs);
    }

    Ok(attrs)
}

/// Find and parse the Attribute Info message from an object header.
fn find_attribute_info(
    header: &ObjectHeader,
    offset_size: u8,
) -> Result<Option<AttributeInfoMessage>, FormatError> {
    for msg in &header.messages {
        if msg.msg_type == MessageType::AttributeInfo {
            let info = AttributeInfoMessage::parse(&msg.data, offset_size)?;
            return Ok(Some(info));
        }
    }
    Ok(None)
}

/// Extract attributes from dense storage (fractal heap + B-tree v2).
fn extract_dense_attributes(
    file_data: &[u8],
    attr_info: &AttributeInfoMessage,
    fh_addr: u64,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<AttributeMessage>, FormatError> {
    // Parse fractal heap
    let fh = FractalHeapHeader::parse(file_data, fh_addr as usize, offset_size, length_size)?;

    // Parse B-tree v2 for name index (type 8)
    let btree_addr = attr_info
        .btree_name_index_address
        .ok_or(FormatError::UnexpectedEof {
            expected: 1,
            available: 0,
        })?;
    let btree_hdr = BTreeV2Header::parse(file_data, btree_addr as usize, offset_size, length_size)?;
    let records = collect_btree_v2_records(file_data, &btree_hdr, offset_size, length_size)?;

    let mut attrs = Vec::new();
    for record in &records {
        // Per HDF5 spec, both type 8 and type 9 records start with heap_id:
        //   Type 8: heap_id(8) + msg_flags(1) + creation_order(4) + hash(4)
        //   Type 9: heap_id(8) + msg_flags(1) + creation_order(4)
        let id_offset = 0;

        if record.data.len() < id_offset + fh.heap_id_length as usize {
            continue;
        }
        let id_bytes = &record.data[id_offset..id_offset + fh.heap_id_length as usize];

        // Read attribute message from fractal heap
        let attr_data = fh.read_managed_object(file_data, id_bytes, offset_size)?;

        // The data in the heap is a complete attribute message
        let attr = AttributeMessage::parse(&attr_data, length_size)?;
        attrs.push(attr);
    }

    Ok(attrs)
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Build a datatype header for testing (8 bytes).
    fn build_dt_header(class: u8, version: u8, bf: [u8; 3], size: u32) -> Vec<u8> {
        let mut buf = vec![0u8; 8];
        buf[0] = (class & 0x0F) | ((version & 0x0F) << 4);
        buf[1] = bf[0];
        buf[2] = bf[1];
        buf[3] = bf[2];
        buf[4..8].copy_from_slice(&size.to_le_bytes());
        buf
    }

    /// Build an f64 LE datatype message.
    fn build_f64_dt() -> Vec<u8> {
        let mut buf = build_dt_header(1, 1, [0x00, 0x00, 0x02], 8);
        let mut props = [0u8; 12];
        props[2..4].copy_from_slice(&64u16.to_le_bytes()); // bit_precision
        props[4] = 52; // exp_location
        props[5] = 11; // exp_size
        props[6] = 0; // mant_location
        props[7] = 52; // mant_size
        props[8..12].copy_from_slice(&1023u32.to_le_bytes()); // exp_bias
        buf.extend_from_slice(&props);
        buf
    }

    /// Build a scalar dataspace (v2).
    fn build_scalar_ds() -> Vec<u8> {
        vec![2, 0, 0, 0] // version=2, rank=0, flags=0, type=0(scalar)
    }

    /// Build a simple 1D dataspace (v1).
    fn build_simple_ds_v1(dim: u64) -> Vec<u8> {
        let mut buf = vec![1u8, 1, 0, 0, 0, 0, 0, 0]; // version=1, rank=1, flags=0, reserved(5)
        buf.extend_from_slice(&dim.to_le_bytes());
        buf
    }

    /// Build a fixed-length string datatype.
    fn build_string_dt(size: u32) -> Vec<u8> {
        // class=3, version=1, padding=NullPad(1), charset=ASCII(0) → bf0=0x01
        build_dt_header(3, 1, [0x01, 0, 0], size)
    }

    #[test]
    fn parse_v1_attribute_f64_scalar() {
        let name = b"temp\0";
        let dt_bytes = build_f64_dt();
        let ds_bytes = build_scalar_ds();

        let name_size = name.len();
        let dt_size = dt_bytes.len();
        let ds_size = ds_bytes.len();

        let mut data = Vec::new();
        data.push(1); // version
        data.push(0); // reserved
        data.extend_from_slice(&(name_size as u16).to_le_bytes());
        data.extend_from_slice(&(dt_size as u16).to_le_bytes());
        data.extend_from_slice(&(ds_size as u16).to_le_bytes());

        // Name padded to 8 bytes
        data.extend_from_slice(name);
        if data.len() % 8 != 0 || data.len() == 8 {
            // Pad name to 8-byte boundary from start of name
            let name_start = 8;
            let name_padded = pad8(name_size);
            while data.len() < name_start + name_padded {
                data.push(0);
            }
        }

        // Datatype padded to 8 bytes
        let dt_start = data.len();
        data.extend_from_slice(&dt_bytes);
        let dt_padded = pad8(dt_size);
        while data.len() < dt_start + dt_padded {
            data.push(0);
        }

        // Dataspace padded to 8 bytes
        let ds_start = data.len();
        data.extend_from_slice(&ds_bytes);
        let ds_padded = pad8(ds_size);
        while data.len() < ds_start + ds_padded {
            data.push(0);
        }

        // Raw data: f64 value 98.6
        data.extend_from_slice(&98.6f64.to_le_bytes());

        let attr = AttributeMessage::parse(&data, 8).unwrap();
        assert_eq!(attr.name, "temp");
        assert_eq!(attr.dataspace.num_elements(), 1);
        let vals = attr.read_as_f64().unwrap();
        assert_eq!(vals.len(), 1);
        assert!((vals[0] - 98.6).abs() < 1e-10);
    }

    #[test]
    fn parse_v2_attribute_fixed_string() {
        let name = b"label\0";
        let dt_bytes = build_string_dt(5);
        let ds_bytes = build_scalar_ds();

        let mut data = Vec::new();
        data.push(2); // version
        data.push(0); // flags
        data.extend_from_slice(&(name.len() as u16).to_le_bytes());
        data.extend_from_slice(&(dt_bytes.len() as u16).to_le_bytes());
        data.extend_from_slice(&(ds_bytes.len() as u16).to_le_bytes());

        // No padding in v2
        data.extend_from_slice(name);
        data.extend_from_slice(&dt_bytes);
        data.extend_from_slice(&ds_bytes);

        // Raw data: "hello"
        data.extend_from_slice(b"hello");

        let attr = AttributeMessage::parse(&data, 8).unwrap();
        assert_eq!(attr.name, "label");
        let s = attr.read_as_string().unwrap();
        assert_eq!(s, "hello");
    }

    #[test]
    fn parse_v3_attribute_utf8() {
        let name = b"note\0";
        let dt_bytes = build_string_dt(3);
        let ds_bytes = build_scalar_ds();

        let mut data = Vec::new();
        data.push(3); // version
        data.push(0); // flags
        data.extend_from_slice(&(name.len() as u16).to_le_bytes());
        data.extend_from_slice(&(dt_bytes.len() as u16).to_le_bytes());
        data.extend_from_slice(&(ds_bytes.len() as u16).to_le_bytes());
        data.push(1); // encoding = UTF-8

        data.extend_from_slice(name);
        data.extend_from_slice(&dt_bytes);
        data.extend_from_slice(&ds_bytes);
        data.extend_from_slice(b"abc");

        let attr = AttributeMessage::parse(&data, 8).unwrap();
        assert_eq!(attr.name, "note");
        let s = attr.read_as_string().unwrap();
        assert_eq!(s, "abc");
    }

    #[test]
    fn parse_v2_attribute_1d_array() {
        let name = b"vals\0";
        let dt_bytes = build_f64_dt();
        let ds_bytes = build_simple_ds_v1(3);

        let mut data = Vec::new();
        data.push(2); // version
        data.push(0); // flags
        data.extend_from_slice(&(name.len() as u16).to_le_bytes());
        data.extend_from_slice(&(dt_bytes.len() as u16).to_le_bytes());
        data.extend_from_slice(&(ds_bytes.len() as u16).to_le_bytes());

        data.extend_from_slice(name);
        data.extend_from_slice(&dt_bytes);
        data.extend_from_slice(&ds_bytes);

        // 3 f64 values
        data.extend_from_slice(&1.0f64.to_le_bytes());
        data.extend_from_slice(&2.0f64.to_le_bytes());
        data.extend_from_slice(&3.0f64.to_le_bytes());

        let attr = AttributeMessage::parse(&data, 8).unwrap();
        assert_eq!(attr.name, "vals");
        let vals = attr.read_as_f64().unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn parse_v1_padding_alignment() {
        // Verify v1 pads name, dt, ds each to 8 bytes
        let name = b"x\0"; // 2 bytes → pad to 8
        let dt_bytes = build_f64_dt(); // 20 bytes → pad to 24
        let ds_bytes = build_scalar_ds(); // 4 bytes → pad to 8

        let mut data = Vec::new();
        data.push(1); // version
        data.push(0); // reserved
        data.extend_from_slice(&(name.len() as u16).to_le_bytes());
        data.extend_from_slice(&(dt_bytes.len() as u16).to_le_bytes());
        data.extend_from_slice(&(ds_bytes.len() as u16).to_le_bytes());

        // Name padded to 8
        data.extend_from_slice(name);
        data.resize(8 + pad8(name.len()), 0);

        // DT padded to 8
        let dt_start = data.len();
        data.extend_from_slice(&dt_bytes);
        data.resize(dt_start + pad8(dt_bytes.len()), 0);

        // DS padded to 8
        let ds_start = data.len();
        data.extend_from_slice(&ds_bytes);
        data.resize(ds_start + pad8(ds_bytes.len()), 0);

        // raw data
        data.extend_from_slice(&42.0f64.to_le_bytes());

        let attr = AttributeMessage::parse(&data, 8).unwrap();
        assert_eq!(attr.name, "x");
        let vals = attr.read_as_f64().unwrap();
        assert_eq!(vals, vec![42.0]);
    }

    #[test]
    fn parse_v2_no_padding() {
        // Same as parse_v2_attribute_fixed_string but verifying no padding
        let name = b"ab\0"; // 3 bytes, no padding
        let dt_bytes = build_string_dt(2); // 8 bytes, no padding
        let ds_bytes = build_scalar_ds(); // 4 bytes, no padding

        let mut data = Vec::new();
        data.push(2);
        data.push(0);
        data.extend_from_slice(&(name.len() as u16).to_le_bytes());
        data.extend_from_slice(&(dt_bytes.len() as u16).to_le_bytes());
        data.extend_from_slice(&(ds_bytes.len() as u16).to_le_bytes());
        data.extend_from_slice(name);
        data.extend_from_slice(&dt_bytes);
        data.extend_from_slice(&ds_bytes);
        data.extend_from_slice(b"hi");

        let attr = AttributeMessage::parse(&data, 8).unwrap();
        assert_eq!(attr.name, "ab");
        assert_eq!(attr.read_as_string().unwrap(), "hi");
    }

    #[test]
    fn truncated_attribute_error() {
        let data = [1u8]; // too short
        let err = AttributeMessage::parse(&data, 8).unwrap_err();
        assert!(matches!(err, FormatError::UnexpectedEof { .. }));
    }

    #[test]
    fn invalid_version_error() {
        let data = [5u8, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let err = AttributeMessage::parse(&data, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidAttributeVersion(5));
    }

    #[test]
    fn extract_attributes_from_header() {
        // Build a fake ObjectHeader with 3 attribute messages
        let mut msgs = Vec::new();
        for i in 0..3 {
            let name = format!("attr{}\0", i);
            let dt_bytes = build_f64_dt();
            let ds_bytes = build_scalar_ds();

            let mut attr_data = Vec::new();
            attr_data.push(2); // version
            attr_data.push(0);
            attr_data.extend_from_slice(&(name.len() as u16).to_le_bytes());
            attr_data.extend_from_slice(&(dt_bytes.len() as u16).to_le_bytes());
            attr_data.extend_from_slice(&(ds_bytes.len() as u16).to_le_bytes());
            attr_data.extend_from_slice(name.as_bytes());
            attr_data.extend_from_slice(&dt_bytes);
            attr_data.extend_from_slice(&ds_bytes);
            attr_data.extend_from_slice(&((i as f64) * 1.0).to_le_bytes());

            msgs.push(crate::object_header::HeaderMessage {
                msg_type: MessageType::Attribute,
                size: attr_data.len(),
                flags: 0,
                creation_order: None,
                data: attr_data,
            });
        }

        let header = ObjectHeader {
            version: 2,
            messages: msgs,
            reference_count: None,
            flags: 0,
            access_time: None,
            modification_time: None,
            change_time: None,
            birth_time: None,
        };

        let attrs = extract_attributes(&header, 8).unwrap();
        assert_eq!(attrs.len(), 3);
        assert_eq!(attrs[0].name, "attr0");
        assert_eq!(attrs[1].name, "attr1");
        assert_eq!(attrs[2].name, "attr2");
    }

    #[test]
    fn find_attribute_by_name() {
        let name = b"target\0";
        let dt_bytes = build_f64_dt();
        let ds_bytes = build_scalar_ds();

        let mut attr_data = Vec::new();
        attr_data.push(2);
        attr_data.push(0);
        attr_data.extend_from_slice(&(name.len() as u16).to_le_bytes());
        attr_data.extend_from_slice(&(dt_bytes.len() as u16).to_le_bytes());
        attr_data.extend_from_slice(&(ds_bytes.len() as u16).to_le_bytes());
        attr_data.extend_from_slice(name);
        attr_data.extend_from_slice(&dt_bytes);
        attr_data.extend_from_slice(&ds_bytes);
        attr_data.extend_from_slice(&99.0f64.to_le_bytes());

        let attr = AttributeMessage::parse(&attr_data, 8).unwrap();
        let attrs = vec![attr];

        assert!(find_attribute(&attrs, "target").is_some());
        assert!(find_attribute(&attrs, "missing").is_none());
    }

    #[test]
    fn read_as_f64_scalar() {
        let name = b"v\0";
        let dt_bytes = build_f64_dt();
        let ds_bytes = build_scalar_ds();

        let mut data = Vec::new();
        data.push(2);
        data.push(0);
        data.extend_from_slice(&(name.len() as u16).to_le_bytes());
        data.extend_from_slice(&(dt_bytes.len() as u16).to_le_bytes());
        data.extend_from_slice(&(ds_bytes.len() as u16).to_le_bytes());
        data.extend_from_slice(name);
        data.extend_from_slice(&dt_bytes);
        data.extend_from_slice(&ds_bytes);
        data.extend_from_slice(&3.14f64.to_le_bytes());

        let attr = AttributeMessage::parse(&data, 8).unwrap();
        let vals = attr.read_as_f64().unwrap();
        assert_eq!(vals, vec![3.14]);
    }

    #[test]
    fn read_as_string_fixed() {
        let name = b"s\0";
        let dt_bytes = build_string_dt(5);
        let ds_bytes = build_scalar_ds();

        let mut data = Vec::new();
        data.push(2);
        data.push(0);
        data.extend_from_slice(&(name.len() as u16).to_le_bytes());
        data.extend_from_slice(&(dt_bytes.len() as u16).to_le_bytes());
        data.extend_from_slice(&(ds_bytes.len() as u16).to_le_bytes());
        data.extend_from_slice(name);
        data.extend_from_slice(&dt_bytes);
        data.extend_from_slice(&ds_bytes);
        data.extend_from_slice(b"world");

        let attr = AttributeMessage::parse(&data, 8).unwrap();
        assert_eq!(attr.read_as_string().unwrap(), "world");
    }

    #[test]
    fn read_as_strings_array() {
        let name = b"arr\0";
        let dt_bytes = build_string_dt(4);
        let ds_bytes = build_simple_ds_v1(2);

        let mut data = Vec::new();
        data.push(2);
        data.push(0);
        data.extend_from_slice(&(name.len() as u16).to_le_bytes());
        data.extend_from_slice(&(dt_bytes.len() as u16).to_le_bytes());
        data.extend_from_slice(&(ds_bytes.len() as u16).to_le_bytes());
        data.extend_from_slice(name);
        data.extend_from_slice(&dt_bytes);
        data.extend_from_slice(&ds_bytes);
        data.extend_from_slice(b"abcdEFGH");

        let attr = AttributeMessage::parse(&data, 8).unwrap();
        let strs = attr.read_as_strings().unwrap();
        assert_eq!(strs, vec!["abcd", "EFGH"]);
    }
}
