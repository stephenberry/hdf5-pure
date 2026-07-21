//! Variable-length data reading (VL strings & VL sequences).
//!
//! VL data elements in HDF5 store their values in the global heap.
//! The raw data for each element contains a global heap ID:
//! `sequence_length(4 LE) + collection_address(offset_size LE) + object_index(4 LE)`.

#[cfg(not(feature = "std"))]
use alloc::{format, string::String, vec::Vec};

use crate::convert::{TryToUsize, is_undefined_addr};
use crate::datatype::{CharacterSet, Datatype};
use crate::error::FormatError;
use crate::global_heap::GlobalHeapIndex;
#[cfg(test)]
use crate::source::BytesSource;
use crate::source::Source;

/// Allocation limits for reading variable-length strings.
///
/// Limits are checked before any string payload is materialized. The payload
/// byte limit covers the bytes referenced by the VL elements; it excludes the
/// `Vec<String>` and `String` allocation metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VlenStringReadOptions {
    max_elements: Option<usize>,
    max_payload_bytes: Option<usize>,
}

impl VlenStringReadOptions {
    /// Create options with no limits.
    pub const fn new() -> Self {
        Self {
            max_elements: None,
            max_payload_bytes: None,
        }
    }

    /// Set the maximum number of VL elements that may be read.
    pub const fn with_max_elements(mut self, max_elements: usize) -> Self {
        self.max_elements = Some(max_elements);
        self
    }

    /// Set the maximum total string payload size in bytes.
    pub const fn with_max_payload_bytes(mut self, max_payload_bytes: usize) -> Self {
        self.max_payload_bytes = Some(max_payload_bytes);
        self
    }

    /// Return the configured element limit.
    pub const fn max_elements(&self) -> Option<usize> {
        self.max_elements
    }

    /// Return the configured payload-byte limit.
    pub const fn max_payload_bytes(&self) -> Option<usize> {
        self.max_payload_bytes
    }
}

/// A parsed variable-length element reference (global heap ID).
#[derive(Debug, Clone)]
pub struct VlElement {
    /// Length of the VL data.
    pub length: u32,
    /// Address of the global heap collection containing the data.
    pub collection_address: u64,
    /// Index of the object within the collection.
    pub object_index: u32,
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

fn read_offset(data: &[u8], pos: usize, offset_size: u8) -> Result<u64, FormatError> {
    let s = offset_size as usize;
    ensure_len(data, pos, s)?;
    let slice = &data[pos..pos + s];
    Ok(match offset_size {
        2 => u16::from_le_bytes([slice[0], slice[1]]) as u64,
        4 => u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]) as u64,
        8 => u64::from_le_bytes([
            slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
        ]),
        _ => return Err(FormatError::InvalidOffsetSize(offset_size)),
    })
}

/// Parse VL global heap references from raw attribute/dataset data.
pub fn parse_vl_references(
    raw_data: &[u8],
    num_elements: u64,
    offset_size: u8,
) -> Result<Vec<VlElement>, FormatError> {
    let elem_size = 4 + offset_size as u64 + 4; // length + address + index
    let total = num_elements
        .checked_mul(elem_size)
        .ok_or(FormatError::OffsetOverflow {
            offset: num_elements,
            length: elem_size,
        })?
        .to_usize()?;
    if raw_data.len() < total {
        return Err(FormatError::UnexpectedEof {
            expected: total,
            available: raw_data.len(),
        });
    }

    let mut elements = Vec::with_capacity(num_elements.to_usize()?);
    let mut pos = 0;

    for _ in 0..num_elements {
        let length = u32::from_le_bytes([
            raw_data[pos],
            raw_data[pos + 1],
            raw_data[pos + 2],
            raw_data[pos + 3],
        ]);
        pos += 4;

        let collection_address = read_offset(raw_data, pos, offset_size)?;
        pos += offset_size as usize;

        let object_index = u32::from_le_bytes([
            raw_data[pos],
            raw_data[pos + 1],
            raw_data[pos + 2],
            raw_data[pos + 3],
        ]);
        pos += 4;

        elements.push(VlElement {
            length,
            collection_address,
            object_index,
        });
    }

    Ok(elements)
}

/// Whether a datatype is one of the string-shaped VL encodings understood by
/// this module.
pub(crate) fn is_vlen_string_datatype(datatype: &Datatype) -> bool {
    match datatype {
        Datatype::VariableLength {
            is_string: true, ..
        } => true,
        Datatype::VariableLength {
            is_string: false,
            base_type,
            ..
        } => matches!(
            base_type.as_ref(),
            Datatype::String {
                size: 1,
                charset: CharacterSet::Ascii,
                ..
            }
        ),
        _ => false,
    }
}

fn check_element_limit(
    num_elements: u64,
    options: VlenStringReadOptions,
) -> Result<(), FormatError> {
    if let Some(limit) = options.max_elements
        && num_elements > limit as u64
    {
        return Err(FormatError::VariableLengthElementLimitExceeded {
            limit,
            actual: num_elements,
        });
    }
    Ok(())
}

fn payload_size(refs: &[VlElement], options: VlenStringReadOptions) -> Result<u64, FormatError> {
    let mut required = 0u64;
    for element in refs {
        required =
            required
                .checked_add(u64::from(element.length))
                .ok_or(FormatError::OffsetOverflow {
                    offset: required,
                    length: u64::from(element.length),
                })?;
    }
    if let Some(limit) = options.max_payload_bytes
        && required > limit as u64
    {
        return Err(FormatError::VariableLengthByteLimitExceeded { limit, required });
    }
    Ok(required)
}

/// Return the total payload bytes named by a set of VL references.
pub fn vlen_string_payload_size(
    raw_data: &[u8],
    num_elements: u64,
    offset_size: u8,
) -> Result<u64, FormatError> {
    check_element_limit(num_elements, VlenStringReadOptions::default())?;
    let refs = parse_vl_references(raw_data, num_elements, offset_size)?;
    payload_size(&refs, VlenStringReadOptions::default())
}

/// Resolve VL strings from a random-access file source and pass them to a
/// visitor one at a time.
pub fn visit_vl_strings_from_source<S, F>(
    source: &S,
    raw_data: &[u8],
    num_elements: u64,
    offset_size: u8,
    length_size: u8,
    base_address: u64,
    options: VlenStringReadOptions,
    mut visitor: F,
) -> Result<(), FormatError>
where
    S: Source + ?Sized,
    F: FnMut(&str),
{
    check_element_limit(num_elements, options)?;
    let refs = parse_vl_references(raw_data, num_elements, offset_size)?;
    payload_size(&refs, options)?;

    let mut collections: Vec<(u64, GlobalHeapIndex)> = Vec::new();
    for element in &refs {
        if element.length == 0
            && (is_undefined_addr(element.collection_address, offset_size)
                || element.collection_address == 0)
        {
            visitor("");
            continue;
        }
        if is_undefined_addr(element.collection_address, offset_size) {
            return Err(FormatError::VlDataError(
                "non-empty VL element has an undefined heap address".into(),
            ));
        }

        let collection_address = element.collection_address.checked_add(base_address).ok_or(
            FormatError::OffsetOverflow {
                offset: element.collection_address,
                length: base_address,
            },
        )?;
        let collection_pos = match collections
            .iter()
            .position(|(address, _)| *address == collection_address)
        {
            Some(pos) => pos,
            None => {
                let collection = GlobalHeapIndex::parse(source, collection_address, length_size)?;
                collections.push((collection_address, collection));
                collections.len() - 1
            }
        };

        let index = u16::try_from(element.object_index).map_err(|_| {
            FormatError::VlDataError(format!(
                "global heap object index {} does not fit u16",
                element.object_index
            ))
        })?;
        let object = collections[collection_pos].1.get_object(index).ok_or(
            FormatError::GlobalHeapObjectNotFound {
                collection_address,
                index,
            },
        )?;
        if u64::from(element.length) > object.size {
            return Err(FormatError::VlDataError(format!(
                "VL element length {} exceeds global heap object size {}",
                element.length, object.size
            )));
        }

        let bytes = source.read_exact_at(object.data_address, element.length as usize)?;
        let string = String::from_utf8_lossy(&bytes);
        visitor(&string);
    }

    Ok(())
}

/// Resolve VL strings from a random-access file source.
pub fn read_vl_strings_from_source<S: Source + ?Sized>(
    source: &S,
    raw_data: &[u8],
    num_elements: u64,
    offset_size: u8,
    length_size: u8,
    base_address: u64,
    options: VlenStringReadOptions,
) -> Result<Vec<String>, FormatError> {
    let mut strings = Vec::new();
    visit_vl_strings_from_source(
        source,
        raw_data,
        num_elements,
        offset_size,
        length_size,
        base_address,
        options,
        |string| strings.push(String::from(string)),
    )?;
    Ok(strings)
}

/// One element of a variable-length string dataset/attribute, read as exact
/// heap bytes rather than a lossily-decoded `String`.
///
/// `None` is a *null* reference (length 0 with an undefined or zero heap
/// address), which the HDF5 model distinguishes from an empty string. `Some`
/// is a real heap object, carrying its exact bytes (possibly empty, possibly
/// containing embedded NULs or non-UTF-8 sequences). Preserving this
/// distinction lets a faithful rewrite reproduce the source byte-for-byte.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum VlByteObject {
    /// A null VL reference (no heap object).
    Null,
    /// A heap object holding these exact bytes.
    Bytes(Vec<u8>),
}

/// Resolve a VL element's exact heap bytes from a random-access source,
/// preserving the null-vs-empty distinction and never lossily decoding.
///
/// This mirrors [`visit_vl_strings_from_source`] but yields raw bytes (and a
/// null marker) instead of a `&str`, so a faithful rewrite can reproduce
/// embedded-NUL and non-UTF-8 payloads exactly.
///
/// `element_size` is the byte width of one base-type element of the sequence.
/// For VL strings the base type is a single byte, so `element_size == 1` and the
/// reference's stored `length` (an element count) equals the byte count. For a
/// non-string VL sequence (e.g. `H5T_VLEN { H5T_NATIVE_DOUBLE }`) the stored
/// `length` counts base-type elements, so the heap object holds
/// `length * element_size` bytes — exactly what is read here.
pub(crate) fn read_vl_byte_objects_from_source<S: Source + ?Sized>(
    source: &S,
    raw_data: &[u8],
    num_elements: u64,
    offset_size: u8,
    length_size: u8,
    base_address: u64,
    element_size: usize,
    options: VlenStringReadOptions,
) -> Result<Vec<VlByteObject>, FormatError> {
    check_element_limit(num_elements, options)?;
    let refs = parse_vl_references(raw_data, num_elements, offset_size)?;
    payload_size(&refs, options)?;

    let mut objects = Vec::with_capacity(refs.len());
    let mut collections: Vec<(u64, GlobalHeapIndex)> = Vec::new();
    for element in &refs {
        if element.length == 0
            && (is_undefined_addr(element.collection_address, offset_size)
                || element.collection_address == 0)
        {
            objects.push(VlByteObject::Null);
            continue;
        }
        if is_undefined_addr(element.collection_address, offset_size) {
            return Err(FormatError::VlDataError(
                "non-empty VL element has an undefined heap address".into(),
            ));
        }

        let collection_address = element.collection_address.checked_add(base_address).ok_or(
            FormatError::OffsetOverflow {
                offset: element.collection_address,
                length: base_address,
            },
        )?;
        let collection_pos = match collections
            .iter()
            .position(|(address, _)| *address == collection_address)
        {
            Some(pos) => pos,
            None => {
                let collection = GlobalHeapIndex::parse(source, collection_address, length_size)?;
                collections.push((collection_address, collection));
                collections.len() - 1
            }
        };

        let index = u16::try_from(element.object_index).map_err(|_| {
            FormatError::VlDataError(format!(
                "global heap object index {} does not fit u16",
                element.object_index
            ))
        })?;
        let object = collections[collection_pos].1.get_object(index).ok_or(
            FormatError::GlobalHeapObjectNotFound {
                collection_address,
                index,
            },
        )?;
        // The heap object holds `length` base-type elements of `element_size`
        // bytes each. Compute the byte count with checked arithmetic so a hostile
        // `length` cannot overflow, and bound it by the heap object's own size.
        let byte_len = (element.length as u64)
            .checked_mul(element_size as u64)
            .ok_or(FormatError::OffsetOverflow {
                offset: u64::from(element.length),
                length: element_size as u64,
            })?;
        if byte_len > object.size {
            return Err(FormatError::VlDataError(format!(
                "VL element length {} ({} bytes) exceeds global heap object size {}",
                element.length, byte_len, object.size
            )));
        }

        let bytes = source.read_exact_at(object.data_address, byte_len.to_usize()?)?;
        objects.push(VlByteObject::Bytes(bytes));
    }

    Ok(objects)
}

/// Resolve VL strings from an in-memory buffer by looking up each element in the
/// global heap. A thin convenience wrapper over
/// [`read_vl_strings_from_source`] used by the unit tests; production callers go
/// straight to the source-based reader so a streaming backend works unchanged.
#[cfg(test)]
pub fn read_vl_strings(
    file_data: &[u8],
    raw_data: &[u8],
    num_elements: u64,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<String>, FormatError> {
    read_vl_strings_from_source(
        &BytesSource::new(file_data),
        raw_data,
        num_elements,
        offset_size,
        length_size,
        0,
        VlenStringReadOptions::default(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a global heap collection at given offset in a file buffer.
    fn build_gcol_at(
        file_data: &mut Vec<u8>,
        offset: usize,
        objects: &[(u16, &[u8])], // (index, data)
    ) {
        let length_size = 8usize;

        // Ensure file_data is large enough
        let header_size = 8 + length_size;
        let mut obj_total = 0usize;
        for (_, data) in objects {
            let padded = (data.len() + 7) & !7;
            obj_total += 8 + length_size + padded;
        }
        obj_total += 2; // free space marker
        let collection_size = header_size + obj_total;
        let needed = offset + collection_size;
        if file_data.len() < needed {
            file_data.resize(needed, 0);
        }

        let mut pos = offset;
        // Signature
        file_data[pos..pos + 4].copy_from_slice(b"GCOL");
        file_data[pos + 4] = 1; // version
        // reserved(3) already 0
        pos += 8;
        file_data[pos..pos + 8].copy_from_slice(&(collection_size as u64).to_le_bytes());
        pos += 8;

        for (index, data) in objects {
            file_data[pos..pos + 2].copy_from_slice(&index.to_le_bytes());
            file_data[pos + 2..pos + 4].copy_from_slice(&1u16.to_le_bytes()); // ref_count
            // reserved(4) already 0
            pos += 8;
            file_data[pos..pos + 8].copy_from_slice(&(data.len() as u64).to_le_bytes());
            pos += 8;
            file_data[pos..pos + data.len()].copy_from_slice(data);
            let padded = (data.len() + 7) & !7;
            pos += padded;
        }
        // free space marker
        file_data[pos..pos + 2].copy_from_slice(&0u16.to_le_bytes());
    }

    /// Build VL reference raw data for given strings at a collection address.
    fn build_vl_refs(
        strings: &[&str],
        collection_address: u64,
        start_index: u16,
        offset_size: u8,
    ) -> Vec<u8> {
        let mut raw = Vec::new();
        for (i, s) in strings.iter().enumerate() {
            raw.extend_from_slice(&(s.len() as u32).to_le_bytes());
            match offset_size {
                4 => raw.extend_from_slice(&(collection_address as u32).to_le_bytes()),
                8 => raw.extend_from_slice(&collection_address.to_le_bytes()),
                _ => panic!("unsupported"),
            }
            raw.extend_from_slice(&(start_index as u32 + i as u32).to_le_bytes());
        }
        raw
    }

    #[test]
    fn parse_vl_references_two_elements() {
        let raw = build_vl_refs(&["hello", "world"], 0x1000, 1, 8);
        let refs = parse_vl_references(&raw, 2, 8).unwrap();
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].length, 5);
        assert_eq!(refs[0].collection_address, 0x1000);
        assert_eq!(refs[0].object_index, 1);
        assert_eq!(refs[1].length, 5);
        assert_eq!(refs[1].object_index, 2);
    }

    #[test]
    fn read_vl_strings_from_heap() {
        let gcol_offset = 256usize;
        let mut file_data = vec![0u8; 512];
        build_gcol_at(&mut file_data, gcol_offset, &[(1, b"Alice"), (2, b"Bob")]);

        let raw = build_vl_refs(&["Alice", "Bob"], gcol_offset as u64, 1, 8);
        let strings = read_vl_strings(&file_data, &raw, 2, 8, 8).unwrap();
        assert_eq!(strings, vec!["Alice", "Bob"]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn read_vl_strings_from_seekable_source() {
        use std::io::Cursor;

        use crate::source::ReadSeekSource;

        let gcol_offset = 256usize;
        let mut file_data = vec![0u8; 512];
        build_gcol_at(&mut file_data, gcol_offset, &[(1, b"Alice"), (2, b"Bob")]);
        let raw = build_vl_refs(&["Alice", "Bob"], gcol_offset as u64, 1, 8);
        let source = ReadSeekSource::new(Cursor::new(file_data)).unwrap();

        let strings = read_vl_strings_from_source(
            &source,
            &raw,
            2,
            8,
            8,
            0,
            VlenStringReadOptions::default(),
        )
        .unwrap();
        assert_eq!(strings, vec!["Alice", "Bob"]);
    }

    #[test]
    fn null_vl_element_empty_string() {
        // length=0, address=undefined
        let mut raw = Vec::new();
        raw.extend_from_slice(&0u32.to_le_bytes()); // length=0
        raw.extend_from_slice(&u64::MAX.to_le_bytes()); // undefined address
        raw.extend_from_slice(&0u32.to_le_bytes()); // index

        let file_data = vec![0u8; 16];
        let strings = read_vl_strings(&file_data, &raw, 1, 8, 8).unwrap();
        assert_eq!(strings, vec![""]);
    }

    #[test]
    fn null_vl_element_zero_address() {
        let mut raw = Vec::new();
        raw.extend_from_slice(&0u32.to_le_bytes());
        raw.extend_from_slice(&0u64.to_le_bytes());
        raw.extend_from_slice(&0u32.to_le_bytes());

        let file_data = vec![0u8; 16];
        let strings = read_vl_strings(&file_data, &raw, 1, 8, 8).unwrap();
        assert_eq!(strings, vec![""]);
    }

    #[test]
    fn parse_vl_references_truncated_error() {
        let raw = vec![0u8; 10]; // too short for 1 element with offset_size=8
        let err = parse_vl_references(&raw, 1, 8).unwrap_err();
        assert!(matches!(err, FormatError::UnexpectedEof { .. }));
    }
}
