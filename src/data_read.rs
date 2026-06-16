//! Raw data reading and typed conversion for HDF5 datasets.

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec, vec::Vec};

use crate::chunk_cache::ChunkCache;
use crate::chunked_read::{
    read_chunked_data, read_chunked_data_cached, read_chunked_data_cached_from_source,
    read_chunked_data_from_source,
};
use crate::convert::{TryToUsize, slice_range};
use crate::data_layout::DataLayout;
use crate::dataspace::Dataspace;
use crate::datatype::{Datatype, DatatypeByteOrder};
use crate::error::FormatError;
use crate::filter_pipeline::FilterPipeline;
use crate::source::FileSource;

/// Read raw bytes for a dataset given its layout and the file data buffer,
/// using default filter/size parameters.
///
/// A test-only convenience wrapper over [`read_raw_data_full`]; the read paths
/// call `read_raw_data_full` directly with the real pipeline and offset sizes.
#[cfg(test)]
pub fn read_raw_data(
    file_data: &[u8],
    layout: &DataLayout,
    dataspace: &Dataspace,
    datatype: &Datatype,
) -> Result<Vec<u8>, FormatError> {
    read_raw_data_full(file_data, layout, dataspace, datatype, None, 8, 8)
}

/// Read raw bytes with full parameters including filter pipeline and sizes.
pub fn read_raw_data_full(
    file_data: &[u8],
    layout: &DataLayout,
    dataspace: &Dataspace,
    datatype: &Datatype,
    pipeline: Option<&FilterPipeline>,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<u8>, FormatError> {
    let num_elements = dataspace.num_elements().to_usize()?;
    let elem_size = datatype.type_size() as usize;
    let expected_size = num_elements
        .checked_mul(elem_size)
        .ok_or(FormatError::OffsetOverflow {
            offset: num_elements as u64,
            length: elem_size as u64,
        })?;

    // Zero-element datasets have no data to read.
    if num_elements == 0 {
        return Ok(Vec::new());
    }

    match layout {
        DataLayout::Compact { data } => {
            if data.len() != expected_size {
                return Err(FormatError::DataSizeMismatch {
                    expected: expected_size,
                    actual: data.len(),
                });
            }
            Ok(data.clone())
        }
        DataLayout::Contiguous { address, size } => {
            let addr = address.ok_or(FormatError::NoDataAllocated)?;
            let r = slice_range(addr, *size)?;
            let sz = r.end - r.start;
            if sz != expected_size {
                return Err(FormatError::DataSizeMismatch {
                    expected: expected_size,
                    actual: sz,
                });
            }
            if r.end > file_data.len() {
                return Err(FormatError::UnexpectedEof {
                    expected: r.end,
                    available: file_data.len(),
                });
            }
            Ok(file_data[r].to_vec())
        }
        DataLayout::Chunked { .. } => read_chunked_data(
            file_data,
            layout,
            dataspace,
            datatype,
            pipeline,
            offset_size,
            length_size,
        ),
        DataLayout::Virtual { .. } => Err(FormatError::UnsupportedVersion(0)),
    }
}

/// Read raw bytes with chunk cache support.
///
/// For chunked layouts the `cache` is used to avoid repeated B-tree
/// traversals and to cache decompressed chunk data.  For compact and
/// contiguous layouts this behaves identically to [`read_raw_data_full`].
pub fn read_raw_data_cached(
    file_data: &[u8],
    layout: &DataLayout,
    dataspace: &Dataspace,
    datatype: &Datatype,
    pipeline: Option<&FilterPipeline>,
    offset_size: u8,
    length_size: u8,
    cache: &ChunkCache,
) -> Result<Vec<u8>, FormatError> {
    match layout {
        DataLayout::Chunked { .. } => read_chunked_data_cached(
            file_data,
            layout,
            dataspace,
            datatype,
            pipeline,
            offset_size,
            length_size,
            cache,
        ),
        _ => read_raw_data_full(
            file_data,
            layout,
            dataspace,
            datatype,
            pipeline,
            offset_size,
            length_size,
        ),
    }
}

// ---------------------------------------------------------------------------
// Streaming variants (read from a `FileSource` instead of a whole-file buffer)
// ---------------------------------------------------------------------------
//
// These mirror `read_raw_data_full` / `read_raw_data` / `read_raw_data_cached`
// but fetch bytes on demand via `FileSource::read_exact_at`, so a 32-bit host
// can read a dataset out of a file larger than its address space. They always
// return an owned `Vec<u8>`: a lazy source cannot hand out a borrow, and the
// public API already returns owned data. The zero-copy `read_raw_data_zerocopy`
// stays the in-memory-only fast path and is intentionally not given a streaming
// variant.

/// Read raw bytes for a dataset from a [`FileSource`] (streaming counterpart of
/// [`read_raw_data_full`]).
pub fn read_raw_data_full_from_source<S: FileSource + ?Sized>(
    source: &S,
    layout: &DataLayout,
    dataspace: &Dataspace,
    datatype: &Datatype,
    pipeline: Option<&FilterPipeline>,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<u8>, FormatError> {
    let num_elements = dataspace.num_elements().to_usize()?;
    let elem_size = datatype.type_size() as usize;
    let expected_size = num_elements
        .checked_mul(elem_size)
        .ok_or(FormatError::OffsetOverflow {
            offset: num_elements as u64,
            length: elem_size as u64,
        })?;

    if num_elements == 0 {
        return Ok(Vec::new());
    }

    match layout {
        DataLayout::Compact { data } => {
            if data.len() != expected_size {
                return Err(FormatError::DataSizeMismatch {
                    expected: expected_size,
                    actual: data.len(),
                });
            }
            Ok(data.clone())
        }
        DataLayout::Contiguous { address, size } => {
            let addr = address.ok_or(FormatError::NoDataAllocated)?;
            let sz = (*size).to_usize()?;
            if sz != expected_size {
                return Err(FormatError::DataSizeMismatch {
                    expected: expected_size,
                    actual: sz,
                });
            }
            // The single point of I/O; `read_exact_at` bounds-checks against the
            // source length (in u64) and errors instead of truncating.
            source.read_exact_at(addr, sz)
        }
        DataLayout::Chunked { .. } => read_chunked_data_from_source(
            source,
            layout,
            dataspace,
            datatype,
            pipeline,
            offset_size,
            length_size,
        ),
        DataLayout::Virtual { .. } => Err(FormatError::UnsupportedVersion(0)),
    }
}

/// Streaming counterpart of [`read_raw_data_cached`].
#[allow(clippy::too_many_arguments)]
pub fn read_raw_data_cached_from_source<S: FileSource + ?Sized>(
    source: &S,
    layout: &DataLayout,
    dataspace: &Dataspace,
    datatype: &Datatype,
    pipeline: Option<&FilterPipeline>,
    offset_size: u8,
    length_size: u8,
    cache: &ChunkCache,
) -> Result<Vec<u8>, FormatError> {
    match layout {
        DataLayout::Chunked { .. } => read_chunked_data_cached_from_source(
            source,
            layout,
            dataspace,
            datatype,
            pipeline,
            offset_size,
            length_size,
            cache,
        ),
        _ => read_raw_data_full_from_source(
            source,
            layout,
            dataspace,
            datatype,
            pipeline,
            offset_size,
            length_size,
        ),
    }
}

fn datatype_name(dt: &Datatype) -> &'static str {
    match dt {
        Datatype::FixedPoint { .. } => "FixedPoint",
        Datatype::FloatingPoint { .. } => "FloatingPoint",
        Datatype::String { .. } => "String",
        Datatype::Time { .. } => "Time",
        Datatype::BitField { .. } => "BitField",
        Datatype::Opaque { .. } => "Opaque",
        Datatype::Compound { .. } => "Compound",
        Datatype::Reference { .. } => "Reference",
        Datatype::Enumeration { .. } => "Enumeration",
        Datatype::VariableLength { .. } => "VariableLength",
        Datatype::Array { .. } => "Array",
    }
}

fn ensure_numeric(dt: &Datatype, expected: &'static str) -> Result<(), FormatError> {
    match dt {
        Datatype::FixedPoint { .. } | Datatype::FloatingPoint { .. } => Ok(()),
        _ => Err(FormatError::TypeMismatch {
            expected,
            actual: datatype_name(dt),
        }),
    }
}

fn get_byte_order(dt: &Datatype) -> DatatypeByteOrder {
    match dt {
        Datatype::FixedPoint { byte_order, .. } => byte_order.clone(),
        Datatype::FloatingPoint { byte_order, .. } => byte_order.clone(),
        _ => DatatypeByteOrder::LittleEndian,
    }
}

fn get_size(dt: &Datatype) -> usize {
    dt.type_size() as usize
}

/// Convert raw bytes to `f64` values.
pub fn read_as_f64(raw: &[u8], datatype: &Datatype) -> Result<Vec<f64>, FormatError> {
    ensure_numeric(datatype, "FloatingPoint or FixedPoint")?;
    let elem_size = get_size(datatype);
    if elem_size == 0 || !raw.len().is_multiple_of(elem_size) {
        return Err(FormatError::DataSizeMismatch {
            expected: 0,
            actual: raw.len(),
        });
    }
    let count = raw.len() / elem_size;

    // Fast path: native-endian f64 can use bulk copy (zero conversion overhead)
    #[cfg(target_endian = "little")]
    if matches!(
        datatype,
        Datatype::FloatingPoint {
            size: 8,
            byte_order: DatatypeByteOrder::LittleEndian,
            ..
        }
    ) {
        let mut result = vec![0.0f64; count];
        // Safety equivalent via from_le_bytes — but we use safe transmute-free copy
        for (i, val) in result.iter_mut().enumerate() {
            let off = i * 8;
            *val = f64::from_le_bytes([
                raw[off],
                raw[off + 1],
                raw[off + 2],
                raw[off + 3],
                raw[off + 4],
                raw[off + 5],
                raw[off + 6],
                raw[off + 7],
            ]);
        }
        return Ok(result);
    }

    let order = get_byte_order(datatype);
    let mut result = Vec::with_capacity(count);

    for i in 0..count {
        let chunk = &raw[i * elem_size..(i + 1) * elem_size];
        let val = convert_to_f64(chunk, datatype, &order)?;
        result.push(val);
    }
    Ok(result)
}

fn convert_to_f64(
    bytes: &[u8],
    dt: &Datatype,
    order: &DatatypeByteOrder,
) -> Result<f64, FormatError> {
    match dt {
        Datatype::FloatingPoint { size, .. } => match size {
            4 => {
                let v = read_f32_bytes(bytes, order);
                Ok(v as f64)
            }
            8 => Ok(read_f64_bytes(bytes, order)),
            _ => Err(FormatError::DataSizeMismatch {
                expected: 8,
                actual: *size as usize,
            }),
        },
        Datatype::FixedPoint { size, signed, .. } => {
            if *signed {
                let v = read_signed_int(bytes, *size as usize, order);
                Ok(v as f64)
            } else {
                let v = read_unsigned_int(bytes, *size as usize, order);
                Ok(v as f64)
            }
        }
        _ => Err(FormatError::TypeMismatch {
            expected: "numeric",
            actual: datatype_name(dt),
        }),
    }
}

/// Convert raw bytes to `i64` values.
pub fn read_as_i64(raw: &[u8], datatype: &Datatype) -> Result<Vec<i64>, FormatError> {
    ensure_numeric(datatype, "FixedPoint (signed)")?;
    let elem_size = get_size(datatype);
    if elem_size == 0 || !raw.len().is_multiple_of(elem_size) {
        return Err(FormatError::DataSizeMismatch {
            expected: 0,
            actual: raw.len(),
        });
    }
    let count = raw.len() / elem_size;
    let order = get_byte_order(datatype);
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let chunk = &raw[i * elem_size..(i + 1) * elem_size];
        let v = read_signed_int(chunk, elem_size, &order);
        result.push(v);
    }
    Ok(result)
}

/// Convert raw bytes to `u64` values.
pub fn read_as_u64(raw: &[u8], datatype: &Datatype) -> Result<Vec<u64>, FormatError> {
    ensure_numeric(datatype, "FixedPoint (unsigned)")?;
    let elem_size = get_size(datatype);
    if elem_size == 0 || !raw.len().is_multiple_of(elem_size) {
        return Err(FormatError::DataSizeMismatch {
            expected: 0,
            actual: raw.len(),
        });
    }
    let count = raw.len() / elem_size;
    let order = get_byte_order(datatype);
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let chunk = &raw[i * elem_size..(i + 1) * elem_size];
        let v = read_unsigned_int(chunk, elem_size, &order);
        result.push(v);
    }
    Ok(result)
}

/// Convert raw bytes to `f32` values.
pub fn read_as_f32(raw: &[u8], datatype: &Datatype) -> Result<Vec<f32>, FormatError> {
    ensure_numeric(datatype, "FloatingPoint")?;
    let elem_size = get_size(datatype);
    if elem_size == 0 || !raw.len().is_multiple_of(elem_size) {
        return Err(FormatError::DataSizeMismatch {
            expected: 0,
            actual: raw.len(),
        });
    }
    let count = raw.len() / elem_size;
    let order = get_byte_order(datatype);
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let chunk = &raw[i * elem_size..(i + 1) * elem_size];
        match datatype {
            Datatype::FloatingPoint { size: 4, .. } => {
                result.push(read_f32_bytes(chunk, &order));
            }
            Datatype::FloatingPoint { size: 8, .. } => {
                result.push(read_f64_bytes(chunk, &order) as f32);
            }
            Datatype::FixedPoint {
                signed: true, size, ..
            } => {
                result.push(read_signed_int(chunk, *size as usize, &order) as f32);
            }
            Datatype::FixedPoint {
                signed: false,
                size,
                ..
            } => {
                result.push(read_unsigned_int(chunk, *size as usize, &order) as f32);
            }
            _ => {
                return Err(FormatError::TypeMismatch {
                    expected: "numeric",
                    actual: datatype_name(datatype),
                });
            }
        }
    }
    Ok(result)
}

/// Convert raw bytes to `i32` values.
pub fn read_as_i32(raw: &[u8], datatype: &Datatype) -> Result<Vec<i32>, FormatError> {
    ensure_numeric(datatype, "FixedPoint")?;
    let elem_size = get_size(datatype);
    if elem_size == 0 || !raw.len().is_multiple_of(elem_size) {
        return Err(FormatError::DataSizeMismatch {
            expected: 0,
            actual: raw.len(),
        });
    }
    let count = raw.len() / elem_size;
    let order = get_byte_order(datatype);
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let chunk = &raw[i * elem_size..(i + 1) * elem_size];
        let v = read_signed_int(chunk, elem_size, &order);
        result.push(v as i32);
    }
    Ok(result)
}

/// Read fixed-length strings from raw bytes.
pub fn read_as_strings(raw: &[u8], datatype: &Datatype) -> Result<Vec<String>, FormatError> {
    match datatype {
        Datatype::String { size, padding, .. } => {
            let elem_size = *size as usize;
            if elem_size == 0 {
                return Ok(Vec::new());
            }
            if !raw.len().is_multiple_of(elem_size) {
                return Err(FormatError::DataSizeMismatch {
                    expected: 0,
                    actual: raw.len(),
                });
            }
            let count = raw.len() / elem_size;
            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let chunk = &raw[i * elem_size..(i + 1) * elem_size];
                let s = match padding {
                    crate::datatype::StringPadding::NullTerminate => {
                        let end = chunk.iter().position(|&b| b == 0).unwrap_or(chunk.len());
                        String::from_utf8_lossy(&chunk[..end]).into_owned()
                    }
                    crate::datatype::StringPadding::NullPad => {
                        let end = chunk.iter().rposition(|&b| b != 0).map_or(0, |p| p + 1);
                        String::from_utf8_lossy(&chunk[..end]).into_owned()
                    }
                    crate::datatype::StringPadding::SpacePad => {
                        let end = chunk.iter().rposition(|&b| b != b' ').map_or(0, |p| p + 1);
                        String::from_utf8_lossy(&chunk[..end]).into_owned()
                    }
                };
                result.push(s);
            }
            Ok(result)
        }
        _ => Err(FormatError::TypeMismatch {
            expected: "String",
            actual: datatype_name(datatype),
        }),
    }
}

// --- Low-level byte conversion helpers ---

fn reorder_bytes(bytes: &[u8], order: &DatatypeByteOrder) -> [u8; 8] {
    let mut buf = [0u8; 8];
    let len = bytes.len().min(8);
    match order {
        DatatypeByteOrder::LittleEndian | DatatypeByteOrder::Vax => {
            buf[..len].copy_from_slice(&bytes[..len]);
        }
        DatatypeByteOrder::BigEndian => {
            // Reverse bytes into LE order
            for i in 0..len {
                buf[i] = bytes[len - 1 - i];
            }
        }
    }
    buf
}

fn read_f64_bytes(bytes: &[u8], order: &DatatypeByteOrder) -> f64 {
    let buf = reorder_bytes(bytes, order);
    f64::from_le_bytes(buf)
}

fn read_f32_bytes(bytes: &[u8], order: &DatatypeByteOrder) -> f32 {
    let mut buf = [0u8; 4];
    let len = bytes.len().min(4);
    match order {
        DatatypeByteOrder::LittleEndian | DatatypeByteOrder::Vax => {
            buf[..len].copy_from_slice(&bytes[..len]);
        }
        DatatypeByteOrder::BigEndian => {
            for i in 0..len {
                buf[i] = bytes[len - 1 - i];
            }
        }
    }
    f32::from_le_bytes(buf)
}

fn read_unsigned_int(bytes: &[u8], size: usize, order: &DatatypeByteOrder) -> u64 {
    let buf = reorder_bytes(bytes, order);
    match size {
        1 => buf[0] as u64,
        2 => u16::from_le_bytes([buf[0], buf[1]]) as u64,
        4 => u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as u64,
        8 => u64::from_le_bytes(buf),
        _ => {
            // Generic: read as LE
            let mut val = 0u64;
            for (i, &byte) in buf.iter().enumerate().take(size.min(8)) {
                val |= (byte as u64) << (i * 8);
            }
            val
        }
    }
}

fn read_signed_int(bytes: &[u8], size: usize, order: &DatatypeByteOrder) -> i64 {
    let buf = reorder_bytes(bytes, order);
    match size {
        1 => buf[0] as i8 as i64,
        2 => i16::from_le_bytes([buf[0], buf[1]]) as i64,
        4 => i32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as i64,
        8 => i64::from_le_bytes(buf),
        _ => {
            let u = read_unsigned_int(bytes, size, order);
            // Sign extend
            let shift = 64 - (size * 8);
            ((u as i64) << shift) >> shift
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataspace::{Dataspace, DataspaceType};
    use crate::datatype::{CharacterSet, StringPadding};

    fn make_f64_le_type() -> Datatype {
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

    fn make_f32_be_type() -> Datatype {
        Datatype::FloatingPoint {
            size: 4,
            byte_order: DatatypeByteOrder::BigEndian,
            bit_offset: 0,
            bit_precision: 32,
            exponent_location: 23,
            exponent_size: 8,
            mantissa_location: 0,
            mantissa_size: 23,
            exponent_bias: 127,
        }
    }

    fn make_i32_le_type() -> Datatype {
        Datatype::FixedPoint {
            size: 4,
            byte_order: DatatypeByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 32,
        }
    }

    fn make_i16_le_type() -> Datatype {
        Datatype::FixedPoint {
            size: 2,
            byte_order: DatatypeByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 16,
        }
    }

    fn make_u8_type() -> Datatype {
        Datatype::FixedPoint {
            size: 1,
            byte_order: DatatypeByteOrder::LittleEndian,
            signed: false,
            bit_offset: 0,
            bit_precision: 8,
        }
    }

    fn make_simple_dataspace(dims: &[u64]) -> Dataspace {
        Dataspace {
            space_type: DataspaceType::Simple,
            rank: dims.len() as u8,
            dimensions: dims.to_vec(),
            max_dimensions: None,
        }
    }

    #[test]
    fn read_f64_compact() {
        let dt = make_f64_le_type();
        let ds = make_simple_dataspace(&[3]);
        let mut data = Vec::new();
        data.extend_from_slice(&1.0f64.to_le_bytes());
        data.extend_from_slice(&2.0f64.to_le_bytes());
        data.extend_from_slice(&3.0f64.to_le_bytes());
        let layout = DataLayout::Compact { data: data.clone() };
        let raw = read_raw_data(&[], &layout, &ds, &dt).unwrap();
        assert_eq!(raw, data);
        let values = read_as_f64(&raw, &dt).unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn read_i32_contiguous() {
        let dt = make_i32_le_type();
        let ds = make_simple_dataspace(&[4]);
        let mut file_data = vec![0u8; 1024];
        let offset = 256usize;
        let vals: Vec<i32> = vec![10, -20, 30, -40];
        for (i, v) in vals.iter().enumerate() {
            let bytes = v.to_le_bytes();
            file_data[offset + i * 4..offset + i * 4 + 4].copy_from_slice(&bytes);
        }
        let layout = DataLayout::Contiguous {
            address: Some(offset as u64),
            size: 16,
        };
        let raw = read_raw_data(&file_data, &layout, &ds, &dt).unwrap();
        let result = read_as_i32(&raw, &dt).unwrap();
        assert_eq!(result, vec![10, -20, 30, -40]);
    }

    #[test]
    fn read_u8_data() {
        let dt = make_u8_type();
        let ds = make_simple_dataspace(&[5]);
        let data = vec![10u8, 20, 30, 40, 50];
        let layout = DataLayout::Compact { data: data.clone() };
        let raw = read_raw_data(&[], &layout, &ds, &dt).unwrap();
        let result = read_as_u64(&raw, &dt).unwrap();
        assert_eq!(result, vec![10, 20, 30, 40, 50]);
    }

    #[test]
    fn read_f32_be() {
        let dt = make_f32_be_type();
        let ds = make_simple_dataspace(&[2]);
        let mut data = Vec::new();
        // Store as big-endian
        data.extend_from_slice(&1.5f32.to_be_bytes());
        data.extend_from_slice(&2.5f32.to_be_bytes());
        let layout = DataLayout::Compact { data: data.clone() };
        let raw = read_raw_data(&[], &layout, &ds, &dt).unwrap();
        let result = read_as_f32(&raw, &dt).unwrap();
        assert_eq!(result, vec![1.5, 2.5]);
    }

    #[test]
    fn read_i16_le() {
        let dt = make_i16_le_type();
        let ds = make_simple_dataspace(&[3]);
        let mut data = Vec::new();
        data.extend_from_slice(&(-100i16).to_le_bytes());
        data.extend_from_slice(&200i16.to_le_bytes());
        data.extend_from_slice(&(-300i16).to_le_bytes());
        let layout = DataLayout::Compact { data: data.clone() };
        let raw = read_raw_data(&[], &layout, &ds, &dt).unwrap();
        let result = read_as_i64(&raw, &dt).unwrap();
        assert_eq!(result, vec![-100, 200, -300]);
    }

    #[test]
    fn read_strings_compact() {
        let dt = Datatype::String {
            size: 5,
            padding: StringPadding::NullPad,
            charset: CharacterSet::Ascii,
        };
        let ds = make_simple_dataspace(&[2]);
        let mut data = Vec::new();
        data.extend_from_slice(b"hello");
        data.extend_from_slice(b"hi\0\0\0");
        let layout = DataLayout::Compact { data: data.clone() };
        let raw = read_raw_data(&[], &layout, &ds, &dt).unwrap();
        let result = read_as_strings(&raw, &dt).unwrap();
        assert_eq!(result, vec!["hello", "hi"]);
    }

    #[test]
    fn type_mismatch_f64_on_string() {
        let dt = Datatype::String {
            size: 4,
            padding: StringPadding::NullTerminate,
            charset: CharacterSet::Ascii,
        };
        let raw = vec![0u8; 8];
        let err = read_as_f64(&raw, &dt).unwrap_err();
        assert!(matches!(err, FormatError::TypeMismatch { .. }));
    }

    #[test]
    fn size_mismatch_compact() {
        let dt = make_f64_le_type();
        let ds = make_simple_dataspace(&[3]);
        let data = vec![0u8; 16]; // wrong: should be 24
        let layout = DataLayout::Compact { data };
        let err = read_raw_data(&[], &layout, &ds, &dt).unwrap_err();
        assert!(matches!(err, FormatError::DataSizeMismatch { .. }));
    }

    #[test]
    fn no_data_allocated() {
        let dt = make_f64_le_type();
        let ds = make_simple_dataspace(&[3]);
        let layout = DataLayout::Contiguous {
            address: None,
            size: 24,
        };
        let err = read_raw_data(&[], &layout, &ds, &dt).unwrap_err();
        assert!(matches!(err, FormatError::NoDataAllocated));
    }

    #[test]
    fn string_type_mismatch_on_read_as_strings() {
        let dt = make_i32_le_type();
        let raw = vec![0u8; 8];
        let err = read_as_strings(&raw, &dt).unwrap_err();
        assert!(matches!(err, FormatError::TypeMismatch { .. }));
    }

    #[test]
    fn read_f64_from_i32() {
        // read_as_f64 should work on FixedPoint types too
        let dt = make_i32_le_type();
        let mut raw = Vec::new();
        raw.extend_from_slice(&42i32.to_le_bytes());
        raw.extend_from_slice(&(-7i32).to_le_bytes());
        let result = read_as_f64(&raw, &dt).unwrap();
        assert_eq!(result, vec![42.0, -7.0]);
    }

    #[test]
    fn read_strings_space_padded() {
        let dt = Datatype::String {
            size: 8,
            padding: StringPadding::SpacePad,
            charset: CharacterSet::Ascii,
        };
        let raw = b"hello   world   ";
        let result = read_as_strings(raw, &dt).unwrap();
        assert_eq!(result, vec!["hello", "world"]);
    }

    #[test]
    fn read_strings_null_terminated() {
        let dt = Datatype::String {
            size: 6,
            padding: StringPadding::NullTerminate,
            charset: CharacterSet::Ascii,
        };
        let raw = b"abc\0\0\0de\0\0\0\0";
        let result = read_as_strings(raw, &dt).unwrap();
        assert_eq!(result, vec!["abc", "de"]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn streaming_contiguous_matches_buffered() {
        use crate::source::{BytesSource, ReadSeekSource};
        let dt = make_f64_le_type();
        let ds = make_simple_dataspace(&[3]);
        let mut file_data = vec![0u8; 1024];
        let offset = 256usize;
        for (i, v) in [1.0f64, 2.0, 3.0].iter().enumerate() {
            file_data[offset + i * 8..offset + i * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
        let layout = DataLayout::Contiguous {
            address: Some(offset as u64),
            size: 24,
        };

        let buffered = read_raw_data_full(&file_data, &layout, &ds, &dt, None, 8, 8).unwrap();
        let from_mem = read_raw_data_full_from_source(
            &BytesSource::new(&file_data),
            &layout,
            &ds,
            &dt,
            None,
            8,
            8,
        )
        .unwrap();
        let from_seek = read_raw_data_full_from_source(
            &ReadSeekSource::new(std::io::Cursor::new(file_data)).unwrap(),
            &layout,
            &ds,
            &dt,
            None,
            8,
            8,
        )
        .unwrap();

        assert_eq!(buffered, from_mem);
        assert_eq!(buffered, from_seek);
        assert_eq!(read_as_f64(&from_seek, &dt).unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn streaming_compact_matches_buffered() {
        use crate::source::BytesSource;
        let dt = make_f64_le_type();
        let ds = make_simple_dataspace(&[2]);
        let mut data = Vec::new();
        for v in [7.0f64, 8.0] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let layout = DataLayout::Compact { data };
        let buffered = read_raw_data_full(&[], &layout, &ds, &dt, None, 8, 8).unwrap();
        let streamed = read_raw_data_full_from_source(
            &BytesSource::new(Vec::new()),
            &layout,
            &ds,
            &dt,
            None,
            8,
            8,
        )
        .unwrap();
        assert_eq!(buffered, streamed);
    }

    #[cfg(feature = "std")]
    #[test]
    fn streaming_contiguous_error_parity() {
        use crate::source::BytesSource;
        let dt = make_f64_le_type();
        let ds = make_simple_dataspace(&[3]);
        let layout = DataLayout::Contiguous {
            address: None,
            size: 24,
        };
        let buffered = read_raw_data_full(&[], &layout, &ds, &dt, None, 8, 8);
        let streamed = read_raw_data_full_from_source(
            &BytesSource::new(Vec::new()),
            &layout,
            &ds,
            &dt,
            None,
            8,
            8,
        );
        assert!(matches!(buffered, Err(FormatError::NoDataAllocated)));
        assert!(matches!(streamed, Err(FormatError::NoDataAllocated)));
    }
}
