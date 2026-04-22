//! Read an HDF5 file with MATLAB conventions into a `MatValue` tree.

use std::collections::HashMap;

use crate::file_writer::AttrValue;
use crate::mat::class::MatClass;
use crate::mat::error::MatError;
use crate::mat::utf16;
use crate::mat::value::{MatValue, NumVec, ScalarNum};
use crate::reader::{Dataset, File, Group};
use crate::types::DType;

/// Parse a MAT v7.3 file into an ordered list of `(name, value)` fields
/// rooted at the file's top level.
pub(crate) fn read_file(bytes: &[u8]) -> Result<Vec<(String, MatValue)>, MatError> {
    // Accept raw HDF5 too — we don't require the MATLAB signature to be
    // present to deserialize successfully.
    let file = File::from_bytes(bytes.to_vec()).map_err(MatError::Hdf5)?;
    let root = file.root();
    read_group(&root)
}

fn read_group(group: &Group<'_>) -> Result<Vec<(String, MatValue)>, MatError> {
    let mut out = Vec::new();

    // Collect datasets first, then subgroups. Preserve iteration order from
    // the HDF5 link table.
    for name in group.datasets().map_err(MatError::Hdf5)? {
        let ds = group.dataset(&name).map_err(MatError::Hdf5)?;
        out.push((name, read_dataset(&ds)?));
    }
    for name in group.groups().map_err(MatError::Hdf5)? {
        let sub = group.group(&name).map_err(MatError::Hdf5)?;
        out.push((name, read_group_as_value(&sub)?));
    }
    Ok(out)
}

fn read_group_as_value(group: &Group<'_>) -> Result<MatValue, MatError> {
    let fields = read_group(group)?;
    Ok(MatValue::Struct(fields))
}

/// Read a single dataset into a `MatValue` using its MATLAB_class attribute
/// (if present) and its HDF5 shape.
fn read_dataset(ds: &Dataset<'_>) -> Result<MatValue, MatError> {
    let attrs = ds.attrs().map_err(MatError::Hdf5)?;
    let class = matlab_class_from_attrs(&attrs)?;
    let shape = ds.shape().map_err(MatError::Hdf5)?;
    let dtype = ds.dtype().map_err(MatError::Hdf5)?;
    let is_empty = is_empty_attr(&attrs) || shape.iter().any(|&d| d == 0);

    let class = class.unwrap_or_else(|| class_from_dtype(&dtype));

    if is_empty {
        // Empty numeric/char: produce an empty 1-D vec of the correct tag.
        return Ok(empty_value_for_class(class));
    }

    match class {
        MatClass::Char => {
            let units = ds.read_u16().map_err(MatError::Hdf5)?;
            let s = utf16::decode_utf16(&units)?;
            Ok(MatValue::String(s))
        }
        MatClass::Logical => read_numeric(ds, &shape, class),
        MatClass::Double
        | MatClass::Single
        | MatClass::Int8
        | MatClass::Int16
        | MatClass::Int32
        | MatClass::Int64
        | MatClass::UInt8
        | MatClass::UInt16
        | MatClass::UInt32
        | MatClass::UInt64 => {
            if is_complex_dtype(&dtype) {
                read_complex(ds, &shape, class)
            } else {
                read_numeric(ds, &shape, class)
            }
        }
        MatClass::Struct => Err(MatError::Custom(
            "dataset has MATLAB_class='struct'; expected a group".into(),
        )),
        MatClass::Cell => Err(MatError::UnsupportedType("cell array")),
    }
}

/// Extract and parse the `MATLAB_class` attribute value, if present.
fn matlab_class_from_attrs(
    attrs: &HashMap<String, AttrValue>,
) -> Result<Option<MatClass>, MatError> {
    let raw = match attrs.get("MATLAB_class") {
        Some(AttrValue::AsciiString(s)) | Some(AttrValue::String(s)) => Some(s.clone()),
        Some(AttrValue::StringArray(v)) if v.len() == 1 => Some(v[0].clone()),
        None => None,
        other => {
            return Err(MatError::Custom(format!(
                "MATLAB_class attribute has unexpected type: {other:?}"
            )))
        }
    };
    match raw {
        Some(s) => Ok(Some(MatClass::parse(&s)?)),
        None => Ok(None),
    }
}

fn is_empty_attr(attrs: &HashMap<String, AttrValue>) -> bool {
    match attrs.get("MATLAB_empty") {
        Some(AttrValue::U32(v)) => *v != 0,
        Some(AttrValue::U64(v)) => *v != 0,
        Some(AttrValue::I64(v)) => *v != 0,
        Some(AttrValue::I32(v)) => *v != 0,
        _ => false,
    }
}

fn empty_value_for_class(class: MatClass) -> MatValue {
    use crate::mat::value::ScalarTag;
    match class {
        MatClass::Char => MatValue::String(String::new()),
        MatClass::Logical => MatValue::Vec1D(NumVec::empty_with_tag(ScalarTag::Bool)),
        MatClass::Double => MatValue::Vec1D(NumVec::empty_with_tag(ScalarTag::F64)),
        MatClass::Single => MatValue::Vec1D(NumVec::empty_with_tag(ScalarTag::F32)),
        MatClass::Int8 => MatValue::Vec1D(NumVec::empty_with_tag(ScalarTag::I8)),
        MatClass::Int16 => MatValue::Vec1D(NumVec::empty_with_tag(ScalarTag::I16)),
        MatClass::Int32 => MatValue::Vec1D(NumVec::empty_with_tag(ScalarTag::I32)),
        MatClass::Int64 => MatValue::Vec1D(NumVec::empty_with_tag(ScalarTag::I64)),
        MatClass::UInt8 => MatValue::Vec1D(NumVec::empty_with_tag(ScalarTag::U8)),
        MatClass::UInt16 => MatValue::Vec1D(NumVec::empty_with_tag(ScalarTag::U16)),
        MatClass::UInt32 => MatValue::Vec1D(NumVec::empty_with_tag(ScalarTag::U32)),
        MatClass::UInt64 => MatValue::Vec1D(NumVec::empty_with_tag(ScalarTag::U64)),
        MatClass::Struct => MatValue::Struct(Vec::new()),
        MatClass::Cell => MatValue::Struct(Vec::new()),
    }
}

/// Infer a `MatClass` from the raw `DType` when no `MATLAB_class` attribute
/// is present (so non-MAT files still work).
fn class_from_dtype(dtype: &DType) -> MatClass {
    match dtype {
        DType::F64 => MatClass::Double,
        DType::F32 => MatClass::Single,
        DType::I8 => MatClass::Int8,
        DType::I16 => MatClass::Int16,
        DType::I32 => MatClass::Int32,
        DType::I64 => MatClass::Int64,
        DType::U8 => MatClass::UInt8,
        DType::U16 => MatClass::UInt16,
        DType::U32 => MatClass::UInt32,
        DType::U64 => MatClass::UInt64,
        DType::String => MatClass::Char,
        DType::VariableLengthString => MatClass::Char,
        _ => MatClass::Double, // fallback guess
    }
}

fn is_complex_dtype(dtype: &DType) -> bool {
    match dtype {
        DType::Compound(fields) => {
            fields.len() == 2
                && fields.iter().any(|(n, _)| n == "real")
                && fields.iter().any(|(n, _)| n == "imag")
        }
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Numeric reading
// ---------------------------------------------------------------------------

fn read_numeric(
    ds: &Dataset<'_>,
    shape: &[u64],
    class: MatClass,
) -> Result<MatValue, MatError> {
    let (rows, cols, total) = shape_decomposition(shape);

    // For a single-element dataset we emit a Scalar of the appropriate class.
    if total == 1 {
        return Ok(MatValue::Scalar(read_scalar(ds, class)?));
    }

    // Read all elements as the MATLAB class's native type.
    let flat = read_all_elements(ds, class)?;

    if rows == 1 || cols == 1 {
        return Ok(MatValue::Vec1D(flat));
    }

    // 2-D matrix: the stored bytes are column-major for a MATLAB [rows, cols]
    // matrix. Transpose into row-major for Rust.
    let matrix = transpose_col_major_to_row_major(flat, rows, cols)?;
    Ok(MatValue::Matrix {
        rows,
        cols,
        vec: matrix,
    })
}

fn shape_decomposition(shape: &[u64]) -> (usize, usize, usize) {
    // HDF5 shape for MATLAB data is [cols, rows] (column-major storage of a
    // [rows, cols] matrix). For a 1-D variant like [1, N] or [N, 1], treat as
    // a vector of length N.
    match shape.len() {
        0 => (1, 1, 1),
        1 => (1, shape[0] as usize, shape[0] as usize),
        2 => {
            let cols_hdf5 = shape[0] as usize;
            let rows_hdf5 = shape[1] as usize;
            // MATLAB matrix has rows = rows_hdf5, cols = cols_hdf5.
            let total = cols_hdf5 * rows_hdf5;
            (rows_hdf5, cols_hdf5, total)
        }
        _ => {
            let total: usize = shape.iter().map(|&d| d as usize).product();
            (1, total, total)
        }
    }
}

fn read_all_elements(ds: &Dataset<'_>, class: MatClass) -> Result<NumVec, MatError> {
    Ok(match class {
        MatClass::Double => NumVec::F64(ds.read_f64().map_err(MatError::Hdf5)?),
        MatClass::Single => NumVec::F32(ds.read_f32().map_err(MatError::Hdf5)?),
        MatClass::Int8 => NumVec::I8(ds.read_i8().map_err(MatError::Hdf5)?),
        MatClass::Int16 => NumVec::I16(ds.read_i16().map_err(MatError::Hdf5)?),
        MatClass::Int32 => NumVec::I32(ds.read_i32().map_err(MatError::Hdf5)?),
        MatClass::Int64 => NumVec::I64(ds.read_i64().map_err(MatError::Hdf5)?),
        MatClass::UInt8 => NumVec::U8(ds.read_u8().map_err(MatError::Hdf5)?),
        MatClass::UInt16 => NumVec::U16(ds.read_u16().map_err(MatError::Hdf5)?),
        MatClass::UInt32 => NumVec::U32(ds.read_u32().map_err(MatError::Hdf5)?),
        MatClass::UInt64 => NumVec::U64(ds.read_u64().map_err(MatError::Hdf5)?),
        MatClass::Logical => {
            let bytes = ds.read_u8().map_err(MatError::Hdf5)?;
            NumVec::Bool(bytes.into_iter().map(|b| b != 0).collect())
        }
        _ => return Err(MatError::Custom(format!("read_numeric: class {class:?}"))),
    })
}

fn read_scalar(ds: &Dataset<'_>, class: MatClass) -> Result<ScalarNum, MatError> {
    Ok(match class {
        MatClass::Double => ScalarNum::F64(ds.read_f64().map_err(MatError::Hdf5)?[0]),
        MatClass::Single => ScalarNum::F32(ds.read_f32().map_err(MatError::Hdf5)?[0]),
        MatClass::Int8 => ScalarNum::I8(ds.read_i8().map_err(MatError::Hdf5)?[0]),
        MatClass::Int16 => ScalarNum::I16(ds.read_i16().map_err(MatError::Hdf5)?[0]),
        MatClass::Int32 => ScalarNum::I32(ds.read_i32().map_err(MatError::Hdf5)?[0]),
        MatClass::Int64 => ScalarNum::I64(ds.read_i64().map_err(MatError::Hdf5)?[0]),
        MatClass::UInt8 => ScalarNum::U8(ds.read_u8().map_err(MatError::Hdf5)?[0]),
        MatClass::UInt16 => ScalarNum::U16(ds.read_u16().map_err(MatError::Hdf5)?[0]),
        MatClass::UInt32 => ScalarNum::U32(ds.read_u32().map_err(MatError::Hdf5)?[0]),
        MatClass::UInt64 => ScalarNum::U64(ds.read_u64().map_err(MatError::Hdf5)?[0]),
        MatClass::Logical => ScalarNum::Bool(ds.read_u8().map_err(MatError::Hdf5)?[0] != 0),
        _ => return Err(MatError::Custom(format!("read_scalar: class {class:?}"))),
    })
}

fn transpose_col_major_to_row_major(
    col_major: NumVec,
    rows: usize,
    cols: usize,
) -> Result<NumVec, MatError> {
    debug_assert_eq!(col_major.len(), rows * cols);

    fn transpose<T: Copy>(v: Vec<T>, rows: usize, cols: usize) -> Vec<T> {
        let mut out = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                out.push(v[c * rows + r]);
            }
        }
        out
    }

    Ok(match col_major {
        NumVec::F64(v) => NumVec::F64(transpose(v, rows, cols)),
        NumVec::F32(v) => NumVec::F32(transpose(v, rows, cols)),
        NumVec::I8(v) => NumVec::I8(transpose(v, rows, cols)),
        NumVec::I16(v) => NumVec::I16(transpose(v, rows, cols)),
        NumVec::I32(v) => NumVec::I32(transpose(v, rows, cols)),
        NumVec::I64(v) => NumVec::I64(transpose(v, rows, cols)),
        NumVec::U8(v) => NumVec::U8(transpose(v, rows, cols)),
        NumVec::U16(v) => NumVec::U16(transpose(v, rows, cols)),
        NumVec::U32(v) => NumVec::U32(transpose(v, rows, cols)),
        NumVec::U64(v) => NumVec::U64(transpose(v, rows, cols)),
        NumVec::Bool(v) => NumVec::Bool(transpose(v, rows, cols)),
    })
}

// ---------------------------------------------------------------------------
// Complex reading
// ---------------------------------------------------------------------------

fn read_complex(
    ds: &Dataset<'_>,
    shape: &[u64],
    class: MatClass,
) -> Result<MatValue, MatError> {
    let (rows, cols, total) = shape_decomposition(shape);
    let bytes = ds.read_u8().map_err(MatError::Hdf5)?;

    match class {
        MatClass::Double => {
            let pairs = parse_complex64_pairs(&bytes, total)?;
            if total == 1 {
                let (re, im) = pairs[0];
                Ok(MatValue::ComplexScalar64 { re, im })
            } else if rows == 1 || cols == 1 {
                Ok(MatValue::ComplexVec64(pairs))
            } else {
                let row_major = transpose_pairs_col_to_row(pairs, rows, cols);
                Ok(MatValue::ComplexMatrix64 {
                    rows,
                    cols,
                    pairs: row_major,
                })
            }
        }
        MatClass::Single => {
            let pairs = parse_complex32_pairs(&bytes, total)?;
            if total == 1 {
                let (re, im) = pairs[0];
                Ok(MatValue::ComplexScalar32 { re, im })
            } else if rows == 1 || cols == 1 {
                Ok(MatValue::ComplexVec32(pairs))
            } else {
                let row_major = transpose_pairs_col_to_row(pairs, rows, cols);
                Ok(MatValue::ComplexMatrix32 {
                    rows,
                    cols,
                    pairs: row_major,
                })
            }
        }
        _ => Err(MatError::Custom(
            "complex compound on non-float class".into(),
        )),
    }
}

fn parse_complex64_pairs(bytes: &[u8], count: usize) -> Result<Vec<(f64, f64)>, MatError> {
    if bytes.len() < count * 16 {
        return Err(MatError::Custom(format!(
            "complex64 raw bytes too short: need {}, have {}",
            count * 16,
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let off = i * 16;
        let re = f64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        let im = f64::from_le_bytes(bytes[off + 8..off + 16].try_into().unwrap());
        out.push((re, im));
    }
    Ok(out)
}

fn parse_complex32_pairs(bytes: &[u8], count: usize) -> Result<Vec<(f32, f32)>, MatError> {
    if bytes.len() < count * 8 {
        return Err(MatError::Custom(format!(
            "complex32 raw bytes too short: need {}, have {}",
            count * 8,
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let off = i * 8;
        let re = f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        let im = f32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap());
        out.push((re, im));
    }
    Ok(out)
}

fn transpose_pairs_col_to_row<T: Copy>(
    col_major: Vec<(T, T)>,
    rows: usize,
    cols: usize,
) -> Vec<(T, T)> {
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            out.push(col_major[c * rows + r]);
        }
    }
    out
}
