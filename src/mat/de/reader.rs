//! Read an HDF5 file with MATLAB conventions into a `MatValue` tree.

use std::collections::HashMap;

use crate::convert::TryToUsize;
use crate::error::FormatError;
use crate::file_writer::AttrValue;
use crate::mat::class::MatClass;
use crate::mat::de::mcos_reader::{self, Mcos, PropValue};
use crate::mat::error::MatError;
use crate::mat::utf16;
use crate::mat::value::{MatValue, NumVec, ScalarNum};
use crate::reader::{Dataset, File, Group, Object};
use crate::types::DType;

/// Maximum struct/cell nesting depth followed when reading. Real MATLAB data
/// nests a handful of levels deep; this bound is generous headroom that still
/// turns a malformed file — e.g. a cell whose object reference points back at
/// itself, or a cyclic group link — into a typed error instead of unbounded
/// recursion and a stack overflow.
const MAX_NESTING_DEPTH: usize = 256;

/// Parse a MAT v7.3 file into an ordered list of `(name, value)` fields
/// rooted at the file's top level.
pub(crate) fn read_file(bytes: &[u8]) -> Result<Vec<(String, MatValue)>, MatError> {
    // Accept raw HDF5 too — we don't require the MATLAB signature to be
    // present to deserialize successfully.
    let file = File::from_bytes(bytes.to_vec()).map_err(MatError::Hdf5)?;
    // Parse the MCOS opaque-object store once; opaque datasets (modern `string`,
    // …) anywhere in the file resolve their payloads against it. `None` when the
    // file has no `#subsystem#` group.
    let mcos = Mcos::parse(&file)?;
    let root = file.root();
    read_group(&root, mcos.as_ref(), 0)
}

fn read_group(
    group: &Group<'_>,
    mcos: Option<&Mcos<'_>>,
    depth: usize,
) -> Result<Vec<(String, MatValue)>, MatError> {
    if depth > MAX_NESTING_DEPTH {
        return Err(MatError::Format(FormatError::NestingDepthExceeded));
    }
    let mut out = Vec::new();

    // Collect datasets first, then subgroups. Preserve iteration order from
    // the HDF5 link table. Skip MATLAB's reserved internal entries — `#refs#`
    // (cell/object-reference payloads) and `#subsystem#` (the MCOS opaque-class
    // store): MATLAB hides them, no variable name can begin with `#`, and their
    // contents are reached by reference from the variables that use them, not
    // by walking the top level.
    for name in group.datasets().map_err(MatError::Hdf5)? {
        if name.starts_with('#') {
            continue;
        }
        let ds = group.dataset(&name).map_err(MatError::Hdf5)?;
        out.push((name, read_dataset(&ds, mcos, depth)?));
    }
    for name in group.groups().map_err(MatError::Hdf5)? {
        if name.starts_with('#') {
            continue;
        }
        let sub = group.group(&name).map_err(MatError::Hdf5)?;
        out.push((name, read_group_as_value(&sub, mcos, depth + 1)?));
    }
    Ok(out)
}

fn read_group_as_value(
    group: &Group<'_>,
    mcos: Option<&Mcos<'_>>,
    depth: usize,
) -> Result<MatValue, MatError> {
    let fields = read_group(group, mcos, depth)?;
    Ok(MatValue::Struct(fields))
}

/// Read a single dataset into a `MatValue` using its MATLAB_class attribute
/// (if present) and its HDF5 shape. `depth` bounds struct/cell nesting.
fn read_dataset(
    ds: &Dataset<'_>,
    mcos: Option<&Mcos<'_>>,
    depth: usize,
) -> Result<MatValue, MatError> {
    if depth > MAX_NESTING_DEPTH {
        return Err(MatError::Format(FormatError::NestingDepthExceeded));
    }
    let attrs = ds.attrs().map_err(MatError::Hdf5)?;

    // An `mxOPAQUE_CLASS` object (`MATLAB_object_decode` set) stores its data in
    // the `#subsystem#` MCOS store, not on this dataset. Decode it there before
    // the builtin-class path — its `MATLAB_class` (e.g. `string`) is not a
    // `MatClass` variant.
    if let Some(decode) = mcos_reader::matlab_object_decode(&attrs) {
        return decode_opaque(ds, &attrs, decode, mcos, depth);
    }

    let class = matlab_class_from_attrs(&attrs)?;
    let shape = ds.shape().map_err(MatError::Hdf5)?;
    let dtype = ds.dtype().map_err(MatError::Hdf5)?;
    let is_empty = is_empty_attr(&attrs) || shape.contains(&0);

    let class = class.unwrap_or_else(|| class_from_dtype(&dtype));

    if is_empty {
        // Complex compound types preserve their shape and class so a 0×0
        // `Matrix<Complex*>` round-trips back to a 0×0 complex matrix
        // rather than collapsing to a numeric empty vec.
        if matches!(class, MatClass::Double | MatClass::Single) && is_complex_dtype(&dtype) {
            return empty_complex_value(class, &shape);
        }
        // An empty struct-classed *dataset* (`MATLAB_empty=1`) is `struct([])`,
        // the marker the serializer writes for `Option::None` inside a
        // sequence. Surface it as `EmptyStructArray` so an `Option` field
        // round-trips to `None` (an empty struct *group*, by contrast, reads as
        // an empty struct via `read_group_as_value`).
        if class == MatClass::Struct {
            return Ok(MatValue::EmptyStructArray);
        }
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
        MatClass::Cell => read_cell(ds, mcos, depth),
    }
}

/// Read a cell-array dataset: a vector of HDF5 object references, each pointing
/// at a member object (struct group, numeric/char dataset, nested cell, …)
/// interned under `#refs#`. Members are decoded in storage order and collected
/// into a flat [`MatValue::Cell`], matching the column-vector layout the
/// serializer writes (the deserializer flattens a cell to a sequence).
fn read_cell(
    ds: &Dataset<'_>,
    mcos: Option<&Mcos<'_>>,
    depth: usize,
) -> Result<MatValue, MatError> {
    let members = ds.dereference().map_err(MatError::Hdf5)?;
    let mut elems = Vec::with_capacity(members.len());
    for member in members {
        elems.push(read_object(&member, mcos, depth + 1)?);
    }
    Ok(MatValue::Cell(elems))
}

/// Decode a dereferenced object (a `#refs#`/`#subsystem#` dataset or struct
/// group) into a `MatValue`, dispatching on whether it is a dataset or group.
fn read_object(
    obj: &Object<'_>,
    mcos: Option<&Mcos<'_>>,
    depth: usize,
) -> Result<MatValue, MatError> {
    match obj {
        Object::Dataset(d) => read_dataset(d, mcos, depth),
        Object::Group(g) => read_group_as_value(g, mcos, depth),
    }
}

/// Decode an `mxOPAQUE_CLASS` dataset (one with `MATLAB_object_decode` set).
///
/// The modern `string` class keeps its dedicated saveobj decoder; every other
/// opaque class resolves its object id(s) from the parent `uint32` metadata and
/// decodes each object through the `#subsystem#/MCOS` FileWrapper property
/// tables — to a typed value (`datetime`, `duration`, `categorical`) or a
/// lossless [`MatValue::Opaque`] for classes without a dedicated decoder.
fn decode_opaque(
    ds: &Dataset<'_>,
    attrs: &HashMap<String, AttrValue>,
    decode: i64,
    mcos: Option<&Mcos<'_>>,
    depth: usize,
) -> Result<MatValue, MatError> {
    let class = mcos_reader::raw_matlab_class(attrs).ok_or_else(|| {
        MatError::Custom("opaque object (MATLAB_object_decode set) has no MATLAB_class".into())
    })?;
    // Only `MATLAB_object_decode == 3` (mxOPAQUE_CLASS / MCOS) is decoded here.
    // `1` (function handle) and `2` (legacy object) have different on-disk
    // layouts; refuse them by name rather than misread.
    if !mcos_reader::is_mcos_decode(decode) {
        return Err(MatError::UnsupportedMatlabClass(class));
    }
    let mcos = mcos.ok_or_else(|| {
        MatError::Custom(
            "file references an MCOS opaque object but has no #subsystem# store".into(),
        )
    })?;

    if mcos_reader::is_string_class(&class, decode) {
        return mcos.decode_string(ds);
    }

    let meta = ds.read_u32().map_err(MatError::Hdf5)?;
    let parsed = mcos_reader::parse_opaque_metadata(&meta)?;
    if parsed.object_ids.is_empty() {
        // An empty opaque array names no objects; resolve its declared class
        // from the parent metadata's trailing class id (falling back to the
        // dataset's `MATLAB_class`) and surface it with no properties so the
        // read still succeeds rather than erroring.
        let class_name = mcos.class_name(parsed.class_id).unwrap_or(class);
        return Ok(MatValue::Opaque {
            class_name,
            fields: Vec::new(),
        });
    }

    let mut values = Vec::with_capacity(parsed.object_ids.len());
    for object_id in parsed.object_ids {
        values.push(decode_opaque_object(mcos, object_id, depth)?);
    }
    // A scalar opaque variable is a single object; a genuine object array
    // becomes a cell of the decoded objects.
    if values.len() == 1 {
        Ok(values.pop().expect("len checked"))
    } else {
        Ok(MatValue::Cell(values))
    }
}

/// Assemble one MCOS object's resolved properties (reading heap cells through
/// the full read path so nested objects/cells decode too) and finish it via a
/// typed decoder or the lossless `Opaque` fallback.
fn decode_opaque_object(
    mcos: &Mcos<'_>,
    object_id: u32,
    depth: usize,
) -> Result<MatValue, MatError> {
    if depth > MAX_NESTING_DEPTH {
        return Err(MatError::Format(FormatError::NestingDepthExceeded));
    }
    let class_name = mcos.object_class_name(object_id)?;
    let mut fields = Vec::new();
    for prop in mcos.properties(object_id)? {
        let value = match prop.value {
            PropValue::Heap(v) => read_object(mcos.heap_object(v)?, Some(mcos), depth + 1)?,
            PropValue::Inline(n) => MatValue::Scalar(mcos_reader::inline_to_scalar(n)),
            PropValue::Name(s) => MatValue::String(s),
        };
        fields.push((prop.name, value));
    }
    mcos_reader::decode_object_fields(class_name, fields)
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
            )));
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

/// Build an empty `ComplexMatrix*` whose `(rows, cols)` matches the dataspace
/// shape, so deserializing into `Matrix<Complex*>` recovers the original
/// shape (0×0, 0×N, N×0) instead of collapsing to 1×0.
fn empty_complex_value(class: MatClass, shape: &[u64]) -> Result<MatValue, MatError> {
    let (rows, cols, _total) = shape_decomposition(shape)?;
    Ok(match class {
        MatClass::Double => MatValue::ComplexMatrix64 {
            rows,
            cols,
            pairs: Vec::new(),
        },
        MatClass::Single => MatValue::ComplexMatrix32 {
            rows,
            cols,
            pairs: Vec::new(),
        },
        _ => unreachable!("empty_complex_value called with non-float class"),
    })
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
        MatClass::Cell => MatValue::Cell(Vec::new()),
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

fn read_numeric(ds: &Dataset<'_>, shape: &[u64], class: MatClass) -> Result<MatValue, MatError> {
    let (rows, cols, total) = shape_decomposition(shape)?;

    // For a single-element dataset we emit a Scalar of the appropriate class.
    if total == 1 {
        return Ok(MatValue::Scalar(read_scalar(ds, class)?));
    }

    // Read all elements as the MATLAB class's native type.
    let flat = read_all_elements(ds, class)?;

    // A 1-D HDF5 dataset (no recorded cols/rows split) is treated as a flat
    // Vec1D. Files produced by MATLAB/this library are always 2-D, but some
    // external tools write true 1-D shapes.
    if shape.len() <= 1 {
        return Ok(MatValue::Vec1D(flat));
    }

    // 2-D dataset. Preserve the MATLAB [rows, cols] shape even when one dim
    // is 1 — Matrix<T> needs this to distinguish row vs column vectors. The
    // deserializer flattens to a plain sequence for Vec<T> callers.
    // Skip the transpose when one dim is 1: column-major and row-major
    // orderings are identical for 1×N / N×1, so the call would be a no-op
    // copy.
    let matrix = if rows == 1 || cols == 1 {
        flat
    } else {
        transpose_col_major_to_row_major(flat, rows, cols)?
    };
    Ok(MatValue::Matrix {
        rows,
        cols,
        vec: matrix,
    })
}

fn shape_decomposition(shape: &[u64]) -> Result<(usize, usize, usize), MatError> {
    // HDF5 shape for MATLAB data is [cols, rows] (column-major storage of a
    // [rows, cols] matrix). For a 1-D variant like [1, N] or [N, 1], treat as
    // a vector of length N. The dimensions come from the file, so each is
    // narrowed to `usize` through the checked conversion that errors (rather
    // than truncating) if a dimension exceeds the platform's pointer width.
    Ok(match shape.len() {
        0 => (1, 1, 1),
        1 => {
            let n = shape[0].to_usize()?;
            (1, n, n)
        }
        2 => {
            let cols_hdf5 = shape[0].to_usize()?;
            let rows_hdf5 = shape[1].to_usize()?;
            // MATLAB matrix has rows = rows_hdf5, cols = cols_hdf5.
            let total = cols_hdf5 * rows_hdf5;
            (rows_hdf5, cols_hdf5, total)
        }
        _ => {
            let mut total: usize = 1;
            for &d in shape {
                total *= d.to_usize()?;
            }
            (1, total, total)
        }
    })
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

fn read_complex(ds: &Dataset<'_>, shape: &[u64], class: MatClass) -> Result<MatValue, MatError> {
    let (rows, cols, total) = shape_decomposition(shape)?;
    let bytes = ds.read_u8().map_err(MatError::Hdf5)?;

    match class {
        MatClass::Double => {
            let pairs = parse_complex64_pairs(&bytes, total)?;
            if total == 1 {
                let (re, im) = pairs[0];
                return Ok(MatValue::ComplexScalar64 { re, im });
            }
            // Preserve the MATLAB [rows, cols] shape even when one dim is 1,
            // so `Matrix<Complex64>` round-trips as a row vs column vector.
            // The deserializer flattens to a plain sequence for
            // `Vec<Complex64>` callers.
            let row_major = if rows == 1 || cols == 1 {
                pairs
            } else {
                transpose_pairs_col_to_row(pairs, rows, cols)
            };
            Ok(MatValue::ComplexMatrix64 {
                rows,
                cols,
                pairs: row_major,
            })
        }
        MatClass::Single => {
            let pairs = parse_complex32_pairs(&bytes, total)?;
            if total == 1 {
                let (re, im) = pairs[0];
                return Ok(MatValue::ComplexScalar32 { re, im });
            }
            let row_major = if rows == 1 || cols == 1 {
                pairs
            } else {
                transpose_pairs_col_to_row(pairs, rows, cols)
            };
            Ok(MatValue::ComplexMatrix32 {
                rows,
                cols,
                pairs: row_major,
            })
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
