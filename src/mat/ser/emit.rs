//! Emit a `MatValue` tree into an HDF5 file with MATLAB v7.3 conventions.

use crate::file_writer::AttrValue;
use crate::mat::class::MatClass;
use crate::mat::error::MatError;
use crate::mat::userblock::{self, USERBLOCK_SIZE};
use crate::mat::utf16;
use crate::type_builders::{
    make_f32_type, make_f64_type, DatasetBuilder, FinishedGroup, GroupBuilder,
};
use crate::writer::FileBuilder;

use crate::mat::value::{MatValue, NumVec, ScalarNum, ScalarTag};

/// Turn a list of top-level `(name, value)` pairs into a MAT 7.3 file.
pub(crate) fn emit_file(fields: Vec<(String, MatValue)>) -> Result<Vec<u8>, MatError> {
    let mut builder = FileBuilder::new();
    builder.with_userblock(USERBLOCK_SIZE);

    for (name, value) in fields {
        if matches!(value, MatValue::Omit) {
            continue;
        }
        emit_at_root(&mut builder, &name, value)?;
    }

    let mut bytes = builder.finish().map_err(MatError::Hdf5)?;
    userblock::write_header(&mut bytes, userblock::DEFAULT_DESCRIPTION);
    Ok(bytes)
}

/// Emit a single named value at the file root.
fn emit_at_root(builder: &mut FileBuilder, name: &str, value: MatValue) -> Result<(), MatError> {
    match value {
        MatValue::Omit => Ok(()),
        MatValue::Struct(fields) => {
            let group = build_struct_group(name, fields)?;
            builder.add_group(group);
            Ok(())
        }
        other => {
            let ds = builder.create_dataset(name);
            apply_value_to_dataset(ds, other)
        }
    }
}

/// Emit a value as a child of a group.
fn emit_into_group(group: &mut GroupBuilder, name: &str, value: MatValue) -> Result<(), MatError> {
    match value {
        MatValue::Omit => Ok(()),
        MatValue::Struct(fields) => {
            let sub = build_struct_group(name, fields)?;
            group.add_group(sub);
            Ok(())
        }
        other => {
            let ds = group.create_dataset(name);
            apply_value_to_dataset(ds, other)
        }
    }
}

/// Build a `FinishedGroup` representing a MATLAB struct.
fn build_struct_group(
    name: &str,
    fields: Vec<(String, MatValue)>,
) -> Result<FinishedGroup, MatError> {
    let mut group = new_group_builder(name);
    // Filter out Omit fields and record the surviving order.
    let mut live_names: Vec<String> = Vec::with_capacity(fields.len());
    for (fname, value) in fields {
        if matches!(value, MatValue::Omit) {
            continue;
        }
        live_names.push(fname.clone());
        emit_into_group(&mut group, &fname, value)?;
    }
    group.set_attr(
        "MATLAB_class",
        AttrValue::AsciiString(MatClass::Struct.as_str().into()),
    );
    group.set_attr("MATLAB_fields", AttrValue::VarLenAsciiArray(live_names));
    Ok(group.finish())
}

fn new_group_builder(name: &str) -> GroupBuilder {
    // Same as FileBuilder::create_group but without needing a FileBuilder.
    // GroupBuilder::new is crate-visible.
    GroupBuilder::new(name)
}

/// Apply a non-struct `MatValue` to the given `DatasetBuilder`, writing data,
/// shape, and the `MATLAB_class` attribute.
fn apply_value_to_dataset(ds: &mut DatasetBuilder, value: MatValue) -> Result<(), MatError> {
    match value {
        MatValue::Omit | MatValue::Struct(_) => {
            unreachable!("emitted as group, not dataset")
        }
        MatValue::Scalar(n) => apply_scalar(ds, n),
        MatValue::Vec1D(v) => apply_vec_1d(ds, v),
        MatValue::Matrix { rows, cols, vec } => apply_matrix(ds, rows, cols, vec),
        MatValue::String(s) => {
            apply_char_string(ds, &s);
            Ok(())
        }
        MatValue::ComplexScalar64 { re, im } => {
            ds.with_complex64_data(&[(re, im)]).with_shape(&[1, 1]);
            set_class(ds, MatClass::Double);
            Ok(())
        }
        MatValue::ComplexScalar32 { re, im } => {
            ds.with_complex32_data(&[(re, im)]).with_shape(&[1, 1]);
            set_class(ds, MatClass::Single);
            Ok(())
        }
        MatValue::ComplexVec64(pairs) => {
            let n = pairs.len() as u64;
            ds.with_complex64_data(&pairs).with_shape(&[1, n]);
            set_class(ds, MatClass::Double);
            Ok(())
        }
        MatValue::ComplexVec32(pairs) => {
            let n = pairs.len() as u64;
            ds.with_complex32_data(&pairs).with_shape(&[1, n]);
            set_class(ds, MatClass::Single);
            Ok(())
        }
        MatValue::ComplexMatrix64 { rows, cols, pairs } => {
            let col_major = transpose_pairs(rows, cols, &pairs);
            ds.with_complex64_data(&col_major)
                .with_shape(&[cols as u64, rows as u64]);
            set_class(ds, MatClass::Double);
            Ok(())
        }
        MatValue::ComplexMatrix32 { rows, cols, pairs } => {
            let col_major = transpose_pairs(rows, cols, &pairs);
            ds.with_complex32_data(&col_major)
                .with_shape(&[cols as u64, rows as u64]);
            set_class(ds, MatClass::Single);
            Ok(())
        }
    }
}

fn apply_scalar(ds: &mut DatasetBuilder, n: ScalarNum) -> Result<(), MatError> {
    match n {
        ScalarNum::Bool(b) => {
            ds.with_u8_data(&[u8::from(b)]).with_shape(&[1, 1]);
            set_class(ds, MatClass::Logical);
            set_logical_decode(ds);
        }
        ScalarNum::F64(x) => {
            ds.with_f64_data(&[x]).with_shape(&[1, 1]);
            set_class(ds, MatClass::Double);
        }
        ScalarNum::F32(x) => {
            ds.with_f32_data(&[x]).with_shape(&[1, 1]);
            set_class(ds, MatClass::Single);
        }
        ScalarNum::I64(x) => {
            ds.with_i64_data(&[x]).with_shape(&[1, 1]);
            set_class(ds, MatClass::Int64);
        }
        ScalarNum::I32(x) => {
            ds.with_i32_data(&[x]).with_shape(&[1, 1]);
            set_class(ds, MatClass::Int32);
        }
        ScalarNum::I16(x) => {
            ds.with_i16_data(&[x]).with_shape(&[1, 1]);
            set_class(ds, MatClass::Int16);
        }
        ScalarNum::I8(x) => {
            ds.with_i8_data(&[x]).with_shape(&[1, 1]);
            set_class(ds, MatClass::Int8);
        }
        ScalarNum::U64(x) => {
            ds.with_u64_data(&[x]).with_shape(&[1, 1]);
            set_class(ds, MatClass::UInt64);
        }
        ScalarNum::U32(x) => {
            ds.with_u32_data(&[x]).with_shape(&[1, 1]);
            set_class(ds, MatClass::UInt32);
        }
        ScalarNum::U16(x) => {
            ds.with_u16_data(&[x]).with_shape(&[1, 1]);
            set_class(ds, MatClass::UInt16);
        }
        ScalarNum::U8(x) => {
            ds.with_u8_data(&[x]).with_shape(&[1, 1]);
            set_class(ds, MatClass::UInt8);
        }
    }
    Ok(())
}

fn apply_vec_1d(ds: &mut DatasetBuilder, v: NumVec) -> Result<(), MatError> {
    let n = v.len() as u64;
    if n == 0 {
        emit_empty(ds, v.tag());
        return Ok(());
    }
    let shape = [1u64, n];
    match v {
        NumVec::Bool(vec) => {
            let bytes: Vec<u8> = vec.into_iter().map(u8::from).collect();
            ds.with_u8_data(&bytes).with_shape(&shape);
            set_class(ds, MatClass::Logical);
            set_logical_decode(ds);
        }
        NumVec::F64(vec) => {
            ds.with_f64_data(&vec).with_shape(&shape);
            set_class(ds, MatClass::Double);
        }
        NumVec::F32(vec) => {
            ds.with_f32_data(&vec).with_shape(&shape);
            set_class(ds, MatClass::Single);
        }
        NumVec::I64(vec) => {
            ds.with_i64_data(&vec).with_shape(&shape);
            set_class(ds, MatClass::Int64);
        }
        NumVec::I32(vec) => {
            ds.with_i32_data(&vec).with_shape(&shape);
            set_class(ds, MatClass::Int32);
        }
        NumVec::I16(vec) => {
            ds.with_i16_data(&vec).with_shape(&shape);
            set_class(ds, MatClass::Int16);
        }
        NumVec::I8(vec) => {
            ds.with_i8_data(&vec).with_shape(&shape);
            set_class(ds, MatClass::Int8);
        }
        NumVec::U64(vec) => {
            ds.with_u64_data(&vec).with_shape(&shape);
            set_class(ds, MatClass::UInt64);
        }
        NumVec::U32(vec) => {
            ds.with_u32_data(&vec).with_shape(&shape);
            set_class(ds, MatClass::UInt32);
        }
        NumVec::U16(vec) => {
            ds.with_u16_data(&vec).with_shape(&shape);
            set_class(ds, MatClass::UInt16);
        }
        NumVec::U8(vec) => {
            ds.with_u8_data(&vec).with_shape(&shape);
            set_class(ds, MatClass::UInt8);
        }
    }
    Ok(())
}

fn apply_matrix(
    ds: &mut DatasetBuilder,
    rows: usize,
    cols: usize,
    vec: NumVec,
) -> Result<(), MatError> {
    debug_assert_eq!(vec.len(), rows * cols);
    // HDF5 shape for a MATLAB [rows × cols] matrix is [cols, rows].
    let shape = [cols as u64, rows as u64];
    match vec {
        NumVec::Bool(row_major) => {
            let col_major = transpose_scalars(rows, cols, &row_major);
            let bytes: Vec<u8> = col_major.into_iter().map(u8::from).collect();
            ds.with_u8_data(&bytes).with_shape(&shape);
            set_class(ds, MatClass::Logical);
            set_logical_decode(ds);
        }
        NumVec::F64(row_major) => {
            let col = transpose_scalars(rows, cols, &row_major);
            ds.with_f64_data(&col).with_shape(&shape);
            set_class(ds, MatClass::Double);
        }
        NumVec::F32(row_major) => {
            let col = transpose_scalars(rows, cols, &row_major);
            ds.with_f32_data(&col).with_shape(&shape);
            set_class(ds, MatClass::Single);
        }
        NumVec::I64(row_major) => {
            let col = transpose_scalars(rows, cols, &row_major);
            ds.with_i64_data(&col).with_shape(&shape);
            set_class(ds, MatClass::Int64);
        }
        NumVec::I32(row_major) => {
            let col = transpose_scalars(rows, cols, &row_major);
            ds.with_i32_data(&col).with_shape(&shape);
            set_class(ds, MatClass::Int32);
        }
        NumVec::I16(row_major) => {
            let col = transpose_scalars(rows, cols, &row_major);
            ds.with_i16_data(&col).with_shape(&shape);
            set_class(ds, MatClass::Int16);
        }
        NumVec::I8(row_major) => {
            let col = transpose_scalars(rows, cols, &row_major);
            ds.with_i8_data(&col).with_shape(&shape);
            set_class(ds, MatClass::Int8);
        }
        NumVec::U64(row_major) => {
            let col = transpose_scalars(rows, cols, &row_major);
            ds.with_u64_data(&col).with_shape(&shape);
            set_class(ds, MatClass::UInt64);
        }
        NumVec::U32(row_major) => {
            let col = transpose_scalars(rows, cols, &row_major);
            ds.with_u32_data(&col).with_shape(&shape);
            set_class(ds, MatClass::UInt32);
        }
        NumVec::U16(row_major) => {
            let col = transpose_scalars(rows, cols, &row_major);
            ds.with_u16_data(&col).with_shape(&shape);
            set_class(ds, MatClass::UInt16);
        }
        NumVec::U8(row_major) => {
            let col = transpose_scalars(rows, cols, &row_major);
            ds.with_u8_data(&col).with_shape(&shape);
            set_class(ds, MatClass::UInt8);
        }
    }
    Ok(())
}

fn apply_char_string(ds: &mut DatasetBuilder, s: &str) {
    let units = utf16::encode_utf16(s);
    let n = units.len() as u64;
    if n == 0 {
        // Empty char: use MATLAB_empty marker with [0, 0] shape.
        ds.with_u16_data(&[]).with_shape(&[0u64, 0]);
        set_class(ds, MatClass::Char);
        set_char_decode(ds);
        ds.set_attr("MATLAB_empty", AttrValue::U32(1));
        return;
    }
    // MATLAB strings are row vectors: MATLAB shape [1, N] → HDF5 [N, 1]
    // (column-major on-disk). This matches libmatio's output and lets
    // MATLAB `strcmp` work without transposing.
    ds.with_u16_data(&units).with_shape(&[n, 1]);
    set_class(ds, MatClass::Char);
    set_char_decode(ds);
}

fn emit_empty(ds: &mut DatasetBuilder, tag: ScalarTag) {
    // Empty numeric array: shape [0, 0], MATLAB_empty = 1.
    match tag {
        ScalarTag::Bool => {
            ds.with_u8_data(&[]).with_shape(&[0u64, 0]);
            set_class(ds, MatClass::Logical);
            set_logical_decode(ds);
        }
        ScalarTag::F64 => {
            ds.with_f64_data(&[]).with_shape(&[0u64, 0]);
            set_class(ds, MatClass::Double);
        }
        ScalarTag::F32 => {
            ds.with_f32_data(&[]).with_shape(&[0u64, 0]);
            set_class(ds, MatClass::Single);
        }
        ScalarTag::I64 => {
            ds.with_i64_data(&[]).with_shape(&[0u64, 0]);
            set_class(ds, MatClass::Int64);
        }
        ScalarTag::I32 => {
            ds.with_i32_data(&[]).with_shape(&[0u64, 0]);
            set_class(ds, MatClass::Int32);
        }
        ScalarTag::I16 => {
            ds.with_i16_data(&[]).with_shape(&[0u64, 0]);
            set_class(ds, MatClass::Int16);
        }
        ScalarTag::I8 => {
            ds.with_i8_data(&[]).with_shape(&[0u64, 0]);
            set_class(ds, MatClass::Int8);
        }
        ScalarTag::U64 => {
            ds.with_u64_data(&[]).with_shape(&[0u64, 0]);
            set_class(ds, MatClass::UInt64);
        }
        ScalarTag::U32 => {
            ds.with_u32_data(&[]).with_shape(&[0u64, 0]);
            set_class(ds, MatClass::UInt32);
        }
        ScalarTag::U16 => {
            ds.with_u16_data(&[]).with_shape(&[0u64, 0]);
            set_class(ds, MatClass::UInt16);
        }
        ScalarTag::U8 => {
            ds.with_u8_data(&[]).with_shape(&[0u64, 0]);
            set_class(ds, MatClass::UInt8);
        }
    }
    ds.set_attr("MATLAB_empty", AttrValue::U32(1));
}

fn set_class(ds: &mut DatasetBuilder, class: MatClass) {
    ds.set_attr(
        "MATLAB_class",
        AttrValue::AsciiString(class.as_str().into()),
    );
}

/// MATLAB writes logical datasets as uint8 storage with `MATLAB_int_decode = 1`
/// in addition to `MATLAB_class = "logical"`. Without this attribute matio
/// (and MATLAB itself) report the variable as an empty/unknown class.
fn set_logical_decode(ds: &mut DatasetBuilder) {
    ds.set_attr("MATLAB_int_decode", AttrValue::I32(1));
}

/// `char` datasets are uint16 storage; MATLAB also expects
/// `MATLAB_int_decode = 2` so the library decodes the uint16 code units as
/// UTF-16 characters rather than a numeric array.
fn set_char_decode(ds: &mut DatasetBuilder) {
    ds.set_attr("MATLAB_int_decode", AttrValue::I32(2));
}

fn transpose_scalars<T: Copy>(rows: usize, cols: usize, row_major: &[T]) -> Vec<T> {
    let mut out = Vec::with_capacity(rows * cols);
    for c in 0..cols {
        for r in 0..rows {
            out.push(row_major[r * cols + c]);
        }
    }
    out
}

fn transpose_pairs<T: Copy>(rows: usize, cols: usize, row_major: &[(T, T)]) -> Vec<(T, T)> {
    let mut out = Vec::with_capacity(rows * cols);
    for c in 0..cols {
        for r in 0..rows {
            out.push(row_major[r * cols + c]);
        }
    }
    out
}

// Silence the "unused import" on the no-test build.
#[allow(dead_code)]
fn _touch() {
    let _ = make_f64_type();
    let _ = make_f32_type();
}
