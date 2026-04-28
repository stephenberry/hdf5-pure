//! Emit a `MatValue` tree into HDF5 bytes through the public [`MatBuilder`]
//! mid-level API.
//!
//! Used by [`super::root::to_bytes_with_options`]. The default-options path
//! (`to_bytes`) keeps the legacy `emit_file` for backwards compatibility.

use crate::mat::builder::{CellWriter, MatBuilder, StructWriter};
use crate::mat::class::MatClass;
use crate::mat::error::MatError;
use crate::mat::options::{Options, StringClass};
use crate::mat::value::{MatValue, NumVec, ScalarNum, ScalarTag};

/// Walk top-level fields and emit through a `MatBuilder`.
pub(crate) fn emit_file_with_options(
    fields: Vec<(String, MatValue)>,
    options: &Options,
) -> Result<Vec<u8>, MatError> {
    let mut mb = MatBuilder::new(options.clone());
    for (name, value) in fields {
        if matches!(value, MatValue::Omit) {
            continue;
        }
        emit_at_root(&mut mb, &name, value)?;
    }
    mb.finish()
}

fn emit_at_root(mb: &mut MatBuilder, name: &str, value: MatValue) -> Result<(), MatError> {
    match value {
        MatValue::Omit => Ok(()),
        MatValue::Struct(fields) => mb
            .struct_(name, |sw| emit_struct_fields(sw, fields))
            .map(|_| ()),
        MatValue::Cell(elements) => mb
            .cell(name, &cell_dims(elements.len()), |cw| {
                emit_cell_elements(cw, elements)
            })
            .map(|_| ()),
        other => emit_leaf_at_builder(mb, name, other),
    }
}

fn emit_struct_fields(
    sw: &mut StructWriter,
    fields: Vec<(String, MatValue)>,
) -> Result<(), MatError> {
    for (name, value) in fields {
        if matches!(value, MatValue::Omit) {
            continue;
        }
        emit_at_struct(sw, &name, value)?;
    }
    Ok(())
}

fn emit_at_struct(sw: &mut StructWriter, name: &str, value: MatValue) -> Result<(), MatError> {
    match value {
        MatValue::Omit => Ok(()),
        MatValue::Struct(fields) => sw
            .struct_(name, |inner| emit_struct_fields(inner, fields))
            .map(|_| ()),
        MatValue::Cell(elements) => sw
            .cell(name, &cell_dims(elements.len()), |cw| {
                emit_cell_elements(cw, elements)
            })
            .map(|_| ()),
        other => emit_leaf_at_struct(sw, name, other),
    }
}

fn emit_cell_elements(
    cw: &mut CellWriter,
    elements: Vec<MatValue>,
) -> Result<(), MatError> {
    for value in elements {
        emit_cell_element(cw, value)?;
    }
    Ok(())
}

fn emit_cell_element(cw: &mut CellWriter, value: MatValue) -> Result<(), MatError> {
    match value {
        MatValue::Omit => {
            cw.push_empty_struct_array()?;
        }
        MatValue::Struct(fields) => {
            cw.push_struct(|sw| emit_struct_fields(sw, fields))?;
        }
        MatValue::Cell(elements) => {
            let dims = cell_dims(elements.len());
            cw.push_cell(&dims, |inner| emit_cell_elements(inner, elements))?;
        }
        MatValue::Scalar(n) => emit_cell_scalar(cw, n)?,
        MatValue::Vec1D(v) => emit_cell_vec(cw, v)?,
        MatValue::Matrix { rows, cols, vec } => emit_cell_matrix(cw, rows, cols, vec)?,
        MatValue::String(s) => {
            cw.push_char(&s)?;
        }
        MatValue::ComplexScalar64 { re, im } => {
            cw.push_complex_f64(&[1, 1], &[(re, im)])?;
        }
        MatValue::ComplexScalar32 { re, im } => {
            cw.push_complex_f32(&[1, 1], &[(re, im)])?;
        }
        MatValue::ComplexVec64(pairs) => {
            cw.push_complex_f64(&[1, pairs.len()], &pairs)?;
        }
        MatValue::ComplexVec32(pairs) => {
            cw.push_complex_f32(&[1, pairs.len()], &pairs)?;
        }
        MatValue::ComplexMatrix64 { rows, cols, pairs } => {
            let col_major = transpose_pairs(rows, cols, &pairs);
            cw.push_complex_f64(&[rows, cols], &col_major)?;
        }
        MatValue::ComplexMatrix32 { rows, cols, pairs } => {
            let col_major = transpose_pairs(rows, cols, &pairs);
            cw.push_complex_f32(&[rows, cols], &col_major)?;
        }
        MatValue::EmptyStructArray => {
            cw.push_empty_struct_array()?;
        }
    }
    Ok(())
}

fn emit_cell_scalar(cw: &mut CellWriter, scalar: ScalarNum) -> Result<(), MatError> {
    match scalar {
        ScalarNum::Bool(b) => {
            cw.push_scalar_logical(b)?;
        }
        ScalarNum::F64(x) => {
            cw.push_scalar_f64(x)?;
        }
        ScalarNum::F32(x) => {
            cw.push_scalar_f32(x)?;
        }
        ScalarNum::I64(x) => {
            cw.push_scalar_i64(x)?;
        }
        ScalarNum::I32(x) => {
            cw.push_scalar_i32(x)?;
        }
        ScalarNum::I16(x) => {
            cw.push_scalar_i16(x)?;
        }
        ScalarNum::I8(x) => {
            cw.push_scalar_i8(x)?;
        }
        ScalarNum::U64(x) => {
            cw.push_scalar_u64(x)?;
        }
        ScalarNum::U32(x) => {
            cw.push_scalar_u32(x)?;
        }
        ScalarNum::U16(x) => {
            cw.push_scalar_u16(x)?;
        }
        ScalarNum::U8(x) => {
            cw.push_scalar_u8(x)?;
        }
    }
    Ok(())
}

fn emit_cell_vec(cw: &mut CellWriter, v: NumVec) -> Result<(), MatError> {
    // 1-D cell-element vectors honor the configured OneDimensionalMode
    // (default ColumnVector → MATLAB shape `[N, 1]`).
    let dims = cw.vector_dims(v.len());
    match v {
        NumVec::Bool(vec) => {
            let bytes: Vec<u8> = vec.into_iter().map(u8::from).collect();
            cw.push_logical(&dims, &bytes)?;
        }
        NumVec::F64(vec) => {
            cw.push_f64(&dims, &vec)?;
        }
        NumVec::F32(vec) => {
            cw.push_f32(&dims, &vec)?;
        }
        NumVec::I64(vec) => {
            cw.push_i64(&dims, &vec)?;
        }
        NumVec::I32(vec) => {
            cw.push_i32(&dims, &vec)?;
        }
        NumVec::I16(vec) => {
            cw.push_i16(&dims, &vec)?;
        }
        NumVec::I8(vec) => {
            cw.push_i8(&dims, &vec)?;
        }
        NumVec::U64(vec) => {
            cw.push_u64(&dims, &vec)?;
        }
        NumVec::U32(vec) => {
            cw.push_u32(&dims, &vec)?;
        }
        NumVec::U16(vec) => {
            cw.push_u16(&dims, &vec)?;
        }
        NumVec::U8(vec) => {
            cw.push_u8(&dims, &vec)?;
        }
    }
    Ok(())
}

fn emit_cell_matrix(
    cw: &mut CellWriter,
    rows: usize,
    cols: usize,
    v: NumVec,
) -> Result<(), MatError> {
    let dims = [rows, cols];
    match v {
        NumVec::Bool(vec) => {
            let col_major = transpose_scalars(rows, cols, &vec);
            let bytes: Vec<u8> = col_major.into_iter().map(u8::from).collect();
            cw.push_logical(&dims, &bytes)?;
        }
        NumVec::F64(vec) => {
            cw.push_f64(&dims, &transpose_scalars(rows, cols, &vec))?;
        }
        NumVec::F32(vec) => {
            cw.push_f32(&dims, &transpose_scalars(rows, cols, &vec))?;
        }
        NumVec::I64(vec) => {
            cw.push_i64(&dims, &transpose_scalars(rows, cols, &vec))?;
        }
        NumVec::I32(vec) => {
            cw.push_i32(&dims, &transpose_scalars(rows, cols, &vec))?;
        }
        NumVec::I16(vec) => {
            cw.push_i16(&dims, &transpose_scalars(rows, cols, &vec))?;
        }
        NumVec::I8(vec) => {
            cw.push_i8(&dims, &transpose_scalars(rows, cols, &vec))?;
        }
        NumVec::U64(vec) => {
            cw.push_u64(&dims, &transpose_scalars(rows, cols, &vec))?;
        }
        NumVec::U32(vec) => {
            cw.push_u32(&dims, &transpose_scalars(rows, cols, &vec))?;
        }
        NumVec::U16(vec) => {
            cw.push_u16(&dims, &transpose_scalars(rows, cols, &vec))?;
        }
        NumVec::U8(vec) => {
            cw.push_u8(&dims, &transpose_scalars(rows, cols, &vec))?;
        }
    }
    Ok(())
}

fn emit_leaf_at_builder(
    mb: &mut MatBuilder,
    name: &str,
    value: MatValue,
) -> Result<(), MatError> {
    match value {
        MatValue::Omit | MatValue::Struct(_) | MatValue::Cell(_) => {
            // Handled by the caller.
            Ok(())
        }
        MatValue::Scalar(n) => emit_scalar_at_builder(mb, name, n),
        MatValue::Vec1D(v) => emit_vec_at_builder(mb, name, v),
        MatValue::Matrix { rows, cols, vec } => emit_matrix_at_builder(mb, name, rows, cols, vec),
        MatValue::String(s) => emit_string_at_builder(mb, name, &s),
        MatValue::ComplexScalar64 { re, im } => {
            mb.write_complex_f64(name, &[1, 1], &[(re, im)]).map(|_| ())
        }
        MatValue::ComplexScalar32 { re, im } => {
            mb.write_complex_f32(name, &[1, 1], &[(re, im)]).map(|_| ())
        }
        MatValue::ComplexVec64(pairs) => mb
            .write_complex_f64(name, &[1, pairs.len()], &pairs)
            .map(|_| ()),
        MatValue::ComplexVec32(pairs) => mb
            .write_complex_f32(name, &[1, pairs.len()], &pairs)
            .map(|_| ()),
        MatValue::ComplexMatrix64 { rows, cols, pairs } => {
            let col_major = transpose_pairs(rows, cols, &pairs);
            mb.write_complex_f64(name, &[rows, cols], &col_major).map(|_| ())
        }
        MatValue::ComplexMatrix32 { rows, cols, pairs } => {
            let col_major = transpose_pairs(rows, cols, &pairs);
            mb.write_complex_f32(name, &[rows, cols], &col_major).map(|_| ())
        }
        MatValue::EmptyStructArray => mb.write_empty_struct_array(name).map(|_| ()),
    }
}

fn emit_leaf_at_struct(
    sw: &mut StructWriter,
    name: &str,
    value: MatValue,
) -> Result<(), MatError> {
    match value {
        MatValue::Omit | MatValue::Struct(_) | MatValue::Cell(_) => Ok(()),
        MatValue::Scalar(n) => emit_scalar_at_struct(sw, name, n),
        MatValue::Vec1D(v) => emit_vec_at_struct(sw, name, v),
        MatValue::Matrix { rows, cols, vec } => emit_matrix_at_struct(sw, name, rows, cols, vec),
        MatValue::String(s) => emit_string_at_struct(sw, name, &s),
        MatValue::ComplexScalar64 { re, im } => {
            sw.write_complex_f64(name, &[1, 1], &[(re, im)]).map(|_| ())
        }
        MatValue::ComplexScalar32 { re, im } => {
            sw.write_complex_f32(name, &[1, 1], &[(re, im)]).map(|_| ())
        }
        MatValue::ComplexVec64(pairs) => sw
            .write_complex_f64(name, &[1, pairs.len()], &pairs)
            .map(|_| ()),
        MatValue::ComplexVec32(pairs) => sw
            .write_complex_f32(name, &[1, pairs.len()], &pairs)
            .map(|_| ()),
        MatValue::ComplexMatrix64 { rows, cols, pairs } => {
            let col_major = transpose_pairs(rows, cols, &pairs);
            sw.write_complex_f64(name, &[rows, cols], &col_major).map(|_| ())
        }
        MatValue::ComplexMatrix32 { rows, cols, pairs } => {
            let col_major = transpose_pairs(rows, cols, &pairs);
            sw.write_complex_f32(name, &[rows, cols], &col_major).map(|_| ())
        }
        MatValue::EmptyStructArray => sw.write_empty_struct_array(name).map(|_| ()),
    }
}

fn emit_scalar_at_builder(
    mb: &mut MatBuilder,
    name: &str,
    scalar: ScalarNum,
) -> Result<(), MatError> {
    match scalar {
        ScalarNum::Bool(b) => mb.write_scalar_logical(name, b),
        ScalarNum::F64(x) => mb.write_scalar_f64(name, x),
        ScalarNum::F32(x) => mb.write_scalar_f32(name, x),
        ScalarNum::I64(x) => mb.write_scalar_i64(name, x),
        ScalarNum::I32(x) => mb.write_scalar_i32(name, x),
        ScalarNum::I16(x) => mb.write_scalar_i16(name, x),
        ScalarNum::I8(x) => mb.write_scalar_i8(name, x),
        ScalarNum::U64(x) => mb.write_scalar_u64(name, x),
        ScalarNum::U32(x) => mb.write_scalar_u32(name, x),
        ScalarNum::U16(x) => mb.write_scalar_u16(name, x),
        ScalarNum::U8(x) => mb.write_scalar_u8(name, x),
    }
    .map(|_| ())
}

fn emit_scalar_at_struct(
    sw: &mut StructWriter,
    name: &str,
    scalar: ScalarNum,
) -> Result<(), MatError> {
    match scalar {
        ScalarNum::Bool(b) => sw.write_scalar_logical(name, b),
        ScalarNum::F64(x) => sw.write_scalar_f64(name, x),
        ScalarNum::F32(x) => sw.write_scalar_f32(name, x),
        ScalarNum::I64(x) => sw.write_scalar_i64(name, x),
        ScalarNum::I32(x) => sw.write_scalar_i32(name, x),
        ScalarNum::I16(x) => sw.write_scalar_i16(name, x),
        ScalarNum::I8(x) => sw.write_scalar_i8(name, x),
        ScalarNum::U64(x) => sw.write_scalar_u64(name, x),
        ScalarNum::U32(x) => sw.write_scalar_u32(name, x),
        ScalarNum::U16(x) => sw.write_scalar_u16(name, x),
        ScalarNum::U8(x) => sw.write_scalar_u8(name, x),
    }
    .map(|_| ())
}

fn emit_vec_at_builder(mb: &mut MatBuilder, name: &str, v: NumVec) -> Result<(), MatError> {
    let dims = mb.vector_dims(v.len());
    if v.len() == 0 {
        return mb.write_empty(name, scalar_class(v.tag()), &dims).map(|_| ());
    }
    match v {
        NumVec::Bool(vec) => {
            let bytes: Vec<u8> = vec.into_iter().map(u8::from).collect();
            mb.write_logical(name, &dims, &bytes)
        }
        NumVec::F64(vec) => mb.write_f64(name, &dims, &vec),
        NumVec::F32(vec) => mb.write_f32(name, &dims, &vec),
        NumVec::I64(vec) => mb.write_i64(name, &dims, &vec),
        NumVec::I32(vec) => mb.write_i32(name, &dims, &vec),
        NumVec::I16(vec) => mb.write_i16(name, &dims, &vec),
        NumVec::I8(vec) => mb.write_i8(name, &dims, &vec),
        NumVec::U64(vec) => mb.write_u64(name, &dims, &vec),
        NumVec::U32(vec) => mb.write_u32(name, &dims, &vec),
        NumVec::U16(vec) => mb.write_u16(name, &dims, &vec),
        NumVec::U8(vec) => mb.write_u8(name, &dims, &vec),
    }
    .map(|_| ())
}

fn emit_vec_at_struct(sw: &mut StructWriter, name: &str, v: NumVec) -> Result<(), MatError> {
    let dims = sw.vector_dims(v.len());
    if v.len() == 0 {
        return sw.write_empty(name, scalar_class(v.tag()), &dims).map(|_| ());
    }
    match v {
        NumVec::Bool(vec) => {
            let bytes: Vec<u8> = vec.into_iter().map(u8::from).collect();
            sw.write_logical(name, &dims, &bytes)
        }
        NumVec::F64(vec) => sw.write_f64(name, &dims, &vec),
        NumVec::F32(vec) => sw.write_f32(name, &dims, &vec),
        NumVec::I64(vec) => sw.write_i64(name, &dims, &vec),
        NumVec::I32(vec) => sw.write_i32(name, &dims, &vec),
        NumVec::I16(vec) => sw.write_i16(name, &dims, &vec),
        NumVec::I8(vec) => sw.write_i8(name, &dims, &vec),
        NumVec::U64(vec) => sw.write_u64(name, &dims, &vec),
        NumVec::U32(vec) => sw.write_u32(name, &dims, &vec),
        NumVec::U16(vec) => sw.write_u16(name, &dims, &vec),
        NumVec::U8(vec) => sw.write_u8(name, &dims, &vec),
    }
    .map(|_| ())
}

fn emit_matrix_at_builder(
    mb: &mut MatBuilder,
    name: &str,
    rows: usize,
    cols: usize,
    v: NumVec,
) -> Result<(), MatError> {
    let dims = [rows, cols];
    match v {
        NumVec::Bool(vec) => {
            let col_major = transpose_scalars(rows, cols, &vec);
            let bytes: Vec<u8> = col_major.into_iter().map(u8::from).collect();
            mb.write_logical(name, &dims, &bytes)
        }
        NumVec::F64(vec) => mb.write_f64(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::F32(vec) => mb.write_f32(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::I64(vec) => mb.write_i64(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::I32(vec) => mb.write_i32(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::I16(vec) => mb.write_i16(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::I8(vec) => mb.write_i8(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::U64(vec) => mb.write_u64(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::U32(vec) => mb.write_u32(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::U16(vec) => mb.write_u16(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::U8(vec) => mb.write_u8(name, &dims, &transpose_scalars(rows, cols, &vec)),
    }
    .map(|_| ())
}

fn emit_matrix_at_struct(
    sw: &mut StructWriter,
    name: &str,
    rows: usize,
    cols: usize,
    v: NumVec,
) -> Result<(), MatError> {
    let dims = [rows, cols];
    match v {
        NumVec::Bool(vec) => {
            let col_major = transpose_scalars(rows, cols, &vec);
            let bytes: Vec<u8> = col_major.into_iter().map(u8::from).collect();
            sw.write_logical(name, &dims, &bytes)
        }
        NumVec::F64(vec) => sw.write_f64(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::F32(vec) => sw.write_f32(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::I64(vec) => sw.write_i64(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::I32(vec) => sw.write_i32(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::I16(vec) => sw.write_i16(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::I8(vec) => sw.write_i8(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::U64(vec) => sw.write_u64(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::U32(vec) => sw.write_u32(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::U16(vec) => sw.write_u16(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::U8(vec) => sw.write_u8(name, &dims, &transpose_scalars(rows, cols, &vec)),
    }
    .map(|_| ())
}

fn emit_string_at_builder(
    mb: &mut MatBuilder,
    name: &str,
    s: &str,
) -> Result<(), MatError> {
    match mb.options().string_class {
        StringClass::Char => mb.write_char(name, s).map(|_| ()),
        StringClass::String => mb.write_string_object(name, &[s.to_owned()], &[1, 1]).map(|_| ()),
    }
}

fn emit_string_at_struct(
    sw: &mut StructWriter,
    name: &str,
    s: &str,
) -> Result<(), MatError> {
    match sw.string_class() {
        StringClass::Char => sw.write_char(name, s).map(|_| ()),
        StringClass::String => sw
            .write_string_object(name, &[s.to_owned()], &[1, 1])
            .map(|_| ()),
    }
}

fn cell_dims(n: usize) -> [usize; 2] {
    // Mirror the legacy emit.rs behavior: 1-D cell is `[n, 1]`.
    if n == 0 { [0, 0] } else { [n, 1] }
}

fn scalar_class(tag: ScalarTag) -> MatClass {
    match tag {
        ScalarTag::Bool => MatClass::Logical,
        ScalarTag::F64 => MatClass::Double,
        ScalarTag::F32 => MatClass::Single,
        ScalarTag::I64 => MatClass::Int64,
        ScalarTag::I32 => MatClass::Int32,
        ScalarTag::I16 => MatClass::Int16,
        ScalarTag::I8 => MatClass::Int8,
        ScalarTag::U64 => MatClass::UInt64,
        ScalarTag::U32 => MatClass::UInt32,
        ScalarTag::U16 => MatClass::UInt16,
        ScalarTag::U8 => MatClass::UInt8,
    }
}

/// Transpose a row-major matrix of shape `[rows, cols]` into column-major.
/// Tiled to keep both reads and writes cache-resident on large matrices.
fn transpose_2d<T: Copy>(rows: usize, cols: usize, row_major: &[T]) -> Vec<T> {
    debug_assert_eq!(row_major.len(), rows * cols);
    let n = rows * cols;
    let mut out: Vec<T> = Vec::with_capacity(n);
    if n == 0 {
        return out;
    }

    const BLK: usize = 32;
    let dst = out.as_mut_ptr();
    for cb in (0..cols).step_by(BLK) {
        let c_end = (cb + BLK).min(cols);
        for rb in (0..rows).step_by(BLK) {
            let r_end = (rb + BLK).min(rows);
            for r in rb..r_end {
                let src_row_base = r * cols;
                for c in cb..c_end {
                    let value = row_major[src_row_base + c];
                    // SAFETY: c < cols and r < rows so c*rows + r < cols*rows = n,
                    // and out has capacity n.
                    unsafe {
                        dst.add(c * rows + r).write(value);
                    }
                }
            }
        }
    }
    // SAFETY: every index 0..n was written above (each (r, c) maps to a unique
    // c * rows + r in 0..n).
    unsafe {
        out.set_len(n);
    }
    out
}

#[inline]
fn transpose_pairs<T: Copy>(rows: usize, cols: usize, row_major: &[(T, T)]) -> Vec<(T, T)> {
    transpose_2d(rows, cols, row_major)
}

#[inline]
fn transpose_scalars<T: Copy>(rows: usize, cols: usize, row_major: &[T]) -> Vec<T> {
    transpose_2d(rows, cols, row_major)
}
