//! `serde::Deserializer` implementations for `MatValue`.
//!
//! Entry: [`MatValueDeserializer::new`] wraps a [`MatValue`] and implements
//! `Deserializer`. Helper access types (struct maps, sequences) live below.

use std::collections::VecDeque;

use serde::de::{
    self, DeserializeSeed, Deserializer, EnumAccess, IntoDeserializer, MapAccess, SeqAccess,
    VariantAccess, Visitor,
};
use serde::forward_to_deserialize_any;

use crate::mat::complex::{COMPLEX32_SENTINEL, COMPLEX64_SENTINEL};
use crate::mat::error::MatError;
use crate::mat::matrix::MATRIX_SENTINEL;
use crate::mat::value::{MatValue, NumVec, ScalarNum};

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub(crate) struct MatValueDeserializer {
    value: MatValue,
}

impl MatValueDeserializer {
    pub(crate) fn new(value: MatValue) -> Self {
        Self { value }
    }
}

impl<'de> Deserializer<'de> for MatValueDeserializer {
    type Error = MatError;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        dispatch_any(self.value, visitor)
    }

    // ----- primitives (coerce from Scalar) -----

    fn deserialize_bool<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        match self.value {
            MatValue::Scalar(ScalarNum::Bool(b)) => visitor.visit_bool(b),
            MatValue::Scalar(ScalarNum::U8(x)) => visitor.visit_bool(x != 0),
            other => mismatch("bool", other),
        }
    }

    fn deserialize_i8<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        let v = to_i64(self.value, "i8")?;
        visitor.visit_i8(v as i8)
    }
    fn deserialize_i16<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        let v = to_i64(self.value, "i16")?;
        visitor.visit_i16(v as i16)
    }
    fn deserialize_i32<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        let v = to_i64(self.value, "i32")?;
        visitor.visit_i32(v as i32)
    }
    fn deserialize_i64<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        let v = to_i64(self.value, "i64")?;
        visitor.visit_i64(v)
    }
    fn deserialize_u8<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        let v = to_u64(self.value, "u8")?;
        visitor.visit_u8(v as u8)
    }
    fn deserialize_u16<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        let v = to_u64(self.value, "u16")?;
        visitor.visit_u16(v as u16)
    }
    fn deserialize_u32<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        let v = to_u64(self.value, "u32")?;
        visitor.visit_u32(v as u32)
    }
    fn deserialize_u64<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        let v = to_u64(self.value, "u64")?;
        visitor.visit_u64(v)
    }
    fn deserialize_f32<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        let v = to_f64(self.value, "f32")?;
        visitor.visit_f32(v as f32)
    }
    fn deserialize_f64<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        let v = to_f64(self.value, "f64")?;
        visitor.visit_f64(v)
    }

    fn deserialize_char<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        match self.value {
            MatValue::String(s) => {
                let mut iter = s.chars();
                let c = iter.next().ok_or_else(|| {
                    MatError::Custom("expected single char, got empty string".into())
                })?;
                if iter.next().is_some() {
                    return Err(MatError::Custom(
                        "expected single char, got multi-char string".into(),
                    ));
                }
                visitor.visit_char(c)
            }
            other => mismatch("char", other),
        }
    }

    fn deserialize_str<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        self.deserialize_string(visitor)
    }

    fn deserialize_string<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        match self.value {
            MatValue::String(s) => visitor.visit_string(s),
            other => mismatch("string", other),
        }
    }

    fn deserialize_bytes<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        self.deserialize_byte_buf(visitor)
    }

    fn deserialize_byte_buf<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        match self.value {
            MatValue::Vec1D(NumVec::U8(v)) => visitor.visit_byte_buf(v),
            other => mismatch("bytes", other),
        }
    }

    // ----- option / unit -----

    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        match self.value {
            MatValue::Omit => visitor.visit_none(),
            other => visitor.visit_some(MatValueDeserializer::new(other)),
        }
    }

    fn deserialize_unit<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        match self.value {
            MatValue::Omit => visitor.visit_unit(),
            other => mismatch("unit", other),
        }
    }

    fn deserialize_unit_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, MatError> {
        self.deserialize_unit(visitor)
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, MatError> {
        visitor.visit_newtype_struct(self)
    }

    // ----- collections -----

    fn deserialize_seq<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        match self.value {
            MatValue::Vec1D(v) => visitor.visit_seq(Vec1DSeq::new(v)),
            MatValue::Matrix { rows, cols, vec } => {
                visitor.visit_seq(MatrixRowsSeq::new(rows, cols, vec))
            }
            MatValue::ComplexVec64(pairs) => {
                visitor.visit_seq(ComplexPairsSeq::Vec64(pairs.into_iter().collect()))
            }
            MatValue::ComplexVec32(pairs) => {
                visitor.visit_seq(ComplexPairsSeq::Vec32(pairs.into_iter().collect()))
            }
            MatValue::ComplexMatrix64 { rows, cols, pairs } => {
                visitor.visit_seq(ComplexMatrixRowsSeq::new64(rows, cols, pairs))
            }
            MatValue::ComplexMatrix32 { rows, cols, pairs } => {
                visitor.visit_seq(ComplexMatrixRowsSeq::new32(rows, cols, pairs))
            }
            // A scalar can deserialize as a length-1 Vec.
            MatValue::Scalar(s) => {
                let v = NumVec::from_single(s);
                visitor.visit_seq(Vec1DSeq::new(v))
            }
            other => mismatch("sequence", other),
        }
    }

    fn deserialize_tuple<V: Visitor<'de>>(
        self,
        _len: usize,
        visitor: V,
    ) -> Result<V::Value, MatError> {
        self.deserialize_seq(visitor)
    }

    fn deserialize_tuple_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _len: usize,
        visitor: V,
    ) -> Result<V::Value, MatError> {
        self.deserialize_seq(visitor)
    }

    fn deserialize_map<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        match self.value {
            MatValue::Struct(fields) => visitor.visit_map(StructMap::new(fields)),
            other => mismatch("map/struct", other),
        }
    }

    fn deserialize_struct<V: Visitor<'de>>(
        self,
        name: &'static str,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, MatError> {
        match name {
            MATRIX_SENTINEL => match self.value {
                MatValue::Matrix { rows, cols, vec } => {
                    visitor.visit_map(MatrixStructMap::new(rows, cols, vec))
                }
                // Allow deserializing a scalar or 1-D vec as a Matrix
                // (treating it as rows=1 or single-element).
                MatValue::Vec1D(v) => {
                    let len = v.len();
                    visitor.visit_map(MatrixStructMap::new(1, len, v))
                }
                MatValue::Scalar(s) => {
                    let v = NumVec::from_single(s);
                    visitor.visit_map(MatrixStructMap::new(1, 1, v))
                }
                other => mismatch("Matrix", other),
            },
            COMPLEX64_SENTINEL => match self.value {
                MatValue::ComplexScalar64 { re, im } => {
                    visitor.visit_map(ComplexStructMap64::new(re, im))
                }
                MatValue::Scalar(ScalarNum::F64(re)) => {
                    visitor.visit_map(ComplexStructMap64::new(re, 0.0))
                }
                other => mismatch("Complex64", other),
            },
            COMPLEX32_SENTINEL => match self.value {
                MatValue::ComplexScalar32 { re, im } => {
                    visitor.visit_map(ComplexStructMap32::new(re, im))
                }
                MatValue::Scalar(ScalarNum::F32(re)) => {
                    visitor.visit_map(ComplexStructMap32::new(re, 0.0))
                }
                other => mismatch("Complex32", other),
            },
            _ => {
                // Plain struct: deserialize from a MATLAB struct group.
                let _ = fields;
                self.deserialize_map(visitor)
            }
        }
    }

    fn deserialize_enum<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, MatError> {
        match self.value {
            MatValue::String(s) => visitor.visit_enum(UnitVariantAccess(s)),
            other => Err(MatError::UnsupportedType(match other.kind() {
                "struct" => "struct enum variant (not supported in v1)",
                _ => "non-unit enum variant",
            })),
        }
    }

    fn deserialize_identifier<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        match self.value {
            MatValue::String(s) => visitor.visit_string(s),
            other => mismatch("identifier", other),
        }
    }

    fn deserialize_ignored_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        let _ = self.value;
        visitor.visit_unit()
    }
}

// ---------------------------------------------------------------------------
// Dispatch for deserialize_any
// ---------------------------------------------------------------------------

fn dispatch_any<'de, V: Visitor<'de>>(value: MatValue, visitor: V) -> Result<V::Value, MatError> {
    match value {
        MatValue::Omit => visitor.visit_none(),
        MatValue::Scalar(s) => match s {
            ScalarNum::Bool(b) => visitor.visit_bool(b),
            ScalarNum::F64(x) => visitor.visit_f64(x),
            ScalarNum::F32(x) => visitor.visit_f32(x),
            ScalarNum::I64(x) => visitor.visit_i64(x),
            ScalarNum::I32(x) => visitor.visit_i32(x),
            ScalarNum::I16(x) => visitor.visit_i16(x),
            ScalarNum::I8(x) => visitor.visit_i8(x),
            ScalarNum::U64(x) => visitor.visit_u64(x),
            ScalarNum::U32(x) => visitor.visit_u32(x),
            ScalarNum::U16(x) => visitor.visit_u16(x),
            ScalarNum::U8(x) => visitor.visit_u8(x),
        },
        MatValue::String(s) => visitor.visit_string(s),
        MatValue::Vec1D(v) => visitor.visit_seq(Vec1DSeq::new(v)),
        MatValue::Matrix { rows, cols, vec } => {
            visitor.visit_seq(MatrixRowsSeq::new(rows, cols, vec))
        }
        MatValue::ComplexScalar64 { re, im } => {
            visitor.visit_map(ComplexStructMap64::new(re, im))
        }
        MatValue::ComplexScalar32 { re, im } => {
            visitor.visit_map(ComplexStructMap32::new(re, im))
        }
        MatValue::ComplexVec64(pairs) => {
            visitor.visit_seq(ComplexPairsSeq::Vec64(pairs.into_iter().collect()))
        }
        MatValue::ComplexVec32(pairs) => {
            visitor.visit_seq(ComplexPairsSeq::Vec32(pairs.into_iter().collect()))
        }
        MatValue::ComplexMatrix64 { rows, cols, pairs } => {
            visitor.visit_seq(ComplexMatrixRowsSeq::new64(rows, cols, pairs))
        }
        MatValue::ComplexMatrix32 { rows, cols, pairs } => {
            visitor.visit_seq(ComplexMatrixRowsSeq::new32(rows, cols, pairs))
        }
        MatValue::Struct(fields) => visitor.visit_map(StructMap::new(fields)),
    }
}

// ---------------------------------------------------------------------------
// Helper coercions
// ---------------------------------------------------------------------------

fn mismatch<'de, T>(expected: &'static str, got: MatValue) -> Result<T, MatError> {
    Err(MatError::Custom(format!(
        "expected {expected}, got {}",
        got.kind()
    )))
}

fn to_i64(v: MatValue, expected: &'static str) -> Result<i64, MatError> {
    match v {
        MatValue::Scalar(ScalarNum::I8(x)) => Ok(x as i64),
        MatValue::Scalar(ScalarNum::I16(x)) => Ok(x as i64),
        MatValue::Scalar(ScalarNum::I32(x)) => Ok(x as i64),
        MatValue::Scalar(ScalarNum::I64(x)) => Ok(x),
        MatValue::Scalar(ScalarNum::U8(x)) => Ok(x as i64),
        MatValue::Scalar(ScalarNum::U16(x)) => Ok(x as i64),
        MatValue::Scalar(ScalarNum::U32(x)) => Ok(x as i64),
        MatValue::Scalar(ScalarNum::U64(x)) if x <= i64::MAX as u64 => Ok(x as i64),
        MatValue::Scalar(ScalarNum::F64(x)) => Ok(x as i64),
        MatValue::Scalar(ScalarNum::F32(x)) => Ok(x as i64),
        MatValue::Scalar(ScalarNum::Bool(b)) => Ok(i64::from(b)),
        other => mismatch(expected, other),
    }
}

fn to_u64(v: MatValue, expected: &'static str) -> Result<u64, MatError> {
    match v {
        MatValue::Scalar(ScalarNum::U8(x)) => Ok(x as u64),
        MatValue::Scalar(ScalarNum::U16(x)) => Ok(x as u64),
        MatValue::Scalar(ScalarNum::U32(x)) => Ok(x as u64),
        MatValue::Scalar(ScalarNum::U64(x)) => Ok(x),
        MatValue::Scalar(ScalarNum::I8(x)) if x >= 0 => Ok(x as u64),
        MatValue::Scalar(ScalarNum::I16(x)) if x >= 0 => Ok(x as u64),
        MatValue::Scalar(ScalarNum::I32(x)) if x >= 0 => Ok(x as u64),
        MatValue::Scalar(ScalarNum::I64(x)) if x >= 0 => Ok(x as u64),
        MatValue::Scalar(ScalarNum::F64(x)) if x >= 0.0 => Ok(x as u64),
        MatValue::Scalar(ScalarNum::F32(x)) if x >= 0.0 => Ok(x as u64),
        MatValue::Scalar(ScalarNum::Bool(b)) => Ok(u64::from(b)),
        other => mismatch(expected, other),
    }
}

fn to_f64(v: MatValue, expected: &'static str) -> Result<f64, MatError> {
    match v {
        MatValue::Scalar(ScalarNum::F64(x)) => Ok(x),
        MatValue::Scalar(ScalarNum::F32(x)) => Ok(x as f64),
        MatValue::Scalar(ScalarNum::I64(x)) => Ok(x as f64),
        MatValue::Scalar(ScalarNum::I32(x)) => Ok(x as f64),
        MatValue::Scalar(ScalarNum::I16(x)) => Ok(x as f64),
        MatValue::Scalar(ScalarNum::I8(x)) => Ok(x as f64),
        MatValue::Scalar(ScalarNum::U64(x)) => Ok(x as f64),
        MatValue::Scalar(ScalarNum::U32(x)) => Ok(x as f64),
        MatValue::Scalar(ScalarNum::U16(x)) => Ok(x as f64),
        MatValue::Scalar(ScalarNum::U8(x)) => Ok(x as f64),
        MatValue::Scalar(ScalarNum::Bool(b)) => Ok(if b { 1.0 } else { 0.0 }),
        other => mismatch(expected, other),
    }
}

// ---------------------------------------------------------------------------
// Vec1D SeqAccess
// ---------------------------------------------------------------------------

struct Vec1DSeq {
    items: VecDeque<MatValue>,
}

impl Vec1DSeq {
    fn new(v: NumVec) -> Self {
        let items = match v {
            NumVec::Bool(vs) => vs
                .into_iter()
                .map(|b| MatValue::Scalar(ScalarNum::Bool(b)))
                .collect(),
            NumVec::F64(vs) => vs
                .into_iter()
                .map(|x| MatValue::Scalar(ScalarNum::F64(x)))
                .collect(),
            NumVec::F32(vs) => vs
                .into_iter()
                .map(|x| MatValue::Scalar(ScalarNum::F32(x)))
                .collect(),
            NumVec::I64(vs) => vs
                .into_iter()
                .map(|x| MatValue::Scalar(ScalarNum::I64(x)))
                .collect(),
            NumVec::I32(vs) => vs
                .into_iter()
                .map(|x| MatValue::Scalar(ScalarNum::I32(x)))
                .collect(),
            NumVec::I16(vs) => vs
                .into_iter()
                .map(|x| MatValue::Scalar(ScalarNum::I16(x)))
                .collect(),
            NumVec::I8(vs) => vs
                .into_iter()
                .map(|x| MatValue::Scalar(ScalarNum::I8(x)))
                .collect(),
            NumVec::U64(vs) => vs
                .into_iter()
                .map(|x| MatValue::Scalar(ScalarNum::U64(x)))
                .collect(),
            NumVec::U32(vs) => vs
                .into_iter()
                .map(|x| MatValue::Scalar(ScalarNum::U32(x)))
                .collect(),
            NumVec::U16(vs) => vs
                .into_iter()
                .map(|x| MatValue::Scalar(ScalarNum::U16(x)))
                .collect(),
            NumVec::U8(vs) => vs
                .into_iter()
                .map(|x| MatValue::Scalar(ScalarNum::U8(x)))
                .collect(),
        };
        Self { items }
    }
}

impl<'de> SeqAccess<'de> for Vec1DSeq {
    type Error = MatError;
    fn next_element_seed<T: DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>, MatError> {
        match self.items.pop_front() {
            Some(v) => seed.deserialize(MatValueDeserializer::new(v)).map(Some),
            None => Ok(None),
        }
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.items.len())
    }
}

// ---------------------------------------------------------------------------
// MatrixRows SeqAccess: yields each row as a Vec1D
// ---------------------------------------------------------------------------

struct MatrixRowsSeq {
    rows_remaining: usize,
    cols: usize,
    row_major_iter: NumVecIter,
}

impl MatrixRowsSeq {
    fn new(rows: usize, cols: usize, vec: NumVec) -> Self {
        Self {
            rows_remaining: rows,
            cols,
            row_major_iter: NumVecIter::new(vec),
        }
    }
}

impl<'de> SeqAccess<'de> for MatrixRowsSeq {
    type Error = MatError;
    fn next_element_seed<T: DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>, MatError> {
        if self.rows_remaining == 0 {
            return Ok(None);
        }
        self.rows_remaining -= 1;
        let row = self.row_major_iter.take_n(self.cols);
        seed.deserialize(MatValueDeserializer::new(MatValue::Vec1D(row)))
            .map(Some)
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.rows_remaining)
    }
}

/// Iterator-by-vec-tag: pops the front `n` elements each call.
struct NumVecIter {
    inner: NumVec,
}
impl NumVecIter {
    fn new(v: NumVec) -> Self {
        Self { inner: v }
    }

    fn take_n(&mut self, n: usize) -> NumVec {
        macro_rules! split {
            ($variant:ident, $vec:expr) => {{
                let tail = $vec.split_off(n);
                let head = std::mem::replace($vec, tail);
                NumVec::$variant(head)
            }};
        }
        match &mut self.inner {
            NumVec::Bool(v) => split!(Bool, v),
            NumVec::F64(v) => split!(F64, v),
            NumVec::F32(v) => split!(F32, v),
            NumVec::I64(v) => split!(I64, v),
            NumVec::I32(v) => split!(I32, v),
            NumVec::I16(v) => split!(I16, v),
            NumVec::I8(v) => split!(I8, v),
            NumVec::U64(v) => split!(U64, v),
            NumVec::U32(v) => split!(U32, v),
            NumVec::U16(v) => split!(U16, v),
            NumVec::U8(v) => split!(U8, v),
        }
    }
}

// ---------------------------------------------------------------------------
// ComplexPairs SeqAccess
// ---------------------------------------------------------------------------

enum ComplexPairsSeq {
    Vec64(VecDeque<(f64, f64)>),
    Vec32(VecDeque<(f32, f32)>),
}

impl<'de> SeqAccess<'de> for ComplexPairsSeq {
    type Error = MatError;
    fn next_element_seed<T: DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>, MatError> {
        match self {
            ComplexPairsSeq::Vec64(q) => match q.pop_front() {
                Some((re, im)) => seed
                    .deserialize(MatValueDeserializer::new(MatValue::ComplexScalar64 {
                        re,
                        im,
                    }))
                    .map(Some),
                None => Ok(None),
            },
            ComplexPairsSeq::Vec32(q) => match q.pop_front() {
                Some((re, im)) => seed
                    .deserialize(MatValueDeserializer::new(MatValue::ComplexScalar32 {
                        re,
                        im,
                    }))
                    .map(Some),
                None => Ok(None),
            },
        }
    }
    fn size_hint(&self) -> Option<usize> {
        Some(match self {
            ComplexPairsSeq::Vec64(q) => q.len(),
            ComplexPairsSeq::Vec32(q) => q.len(),
        })
    }
}

// ---------------------------------------------------------------------------
// ComplexMatrixRows SeqAccess: yields each row as ComplexVec*
// ---------------------------------------------------------------------------

enum ComplexMatrixRowsSeq {
    Mat64 {
        rows_remaining: usize,
        cols: usize,
        row_major: Vec<(f64, f64)>, // consumed front-to-back
    },
    Mat32 {
        rows_remaining: usize,
        cols: usize,
        row_major: Vec<(f32, f32)>,
    },
}

impl ComplexMatrixRowsSeq {
    fn new64(rows: usize, cols: usize, row_major: Vec<(f64, f64)>) -> Self {
        ComplexMatrixRowsSeq::Mat64 {
            rows_remaining: rows,
            cols,
            row_major,
        }
    }
    fn new32(rows: usize, cols: usize, row_major: Vec<(f32, f32)>) -> Self {
        ComplexMatrixRowsSeq::Mat32 {
            rows_remaining: rows,
            cols,
            row_major,
        }
    }
}

impl<'de> SeqAccess<'de> for ComplexMatrixRowsSeq {
    type Error = MatError;
    fn next_element_seed<T: DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>, MatError> {
        match self {
            ComplexMatrixRowsSeq::Mat64 {
                rows_remaining,
                cols,
                row_major,
            } => {
                if *rows_remaining == 0 {
                    return Ok(None);
                }
                *rows_remaining -= 1;
                let tail = row_major.split_off(*cols);
                let head = std::mem::replace(row_major, tail);
                seed.deserialize(MatValueDeserializer::new(MatValue::ComplexVec64(head)))
                    .map(Some)
            }
            ComplexMatrixRowsSeq::Mat32 {
                rows_remaining,
                cols,
                row_major,
            } => {
                if *rows_remaining == 0 {
                    return Ok(None);
                }
                *rows_remaining -= 1;
                let tail = row_major.split_off(*cols);
                let head = std::mem::replace(row_major, tail);
                seed.deserialize(MatValueDeserializer::new(MatValue::ComplexVec32(head)))
                    .map(Some)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Struct MapAccess
// ---------------------------------------------------------------------------

struct StructMap {
    fields: VecDeque<(String, MatValue)>,
    pending_value: Option<MatValue>,
}

impl StructMap {
    fn new(fields: Vec<(String, MatValue)>) -> Self {
        Self {
            fields: fields.into_iter().collect(),
            pending_value: None,
        }
    }
}

impl<'de> MapAccess<'de> for StructMap {
    type Error = MatError;
    fn next_key_seed<K: DeserializeSeed<'de>>(
        &mut self,
        seed: K,
    ) -> Result<Option<K::Value>, MatError> {
        match self.fields.pop_front() {
            Some((k, v)) => {
                self.pending_value = Some(v);
                seed.deserialize(StringRefDe(k)).map(Some)
            }
            None => Ok(None),
        }
    }
    fn next_value_seed<V: DeserializeSeed<'de>>(
        &mut self,
        seed: V,
    ) -> Result<V::Value, MatError> {
        let v = self
            .pending_value
            .take()
            .ok_or_else(|| MatError::Custom("next_value before next_key".into()))?;
        seed.deserialize(MatValueDeserializer::new(v))
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.fields.len())
    }
}

/// Small helper Deserializer that yields a String as identifier/str.
struct StringRefDe(String);

impl<'de> Deserializer<'de> for StringRefDe {
    type Error = MatError;
    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        visitor.visit_string(self.0)
    }
    forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        bytes byte_buf option unit unit_struct newtype_struct seq tuple
        tuple_struct map struct enum identifier ignored_any
    }
}

// ---------------------------------------------------------------------------
// Matrix sentinel MapAccess: yields {rows, cols, data}
// ---------------------------------------------------------------------------

struct MatrixStructMap {
    state: MatrixState,
    rows: usize,
    cols: usize,
    data: Option<NumVec>,
}

enum MatrixState {
    NeedRowsKey,
    NeedRowsValue,
    NeedColsKey,
    NeedColsValue,
    NeedDataKey,
    NeedDataValue,
    Done,
}

impl MatrixStructMap {
    fn new(rows: usize, cols: usize, vec: NumVec) -> Self {
        Self {
            state: MatrixState::NeedRowsKey,
            rows,
            cols,
            data: Some(vec),
        }
    }
}

impl<'de> MapAccess<'de> for MatrixStructMap {
    type Error = MatError;

    fn next_key_seed<K: DeserializeSeed<'de>>(
        &mut self,
        seed: K,
    ) -> Result<Option<K::Value>, MatError> {
        let (key, next_state) = match self.state {
            MatrixState::NeedRowsKey => ("rows", MatrixState::NeedRowsValue),
            MatrixState::NeedColsKey => ("cols", MatrixState::NeedColsValue),
            MatrixState::NeedDataKey => ("data", MatrixState::NeedDataValue),
            MatrixState::Done => return Ok(None),
            _ => return Err(MatError::Custom("matrix map state desync".into())),
        };
        self.state = next_state;
        seed.deserialize(StringRefDe(key.to_string())).map(Some)
    }

    fn next_value_seed<V: DeserializeSeed<'de>>(
        &mut self,
        seed: V,
    ) -> Result<V::Value, MatError> {
        match self.state {
            MatrixState::NeedRowsValue => {
                self.state = MatrixState::NeedColsKey;
                seed.deserialize((self.rows as u64).into_deserializer())
            }
            MatrixState::NeedColsValue => {
                self.state = MatrixState::NeedDataKey;
                seed.deserialize((self.cols as u64).into_deserializer())
            }
            MatrixState::NeedDataValue => {
                self.state = MatrixState::Done;
                let data = self.data.take().unwrap();
                seed.deserialize(MatValueDeserializer::new(MatValue::Vec1D(data)))
            }
            _ => Err(MatError::Custom("matrix map value before key".into())),
        }
    }

    fn size_hint(&self) -> Option<usize> {
        Some(match self.state {
            MatrixState::NeedRowsKey | MatrixState::NeedRowsValue => 3,
            MatrixState::NeedColsKey | MatrixState::NeedColsValue => 2,
            MatrixState::NeedDataKey | MatrixState::NeedDataValue => 1,
            MatrixState::Done => 0,
        })
    }
}

// ---------------------------------------------------------------------------
// Complex sentinel MapAccess
// ---------------------------------------------------------------------------

struct ComplexStructMap64 {
    state: ComplexState,
    re: f64,
    im: f64,
}

struct ComplexStructMap32 {
    state: ComplexState,
    re: f32,
    im: f32,
}

enum ComplexState {
    NeedRealKey,
    NeedRealValue,
    NeedImagKey,
    NeedImagValue,
    Done,
}

impl ComplexStructMap64 {
    fn new(re: f64, im: f64) -> Self {
        Self {
            state: ComplexState::NeedRealKey,
            re,
            im,
        }
    }
}
impl ComplexStructMap32 {
    fn new(re: f32, im: f32) -> Self {
        Self {
            state: ComplexState::NeedRealKey,
            re,
            im,
        }
    }
}

macro_rules! impl_complex_map {
    ($map:ty, $re:ty, $de:expr) => {
        impl<'de> MapAccess<'de> for $map {
            type Error = MatError;

            fn next_key_seed<K: DeserializeSeed<'de>>(
                &mut self,
                seed: K,
            ) -> Result<Option<K::Value>, MatError> {
                let (key, next) = match self.state {
                    ComplexState::NeedRealKey => ("real", ComplexState::NeedRealValue),
                    ComplexState::NeedImagKey => ("imag", ComplexState::NeedImagValue),
                    ComplexState::Done => return Ok(None),
                    _ => return Err(MatError::Custom("complex map state desync".into())),
                };
                self.state = next;
                seed.deserialize(StringRefDe(key.to_string())).map(Some)
            }
            fn next_value_seed<V: DeserializeSeed<'de>>(
                &mut self,
                seed: V,
            ) -> Result<V::Value, MatError> {
                match self.state {
                    ComplexState::NeedRealValue => {
                        self.state = ComplexState::NeedImagKey;
                        seed.deserialize($de(self.re))
                    }
                    ComplexState::NeedImagValue => {
                        self.state = ComplexState::Done;
                        seed.deserialize($de(self.im))
                    }
                    _ => Err(MatError::Custom("complex map value before key".into())),
                }
            }
            fn size_hint(&self) -> Option<usize> {
                Some(match self.state {
                    ComplexState::NeedRealKey | ComplexState::NeedRealValue => 2,
                    ComplexState::NeedImagKey | ComplexState::NeedImagValue => 1,
                    ComplexState::Done => 0,
                })
            }
        }
        // Touch $re so the pattern arg is reachable for documentation of the
        // type being produced; this is a no-op at runtime.
        #[allow(dead_code)]
        const _: fn($re) = |_| {};
    };
}

impl_complex_map!(
    ComplexStructMap64,
    f64,
    |x: f64| x.into_deserializer()
);
impl_complex_map!(
    ComplexStructMap32,
    f32,
    |x: f32| x.into_deserializer()
);

// ---------------------------------------------------------------------------
// Enum unit variant
// ---------------------------------------------------------------------------

struct UnitVariantAccess(String);

impl<'de> EnumAccess<'de> for UnitVariantAccess {
    type Error = MatError;
    type Variant = UnitVariantOnly;
    fn variant_seed<V: DeserializeSeed<'de>>(
        self,
        seed: V,
    ) -> Result<(V::Value, UnitVariantOnly), MatError> {
        let value = seed.deserialize(StringRefDe(self.0))?;
        Ok((value, UnitVariantOnly))
    }
}

struct UnitVariantOnly;

impl<'de> VariantAccess<'de> for UnitVariantOnly {
    type Error = MatError;
    fn unit_variant(self) -> Result<(), MatError> {
        Ok(())
    }
    fn newtype_variant_seed<T: DeserializeSeed<'de>>(self, _seed: T) -> Result<T::Value, MatError> {
        Err(MatError::UnsupportedType("newtype enum variant"))
    }
    fn tuple_variant<V: Visitor<'de>>(
        self,
        _len: usize,
        _visitor: V,
    ) -> Result<V::Value, MatError> {
        Err(MatError::UnsupportedType("tuple enum variant"))
    }
    fn struct_variant<V: Visitor<'de>>(
        self,
        _fields: &'static [&'static str],
        _visitor: V,
    ) -> Result<V::Value, MatError> {
        Err(MatError::UnsupportedType("struct enum variant"))
    }
}

// ---------------------------------------------------------------------------
// NumVec helper: build a 1-element NumVec from a scalar
// ---------------------------------------------------------------------------

impl NumVec {
    pub(crate) fn from_single(s: ScalarNum) -> NumVec {
        match s {
            ScalarNum::Bool(b) => NumVec::Bool(vec![b]),
            ScalarNum::F64(x) => NumVec::F64(vec![x]),
            ScalarNum::F32(x) => NumVec::F32(vec![x]),
            ScalarNum::I64(x) => NumVec::I64(vec![x]),
            ScalarNum::I32(x) => NumVec::I32(vec![x]),
            ScalarNum::I16(x) => NumVec::I16(vec![x]),
            ScalarNum::I8(x) => NumVec::I8(vec![x]),
            ScalarNum::U64(x) => NumVec::U64(vec![x]),
            ScalarNum::U32(x) => NumVec::U32(vec![x]),
            ScalarNum::U16(x) => NumVec::U16(vec![x]),
            ScalarNum::U8(x) => NumVec::U8(vec![x]),
        }
    }
}

// Silence unused imports in some builds.
#[allow(dead_code)]
fn _touch<E: de::Error>() -> E {
    E::custom("x")
}
