//! Intermediate value tree built by the serializer, later emitted to an
//! HDF5 file with MATLAB conventions.

use crate::mat::error::MatError;

/// A scalar numeric value tagged by its Rust type.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ScalarNum {
    Bool(bool),
    F64(f64),
    F32(f32),
    I64(i64),
    I32(i32),
    I16(i16),
    I8(i8),
    U64(u64),
    U32(u32),
    U16(u16),
    U8(u8),
}

impl ScalarNum {
    pub(crate) fn tag(&self) -> ScalarTag {
        match self {
            ScalarNum::Bool(_) => ScalarTag::Bool,
            ScalarNum::F64(_) => ScalarTag::F64,
            ScalarNum::F32(_) => ScalarTag::F32,
            ScalarNum::I64(_) => ScalarTag::I64,
            ScalarNum::I32(_) => ScalarTag::I32,
            ScalarNum::I16(_) => ScalarTag::I16,
            ScalarNum::I8(_) => ScalarTag::I8,
            ScalarNum::U64(_) => ScalarTag::U64,
            ScalarNum::U32(_) => ScalarTag::U32,
            ScalarNum::U16(_) => ScalarTag::U16,
            ScalarNum::U8(_) => ScalarTag::U8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScalarTag {
    Bool,
    F64,
    F32,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
}

/// A typed 1-D array of a single primitive class.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum NumVec {
    Bool(Vec<bool>),
    F64(Vec<f64>),
    F32(Vec<f32>),
    I64(Vec<i64>),
    I32(Vec<i32>),
    I16(Vec<i16>),
    I8(Vec<i8>),
    U64(Vec<u64>),
    U32(Vec<u32>),
    U16(Vec<u16>),
    U8(Vec<u8>),
}

impl NumVec {
    pub(crate) fn len(&self) -> usize {
        match self {
            NumVec::Bool(v) => v.len(),
            NumVec::F64(v) => v.len(),
            NumVec::F32(v) => v.len(),
            NumVec::I64(v) => v.len(),
            NumVec::I32(v) => v.len(),
            NumVec::I16(v) => v.len(),
            NumVec::I8(v) => v.len(),
            NumVec::U64(v) => v.len(),
            NumVec::U32(v) => v.len(),
            NumVec::U16(v) => v.len(),
            NumVec::U8(v) => v.len(),
        }
    }

    pub(crate) fn tag(&self) -> ScalarTag {
        match self {
            NumVec::Bool(_) => ScalarTag::Bool,
            NumVec::F64(_) => ScalarTag::F64,
            NumVec::F32(_) => ScalarTag::F32,
            NumVec::I64(_) => ScalarTag::I64,
            NumVec::I32(_) => ScalarTag::I32,
            NumVec::I16(_) => ScalarTag::I16,
            NumVec::I8(_) => ScalarTag::I8,
            NumVec::U64(_) => ScalarTag::U64,
            NumVec::U32(_) => ScalarTag::U32,
            NumVec::U16(_) => ScalarTag::U16,
            NumVec::U8(_) => ScalarTag::U8,
        }
    }

    pub(crate) fn empty_with_tag(tag: ScalarTag) -> Self {
        match tag {
            ScalarTag::Bool => NumVec::Bool(Vec::new()),
            ScalarTag::F64 => NumVec::F64(Vec::new()),
            ScalarTag::F32 => NumVec::F32(Vec::new()),
            ScalarTag::I64 => NumVec::I64(Vec::new()),
            ScalarTag::I32 => NumVec::I32(Vec::new()),
            ScalarTag::I16 => NumVec::I16(Vec::new()),
            ScalarTag::I8 => NumVec::I8(Vec::new()),
            ScalarTag::U64 => NumVec::U64(Vec::new()),
            ScalarTag::U32 => NumVec::U32(Vec::new()),
            ScalarTag::U16 => NumVec::U16(Vec::new()),
            ScalarTag::U8 => NumVec::U8(Vec::new()),
        }
    }

    /// Push a scalar into this vec, requiring matching tags.
    pub(crate) fn push(&mut self, v: ScalarNum) -> Result<(), MatError> {
        match (self, v) {
            (NumVec::Bool(vec), ScalarNum::Bool(x)) => vec.push(x),
            (NumVec::F64(vec), ScalarNum::F64(x)) => vec.push(x),
            (NumVec::F32(vec), ScalarNum::F32(x)) => vec.push(x),
            (NumVec::I64(vec), ScalarNum::I64(x)) => vec.push(x),
            (NumVec::I32(vec), ScalarNum::I32(x)) => vec.push(x),
            (NumVec::I16(vec), ScalarNum::I16(x)) => vec.push(x),
            (NumVec::I8(vec), ScalarNum::I8(x)) => vec.push(x),
            (NumVec::U64(vec), ScalarNum::U64(x)) => vec.push(x),
            (NumVec::U32(vec), ScalarNum::U32(x)) => vec.push(x),
            (NumVec::U16(vec), ScalarNum::U16(x)) => vec.push(x),
            (NumVec::U8(vec), ScalarNum::U8(x)) => vec.push(x),
            _ => return Err(MatError::MixedSequenceElementTypes),
        }
        Ok(())
    }

    /// Append another vec of the same tag.
    pub(crate) fn extend(&mut self, other: NumVec) -> Result<(), MatError> {
        match (self, other) {
            (NumVec::Bool(a), NumVec::Bool(b)) => a.extend(b),
            (NumVec::F64(a), NumVec::F64(b)) => a.extend(b),
            (NumVec::F32(a), NumVec::F32(b)) => a.extend(b),
            (NumVec::I64(a), NumVec::I64(b)) => a.extend(b),
            (NumVec::I32(a), NumVec::I32(b)) => a.extend(b),
            (NumVec::I16(a), NumVec::I16(b)) => a.extend(b),
            (NumVec::I8(a), NumVec::I8(b)) => a.extend(b),
            (NumVec::U64(a), NumVec::U64(b)) => a.extend(b),
            (NumVec::U32(a), NumVec::U32(b)) => a.extend(b),
            (NumVec::U16(a), NumVec::U16(b)) => a.extend(b),
            (NumVec::U8(a), NumVec::U8(b)) => a.extend(b),
            _ => return Err(MatError::MixedSequenceElementTypes),
        }
        Ok(())
    }
}

/// Intermediate tree node produced by the value serializer.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum MatValue {
    /// Placeholder for `Option::None` — the containing struct drops the field.
    Omit,
    /// Numeric / logical scalar.
    Scalar(ScalarNum),
    /// 1-D numeric array → stored as `[1, N]`.
    Vec1D(NumVec),
    /// 2-D numeric array in row-major order → stored column-major with shape
    /// `[cols, rows]`.
    Matrix {
        rows: usize,
        cols: usize,
        vec: NumVec,
    },
    /// UTF-16 `char` string → stored as `uint16 [1, N]`.
    String(String),
    /// Complex scalar (`double` class).
    ComplexScalar64 { re: f64, im: f64 },
    /// Complex scalar (`single` class).
    ComplexScalar32 { re: f32, im: f32 },
    /// Complex 1-D array (`double`). Pairs laid out `[(re, im), ...]`.
    ComplexVec64(Vec<(f64, f64)>),
    /// Complex 1-D array (`single`).
    ComplexVec32(Vec<(f32, f32)>),
    /// Complex 2-D matrix (`double`), row-major pairs.
    ComplexMatrix64 {
        rows: usize,
        cols: usize,
        pairs: Vec<(f64, f64)>,
    },
    /// Complex 2-D matrix (`single`).
    ComplexMatrix32 {
        rows: usize,
        cols: usize,
        pairs: Vec<(f32, f32)>,
    },
    /// Ordered, named fields — serialized as a MATLAB struct group.
    Struct(Vec<(String, MatValue)>),
}

impl MatValue {
    /// Return a short human-readable description for error messages.
    pub(crate) fn kind(&self) -> &'static str {
        match self {
            MatValue::Omit => "none",
            MatValue::Scalar(_) => "scalar",
            MatValue::Vec1D(_) => "1-D vector",
            MatValue::Matrix { .. } => "2-D matrix",
            MatValue::String(_) => "string",
            MatValue::ComplexScalar64 { .. } | MatValue::ComplexScalar32 { .. } => {
                "complex scalar"
            }
            MatValue::ComplexVec64(_) | MatValue::ComplexVec32(_) => "complex vector",
            MatValue::ComplexMatrix64 { .. } | MatValue::ComplexMatrix32 { .. } => {
                "complex matrix"
            }
            MatValue::Struct(_) => "struct",
        }
    }
}
