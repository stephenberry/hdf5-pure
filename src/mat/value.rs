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

    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the scalar at `index`, or `None` if out of bounds. Lets callers
    /// walk a numeric vector element-by-element without materializing a
    /// `Vec<MatValue>` of boxed scalars.
    pub(crate) fn get(&self, index: usize) -> Option<ScalarNum> {
        match self {
            NumVec::Bool(v) => v.get(index).copied().map(ScalarNum::Bool),
            NumVec::F64(v) => v.get(index).copied().map(ScalarNum::F64),
            NumVec::F32(v) => v.get(index).copied().map(ScalarNum::F32),
            NumVec::I64(v) => v.get(index).copied().map(ScalarNum::I64),
            NumVec::I32(v) => v.get(index).copied().map(ScalarNum::I32),
            NumVec::I16(v) => v.get(index).copied().map(ScalarNum::I16),
            NumVec::I8(v) => v.get(index).copied().map(ScalarNum::I8),
            NumVec::U64(v) => v.get(index).copied().map(ScalarNum::U64),
            NumVec::U32(v) => v.get(index).copied().map(ScalarNum::U32),
            NumVec::U16(v) => v.get(index).copied().map(ScalarNum::U16),
            NumVec::U8(v) => v.get(index).copied().map(ScalarNum::U8),
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

    /// Like [`empty_with_tag`](Self::empty_with_tag) but reserves `cap` elements
    /// up front, so a known-size fill via [`push`](Self::push) /
    /// [`extend`](Self::extend) never reallocates.
    pub(crate) fn with_capacity_for_tag(tag: ScalarTag, cap: usize) -> Self {
        match tag {
            ScalarTag::Bool => NumVec::Bool(Vec::with_capacity(cap)),
            ScalarTag::F64 => NumVec::F64(Vec::with_capacity(cap)),
            ScalarTag::F32 => NumVec::F32(Vec::with_capacity(cap)),
            ScalarTag::I64 => NumVec::I64(Vec::with_capacity(cap)),
            ScalarTag::I32 => NumVec::I32(Vec::with_capacity(cap)),
            ScalarTag::I16 => NumVec::I16(Vec::with_capacity(cap)),
            ScalarTag::I8 => NumVec::I8(Vec::with_capacity(cap)),
            ScalarTag::U64 => NumVec::U64(Vec::with_capacity(cap)),
            ScalarTag::U32 => NumVec::U32(Vec::with_capacity(cap)),
            ScalarTag::U16 => NumVec::U16(Vec::with_capacity(cap)),
            ScalarTag::U8 => NumVec::U8(Vec::with_capacity(cap)),
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
    /// Ordered, named fields. Serialized as a MATLAB struct group.
    Struct(Vec<(String, MatValue)>),
    /// Heterogeneous sequence (`MATLAB_class = "cell"`). Each element is
    /// interned under `#refs#` and the parent dataset stores object
    /// references in element order. The IR carries no shape: the writer
    /// always emits a column-vector layout (`[n, 1]` MATLAB shape, `[1, n]`
    /// HDF5 shape), and the deserializer flattens to a 1-D sequence. If
    /// multi-dim cells ever ship, add a `dims` field then.
    Cell(Vec<MatValue>),
    /// Empty struct array placeholder for `None` inside a sequence. Renders
    /// as MATLAB's `struct([])` (a `[0, 0]` empty marker with
    /// `MATLAB_class="struct"` and `MATLAB_empty=1`).
    EmptyStructArray,
    /// A MATLAB struct *array* (`1×N` / `N×1` / `M×N` struct with fields).
    ///
    /// On disk MATLAB stores a struct array as a `MATLAB_class="struct"` group
    /// whose every field is a dataset of object references — one reference per
    /// array element — i.e. a struct-of-arrays. The reader transposes that into
    /// this array-of-structs: `elements` lists each element's fields in
    /// row-major order, and `rows`/`cols` carry the array shape so the
    /// deserializer mirrors [`MatValue::Matrix`] (a `1×N`/`N×1` array flattens
    /// to a sequence of structs → `Vec<T>`; a true `M×N` array yields a
    /// sequence of rows → `Vec<Vec<T>>`).
    ///
    /// Read-only: the serializer lowers a `Vec<Struct>` to a cell array (see
    /// [`MatValue::Cell`]), never to this native struct-array layout, so it
    /// never produces this variant.
    StructArray {
        rows: usize,
        cols: usize,
        elements: Vec<Vec<(String, MatValue)>>,
    },
    /// A decoded MATLAB MCOS opaque object (`MATLAB_object_decode = 3`).
    ///
    /// Aside from the modern `string` class (which lowers to [`MatValue::String`]
    /// / [`MatValue::Cell`]), MATLAB stores `datetime`, `duration`,
    /// `categorical`, `table`, `containers.Map`, `dictionary`, user `classdef`
    /// instances, … as opaque objects in the hidden `#subsystem#/MCOS` store.
    /// This variant carries the MATLAB class name and the object's resolved
    /// properties in declaration order:
    ///
    /// - For a class with a dedicated decoder (`datetime`, `duration`,
    ///   `categorical`) `fields` holds the decoded logical components (e.g.
    ///   datetime's `millis_utc` / `sub_ms`), which deserialize into the
    ///   matching public type ([`MatDatetime`](crate::mat::MatDatetime), …) or
    ///   any struct with the same field names.
    /// - For every other opaque class `fields` holds the raw property values,
    ///   so the object is still losslessly readable as a struct rather than
    ///   failing the whole file.
    ///
    /// Read-only: the serializer never produces this variant (writing MCOS
    /// opaque objects beyond `string` is not supported).
    Opaque {
        /// The MATLAB class name (`"datetime"`, `"categorical"`, `"table"`, …).
        class_name: String,
        /// Resolved properties in declaration order.
        fields: Vec<(String, MatValue)>,
    },
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
            MatValue::ComplexScalar64 { .. } | MatValue::ComplexScalar32 { .. } => "complex scalar",
            MatValue::ComplexVec64(_) | MatValue::ComplexVec32(_) => "complex vector",
            MatValue::ComplexMatrix64 { .. } | MatValue::ComplexMatrix32 { .. } => "complex matrix",
            MatValue::Struct(_) => "struct",
            MatValue::Cell(_) => "cell array",
            MatValue::EmptyStructArray => "empty struct array",
            MatValue::StructArray { .. } => "struct array",
            MatValue::Opaque { .. } => "opaque object",
        }
    }
}
