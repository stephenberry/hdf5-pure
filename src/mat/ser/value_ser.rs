//! `ValueSerializer`: the core `serde::Serializer` that produces a
//! [`MatValue`] from any serializable input.
//!
//! The serializer makes one pass to collect everything into the intermediate
//! tree, then the emitter walks the tree to build the HDF5 file.

use serde::ser::{
    self, Impossible, Serialize, SerializeMap, SerializeSeq, SerializeStruct, SerializeTuple,
    SerializeTupleStruct, Serializer,
};

use crate::mat::complex::{COMPLEX32_SENTINEL, COMPLEX64_SENTINEL};
use crate::mat::error::MatError;
use crate::mat::matrix::MATRIX_SENTINEL;
use crate::mat::utf16;

use crate::mat::value::{MatValue, NumVec, ScalarNum, ScalarTag};

// ---------------------------------------------------------------------------
// Public entry: serialize a value into a MatValue
// ---------------------------------------------------------------------------

pub(crate) fn to_value<T: Serialize + ?Sized>(value: &T) -> Result<MatValue, MatError> {
    value.serialize(ValueSerializer)
}

// ---------------------------------------------------------------------------
// ValueSerializer
// ---------------------------------------------------------------------------

pub(crate) struct ValueSerializer;

impl Serializer for ValueSerializer {
    type Ok = MatValue;
    type Error = MatError;

    type SerializeSeq = SeqSer;
    type SerializeTuple = SeqSer;
    type SerializeTupleStruct = SeqSer;
    type SerializeTupleVariant = Impossible<MatValue, MatError>;
    type SerializeMap = MapSer;
    type SerializeStruct = StructSer;
    type SerializeStructVariant = Impossible<MatValue, MatError>;

    // ----- primitives -----

    fn serialize_bool(self, v: bool) -> Result<MatValue, MatError> {
        Ok(MatValue::Scalar(ScalarNum::Bool(v)))
    }
    fn serialize_i8(self, v: i8) -> Result<MatValue, MatError> {
        Ok(MatValue::Scalar(ScalarNum::I8(v)))
    }
    fn serialize_i16(self, v: i16) -> Result<MatValue, MatError> {
        Ok(MatValue::Scalar(ScalarNum::I16(v)))
    }
    fn serialize_i32(self, v: i32) -> Result<MatValue, MatError> {
        Ok(MatValue::Scalar(ScalarNum::I32(v)))
    }
    fn serialize_i64(self, v: i64) -> Result<MatValue, MatError> {
        Ok(MatValue::Scalar(ScalarNum::I64(v)))
    }
    fn serialize_i128(self, _v: i128) -> Result<MatValue, MatError> {
        Err(MatError::UnsupportedType(
            "i128 (MATLAB has no 128-bit integer)",
        ))
    }
    fn serialize_u8(self, v: u8) -> Result<MatValue, MatError> {
        Ok(MatValue::Scalar(ScalarNum::U8(v)))
    }
    fn serialize_u16(self, v: u16) -> Result<MatValue, MatError> {
        Ok(MatValue::Scalar(ScalarNum::U16(v)))
    }
    fn serialize_u32(self, v: u32) -> Result<MatValue, MatError> {
        Ok(MatValue::Scalar(ScalarNum::U32(v)))
    }
    fn serialize_u64(self, v: u64) -> Result<MatValue, MatError> {
        Ok(MatValue::Scalar(ScalarNum::U64(v)))
    }
    fn serialize_u128(self, _v: u128) -> Result<MatValue, MatError> {
        Err(MatError::UnsupportedType("u128"))
    }
    fn serialize_f32(self, v: f32) -> Result<MatValue, MatError> {
        Ok(MatValue::Scalar(ScalarNum::F32(v)))
    }
    fn serialize_f64(self, v: f64) -> Result<MatValue, MatError> {
        Ok(MatValue::Scalar(ScalarNum::F64(v)))
    }

    fn serialize_char(self, v: char) -> Result<MatValue, MatError> {
        let mut buf = [0u8; 4];
        Ok(MatValue::String(v.encode_utf8(&mut buf).to_string()))
    }

    fn serialize_str(self, v: &str) -> Result<MatValue, MatError> {
        Ok(MatValue::String(v.to_owned()))
    }

    fn serialize_bytes(self, v: &[u8]) -> Result<MatValue, MatError> {
        Ok(MatValue::Vec1D(NumVec::U8(v.to_vec())))
    }

    // ----- option / unit / newtype -----

    fn serialize_none(self) -> Result<MatValue, MatError> {
        Ok(MatValue::Omit)
    }

    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<MatValue, MatError> {
        value.serialize(self)
    }

    fn serialize_unit(self) -> Result<MatValue, MatError> {
        Err(MatError::UnsupportedType("() / unit"))
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<MatValue, MatError> {
        Err(MatError::UnsupportedType("unit struct"))
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _idx: u32,
        variant: &'static str,
    ) -> Result<MatValue, MatError> {
        Ok(MatValue::String(variant.to_owned()))
    }

    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _name: &'static str,
        value: &T,
    ) -> Result<MatValue, MatError> {
        // Transparent newtype — pass through.
        value.serialize(self)
    }

    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _name: &'static str,
        _idx: u32,
        _variant: &'static str,
        _value: &T,
    ) -> Result<MatValue, MatError> {
        Err(MatError::UnsupportedType("newtype enum variant"))
    }

    // ----- sequences -----

    fn serialize_seq(self, len: Option<usize>) -> Result<SeqSer, MatError> {
        Ok(SeqSer::new(len))
    }
    fn serialize_tuple(self, len: usize) -> Result<SeqSer, MatError> {
        Ok(SeqSer::new(Some(len)))
    }
    fn serialize_tuple_struct(self, _name: &'static str, len: usize) -> Result<SeqSer, MatError> {
        Ok(SeqSer::new(Some(len)))
    }
    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _idx: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant, MatError> {
        Err(MatError::UnsupportedType("tuple enum variant"))
    }

    fn serialize_map(self, _len: Option<usize>) -> Result<MapSer, MatError> {
        Ok(MapSer::new())
    }

    fn serialize_struct(self, name: &'static str, _len: usize) -> Result<StructSer, MatError> {
        Ok(match name {
            MATRIX_SENTINEL => StructSer::Matrix(MatrixFields::default()),
            COMPLEX64_SENTINEL => StructSer::Complex64(ComplexFields::default()),
            COMPLEX32_SENTINEL => StructSer::Complex32(ComplexFields::default()),
            _ => StructSer::Plain(PlainStructFields::default()),
        })
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _idx: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant, MatError> {
        Err(MatError::UnsupportedType("struct enum variant"))
    }
}

// ---------------------------------------------------------------------------
// Sequence serializer: handles Vec<T>, [T; N], tuples, tuple structs
// ---------------------------------------------------------------------------

pub(crate) struct SeqSer {
    elements: Vec<MatValue>,
}

impl SeqSer {
    fn new(len: Option<usize>) -> Self {
        Self {
            elements: Vec::with_capacity(len.unwrap_or(0)),
        }
    }

    fn push<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), MatError> {
        let value = v.serialize(ValueSerializer)?;
        self.elements.push(value);
        Ok(())
    }

    fn finish(self) -> Result<MatValue, MatError> {
        unify_sequence(self.elements)
    }
}

impl SerializeSeq for SeqSer {
    type Ok = MatValue;
    type Error = MatError;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), MatError> {
        self.push(value)
    }
    fn end(self) -> Result<MatValue, MatError> {
        self.finish()
    }
}

impl SerializeTuple for SeqSer {
    type Ok = MatValue;
    type Error = MatError;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), MatError> {
        self.push(value)
    }
    fn end(self) -> Result<MatValue, MatError> {
        self.finish()
    }
}

impl SerializeTupleStruct for SeqSer {
    type Ok = MatValue;
    type Error = MatError;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), MatError> {
        self.push(value)
    }
    fn end(self) -> Result<MatValue, MatError> {
        self.finish()
    }
}

/// Decide what a finished sequence of elements means. A sequence whose
/// elements all share the same numeric shape (scalars of one tag, vectors of
/// one tag and length, complex of one width) collapses to a numeric vec/matrix
/// or complex vec/matrix. Anything else (mixed tags, ragged inner vectors,
/// sequences of structs, sequences containing `None`) lowers to a MATLAB cell
/// array; the emitter interns each element under `#refs#`.
fn unify_sequence(elements: Vec<MatValue>) -> Result<MatValue, MatError> {
    if elements.is_empty() {
        // Unknown element type; default to an empty f64 array.
        return Ok(MatValue::Vec1D(NumVec::F64(Vec::new())));
    }

    let elements = match try_unify_homogeneous(elements) {
        Ok(unified) => return Ok(unified),
        Err(elements) => elements,
    };

    // Heterogeneous: lower to a cell array, mapping `Omit` to `struct([])`.
    let cell_elements: Vec<MatValue> = elements
        .into_iter()
        .map(|e| match e {
            MatValue::Omit => MatValue::EmptyStructArray,
            other => other,
        })
        .collect();
    Ok(MatValue::Cell(cell_elements))
}

/// Try the homogeneous fast paths. Returns the original `Vec` back via
/// `Err(_)` when no path matches, so the cell-array fallback can take
/// ownership without re-cloning each element. (Cloning a `Vec1D` or
/// `ComplexVec*` of the inner shape would double peak allocation on the
/// matrix path for large `Vec<Vec<T>>` inputs.)
fn try_unify_homogeneous(elements: Vec<MatValue>) -> Result<MatValue, Vec<MatValue>> {
    debug_assert!(!elements.is_empty());

    // ----- all elements are numeric scalars of the same tag → Vec1D -----
    if let Some(MatValue::Scalar(first)) = elements.first() {
        let first_tag = first.tag();
        if elements
            .iter()
            .all(|e| matches!(e, MatValue::Scalar(s) if s.tag() == first_tag))
        {
            let mut vec = NumVec::empty_with_tag(first_tag);
            for e in elements {
                let MatValue::Scalar(s) = e else {
                    unreachable!()
                };
                vec.push(s).expect("tag check held");
            }
            return Ok(MatValue::Vec1D(vec));
        }
    }

    // ----- all elements are Vec1D of same tag & length → Matrix -----
    if let Some(MatValue::Vec1D(first)) = elements.first() {
        let first_tag = first.tag();
        let first_len = first.len();
        if elements.iter().all(
            |e| matches!(e, MatValue::Vec1D(v) if v.tag() == first_tag && v.len() == first_len),
        ) {
            let rows = elements.len();
            let mut flat = NumVec::empty_with_tag(first_tag);
            for e in elements {
                let MatValue::Vec1D(v) = e else {
                    unreachable!()
                };
                flat.extend(v).expect("tag check held");
            }
            return Ok(MatValue::Matrix {
                rows,
                cols: first_len,
                vec: flat,
            });
        }
    }

    // ----- all complex scalars of one width → ComplexVec (f64 case) -----
    if elements
        .iter()
        .all(|e| matches!(e, MatValue::ComplexScalar64 { .. }))
    {
        let pairs: Vec<(f64, f64)> = elements
            .into_iter()
            .map(|e| match e {
                MatValue::ComplexScalar64 { re, im } => (re, im),
                _ => unreachable!(),
            })
            .collect();
        return Ok(MatValue::ComplexVec64(pairs));
    }
    // ----- f32 complex-scalar variant of the above -----
    if elements
        .iter()
        .all(|e| matches!(e, MatValue::ComplexScalar32 { .. }))
    {
        let pairs: Vec<(f32, f32)> = elements
            .into_iter()
            .map(|e| match e {
                MatValue::ComplexScalar32 { re, im } => (re, im),
                _ => unreachable!(),
            })
            .collect();
        return Ok(MatValue::ComplexVec32(pairs));
    }

    // ----- all elements are ComplexVec of same length → ComplexMatrix -----
    if let Some(MatValue::ComplexVec64(first)) = elements.first() {
        let first_len = first.len();
        if elements
            .iter()
            .all(|e| matches!(e, MatValue::ComplexVec64(v) if v.len() == first_len))
        {
            let rows = elements.len();
            let mut pairs: Vec<(f64, f64)> = Vec::with_capacity(rows * first_len);
            for e in elements {
                let MatValue::ComplexVec64(v) = e else {
                    unreachable!()
                };
                pairs.extend(v);
            }
            return Ok(MatValue::ComplexMatrix64 {
                rows,
                cols: first_len,
                pairs,
            });
        }
    }
    if let Some(MatValue::ComplexVec32(first)) = elements.first() {
        let first_len = first.len();
        if elements
            .iter()
            .all(|e| matches!(e, MatValue::ComplexVec32(v) if v.len() == first_len))
        {
            let rows = elements.len();
            let mut pairs: Vec<(f32, f32)> = Vec::with_capacity(rows * first_len);
            for e in elements {
                let MatValue::ComplexVec32(v) = e else {
                    unreachable!()
                };
                pairs.extend(v);
            }
            return Ok(MatValue::ComplexMatrix32 {
                rows,
                cols: first_len,
                pairs,
            });
        }
    }

    Err(elements)
}

// ---------------------------------------------------------------------------
// Map serializer: HashMap<String, T> → struct
// ---------------------------------------------------------------------------

pub(crate) struct MapSer {
    fields: Vec<(String, MatValue)>,
    pending_key: Option<String>,
}

impl MapSer {
    fn new() -> Self {
        Self {
            fields: Vec::new(),
            pending_key: None,
        }
    }
}

impl SerializeMap for MapSer {
    type Ok = MatValue;
    type Error = MatError;

    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<(), MatError> {
        let key_val = key.serialize(ValueSerializer)?;
        let key_str = match key_val {
            MatValue::String(s) => s,
            other => {
                return Err(MatError::UnsupportedType(match other.kind() {
                    "struct" => "map with non-string keys (struct as key)",
                    _ => "map with non-string keys",
                }));
            }
        };
        self.pending_key = Some(key_str);
        Ok(())
    }

    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), MatError> {
        let key = self.pending_key.take().ok_or_else(|| {
            MatError::Custom("serialize_value called before serialize_key".into())
        })?;
        let val = value.serialize(ValueSerializer)?;
        if !matches!(val, MatValue::Omit) {
            self.fields.push((key, val));
        }
        Ok(())
    }

    fn end(self) -> Result<MatValue, MatError> {
        Ok(MatValue::Struct(self.fields))
    }
}

// ---------------------------------------------------------------------------
// Struct serializer: dispatches between Matrix sentinel, Complex sentinels,
// and a plain MATLAB-struct group.
// ---------------------------------------------------------------------------

pub(crate) enum StructSer {
    Matrix(MatrixFields),
    Complex64(ComplexFields<f64>),
    Complex32(ComplexFields<f32>),
    Plain(PlainStructFields),
}

#[derive(Default)]
pub(crate) struct MatrixFields {
    rows: Option<usize>,
    cols: Option<usize>,
    data: Option<MatValue>,
}

#[derive(Default)]
pub(crate) struct ComplexFields<T> {
    real: Option<T>,
    imag: Option<T>,
}

#[derive(Default)]
pub(crate) struct PlainStructFields {
    fields: Vec<(String, MatValue)>,
}

impl SerializeStruct for StructSer {
    type Ok = MatValue;
    type Error = MatError;

    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), MatError> {
        match self {
            StructSer::Matrix(fields) => match key {
                "rows" => {
                    let v = value.serialize(ValueSerializer)?;
                    fields.rows = Some(expect_usize(v, "Matrix::rows")?);
                }
                "cols" => {
                    let v = value.serialize(ValueSerializer)?;
                    fields.cols = Some(expect_usize(v, "Matrix::cols")?);
                }
                "data" => {
                    let v = value.serialize(ValueSerializer)?;
                    fields.data = Some(v);
                }
                other => {
                    return Err(MatError::Custom(format!(
                        "unexpected field {other:?} on Matrix sentinel"
                    )));
                }
            },
            StructSer::Complex64(fields) => match key {
                "real" => fields.real = Some(expect_f64(value.serialize(ValueSerializer)?)?),
                "imag" => fields.imag = Some(expect_f64(value.serialize(ValueSerializer)?)?),
                other => {
                    return Err(MatError::Custom(format!(
                        "unexpected field {other:?} on Complex64 sentinel"
                    )));
                }
            },
            StructSer::Complex32(fields) => match key {
                "real" => fields.real = Some(expect_f32(value.serialize(ValueSerializer)?)?),
                "imag" => fields.imag = Some(expect_f32(value.serialize(ValueSerializer)?)?),
                other => {
                    return Err(MatError::Custom(format!(
                        "unexpected field {other:?} on Complex32 sentinel"
                    )));
                }
            },
            StructSer::Plain(ps) => {
                let v = value.serialize(ValueSerializer)?;
                ps.fields.push((key.to_owned(), v));
            }
        }
        Ok(())
    }

    fn end(self) -> Result<MatValue, MatError> {
        match self {
            StructSer::Plain(ps) => Ok(MatValue::Struct(ps.fields)),
            StructSer::Matrix(fields) => matrix_from_fields(fields),
            StructSer::Complex64(fields) => Ok(MatValue::ComplexScalar64 {
                re: fields
                    .real
                    .ok_or_else(|| MatError::MissingField("real".into()))?,
                im: fields
                    .imag
                    .ok_or_else(|| MatError::MissingField("imag".into()))?,
            }),
            StructSer::Complex32(fields) => Ok(MatValue::ComplexScalar32 {
                re: fields
                    .real
                    .ok_or_else(|| MatError::MissingField("real".into()))?,
                im: fields
                    .imag
                    .ok_or_else(|| MatError::MissingField("imag".into()))?,
            }),
        }
    }
}

fn matrix_from_fields(fields: MatrixFields) -> Result<MatValue, MatError> {
    let rows = fields
        .rows
        .ok_or_else(|| MatError::MissingField("rows".into()))?;
    let cols = fields
        .cols
        .ok_or_else(|| MatError::MissingField("cols".into()))?;
    let data = fields
        .data
        .ok_or_else(|| MatError::MissingField("data".into()))?;
    let vec = match data {
        MatValue::Vec1D(v) => v,
        // Serializing `Vec<T>` of length 0 with T unknown yields F64.
        // If rows*cols == 0 we let it pass as an empty NumVec of that tag.
        MatValue::Scalar(s) if rows * cols == 1 => {
            let mut nv = NumVec::empty_with_tag(s.tag());
            nv.push(s)?;
            nv
        }
        other => {
            return Err(MatError::Custom(format!(
                "Matrix::data must be a Vec<T>, got {}",
                other.kind()
            )));
        }
    };
    if vec.len() != rows * cols {
        return Err(MatError::Custom(format!(
            "Matrix::data length {} does not match rows*cols = {}",
            vec.len(),
            rows * cols
        )));
    }
    Ok(MatValue::Matrix { rows, cols, vec })
}

fn expect_usize(v: MatValue, field: &str) -> Result<usize, MatError> {
    match v {
        MatValue::Scalar(ScalarNum::U64(x)) => Ok(x as usize),
        MatValue::Scalar(ScalarNum::U32(x)) => Ok(x as usize),
        MatValue::Scalar(ScalarNum::I64(x)) if x >= 0 => Ok(x as usize),
        MatValue::Scalar(ScalarNum::I32(x)) if x >= 0 => Ok(x as usize),
        MatValue::Scalar(ScalarNum::U16(x)) => Ok(x as usize),
        MatValue::Scalar(ScalarNum::U8(x)) => Ok(x as usize),
        other => Err(MatError::Custom(format!(
            "{field} must be an unsigned integer, got {}",
            other.kind()
        ))),
    }
}

fn expect_f64(v: MatValue) -> Result<f64, MatError> {
    match v {
        MatValue::Scalar(ScalarNum::F64(x)) => Ok(x),
        MatValue::Scalar(ScalarNum::F32(x)) => Ok(x as f64),
        other => Err(MatError::Custom(format!(
            "Complex field must be f64, got {}",
            other.kind()
        ))),
    }
}

fn expect_f32(v: MatValue) -> Result<f32, MatError> {
    match v {
        MatValue::Scalar(ScalarNum::F32(x)) => Ok(x),
        MatValue::Scalar(ScalarNum::F64(x)) => Ok(x as f32),
        other => Err(MatError::Custom(format!(
            "Complex field must be f32, got {}",
            other.kind()
        ))),
    }
}

// Silence unused-import warnings for items only referenced in specific
// serializer methods.
#[allow(dead_code)]
fn _touch_utf16() -> Vec<u16> {
    utf16::encode_utf16("x")
}

#[allow(dead_code)]
fn _touch_tag(_: ScalarTag) {}

#[allow(dead_code)]
fn _touch_ser_err<E: ser::Error>() -> E {
    E::custom("x")
}
