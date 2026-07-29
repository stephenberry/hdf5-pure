//! `ValueSerializer`: the core `serde::Serializer` that produces a
//! [`MatValue`] from any serializable input.
//!
//! The serializer makes one pass to collect everything into the intermediate
//! tree, then the emitter walks the tree to build the HDF5 file.

use serde::ser::{
    self, Impossible, Serialize, SerializeMap, SerializeSeq, SerializeStruct, SerializeTuple,
    SerializeTupleStruct, Serializer,
};

use crate::mat::complex::complex_tag_for_sentinel;
use crate::mat::error::MatError;
use crate::mat::matrix::{MATRIX_SENTINEL, complex_tag_for_matrix_sentinel};
use crate::mat::options::{EmptySequencePolicy, NullPolicy, Options, UnitVariantEncoding};
use crate::mat::utf16;

use crate::mat::value::{
    ComplexNum, ComplexTag, ComplexVec, MatValue, NumVec, ScalarNum, ScalarTag,
};

// ---------------------------------------------------------------------------
// Public entry: serialize a value into a MatValue
// ---------------------------------------------------------------------------

pub(crate) fn to_value<T: Serialize + ?Sized>(
    value: &T,
    options: &Options,
) -> Result<MatValue, MatError> {
    value.serialize(ValueSerializer::new(options))
}

/// Lower a `None` / unit / unit-struct per [`Options::null_policy`].
fn null_value(opts: &Options) -> Result<MatValue, MatError> {
    match opts.null_policy {
        NullPolicy::EmptyStructArray => Ok(MatValue::EmptyStructArray),
        NullPolicy::Omit => Ok(MatValue::Omit),
        NullPolicy::Error => Err(MatError::UnsupportedType(
            "null value under NullPolicy::Error",
        )),
    }
}

// ---------------------------------------------------------------------------
// ValueSerializer
// ---------------------------------------------------------------------------

/// Borrows the caller's [`Options`] so that the handful of decisions with no
/// single right answer (null lowering, unit-variant encoding, the class of an
/// empty sequence) are made where the input is still in view, rather than
/// being baked into the value tree and second-guessed by the emitter.
#[derive(Clone, Copy)]
pub(crate) struct ValueSerializer<'a> {
    opts: &'a Options,
}

impl<'a> ValueSerializer<'a> {
    pub(crate) fn new(opts: &'a Options) -> Self {
        Self { opts }
    }
}

impl<'a> Serializer for ValueSerializer<'a> {
    type Ok = MatValue;
    type Error = MatError;

    type SerializeSeq = SeqSer<'a>;
    type SerializeTuple = SeqSer<'a>;
    type SerializeTupleStruct = SeqSer<'a>;
    type SerializeTupleVariant = Impossible<MatValue, MatError>;
    type SerializeMap = MapSer<'a>;
    type SerializeStruct = StructSer<'a>;
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
        null_value(self.opts)
    }

    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<MatValue, MatError> {
        value.serialize(self)
    }

    fn serialize_unit(self) -> Result<MatValue, MatError> {
        // Unit lowers exactly like `serialize_none` above. The common way to
        // hit it is `serde_json::Value::Null`, which serializes via
        // `serialize_unit`; routing both through `null_policy` means the two
        // spellings of "no value" cannot come to disagree.
        null_value(self.opts)
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<MatValue, MatError> {
        null_value(self.opts)
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        idx: u32,
        variant: &'static str,
    ) -> Result<MatValue, MatError> {
        match self.opts.unit_variant_encoding {
            UnitVariantEncoding::Name => Ok(MatValue::String(variant.to_owned())),
            UnitVariantEncoding::Index => Ok(MatValue::Scalar(ScalarNum::U32(idx))),
        }
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

    fn serialize_seq(self, len: Option<usize>) -> Result<SeqSer<'a>, MatError> {
        Ok(SeqSer::new(len, self.opts))
    }
    fn serialize_tuple(self, len: usize) -> Result<SeqSer<'a>, MatError> {
        Ok(SeqSer::new(Some(len), self.opts))
    }
    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        len: usize,
    ) -> Result<SeqSer<'a>, MatError> {
        Ok(SeqSer::new(Some(len), self.opts))
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

    fn serialize_map(self, _len: Option<usize>) -> Result<MapSer<'a>, MatError> {
        Ok(MapSer::new(self.opts))
    }

    fn serialize_struct(self, name: &'static str, len: usize) -> Result<StructSer<'a>, MatError> {
        let kind = if let Some(tag) = complex_tag_for_sentinel(name) {
            StructKind::Complex(tag, ComplexFields::default())
        } else if let Some(tag) = complex_tag_for_matrix_sentinel(name) {
            StructKind::Matrix(MatrixFields::default(), MatrixKind::Complex(tag))
        } else {
            match name {
                MATRIX_SENTINEL => StructKind::Matrix(MatrixFields::default(), MatrixKind::Numeric),
                // serde supplies the exact field count, so the field Vec can be
                // sized once instead of growing by reallocation.
                _ => StructKind::Plain(PlainStructFields::with_capacity(len)),
            }
        };
        Ok(StructSer {
            opts: self.opts,
            kind,
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

pub(crate) struct SeqSer<'a> {
    accum: SeqAccum,
    opts: &'a Options,
}

/// How a sequence's elements are held while it is being collected.
///
/// A flat numeric or complex array is the common large case, and holding it
/// as one `MatValue` per element costs 56 bytes for what packs into 4. So the
/// accumulator stays packed for as long as the elements agree, and only falls
/// back to one-value-per-element when one of them breaks the pattern. That
/// fallback is where the sequence was always going to end up anyway (a cell
/// array, or a matrix built from equal-length rows), so nothing is lost by
/// paying for it there instead of everywhere.
enum SeqAccum {
    /// Nothing pushed yet; the first element picks the representation.
    Empty { cap: usize },
    /// Every element so far is a numeric scalar of one tag.
    Numeric(NumVec),
    /// Every element so far is a complex scalar of one tag.
    Complex(ComplexVec),
    /// Anything else, held individually for `unify_sequence` to interpret.
    Mixed(Vec<MatValue>),
}

impl SeqAccum {
    fn push(&mut self, value: MatValue) -> Result<(), MatError> {
        match self {
            SeqAccum::Empty { cap } => {
                let cap = *cap;
                *self = match value {
                    MatValue::Scalar(s) => {
                        let mut v = NumVec::with_capacity_for_tag(s.tag(), cap);
                        v.push(s)?;
                        SeqAccum::Numeric(v)
                    }
                    MatValue::ComplexScalar(c) => {
                        let mut v = ComplexVec::with_capacity_for_tag(c.tag(), cap);
                        v.push(c)?;
                        SeqAccum::Complex(v)
                    }
                    other => {
                        let mut v = Vec::with_capacity(cap);
                        v.push(other);
                        SeqAccum::Mixed(v)
                    }
                };
                Ok(())
            }
            SeqAccum::Numeric(v) => match value {
                MatValue::Scalar(s) if s.tag() == v.tag() => v.push(s),
                other => self.spill_and_push(other),
            },
            SeqAccum::Complex(v) => match value {
                MatValue::ComplexScalar(c) if c.tag() == v.tag() => v.push(c),
                other => self.spill_and_push(other),
            },
            SeqAccum::Mixed(v) => {
                v.push(value);
                Ok(())
            }
        }
    }

    /// Expand a packed accumulator back to one `MatValue` per element, then
    /// push the element that broke the pattern.
    fn spill_and_push(&mut self, value: MatValue) -> Result<(), MatError> {
        let packed = std::mem::replace(self, SeqAccum::Empty { cap: 0 });
        let (len, elements): (usize, Box<dyn Iterator<Item = MatValue>>) = match packed {
            SeqAccum::Numeric(nums) => (
                nums.len(),
                Box::new(nums.into_scalars().map(MatValue::Scalar)),
            ),
            SeqAccum::Complex(pairs) => (
                pairs.len(),
                Box::new(pairs.into_pairs().map(MatValue::ComplexScalar)),
            ),
            SeqAccum::Empty { .. } | SeqAccum::Mixed(_) => {
                unreachable!("only a packed accumulator spills")
            }
        };
        let mut v = Vec::with_capacity(len + 1);
        v.extend(elements);
        v.push(value);
        *self = SeqAccum::Mixed(v);
        Ok(())
    }
}

impl<'a> SeqSer<'a> {
    fn new(len: Option<usize>, opts: &'a Options) -> Self {
        Self {
            accum: SeqAccum::Empty {
                cap: len.unwrap_or(0),
            },
            opts,
        }
    }

    fn push<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), MatError> {
        let value = v.serialize(ValueSerializer::new(self.opts))?;
        self.accum.push(value)
    }

    fn finish(self) -> Result<MatValue, MatError> {
        match self.accum {
            SeqAccum::Empty { .. } => unify_sequence(Vec::new(), self.opts),
            SeqAccum::Numeric(v) => Ok(MatValue::Vec1D(v)),
            SeqAccum::Complex(v) => Ok(MatValue::ComplexVec1D(v)),
            SeqAccum::Mixed(v) => unify_sequence(v, self.opts),
        }
    }
}

impl SerializeSeq for SeqSer<'_> {
    type Ok = MatValue;
    type Error = MatError;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), MatError> {
        self.push(value)
    }
    fn end(self) -> Result<MatValue, MatError> {
        self.finish()
    }
}

impl SerializeTuple for SeqSer<'_> {
    type Ok = MatValue;
    type Error = MatError;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), MatError> {
        self.push(value)
    }
    fn end(self) -> Result<MatValue, MatError> {
        self.finish()
    }
}

impl SerializeTupleStruct for SeqSer<'_> {
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
fn unify_sequence(elements: Vec<MatValue>, opts: &Options) -> Result<MatValue, MatError> {
    if elements.is_empty() {
        // No element revealed its type, so the MATLAB class is the caller's
        // to pick; see `EmptySequencePolicy`.
        return Ok(match opts.empty_sequence_policy {
            EmptySequencePolicy::DoubleArray => MatValue::Vec1D(NumVec::F64(Vec::new())),
            EmptySequencePolicy::Cell => MatValue::Cell(Vec::new()),
        });
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
            let mut vec = NumVec::with_capacity_for_tag(first_tag, elements.len());
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
            let mut flat = NumVec::with_capacity_for_tag(first_tag, rows * first_len);
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

    // ----- all complex scalars of one class → ComplexVec1D -----
    if let Some(MatValue::ComplexScalar(first)) = elements.first() {
        let first_tag = first.tag();
        if elements
            .iter()
            .all(|e| matches!(e, MatValue::ComplexScalar(n) if n.tag() == first_tag))
        {
            let mut pairs = ComplexVec::with_capacity_for_tag(first_tag, elements.len());
            for e in elements {
                let MatValue::ComplexScalar(n) = e else {
                    unreachable!()
                };
                pairs.push(n).expect("tag check held");
            }
            return Ok(MatValue::ComplexVec1D(pairs));
        }
    }

    // ----- all elements are ComplexVec1D of one class & length → ComplexMatrix -----
    if let Some(MatValue::ComplexVec1D(first)) = elements.first() {
        let first_tag = first.tag();
        let first_len = first.len();
        if elements.iter().all(
            |e| matches!(e, MatValue::ComplexVec1D(v) if v.tag() == first_tag && v.len() == first_len),
        ) {
            let rows = elements.len();
            let mut pairs = ComplexVec::with_capacity_for_tag(first_tag, rows * first_len);
            for e in elements {
                let MatValue::ComplexVec1D(v) = e else {
                    unreachable!()
                };
                pairs.extend(v).expect("tag check held");
            }
            return Ok(MatValue::ComplexMatrix {
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

pub(crate) struct MapSer<'a> {
    fields: Vec<(String, MatValue)>,
    pending_key: Option<String>,
    opts: &'a Options,
}

impl<'a> MapSer<'a> {
    fn new(opts: &'a Options) -> Self {
        Self {
            fields: Vec::new(),
            pending_key: None,
            opts,
        }
    }
}

impl SerializeMap for MapSer<'_> {
    type Ok = MatValue;
    type Error = MatError;

    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<(), MatError> {
        let key_val = key.serialize(ValueSerializer::new(self.opts))?;
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
        let val = value.serialize(ValueSerializer::new(self.opts))?;
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

pub(crate) struct StructSer<'a> {
    opts: &'a Options,
    kind: StructKind,
}

pub(crate) enum StructKind {
    Matrix(MatrixFields, MatrixKind),
    Complex(ComplexTag, ComplexFields),
    Plain(PlainStructFields),
}

/// Element class hint for a `Matrix<T>` sentinel. Carried through from the
/// chosen sentinel name (see `matrix::MatElement`) so that empty matrices,
/// where the inner `Vec<T>` cannot reveal `T`, still emit with the right
/// MATLAB class.
#[derive(Clone, Copy)]
pub(crate) enum MatrixKind {
    Numeric,
    Complex(ComplexTag),
}

#[derive(Default)]
pub(crate) struct MatrixFields {
    rows: Option<usize>,
    cols: Option<usize>,
    data: Option<MatValue>,
}

/// The two fields of a complex sentinel, held as tagged scalars: the component
/// class comes from the sentinel, and `end` checks the fields against it.
#[derive(Default)]
pub(crate) struct ComplexFields {
    real: Option<ScalarNum>,
    imag: Option<ScalarNum>,
}

#[derive(Default)]
pub(crate) struct PlainStructFields {
    fields: Vec<(String, MatValue)>,
}

impl PlainStructFields {
    fn with_capacity(n: usize) -> Self {
        Self {
            fields: Vec::with_capacity(n),
        }
    }
}

impl SerializeStruct for StructSer<'_> {
    type Ok = MatValue;
    type Error = MatError;

    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), MatError> {
        let vs = ValueSerializer::new(self.opts);
        match &mut self.kind {
            StructKind::Matrix(fields, _) => match key {
                "rows" => {
                    let v = value.serialize(vs)?;
                    fields.rows = Some(expect_usize(v, "Matrix::rows")?);
                }
                "cols" => {
                    let v = value.serialize(vs)?;
                    fields.cols = Some(expect_usize(v, "Matrix::cols")?);
                }
                "data" => {
                    let v = value.serialize(vs)?;
                    fields.data = Some(v);
                }
                other => {
                    return Err(MatError::Custom(format!(
                        "unexpected field {other:?} on Matrix sentinel"
                    )));
                }
            },
            StructKind::Complex(tag, fields) => match key {
                "real" => fields.real = Some(expect_component(value.serialize(vs)?)?),
                "imag" => fields.imag = Some(expect_component(value.serialize(vs)?)?),
                other => {
                    return Err(MatError::Custom(format!(
                        "unexpected field {other:?} on the complex {} sentinel",
                        tag.class().as_str()
                    )));
                }
            },
            StructKind::Plain(ps) => {
                let v = value.serialize(vs)?;
                ps.fields.push((key.to_owned(), v));
            }
        }
        Ok(())
    }

    fn end(self) -> Result<MatValue, MatError> {
        match self.kind {
            StructKind::Plain(ps) => Ok(MatValue::Struct(ps.fields)),
            StructKind::Matrix(fields, kind) => matrix_from_fields(fields, kind),
            StructKind::Complex(tag, fields) => {
                let re = fields
                    .real
                    .ok_or_else(|| MatError::MissingField("real".into()))?;
                let im = fields
                    .imag
                    .ok_or_else(|| MatError::MissingField("imag".into()))?;
                let n = ComplexNum::from_components(tag, re, im).ok_or_else(|| {
                    MatError::Custom(format!(
                        "complex {} fields must both be {}",
                        tag.class().as_str(),
                        tag.class().as_str()
                    ))
                })?;
                Ok(MatValue::ComplexScalar(n))
            }
        }
    }
}

fn matrix_from_fields(fields: MatrixFields, kind: MatrixKind) -> Result<MatValue, MatError> {
    let rows = fields
        .rows
        .ok_or_else(|| MatError::MissingField("rows".into()))?;
    let cols = fields
        .cols
        .ok_or_else(|| MatError::MissingField("cols".into()))?;
    let data = fields
        .data
        .ok_or_else(|| MatError::MissingField("data".into()))?;
    // `rows` and `cols` arrive from the serialized input, so the product can
    // overflow. Refuse it here: a wrapped total would agree with a short data
    // vector and let the pair through to a writer that transposes through a raw
    // pointer sized from that same product.
    let total = rows.checked_mul(cols).ok_or_else(|| {
        MatError::Custom(format!(
            "Matrix dimensions {rows}x{cols} overflow the address space"
        ))
    })?;
    let length_check = |actual: usize| -> Result<(), MatError> {
        if actual != total {
            return Err(MatError::Custom(format!(
                "Matrix::data length {} does not match rows*cols = {}",
                actual, total
            )));
        }
        Ok(())
    };
    match (kind, data) {
        // Numeric Matrix<T>: data unifies to Vec1D of T's tag. Matrix<Complex*>
        // does not land here; it carries `T::SENTINEL = MATRIX_COMPLEX{32,64}_SENTINEL`
        // and routes to the dedicated Complex64 / Complex32 arms below.
        (MatrixKind::Numeric, MatValue::Vec1D(vec)) => {
            length_check(vec.len())?;
            Ok(MatValue::Matrix { rows, cols, vec })
        }
        // Matrix<Complex*>: a ComplexVec1D of the sentinel's class in the data
        // slot, OR an empty Vec1D (the f64-default that an empty
        // `Vec<Complex*>` collapses to in the seq path: with no elements
        // observed, the seq unification can't recover T). The dedicated
        // sentinel here lets us recover the class.
        (MatrixKind::Complex(tag), MatValue::ComplexVec1D(pairs)) if pairs.tag() == tag => {
            length_check(pairs.len())?;
            Ok(MatValue::ComplexMatrix { rows, cols, pairs })
        }
        (MatrixKind::Complex(tag), MatValue::Vec1D(vec)) if vec.is_empty() => {
            length_check(0)?;
            Ok(MatValue::ComplexMatrix {
                rows,
                cols,
                pairs: ComplexVec::empty_with_tag(tag),
            })
        }
        (_, other) => Err(MatError::Custom(format!(
            "Matrix::data must be a Vec<T>, got {}",
            other.kind()
        ))),
    }
}

fn expect_usize(v: MatValue, field: &str) -> Result<usize, MatError> {
    match v {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "serde scalar accessor; conversion of a user-level MAT scalar (e.g. a Matrix rows/cols field) to usize, not a file-derived size"
        )]
        MatValue::Scalar(ScalarNum::U64(x)) => Ok(x as usize),
        MatValue::Scalar(ScalarNum::U32(x)) => Ok(x as usize),
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the `x >= 0` guard keeps the I64 non-negative; serde scalar accessor converting a user-level MAT scalar to usize, not a file-derived size"
        )]
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

/// A complex sentinel's `real`/`imag` field, kept at the width it was
/// serialized at. `end` checks it against the sentinel's class.
fn expect_component(v: MatValue) -> Result<ScalarNum, MatError> {
    match v {
        MatValue::Scalar(s) => Ok(s),
        other => Err(MatError::Custom(format!(
            "a complex field must be a numeric scalar, got {}",
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
