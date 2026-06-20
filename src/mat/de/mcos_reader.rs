//! Reader for MATLAB's MCOS opaque-object subsystem (`#subsystem#`).
//!
//! MATLAB stores its non-builtin classes — the modern `string`, plus
//! `datetime`, `categorical`, `table`, `containers.Map`, … — as
//! `mxOPAQUE_CLASS` objects (`MATLAB_object_decode = 3`). The object's data does
//! not live on its own dataset; instead the dataset carries a small `uint32`
//! metadata array that names an *object id*, and the actual property data is
//! interned in the hidden `#subsystem#/MCOS` store.
//!
//! On-disk shape (reverse-engineered against this crate's own writer, which
//! mirrors MATLAB byte-for-byte for `string`):
//!
//! - The opaque **parent dataset** holds `uint32`
//!   `[0xDD000000, ndims, dim0, …, object_id0, …, class_id]`. The reserved
//!   header is `2 + ndims` words; then one *object id* per element
//!   (`prod(dims)` of them); then a trailing *class id*.
//! - `/#subsystem#/MCOS` is a `FileWrapper__` object-reference dataset whose
//!   cell array is `[metadata blob, canonical empty, saveobj_0, saveobj_1, …,
//!   <trailing default/alias cells>]`. The saveobj payload for object id `k`
//!   (1-based) is therefore cell `k + 1` ([`MCOS_RESERVED_CELL_PREFIX`] leading
//!   cells before the first payload).
//!
//! This module currently decodes the `string` class; other opaque classes are
//! refused with a typed [`MatError::UnsupportedMatlabClass`] rather than
//! misread. The subsystem parse is the foundation the remaining classes build
//! on.

use std::collections::HashMap;

use crate::convert::TryToUsize;
use crate::file_writer::AttrValue;
use crate::mat::error::MatError;
use crate::mat::string_object::{
    MATLAB_CLASS_STRING, MATLAB_OBJECT_DECODE_OPAQUE, MATLAB_STRING_SAVEOBJ_VERSION,
    MCOS_MAGIC_NUMBER,
};
use crate::mat::value::MatValue;
use crate::reader::{Dataset, File, Object};
use crate::types::DType;

/// Number of reserved cells at the front of the `#subsystem#/MCOS` cell array
/// before the first object's saveobj payload: the `FileWrapper__` metadata blob
/// and the canonical-empty placeholder. The saveobj for 1-based object id `k`
/// lives at cell `MCOS_RESERVED_CELL_PREFIX + (k - 1)`.
const MCOS_RESERVED_CELL_PREFIX: usize = 2;

/// The parsed `#subsystem#/MCOS` opaque-object store for one file.
///
/// Holds the dereferenced MCOS cell array; opaque parent datasets index into it
/// by object id. Parsed once per file and shared across every opaque dataset
/// read (top-level variables and struct fields alike reference the same store).
pub(crate) struct Mcos<'f> {
    cells: Vec<Object<'f>>,
}

impl<'f> Mcos<'f> {
    /// Parse the `#subsystem#/MCOS` store, or `Ok(None)` if the file has no
    /// `#subsystem#` group (a file with no opaque objects).
    pub(crate) fn parse(file: &'f File) -> Result<Option<Self>, MatError> {
        let subsystem = match file.group("#subsystem#") {
            Ok(g) => g,
            // No subsystem group: the file holds no opaque objects.
            Err(_) => return Ok(None),
        };
        let mcos = subsystem.dataset("MCOS").map_err(MatError::Hdf5)?;
        let cells = mcos.dereference().map_err(MatError::Hdf5)?;
        Ok(Some(Self { cells }))
    }

    /// Read the saveobj `uint64` payload for a 1-based object id.
    fn saveobj_payload(&self, object_id: u32) -> Result<Vec<u64>, MatError> {
        if object_id == 0 {
            return Err(MatError::Custom(
                "opaque object id 0 is invalid (ids are 1-based)".into(),
            ));
        }
        let idx = MCOS_RESERVED_CELL_PREFIX + (object_id as usize - 1);
        let cell = self.cells.get(idx).ok_or_else(|| {
            MatError::Custom(format!(
                "opaque object id {object_id} maps to MCOS cell {idx}, but the store has {} cells",
                self.cells.len()
            ))
        })?;
        match cell {
            Object::Dataset(d) => {
                // The saveobj payload is always a `uint64` array. Refuse any
                // other datatype rather than letting `read_u64` widen a
                // narrower or floating-point cell into `u64` — the packing
                // (4 UTF-16 units per 64-bit word) is only valid for `uint64`,
                // so a divergent layout must error, not be reinterpreted.
                let dtype = d.dtype().map_err(MatError::Hdf5)?;
                if dtype != DType::U64 {
                    return Err(MatError::Custom(format!(
                        "MCOS saveobj cell {idx} has datatype {dtype:?}; expected uint64"
                    )));
                }
                d.read_u64().map_err(MatError::Hdf5)
            }
            Object::Group(_) => Err(MatError::Custom(format!(
                "MCOS cell {idx} is a group; expected a saveobj dataset"
            ))),
        }
    }
}

/// Decode an opaque (`mxOPAQUE_CLASS`) dataset into a [`MatValue`].
///
/// `decode` is the `MATLAB_object_decode` value; `attrs` are the parent
/// dataset's attributes (carrying `MATLAB_class`). Only `string`
/// (`decode == 3`) is decoded so far; any other opaque class is refused by name.
pub(crate) fn decode_opaque(
    ds: &Dataset<'_>,
    attrs: &HashMap<String, AttrValue>,
    decode: i64,
    mcos: Option<&Mcos<'_>>,
) -> Result<MatValue, MatError> {
    let class = raw_matlab_class(attrs).ok_or_else(|| {
        MatError::Custom("opaque object (MATLAB_object_decode set) has no MATLAB_class".into())
    })?;

    if decode == i64::from(MATLAB_OBJECT_DECODE_OPAQUE) && class == MATLAB_CLASS_STRING {
        let mcos = mcos.ok_or_else(|| {
            MatError::Custom(
                "file references an MCOS opaque object but has no #subsystem# store".into(),
            )
        })?;
        return decode_string_object(ds, mcos);
    }

    // Every other modern MATLAB type (datetime, categorical, table,
    // containers.Map, dictionary, enum, user classdefs, …) is also an MCOS
    // opaque object; decoding them is tracked follow-up work. Refuse by name
    // rather than misread.
    Err(MatError::UnsupportedMatlabClass(class))
}

/// Decode a `MATLAB_class = "string"` opaque dataset.
fn decode_string_object(ds: &Dataset<'_>, mcos: &Mcos<'_>) -> Result<MatValue, MatError> {
    let metadata = ds.read_u32().map_err(MatError::Hdf5)?;
    let object_ids = parse_opaque_object_ids(&metadata)?;

    // The serde writer emits one scalar `string` object, but a single object can
    // encode a string array and a dataset can name several objects; collect
    // every decoded value and represent the result accordingly.
    let mut values: Vec<String> = Vec::new();
    for object_id in object_ids {
        let payload = mcos.saveobj_payload(object_id)?;
        values.extend(decode_string_saveobj(&payload)?);
    }

    // Intentional lowering (not a shape-losing bug): a single value becomes a
    // scalar `String`; zero values (an empty `0×0 string`) or a genuine string
    // array become a cell of strings, which deserializes to `Vec<String>`. The
    // serde writer only ever emits scalar `string` objects, so the `1` arm is
    // the path these tests exercise; the cell arm is the forward-compatible
    // representation for real-MATLAB string arrays.
    Ok(match values.len() {
        1 => MatValue::String(values.pop().expect("len checked")),
        _ => MatValue::Cell(values.into_iter().map(MatValue::String).collect()),
    })
}

/// Extract the 1-based object ids from an opaque parent dataset's `uint32`
/// metadata: `[MAGIC, ndims, dims…, object_ids…, class_id]`.
fn parse_opaque_object_ids(data: &[u32]) -> Result<Vec<u32>, MatError> {
    // Minimum: MAGIC, ndims, (≥0 dims), (≥0 ids), class_id.
    if data.len() < 3 {
        return Err(MatError::Custom(format!(
            "opaque metadata too short: {} words",
            data.len()
        )));
    }
    if data[0] != MCOS_MAGIC_NUMBER {
        return Err(MatError::Custom(format!(
            "opaque metadata magic mismatch: {:#x}",
            data[0]
        )));
    }
    let ndims = (data[1] as u64).to_usize()?;
    let dims_end = 2usize
        .checked_add(ndims)
        .ok_or_else(|| MatError::Custom("opaque metadata ndims overflow".into()))?;
    // Need the dims block plus at least the trailing class id.
    if data.len() < dims_end + 1 {
        return Err(MatError::Custom(
            "opaque metadata truncated before object ids".into(),
        ));
    }
    let num_objects = checked_product(&data[2..dims_end])?;
    let ids_end = dims_end
        .checked_add(num_objects)
        .ok_or_else(|| MatError::Custom("opaque metadata object count overflow".into()))?;
    // Exactly one trailing class id must follow the object ids.
    if data.len() != ids_end + 1 {
        return Err(MatError::Custom(format!(
            "opaque metadata length {} does not match header (expected {})",
            data.len(),
            ids_end + 1
        )));
    }
    Ok(data[dims_end..ids_end].to_vec())
}

/// Decode a `string` saveobj payload into its string values.
///
/// Layout: `[VERSION, ndims, dims…, lens…, UTF-16 packed 4 units per u64]`.
/// A length of `u64::MAX` is MATLAB's `<missing>` sentinel (no code units),
/// decoded here as an empty string.
fn decode_string_saveobj(payload: &[u64]) -> Result<Vec<String>, MatError> {
    if payload.len() < 2 {
        return Err(MatError::Custom("string saveobj payload too short".into()));
    }
    if payload[0] != MATLAB_STRING_SAVEOBJ_VERSION {
        return Err(MatError::Custom(format!(
            "unsupported string saveobj version: {}",
            payload[0]
        )));
    }
    let ndims = payload[1].to_usize()?;
    let dims_end = 2usize
        .checked_add(ndims)
        .ok_or_else(|| MatError::Custom("string saveobj ndims overflow".into()))?;
    if payload.len() < dims_end {
        return Err(MatError::Custom(
            "string saveobj truncated before lengths".into(),
        ));
    }
    let count = checked_product_u64(&payload[2..dims_end])?;
    let lens_end = dims_end
        .checked_add(count)
        .ok_or_else(|| MatError::Custom("string saveobj count overflow".into()))?;
    if payload.len() < lens_end {
        return Err(MatError::Custom(
            "string saveobj truncated before code units".into(),
        ));
    }
    let lens = &payload[dims_end..lens_end];

    // Total UTF-16 code units across all (non-missing) strings.
    let mut total_units: usize = 0;
    for &len in lens {
        if len != u64::MAX {
            total_units = total_units
                .checked_add(len.to_usize()?)
                .ok_or_else(|| MatError::Custom("string saveobj unit count overflow".into()))?;
        }
    }

    // Unpack `total_units` code units from the packed u64 words (4 per word).
    let mut units: Vec<u16> = Vec::with_capacity(total_units);
    'words: for &word in &payload[lens_end..] {
        for shift in [0u32, 16, 32, 48] {
            if units.len() == total_units {
                break 'words;
            }
            #[expect(
                clippy::cast_possible_truncation,
                reason = "extracting one packed 16-bit UTF-16 code unit from a 64-bit word"
            )]
            units.push((word >> shift) as u16);
        }
    }
    if units.len() < total_units {
        return Err(MatError::Custom(
            "string saveobj code-unit data is shorter than the declared lengths".into(),
        ));
    }

    let mut strings = Vec::with_capacity(lens.len());
    let mut pos = 0usize;
    for &len in lens {
        if len == u64::MAX {
            strings.push(String::new());
            continue;
        }
        let len = len.to_usize()?;
        let slice = &units[pos..pos + len];
        pos += len;
        let s = String::from_utf16(slice).map_err(|e| MatError::Utf16Decode(e.to_string()))?;
        strings.push(s);
    }
    Ok(strings)
}

/// Read the raw `MATLAB_class` attribute string without parsing it into a
/// builtin [`MatClass`](crate::mat::class::MatClass) (opaque class names such as
/// `string` are not builtin variants).
fn raw_matlab_class(attrs: &HashMap<String, AttrValue>) -> Option<String> {
    match attrs.get("MATLAB_class") {
        Some(AttrValue::AsciiString(s)) | Some(AttrValue::String(s)) => Some(s.clone()),
        Some(AttrValue::StringArray(v)) if v.len() == 1 => Some(v[0].clone()),
        _ => None,
    }
}

/// `MATLAB_object_decode` value for a dataset, if present, normalized to `i64`.
/// A non-zero value marks an opaque object; `None`/`0` is an ordinary dataset.
pub(crate) fn matlab_object_decode(attrs: &HashMap<String, AttrValue>) -> Option<i64> {
    let v = match attrs.get("MATLAB_object_decode")? {
        AttrValue::I64(v) => *v,
        AttrValue::I32(v) => i64::from(*v),
        AttrValue::U32(v) => i64::from(*v),
        AttrValue::U64(v) => i64::try_from(*v).ok()?,
        _ => return None,
    };
    (v != 0).then_some(v)
}

fn checked_product(values: &[u32]) -> Result<usize, MatError> {
    let mut acc = 1usize;
    for &v in values {
        acc = acc
            .checked_mul((v as u64).to_usize()?)
            .ok_or_else(|| MatError::Custom("opaque dimension product overflow".into()))?;
    }
    Ok(acc)
}

fn checked_product_u64(values: &[u64]) -> Result<usize, MatError> {
    let mut acc = 1usize;
    for &v in values {
        acc = acc
            .checked_mul(v.to_usize()?)
            .ok_or_else(|| MatError::Custom("string saveobj dimension product overflow".into()))?;
    }
    Ok(acc)
}
