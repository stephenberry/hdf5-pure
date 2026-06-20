//! Reader for MATLAB's MCOS opaque-object subsystem (`#subsystem#`).
//!
//! MATLAB stores its non-builtin classes — the modern `string`, plus
//! `datetime`, `duration`, `categorical`, `table`, `containers.Map`, … — as
//! `mxOPAQUE_CLASS` objects (`MATLAB_object_decode = 3`). The object's data does
//! not live on its own dataset; instead the dataset carries a small `uint32`
//! metadata array that names one or more *object ids*, and the actual property
//! data is interned in the hidden `#subsystem#/MCOS` store.
//!
//! # The `#subsystem#/MCOS` store
//!
//! `#subsystem#/MCOS` is a `FileWrapper__` object-reference dataset; dereferenced
//! it yields a cell array laid out as
//! `[metadata blob, canonical empty, <property-value heap…>, <trailing shared
//! cells>]`. The two leading cells are reserved ([`MCOS_RESERVED_CELL_PREFIX`]),
//! which is the source of the `+2` heap offset below.
//!
//! Cell 0 is a `uint8` **FileWrapper metadata blob** describing every opaque
//! object in the file. Its byte layout (little-endian `u32` throughout,
//! cross-validated against the `matio`, `MatFileHandler`, and `foreverallama`
//! parsers, and against this crate's own writer):
//!
//! - bytes `0..4`   `version` (2..=4; this crate writes 4)
//! - bytes `4..8`   `num_strings` — count of names in the heap
//! - bytes `8..40`  eight `u32` **region offsets** (byte offsets from blob start)
//! - bytes `40..`   the **name heap**: `num_strings` NUL-terminated names
//!   (1-based indices; index 0 means "absent")
//!
//! followed by five region tables delimited by the offsets, in this order:
//!
//! | region | range | meaning |
//! |--------|-------|---------|
//! | names  | `40 .. offsets[0]` | name heap |
//! | R1     | `offsets[0] .. offsets[1]` | **class table** (`namespace_idx, name_idx, 0, 0` per class) |
//! | R2     | `offsets[1] .. offsets[2]` | **type-1 / "saveobj" property table** |
//! | R3     | `offsets[2] .. offsets[3]` | **object table** (`class_id, 0, 0, saveobj_id, normalobj_id, dep_id`) |
//! | R4     | `offsets[3] .. offsets[4]` | **type-2 / "normal" property table** |
//! | R5     | `offsets[4] .. offsets[5]` | dynamic-property table (unused here) |
//!
//! The class and object tables carry a leading reserved entry (ids are 1-based).
//! Each property table is a sequence of blocks `[nprops, (name_idx, field_type,
//! value)…]` padded to an 8-byte boundary; an object reaches its block via the
//! `saveobj_id` (R2) or `normalobj_id` (R4) from its object record. A property's
//! `field_type` selects how `value` is read: `0` → a name-heap index (an
//! enumeration-member/string name), `1` → a 0-based index into the property
//! heap (resolved as cell `value + 2`), `2` → an inline integer/logical.
//!
//! # What this module decodes
//!
//! The modern `string` class keeps a dedicated decoder (its `saveobj` payload is
//! a self-describing `uint64` blob). `datetime`, `duration`, and `categorical`
//! are decoded from their resolved properties. Every other opaque class is
//! surfaced losslessly as [`MatValue::Opaque`] (class name plus raw properties)
//! rather than misread or refused outright.

use std::collections::HashMap;

use crate::convert::TryToUsize;
use crate::file_writer::AttrValue;
use crate::mat::error::MatError;
use crate::mat::string_object::{
    MATLAB_CLASS_STRING, MATLAB_OBJECT_DECODE_OPAQUE, MATLAB_STRING_SAVEOBJ_VERSION,
    MCOS_MAGIC_NUMBER,
};
use crate::mat::value::{MatValue, NumVec, ScalarNum};
use crate::reader::{Dataset, File, Object};
use crate::types::DType;

/// Number of reserved cells at the front of the `#subsystem#/MCOS` cell array
/// before the property-value heap: the `FileWrapper__` metadata blob (cell 0)
/// and the canonical-empty placeholder (cell 1). A `field_type == 1` property
/// whose stored value is the 0-based heap index `v` therefore lives at cell
/// `MCOS_RESERVED_CELL_PREFIX + v`.
const MCOS_RESERVED_CELL_PREFIX: usize = 2;

/// Length of the fixed FileWrapper blob header (version, num_strings, eight
/// region offsets) before the name heap begins.
const FILEWRAPPER_HEADER_LEN: usize = 40;

/// Number of `u32` region offsets in the FileWrapper header (bytes 8..40).
const FILEWRAPPER_NUM_OFFSETS: usize = 8;

/// `u32` words per class-table entry: `(namespace_idx, name_idx, reserved,
/// reserved)`.
const CLASS_ENTRY_WORDS: usize = 4;

/// `u32` words per object-table entry: `(class_id, reserved, reserved,
/// saveobj_id, normalobj_id, dependency_id)`.
const OBJECT_ENTRY_WORDS: usize = 6;

/// `u32` words per property triple: `(name_idx, field_type, value)`.
const PROP_TRIPLE_WORDS: usize = 3;

/// `field_type` selecting a name-heap index (an enumeration-member or string
/// name): `value` is a 1-based index into the name heap.
const FIELD_TYPE_NAME: u32 = 0;
/// `field_type` selecting a property-heap cell: `value` is a 0-based index into
/// the property-value heap (resolved as cell `value + 2`).
const FIELD_TYPE_HEAP: u32 = 1;
/// `field_type` selecting an inline literal: `value` is an integer/logical
/// stored directly in the triple.
const FIELD_TYPE_INLINE: u32 = 2;

/// The parsed `#subsystem#/MCOS` opaque-object store for one file.
///
/// Holds the dereferenced MCOS cell array (the property-value heap) and the
/// parsed [`FileWrapper`] metadata. Opaque parent datasets index into it by
/// object id. Parsed once per file and shared across every opaque dataset read.
pub(crate) struct Mcos<'f> {
    cells: Vec<Object<'f>>,
    wrapper: FileWrapper,
}

/// The parsed FileWrapper metadata blob (cell 0 of the MCOS store).
struct FileWrapper {
    /// Name heap; `names[i]` is 1-based name index `i + 1`.
    names: Vec<String>,
    /// Class table; index 0 is the reserved leading entry, so a 1-based
    /// `class_id` indexes directly.
    classes: Vec<ClassEntry>,
    /// Object table; index 0 is the reserved leading entry, so a 1-based
    /// `object_id` indexes directly.
    objects: Vec<ObjectRecord>,
    /// Type-1 ("saveobj") property blocks; block 0 is the reserved leading
    /// block, so a 1-based `saveobj_id` indexes directly.
    seg1: Vec<PropBlock>,
    /// Type-2 ("normal") property blocks; block 0 is the reserved leading
    /// block, so a 1-based `normalobj_id` indexes directly.
    seg2: Vec<PropBlock>,
}

/// One class-table entry: a package/namespace name and the class name, each a
/// 1-based name-heap index (`0` = none).
struct ClassEntry {
    namespace_idx: u32,
    name_idx: u32,
}

/// One object-table entry. Exactly one of `saveobj_id` / `normalobj_id` is
/// non-zero; it selects the object's property block in R2 / R4 respectively.
struct ObjectRecord {
    class_id: u32,
    saveobj_id: u32,
    normalobj_id: u32,
    #[expect(
        dead_code,
        reason = "dependency_id (R5 dynamic-property link) is parsed for completeness; \
                  the decoded value classes do not use dynamic properties yet"
    )]
    dependency_id: u32,
}

/// One property block: the property triples for a single object.
struct PropBlock {
    props: Vec<Triple>,
}

/// One property triple: the property name and its typed value reference.
struct Triple {
    name_idx: u32,
    field_type: u32,
    value: u32,
}

/// A property resolved to its name and a reference to its value.
pub(crate) struct PropRef {
    pub(crate) name: String,
    pub(crate) value: PropValue,
}

/// How a resolved property's value is reached.
pub(crate) enum PropValue {
    /// A property-heap cell index (0-based into the heap; resolve via
    /// [`Mcos::heap_object`]).
    Heap(u32),
    /// An inline integer/logical literal.
    Inline(u32),
    /// A name-heap string (enumeration member / string property).
    Name(String),
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

        // Cell 0 is the FileWrapper metadata blob (a uint8 array).
        let blob = match cells.first() {
            Some(Object::Dataset(d)) => d.read_u8().map_err(MatError::Hdf5)?,
            Some(Object::Group(_)) => {
                return Err(MatError::Custom(
                    "#subsystem#/MCOS cell 0 is a group; expected the FileWrapper metadata blob"
                        .into(),
                ));
            }
            None => {
                return Err(MatError::Custom(
                    "#subsystem#/MCOS store is empty; expected at least the FileWrapper blob"
                        .into(),
                ));
            }
        };
        let wrapper = FileWrapper::parse(&blob)?;
        Ok(Some(Self { cells, wrapper }))
    }

    /// The full class name for a 1-based `class_id` (e.g. `"datetime"`,
    /// `"containers.Map"`).
    pub(crate) fn class_name(&self, class_id: u32) -> Result<String, MatError> {
        let entry = self
            .wrapper
            .classes
            .get(class_id.to_usize()?)
            .ok_or_else(|| MatError::Custom(format!("MCOS class id {class_id} out of range")))?;
        let name = self.name(entry.name_idx)?;
        if entry.namespace_idx != 0 {
            Ok(format!("{}.{name}", self.name(entry.namespace_idx)?))
        } else {
            Ok(name)
        }
    }

    /// The class name of a 1-based `object_id`, resolved through its object
    /// record's `class_id`.
    pub(crate) fn object_class_name(&self, object_id: u32) -> Result<String, MatError> {
        self.class_name(self.object_record(object_id)?.class_id)
    }

    /// Resolve every property of a 1-based `object_id` to its name and value
    /// reference, in declaration order. Picks the type-1 (`saveobj`) or type-2
    /// (`normal`) block as the object record dictates.
    pub(crate) fn properties(&self, object_id: u32) -> Result<Vec<PropRef>, MatError> {
        let record = self.object_record(object_id)?;
        let (blocks, block_id) = if record.saveobj_id != 0 {
            (&self.wrapper.seg1, record.saveobj_id)
        } else {
            (&self.wrapper.seg2, record.normalobj_id)
        };
        // block_id 0 means the object has no properties in that segment.
        if block_id == 0 {
            return Ok(Vec::new());
        }
        let block = blocks.get(block_id.to_usize()?).ok_or_else(|| {
            MatError::Custom(format!(
                "MCOS object {object_id} references property block {block_id} out of range"
            ))
        })?;

        let mut out = Vec::with_capacity(block.props.len());
        for triple in &block.props {
            let name = self.name(triple.name_idx)?;
            let value = match triple.field_type {
                FIELD_TYPE_NAME => PropValue::Name(self.name(triple.value)?),
                FIELD_TYPE_HEAP => PropValue::Heap(triple.value),
                FIELD_TYPE_INLINE => PropValue::Inline(triple.value),
                other => {
                    return Err(MatError::Custom(format!(
                        "MCOS property {name:?} has unknown field_type {other}"
                    )));
                }
            };
            out.push(PropRef { name, value });
        }
        Ok(out)
    }

    /// The property-heap cell for a 0-based `field_type == 1` heap value.
    pub(crate) fn heap_object(&self, value: u32) -> Result<&Object<'f>, MatError> {
        let idx = MCOS_RESERVED_CELL_PREFIX
            .checked_add(value.to_usize()?)
            .ok_or_else(|| MatError::Custom("MCOS heap index overflow".into()))?;
        self.cells.get(idx).ok_or_else(|| {
            MatError::Custom(format!(
                "MCOS heap value {value} maps to cell {idx}, but the store has {} cells",
                self.cells.len()
            ))
        })
    }

    /// The object record for a 1-based `object_id`.
    fn object_record(&self, object_id: u32) -> Result<&ObjectRecord, MatError> {
        if object_id == 0 {
            return Err(MatError::Custom(
                "opaque object id 0 is invalid (ids are 1-based)".into(),
            ));
        }
        self.wrapper
            .objects
            .get(object_id.to_usize()?)
            .ok_or_else(|| {
                MatError::Custom(format!(
                    "MCOS object id {object_id} out of range ({} objects)",
                    self.wrapper.objects.len().saturating_sub(1)
                ))
            })
    }

    /// Look up a 1-based name-heap index. Index 0 is MATLAB's "absent" sentinel
    /// and resolves to the empty string.
    fn name(&self, idx: u32) -> Result<String, MatError> {
        if idx == 0 {
            return Ok(String::new());
        }
        self.wrapper
            .names
            .get(idx.to_usize()? - 1)
            .cloned()
            .ok_or_else(|| MatError::Custom(format!("MCOS name index {idx} out of range")))
    }

    /// Decode a `MATLAB_class = "string"` opaque dataset.
    ///
    /// `string` keeps a dedicated path: its `saveobj` payload is a
    /// self-describing `uint64` blob in the property heap rather than a set of
    /// named properties, so it is read directly instead of through the property
    /// tables.
    pub(crate) fn decode_string(&self, ds: &Dataset<'_>) -> Result<MatValue, MatError> {
        let metadata = ds.read_u32().map_err(MatError::Hdf5)?;
        let object_ids = parse_opaque_metadata(&metadata)?.object_ids;

        // A single object can encode a string array, and a dataset can name
        // several objects; collect every decoded value.
        let mut values: Vec<String> = Vec::new();
        for object_id in object_ids {
            let payload = self.saveobj_payload(object_id)?;
            values.extend(decode_string_saveobj(&payload)?);
        }

        // A single value becomes a scalar `String`; zero values (empty `0×0
        // string`) or a genuine string array become a cell of strings.
        Ok(match values.len() {
            1 => MatValue::String(values.pop().expect("len checked")),
            _ => MatValue::Cell(values.into_iter().map(MatValue::String).collect()),
        })
    }

    /// Read the saveobj `uint64` payload for a 1-based object id.
    ///
    /// The string writer lays out one saveobj cell per object directly after the
    /// reserved prefix, so object id `k` maps to cell `MCOS_RESERVED_CELL_PREFIX
    /// + (k - 1)`.
    fn saveobj_payload(&self, object_id: u32) -> Result<Vec<u64>, MatError> {
        if object_id == 0 {
            return Err(MatError::Custom(
                "opaque object id 0 is invalid (ids are 1-based)".into(),
            ));
        }
        let idx = MCOS_RESERVED_CELL_PREFIX + (object_id.to_usize()? - 1);
        let cell = self.cells.get(idx).ok_or_else(|| {
            MatError::Custom(format!(
                "opaque object id {object_id} maps to MCOS cell {idx}, but the store has {} cells",
                self.cells.len()
            ))
        })?;
        match cell {
            Object::Dataset(d) => {
                // The saveobj payload is always a `uint64` array. Refuse any
                // other datatype rather than letting `read_u64` widen a narrower
                // or floating-point cell into `u64`.
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

impl FileWrapper {
    /// Parse the FileWrapper metadata blob (cell 0 of the MCOS store).
    fn parse(blob: &[u8]) -> Result<Self, MatError> {
        if blob.len() < FILEWRAPPER_HEADER_LEN {
            return Err(MatError::Custom(format!(
                "FileWrapper blob too short: {} bytes (need at least {FILEWRAPPER_HEADER_LEN})",
                blob.len()
            )));
        }
        // version (blob[0..4]) is read but not gated: the region layout is
        // self-describing via the offsets, so a 2..=4 version parses uniformly.
        let num_strings = read_u32(blob, 4)?.to_usize()?;

        let mut offsets = [0usize; FILEWRAPPER_NUM_OFFSETS];
        for (i, slot) in offsets.iter_mut().enumerate() {
            let v = read_u32(blob, 8 + i * 4)?.to_usize()?;
            if v > blob.len() {
                return Err(MatError::Custom(format!(
                    "FileWrapper region offset {i} = {v} exceeds blob length {}",
                    blob.len()
                )));
            }
            *slot = v;
        }
        // The five region offsets used here must be non-decreasing and start
        // past the header so the name heap and tables carve cleanly.
        if offsets[0] < FILEWRAPPER_HEADER_LEN {
            return Err(MatError::Custom(
                "FileWrapper name-heap offset overlaps the header".into(),
            ));
        }
        for w in offsets.windows(2) {
            if w[1] < w[0] {
                return Err(MatError::Custom(
                    "FileWrapper region offsets are not non-decreasing".into(),
                ));
            }
        }

        let names = parse_names(blob, FILEWRAPPER_HEADER_LEN, offsets[0], num_strings)?;
        let classes = parse_class_table(blob, offsets[0], offsets[1])?;
        let seg1 = parse_property_blocks(blob, offsets[1], offsets[2])?;
        let objects = parse_object_table(blob, offsets[2], offsets[3])?;
        let seg2 = parse_property_blocks(blob, offsets[3], offsets[4])?;

        Ok(Self {
            names,
            classes,
            objects,
            seg1,
            seg2,
        })
    }
}

/// Parse `num` NUL-terminated names from `blob[start..end]`.
fn parse_names(blob: &[u8], start: usize, end: usize, num: usize) -> Result<Vec<String>, MatError> {
    let region = blob
        .get(start..end)
        .ok_or_else(|| MatError::Custom("FileWrapper name heap out of range".into()))?;
    let mut names = Vec::with_capacity(num);
    let mut iter = region.split(|&b| b == 0);
    for _ in 0..num {
        let bytes = iter.next().ok_or_else(|| {
            MatError::Custom("FileWrapper name heap ended before all names were read".into())
        })?;
        // Names are ASCII identifiers; decode lossily rather than fail the whole
        // read on an unexpected byte.
        names.push(String::from_utf8_lossy(bytes).into_owned());
    }
    Ok(names)
}

/// Parse the class table. The leading reserved entry is kept as `classes[0]` so
/// a 1-based `class_id` indexes directly.
fn parse_class_table(blob: &[u8], start: usize, end: usize) -> Result<Vec<ClassEntry>, MatError> {
    let stride = CLASS_ENTRY_WORDS * 4;
    let mut classes = Vec::new();
    let mut off = start;
    while off + stride <= end {
        classes.push(ClassEntry {
            namespace_idx: read_u32(blob, off)?,
            name_idx: read_u32(blob, off + 4)?,
        });
        off += stride;
    }
    Ok(classes)
}

/// Parse the object table. The leading reserved entry is kept as `objects[0]` so
/// a 1-based `object_id` indexes directly.
fn parse_object_table(
    blob: &[u8],
    start: usize,
    end: usize,
) -> Result<Vec<ObjectRecord>, MatError> {
    let stride = OBJECT_ENTRY_WORDS * 4;
    let mut objects = Vec::new();
    let mut off = start;
    while off + stride <= end {
        objects.push(ObjectRecord {
            class_id: read_u32(blob, off)?,
            // words at off+4, off+8 are reserved zeros
            saveobj_id: read_u32(blob, off + 12)?,
            normalobj_id: read_u32(blob, off + 16)?,
            dependency_id: read_u32(blob, off + 20)?,
        });
        off += stride;
    }
    Ok(objects)
}

/// Parse a property-table region into blocks. Each block is `[nprops,
/// (name_idx, field_type, value)…]` padded to an 8-byte boundary; the region
/// opens with a reserved empty block (kept as block 0).
fn parse_property_blocks(
    blob: &[u8],
    start: usize,
    end: usize,
) -> Result<Vec<PropBlock>, MatError> {
    let triple_stride = PROP_TRIPLE_WORDS * 4;
    let mut blocks = Vec::new();
    let mut pos = start;
    while pos + 4 <= end {
        let nprops = read_u32(blob, pos)?.to_usize()?;
        pos += 4;
        let triples_bytes = nprops
            .checked_mul(triple_stride)
            .ok_or_else(|| MatError::Custom("MCOS property block size overflow".into()))?;
        let triples_end = pos
            .checked_add(triples_bytes)
            .ok_or_else(|| MatError::Custom("MCOS property block size overflow".into()))?;
        if triples_end > end {
            return Err(MatError::Custom(
                "MCOS property block overruns its region".into(),
            ));
        }
        let mut props = Vec::with_capacity(nprops);
        for _ in 0..nprops {
            props.push(Triple {
                name_idx: read_u32(blob, pos)?,
                field_type: read_u32(blob, pos + 4)?,
                value: read_u32(blob, pos + 8)?,
            });
            pos += triple_stride;
        }
        blocks.push(PropBlock { props });
        // Pad to the next 8-byte boundary (absolute offset in the blob).
        let rem = pos % 8;
        if rem != 0 {
            pos += 8 - rem;
        }
    }
    Ok(blocks)
}

/// Read a little-endian `u32` at byte offset `off`, bounds-checked.
fn read_u32(blob: &[u8], off: usize) -> Result<u32, MatError> {
    let bytes = blob
        .get(off..off + 4)
        .ok_or_else(|| MatError::Custom(format!("FileWrapper read past end at byte {off}")))?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

/// Turn an inline `field_type == 2` literal into a scalar: `0`/`1` are MATLAB
/// logicals, anything else a `uint32`.
pub(crate) fn inline_to_scalar(value: u32) -> ScalarNum {
    if value <= 1 {
        ScalarNum::Bool(value != 0)
    } else {
        ScalarNum::U32(value)
    }
}

/// Finish an opaque object given its class name and resolved properties:
/// dispatch to a typed decoder, or fall back to a lossless [`MatValue::Opaque`].
pub(crate) fn decode_object_fields(
    class_name: String,
    fields: Vec<(String, MatValue)>,
) -> Result<MatValue, MatError> {
    match class_name.as_str() {
        "datetime" => decode_datetime(class_name, fields),
        "duration" => decode_duration(class_name, fields),
        "categorical" => decode_categorical(class_name, fields),
        // Every other opaque class (table, containers.Map, dictionary, user
        // classdefs, …) is surfaced losslessly with its raw properties.
        _ => Ok(MatValue::Opaque { class_name, fields }),
    }
}

/// Decode a `datetime` object. Its `data` property is a complex `double` whose
/// real part is milliseconds since the Unix epoch (1970-01-01 UTC) and whose
/// imaginary part is a sub-millisecond correction. Both halves are surfaced raw
/// and losslessly (`millis_utc`, `sub_ms`); `tz`/`fmt` are display metadata.
fn decode_datetime(
    class_name: String,
    mut fields: Vec<(String, MatValue)>,
) -> Result<MatValue, MatError> {
    let data = take_field(&mut fields, "data")
        .ok_or_else(|| MatError::Custom("datetime object is missing its `data` property".into()))?;
    let pairs = complex_pairs(data, "datetime `data`")?;
    let millis_utc: Vec<f64> = pairs.iter().map(|&(re, _)| re).collect();
    let sub_ms: Vec<f64> = pairs.iter().map(|&(_, im)| im).collect();

    let mut out = vec![
        (
            "millis_utc".to_owned(),
            MatValue::Vec1D(NumVec::F64(millis_utc)),
        ),
        ("sub_ms".to_owned(), MatValue::Vec1D(NumVec::F64(sub_ms))),
    ];
    // `tz` is sometimes named `tmz`; carry whichever is present.
    if let Some(tz) = take_field(&mut fields, "tz").or_else(|| take_field(&mut fields, "tmz")) {
        out.push(("tz".to_owned(), string_or_empty(tz)));
    }
    if let Some(fmt) = take_field(&mut fields, "fmt") {
        out.push(("fmt".to_owned(), string_or_empty(fmt)));
    }
    Ok(MatValue::Opaque {
        class_name,
        fields: out,
    })
}

/// Decode a `duration` object. Its `millis` property is a real `double` in
/// milliseconds; `fmt` is a display unit.
fn decode_duration(
    class_name: String,
    mut fields: Vec<(String, MatValue)>,
) -> Result<MatValue, MatError> {
    let millis = take_field(&mut fields, "millis").ok_or_else(|| {
        MatError::Custom("duration object is missing its `millis` property".into())
    })?;
    let millis = numeric_f64_vec(millis, "duration `millis`")?;
    let mut out = vec![("millis".to_owned(), MatValue::Vec1D(NumVec::F64(millis)))];
    if let Some(fmt) = take_field(&mut fields, "fmt") {
        out.push(("fmt".to_owned(), string_or_empty(fmt)));
    }
    Ok(MatValue::Opaque {
        class_name,
        fields: out,
    })
}

/// Decode a `categorical` object: integer `codes` (1-based, `0` =
/// `<undefined>`), the `categoryNames` cell, and the inline `isOrdinal` /
/// `isProtected` logicals.
fn decode_categorical(
    class_name: String,
    mut fields: Vec<(String, MatValue)>,
) -> Result<MatValue, MatError> {
    // An empty or fully-default categorical (e.g. `categorical({})`) is stored
    // by MATLAB with *no* properties at all, so `codes` can be legitimately
    // absent. Decode that as an empty categorical (empty codes, no categories)
    // rather than erroring, which would otherwise abort the whole-file read.
    let codes = take_field(&mut fields, "codes")
        .map(flatten_to_1d)
        .unwrap_or_else(|| MatValue::Vec1D(NumVec::U32(Vec::new())));
    let categories = take_field(&mut fields, "categoryNames")
        .or_else(|| take_field(&mut fields, "categories"))
        .unwrap_or(MatValue::Cell(Vec::new()));
    let is_ordinal =
        take_field(&mut fields, "isOrdinal").unwrap_or(MatValue::Scalar(ScalarNum::Bool(false)));
    let is_protected =
        take_field(&mut fields, "isProtected").unwrap_or(MatValue::Scalar(ScalarNum::Bool(false)));

    Ok(MatValue::Opaque {
        class_name,
        fields: vec![
            ("codes".to_owned(), codes),
            ("categories".to_owned(), categories),
            ("is_ordinal".to_owned(), is_ordinal),
            ("is_protected".to_owned(), is_protected),
        ],
    })
}

/// Remove and return the first field named `name`, if present.
fn take_field(fields: &mut Vec<(String, MatValue)>, name: &str) -> Option<MatValue> {
    let pos = fields.iter().position(|(n, _)| n == name)?;
    Some(fields.remove(pos).1)
}

/// Coerce a decoded property to a `String` (empty if it is not a char string).
fn string_or_empty(value: MatValue) -> MatValue {
    match value {
        MatValue::String(_) => value,
        _ => MatValue::String(String::new()),
    }
}

/// Flatten a real numeric property into a `Vec<f64>`.
fn numeric_f64_vec(value: MatValue, what: &str) -> Result<Vec<f64>, MatError> {
    match value {
        MatValue::Scalar(s) => Ok(vec![scalar_to_f64(s)]),
        MatValue::Vec1D(v) => Ok(numvec_to_f64(v)),
        MatValue::Matrix { vec, .. } => Ok(numvec_to_f64(vec)),
        other => Err(MatError::Custom(format!(
            "expected a real numeric array for {what}, got {}",
            other.kind()
        ))),
    }
}

/// Flatten a complex (or real) numeric property into `(re, im)` pairs.
fn complex_pairs(value: MatValue, what: &str) -> Result<Vec<(f64, f64)>, MatError> {
    Ok(match value {
        MatValue::ComplexScalar64 { re, im } => vec![(re, im)],
        MatValue::ComplexScalar32 { re, im } => vec![(f64::from(re), f64::from(im))],
        MatValue::ComplexVec64(pairs) => pairs,
        MatValue::ComplexVec32(pairs) => pairs
            .into_iter()
            .map(|(r, i)| (f64::from(r), f64::from(i)))
            .collect(),
        MatValue::ComplexMatrix64 { pairs, .. } => pairs,
        MatValue::ComplexMatrix32 { pairs, .. } => pairs
            .into_iter()
            .map(|(r, i)| (f64::from(r), f64::from(i)))
            .collect(),
        // A datetime with no sub-millisecond component may store `data` as a
        // plain real double; treat the imaginary part as zero.
        MatValue::Scalar(s) => vec![(scalar_to_f64(s), 0.0)],
        MatValue::Vec1D(v) => numvec_to_f64(v).into_iter().map(|r| (r, 0.0)).collect(),
        MatValue::Matrix { vec, .. } => numvec_to_f64(vec).into_iter().map(|r| (r, 0.0)).collect(),
        other => {
            return Err(MatError::Custom(format!(
                "expected a complex double array for {what}, got {}",
                other.kind()
            )));
        }
    })
}

/// Flatten a numeric property to a 1-D vector, preserving its integer width.
/// A 2-D categorical's `codes` matrix is already stored row-major, so its
/// backing vector is taken as-is; a scalar becomes a length-1 vector. Used so
/// `categorical` `codes` always surface as a flat 1-D array (matching how
/// `datetime`/`duration` arrays flatten), regardless of the source rank.
fn flatten_to_1d(value: MatValue) -> MatValue {
    match value {
        MatValue::Matrix { vec, .. } => MatValue::Vec1D(vec),
        MatValue::Scalar(s) => MatValue::Vec1D(scalar_to_numvec(s)),
        other => other,
    }
}

/// Wrap a numeric scalar in a length-1 [`NumVec`] of the same width.
fn scalar_to_numvec(s: ScalarNum) -> NumVec {
    match s {
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
        ScalarNum::Bool(b) => NumVec::Bool(vec![b]),
    }
}

fn numvec_to_f64(v: NumVec) -> Vec<f64> {
    match v {
        NumVec::F64(x) => x,
        NumVec::F32(x) => x.into_iter().map(f64::from).collect(),
        NumVec::I64(x) => x.into_iter().map(|v| v as f64).collect(),
        NumVec::I32(x) => x.into_iter().map(f64::from).collect(),
        NumVec::I16(x) => x.into_iter().map(f64::from).collect(),
        NumVec::I8(x) => x.into_iter().map(f64::from).collect(),
        NumVec::U64(x) => x.into_iter().map(|v| v as f64).collect(),
        NumVec::U32(x) => x.into_iter().map(f64::from).collect(),
        NumVec::U16(x) => x.into_iter().map(f64::from).collect(),
        NumVec::U8(x) => x.into_iter().map(f64::from).collect(),
        NumVec::Bool(x) => x.into_iter().map(|b| if b { 1.0 } else { 0.0 }).collect(),
    }
}

fn scalar_to_f64(s: ScalarNum) -> f64 {
    match s {
        ScalarNum::F64(x) => x,
        ScalarNum::F32(x) => f64::from(x),
        ScalarNum::I64(x) => x as f64,
        ScalarNum::I32(x) => f64::from(x),
        ScalarNum::I16(x) => f64::from(x),
        ScalarNum::I8(x) => f64::from(x),
        ScalarNum::U64(x) => x as f64,
        ScalarNum::U32(x) => f64::from(x),
        ScalarNum::U16(x) => f64::from(x),
        ScalarNum::U8(x) => f64::from(x),
        ScalarNum::Bool(b) => {
            if b {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// Metadata parsed from an opaque parent dataset's `uint32` array.
pub(crate) struct OpaqueMeta {
    pub(crate) object_ids: Vec<u32>,
    pub(crate) class_id: u32,
}

/// Parse an opaque parent dataset's `uint32` metadata array:
/// `[MAGIC, ndims, dims…, object_ids…, class_id]`.
pub(crate) fn parse_opaque_metadata(data: &[u32]) -> Result<OpaqueMeta, MatError> {
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
    Ok(OpaqueMeta {
        object_ids: data[dims_end..ids_end].to_vec(),
        class_id: data[ids_end],
    })
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
pub(crate) fn raw_matlab_class(attrs: &HashMap<String, AttrValue>) -> Option<String> {
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

/// Whether a `MATLAB_object_decode` value marks an `mxOPAQUE_CLASS` (MCOS)
/// object — the only kind decoded here. `1` (function handle) and `2` (legacy
/// object) have different on-disk layouts and are not yet supported.
pub(crate) fn is_mcos_decode(decode: i64) -> bool {
    decode == i64::from(MATLAB_OBJECT_DECODE_OPAQUE)
}

/// `MATLAB_object_decode` value for the modern `string` class.
pub(crate) fn is_string_class(class: &str, decode: i64) -> bool {
    is_mcos_decode(decode) && class == MATLAB_CLASS_STRING
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::string_object::build_string_filewrapper_metadata;

    // The production string writer emits a FileWrapper blob that was
    // reverse-engineered from real MATLAB output, so parsing it validates the
    // parser against real-MATLAB-derived bytes (not against itself).
    #[test]
    fn parses_production_string_filewrapper_blob() {
        let blob = build_string_filewrapper_metadata(1);
        let fw = FileWrapper::parse(&blob).expect("production blob must parse");

        // Name heap: "any", "string".
        assert_eq!(fw.names, vec!["any".to_owned(), "string".to_owned()]);

        // One real class after the reserved leading entry: class 1 = "string".
        assert_eq!(fw.classes.len(), 2);
        assert_eq!(fw.classes[1].namespace_idx, 0);
        assert_eq!(fw.classes[1].name_idx, 2);

        // One real object after the reserved leading entry: class 1, stored in
        // the type-1 ("saveobj") segment.
        assert_eq!(fw.objects.len(), 2);
        assert_eq!(fw.objects[1].class_id, 1);
        assert_eq!(fw.objects[1].saveobj_id, 1);
        assert_eq!(fw.objects[1].normalobj_id, 0);

        // The saveobj property block points at heap value 0 (cell index 2).
        assert_eq!(fw.seg1.len(), 2);
        let block = &fw.seg1[1];
        assert_eq!(block.props.len(), 1);
        assert_eq!(block.props[0].field_type, FIELD_TYPE_HEAP);
        assert_eq!(block.props[0].value, 0);
    }

    #[test]
    fn parses_multi_object_production_blob() {
        let blob = build_string_filewrapper_metadata(3);
        let fw = FileWrapper::parse(&blob).expect("production blob must parse");
        // Three objects after the reserved entry, each in the saveobj segment.
        assert_eq!(fw.objects.len(), 4);
        for id in 1..=3u32 {
            assert_eq!(fw.objects[id as usize].class_id, 1);
            assert_eq!(fw.objects[id as usize].saveobj_id, id);
        }
        // Each object's block points at heap value id-1 (cell id+1).
        assert_eq!(fw.seg1.len(), 4);
        for id in 1..=3u32 {
            let block = &fw.seg1[id as usize];
            assert_eq!(block.props[0].value, id - 1);
        }
    }

    #[test]
    fn rejects_short_blob() {
        assert!(matches!(
            FileWrapper::parse(&[0u8; 8]),
            Err(MatError::Custom(_))
        ));
    }

    #[test]
    fn inline_logical_and_integer() {
        assert!(matches!(inline_to_scalar(0), ScalarNum::Bool(false)));
        assert!(matches!(inline_to_scalar(1), ScalarNum::Bool(true)));
        assert!(matches!(inline_to_scalar(7), ScalarNum::U32(7)));
    }

    #[test]
    fn parse_opaque_metadata_extracts_ids_and_class() {
        // [MAGIC, ndims=2, dims=1×1, object_id=4, class_id=2]
        let meta = [MCOS_MAGIC_NUMBER, 2, 1, 1, 4, 2];
        let parsed = parse_opaque_metadata(&meta).unwrap();
        assert_eq!(parsed.object_ids, vec![4]);
        assert_eq!(parsed.class_id, 2);
    }

    #[test]
    fn parse_opaque_metadata_rejects_bad_magic() {
        let meta = [0x1234_5678, 2, 1, 1, 1, 1];
        assert!(parse_opaque_metadata(&meta).is_err());
    }
}
