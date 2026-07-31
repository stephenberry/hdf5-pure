//! A gallery of every value this crate renders with `Display`, for review.
//!
//! ```bash
//! cargo test display_gallery -- --nocapture
//! ```
//!
//! writes `display_gallery.md` in the crate root (override the path with
//! `DISPLAY_GALLERY_OUT`) and prints it. Run it once at the commit before the
//! `Display` impls and once after for a before/after pair: this file compiles
//! against both versions, so the two documents differ only in what the crate
//! renders. See `render!` and `AttrTypeName` below for how.
//!
//! It lives in `src/` rather than `examples/` because an example is a separate
//! crate, and half of these values cannot be spelled from outside: `Filter`,
//! `CompoundMember` and `EnumMember` are `#[non_exhaustive]` structs,
//! `MessageType` and `ChunkIndex::from_layout` are crate-private.

use std::fmt;

use crate::datatype::{
    CharacterSet, CompoundMember, Datatype, DatatypeByteOrder, EnumMember, ReferenceType,
    StringPadding,
};
use crate::error::Error;
use crate::layout_info::{ChunkIndex, Filter, Layout};
use crate::message_type::MessageType;
use crate::reader::File;
use crate::type_builders::{AttrValue, make_f32_type, make_f64_type, make_i32_type, make_i64_type};
use crate::types::DType;
use crate::writer::FileBuilder;

// ---- Version shims -----------------------------------------------------
//
// Both snapshots come from this one file, which therefore has to compile
// against the version where these `Display` impls and `AttrValue::type_name`
// do not exist yet.

/// Carries a value to render. See [`render`].
struct Show<T>(T);

#[allow(dead_code)] // whichever of the two the version does not need
trait ViaDisplay {
    fn render(&self) -> String;
}

#[allow(dead_code)] // whichever of the two the version does not need
trait ViaDebug {
    fn render(&self) -> String;
}

impl<T: fmt::Display> ViaDisplay for Show<T> {
    fn render(&self) -> String {
        self.0.to_string()
    }
}

impl<T: fmt::Debug> ViaDebug for &Show<T> {
    fn render(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// Render a value through `Display` where the type has one, and through `Debug`
/// where it does not — which is precisely what a caller had before.
///
/// Called on a `&Show<T>`, method lookup takes `ViaDisplay` before `ViaDebug`
/// because its receiver needs no further reference (the autoref trick `anyhow`
/// uses), and falls to `ViaDebug` when `T: Display` does not hold.
macro_rules! render {
    ($value:expr) => {{
        let shown = Show(&$value);
        (&shown).render()
    }};
}

/// Stands in for `AttrValue::type_name` on the version that has none. An
/// inherent method wins over a trait one, so the real method is used where it
/// exists.
#[allow(dead_code)] // used only on the version before `type_name` existed
trait AttrTypeName {
    fn type_name(&self) -> &'static str;
}

impl AttrTypeName for AttrValue {
    fn type_name(&self) -> &'static str {
        "(no type_name on this version)"
    }
}

// ---- The gallery -------------------------------------------------------

/// One rendered value: what it is, and how it comes out.
type Row = (String, String);

/// A source expression as one line, since `stringify!` keeps the line breaks
/// rustfmt put in a long literal and a table cell holds no newline.
fn one_line(source: &str) -> String {
    source.split_whitespace().collect::<Vec<&str>>().join(" ")
}

/// Push a row, labeling it with the expression itself unless given a label.
macro_rules! row {
    ($rows:expr, $value:expr) => {
        $rows.push((
            format!("`{}`", one_line(stringify!($value))),
            render!($value),
        ))
    };
    ($rows:expr, $label:expr, $value:expr) => {
        $rows.push(($label.to_owned(), render!($value)))
    };
}

/// The value an attribute holds.
fn attr_values() -> Vec<Row> {
    let mut rows = Vec::new();

    row!(rows, AttrValue::F64(1.5));
    row!(rows, AttrValue::F64(1.0));
    row!(rows, AttrValue::I32(-7));
    row!(rows, AttrValue::I64(7));
    row!(rows, AttrValue::U32(3));
    row!(rows, AttrValue::U64(u64::MAX));
    row!(rows, AttrValue::String("metres".into()));
    row!(rows, AttrValue::AsciiString("lab_a".into()));
    row!(rows, AttrValue::F64Array(vec![1.0, 2.5]));
    row!(rows, AttrValue::F64Array(vec![]));
    row!(rows, AttrValue::I64Array(vec![1, 2, 3]));
    row!(rows, AttrValue::U64Array(vec![u64::MAX]));
    row!(rows, AttrValue::StringArray(vec!["a".into(), "b".into()]));
    row!(rows, AttrValue::AsciiStringArray(vec!["double".into()]));
    row!(
        rows,
        AttrValue::VarLenAsciiArray(vec!["x".into(), "y".into()])
    );
    row!(
        rows,
        "`AttrValue::I64Array` of 13 elements",
        AttrValue::I64Array((0..13).collect())
    );

    rows
}

/// The name of the type an attribute holds.
fn attr_type_names() -> Vec<Row> {
    let mut rows = Vec::new();

    row!(rows, AttrValue::F64(0.0).type_name());
    row!(rows, AttrValue::F64Array(vec![]).type_name());
    row!(rows, AttrValue::I32(0).type_name());
    row!(rows, AttrValue::I64(0).type_name());
    row!(rows, AttrValue::I64Array(vec![]).type_name());
    row!(rows, AttrValue::U32(0).type_name());
    row!(rows, AttrValue::U64(0).type_name());
    row!(rows, AttrValue::U64Array(vec![]).type_name());
    row!(rows, AttrValue::String(String::new()).type_name());
    row!(rows, AttrValue::StringArray(vec![]).type_name());
    row!(rows, AttrValue::AsciiString(String::new()).type_name());
    row!(rows, AttrValue::AsciiStringArray(vec![]).type_name());
    row!(rows, AttrValue::VarLenAsciiArray(vec![]).type_name());

    rows
}

/// A parsed datatype, one per class the format defines.
fn datatypes() -> Vec<Row> {
    let mut rows = Vec::new();

    row!(rows, "a 4-byte signed integer", make_i32_type());
    row!(rows, "an 8-byte float", make_f64_type());
    row!(
        rows,
        "a 2-byte unsigned big-endian integer",
        Datatype::FixedPoint {
            size: 2,
            byte_order: DatatypeByteOrder::BigEndian,
            signed: false,
            bit_offset: 0,
            bit_precision: 16,
        }
    );
    row!(
        rows,
        "a 4-byte integer carrying 24 bits",
        Datatype::FixedPoint {
            size: 4,
            byte_order: DatatypeByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 24,
        }
    );
    row!(
        rows,
        "a Vax-ordered 4-byte float",
        Datatype::FloatingPoint {
            size: 4,
            byte_order: DatatypeByteOrder::Vax,
            bit_offset: 0,
            bit_precision: 32,
            exponent_location: 23,
            exponent_size: 8,
            mantissa_location: 0,
            mantissa_size: 23,
            exponent_bias: 127,
        }
    );
    row!(
        rows,
        "a 4-byte time",
        Datatype::Time {
            size: 4,
            byte_order: DatatypeByteOrder::LittleEndian,
            bit_precision: 32,
        }
    );
    row!(
        rows,
        "a 16-byte UTF-8 null-padded string",
        Datatype::String {
            size: 16,
            padding: StringPadding::NullPad,
            charset: CharacterSet::Utf8,
        }
    );
    row!(
        rows,
        "an 8-byte ASCII space-padded string",
        Datatype::String {
            size: 8,
            padding: StringPadding::SpacePad,
            charset: CharacterSet::Ascii,
        }
    );
    row!(
        rows,
        "a 1-byte bit field",
        Datatype::BitField {
            size: 1,
            byte_order: DatatypeByteOrder::LittleEndian,
            bit_offset: 0,
            bit_precision: 8,
        }
    );
    row!(
        rows,
        "a 2-byte bit field carrying bits 3..7",
        Datatype::BitField {
            size: 2,
            byte_order: DatatypeByteOrder::LittleEndian,
            bit_offset: 3,
            bit_precision: 4,
        }
    );
    row!(
        rows,
        "a 3-byte opaque type tagged `rgb`",
        Datatype::Opaque {
            size: 3,
            tag: b"rgb".to_vec(),
        }
    );
    row!(
        rows,
        "a 4-byte opaque type whose tag holds a quote and a NUL",
        Datatype::Opaque {
            size: 4,
            tag: b"a\"b\x00".to_vec(),
        }
    );
    row!(
        rows,
        "an untagged 8-byte opaque type",
        Datatype::Opaque {
            size: 8,
            tag: Vec::new(),
        }
    );
    row!(
        rows,
        "a compound of `x: f32` and `n: i64`",
        Datatype::Compound {
            size: 12,
            members: vec![
                CompoundMember {
                    name: "x".into(),
                    byte_offset: 0,
                    datatype: make_f32_type(),
                },
                CompoundMember {
                    name: "n".into(),
                    byte_offset: 4,
                    datatype: make_i64_type(),
                },
            ],
        }
    );
    row!(
        rows,
        "an object reference",
        Datatype::Reference {
            size: 8,
            ref_type: ReferenceType::Object,
        }
    );
    row!(
        rows,
        "a dataset-region reference",
        Datatype::Reference {
            size: 12,
            ref_type: ReferenceType::DatasetRegion,
        }
    );
    row!(
        rows,
        "an i32 enumeration of `red`, `green`, `blue`",
        Datatype::Enumeration {
            size: 4,
            base_type: Box::new(make_i32_type()),
            members: vec![
                EnumMember {
                    name: "red".into(),
                    value: 0i32.to_le_bytes().to_vec(),
                },
                EnumMember {
                    name: "green".into(),
                    value: 1i32.to_le_bytes().to_vec(),
                },
                EnumMember {
                    name: "blue".into(),
                    value: 2i32.to_le_bytes().to_vec(),
                },
            ],
        }
    );
    row!(
        rows,
        "a variable-length UTF-8 string",
        Datatype::VariableLength {
            is_string: true,
            padding: Some(StringPadding::NullTerminate),
            charset: Some(CharacterSet::Utf8),
            base_type: Box::new(Datatype::String {
                size: 1,
                padding: StringPadding::NullTerminate,
                charset: CharacterSet::Utf8,
            }),
        }
    );
    row!(
        rows,
        "a variable-length string with no recorded charset",
        Datatype::VariableLength {
            is_string: true,
            padding: None,
            charset: None,
            base_type: Box::new(Datatype::String {
                size: 1,
                padding: StringPadding::NullTerminate,
                charset: CharacterSet::Ascii,
            }),
        }
    );
    row!(
        rows,
        "a variable-length sequence of i32",
        Datatype::VariableLength {
            is_string: false,
            padding: None,
            charset: None,
            base_type: Box::new(make_i32_type()),
        }
    );
    row!(
        rows,
        "a 2x3 array of u8",
        Datatype::Array {
            base_type: Box::new(Datatype::FixedPoint {
                size: 1,
                byte_order: DatatypeByteOrder::LittleEndian,
                signed: false,
                bit_offset: 0,
                bit_precision: 8,
            }),
            dimensions: vec![2, 3],
        }
    );
    row!(
        rows,
        "a 4-element array of f32",
        Datatype::Array {
            base_type: Box::new(make_f32_type()),
            dimensions: vec![4],
        }
    );

    rows
}

/// The enums a datatype is described with.
fn datatype_parts() -> Vec<Row> {
    let mut rows = Vec::new();

    row!(rows, DatatypeByteOrder::LittleEndian);
    row!(rows, DatatypeByteOrder::BigEndian);
    row!(rows, DatatypeByteOrder::Vax);
    row!(rows, StringPadding::NullTerminate);
    row!(rows, StringPadding::NullPad);
    row!(rows, StringPadding::SpacePad);
    row!(rows, CharacterSet::Ascii);
    row!(rows, CharacterSet::Utf8);
    row!(rows, ReferenceType::Object);
    row!(rows, ReferenceType::DatasetRegion);

    rows
}

/// The curated element type a caller reads off a dataset.
fn dtypes() -> Vec<Row> {
    let mut rows = Vec::new();

    row!(rows, DType::F64);
    row!(rows, DType::VariableLengthString);
    row!(rows, DType::Array(Box::new(DType::F32), vec![2, 3]));
    row!(rows, DType::Array(Box::new(DType::U8), vec![4]));

    // What `classify_datatype` builds for a class it does not curate: the
    // summary of the whole `Datatype`.
    let time = Datatype::Time {
        size: 4,
        byte_order: DatatypeByteOrder::LittleEndian,
        bit_precision: 32,
    };
    row!(
        rows,
        "`DType::Other` for a 4-byte time type",
        DType::Other(render!(time))
    );

    let opaque = Datatype::Opaque {
        size: 3,
        tag: b"rgb".to_vec(),
    };
    row!(
        rows,
        "`DType::Other` for a 3-byte opaque type tagged `rgb`",
        DType::Other(render!(opaque))
    );

    rows
}

/// An object-header message, as `Error::MissingMessage` names it.
fn message_types() -> Vec<Row> {
    let mut rows = Vec::new();

    row!(rows, MessageType::Nil);
    row!(rows, MessageType::Dataspace);
    row!(rows, MessageType::LinkInfo);
    row!(rows, MessageType::Datatype);
    row!(rows, MessageType::FillValueOld);
    row!(rows, MessageType::FillValue);
    row!(rows, MessageType::Link);
    row!(rows, MessageType::DataLayout);
    row!(rows, MessageType::GroupInfo);
    row!(rows, MessageType::FilterPipeline);
    row!(rows, MessageType::Attribute);
    row!(rows, MessageType::ObjectHeaderContinuation);
    row!(rows, MessageType::SymbolTable);
    row!(rows, MessageType::ObjectModificationTime);
    row!(rows, MessageType::BTreeKValues);
    row!(rows, MessageType::SharedMessageTable);
    row!(rows, MessageType::AttributeInfo);
    row!(rows, MessageType::ObjectReferenceCount);
    row!(rows, MessageType::FileSpaceInfo);
    row!(rows, MessageType::Unknown(0x00ff));

    rows
}

/// How a dataset's raw data is arranged, and how its chunks are indexed.
fn layouts() -> Vec<Row> {
    let mut rows = Vec::new();

    row!(rows, Layout::Compact { size: 40 });
    row!(
        rows,
        Layout::Contiguous {
            address: Some(0x2a0),
            size: 128,
        }
    );
    row!(
        rows,
        Layout::Contiguous {
            address: None,
            size: 64,
        }
    );
    row!(
        rows,
        Layout::Chunked {
            chunk_shape: vec![1000],
            index: ChunkIndex::BTreeV1,
        }
    );
    row!(
        rows,
        Layout::Chunked {
            chunk_shape: vec![4, 8],
            index: ChunkIndex::ExtensibleArray,
        }
    );
    row!(rows, Layout::Virtual);

    row!(rows, ChunkIndex::BTreeV1);
    row!(rows, ChunkIndex::SingleChunk);
    row!(rows, ChunkIndex::Implicit);
    row!(rows, ChunkIndex::FixedArray);
    row!(rows, ChunkIndex::ExtensibleArray);
    row!(rows, ChunkIndex::BTreeV2);

    rows
}

/// One filter of a dataset's pipeline.
fn filters() -> Vec<Row> {
    let filter = |id: u16, name: Option<&str>, is_optional: bool, client_data: Vec<u32>| Filter {
        id,
        name: name.map(str::to_owned),
        is_optional,
        client_data,
    };

    let mut rows = Vec::new();

    row!(rows, "deflate at level 6", filter(1, None, false, vec![6]));
    row!(
        rows,
        "shuffle over 8-byte elements",
        filter(2, None, false, vec![8])
    );
    row!(rows, "fletcher32", filter(3, None, false, Vec::new()));
    row!(rows, "szip", filter(4, None, false, vec![4, 32]));
    row!(rows, "nbit", filter(5, None, false, Vec::new()));
    row!(rows, "scale-offset", filter(6, None, false, vec![0, 2]));
    row!(rows, "lzf", filter(32000, None, false, Vec::new()));
    row!(rows, "zfp", filter(32013, None, false, vec![5, 0, 0, 0]));
    row!(
        rows,
        "an unregistered filter the file names",
        filter(40000, Some("custom"), true, Vec::new())
    );
    row!(
        rows,
        "an unregistered filter the file does not name",
        filter(40001, None, false, Vec::new())
    );

    rows
}

/// The error messages this change rewords.
fn errors() -> Vec<Row> {
    let mut rows = Vec::new();

    row!(rows, Error::MissingMessage(MessageType::DataLayout));
    row!(rows, Error::MissingMessage(MessageType::Dataspace));
    row!(rows, ChunkIndex::from_layout(4, Some(9)).unwrap_err());
    row!(rows, ChunkIndex::from_layout(9, None).unwrap_err());

    rows
}

/// The same types as they come off a file this crate writes, which is where a
/// caller meets them.
fn from_a_written_file() -> Vec<Row> {
    let signal: Vec<f64> = (0..4000).map(|i| f64::from(i % 97) * 0.5).collect();
    let fast: Vec<f32> = (0..4000).map(|i| i as f32).collect();

    let mut builder = FileBuilder::new();
    builder.set_attr("title", AttrValue::String("experiment 7".into()));
    builder.set_attr("frequency", AttrValue::F64(50.0));
    builder.set_attr("operator", AttrValue::AsciiString("lab_a".into()));
    builder.set_attr("channels", AttrValue::I64Array((0..12).collect()));
    builder
        .create_dataset("signal")
        .with_f64_data(&signal)
        .with_chunks(&[1000])
        .with_shuffle()
        .with_deflate(6);
    builder
        .create_dataset("fast")
        .with_f32_data(&fast)
        .with_chunks(&[1000])
        .with_lzf()
        .with_fletcher32();
    builder
        .create_dataset("tiny")
        .with_f64_data(&[1.0, 2.0, 3.0]);
    builder
        .create_dataset("growable")
        .with_f64_data(&[1.0, 2.0, 3.0, 4.0])
        .with_chunks(&[4])
        .with_maxshape(&[u64::MAX]);
    builder
        .create_dataset("names")
        .with_vlen_strings(&["alpha", "beta"]);

    let file = File::from_bytes(builder.finish().expect("write the gallery file"))
        .expect("read the gallery file back");

    let mut rows = Vec::new();

    for name in ["signal", "fast", "tiny", "growable", "names"] {
        let dataset = file.dataset(name).expect("open the dataset");

        rows.push((
            format!("`/{name}` datatype"),
            render!(dataset.datatype().expect("read the datatype")),
        ));
        rows.push((
            format!("`/{name}` layout"),
            render!(dataset.layout().expect("read the layout")),
        ));

        let pipeline = dataset.filter_pipeline();
        if !pipeline.is_empty() {
            let rendered: Vec<String> = pipeline.iter().map(|filter| render!(filter)).collect();
            rows.push((format!("`/{name}` filter pipeline"), rendered.join(" → ")));
        }
    }

    let attrs = file.root().attrs().expect("read the root attributes");
    let mut names: Vec<&String> = attrs.keys().collect();
    names.sort();
    for name in names {
        let value = &attrs[name];
        rows.push((format!("`/` attribute `{name}`"), render!(value)));
        rows.push((
            format!("`/` attribute `{name}` type name"),
            render!(value.type_name()),
        ));
    }

    rows
}

// ---- The document ------------------------------------------------------

/// A markdown table cell: the rendering is arbitrary text, so it is quoted as
/// code with any pipe escaped.
fn cell(text: &str) -> String {
    format!("`{}`", text.replace('|', "\\|"))
}

fn section(doc: &mut String, title: &str, rows: &[Row]) {
    doc.push_str(&format!(
        "\n## {title}\n\n| item | rendering |\n| --- | --- |\n"
    ));
    for (label, rendering) in rows {
        doc.push_str(&format!("| {} | {} |\n", label, cell(rendering)));
    }
}

fn document() -> String {
    let mut doc = String::new();
    doc.push_str(&format!(
        "# hdf5-pure display gallery\n\n\
         Every value the crate describes to a caller, as it comes out. Rendered by \
         `src/display_gallery.rs` at crate version {}, through `Display` where the type \
         has one and through `Debug` where it does not.\n",
        env!("CARGO_PKG_VERSION")
    ));

    section(&mut doc, "Attribute values", &attr_values());
    section(&mut doc, "Attribute type names", &attr_type_names());
    section(&mut doc, "Datatypes", &datatypes());
    section(&mut doc, "Datatype parts", &datatype_parts());
    section(&mut doc, "Element types", &dtypes());
    section(&mut doc, "Header messages", &message_types());
    section(&mut doc, "Layouts and chunk indexes", &layouts());
    section(&mut doc, "Filters", &filters());
    section(&mut doc, "Error messages", &errors());
    section(
        &mut doc,
        "Read back off a written file",
        &from_a_written_file(),
    );

    doc
}

#[test]
fn display_gallery() {
    let doc = document();
    println!("{doc}");

    let path = std::env::var("DISPLAY_GALLERY_OUT")
        .unwrap_or_else(|_| format!("{}/display_gallery.md", env!("CARGO_MANIFEST_DIR")));
    std::fs::write(&path, &doc).expect("write the gallery");
    println!("wrote {path}");
}
