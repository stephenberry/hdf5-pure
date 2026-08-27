# Data Types

This page is the reference for every datatype hdf5-pure can write, read, and model. It maps the high-level dataset and attribute methods to their HDF5 encodings and documents the low-level datatype model (`Datatype` and its helpers) that those methods build on. For task-oriented walkthroughs see the [writing](../guide/writing.md), [reading](../guide/reading.md), and [compound types](../guide/compound-types.md) guides.

## Dataset write methods

`DatasetBuilder` exposes a typed method per supported element type. Each sets both the dataset's data and its HDF5 datatype, and defaults the shape to the 1-D `[len]` unless `with_shape` has already set one. All integer and float types are written little-endian.

| Method | HDF5 type |
|---|---|
| `with_data` (generic over [`H5Element`](#the-h5element-trait)) | Inferred from the element type |
| `with_f64_data` | IEEE 64-bit float |
| `with_f32_data` | IEEE 32-bit float |
| `with_i8_data` / `with_i16_data` / `with_i32_data` / `with_i64_data` | Signed integers |
| `with_u8_data` / `with_u16_data` / `with_u32_data` / `with_u64_data` | Unsigned integers |
| `with_ascii_strings` / `with_strings` (and their `_sized` forms) | Fixed-width strings, null-padded, ASCII or UTF-8 |
| `with_vlen_strings` | Variable-length UTF-8 strings (global heap) |
| `with_complex32_data` | Compound `{real: f32, imag: f32}` |
| `with_complex64_data` | Compound `{real: f64, imag: f64}` |
| `with_compound_data` | Arbitrary compound types (explicit datatype + raw bytes) |
| `with_compound_values` | Safely encoded numeric tuples (field by field) |
| `with_enum_i32_data` / `with_enum_u8_data` | Enumeration types |
| `with_array_data` | Fixed-size array types |
| `with_path_references` | Object references (resolved by path) |
| `with_dtype` + `with_shape` | Empty / zero-dimension datasets |

The generic `with_data(&[T])` is the counterpart of the typed family; it dispatches to the matching `with_*_data` method based on `T`. See the [generic I/O guide](../guide/generic-io.md).

```rust
use hdf5_pure::FileBuilder;

let mut builder = FileBuilder::new();
builder.create_dataset("temperature")
    .with_f64_data(&[22.5, 23.1, 21.8])
    .with_shape(&[3]);
```

`with_complex32_data` and `with_complex64_data` accept `&[(f32, f32)]` / `&[(f64, f64)]` and store each pair as a two-field compound. `with_compound_data` takes an explicit [`Datatype`](#the-datatype-model), the raw element bytes, and the element count, so the caller is responsible for the bytes matching the datatype's on-disk layout. `with_array_data` takes the array's base [`Datatype`](#the-datatype-model), the array dimensions, the raw element bytes, and the element count, and builds the `Datatype::Array` for you. `with_compound_values` is the safe alternative for numeric tuples: it encodes each field explicitly via the [`CompoundType`](#compoundtype-and-compoundfield) trait without copying Rust tuple padding. `with_enum_i32_data` / `with_enum_u8_data` take a datatype (built with [`EnumTypeBuilder`](#enumtypebuilder)) plus the raw values.

!!! note
    There is also a lower-level `with_raw_data(datatype, raw_data, num_elements)` that writes an explicit datatype and its raw bytes verbatim; `with_compound_data` delegates to it.

## Dataset read methods

`Dataset` reads return a flat `Vec<T>` in row-major order. The typed `read_*` methods coerce the stored bytes into the requested type, so a conversion can be lossy if the requested type does not match the stored datatype.

| Method | Returns |
|---|---|
| `read_f64` / `read_f32` | `Vec<f64>` / `Vec<f32>` |
| `read_i8` / `read_i16` / `read_i32` / `read_i64` | signed-integer vectors |
| `read_u8` / `read_u16` / `read_u32` / `read_u64` | unsigned-integer vectors |
| `read::<T>()` (generic over [`H5Element`](#the-h5element-trait)) | `Vec<T>` |
| `read_compound::<T>()` (over [`CompoundType`](#compoundtype-and-compoundfield)) | `Vec<T>` |
| `read_string` | `Vec<String>` (fixed- and variable-length string datasets) |
| `read_raw` | `Vec<u8>` (complete unfiltered record bytes) |
| `read_array` (`ndarray` feature) | `Array<T, D>` (static rank `D`) |
| `read_array_dyn` (`ndarray` feature) | `ArrayD<T>` (runtime rank) |

```rust
use hdf5_pure::File;

let file = File::open("output.h5").unwrap();
let ds = file.dataset("temperature").unwrap();
let values = ds.read_f64().unwrap();
```

`read::<T>()` requests delivery as `T`; it is not an assertion about the stored datatype, so pick `T` to match the stored type for a lossless read. `read_compound::<T>()` decodes each element using its exact on-disk datatype, reading field offsets from the file. `read_raw` returns the unfiltered record bytes verbatim, which pairs with `Dataset::datatype()` (see below) when you need to inspect or re-emit an arbitrary type. `read_array` and `read_array_dyn` require the `ndarray` feature; see the [ndarray guide](../guide/ndarray.md).

There are also VL-string reads beyond `read_string`: `read_vlen_strings(options)` and the streaming `visit_vlen_strings(options, f)`, bounded by `VlenStringReadOptions`, with `vlen_string_payload_size()` reporting the payload size up front. See the [variable-length strings guide](../guide/vlen-strings.md).

## Inspecting a dataset's type

Two accessors describe an existing dataset's type:

- `Dataset::dtype()` returns a simplified [`DType`](#the-dtype-classification) classification.
- `Dataset::datatype()` returns the full low-level [`Datatype`](#the-datatype-model), including exact compound field offsets.

## Attribute values

`AttrValue` is the write-side attribute enum. Reading attributes yields the same enum (a `HashMap<String, AttrValue>` from `attrs()`), and a number keeps the width it is stored at: a 2-byte signed attribute arrives as `I16` / `I16Array`, a 4-byte float as `F32` / `F32Array`. What the reader does normalize is the layout *inside* that width — every numeric variant is little-endian at its full precision and, for a float, in IEEE 754 layout, so a big-endian attribute reads correctly and would be rewritten in this crate's own layout. An integer width with no Rust integer of its own, 3 bytes say, comes back as `I64` / `U64`. An enumeration attribute decodes through its integer base type, the same view the numeric readers take of an enum dataset, so its codes arrive and its member names do not — this is how an h5py `np.bool_` attribute, stored as `enum[FALSE, TRUE]`, reads back as `0` or `1`. `repack` does not go through `AttrValue`, so an attribute keeps whatever encoding it had when copied.

`attr_datatypes()` reports what that normalization drops: the [`Datatype`](#the-datatype-model) of every attribute, including the ones `attrs()` omits for having no `AttrValue` at all. It is the attribute equivalent of `Dataset::datatype()`, and it is what identifies a `np.bool_` attribute as boolean rather than as an ordinary one-byte integer. Like `Dataset::datatype()` it resolves a committed (shared) datatype to the type it names, though neither reports that type's *name*, and neither channel exposes an attribute's rank; see the [groups and attributes guide](../guide/groups-attributes.md#reading-an-attributes-datatype).

| Variant | HDF5 encoding |
|---|---|
| `AttrValue::F32` / `F32Array` | 32-bit float scalar / array |
| `AttrValue::F64` / `F64Array` | 64-bit float scalar / array |
| `AttrValue::I8` / `I16` / `I32` / `I64` | Signed integer scalar, at that width |
| `AttrValue::I8Array` / `I16Array` / `I32Array` / `I64Array` | Signed integer array, at that width |
| `AttrValue::U8` / `U16` / `U32` / `U64` | Unsigned integer scalar, at that width |
| `AttrValue::U8Array` / `U16Array` / `U32Array` / `U64Array` | Unsigned integer array, at that width |
| `AttrValue::String` | UTF-8 null-padded string |
| `AttrValue::StringArray` | UTF-8 null-padded string array |
| `AttrValue::AsciiString` | Fixed-width ASCII string |
| `AttrValue::AsciiStringArray` | Array of fixed-width ASCII strings (null-padded to the longest element) |
| `AttrValue::VarLenAsciiArray` | MATLAB's variable-length ASCII string array — a VLEN sequence of one-byte strings (stored in a global heap collection) |
| `AttrValue::VarLenString` / `AttrValue::VarLenStringArray` | Variable-length UTF-8 string, scalar or array: `H5T_STRING` with `STRSIZE = H5T_VARIABLE` (stored in a global heap collection) |
| `AttrValue::VarLenAsciiString` / `AttrValue::VarLenAsciiStringArray` | The same, with `CSET = ASCII` |
| `AttrValue::StringSized` / `AsciiStringSized` (and their `*ArraySized` forms) | Fixed-width string of a width you declare, built by `AttrValue::string_sized` / `ascii_string_sized` and their array siblings |

```rust
use hdf5_pure::{FileBuilder, AttrValue};

let mut builder = FileBuilder::new();
builder.set_attr("version", AttrValue::I64(2));
builder.set_attr("unit", AttrValue::AsciiString("m/s".into()));

// A slot of a chosen width, holding a shorter value: `H5T_C_S1` plus
// `H5Tset_size(64)`. Writing it again at the same width leaves the slot the
// size it was, where `AsciiString` would resize it to the new content.
builder.set_attr("label", AttrValue::ascii_string_sized("ok", 64).unwrap());
```

The `*_sized` constructors return a `Result`: a value longer than the declared width is refused with `FormatError::FixedStringTooLong` rather than stored as a prefix, and a width of zero with `FormatError::ZeroFixedStringWidth`. `width` counts bytes, so a UTF-8 value with multi-byte characters takes more of it than its character count suggests.

`AsciiStringArray` and `VarLenAsciiArray` exist for MATLAB interoperability (the `MATLAB_fields` pattern); the `VarLenString` family writes the standard variable-length string datatype instead, which is the one h5py and the C library write and read. See the [groups and attributes guide](../guide/groups-attributes.md).

## The datatype model

For full control, the crate re-exports the low-level datatype types from `datatype`. These describe what is actually stored on disk and back every typed helper above.

### The `Datatype` enum

`Datatype` is the parsed HDF5 datatype. Its variants cover the HDF5 type classes:

| Variant | Class | Notes |
|---|---|---|
| `FixedPoint` | 0 | Integer; fields `size`, `byte_order`, `signed`, `bit_offset`, `bit_precision` |
| `FloatingPoint` | 1 | IEEE float; fields include `size`, `byte_order`, exponent/mantissa layout, `exponent_bias` |
| `Time` | 2 | `size`, `bit_precision` |
| `String` | 3 | Fixed-length string; `size`, `padding`, `charset` |
| `BitField` | 4 | `size`, `byte_order`, `bit_offset`, `bit_precision` |
| `Opaque` | 5 | `size`, `tag` |
| `Compound` | 6 | `size`, `members: Vec<CompoundMember>` |
| `Reference` | 7 | `size`, `ref_type: ReferenceType` |
| `Enumeration` | 8 | `size`, `base_type`, `members: Vec<EnumMember>` |
| `VariableLength` | 9 | `is_string`, `padding`, `charset`, `base_type` |
| `Array` | 10 | `base_type`, `dimensions: Vec<u32>` |

`Datatype::type_size()` returns the on-disk size in bytes.

A `FixedPoint` or `FloatingPoint` element **wider than 8 bytes** parses and reports its type like any other, but the typed *numeric* readers refuse to decode it rather than return part of a value (`read_raw`, `read_u8` and `read_i8` still hand back the bytes) — see [Limitations](limitations.md#numeric-element-width).

### Helper enums

| Enum | Variants |
|---|---|
| `DatatypeByteOrder` | `LittleEndian`, `BigEndian`, `Vax` |
| `StringPadding` | `NullTerminate`, `NullPad`, `SpacePad` |
| `CharacterSet` | `Ascii`, `Utf8` |
| `ReferenceType` | `Object`, `DatasetRegion` |

### `CompoundMember` and `EnumMember`

`CompoundMember` describes one field of a compound type:

| Field | Type | Meaning |
|---|---|---|
| `name` | `String` | Member name |
| `byte_offset` | `u64` | Offset within the compound |
| `datatype` | `Datatype` | Member datatype |

`EnumMember` describes one enumeration entry:

| Field | Type | Meaning |
|---|---|---|
| `name` | `String` | Member name |
| `value` | `Vec<u8>` | Raw value bytes (length = base type size) |

### Scalar constructors

The `make_*_type` functions return a canonical little-endian `Datatype` for each scalar:

`make_f32_type`, `make_f64_type`, `make_i8_type`, `make_i16_type`, `make_i32_type`, `make_i64_type`, `make_u8_type`, `make_u16_type`, `make_u32_type`, `make_u64_type`.

These are the building blocks for compound, enum, and array datatypes.

### `CompoundTypeBuilder`

Builds a compound datatype with fields laid out contiguously in insertion order (offsets are computed automatically). `new()` starts an empty builder; `field(name, datatype)` adds an arbitrary field, and the typed helpers `f64_field`, `f32_field`, `i8_field`, `i16_field`, `i32_field`, `i64_field`, `u8_field`, `u16_field`, `u32_field`, `u64_field` add a named scalar field. `build()` returns `Result<Datatype, FormatError>`, refusing a builder with no fields and one whose fields pack to zero bytes.

```rust
use hdf5_pure::CompoundTypeBuilder;

let dt = CompoundTypeBuilder::new()
    .i32_field("id")
    .f64_field("value")
    .build()
    .unwrap();
```

### `ExplicitCompoundTypeBuilder`

For an `H5Tinsert`-style layout with explicit offsets and a fixed total size, `CompoundTypeBuilder::with_size(size)` returns an `ExplicitCompoundTypeBuilder`. Each `field(name, byte_offset, datatype)` (and the typed `*_field(name, byte_offset)` variants) places a field at a chosen offset. `build()` returns `Result<Datatype, FormatError>` after validating that field names are unique, every field fits within `size`, and no two fields overlap.

```rust
use hdf5_pure::CompoundTypeBuilder;

let dt = CompoundTypeBuilder::with_size(16)
    .i32_field("id", 0)
    .f64_field("value", 8)
    .build()
    .unwrap();
```

### `EnumTypeBuilder`

Builds an enumeration datatype over an `i32` or `u8` base type. Start with `i32_based()` or `u8_based()`, add members with `value(name, i32)` or `u8_value(name, u8)`, then call `build()`.

```rust
use hdf5_pure::EnumTypeBuilder;

let dt = EnumTypeBuilder::i32_based()
    .value("Red", 0)
    .value("Green", 1)
    .value("Blue", 2)
    .build()
    .unwrap();
```

## `CompoundType` and `CompoundField`

`CompoundType` is the trait that `with_compound_values` and `read_compound` are generic over. Built-in implementations cover numeric tuples of one through twelve fields, with fields named `"0"`, `"1"`, and so on in the file; the Rust tuple's memory representation is never inspected. Custom structs can implement it by encoding each field explicitly and decoding fields according to the offsets in the supplied `Datatype`. `CompoundField` is the per-field counterpart used to encode and decode a single field. See the [compound types guide](../guide/compound-types.md).

## The `H5Element` trait

`H5Element` is the sealed trait that bounds the generic `with_data`, `read`, `read_array`, and `read_array_dyn` methods. It is implemented exactly for the scalar types: `f32`, `f64`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`. Because it is sealed, the set of implementors is fixed by the crate and cannot be extended downstream.

```rust
use hdf5_pure::{File, FileBuilder, H5Element, Error};

fn store<T: H5Element>(fb: &mut FileBuilder, name: &str, values: &[T]) {
    fb.create_dataset(name).with_data(values);
}

fn load<T: H5Element>(file: &File, name: &str) -> Result<Vec<T>, Error> {
    file.dataset(name)?.read::<T>()
}
```

## The `DType` classification

`DType` is a simplified, user-friendly classification returned by `Dataset::dtype()`. It maps a parsed `Datatype` onto a coarser enum and implements `Display`. Variants:

| Variant | Meaning |
|---|---|
| `F32` / `F64` | 4-/8-byte float |
| `I8` / `I16` / `I32` / `I64` | signed integers |
| `U8` / `U16` / `U32` / `U64` | unsigned integers |
| `String` | fixed-length string |
| `VariableLengthString` | variable-length string |
| `ObjectReference` | HDF5 object reference (8-byte address) |
| `Compound(Vec<(String, DType)>)` | compound with classified fields |
| `Enum(Vec<String>)` | enumeration with member names |
| `Array(Box<DType>, Vec<u32>)` | fixed-size array of a base type |
| `Other(Box<Datatype>)` | anything not classified above, carrying the type itself |

!!! tip
    Use `DType` for a quick human-readable summary; use `Dataset::datatype()` when you need exact field offsets, bit precision, or byte order. A type nested in a compound field or an array base has no such fallback, which is why `Other` carries the `Datatype` rather than a description of it.
