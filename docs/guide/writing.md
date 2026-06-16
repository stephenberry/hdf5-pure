# Writing Files

This page covers building HDF5 files with `FileBuilder`: creating datasets from typed Rust slices, attaching attributes, and serializing the result either to memory or to disk. It is the foundation for everything else you write to a file.

!!! tip "Runnable example"
    A complete, self-checking version of this workflow lives in [`examples/quickstart.rs`](https://github.com/stephenberry/hdf5-pure/blob/main/examples/quickstart.rs). Run it with:

    ```bash
    cargo run --example quickstart
    ```

## The `FileBuilder` workflow

A file is assembled with `FileBuilder`. You start one with `FileBuilder::new()`, add datasets and groups, attach attributes, and finally serialize. `create_dataset(name)` returns a `DatasetBuilder` whose typed setters supply both the data and (by default) the shape:

```rust
use hdf5_pure::{FileBuilder, AttrValue};

let mut builder = FileBuilder::new();

builder
    .create_dataset("temperature")
    .with_f64_data(&[22.5, 23.1, 21.8])
    .set_attr("unit", AttrValue::AsciiString("degC".into()));

builder.set_attr("version", AttrValue::I64(2));

builder.write("output.h5").unwrap();
```

`create_dataset` returns a `&mut DatasetBuilder`, so the typed setters chain. The builder owns the dataset until the file is serialized; there is no separate "commit" step per dataset.

## Typed data setters and shape

Each scalar type has a dedicated setter. Calling one sets both the element datatype and the data. The shape defaults to `[len]`, the one-dimensional shape matching the slice length, so `with_shape` is optional for flat 1-D data and only needed when you want a different rank:

```rust
use hdf5_pure::FileBuilder;

let mut builder = FileBuilder::new();

// 1-D: shape defaults to [6].
builder.create_dataset("flat").with_f64_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

// 2-D: same six values laid out row-major as [2, 3].
builder
    .create_dataset("grid")
    .with_f64_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    .with_shape(&[2, 3]);
```

The element type of a dataset comes from the setter you call:

| Method | HDF5 type |
|---|---|
| `with_f64_data` | IEEE 64-bit float |
| `with_f32_data` | IEEE 32-bit float |
| `with_i8_data` / `with_i16_data` / `with_i32_data` / `with_i64_data` | Signed integers (8/16/32/64-bit) |
| `with_u8_data` / `with_u16_data` / `with_u32_data` / `with_u64_data` | Unsigned integers (8/16/32/64-bit) |

This is the common subset. Compound, enumeration, array, complex, and object-reference datatypes have their own setters; see [compound and complex types](compound-types.md) for those.

!!! note
    Data is stored row-major (C order), which is what HDF5 uses on disk. When you provide a multi-dimensional `with_shape`, the flat slice is interpreted in row-major order.

## Generic writing over the element type

The typed setters have a generic counterpart, `with_data(&[T])`, bounded by the sealed `H5Element` trait. It infers the datatype from `T`, letting you write code that is generic over any supported scalar:

```rust
use hdf5_pure::{FileBuilder, H5Element};

fn store<T: H5Element>(fb: &mut FileBuilder, name: &str, values: &[T]) {
    fb.create_dataset(name).with_data(values);
}

let mut fb = FileBuilder::new();
store(&mut fb, "counts", &[1u32, 2, 3]);
```

See [Generic I/O](generic-io.md) for the full `with_data` / `read::<T>()` round trip and the list of types implementing `H5Element`.

## Attributes

Attributes attach metadata to a dataset or to a group. On a dataset, `set_attr` is part of the builder chain; on the file root, `FileBuilder::set_attr` attaches an attribute to the root group:

```rust
use hdf5_pure::{FileBuilder, AttrValue};

let mut builder = FileBuilder::new();

builder
    .create_dataset("temperature")
    .with_f64_data(&[22.5, 23.1, 21.8])
    .set_attr("unit", AttrValue::AsciiString("degC".into()));

// Root-group attribute.
builder.set_attr("version", AttrValue::I64(2));
```

Attribute values are `AttrValue` variants (`F64`, `I64`, `AsciiString`, and others). The full set of variants and their HDF5 encodings is covered under [groups and attributes](groups-attributes.md).

## Groups

`create_group(name)` returns a `GroupBuilder` you populate the same way as the root, then hand back to the file with `add_group`:

```rust
use hdf5_pure::{FileBuilder, AttrValue};

let mut builder = FileBuilder::new();

let mut grp = builder.create_group("sensors");
grp.create_dataset("pressure").with_f32_data(&[101.3, 101.5]);
grp.set_attr("location", AttrValue::AsciiString("lab_a".into()));
builder.add_group(grp.finish());
```

`GroupBuilder::finish()` produces a `FinishedGroup`, which `add_group` inserts into the file. Nested hierarchies and group attributes are covered in detail on the [groups and attributes](groups-attributes.md) page.

## Empty and zero-dimension datasets

To create a dataset without supplying data, set the datatype and shape explicitly with `with_dtype` and `with_shape`. This is how you write an empty (zero-length) or zero-dimension (scalar-shaped) dataset:

```rust
use hdf5_pure::{FileBuilder, make_f64_type};

let mut builder = FileBuilder::new();

builder
    .create_dataset("placeholder")
    .with_dtype(make_f64_type())
    .with_shape(&[0]);
```

`with_dtype` takes a `Datatype`, which the crate's `make_*_type` constructors produce (for example `make_f64_type()`).

## Serializing: `finish()` vs `write(path)`

When the file is fully assembled, choose how to materialize it:

| Method | Returns | Use when |
|---|---|---|
| `finish()` | `Result<Vec<u8>, Error>` | You want the file image in memory (WASM-friendly, no filesystem) |
| `write(path)` | `Result<(), Error>` | You want the file written to disk |

```rust
use hdf5_pure::FileBuilder;

let mut builder = FileBuilder::new();
builder.create_dataset("x").with_f64_data(&[1.0, 2.0]);

// In memory: no filesystem touched, just the serialized bytes.
let bytes: Vec<u8> = builder.finish().unwrap();

// Or straight to disk.
// builder.write("output.h5").unwrap();
```

The in-memory `Vec<u8>` is exactly the bytes that `write` would put on disk, so it round-trips through `File::from_bytes`. This is what makes writing usable in environments without a filesystem.

!!! note
    `FileBuilder` is part of the high-level API gated behind the `std` feature (enabled by default), so both `finish` and `write` require `std`. The difference is the filesystem: `finish` returns the file image in memory and never touches disk, while `write` writes those same bytes to a path.

## Next steps

- [Reading files](reading.md) to load what you wrote back, including from the in-memory bytes.
- [Compression](compression.md) for chunking, deflate, shuffle, and scale-offset filters.
- [Portability](../interop/portability.md) for how these files interoperate with the reference HDF5 C library, h5py, and MATLAB.
