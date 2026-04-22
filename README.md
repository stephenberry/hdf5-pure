# hdf5-pure

Pure-Rust HDF5 reader/writer. No C dependencies, no build scripts, WASM-compatible.

## Features

- **Write** HDF5 files with datasets, groups, attributes, and nested hierarchies
- **Read** HDF5 files (v0/v1/v2/v3 superblocks, v1/v2 object headers, contiguous/chunked/compact storage)
- **No C dependencies** — compiles to `wasm32-unknown-unknown` with `--no-default-features`
- **MATLAB v7.3 compatible** — userblock support, fixed-length ASCII attributes, variable-length string arrays, object references
- Deflate and shuffle compression
- Compound types, enumerations, array types
- Complex number datasets (as compound `{real, imag}`)

## Quick start

### Writing

```rust
use hdf5_pure::{FileBuilder, AttrValue};

let mut builder = FileBuilder::new();

// Datasets
builder.create_dataset("temperature")
    .with_f64_data(&[22.5, 23.1, 21.8])
    .with_shape(&[3]);

// Groups with nested datasets
let mut grp = builder.create_group("sensors");
grp.create_dataset("pressure").with_f32_data(&[101.3, 101.5]);
grp.set_attr("location", AttrValue::AsciiString("lab_a".into()));
builder.add_group(grp.finish());

// Attributes on the root group
builder.set_attr("version", AttrValue::I64(2));

builder.write("output.h5").unwrap();
```

### Reading

```rust
use hdf5_pure::File;

let file = File::open("output.h5").unwrap();
let ds = file.dataset("temperature").unwrap();

println!("shape: {:?}", ds.shape().unwrap());    // [3]
println!("data:  {:?}", ds.read_f64().unwrap());  // [22.5, 23.1, 21.8]

let attrs = file.root().attrs().unwrap();
println!("version: {:?}", attrs.get("version"));  // Some(I64(2))
```

### In-memory (WASM)

```rust
use hdf5_pure::FileBuilder;

let mut builder = FileBuilder::new();
builder.create_dataset("x").with_f64_data(&[1.0, 2.0]);

let bytes: Vec<u8> = builder.finish().unwrap(); // no filesystem needed
```

## Supported data types

### Datasets

| Method | HDF5 type |
|---|---|
| `with_f64_data` | IEEE 64-bit float |
| `with_f32_data` | IEEE 32-bit float |
| `with_i8_data` / `with_i16_data` / `with_i32_data` / `with_i64_data` | Signed integers |
| `with_u8_data` / `with_u16_data` / `with_u32_data` / `with_u64_data` | Unsigned integers |
| `with_complex32_data` | Compound `{real: f32, imag: f32}` |
| `with_complex64_data` | Compound `{real: f64, imag: f64}` |
| `with_compound_data` | Arbitrary compound types |
| `with_enum_i32_data` / `with_enum_u8_data` | Enumeration types |
| `with_array_data` | Fixed-size array types |
| `with_path_references` | Object references (resolved by path) |
| `with_dtype` + `with_shape` | Empty/zero-dimension datasets |

### Attributes

| Variant | HDF5 encoding |
|---|---|
| `AttrValue::F64` / `F64Array` | 64-bit float scalar/array |
| `AttrValue::I32` / `I64` / `I64Array` | Signed integer scalar/array |
| `AttrValue::U32` / `U64` | Unsigned integer scalar |
| `AttrValue::String` / `StringArray` | UTF-8 null-padded string |
| `AttrValue::AsciiString` | Fixed-length ASCII string |
| `AttrValue::VarLenAsciiArray` | Variable-length ASCII string array (global heap) |

## Compression

```rust
// Deflate (zlib)
builder.create_dataset("compressed")
    .with_f64_data(&data)
    .with_chunks(&[100])
    .with_deflate(6);

// Shuffle + deflate
builder.create_dataset("shuffled")
    .with_f64_data(&data)
    .with_chunks(&[100])
    .with_shuffle()
    .with_deflate(6);
```

## Userblock (MATLAB v7.3)

```rust
let mut builder = FileBuilder::new();
builder.with_userblock(512);
builder.create_dataset("data").with_f64_data(&[1.0]);

let mut bytes = builder.finish().unwrap();
// Write MATLAB header into userblock
bytes[126] = b'I';
bytes[127] = b'M';
```

## MATLAB struct pattern

```rust
use hdf5_pure::{FileBuilder, AttrValue};

let mut builder = FileBuilder::new();
let mut grp = builder.create_group("my_struct");

let mut fields = Vec::new();
for (name, data) in [("x", vec![1.0, 2.0]), ("y", vec![3.0, 4.0])] {
    fields.push(name.to_string());
    grp.create_dataset(name).with_f64_data(&data)
        .set_attr("MATLAB_class", AttrValue::AsciiString("double".into()));
}

grp.set_attr("MATLAB_class", AttrValue::AsciiString("struct".into()));
grp.set_attr("MATLAB_fields", AttrValue::VarLenAsciiArray(fields));
builder.add_group(grp.finish());
```

## MATLAB v7.3 `.mat` via serde

With the `serde` feature, Rust structs can be serialized directly to `.mat`
v7.3 files and back:

```rust
use hdf5_pure::mat::{self, Complex64, Matrix};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct Experiment {
    name: String,
    trial: u32,
    samples: Vec<f64>,
    data: Matrix<f64>,
    waveform: Vec<Complex64>,
    config: Config,
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct Config { threshold: f64, tag: String }

let e = Experiment {
    name: "run1".into(), trial: 3,
    samples: vec![1.0, 2.0, 3.0],
    data: Matrix::from_row_major(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    waveform: vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)],
    config: Config { threshold: 0.5, tag: "prod".into() },
};

mat::to_file(&e, "experiment.mat").unwrap();
let back: Experiment = mat::from_file("experiment.mat").unwrap();
assert_eq!(back, e);
```

The top-level value must be a struct (or `HashMap<String, _>`); each field
becomes a MATLAB variable. Mapping:

| Rust | HDF5 / MATLAB encoding |
|---|---|
| `f64`, `f32`, `i*`, `u*` | scalar dataset `[1,1]`, `MATLAB_class = "double"` / `"single"` / `"int*"` / `"uint*"` |
| `bool` | `uint8` scalar, `MATLAB_class = "logical"` |
| `String` / `&str` | `uint16` `[1, N]` UTF-16LE, `MATLAB_class = "char"` |
| `Vec<T>` of numeric `T` | `[1, N]` row vector |
| `Matrix<T>` or `Vec<Vec<T>>` | column-major 2-D dataset, HDF5 shape `[cols, rows]` |
| `Complex32` / `Complex64` | compound `{real, imag}` dataset |
| nested struct | HDF5 group with `MATLAB_class = "struct"`, `MATLAB_fields` |
| `Option<T>` | omitted if `None` |
| unit enum variant | UTF-16 char dataset holding the variant name |

Not supported in this release: cell arrays, non-unit enum variants, MATLAB
objects (`classdef`), datetime / categorical types.

## Cargo features

| Feature | Default | Description |
|---|---|---|
| `std` | yes | File I/O, high-level reader API |
| `checksum` | yes | Jenkins hash for v2+ object headers |
| `deflate` | yes | Deflate compression (pure Rust backend) |
| `serde` | no | Serialize/deserialize MATLAB v7.3 `.mat` files via serde |
| `fast-checksum` | no | Hardware-accelerated CRC32 via `crc32fast` |
| `fast-deflate` | no | zlib-ng backend for deflate via `flate2/zlib-ng` |
| `mmap` | no | Memory-mapped file reading via `memmap2` |
| `parallel` | no | Parallel chunk processing via `rayon` |
| `provenance` | no | SHA-256 data provenance tracking |

For WASM, disable default features:

```toml
[dependencies]
hdf5-pure = { version = "0.1", default-features = false, features = ["checksum"] }
```

## Acknowledgements

The HDF5 format parsing and low-level I/O modules are derived from [rustyhdf5](https://github.com/rustystack/rustyhdf5) by the RustyStack project (MIT licensed).

## License

MIT
