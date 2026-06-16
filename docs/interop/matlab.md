# MATLAB v7.3 (.mat) Files

A MATLAB v7.3 `.mat` file is an HDF5 file dressed in MATLAB conventions: a 512-byte userblock carrying the `MATLAB 7.3 MAT-file` signature, a `MATLAB_class` attribute on every dataset and group, column-major 2-D arrays, and UTF-16 strings. This page covers the high-level serde path that writes and reads `.mat` files from ordinary Rust structs, the supported type mapping, MATLAB cell arrays, and the lower-level conventions for hand-built files.

!!! tip "Runnable example"
    The [`matlab_fixtures`](https://github.com/stephenberry/hdf5-pure/blob/main/examples/matlab_fixtures.rs) example writes a directory of `.mat` v7.3 fixtures (scalars, vectors, matrices, strings, nested structs, complex data, cell arrays, and edge shapes) for verification in MATLAB and Octave. Run it with:

    ```bash
    cargo run --example matlab_fixtures --features serde
    ```

## Requires the `serde` feature

The high-level `.mat` API is gated on the `serde` feature, which is off by default. The `mat::Matrix`, `mat::Complex32`, `mat::Complex64`, and `mat::MatElement` items, along with `mat::to_file` / `mat::from_file`, are only available when it is enabled. See the [features reference](../reference/features.md) for the full list.

```toml
[dependencies]
hdf5-pure = { version = "0.14", features = ["serde"] }
serde = { version = "1", features = ["derive"] }
```

## Serializing a struct to `.mat`

Any type deriving `serde::Serialize` / `Deserialize` round-trips through `mat::to_file` and `mat::from_file`. The top-level value must be a struct with named fields (or a `HashMap<String, _>`); each field becomes a top-level MATLAB variable.

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

To work with bytes instead of the filesystem, use `mat::to_bytes` and `mat::from_bytes`, which take and return a `Vec<u8>` / `&[u8]`. Both are equally subject to the `serde` feature gate.

## Type mapping

The serializer maps Rust types to HDF5 datasets and the MATLAB classes MATLAB expects on read:

| Rust | HDF5 / MATLAB encoding |
|---|---|
| `f64`, `f32`, `i*`, `u*` | scalar dataset `[1,1]`, `MATLAB_class = "double"` / `"single"` / `"int*"` / `"uint*"` |
| `bool` | `uint8` scalar, `MATLAB_class = "logical"` |
| `String` / `&str` | `uint16` `[1, N]` UTF-16LE, `MATLAB_class = "char"` |
| `Vec<T>` of numeric `T` | `[1, N]` row vector |
| `Matrix<T>` or `Vec<Vec<T>>` of same length | column-major 2-D dataset, HDF5 shape `[cols, rows]` |
| `Complex32` / `Complex64` | compound `{real, imag}` dataset |
| nested struct | HDF5 group with `MATLAB_class = "struct"`, `MATLAB_fields` |
| `Option<T>` (struct field) | omitted if `None` |
| unit enum variant | UTF-16 char dataset holding the variant name |
| `Vec<Struct>` / `Vec<Option<T>>` / ragged `Vec<Vec<T>>` | cell array (`MATLAB_class = "cell"`, object references into `#refs#`); `None` slots become `struct([])` |

### Matrices and the column-major convention

Rust is row-major; MATLAB is column-major. The `mat::Matrix<T>` newtype carries the Rust-side `rows`/`cols` and a row-major `data` vector, and the serializer transposes to column-major byte order and stores the HDF5 dataset with shape `[cols, rows]` so MATLAB sees the intended `rows × cols` matrix. Build one with `Matrix::from_row_major(rows, cols, data)` (it panics if `data.len() != rows * cols`) or `Matrix::zeros(rows, cols)`; read the parts back with `rows()`, `cols()`, `data()`, and `into_data()`.

```rust
use hdf5_pure::mat::{self, Matrix};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct Frame { a: Matrix<f64> }

let v = Frame {
    a: Matrix::from_row_major(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
};
mat::to_file(&v, "matrix.mat").unwrap();
```

A bare `Vec<Vec<T>>` whose rows all share a length is also recognized as a 2-D matrix, but `Matrix` is the unambiguous API. The element type `T` is bounded by the sealed `mat::MatElement` trait, which is implemented for `f32`/`f64`, the 8/16/32/64-bit signed and unsigned integers, `bool`, `Complex32`, and `Complex64`. The trait is sealed because MAT v7.3 admits only this fixed set of numeric classes; you cannot implement it for other types.

### Complex numbers

`Complex32` and `Complex64` are compound `{real, imag}` newtypes constructed with `Complex64::new(re, im)` (or the `re` / `im` fields directly). A bare value becomes a compound scalar `[1, 1]`; a `Vec<Complex64>` becomes a compound dataset `[1, N]`. The on-disk layout is the same `{real, imag}` compound MATLAB uses for complex arrays. For a deeper treatment of HDF5 compound datasets see the [compound types guide](../guide/compound-types.md).

## Cell arrays

A sequence whose elements do not unify into a single numeric matrix lowers to a MATLAB cell array rather than erroring. Each element is interned under the conventional `#refs#` group, and the parent dataset stores HDF5 object references with `MATLAB_class = "cell"`. This covers `Vec<Struct>`, `Vec<Option<T>>` with interspersed `None`, nested cells of cells, and ragged `Vec<Vec<T>>`. An `Option::None` slot inside a sequence becomes `struct([])` so every cell slot has a defined MATLAB type.

```rust
use hdf5_pure::mat;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct Point { x: f64, y: f64 }

#[derive(Serialize, Deserialize)]
struct Capture {
    /// 3x1 cell array of struct.
    path: Vec<Point>,
    /// 3x1 cell array; the `None` slot becomes `struct([])`.
    optionals: Vec<Option<Point>>,
    /// Outer 2x1 cell of cells; rows-of-variable-length-records shape.
    grid: Vec<Vec<Option<Point>>>,
    /// Ragged numerics also fall back to cell rather than erroring.
    ragged: Vec<Vec<f64>>,
}
```

In MATLAB this loads as `iscell(path) == true`, with elements addressed as `path{1}.x`, and so on. Empty `None` slots load as `struct([])` (`isempty(fieldnames(...))`).

!!! note "Reader compatibility"
    Cell arrays load correctly in MATLAB, libmatio (the reference C library), Julia's `MAT.jl`, and Python via `pymatreader` / `hdf5storage`. GNU Octave 11's `load` does not yet follow object references for v7.3 cells (it warns "unknown datatype"); load such files with one of the other tools instead.

## Not supported

This release does not encode non-unit enum variants, MATLAB `classdef` objects, or `datetime` / `categorical` types. Unit enum variants are supported and serialize to a UTF-16 char dataset holding the variant name.

## Hand-built files (low-level conventions)

If you are not using serde, you can apply the MATLAB conventions yourself on top of `FileBuilder`. Two pieces matter: the userblock header and the `MATLAB_class` / `MATLAB_fields` attributes.

### Userblock header

MATLAB expects a 512-byte userblock beginning with the `MATLAB 7.3 MAT-file` signature. Reserve the block with `with_userblock(512)` and write the header bytes into the leading region after finishing:

```rust
use hdf5_pure::FileBuilder;

let mut builder = FileBuilder::new();
builder.with_userblock(512);
builder.create_dataset("data").with_f64_data(&[1.0]);

let mut bytes = builder.finish().unwrap();
// Write MATLAB header into userblock
bytes[126] = b'I';
bytes[127] = b'M';
```

### Struct pattern

A MATLAB struct is an HDF5 group carrying `MATLAB_class = "struct"` and a `MATLAB_fields` list naming its fields, with each field a child dataset that carries its own `MATLAB_class`. Use `AttrValue::AsciiString` for the fixed-length ASCII class names and `AttrValue::VarLenAsciiArray` for the variable-length field-name array:

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

See the [groups and attributes guide](../guide/groups-attributes.md) for more on the `AttrValue` variants used here.
