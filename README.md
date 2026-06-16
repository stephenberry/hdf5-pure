# hdf5-pure

Pure-Rust HDF5 reader, writer, and in-place editor. No C dependencies, no build scripts, WASM-compatible.

**📖 [Documentation site](https://stephenberry.github.io/hdf5-pure/)** · [API reference (docs.rs)](https://docs.rs/hdf5-pure) · [Examples](examples) · [Changelog](CHANGELOG.md)

## Features

- **Write** HDF5 files with datasets, groups, attributes, and nested hierarchies
- **Read** HDF5 files (v0/v1/v2/v3 superblocks, v1/v2 object headers, contiguous/chunked/compact storage)
- **Edit in place** — add, delete (`H5Ldelete`), and copy (`H5Ocopy`) datasets and groups in an existing file without reading it all in and rewriting it; the cost is proportional to what changes, not the file size, and an exclusive OS advisory lock guards against concurrent writers
- **SWMR** (single-writer / multiple-reader) append and refreshing read for 1-D unlimited datasets, interoperable with the reference C library and h5py
- **No C dependencies** — pure Rust, so it compiles to `wasm32-unknown-unknown` and to bare-metal `no_std` (with `alloc`)
- **MATLAB v7.3 compatible** — userblock support, fixed-length ASCII attributes, variable-length string arrays, object references
- Deflate, shuffle, and scale-offset (lossless integer / lossy float) compression
- Compound types, enumerations, array types
- Complex number datasets (as compound `{real, imag}`)

## Examples

Runnable, self-checking examples live in [`examples/`](examples). Run any with `cargo run --example <name>`:

| Example | What it shows |
|---|---|
| `quickstart` | Build a file in memory and read it back |
| `generic_io` | Read/write generically over the element type (`with_data` / `read::<T>`) |
| `groups_and_attributes` | Nested groups and attributes of several types |
| `compression` | Deflate, shuffle, and scale-offset filters |
| `compound_types` | Compound (struct-like) records and complex numbers |
| `ndarray_io` | N-dimensional array I/O (needs `--features ndarray`) |
| `edit_in_place` | Add, copy, and delete objects with `EditSession` |
| `repack` | Shrink a file and drop objects with `repack` |
| `swmr` | Single-writer / multiple-reader append and refresh |
| `file_space` | File-space strategy and persistent free-space reuse across sessions |

The `matlab_fixtures` example (run with `--features serde`) writes `.mat` v7.3 files for verification in MATLAB/Octave.

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

### Generic over the element type

The typed `with_f64_data` / `read_f64` family has a generic counterpart so you
can write code that works for any supported scalar: `with_data(&[T])` to write
and `read::<T>()` to read, bounded by the sealed `H5Element` trait (implemented
for `f32`/`f64` and the 8/16/32/64-bit signed and unsigned integers).

```rust
use hdf5_pure::{File, FileBuilder, H5Element, Error};

fn store<T: H5Element>(fb: &mut FileBuilder, name: &str, values: &[T]) {
    fb.create_dataset(name).with_data(values);
}

fn load<T: H5Element>(file: &File, name: &str) -> Result<Vec<T>, Error> {
    file.dataset(name)?.read::<T>()
}

let mut fb = FileBuilder::new();
store(&mut fb, "counts", &[1u32, 2, 3]);
let file = File::from_bytes(fb.finish().unwrap()).unwrap();

let counts: Vec<u32> = load(&file, "counts").unwrap();  // [1, 2, 3]
```

`read::<T>()` requests delivery as `T` (coercing like `read_f64`); it is not an
assertion about the stored datatype, so pick `T` to match the stored type for a
lossless read. For N-dimensional arrays see the `ndarray` feature below.

### Editing in place

`EditSession` opens an existing file and adds, deletes, or copies objects, or edits compact group attributes without reading it all in and rewriting it. New data and the rebuilt object headers are appended at the end of the file and the superblock is repointed last, so the cost is proportional to what changes and a failed commit leaves the file valid.

```rust,no_run
use hdf5_pure::{AttrValue, EditSession};

let mut session = EditSession::open("output.h5").unwrap();

session.create_group("run2");
session.set_group_attr("run2", "kind", AttrValue::AsciiString("trial".into()));
session.create_dataset("run2/signal").with_f64_data(&[1.0, 2.0, 3.0]);
session.copy("temperature", "temperature_backup");  // H5Ocopy
session.delete("sensors/pressure");                 // H5Ldelete

session.commit().unwrap();  // apply everything in place
```

Contiguous, unfiltered datasets and compact-link groups are supported, and the editor edits files across every on-disk format the reference C library and h5py produce — version 0/1/2/3 superblocks, single- and multi-chunk object headers (a multi-chunk header is collapsed into one chunk on rewrite, and a version 0/1 symbol-table group on the edited path is converted to the latest compact-link format). It refuses, rather than silently degrade the file, anything it cannot reproduce faithfully — a userblock (non-zero base address), chunked/compressed additions, dense-storage headers on the edited path, or copying an existing version-1 object. Within a session the space a deletion frees is reused for later writes and the file is truncated when the freed bytes reach the end, so add/delete churn stays bounded instead of only ever growing; for guaranteed compaction across a reopen, see `repack` below.

`EditSession::open` takes an exclusive OS advisory lock (the analogue of `H5Pset_file_locking`), so a second editor or any concurrent writer gets `Error::FileLocked` rather than racing on the file. The kernel releases the lock on any process exit, including a crash, so a crashed editor never leaves a stale lock behind. Override the policy with `EditSession::open_with_locking` and the `FileLocking` enum, or set `HDF5_USE_FILE_LOCKING=FALSE` for filesystems (such as some network mounts) where locking is unavailable. `SwmrWriter` and the readers intentionally take no lock: SWMR is single-writer by contract and built for concurrent reads.

### Reclaiming space (`repack`)

Deleting an object inside an `EditSession` reuses the freed space within the session, but a single delete-then-close cannot shrink a file whose freed region is not at the very end — the same reason the HDF5 C library ships `h5repack`. `repack` is the guaranteed-shrink answer: it reads every surviving object and rewrites the whole file compact, optionally dropping objects.

```rust,no_run
use hdf5_pure::{repack, RepackOptions};

// Drop a dataset and a whole group subtree, then write a fresh, compact file.
let options = RepackOptions::new()
    .drop_path("scratch")
    .drop_path("runs/aborted");
repack("input.h5", "compact.h5", &options).unwrap();
```

`repack` never silently degrades data: every surviving object is reproduced byte-for-byte — datatype, shape, chunking, supported filters, raw data, and attributes — or the whole operation fails with `Error::RepackUnsupported` naming the object, leaving no output file. It reproduces fixed-point, floating-point, string, bit-field, opaque, compound, enumeration, and array datatypes, contiguous or chunked, filtered with deflate, shuffle, fletcher32, and/or lossless integer scale-offset. Anything it cannot reproduce exactly — variable-length, time, and reference datatypes, virtual layouts, lossy filters (float D-scale scale-offset, ZFP, SZIP), or an attribute the reader cannot decode — it refuses by name rather than write a file that quietly differs.

### File-space strategy

Mirroring `H5Pset_file_space_strategy` and `H5Pset_file_space_page_size`, a written file can record how it manages free space. The strategy is stored in a superblock-extension message, so the reference HDF5 C library and a later reopen observe it; `File::file_space_strategy()` reads it back.

```rust,no_run
use hdf5_pure::{File, FileBuilder, FileSpaceStrategy};

let mut b = FileBuilder::new();
b.create_dataset("d").with_i32_data(&[1, 2, 3]);
b.with_file_space_strategy(FileSpaceStrategy::Page, false, 1)  // strategy, persist, threshold
    .with_file_space_page_size(8192);
b.write("out.h5").unwrap();

assert_eq!(File::open("out.h5").unwrap().file_space_strategy(), Some(FileSpaceStrategy::Page));
```

Passing `persist = true` persists free space across reopen: a file created this way records the regions an `EditSession` frees in on-disk free-space-manager blocks (`FSHD`/`FSSE`), so later sessions — this crate's and the reference C library's — recover and reuse them. `File::persisted_free_space()` returns the tracked free regions, and `EditSession::open` seeds its free list from them so reuse spans sessions rather than just the open session.

### Streaming large files

`File::open(path)` reads the whole file into memory. To read a file too large to buffer (for example a multi-gigabyte file produced on a 32-bit host, where it exceeds the address space), open it with `File::open_streaming(path)` instead. It fetches metadata and dataset chunks from the file on demand rather than buffering it whole, so it never holds the entire file in memory at once: peak memory tracks the data you actually read (one dataset, decompressed, with its chunks fetched on demand) plus the metadata being parsed, not the whole file.

```rust
use hdf5_pure::File;

let file = File::open_streaming("huge.h5").unwrap();
let ds = file.dataset("signal").unwrap();
let values = ds.read_f64().unwrap();  // only this dataset's chunks are read
```

The reading API is identical to `File::open`; only the backing store differs. Dataset reads are fully supported: contiguous, compact, and every chunk-index layout (B-tree v1, fixed array, and extensible array). Two limits apply to the streaming backend that in-memory reading does not have: only latest-format (v2) groups resolve along a path (a v1 symbol-table group is rejected), and reading attributes is not yet supported. `open_streaming` requires the `std` filesystem.

Use `File::open_streaming_with_options` to bound retained metadata and dataset chunk cache memory. `MetadataCacheConfig` mirrors the memory-budget role of `H5Pset_mdc_config`; `ChunkCacheConfig` mirrors the raw-data chunk-cache settings from `H5Pset_cache`, controlling decompressed chunk data and whether parsed chunk indexes are retained between repeated reads of the same dataset.

```rust
use hdf5_pure::{ChunkCacheConfig, File, FileAccessOptions, MetadataCacheConfig};

let options = FileAccessOptions::new()
    .with_metadata_cache(MetadataCacheConfig::new(8 * 1024 * 1024).with_max_entry_bytes(64 * 1024))
    .with_chunk_cache(ChunkCacheConfig::from_h5p_cache(521, 256 * 1024));
let file = File::open_streaming_with_options("huge.h5", options).unwrap();
```

To override the chunk cache for a single dataset (HDF5's per-dataset access property list, `H5Pset_chunk_cache`), open it with `dataset_with_options`. The override replaces the file-wide default for that one dataset; other datasets keep the default. `Dataset::chunk_cache_config()` reports the effective setting (`H5Pget_chunk_cache`).

```rust
use hdf5_pure::{ChunkCacheConfig, DatasetAccessOptions, File};

let file = File::open("data.h5").unwrap();
// This dataset is read once front-to-back: skip caching its decompressed chunks.
let dapl = DatasetAccessOptions::new().with_chunk_cache(ChunkCacheConfig::disabled());
let ds = file.dataset_with_options("scan", dapl).unwrap();
let values = ds.read_f64().unwrap();
```

To confirm the cache is behaving as configured, `Dataset::chunk_cache_stats()` reports a read-only snapshot (index loaded, retained chunks, retained bytes) after a read.

```rust
use hdf5_pure::File;

let file = File::open("data.h5").unwrap();
let ds = file.dataset("signal").unwrap();
let _ = ds.read_f64().unwrap();
let stats = ds.chunk_cache_stats();
assert!(stats.cached_chunks() > 0); // chunks were retained for reuse
```

### In-memory (WASM)

```rust
use hdf5_pure::FileBuilder;

let mut builder = FileBuilder::new();
builder.create_dataset("x").with_f64_data(&[1.0, 2.0]);

let bytes: Vec<u8> = builder.finish().unwrap(); // no filesystem needed
```

### N-dimensional arrays (`ndarray` feature)

Enable the `ndarray` feature to write and read datasets of any rank as
[`ndarray`](https://docs.rs/ndarray) arrays. Shape and datatype are taken from
the array, and data is stored row-major (C order), matching HDF5:

```rust
use hdf5_pure::{File, FileBuilder};
use ndarray::{array, Array2};

let a: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

let mut fb = FileBuilder::new();
fb.create_dataset("m").with_ndarray(&a);          // shape [2, 3], f64
let bytes = fb.finish().unwrap();

let file = File::from_bytes(bytes).unwrap();
let back: Array2<f64> = file.dataset("m").unwrap().read_array().unwrap();
assert_eq!(a, back);

// When the rank is only known at runtime:
let dynamic = file.dataset("m").unwrap().read_array_dyn::<f64>().unwrap(); // ArrayD<f64>
```

`with_ndarray` accepts owned arrays or views; non-standard layouts (transposed,
Fortran-order, or strided) are repacked to row-major on write, and chunking and
compression chain as usual (`.with_ndarray(&a).with_chunks(&[64, 64]).with_deflate(6)`).

## SWMR (single writer, multiple readers)

A single process can append to an unlimited dataset in place while other processes read it concurrently. The writer appends chunks and flushes in dependency order so readers only ever observe a consistent prefix; readers re-read to pick up new data. This interoperates with the reference HDF5 C library and h5py in both directions.

The dataset must have one unlimited dimension and be chunked (it is indexed by an Extensible Array, which the latest format selects automatically). Create it the usual way:

```rust
use hdf5_pure::FileBuilder;

let mut builder = FileBuilder::new();
builder.create_dataset("log")
    .with_i32_data(&[0, 1, 2])   // initial rows
    .with_shape(&[3])
    .with_maxshape(&[u64::MAX])  // one unlimited dimension
    .with_chunks(&[1]);
builder.write("stream.h5").unwrap();
```

Append in place (each call flushes durably; the file stays valid for concurrent readers throughout):

```rust
use hdf5_pure::SwmrWriter;

let mut writer = SwmrWriter::open("stream.h5").unwrap();
writer.append_i32("log", &[3, 4, 5]).unwrap();
writer.append_i32("log", &[6, 7]).unwrap();
writer.close().unwrap(); // clears the SWMR flag; or just drop the writer
```

Follow a growing file from another process (or the reference C library / h5py writing in SWMR mode):

```rust
use hdf5_pure::File;

let mut file = File::open_swmr("stream.h5").unwrap();
let n = file.dataset("log").unwrap().shape().unwrap()[0];
// ... later, after the writer appends ...
file.refresh().unwrap();                 // re-read appended data
let ds = file.dataset("log").unwrap();
println!("now {} rows", ds.shape().unwrap()[0]);
```

Supported subset: one unlimited dimension, chunked, unfiltered (no compression on the appended dataset), chunk-aligned appends, no userblock. Growth is unbounded. SWMR requires the `std` filesystem (not the in-memory/WASM path). If a writer process exits without `close()`, the file is left marked as having an active SWMR writer; recover it with `SwmrWriter::clear_swmr_flag(path)` (the equivalent of `h5clear`).

## Supported data types

### Datasets

| Method | HDF5 type |
|---|---|
| `with_data` (generic, any scalar below) | Inferred from the element type |
| `with_f64_data` | IEEE 64-bit float |
| `with_f32_data` | IEEE 32-bit float |
| `with_i8_data` / `with_i16_data` / `with_i32_data` / `with_i64_data` | Signed integers |
| `with_u8_data` / `with_u16_data` / `with_u32_data` / `with_u64_data` | Unsigned integers |
| `with_complex32_data` | Compound `{real: f32, imag: f32}` |
| `with_complex64_data` | Compound `{real: f64, imag: f64}` |
| `with_compound_data` | Arbitrary compound types |
| `with_compound_values` | Safely encoded numeric tuples |
| `with_enum_i32_data` / `with_enum_u8_data` | Enumeration types |
| `with_array_data` | Fixed-size array types |
| `with_path_references` | Object references (resolved by path) |
| `with_dtype` + `with_shape` | Empty/zero-dimension datasets |

### Compound types

Numeric tuples are encoded field by field, without relying on Rust tuple layout:

```rust
use hdf5_pure::{File, FileBuilder};

let values = [(1i8, 20u64, 3.5f32), (2, 30, 4.5)];
let mut builder = FileBuilder::new();
builder.create_dataset("records")
    .with_compound_values(&values)
    .unwrap();
let bytes = builder.finish().unwrap();

let file = File::from_bytes(bytes).unwrap();
let records = file.dataset("records").unwrap()
    .read_compound::<(i8, u64, f32)>()
    .unwrap();
assert_eq!(records, values);
```

Use `CompoundTypeBuilder::with_size` for an `H5Tinsert`-style layout with explicit offsets and padding. `Dataset::datatype` exposes the exact field offsets from existing files, while `Dataset::read_raw` returns their complete unfiltered record bytes.

### Variable-length string reads

`Dataset::read_string` reads both fixed-length and variable-length HDF5 string datasets. VL reads can be bounded before payload allocation, or consumed one string at a time:

```rust
use hdf5_pure::{File, VlenStringReadOptions};

let file = File::open_streaming("strings.h5").unwrap();
let dataset = file.dataset("names").unwrap();

let payload_bytes = dataset.vlen_string_payload_size().unwrap();
let options = VlenStringReadOptions::new()
    .with_max_elements(100_000)
    .with_max_payload_bytes(64 * 1024 * 1024);

dataset.visit_vlen_strings(options, |value| {
    println!("{value}");
}).unwrap();

println!("{payload_bytes} bytes of string payload");
```

The payload limit covers bytes referenced by VL elements and excludes Rust container metadata. Shared global heap collections are indexed once per read, and only the referenced object payloads are fetched. This works with both `File::open` and `File::open_streaming`.

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

### Scale-offset (HDF5 filter id 6)

Scale-offset stores each chunk's values as offsets from the chunk minimum,
packed into the fewest bits the chunk's range needs. It is a built-in HDF5
filter, so files we write are readable by the reference C library, h5py, and
MATLAB, and files those tools produce are readable by us.

```rust
use hdf5_pure::ScaleOffset;

// Integer mode is lossless. `0` lets the encoder pick the bit width per chunk.
builder.create_dataset("counts")
    .with_i32_data(&counts)
    .with_chunks(&[1000])
    .with_scale_offset(ScaleOffset::Integer(0));

// Float D-scale is lossy: values are rounded to N decimal digits before packing.
builder.create_dataset("readings")
    .with_f64_data(&readings)
    .with_chunks(&[1000])
    .with_scale_offset(ScaleOffset::FloatDScale(3))  // keep 3 decimal digits
    .with_deflate(6);                                // may be followed by deflate
```

| Mode | Datatype | Loss |
|---|---|---|
| `ScaleOffset::Integer(minbits)` | signed/unsigned integers | lossless |
| `ScaleOffset::FloatDScale(decimals)` | `f32` / `f64` | lossy to `decimals` digits |

### ZFP (optional, `zfp` feature)

Pure-Rust fixed-rate port of the LLNL/zfp codec, registered HDF5 filter
ID 32013. Byte-for-byte interoperable with the reference H5Z-ZFP plugin:
files we write are readable by `h5py` + `hdf5plugin`, and files those tools
produce are readable by us. Supported slice:

- Scalar types: `f32`, `f64`, `i32`, `i64`
- Ranks: 1D, 2D, 3D, 4D (per-block sizes 4, 16, 64, 256)
- Mode: fixed-rate (`rate` bits per value)

```rust
// Compile with `--features zfp`
builder.create_dataset("temperature")
    .with_f32_data(&data)
    .with_shape(&[ny, nx])
    .with_chunks(&[ny, nx])
    .with_zfp(16.0);  // 16 bits per value
```

Interop is enforced by `src/zfp_crosscheck.rs`, which compares against
fixtures produced by `h5py` + `hdf5plugin`. See `tests/fixtures/zfp/regen.py`
for the generator — run it after any codec change.

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
| `Matrix<T>` or `Vec<Vec<T>>` of same length | column-major 2-D dataset, HDF5 shape `[cols, rows]` |
| `Complex32` / `Complex64` | compound `{real, imag}` dataset |
| nested struct | HDF5 group with `MATLAB_class = "struct"`, `MATLAB_fields` |
| `Option<T>` (struct field) | omitted if `None` |
| unit enum variant | UTF-16 char dataset holding the variant name |
| `Vec<Struct>` / `Vec<Option<T>>` / ragged `Vec<Vec<T>>` | cell array (`MATLAB_class = "cell"`, object references into `#refs#`); `None` slots become `struct([])` |

### Cell array pattern

Sequences that don't unify into a numeric matrix lower to a MATLAB cell array. Each element is interned under the conventional `#refs#` group and the parent dataset stores object references.

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

In MATLAB this loads as `iscell(path) == true`, `path{1}.x`, etc. Empty `None` slots load as `struct([])` (`isempty(fieldnames(...))`).

**Reader compatibility.** Cell arrays load correctly in MATLAB, libmatio (reference C library), Julia's `MAT.jl`, and Python via `pymatreader` / `hdf5storage`. GNU Octave 11's `load` does not yet follow object references for v7.3 cells (warns "unknown datatype"); load such files with one of the above instead.

Not supported in this release: non-unit enum variants, MATLAB objects (`classdef`), datetime / categorical types.

## Cargo features

| Feature | Default | Description |
|---|---|---|
| `std` | yes | File I/O, high-level reader API |
| `checksum` | yes | Jenkins hash for v2+ object headers |
| `deflate` | yes | Deflate compression (pure Rust backend) |
| `serde` | no | Serialize/deserialize MATLAB v7.3 `.mat` files via serde |
| `fast-deflate` | no | zlib-ng backend for deflate via `flate2/zlib-ng` |
| `ndarray` | no | N-dimensional array I/O via the [`ndarray`](https://docs.rs/ndarray) crate |
| `parallel` | no | Parallel chunk processing via `rayon` |
| `provenance` | no | SHA-256 data provenance tracking |
| `zfp` | no | ZFP fixed-rate compression (HDF5 filter 32013), f32/f64/i32/i64 × 1D–4D |

For bare-metal `no_std`, disable default features (keep `checksum` for object-header validation):

```toml
[dependencies]
hdf5-pure = { version = "0.15", default-features = false, features = ["checksum"] }
```

The high-level `File` / `FileBuilder` API is `std`-gated, so a `no_std` build exposes only the lower-level primitives. WebAssembly builds keep the default features, since `std` is available on `wasm32-unknown-unknown`.

## Acknowledgements

The HDF5 format parsing and low-level I/O modules are derived from rustyhdf5 by the RustyStack project (MIT licensed).

## License

MIT
