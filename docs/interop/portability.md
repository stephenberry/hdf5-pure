# Portability: WASM, no_std & the C Library

`hdf5-pure` is pure Rust with no C dependencies and no build-time linkage to
libhdf5, which gives it three portability properties that the reference library
cannot offer: it compiles to WebAssembly, it compiles for `no_std` targets with
`alloc`, and the files it produces are byte-compatible with the rest of the HDF5
ecosystem. This page covers all three, including what is and is not available
without `std`.

## WebAssembly

`hdf5-pure` builds for `wasm32-unknown-unknown` with no extra toolchain. The key
fact to get right: the Rust `std` library **is** available on that target, so
you keep the default features (which include `std`). The crate's high-level
reader and writer are gated behind `std`, so turning default features off would
compile `File` and `FileBuilder` away — exactly what you do *not* want for a
WASM app.

```bash
rustup target add wasm32-unknown-unknown
cargo build --target wasm32-unknown-unknown
```

In the browser you use the **in-memory** path, which never touches a filesystem:
[`FileBuilder::finish`](../guide/writing.md) returns the complete file as a
`Vec<u8>` you can hand to JavaScript, and `File::from_bytes` parses bytes you
get back.

```rust
use hdf5_pure::FileBuilder;

let mut builder = FileBuilder::new();
builder.create_dataset("x").with_f64_data(&[1.0, 2.0]);

let bytes: Vec<u8> = builder.finish().unwrap(); // in memory, no filesystem
```

Reading is symmetric:

```rust
use hdf5_pure::File;

let file = File::from_bytes(bytes).unwrap();
let values = file.dataset("x").unwrap().read_f64().unwrap();
```

The path-based entry points (`File::open`, `FileBuilder::write`, `EditSession`,
`SwmrWriter`) still compile for WASM, but they cannot reach a filesystem at
runtime in the browser. Build your WASM code around `finish` and `from_bytes`.

!!! tip "Trimming the build"

    `deflate` is on by default (a pure-Rust backend, so it compiles to WASM
    fine). If you only handle uncompressed datasets you can drop it with
    `default-features = false, features = ["std", "checksum"]` — but keep `std`,
    or the high-level API disappears.

## no_std with alloc

With the default features off, the crate is `#![no_std]` and relies only on
`alloc` — it allocates `Vec`s and similar but never calls the operating system.
It compiles for freestanding targets, and CI builds `thumbv7em-none-eabi` to
keep that honest.

There is an important limitation today. The high-level, path-and-image API —
`File`, `FileBuilder`, `EditSession`, `SwmrWriter`, `repack`, and the `mat`
module — is `std`-gated, so a pure-`no_std` build (`--no-default-features`)
*compiles* but does not expose the whole-file reader and writer. What stays
available without `std` is the lower-level surface: the datatype constructors
(`make_f64_type` and friends), the `DatasetBuilder` / `GroupBuilder` and the
compound/enum type builders, `ScaleOffset`, and the format primitives. So
`no_std` is a supported *compilation* target for embedding the format
machinery; building or reading a complete file still needs `std` — which, as
shown above, is available on `wasm32-unknown-unknown`.

| Capability | API | Requires `std` |
|---|---|:---:|
| Datatype & builder primitives | `make_*_type`, `DatasetBuilder`, `GroupBuilder`, `ScaleOffset` | no (`alloc` only) |
| Build a whole file in memory | `FileBuilder::new` / `FileBuilder::finish` | yes |
| Parse a file from memory | `File::from_bytes` | yes |
| Open a file by path | `File::open` | yes |
| Streaming read by path | `File::open_streaming` | yes |
| SWMR follow read by path | `File::open_swmr` | yes |
| Write a file to a path | `FileBuilder::write` | yes |
| Edit a file in place | [`EditSession`](../guide/editing.md) | yes |
| Append in SWMR mode | [`SwmrWriter`](../guide/swmr.md) | yes |
| Compact a file | [`repack`](../guide/repack.md) | yes |
| MATLAB `.mat` via serde | `mat` module | yes (`serde`) |
| N-dimensional array I/O | `with_ndarray` / `read_array` | yes (`ndarray`) |

!!! note

    The `ndarray` and `serde` features both imply `std`, because they build on
    the path-based `File` / `Dataset` reader and writer APIs. See the
    [features reference](../reference/features.md) for the full feature matrix
    and the [installation guide](../getting-started/installation.md) for
    dependency setup.

## Reference-library interoperability

`hdf5-pure` does not define its own dialect of HDF5: it writes and reads the
standard on-disk format. Files this crate writes are readable by the reference
HDF5 C library, by `h5py`, and by MATLAB; files those tools produce are readable
here. This holds for the format features the crate supports — multiple
superblock versions, object header layouts, contiguous and chunked storage, and
the built-in deflate, shuffle, and scale-offset filters.

Interoperability is not asserted by hand. It is enforced by byte-level
crosscheck tests that compare the bytes this crate emits against fixtures
produced by the reference toolchain, so a regression in the on-disk layout fails
the test suite rather than slipping out as a quietly incompatible file. The same
discipline backs the optional [ZFP filter](../guide/compression.md)
(`src/zfp_crosscheck.rs` compares against `h5py` + `hdf5plugin`) and the MATLAB
`.mat` path. For the cross-tool story in depth, see the
[MATLAB interop page](matlab.md).

### 32-bit safety

The same crosscheck discipline extends to 32-bit hosts. Every offset and length
read out of a file is narrowed through checked conversions, so a 64-bit value
that does not fit a 32-bit `usize` produces an error rather than a silent
truncation. This is why a file too large for 32-bit address space should be read
with `File::open_streaming` (see [streaming](../guide/streaming.md)) instead of
`File::open`. CI exercises the suite on `i686` under QEMU.

### Memory safety

The crate is almost entirely safe Rust. The only non-trivial `unsafe` is the
cache-line-aligned chunk buffer in the chunk cache, and it is exercised under
Miri with strict provenance in CI. The [architecture page](../about/architecture.md)
covers the safety and robustness guarantees in more detail.
