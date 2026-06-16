# Architecture

`hdf5-pure` is a single Rust crate that reads, writes, and edits HDF5 files with
no C dependencies. This page explains how it is built and the principles that
shape it.

## Design goals

- **Zero C dependencies.** The HDF5 binary format is implemented directly in
  Rust. There is no `libhdf5`, no `build.rs` linking step, and no system
  package to install.
- **Portable.** The crate compiles to `wasm32-unknown-unknown` and to bare-metal
  `no_std` targets (with `alloc`). The high-level filesystem API is `std`-gated;
  the in-memory parsing and serialization machinery is not.
- **Interoperable.** Files this crate writes are read by the reference HDF5 C
  library, h5py, and MATLAB, and vice versa. Interop is verified by crosscheck
  tests that compare byte-for-byte against fixtures produced by those tools.
- **Faithful or nothing.** Every operation that cannot reproduce data exactly
  refuses with a named error rather than silently writing a file that quietly
  differs. See [Fidelity](#fidelity-refuse-rather-than-degrade) below.

## The pipeline

Internally the crate is layered. Lower layers know nothing about the layers
above them; data flows up from raw bytes to typed values.

| Layer | Responsibility |
|---|---|
| **Primitives** | Errors, checked integer conversions, checksums, byte sources, the file signature |
| **Format structures** | Superblocks, object headers, datatypes, dataspaces, data layouts, links, heaps (local, global, fractal), B-trees (v1/v2), symbol tables |
| **Filters** | The filter pipeline plus deflate, shuffle, scale-offset, and the optional ZFP codec |
| **Engine** | The chunked read/write core, chunk indexes (B-tree v1, fixed array, extensible array), the chunk cache, the file writer, attribute and group machinery |
| **High-level API** | `File`, `Dataset`, `Group`, `FileBuilder`, `EditSession`, `SwmrWriter`, and the `ndarray` integration |
| **MATLAB v7.3** | The serde-based `.mat` reader/writer, built on top of the engine and the high-level API |

The read and write paths are mutually recursive — index structures call back
into the writer, the writer drives chunked I/O, and so on — so the engine is a
single strongly-connected component that cannot be cleanly split into separate
"reader" and "writer" crates without dependency inversion.

## On-disk format coverage

The reader and editor handle the formats the reference C library and h5py
produce in the wild:

- **Superblocks** version 0, 1, 2, and 3.
- **Object headers** version 1 (with continuation blocks) and version 2,
  including multi-chunk headers.
- **Storage layouts** — contiguous, compact, and chunked.
- **Chunk indexes** — B-tree v1, fixed array (including the paged data-block
  layout), and extensible array (which also backs [SWMR](../guide/swmr.md)
  append).
- **Groups** — both the old symbol-table form (v0/v1) and the modern
  compact-link and dense (fractal-heap + v2 B-tree) forms.
- **Datatypes** — fixed-point, floating-point, string (fixed and
  variable-length), bit-field, opaque, compound, enumeration, array, and
  reference classes.

The crate itself writes one canonical modern format (the HDF5 1.10+ version-3
superblock with latest-format object headers), so its output is compact and
consistent while its reader remains broad.

## Fidelity: refuse rather than degrade

A recurring theme across the [editing](../guide/editing.md),
[repack](../guide/repack.md), and [streaming](../guide/streaming.md) APIs is
that they decline anything they cannot reproduce exactly. An `EditSession` that
cannot reproduce an object faithfully fails with `Error::EditUnsupported`;
`repack` fails with `Error::RepackUnsupported`, naming the object, and writes no
output file. The guarantee this buys you: an operation either produces a file
that is correct down to the bytes, or it produces no file and tells you why.

## Safety and robustness

- **Almost entirely safe Rust.** The only non-trivial `unsafe` is the
  cache-line-aligned chunk buffer in the chunk cache, which is exercised under
  [Miri](https://github.com/rust-lang/miri) with strict provenance in CI.
- **32-bit safe.** Every file-derived offset and length is narrowed through
  checked conversions, so a 64-bit value that does not fit a 32-bit `usize`
  errors instead of truncating. CI runs the suite on `i686` under QEMU and
  builds for `thumbv7em-none-eabi`.
- **Property-tested.** The write/read roundtrip and parser robustness are
  covered by property-based tests in addition to the example- and
  fixture-driven suites.

## Provenance

The HDF5 format parsing and low-level I/O modules are derived from rustyhdf5 by
the RustyStack project (MIT licensed).
