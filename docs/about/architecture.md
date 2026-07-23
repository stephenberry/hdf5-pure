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

## Write paths

Three paths put bytes on disk, with different cost models:

| Path | Entry point | What it writes |
|---|---|---|
| **Whole-file writer** | [`FileBuilder`](../guide/writing.md) | Serializes a brand-new file from scratch |
| **In-place editors** | [`File::open_rw`, `File::open_rw_bounded`](../guide/editing.md) | Appends new bytes to the existing file and patches a small, fixed set of locations |
| **Repack** | [`repack`](../guide/repack.md) | Reads a source file and writes a fresh, compact copy to a separate destination through `FileBuilder` |

`File::open_rw` is an **append-and-patch** editor. It reads the file into a whole-file in-memory mirror at open — a statement about *memory* (`O(file size)`), not about what gets written back. An immediate `Dataset::append` writes the new chunks, fills Extensible-Array index slots, and publishes the grown dataspace dimension last under `fsync` barriers, so each append is durable and crash-atomic before the call returns. Every other edit is staged and applied by one `commit()`: new data and new object headers are appended at the end of the file (or placed into space freed by earlier commits), a rewritten header for each touched group and its ancestors up to the root is appended, and the superblock is repointed at the new root **last**, as the single crash-atomic commit point. A same-length value overwrite patches the existing bytes where they lie. Space freed by deletions and relocations goes to a session free list for reuse; a freed run reaching the end of the file is truncated, and on a file that persists its free space the free list is instead serialized into on-disk free-space managers that survive reopen.

**Nothing re-serializes the file on commit.** The cost of a commit is proportional to the edit, not to the file size: a staged append re-encodes at most the trailing partial chunk and carries every other kept chunk into the rebuilt index by metadata alone, and no existing object moves except the object headers on the edited path. A failed or interrupted commit leaves the file valid: structural changes become visible only at the superblock repoint, so a crash before it leaves the old object tree in place. (Same-length value overwrites patch live data blocks directly and are the one staged edit outside that gate.)

`File::open_rw_bounded` is the same append engine minus the mirror: reads are positioned I/O through bounded caches (the read-write sibling of [`open_streaming`](../guide/streaming.md)), `Dataset::append` reads and patches only the metadata windows it touches with the same crash-atomicity, and `close()` rewrites the on-disk free-space managers of a file that persists them — including the per-page-type managers of a paged file, whose appends stay page-homogeneous. It refuses the staged edit surface rather than emulate it with unbounded memory. The two editors share the Extensible-Array append planner and the free-space-manager serialization; the [capability matrix](../guide/editing.md#choosing-a-write-path) documents which operations each supports on which file-space strategy.

The [SWMR writer](../guide/swmr.md) is a restriction of the same immediate append engine — unfiltered, chunk-aligned appends only — chosen so a concurrent reader never observes a torn view.

`repack` is the one operation that rewrites a file from scratch, and it never does so in place: it reads every surviving object and writes a fresh, compact copy at a separate destination path, refusing (`Error::RepackUnsupported`) anything it cannot reproduce faithfully.

## On-disk format coverage

The reader and editor handle the formats the reference C library and h5py
produce in the wild:

- **Superblocks** version 0, 1, 2, and 3.
- **Object headers** version 1 (with continuation blocks) and version 2,
  including multi-chunk headers.
- **Storage layouts** — contiguous, compact, and chunked.
- **Chunk indexes** — B-tree v1, fixed array (including the paged data-block
  layout), and extensible array (which also backs [SWMR](../guide/swmr.md)
  append and the in-place append paths — `EditSession::append_dataset` and the
  owned-handle `File::open_rw` + `Dataset::append`).
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
