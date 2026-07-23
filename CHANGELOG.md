# Changelog

All notable changes to this crate are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this crate follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) under Cargo's pre-1.0 conventions: a `0.x.0` bump may be breaking, `0.x.y` is not.

## [Unreleased]

### Fixed

- `Dataset::read_raw_rows` and the typed `read_*_rows` now stream a row window of an inner-chunked dataset by decoding only the chunks the window overlaps, instead of falling back to a whole read, so peak memory scales with the window plus one chunk rather than the dataset ([#183](https://github.com/stephenberry/hdf5-pure/pull/183)).

## [0.23.2] - 2026-07-23

Two fixes to the windowed row-read API introduced in 0.23.0: a full-range `Dataset::read_raw_rows` / `read_*_rows` window now delegates to the whole read instead of paying a full-size copy on top of it on layouts whose windowed reads fall back to one (inner-chunked storage, variable-length strings), and `Dataset::read_string_rows` now slices a multi-dimensional variable-length string dataset by row rather than by first-dimension index. Non-breaking patch.

### Fixed

- `Dataset::read_raw_rows` and the typed `read_*_rows` now delegate to a whole read when the window covers every row, so full-range windows on layouts whose windowed reads fall back to a whole read (inner-chunked storage, variable-length strings) no longer pay a full-size copy on top of it ([#181](https://github.com/stephenberry/hdf5-pure/pull/181)).
- `Dataset::read_string_rows` on a multi-dimensional variable-length string dataset now slices by row — each row spanning its inner dimensions — instead of treating the flat element array as one string per row, so a windowed read returns the same rows as `read_raw_rows` ([#182](https://github.com/stephenberry/hdf5-pure/pull/182)).

## [0.23.1] - 2026-07-23

Two file-space fixes from documenting and fuzz-testing the paged and persisted surface ([#178](https://github.com/stephenberry/hdf5-pure/issues/178)): a fresh `persist = true` file with a non-paged strategy now records a defined end-of-allocation, so an assertion-enabled build of the reference C library opens it instead of aborting, and the `File::open_rw_bounded` refusal for a non-persisting paged file now advises the right recovery. Non-breaking patch.

### Fixed

- A fresh file written with `persist = true` and a non-paged file-space strategy (`FsmAggr`/`Aggr`/`None`) now records a defined end-of-allocation in its File Space Info message instead of the undefined sentinel, so an assertion-enabled build of the reference C library opens it instead of aborting; release builds already tolerated it ([#178](https://github.com/stephenberry/hdf5-pure/issues/178)).
- The refusal when opening a *non-persisting* paged file with `File::open_rw_bounded` no longer points at `File::open_rw` (which also refuses a paged file); it now advises recreating the file with `persist = true`, the way to grow a paged file in place ([#178](https://github.com/stephenberry/hdf5-pure/issues/178)).

## [0.23.0] - 2026-07-22

Paged file-space support lands ([#173](https://github.com/stephenberry/hdf5-pure/issues/173)): `FileBuilder::with_file_space_strategy(FileSpaceStrategy::Page, …)` now writes a **genuine paged file** — page-aligned allocations with metadata and raw data in separate pages and per-page-type free-space managers — and `File::open_rw_bounded` grows a file that persists its free space, including a paged one, with bounded memory, rewriting its managers at `File::close` so the reference C library reads the result. Also new: `Dataset::read_raw_rows` and the typed `read_*_rows` stream a `[start, start + count)` leading-dimension row window without materializing the whole dataset ([#170](https://github.com/stephenberry/hdf5-pure/pull/170)). Additive minor bump.

### Added

- `FileBuilder::with_file_space_strategy(FileSpaceStrategy::Page, …)` now writes a **genuine paged file** instead of only recording the label: allocations are aligned to `with_file_space_page_size`, metadata and raw data occupy separate pages, and each page's free tail is tracked in a per-page-type free-space manager, so the reference C library reads it as a paged file, parses the managers (`H5Fget_freespace`), and re-paginates it on write ([#173](https://github.com/stephenberry/hdf5-pure/issues/173)).
- `File::open_rw_bounded` now grows a file that persists its free space (`H5Pset_file_space_strategy(persist = true)`): its on-disk free-space managers are seeded on open and rewritten at `File::close`, so bounded-memory appends round-trip through the reference C library. This includes a genuine **paged** file (`H5F_FSPACE_STRATEGY_PAGE`), whose appends are kept page-homogeneous (raw and metadata in separate pages) and whose per-page-type managers are rewritten at close; a paged file without persisted free space is refused ([#173](https://github.com/stephenberry/hdf5-pure/issues/173)).
- `Dataset::read_raw_rows` and the typed `read_f64_rows`/`read_f32_rows`/`read_i8_rows`/`read_i16_rows`/`read_i32_rows`/`read_i64_rows`/`read_u8_rows`/`read_u16_rows`/`read_u32_rows`/`read_u64_rows`/`read_string_rows` read a leading-dimension row window `[start, start + count)` without materializing the whole dataset, so a large dataset can be streamed a fixed number of rows at a time; inner-chunked and variable-length string windows fall back to a whole read sliced to the window ([#170](https://github.com/stephenberry/hdf5-pure/pull/170)).

## [0.22.0] - 2026-07-22

The owned-handle API lands ([#148](https://github.com/stephenberry/hdf5-pure/issues/148)): `Dataset`, `Group`, and `Object` are now owned handles with no `<'f>` lifetime and `File` is cheaply cloneable, so a handle can be stored, cached, sent across threads, and outlive its `File`. A file opened with `File::open_rw` or `File::create` reads, appends, edits, and commits through those handles (`Dataset::append` is immediate and crash-atomic), with `File::open_swmr_writer` for lock-free SWMR appends and `File::open_rw_bounded` for reading and appending with memory bounded independent of file size. The legacy `EditSession`, `SwmrWriter`, and `AppendWriter` are deprecated in favor of it. Also new: filtered in-place append ([#144](https://github.com/stephenberry/hdf5-pure/issues/144)), layout/filter and live-space introspection ([#149](https://github.com/stephenberry/hdf5-pure/issues/149), [#150](https://github.com/stephenberry/hdf5-pure/issues/150)), and configurable fill values ([#151](https://github.com/stephenberry/hdf5-pure/issues/151)). **Breaking:** the handle lifetime is gone (drop `Dataset<'_>`), and `File::refresh` now reports outstanding handles at runtime with `Error::HandlesOutstanding`.

### Breaking

- **Breaking:** `Dataset`, `Group`, and `Object` are now owned handles with no `<'f>` lifetime — `File::dataset`/`group`/`root` hand back handles that share ownership of the open file (internally `Arc`), so a handle can be stored in a struct, cached, sent across threads, and outlive the `File` value it came from, and `File` is now cheaply cloneable. Code that never named the handle lifetime is unaffected; code that wrote `Dataset<'_>` should drop the lifetime, and `File::refresh` now returns `Error::HandlesOutstanding` when a handle or `File` clone is still alive instead of enforcing it at compile time ([#148](https://github.com/stephenberry/hdf5-pure/issues/148)).

### Added

- `File::open_rw_bounded` (and `_with_options`) opens a file for reading and appending with **bounded memory** — no whole-file mirror: streaming-grade reads plus the same immediate, crash-atomic `Dataset::append` as `open_rw`, with large appends applied in whole-chunk batches so peak memory stays at the configured caches plus a few chunks regardless of file or call size. The staged edit surface returns the new `Error::BoundedStagedUnsupported` ([#147](https://github.com/stephenberry/hdf5-pure/issues/147)).
- `File::open_rw` opens a file for reading **and** writing through owned handles, and `File::create` builds a new file the same way: `Dataset::append` grows a chunked, unlimited, Extensible-Array-indexed dataset in place (immediate and crash-atomic, reading back through the same handle), while `Dataset::write`/`set_attr`/`remove_attr`, `Group::create_dataset`/`create_group`/`delete`, and `File::copy` stage edits that `File::commit` applies as one transaction. A write on a read-only file returns `Error::ReadOnly` ([#148](https://github.com/stephenberry/hdf5-pure/issues/148)).
- The owned-handle write surface reaches parity with `EditSession`: `Dataset::append_staged` grows a dataset with a rebuilt index staged until `commit` — including the **filtered** and non-chunk-aligned appends the immediate `Dataset::append` refuses; `File::copy_from` stages a cross-file `H5Ocopy` from a buffered read-only file; `Group::set_attr`/`remove_attr` edit a group's (or the root's) compact attributes; `File::space_accounting` and `File::has_staged_edits` report live space use and whether a commit is pending; and `File::open_rw_with_locking` opens with an explicit `FileLocking` policy. `File::close` now seals the file, so a write through a surviving handle returns the new `Error::FileClosed` ([#148](https://github.com/stephenberry/hdf5-pure/issues/148)).
- `File::open_swmr_writer` opens a file for SWMR (single-writer/multiple-reader) appending through owned handles: it takes **no** OS lock (so concurrent readers, and Windows' mandatory locks, are never blocked) and raises the superblock's SWMR-write flag, cleared on `File::close`. Only immediate `Dataset::append` is allowed, over the unfiltered, chunk-aligned SWMR subset; the staged edit surface returns the new `Error::SwmrStagedUnsupported`, and `File::clear_swmr_flag` recovers a flag left set by a crashed writer ([#148](https://github.com/stephenberry/hdf5-pure/issues/148)).
- `EditSession::append_inplace` grows an existing **chunked, unlimited, Extensible-Array dataset** in place at amortized `O(1)` cost — immediate and crash-atomic, needing no `commit` — and can be interleaved with the session's staged group/dataset/attribute/delete edits on one open file, with no reopening between the fast appends and the tree edits. Unfiltered datasets accept any-length appends, filtered datasets whole chunks only; a userblock or pre-v2 file, an unallocated or non-Extensible-Array index, or a multi-hard-link dataset is refused with `Error::AppendInPlaceUnsupported` (use `append_dataset` instead) ([#146](https://github.com/stephenberry/hdf5-pure/issues/146)).
- `EditSession::set_dataset_attr` / `remove_dataset_attr` add, update, or remove a compact **dataset** attribute — fixed-size or variable-length string — staged until `commit`; a dense (fractal-heap) attribute store or a multi-hard-link dataset is refused ([#146](https://github.com/stephenberry/hdf5-pure/issues/146)).
- `EditSession::append_dataset` grows an existing **chunked, unlimited dataset** in place along its first dimension — **filtered** (deflate/shuffle/fletcher32/scale-offset, and ZFP with the `zfp` feature) or not, and of any length (a trailing partial chunk is rewritten) — without requiring SWMR; existing chunk data stays put while the appended chunks and a rebuilt Extensible-Array index are added, and the result reads back in the reference C library and h5py. Datasets that are not Extensible-Array-indexed (a version-1 B-tree, fixed-array, or single-chunk index), higher than rank 1, use a filter this engine cannot re-encode, or have more than one hard link are refused ([#144](https://github.com/stephenberry/hdf5-pure/issues/144)).
- `Dataset` gains read-only introspection — `is_chunked`, `maxshape`, `chunk_shape`, and `filters` — so callers can check a dataset's storage, extensibility, and filter pipeline (for example append eligibility) without decoding any data ([#144](https://github.com/stephenberry/hdf5-pure/issues/144)).
- `Dataset::layout`, `chunk_index`, `chunks`, and `filter_pipeline` expose the full storage layout and filter pipeline through the curated `Layout`, `ChunkIndex`, `Chunk`, and `Filter` types — the storage class, chunk-index kind (with `ChunkIndex::supports_inplace_append`), and each chunk's absolute file address, on-disk size, and filter mask, plus each filter's id, name, optional flag, and client data — so a caller can locate and read one chunk at a time without materializing the dataset. Enumerating a version-2 B-tree index's chunks is not yet supported ([#149](https://github.com/stephenberry/hdf5-pure/issues/149)).
- `EditSession::space_accounting` reports a mutating session's live space usage as a `SpaceAccounting` — the current logical file size, the total reusable free bytes, and the reusable free regions as absolute `(offset, length)` pairs — the active-editor counterpart of `File::file_size` and `persisted_free_space`; it reflects committed state plus immediate in-place appends, not edits still staged for `commit` ([#150](https://github.com/stephenberry/hdf5-pure/issues/150)).
- `DatasetBuilder::with_fill_value` records a dataset's fill value — the value HDF5 reports for never-written elements — and `Dataset::fill_value` reads one back, from this crate's files as well as the reference C library's and h5py's; the fill value's type must match the dataset datatype ([#151](https://github.com/stephenberry/hdf5-pure/issues/151)).

### Deprecated

- `AppendWriter` is deprecated in favor of `File::open_rw` plus `Dataset::append`, which offers the same amortized `O(1)` in-place append through one open file that also reads and edits; it still works and will be removed in a later release ([#148](https://github.com/stephenberry/hdf5-pure/issues/148)).
- `SwmrWriter` and `EditSession` are deprecated in favor of the owned-handle API and will be removed in a later release: open with `File::open_swmr_writer` or `File::open_rw` and mutate through owned `Dataset`/`Group` handles that read and write one file by name (`Dataset::append`/`append_staged`/`write`, `Group::create_dataset`/`create_group`/`delete`, `File::copy_from`/`commit`/`clear_swmr_flag`) ([#148](https://github.com/stephenberry/hdf5-pure/issues/148)).

### Fixed

- Reading through a `Dataset` handle after appending through that same handle no longer returns stale data: the append now invalidates the handle's cached chunk index, which previously still pointed at the relocated trailing chunk ([#147](https://github.com/stephenberry/hdf5-pure/issues/147)).
- Variable-length string/sequence reads and `Dataset::chunks` introspection now work on read-write files (`File::open_rw` and the new bounded mode): these paths previously read the global heap through an empty byte view on the mirror backend and failed with an EOF error ([#147](https://github.com/stephenberry/hdf5-pure/issues/147)).
- Reading an attribute or dataset whose dataspace declares dimensions whose product overflows `u64` no longer panics: the element count now saturates so the size and limit checks reject the file as a format error ([#142](https://github.com/stephenberry/hdf5-pure/issues/142)).
- docs.rs now documents the full public API — the `ndarray`, `serde` (`mat`), `zfp`, `provenance`, and `parallel` surfaces, previously hidden by a default-features-only build — and repairs the broken rustdoc intra-doc links across the public API ([#154](https://github.com/stephenberry/hdf5-pure/pull/154)).

## [0.21.2] - 2026-07-14

The `.mat` serializer now drops a struct field that serializes as a Rust unit `()` — most commonly a `serde_json::Value::Null` — like `Option::None` instead of aborting the encode. Parser hardening: the buffered and streaming readers agree on a malformed v1 object header, and crafted files return a format error instead of panicking on an arithmetic overflow across the metadata parsers. Non-breaking patch.

### Fixed

- `mat::to_bytes` no longer aborts the whole encode when a struct field serializes as a Rust unit `()` — most commonly a `serde_json::Value::Null` field: the field is now dropped like `Option::None` (read it back with `#[serde(default)]`) instead of failing with `UnsupportedType("() / unit")` ([#141](https://github.com/stephenberry/hdf5-pure/pull/141)).
- The buffered and streaming readers now agree on a malformed v1 object header: the buffered path stops at the declared object-header size instead of reading (and following) a chunk-0 message that overruns it ([#140](https://github.com/stephenberry/hdf5-pure/pull/140)).
- Parsing a crafted file now returns a format error instead of panicking on an arithmetic overflow, hardening address and size computations across the metadata parsers (local heap, symbol table, datatype sizing, and the chunk/fixed-array/extensible-array indexes) ([#140](https://github.com/stephenberry/hdf5-pure/pull/140)).

## [0.21.1] - 2026-07-08

Base-address normalization now rejects a `u64` overflow with an `OffsetOverflow` error instead of panicking or silently wrapping, hardening the parser against a crafted superblock base address. The check covers the superblock root-group address on both the read and edit paths and group-child object-header addresses. Non-breaking patch.

### Fixed

- Reject base-address normalization that overflows `u64` instead of panicking or wrapping, covering the superblock root-group address (read and edit paths) and group-child object-header addresses.

## [0.21.0] - 2026-07-02

`EditSession` gains three in-place additions: an **empty (zero-element) contiguous dataset** and a **provenance-tagged dataset** (`DatasetBuilder::with_provenance`, behind the `provenance` feature); a **variable-length attribute value** (`AttrValue::VarLenAsciiArray`) and a **variable-length-string dataset** (`DatasetBuilder::with_vlen_strings`); and an **object-reference dataset** (`DatasetBuilder::with_path_references`). Chunked/extensible variants of each stay refused. Additive minor bump.

### Added

- `EditSession` now adds, in place, an **empty (zero-element) contiguous dataset** and a **provenance-tagged dataset** (`DatasetBuilder::with_provenance`, behind the `provenance` feature); a chunked/extensible empty dataset stays refused ([#105](https://github.com/stephenberry/hdf5-pure/issues/105)).
- `EditSession` now adds, in place, a dataset, group, or root attribute with a **variable-length value** (`AttrValue::VarLenAsciiArray`) and a **variable-length-string dataset** (`DatasetBuilder::with_vlen_strings`); dense-attribute storage and a chunked/extensible variable-length-string dataset stay refused ([#105](https://github.com/stephenberry/hdf5-pure/issues/105)).
- `EditSession` now adds, in place, an **object-reference dataset** (`DatasetBuilder::with_path_references`); a target the same commit is still writing is refused rather than resolved to a stale address, and a chunked/extensible reference dataset stays refused ([#105](https://github.com/stephenberry/hdf5-pure/issues/105)).

### Fixed

- `EditSession::create_dataset(...).with_vlen_strings(...)` no longer silently corrupts the added dataset: `commit()` now writes and patches its global heap collection, so the dataset reads back instead of failing with `InvalidGlobalHeapSignature` ([#105](https://github.com/stephenberry/hdf5-pure/issues/105)).

## [0.20.1] - 2026-07-01

HDF5 **enumeration datasets** now read back through the typed integer/float readers via their integer base type, so an enum dataset written with `EnumTypeBuilder` / `DatasetBuilder::with_enum_i32_data` reads its codes instead of failing with a `TypeMismatch`. Non-breaking patch.

### Fixed

- Typed integer and float readers (`Dataset::read_i32`, `read_u8`, …) now decode an **HDF5 enumeration dataset** as its integer base type, so an enum dataset written with `EnumTypeBuilder` / `DatasetBuilder::with_enum_i32_data` reads its codes back instead of failing with a `TypeMismatch`; member names stay available via `DType::Enum`, and no name-based enum-to-enum conversion is performed ([#129](https://github.com/stephenberry/hdf5-pure/issues/129)).

## [0.20.0] - 2026-06-24

MATLAB **struct arrays** now read: a `MATLAB_class="struct"` group whose fields are datasets of per-element object references is transposed into an array-of-structs, so `mat::from_file` / `mat::from_bytes` read a `1×N` / `N×1` struct array into `Vec<T>` and an `M×N` array into `Vec<Vec<T>>`. Additive minor bump.

### Added

- MATLAB **struct arrays** now deserialize: a `MATLAB_class="struct"` group whose fields are datasets of per-element object references is transposed into an array-of-structs, so `mat::from_bytes` / `mat::from_file` read a `1×N` / `N×1` struct array into `Vec<T>` and an `M×N` array into `Vec<Vec<T>>` — previously refused with a `Reference` type mismatch. A scalar struct still reads as a single struct ([#127](https://github.com/stephenberry/hdf5-pure/issues/127)).

## [0.19.0] - 2026-06-22

`EditSession` now edits files that carry a **userblock** (non-zero base address), such as MATLAB v7.3 `.mat` files: it reads and writes addresses relative to the base and preserves the userblock bytes, so every edit works — value overwrites, additions, relocating overwrites of every layout with old storage reclaimed, object deletion, in-file and cross-file copy, group creation, and compact attributes — with only cross-file copy from a userblock *source* still refused. Also fixes reading and repacking a chunked dataset from such a file. Additive minor bump.

### Added

- `EditSession` now opens and edits files that carry a **userblock** (non-zero base address), such as MATLAB v7.3 `.mat` files: it reads and writes addresses relative to the base and preserves the userblock bytes verbatim. Every edit is supported — value overwrites, additions, relocating overwrites of every layout (with old storage reclaimed), object deletion, in-file and cross-file copy, group creation, and compact attributes; only cross-file copy from a userblock *source* is still refused ([#104](https://github.com/stephenberry/hdf5-pure/issues/104)).

### Fixed

- Reading and repacking a **chunked dataset from a file with a userblock** (non-zero base address) now works; previously the base address was applied only to contiguous data, so chunked reads from such a file failed ([#104](https://github.com/stephenberry/hdf5-pure/issues/104)).

## [0.18.0] - 2026-06-20

Broad MATLAB v7.3 read support for MCOS opaque types — cell arrays, the modern `string` class, `datetime` / `duration` / `categorical`, `table` / `timetable`, enumeration arrays, and `containers.Map`, including objects nested inside structs, cells, and table columns, all resolved through the file's `#subsystem#`/MCOS store. Also adds in-place overwrite and copy of chunked & filtered datasets in `EditSession`, a faster MAT write path, and two compound-datatype read fixes. **Breaking:** `MatError` is now `#[non_exhaustive]`; minor bump.

### Added

- `EditSession::write_dataset` now overwrites **chunked and filtered** datasets in place: unfiltered chunks (and filtered chunks that re-encode to the same size or smaller) are written into their existing slots — a shrinking filtered overwrite rebuilds the fixed-/extensible-array index in place to record the new sizes — while one whose re-encoded chunks no longer fit is rebuilt and relocated with the old storage reclaimed. A version-2 B-tree chunk index is still refused ([#101](https://github.com/stephenberry/hdf5-pure/issues/101)).
- `EditSession::copy` / `copy_from` now copy a **chunked or filtered** dataset, preserving its chunk payloads and filter pipeline byte-for-byte (the chunk index is rebuilt at the new location, so a B-tree-v1 or implicit-indexed source becomes an equivalent v4 index); a version-2 B-tree index or a sparse chunk grid is still refused ([#101](https://github.com/stephenberry/hdf5-pure/issues/101)).
- MATLAB **cell arrays** now deserialize: `mat::from_bytes` / `mat::from_file` resolve each element's `#refs#` object reference and rebuild the sequence, so `Vec<Struct>`, ragged `Vec<Vec<T>>`, `Vec<Option<T>>` (with `None` slots restored), and nested cells round-trip — previously refused with `UnsupportedType("cell array")`. New public `Dataset::dereference` and `Object` resolve an HDF5 object reference (`H5R_OBJECT`) to the group or dataset it names, and MATLAB's reserved `#refs#` / `#subsystem#` groups are skipped on read ([#114](https://github.com/stephenberry/hdf5-pure/issues/114)).
- The modern MATLAB **`string`** class now deserializes: an opaque (`MATLAB_object_decode=3`) `string` dataset's object id is resolved against the `#subsystem#/MCOS` store and its UTF-16 saveobj payload decoded, so values written with `Options::with_modern_strings()` round-trip and a scalar `string` reads back as a Rust `String` ([#114](https://github.com/stephenberry/hdf5-pure/issues/114)).
- MATLAB **`datetime`**, **`duration`**, and **`categorical`** now deserialize into the new public `MatDatetime` / `MatDuration` / `MatCategorical` types (Unix-epoch millisecond instants, durations in milliseconds, and category codes plus names — lossless, with `nanoseconds()` / `seconds()` / `labels()` helpers). Any other MCOS opaque class (`table`, `containers.Map`, `dictionary`, user `classdef`s, …) is surfaced losslessly as its raw property map rather than refused, so unknown opaque variables still read; function handles and legacy objects (`MATLAB_object_decode` 1/2) remain refused by name. **Breaking:** `MatError` is now `#[non_exhaustive]` ([#114](https://github.com/stephenberry/hdf5-pure/issues/114)).
- Nested MATLAB **MCOS objects now decode**: a `string` / `datetime` / `duration` / `categorical` / struct / user-class value embedded inside another opaque object resolves to its real value instead of the raw `uint32` reference metadata, so a nested `datetime` (in a struct, a cell, or a table column) reads back decoded ([#114](https://github.com/stephenberry/hdf5-pure/issues/114)).
- MATLAB **`table`** and **`timetable`** variables now read. Each column is addressable by its variable name, so a table deserializes straight into your own struct (field name = column name) — `string` / `datetime` / `duration` / `categorical` / struct / user-class columns included — or into the new public `MatTable` / `MatTimetable` for schema-agnostic access through the `MatColumn` enum, with row names and timetable row-times exposed. Numeric columns surface as `f64` through `MatColumn` (read the typed-struct path for exact integer width); a table's `Properties` (units, descriptions, …) is not yet surfaced ([#114](https://github.com/stephenberry/hdf5-pure/issues/114)).
- MATLAB **enumeration** arrays now deserialize into the new public `MatEnum` (the class name plus each element's member name, row-major), wherever they appear — a top-level variable, a user-class property, a cell, or a struct field. The underlying value backing each member is not surfaced ([#114](https://github.com/stephenberry/hdf5-pure/issues/114)).
- MATLAB **`containers.Map`** variables now deserialize as a `key -> value` map: a string/char-keyed map reads straight into a `HashMap<String, V>` / `BTreeMap<String, V>` or a struct keyed by the map's keys, and numeric keys are presented as strings (`1.0` -> `"1"`). The `dictionary` type still reads losslessly as its raw property map; a typed `MatMap` introspection view is not yet provided ([#114](https://github.com/stephenberry/hdf5-pure/issues/114)).

### Fixed

- Read HDF5 **version-1 and version-2 compound datatypes** correctly: the member layout was misparsed (the v1 dimension block skipped one 4-byte reserved field, and v2 names were left unpadded), so complex data written by MATLAB and older HDF5 writers — including real-MATLAB `datetime` arrays — now decodes instead of failing with a type mismatch ([#114](https://github.com/stephenberry/hdf5-pure/issues/114)).
- An empty `datetime` or `duration` object stored with no `data` / `millis` property (e.g. a zero-row timetable's row-times) now decodes as empty instead of aborting the whole-file read ([#114](https://github.com/stephenberry/hdf5-pure/issues/114)).

### Performance

- Serializing a MATLAB v7.3 file is faster: the default `mat::to_bytes` write path now shares the cache-tiled column-major transpose (≈8% faster on a 512×512 `f64` matrix) instead of a strided copy, and numeric/field buffers across the read and write paths are pre-sized or filled in a single pass. Reading a numeric array no longer materializes an intermediate boxed-scalar buffer, and a `uint32` array nested under an MCOS object is decoded once instead of twice ([#122](https://github.com/stephenberry/hdf5-pure/pull/122)).

## [0.17.0] - 2026-06-18

Repack now reproduces three more datatype classes faithfully — non-string variable-length sequences, object-reference datasets, and time datatypes — and the dataset read and write hot paths are several times faster (bulk numeric decode, contiguous-row chunk scatter, compress-once filtered writes). **Breaking:** `Datatype::Time` gained a `byte_order` field, so code matching that variant must account for it; minor bump.

### Added

- Repack now reproduces three more datatype classes faithfully: non-string variable-length sequences (re-staged through a fresh global heap), object-reference datasets (each address rewritten to its target's new location in the compacted file), and time datatypes (byte order preserved). Chunked/filtered/resizable VL and reference datasets, region or non-8-byte object references, and an object reference to a dropped or out-of-hierarchy target are still refused by name ([#107](https://github.com/stephenberry/hdf5-pure/issues/107)).
- **Breaking:** `Datatype::Time` gained a `byte_order` field so a time type's byte order survives a read/serialize round-trip (it was previously dropped on read and forced little-endian); code matching the `Time` variant must account for the new field ([#107](https://github.com/stephenberry/hdf5-pure/issues/107)).

### Fixed

- A null or empty variable-length element now writes a zero heap address (HDF5's null-reference convention) instead of an all-ones undefined-address sentinel, which the reference C library rejected as a bad heap index when reading such an element back ([#107](https://github.com/stephenberry/hdf5-pure/issues/107)).

### Performance

- Decoding a numeric dataset into a typed `Vec` (`Dataset::read_i32`/`read_u16`/`read_f64` and siblings) now bulk-decodes native-/big-endian standard-layout values instead of going element by element, making integer reads several times faster (≈15× for `read_i32`, ≈9× for `read_u16` on a 1M-element array); sub-byte-precision and unusual layouts keep the exact same results ([#113](https://github.com/stephenberry/hdf5-pure/pull/113)).
- Reading a chunked dataset now scatters each chunk into the output one contiguous row at a time rather than element by element, ≈3× faster chunk assembly (a 1024×1024 uncompressed read drops from ~7.6 ms to ~2.2 ms) ([#113](https://github.com/stephenberry/hdf5-pure/pull/113)).
- Writing a chunked, filtered dataset now compresses each chunk once instead of twice (the object-header sizing pass no longer recompresses), ≈2–3× faster compressed writes (a 1024×1024 shuffle+deflate write drops from ~45 ms to ~16 ms) ([#113](https://github.com/stephenberry/hdf5-pure/pull/113)).
- The byte-shuffle filter is specialized for the common element widths, the chunk cache no longer copies decompressed chunks in and out on the hot path, and the deflate decoder pre-sizes its output buffer ([#113](https://github.com/stephenberry/hdf5-pure/pull/113)).

## [0.16.0] - 2026-06-18

Centers on `repack`: it now copies compressed chunks **verbatim** (so lossy filters survive byte-exact) and runs **fully out-of-core**, and gains variable-length-string support. Also adds in-place dataset-value overwrite, dense-attribute and cross-file object copy, in-place addition of chunked/filtered/extensible datasets, and free-space reclaim for chunked deletes; plus reader hardening (a multi-filter chunk-mask corruption fix, sub-byte integer precision, decompression-bomb bounds, and safer B-tree/heap refusals). Additive minor bump.

### Added

- Repack now copies a chunked dataset's compressed chunks **verbatim** instead of decompressing and re-compressing them, eliminating the per-dataset decompression blowup and the decompress→recompress round-trip, so **lossy** filters now survive byte-exact — float D-scale scale-offset, ZFP, SZIP, and even filters this crate cannot itself apply ([#82](https://github.com/stephenberry/hdf5-pure/issues/82), [#84](https://github.com/stephenberry/hdf5-pure/issues/84), [#85](https://github.com/stephenberry/hdf5-pure/issues/85)). The verbatim path covers a fully-allocated chunk grid; a sparse chunked or a contiguous/compact filtered dataset still re-encodes and refuses a lossy filter by name.
- Repack is now **fully out-of-core**, closing [#82](https://github.com/stephenberry/hdf5-pure/issues/82): it streams the source (`File::open_streaming`) and the output (`FileBuilder::finish_to`, a `std::io::Write` sink), so peak memory is bounded by one chunk plus the file's metadata regardless of dataset size. This extended the streaming reader to also read attributes (compact, shared, dense, and VL-string) and traverse v1 symbol-table groups ([#27](https://github.com/stephenberry/hdf5-pure/issues/27)).
- Variable-length string dataset writing and repack: `DatasetBuilder::with_vlen_strings(&[&str])` writes a contiguous VL UTF-8 string dataset (1D, or ND via `with_shape`), matching the C library's `H5Tvlen_create(H5T_C_S1)` layout so the C library and h5py read it back. Repack now round-trips contiguous/compact VL-string datasets, preserving charset, padding, the null-vs-empty distinction, embedded NULs, and non-UTF-8 bytes; chunked, filtered, or resizable VL-string datasets and non-string VL datatypes are still refused by name ([#83](https://github.com/stephenberry/hdf5-pure/issues/83)).
- In-place overwrite of dataset values: `EditSession::write_dataset(path)` replaces an existing contiguous or compact dataset's values (HDF5's `H5Dwrite` whole-dataset write), returning the same `DatasetBuilder` as `create_dataset`. The replacement must match the on-disk datatype and shape; a same-length contiguous overwrite writes straight into the existing data block, while a length change or a compact dataset relocates the header like an addition. Chunked and filtered datasets, and a relocating overwrite of a multiply-hard-linked dataset, are refused by name ([#79](https://github.com/stephenberry/hdf5-pure/issues/79)).
- Object copy now reproduces dense (fractal-heap) attribute storage: above the compact threshold of 8 attributes HDF5 stores attributes in a fractal heap indexed by a B-tree v2, and `EditSession::copy` and `copy_from` previously refused such objects. They now read the source attributes and re-emit them into a fresh destination-local heap, same-file and cross-file ([#87](https://github.com/stephenberry/hdf5-pure/issues/87)). For now a single direct block is emitted: a set too large for one direct block is refused by name, as is a cross-file dense set whose values are variable-length or reference data.
- Cross-file object copy: `EditSession::copy_from` copies a dataset or whole group subtree out of a *separate* open `File` into the file being edited — the cross-file form of HDF5's `H5Ocopy`, alongside the same-file `EditSession::copy` ([#78](https://github.com/stephenberry/hdf5-pure/issues/78)). The source is read and validated eagerly, so it returns a `Result`. Because the copy is verbatim, it refuses by name anything whose stored bytes embed a source-file address — variable-length and reference data or attributes, and any shared header message. The source must be a buffered file (`File::open` / `File::from_bytes`, not `open_streaming`) with 8-byte offsets and no userblock.
- Free-space reclaim for chunked datasets on in-place delete: deleting a chunked dataset (or a group whose subtree contains one) now returns its chunk data blocks and chunk-index structure to the free list, reused by a later commit and truncated away when the freed run reaches end-of-file, where previously a chunked dataset's storage was left as dead bytes ([#77](https://github.com/stephenberry/hdf5-pure/issues/77)). Covers single-chunk, implicit, fixed array, extensible array, and v1 B-tree indexes; a v2 B-tree index, an out-of-bounds or overlapping span, or VL global-heap data is left in place rather than risk freeing live bytes.
- In-place add of chunked, filtered, and extensible datasets: `EditSession::create_dataset` now accepts `with_chunks`, the writer's filters (`with_deflate`, `with_shuffle`, `with_fletcher32`, `with_scale_offset`, `with_zfp`), and `with_maxshape` (optionally unlimited) — previously only contiguous, unfiltered datasets ([#76](https://github.com/stephenberry/hdf5-pure/issues/76)). The added object header is byte-identical to a freshly written one, and the prior root stays intact until the superblock is repointed last.

### Fixed

- Reading a virtual (VDS) dataset now fails with a clear `FormatError::UnsupportedVirtualLayout` instead of a misleading `UnsupportedVersion(0)` (which rendered as "unsupported superblock version: 0"); VDS reading is tracked as a planned feature ([#111](https://github.com/stephenberry/hdf5-pure/issues/111)).
- Multi-filter chunks where only *some* filters were skipped for a chunk (the per-chunk `filter_mask`, e.g. shuffle+gzip on an incompressible chunk that the C library stores shuffled but not deflated) now have the surviving filters reversed instead of being returned raw, fixing silent value corruption on spec-valid files ([#97](https://github.com/stephenberry/hdf5-pure/issues/97)).
- Integers with sub-byte precision or a non-zero bit offset (`H5Tset_precision` / `H5Tset_offset`) now decode correctly in the dataset and attribute readers — masked to the significant bits and sign-extended at the precision boundary — instead of returning the raw stored word with its padding bits; compound fields with such layouts are still refused by name ([#97](https://github.com/stephenberry/hdf5-pure/issues/97)).
- A malformed v1 B-tree with a cyclic or pathologically deep internal node — in either the chunk index or a group's symbol table — now errors instead of recursing until the stack overflows and aborts the process; traversal is bounded by a depth cap ([#97](https://github.com/stephenberry/hdf5-pure/issues/97)).
- Deflate-compressed chunks are now bounded to their expected decompressed size: a chunk that inflates past it (a decompression bomb) or decodes to the wrong length is refused with `FormatError::DecompressionError` / `DataSizeMismatch` instead of allocating unbounded memory or silently zero-filling the result ([#97](https://github.com/stephenberry/hdf5-pure/issues/97)).
- A truncated or corrupt fixed-rate ZFP chunk now decodes without panicking (`zfp` feature) instead of aborting on an out-of-range slice past the end of the buffer ([#97](https://github.com/stephenberry/hdf5-pure/issues/97)).
- Reading an object from a *filtered* fractal managed heap is now refused cleanly with `FormatError::UnsupportedFilteredHeapObject` instead of silently misparsing it (the indirect-block child-pointer walk used the wrong stride for filter-encoded blocks) ([#80](https://github.com/stephenberry/hdf5-pure/issues/80)).
- Object copy (`EditSession::copy` and `copy_from`) no longer refuses an object whose Attribute Info message carries an *undefined* fractal-heap address — the reference C library and h5py emit that message (to record attribute creation order) alongside compact, inline attributes, and the editor mistook its mere presence for dense storage. It now inspects the heap address and refuses only genuine dense storage, on both the same-file and cross-file paths ([#78](https://github.com/stephenberry/hdf5-pure/issues/78)).
- In-place delete now reclaims an object's storage only when the link being removed is its *last* hard link; previously it freed the blocks unconditionally, so deleting one of several hard links returned still-referenced storage and silently corrupted the surviving link once those bytes were reused ([#77](https://github.com/stephenberry/hdf5-pure/issues/77)). The editor now counts every hard link before reclaiming and leaves a multiply-linked object's storage in place (a safe leak the repack path still compacts).
- Malformed chunk geometry is now refused up front by both `FileBuilder` and `EditSession` (`FormatError::InvalidChunkGeometry` / `Error::EditUnsupported`) instead of panicking in the chunk splitter: a chunk rank that disagrees with the shape, a zero chunk dimension, a max shape of the wrong rank or smaller than the current shape, chunking a scalar, and an element count that overflows `u64` ([#76](https://github.com/stephenberry/hdf5-pure/issues/76)). Zero-element extensible datasets remain valid.

## [0.15.0] - 2026-06-16

Adds generic element-typed dataset I/O, file- and dataset-level cache tuning, in-place group attribute editing, OS advisory file locking for the editor, and a gallery of runnable examples; also hardens the 32-bit/WASM readers against silent truncation. Additive minor bump, with two intended behavior changes (editor file locking and the new truncation guards) noted below.

### Added

- Generic, type-parameterized dataset I/O: `DatasetBuilder::with_data(&[T])` writes any supported scalar and `Dataset::read::<T>()` reads one back, so you can write code generic over the element type instead of reaching for `with_i64_data` / `read_i64` and friends. Backed by the now feature-independent `H5Element` bound (previously available only with the `ndarray` feature). Both delegate to the existing typed methods, so behavior is unchanged ([#53](https://github.com/stephenberry/hdf5-pure/issues/53)).
- File-access options applied at open time via `FileAccessOptions` and the matching `*_with_options` constructors (`File::open_with_options`, `open_streaming_with_options`, `open_swmr_with_options`, `from_bytes_with_options`): `MetadataCacheConfig` bounds the streaming reader's metadata cache and `ChunkCacheConfig` tunes the chunk cache ([#65](https://github.com/stephenberry/hdf5-pure/pull/65)).
- Per-dataset chunk-cache control: `File::dataset_with_options` / `Group::dataset_with_options` take a `DatasetAccessOptions` that overrides the file-wide chunk-cache default for a single dataset, mirroring HDF5's `H5Pset_chunk_cache` access property list. `Dataset::chunk_cache_config()` reports the effective setting ([#48](https://github.com/stephenberry/hdf5-pure/issues/48)).
- `ChunkCacheConfig::from_h5p_cache(rdcc_nslots, rdcc_nbytes)` builds a chunk-cache config straight from HDF5's `H5Pset_cache` raw-data parameters ([#66](https://github.com/stephenberry/hdf5-pure/pull/66)).
- `Dataset::chunk_cache_stats()` reports a read-only snapshot of a dataset's chunk-cache occupancy (index loaded, retained chunks, retained bytes), so callers can confirm their chunk-cache tuning is taking effect ([#68](https://github.com/stephenberry/hdf5-pure/pull/68)).
- In-place group attribute editing: `EditSession::set_group_attr` adds or replaces a compact group attribute and `EditSession::remove_group_attr` removes one, without rewriting the file ([#64](https://github.com/stephenberry/hdf5-pure/pull/64)).
- OS advisory file locking for the in-place editor, the crash-safe half of HDF5's concurrency model and the analogue of `H5Pset_file_locking`. `EditSession::open` takes an exclusive lock, so a second editor (or any concurrent writer) gets the new `Error::FileLocked`; the kernel releases it on any process exit, including a crash, so a crashed editor never leaves a stale lock. Control it with the new `FileLocking` policy (`EditSession::open_with_locking`) or `HDF5_USE_FILE_LOCKING=FALSE` for filesystems where locking is unavailable. `SwmrWriter` and the readers intentionally take no lock: SWMR is single-writer-by-contract and built for concurrent reads, and `std`'s whole-file lock would block readers (fatally on Windows, where locks are mandatory) ([#73](https://github.com/stephenberry/hdf5-pure/issues/73)).
- A gallery of runnable, self-checking examples in `examples/` covering the core API: write/read, generic element I/O, groups & attributes, compression, compound & complex types, ndarray, in-place editing, repack, SWMR, and file-space strategy. Run any with `cargo run --example <name>` ([#54](https://github.com/stephenberry/hdf5-pure/issues/54)).

### Changed

- 32-bit / WASM hardening: the chunked-data and MATLAB matrix readers now return an error instead of silently truncating when a file-derived dimension or element count exceeds the platform's pointer width. Every remaining narrowing `as` cast in the library is now either a checked conversion or carries an `#[expect(…, reason = "…")]` justifying why it is bounded, enforced by a hard deny of the narrowing-cast lints on a 32-bit CI target — replacing the previous count-based ratchet, which a new cast could slip past by removing an unrelated one ([#72](https://github.com/stephenberry/hdf5-pure/issues/72)).

### Fixed

- Read dense groups and dense attributes whose link/attribute names are very long (stored as fractal-heap "huge" objects); previously failed with `InvalidObjectHeaderVersion` ([#63](https://github.com/stephenberry/hdf5-pure/pull/63)).
- `EditSession` now clears the superblock's write/SWMR consistency flag on commit instead of preserving whatever the source file carried, so editing a file an interrupted SWMR writer left flagged produces a cleanly-closed file the reference C library can reopen ([#73](https://github.com/stephenberry/hdf5-pure/issues/73)).

## [0.14.0] - 2026-06-15

Completes free-space management ([#21](https://github.com/stephenberry/hdf5-pure/issues/21)) and closes several interoperability gaps with the reference HDF5 C library. Additive minor bump.

### Added

- File-space strategy on the file-creation property list: `FileBuilder::with_file_space_strategy` and `with_file_space_page_size`, read back with `File::file_space_strategy()` / `File::file_space_info()` ([#55](https://github.com/stephenberry/hdf5-pure/pull/55)). Mirrors `H5Pset_file_space_strategy` / `H5Pset_file_space_page_size`.
- `File::persisted_free_space()` reads the on-disk free-space managers of a file written with `persist = true` ([#56](https://github.com/stephenberry/hdf5-pure/pull/56)).
- `EditSession` persists free space across reopen: it seeds its free list from the on-disk managers and writes it back on commit, so freed space is reused by later sessions instead of leaking ([#58](https://github.com/stephenberry/hdf5-pure/pull/58)).

### Fixed

- The reference C library can now add objects to files this crate writes (group headers were missing a Group Info message, which the C library requires before inserting a link) ([#59](https://github.com/stephenberry/hdf5-pure/pull/59)).
- Read large dense groups whose fractal heap grows a multi-row root indirect block (~150+ links) ([#60](https://github.com/stephenberry/hdf5-pure/pull/60)).
- Read large dense groups whose name index is a 3-or-more-level v2 B-tree (~26k+ links) ([#62](https://github.com/stephenberry/hdf5-pure/pull/62)).

## [0.13.0] - 2026-06-15

Free-space management ([#21](https://github.com/stephenberry/hdf5-pure/issues/21), [#45](https://github.com/stephenberry/hdf5-pure/pull/45)).

### Added

- `EditSession` now reuses space freed by earlier commits and truncates the file when free space reaches the end, so add/delete churn stays bounded instead of growing the file every commit.
- Whole-file `repack(src, dst, &RepackOptions)` rewrites a file with no dead space, optionally dropping objects (`RepackOptions::new().drop_path("grp/old")`). It refuses with `Error::RepackUnsupported` rather than silently degrade anything it cannot reproduce exactly (e.g. variable-length, reference, or lossy-filtered data).

### Fixed

- `Datatype::serialize` produced empty bytes for the time, bit-field, and opaque datatype classes, corrupting any datatype message that used one of them ([#45](https://github.com/stephenberry/hdf5-pure/pull/45)).

## [0.12.1] - 2026-06-10

Internal robustness and tests ([#26](https://github.com/stephenberry/hdf5-pure/issues/26)); no public API or on-disk-format change.

### Added

- Property-based tests for the write/read roundtrip and parser robustness ([#44](https://github.com/stephenberry/hdf5-pure/pull/44)).
- A Miri CI job covering the crate's only non-trivial `unsafe` (the aligned chunk buffer) ([#43](https://github.com/stephenberry/hdf5-pure/pull/43)).

### Changed

- Internal cleanup of B-tree v1 size arithmetic into named helpers ([#42](https://github.com/stephenberry/hdf5-pure/pull/42)).

## [0.12.0] - 2026-06-10

### Added

- `EditSession` edits object headers that span multiple chunks (e.g. objects carrying several attributes) ([#32](https://github.com/stephenberry/hdf5-pure/issues/32)).
- `EditSession` edits version 0/1 (symbol-table) files in place — the default format from the C library and h5py ([#32](https://github.com/stephenberry/hdf5-pure/issues/32)). Adding and deleting is supported; copying a version-1 object is not.

### Fixed

- `EditSession::commit` now `fsync`s appended data before repointing the root, making its "repoint last" crash-safety guarantee real ([#32](https://github.com/stephenberry/hdf5-pure/issues/32)).

## [0.11.0] - 2026-06-09

### Added

- In-place file editing via `EditSession` ([#32](https://github.com/stephenberry/hdf5-pure/issues/32)): `open(path)`, then `create_dataset` / `create_group` / `delete` / `copy`, applied by `commit()`. Changes are appended and the superblock repointed last, so cost scales with the edit, not the file size, and a failed commit leaves the file valid. It refuses with `Error::EditUnsupported` cases it cannot reproduce faithfully (userblocks, pre-1.10 formats, dense storage, chunked/compressed new datasets). Freed space is not reclaimed (see [#21](https://github.com/stephenberry/hdf5-pure/issues/21)).
- File inspection: `is_hdf5(path)` / `is_hdf5_bytes(&[u8])`, `File::file_size()`, and `File::libver_bound()` (new `LibVer` enum) ([#32](https://github.com/stephenberry/hdf5-pure/issues/32)).
- `FileBuilder::with_libver_bounds(low, high)`, mirroring `H5Pset_libver_bounds` ([#32](https://github.com/stephenberry/hdf5-pure/issues/32)). This crate writes one format (the 1.10+ version-3 superblock), so it acts as a compatibility guard: `finish()` fails with `FormatError::LibverBoundsUnsatisfiable` if the bounds exclude that format.

## [0.10.0] - 2026-06-09

### Changed

- **Breaking:** the public API is now a curated surface; internal format modules are `pub(crate)` ([#33](https://github.com/stephenberry/hdf5-pure/issues/33)). Code using the documented reader/writer/builder API is unaffected; code reaching into internal module paths (e.g. `hdf5_pure::object_header::…`) must stop.

### Added

- `Dataset::verify_provenance` (feature `provenance`) checks a dataset against the `_provenance_sha256` hash written by `with_provenance`.

### Removed

- The `fast-checksum` feature and its `crc32fast` dependency — it gated unused CRC32 code (HDF5 uses lookup3). Drop it from any feature list that named it.
- Several internal subsystems that were never wired into the reader or writer.

## [0.9.0] - 2026-06-08

### Removed

- **Breaking:** `parallel_read::decompress_chunks_parallel` and `decompress_chunks_sequential` — public but unused ([#33](https://github.com/stephenberry/hdf5-pure/issues/33)). Reader/writer code is unaffected. CI now runs `cargo-semver-checks` to catch unintended API changes.

## [0.8.0] - 2026-06-05

### Added

- Streaming reads for files too large to buffer: `File::open_streaming(path)` reads metadata and chunks on demand instead of loading the whole file ([#27](https://github.com/stephenberry/hdf5-pure/issues/27)). Streams contiguous, compact, and all chunk-index layouts; limited to latest-format groups, and attribute reading is not yet supported. The buffered `File::open` path is unchanged.
- 32-bit and bare-metal robustness ([#27](https://github.com/stephenberry/hdf5-pure/issues/27)): file offsets/lengths that do not fit the platform now error (`ValueTooLargeForPlatform` / `OffsetOverflow`) instead of truncating. CI runs the suite on 32-bit (i686) and builds for `thumbv7em-none-eabi` `no_std`.
- N-dimensional array I/O via the optional `ndarray` feature ([#24](https://github.com/stephenberry/hdf5-pure/issues/24)): `DatasetBuilder::with_ndarray` and `Dataset::read_array` / `read_array_dyn`. Off by default; implies `std`.

### Changed

- Writing a dataset whose shape disagrees with the data now fails with `FormatError::ShapeDataMismatch` instead of producing an unreadable file.

### Removed

- The `mmap` feature and its `memmap2` dependency — declared but never implemented ([#24](https://github.com/stephenberry/hdf5-pure/issues/24)). Drop it if you named it.

## [0.7.0] - 2026-06-03

### Added

- SWMR (single-writer / multiple-reader) support for 1-D, unlimited, Extensible-Array-indexed datasets ([#17](https://github.com/stephenberry/hdf5-pure/issues/17)):
  - `File::open_swmr(path)` plus `File::refresh()` re-read data appended by a concurrent writer.
  - `SwmrWriter::open(path)` appends chunks in place (`append_i32` / `append_f64` / `append_raw`), ordered so a reader or a crashed writer only ever sees a consistent prefix. `close()` clears the SWMR flag; `clear_swmr_flag(path)` recovers a file left flagged by a crash.
  - Limited to unfiltered, chunk-aligned, single-unlimited-dimension datasets; unsupported targets are rejected with `Error::SwmrAppendUnsupported`. Requires `std`.

### Changed

- **Breaking:** `Error` and `FormatError` are now `#[non_exhaustive]`; `match` over them needs a wildcard arm. Future variant additions are now non-breaking.

### Fixed

- Extensible Array chunk index: reading more than 20 chunks returned wrong data and writing more than 244 silently dropped the excess ([#17](https://github.com/stephenberry/hdf5-pure/issues/17)).

## [0.6.0]

### Added

- Scale-offset filter (HDF5 filter id 6), read and write, via `.with_scale_offset(mode)` ([#13](https://github.com/stephenberry/hdf5-pure/issues/13)). Integer mode is lossless; float decimal-scaling is lossy. Datasets compressed with it by other tools now decode instead of failing with `UnsupportedFilter(6)`.

## [0.5.1]

### Fixed

- Chunked datasets indexed by a Fixed Array now use the paged data block layout above the page size (>1024 chunks at the default), and the reader decodes them; previously such files were written corrupt and rejected on read ([#14](https://github.com/stephenberry/hdf5-pure/issues/14)).

## [0.5.0]

### Added

- serde roundtrip for `Matrix<Complex64>` / `Matrix<Complex32>`, including empty matrices (which previously lost their complex class).
- Sealed `mat::MatElement` trait, so an unsupported element type is a compile error rather than a silent class loss.

### Changed

- **Breaking:** `Matrix<T>` serde now requires `T: MatElement` instead of `T: 'static`. Such uses previously produced malformed MAT files at runtime.
- The MAT deserializer flattens 1×N and N×1 values to a 1-D sequence in `deserialize_any` (matching `deserialize_seq`).
- Numeric/complex readers preserve 1×N / N×1 shape at the value layer; any flattening happens at the serde level.

[Unreleased]: https://github.com/stephenberry/hdf5-pure/compare/v0.23.2...HEAD
[0.23.2]: https://github.com/stephenberry/hdf5-pure/compare/v0.23.1...v0.23.2
[0.23.1]: https://github.com/stephenberry/hdf5-pure/compare/v0.23.0...v0.23.1
[0.23.0]: https://github.com/stephenberry/hdf5-pure/compare/v0.22.0...v0.23.0
[0.22.0]: https://github.com/stephenberry/hdf5-pure/compare/v0.21.2...v0.22.0
[0.21.2]: https://github.com/stephenberry/hdf5-pure/compare/v0.21.1...v0.21.2
[0.21.1]: https://github.com/stephenberry/hdf5-pure/compare/v0.21.0...v0.21.1
[0.21.0]: https://github.com/stephenberry/hdf5-pure/compare/v0.20.1...v0.21.0
[0.20.1]: https://github.com/stephenberry/hdf5-pure/compare/v0.20.0...v0.20.1
[0.20.0]: https://github.com/stephenberry/hdf5-pure/compare/v0.19.0...v0.20.0
[0.19.0]: https://github.com/stephenberry/hdf5-pure/compare/v0.18.0...v0.19.0
[0.18.0]: https://github.com/stephenberry/hdf5-pure/compare/v0.17.0...v0.18.0
[0.17.0]: https://github.com/stephenberry/hdf5-pure/compare/v0.16.0...v0.17.0
[0.16.0]: https://github.com/stephenberry/hdf5-pure/compare/v0.15.0...v0.16.0
[0.15.0]: https://github.com/stephenberry/hdf5-pure/compare/v0.14.0...v0.15.0
[0.14.0]: https://github.com/stephenberry/hdf5-pure/compare/v0.13.0...v0.14.0
[0.13.0]: https://github.com/stephenberry/hdf5-pure/compare/v0.12.1...v0.13.0
[0.12.1]: https://github.com/stephenberry/hdf5-pure/compare/v0.12.0...v0.12.1
[0.12.0]: https://github.com/stephenberry/hdf5-pure/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/stephenberry/hdf5-pure/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/stephenberry/hdf5-pure/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/stephenberry/hdf5-pure/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/stephenberry/hdf5-pure/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/stephenberry/hdf5-pure/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/stephenberry/hdf5-pure/releases/tag/v0.6.0
[0.5.1]: https://github.com/stephenberry/hdf5-pure/releases/tag/v0.5.1
[0.5.0]: https://github.com/stephenberry/hdf5-pure/releases/tag/v0.5.0
