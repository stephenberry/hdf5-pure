# Changelog

All notable changes to this crate are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this crate follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) under Cargo's pre-1.0 conventions: a `0.x.0` bump may be breaking, `0.x.y` is not.

## [Unreleased]

## [0.10.0] - 2026-06-09

### Changed

- **The public API is now a curated surface; the internal format modules are no longer exposed** ([#33](https://github.com/stephenberry/hdf5-pure/issues/33)). Previously almost every module was `pub` at the crate root (`btree_v1`, `fractal_heap`, `object_header`, `chunked_write`, `data_read`, …), so internal on-disk-format machinery — and any dead code in it — was part of the published API surface. That is the root cause of the 0.9.0 dead-`pub fn` slip. Those modules are now `pub(crate)`, and the supported public API is the curated set re-exported at the crate root: `File`, `Dataset`, `Group`, `FileBuilder`, `SwmrWriter`, `DatasetBuilder`, `GroupBuilder`, `FinishedGroup`, `CompoundTypeBuilder`, `EnumTypeBuilder`, the `make_*_type` constructors, `Datatype`, `ScaleOffset`, `AttrValue`, `DType`, `H5Element`, `Error`, `FormatError`, and the `mat` module. **Code using the documented reader/writer/builder API is unaffected**; code that reached into internal module paths (e.g. `hdf5_pure::object_header::…`) must stop doing so, hence the `0.x.0` bump. The byte-level ZFP crosscheck, which exercised the internal codec entry points, moved from `tests/` to an in-crate test.

### Removed

- The `fast-checksum` feature and its `crc32fast` dependency. The feature gated a CRC32 implementation that nothing ever called — HDF5 uses Bob Jenkins' lookup3 hash for all metadata checksums, never CRC32 — so enabling it was a no-op. Drop it from any feature list that named it.
- Several internal subsystems that were built but never wired into the reader or writer: the SOHM (shared object header message) table parser, the `metadata_index` / independent-parallel-write module, the `group_info` message parser, the adaptive "sweep" chunk-read path and its cache instrumentation, the batch object-header writer, the unadopted `source::Cursor` abstraction, the compound/enum/object-reference/region-reference/flat-array data readers, and the unexposed provenance `verify_dataset` / `VerifyResult` (provenance writing is unaffected). None were reachable through the public API.

### Notes

- Encapsulating the internals surfaced a large amount of code that compiled only because it was `pub`. Each item was audited: genuinely abandoned subsystems were removed (see above), while struct fields that are parsed from the on-disk format for completeness but not yet read are kept with localized `#[allow(dead_code)]` and a note. `dead_code` enforcement is restored for `std` builds; `no_std` builds allow it because the high-level reader/writer entry points that consume the parsing machinery are `std`-gated and absent there.

## [0.9.0] - 2026-06-08

### Removed

- `parallel_read::decompress_chunks_parallel` and `parallel_read::decompress_chunks_sequential` ([#33](https://github.com/stephenberry/hdf5-pure/issues/33)). Both were public but never called: `decompress_chunks_parallel` was a legacy `par_iter` decompression path superseded by the lane-partitioned implementation that the reader actually uses, and `decompress_chunks_sequential` duplicated a fallback that `chunked_read` already implements inline. They were reachable only with the `parallel` feature enabled and have no realistic use outside the crate. Removing them is a breaking change to that feature's public surface, hence the `0.x.0` bump; code using the reader and writer APIs is unaffected. CI now runs [`cargo-semver-checks`](https://github.com/obi1kenobi/cargo-semver-checks) against the last published release so an unintended public-API change like this is caught before it ships.

## [0.8.0] - 2026-06-05

### Added

- Streaming reads for files too large to buffer, via `File::open_streaming(path)` ([#27](https://github.com/stephenberry/hdf5-pure/issues/27)). The reader fetches metadata and dataset chunks from the file on demand through a `Read + Seek` source instead of loading the whole file into memory, so a host can read a multi-gigabyte file without an equivalent allocation; the motivating case is a 32-bit target (where the file exceeds the address space) reading data produced by the reference C library. The reading API is unchanged and only the backing store differs: the in-memory `File::open` / `File::from_bytes` / `File::open_swmr` paths are byte- and performance-identical to before (they keep the chunk cache and full v1/v2 group support), while the streaming backend reads each region on demand. Contiguous, compact, and all chunk-index layouts (B-tree v1, fixed array, and extensible array) stream. Two limits apply to the streaming backend that the in-memory path does not have: only latest-format (v2) groups resolve along a path (a v1 symbol-table group is rejected), and attribute reading is not yet supported. Verified end-to-end against the buffered reader on a 32-bit (i686) target under emulation.
- 32-bit and bare-metal robustness ([#27](https://github.com/stephenberry/hdf5-pure/issues/27)). File-derived offset and length values that previously narrowed to `usize` with a silent truncation on 32-bit platforms are now converted through checked helpers that return a `FormatError` (`ValueTooLargeForPlatform` or `OffsetOverflow`) instead of truncating, so a 64-bit value that cannot be represented on the host is reported rather than used to read from the wrong place. CI now cross-compiles and runs the test suite on a 32-bit (i686) target under QEMU, lints it with the `cast_possible_truncation` and `cast_possible_wrap` lints denied (a ratchet prevents new truncating casts from creeping in), and builds the crate for a bare-metal `thumbv7em-none-eabi` `no_std` target.
- N-dimensional array I/O via the optional `ndarray` feature ([#24](https://github.com/stephenberry/hdf5-pure/issues/24)). `DatasetBuilder::with_ndarray` writes any `ndarray` array or view, inferring the dataset shape and datatype from it, and `Dataset::read_array` / `Dataset::read_array_dyn` read a dataset back into a typed `ndarray::Array` (fixed rank) or `ArrayD` (runtime rank). Data is stored row-major (C order), so files are byte-compatible with the reference HDF5 library (verified by cross-check); non-standard-layout inputs (transposed, Fortran-order, or strided views) are repacked to row-major on write, and chunking/compression chain as usual. The feature is off by default and implies `std`, keeping the core build dependency-free and WASM/`no_std`-clean. The supported element types (`f32`, `f64`, and the signed and unsigned 8/16/32/64-bit integers) are described by the new sealed `H5Element` trait.

### Changed

- Writing a dataset whose shape disagrees with the supplied data now fails fast with `FormatError::ShapeDataMismatch` (the element count implied by the shape must match the data length) instead of silently producing a file that cannot be read back.

### Removed

- The `mmap` feature and its `memmap2` dependency, which were declared but never implemented — enabling `mmap` did nothing ([#24](https://github.com/stephenberry/hdf5-pure/issues/24)). Downstream code that named the feature should drop it.

## [0.7.0] - 2026-06-03

### Added

- SWMR (single-writer / multiple-reader) support for one-dimensional, unlimited, Extensible-Array-indexed datasets ([#17](https://github.com/stephenberry/hdf5-pure/issues/17)).
  - **Refreshing reader.** `File::open_swmr(path)` retains a live file handle and `File::refresh()` re-reads data appended by a concurrent writer, then re-parses the superblock; datasets fetched after a refresh observe the appended chunks and the extended dimension. Interoperates as a consumer of files written by the reference C library or h5py in SWMR mode.
  - **In-place append writer.** `SwmrWriter::open(path)` opens an existing latest-format file (created by this crate, the C library, or h5py), sets the superblock SWMR-write flag, and `append_i32` / `append_f64` / `append_raw` append chunks to an unlimited dataset in place: each chunk is written at end-of-file, its address stored into the next free chunk-index slot, the index grown by appending new data blocks, super blocks, and paged data blocks only when a block boundary is crossed (never relocating existing data), and the dataspace dimension, array-header counts, and superblock end-of-file patched. Writes are ordered child-before-parent (raw data → superblock end-of-file → chunk-index count → dataspace dimension) with an `fsync` barrier after each phase, so an interrupted append or a concurrent reader only ever observes a consistent prefix; the reader bounds chunk reads by the published count and dimension to ignore slots a writer wrote ahead of the commit. Growth is unbounded (super blocks and paged data blocks past 131060 chunks are allocated incrementally). `close()` clears the SWMR flag; `SwmrWriter::clear_swmr_flag(path)` recovers a file left flagged by a crashed writer. Reopening a file after a writer crashed mid-append rolls forward from the last committed length (the dataspace dimension), overwriting any chunks a previous writer wrote but never committed, so a recovered append never leaves a gap or resurfaces uncommitted data.
  - **Verified interop.** Cross-checked against the reference C library in both directions (hdf5-pure appends → C reads; C creates → hdf5-pure appends → C reads) across the inline, direct-block, super-block, and paged ranges, and end-to-end with h5py: h5py opens the file with `swmr=True` and reads hdf5-pure's appended data while the writer holds it open.
  - Current subset: unfiltered datasets, chunk-aligned appends, a single unlimited dimension, and no userblock. Unsupported targets (a filtered dataset, a non-rank-1 or non-Extensible-Array dataset, or a non-latest-format v0/v1 superblock) are rejected up front with a specific `Error::SwmrAppendUnsupported(reason)` and the file is left unmodified, rather than producing an inconsistent file. An append that fails partway (including an underlying I/O error) never publishes the new length, so a reader still sees the prior consistent prefix. SWMR read/append require the `std` filesystem and are unavailable on the in-memory/WASM path.

### Changed

- The public `Error` and `FormatError` enums are now `#[non_exhaustive]`. Downstream code that matches them must include a wildcard (`_ =>`) arm. This is a one-time break that makes every future variant addition non-breaking; constructing and `matches!`-testing existing variants is unaffected.

### Fixed

- Extensible Array chunk index (used for chunked datasets with one unlimited dimension): reading a dataset with more than 20 chunks along the unlimited axis silently returned wrong data, and writing more than 244 chunks silently dropped the excess. The reader's data-block size progression did not match the HDF5 format (it diverged past the inline elements plus the first data block), and the writer never emitted super blocks or paged data blocks. Reader and writer now share a single block-size geometry derived from the array's creation parameters; the writer emits super blocks and, above 131060 chunks, paged data blocks; and the array header statistics match the reference C library byte-for-byte. Verified against the reference C HDF5 library in both directions across the inline, direct-block, super-block, and paged ranges. Found while implementing SWMR support ([#17](https://github.com/stephenberry/hdf5-pure/issues/17)).

## [0.6.0]

### Added

- Scale-offset filter (HDF5 filter id 6), read and write. Integer mode (`ScaleOffset::Integer`) is lossless; floating-point decimal-scaling mode (`ScaleOffset::FloatDScale`) is lossy to the requested number of decimal digits. Enable on a dataset with `.with_scale_offset(mode)`; it may be combined with deflate. Datasets compressed with scale-offset by other tools (the reference C library, h5py, MATLAB) now decode instead of failing with `UnsupportedFilter(6)`. Verified both directions against the reference C HDF5 library. ([#13](https://github.com/stephenberry/hdf5-pure/issues/13))

## [0.5.1]

### Fixed

- Chunked datasets indexed by a Fixed Array now use the paged data block layout when the chunk count exceeds the page size (`2^max_nelmts_bits`, i.e. more than 1024 chunks at the default). Previously the writer always emitted a flat data block while still advertising the paged page size, producing files that a spec-compliant reader rejects as corrupt. The reader likewise now decodes paged Fixed Array data blocks (page-init bitmap plus fixed-stride, individually checksummed pages) instead of returning `paged Fixed Array data blocks not yet supported`. Verified both directions against the reference C HDF5 library. ([#14](https://github.com/stephenberry/hdf5-pure/issues/14))

## [0.5.0]

### Added

- `Matrix<Complex64>` and `Matrix<Complex32>` serde roundtrip support, including the empty (`0×0`, `0×N`, `N×0`) path. Empty complex matrices now write a compound `{real, imag}` HDF5 dataset on disk and preserve their shape across roundtrip; previously they collapsed to an `f64`-empty dataset and lost the complex class.
- Sealed `mat::MatElement` trait. Implemented for `f32`, `f64`, all signed and unsigned integer primitives (`i8` through `i64`, `u8` through `u64`), `bool`, `Complex32`, and `Complex64`. Adding a new element type to `Matrix<T>` now requires a corresponding `MatElement` impl plus matching dispatch in the MAT (de)serializer. Missing dispatch surfaces as a compile error rather than silent class loss on the empty-matrix path.

### Changed

- **Breaking.** `Matrix<T>`'s `Serialize` / `Deserialize` impls now require `T: MatElement` instead of `T: 'static`. Downstream code parameterizing `Matrix<T>` with a non-numeric `T` (anything outside the impl set above) will no longer compile. Such uses already produced malformed MAT files at runtime, so the new bound converts a runtime failure into a compile error.
- The MAT serde deserializer now flattens 1×N and N×1 `Matrix` / `ComplexMatrix` values to a 1-D sequence inside `deserialize_any`, matching the existing behavior of `deserialize_seq`. This means untagged enums, `serde::de::Content` roundtrips, and custom `Visitor` impls that previously discriminated on the 2-D rows-of-rows shape when one axis was 1 will now see a flat sequence. Values with both axes greater than 1 still surface as a 2-D rows-of-rows.
- Numeric / complex dataset readers no longer collapse a 1×N or N×1 dataset to a flat vector at the value layer. Shape is preserved through `MatValue::Matrix` / `ComplexMatrix`, and any flattening for `Vec<T>` callers happens at the serde-deserializer level (above). Direct consumers of `pub(crate)` value APIs are unaffected; this is an internal cleanup that fixes column-vector roundtrip ambiguity.

[Unreleased]: https://github.com/stephenberry/hdf5-pure/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/stephenberry/hdf5-pure/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/stephenberry/hdf5-pure/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/stephenberry/hdf5-pure/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/stephenberry/hdf5-pure/releases/tag/v0.6.0
[0.5.1]: https://github.com/stephenberry/hdf5-pure/releases/tag/v0.5.1
[0.5.0]: https://github.com/stephenberry/hdf5-pure/releases/tag/v0.5.0
