# Changelog

All notable changes to this crate are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this crate follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) under Cargo's pre-1.0 conventions: a `0.x.0` bump may be breaking, `0.x.y` is not.

## [Unreleased]

### Added

- SWMR (single-writer / multiple-reader) support for one-dimensional, unlimited, Extensible-Array-indexed datasets ([#17](https://github.com/stephenberry/hdf5-pure/issues/17)).
  - **Refreshing reader.** `File::open_swmr(path)` retains a live file handle and `File::refresh()` re-reads data appended by a concurrent writer, then re-parses the superblock; datasets fetched after a refresh observe the appended chunks and the extended dimension. Interoperates as a consumer of files written by the reference C library or h5py in SWMR mode.
  - **In-place append writer.** `SwmrWriter::open(path)` opens an existing latest-format file (created by this crate, the C library, or h5py) and `append_i32` / `append_f64` / `append_raw` append chunks to an unlimited dataset in place: each chunk is written at end-of-file, its address stored into the next free chunk-index slot, the index grown by appending new data blocks / super blocks only when a block boundary is crossed (never relocating existing data), and the dataspace dimension, array-header counts, and superblock end-of-file patched, with writes issued child-before-parent and `fsync`-barriered between phases. Verified against the reference C library in both directions (hdf5-pure appends → C reads; C creates → hdf5-pure appends → C reads) across the inline, direct-block, and super-block ranges.
  - Current subset: unfiltered datasets, chunk-aligned appends, a single unlimited dimension, no userblock, and up to the paged-data-block boundary (131056 chunks on the unlimited axis); larger appends, filtered datasets, and the concurrent-reader superblock flag handshake are not yet implemented and return a clear error rather than producing an inconsistent file. SWMR read/append require the `std` filesystem and are unavailable on the in-memory/WASM path.

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

[Unreleased]: https://github.com/stephenberry/hdf5-pure/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/stephenberry/hdf5-pure/releases/tag/v0.6.0
[0.5.1]: https://github.com/stephenberry/hdf5-pure/releases/tag/v0.5.1
[0.5.0]: https://github.com/stephenberry/hdf5-pure/releases/tag/v0.5.0
