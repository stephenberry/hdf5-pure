# Changelog

All notable changes to this crate are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this crate follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) under Cargo's pre-1.0 conventions: a `0.x.0` bump may be breaking, `0.x.y` is not.

## [Unreleased]

### Added

- Repack now copies a chunked dataset's compressed chunks **verbatim** instead of decompressing and re-compressing them. Each chunk is read one at a time and written byte-for-byte with the source's filter-pipeline message carried through unchanged, so the repacked chunks are identical to the source's. This eliminates the per-dataset decompression blowup that made repack fail on large compressed files ([#82](https://github.com/stephenberry/hdf5-pure/issues/82)) and the decompress→recompress round-trip ([#84](https://github.com/stephenberry/hdf5-pure/issues/84)), and because nothing is ever decoded, **lossy** filters now survive byte-exact: float D-scale scale-offset, ZFP, SZIP, and even filters this crate cannot itself apply are all preserved ([#85](https://github.com/stephenberry/hdf5-pure/issues/85)). Verified by reading the repacked datasets back identically and by the reference C library decoding repacked deflate- and scale-offset-filtered datasets. The verbatim path applies to a dataset whose chunk grid is fully allocated (the common case); a *sparse* chunked dataset (with unallocated chunk-grid holes) still falls back to the read-raw + re-encode path, which requires a lossless filter, and a contiguous/compact filtered dataset likewise re-encodes — a lossy filter on either is still refused by name. Note this does not make repack fully out-of-core: the whole-file writer still buffers the output and `File::open` loads the source into memory; what is removed is the per-dataset decompression and the lossy-filter loss.
- Variable-length string dataset writing and repack support. `DatasetBuilder::with_vlen_strings(&[&str])` writes a contiguous variable-length UTF-8 string dataset (1D, or ND via `with_shape`), staging each element in a fresh global heap and patching the reference addresses once the data layout is known. The base element type matches the reference C library's `H5Tvlen_create(H5T_C_S1)` shape byte-for-byte, so the C library and h5py read these datasets back into `VarLenUnicode`/`VarLenAscii` without a conversion-path error. Repack now round-trips contiguous/compact VL-string datasets (both the `is_string: true` shape and the MATLAB VLEN-of-1-byte-ASCII-string shape) instead of refusing them: each element's exact heap bytes are read and re-staged, preserving charset, padding, the null-vs-empty distinction, embedded NULs, and non-UTF-8 payloads. Chunked, filtered, or resizable VL-string datasets are still refused by name (their element references live inside compressed chunks written before the global heap addresses are known), as are non-string variable-length datatypes ([#83](https://github.com/stephenberry/hdf5-pure/issues/83)).
- In-place overwrite of dataset values: `EditSession::write_dataset(path)` replaces an existing contiguous or compact dataset's values (HDF5's `H5Dwrite` whole-dataset write), returning the same `DatasetBuilder` as `create_dataset`. The replacement must match the on-disk datatype and shape (it is a value write, not a reshape or retype) — both are compared structurally so a file written by this crate, the reference C library, or h5py is accepted regardless of harmless encoding differences. A same-length contiguous overwrite is the cheapest edit possible: the new bytes are written straight into the existing data block, so no object header is rewritten and the superblock root is not flipped, and the synced data write is the commit's linearization point. When the length differs (for example filling a dataset the C library created but never wrote, whose data address was undefined) or the dataset is compact, the header relocates like an addition and the parent group's link is patched. Chunked and filtered datasets are refused by name, as is a relocating overwrite of a dataset reachable through more than one hard link (only the one named link could be repointed); the result is read back faithfully by the reference C library ([#79](https://github.com/stephenberry/hdf5-pure/issues/79)).
- Cross-file object copy: `EditSession::copy_from` copies a dataset or a whole group subtree out of a *separate* open file (a `File` reader) into the file being edited — the cross-file form of HDF5's `H5Ocopy`, alongside the existing same-file `EditSession::copy`. The source object's headers and data are reproduced byte-for-byte and linked into the destination like any other addition, so the copy reads back identically (verified against the source) and the destination is unchanged until commit, preserving the editor's crash-safety guarantee. The source subtree is read and validated eagerly (the `File` borrow need not outlive the call), so `copy_from` returns a `Result`. Because the copy is verbatim, the cross-file path refuses — by name, before any byte is written — anything whose stored bytes embed a *source-file* absolute address that cannot be translated into another file: variable-length and reference datasets and attributes, and any shared (committed datatype, or SOHM-shared dataspace / fill value / filter pipeline) header message, all of which the same-file copy instead keeps valid by sharing the source file's global heaps and objects. The source must be a buffered file (`File::open` / `File::from_bytes`, not `open_streaming`) using 8-byte offsets and no userblock ([#78](https://github.com/stephenberry/hdf5-pure/issues/78)).
- Free-space reclaim for chunked datasets on in-place delete: deleting a chunked dataset (or a group whose subtree contains one) now returns its chunk data blocks *and* its chunk-index structure to the session free list, where a later commit reuses them and the file is truncated when the freed run reaches end-of-file — previously only contiguous datasets, group headers, and superseded headers were reclaimed, and a chunked dataset's storage was left behind as dead bytes. Covers every chunk index this crate and the reference C library write: single-chunk, implicit, fixed array, extensible array, and the version 1 B-tree of older/foreign files. Reclaim stays conservative: an index it cannot enumerate exhaustively (a version 2 B-tree) or any computed span that falls outside the file or overlaps another leaves the dataset's bytes in place rather than risk freeing a region still in use, and variable-length global-heap data (whose collections can be shared between objects) is still never reclaimed. The result is read back faithfully by the reference C library ([#77](https://github.com/stephenberry/hdf5-pure/issues/77)).
- In-place add of chunked, filtered, and extensible datasets: `EditSession::create_dataset` now accepts chunked storage (`with_chunks`), any filter the whole-file writer supports (`with_deflate`, `with_shuffle`, `with_fletcher32`, `with_scale_offset`, `with_zfp`), and extensible (optionally unlimited) dimensions (`with_maxshape`) — previously only contiguous, unfiltered datasets could be added in place. The chunk data, index, and filter pipeline are produced by the same builder the whole-file writer uses and appended at end-of-file, so the added dataset's object header is byte-identical to a freshly written one and is read back faithfully by the reference C library. The filter pipeline is validated before any byte is written, and a chunked dataset's append leaves the prior root intact until the superblock is repointed last, preserving the editor's crash-safety guarantee ([#76](https://github.com/stephenberry/hdf5-pure/issues/76)).

### Fixed

- Reading an object from a *filtered* fractal managed heap is now refused cleanly (`FormatError::UnsupportedFilteredHeapObject`) instead of silently misparsing it. The indirect-block child-pointer walk advanced past each direct-block entry by a hardcoded 4 bytes when the heap stored I/O-filter-encoded blocks, where the real layout inserts a `filtered_size` (length-size bytes) plus a 4-byte `filter_mask` after every child address; the wrong stride misaligned every subsequent child address and would have returned wrong bytes. Filtered managed heaps are now refused at the start of the managed-object read paths (buffered and streaming) — matching the existing refusal on the huge-object paths — so a filtered heap is never read incorrectly. Reading filtered fractal heaps remains unimplemented and is now consistently a typed refusal rather than a partial, incorrect parse ([#80](https://github.com/stephenberry/hdf5-pure/issues/80)).
- Object copy (`EditSession::copy` and `copy_from`) no longer refuses an object that carries an Attribute Info message whose fractal-heap address is *undefined*. The reference C library and h5py emit that message — alongside compact, inline attributes — for latest-format objects to record attribute creation order; the editor mistook its mere presence for dense (fractal-heap) attribute storage and refused the copy, so copying a typical C-library- or h5py-written *attributed* object failed. The copy path now inspects the heap address (mirroring its existing handling of dense link storage) and refuses only genuine dense storage — more than the compact threshold of 8 attributes — so such objects copy, byte-for-byte verbatim and read back by the reference C library, on both the same-file and cross-file paths ([#78](https://github.com/stephenberry/hdf5-pure/issues/78)).
- In-place delete now reclaims an object's storage only when the link being removed is its *last* hard link. HDF5 objects can have several hard links; the editor previously freed a deleted object's blocks unconditionally, so deleting one of several hard links returned still-referenced storage to the free list and corrupted the surviving link once those bytes were reused (silently, unreadable by this crate and the reference C library alike). The editor now counts every hard link in the file before reclaiming and leaves a multiply-linked object's storage in place (a safe leak the repack path still compacts), fixing the corruption for contiguous datasets and preventing it for the chunked datasets reclaimed by [#77](https://github.com/stephenberry/hdf5-pure/issues/77). As defense in depth, the spans a commit frees are de-duplicated and checked for mutual disjointness and in-bounds before reaching the free list ([#77](https://github.com/stephenberry/hdf5-pure/issues/77)).
- Malformed chunk geometry is now refused up front by both `FileBuilder` and `EditSession` instead of panicking deep in the chunk splitter (an out-of-bounds index or a divide-by-zero) or producing an unreadable dataset: chunk dimensions whose rank disagrees with the shape, a zero chunk dimension, a maximum shape of the wrong rank or smaller than the current shape, and chunking a scalar dataset all return a descriptive error (`FormatError::InvalidChunkGeometry` from the writer, `Error::EditUnsupported` from the editor). An absurd shape whose element count overflows `u64` is likewise reported rather than panicking a debug build. Zero-element extensible datasets (e.g. shape `[0]` with an unlimited maximum) remain valid ([#76](https://github.com/stephenberry/hdf5-pure/issues/76)).

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

[Unreleased]: https://github.com/stephenberry/hdf5-pure/compare/v0.15.0...HEAD
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
