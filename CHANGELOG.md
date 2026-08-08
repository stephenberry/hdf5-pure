# Changelog

All notable changes to this crate are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this crate follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) under Cargo's pre-1.0 conventions: a `0.x.0` bump may be breaking, `0.x.y` is not.

## [Unreleased]

### Added

- `FileBuilder::with_libver_bounds` selects the on-disk format rather than only validating it: an upper bound of `LibVer::V18` writes the HDF5 1.8 format, and anything reaching 1.10 writes the 1.10 one ([#247](https://github.com/stephenberry/hdf5-pure/pull/247)).
- `FormatError::LibverTooOldForContent` reports content the requested bound cannot express, rather than silently upgrading the file — a chunked, filtered, or resizable dataset, or any file-space setting ([#247](https://github.com/stephenberry/hdf5-pure/pull/247)).
- `FileAccessProperties::with_libver_bounds` holds an editing session to a format, so `File::open_rw` refuses an addition that would make the file need a newer library instead of making it silently ([#247](https://github.com/stephenberry/hdf5-pure/pull/247)).
- `RepackOptions::with_libver_bounds` makes a repack's output format a guarantee. Without it, `repack` now carries the source file's format forward, upgrading only where the content leaves no choice — it used to rewrite every file in the 1.10 format ([#247](https://github.com/stephenberry/hdf5-pure/pull/247)).
- `LibVer::WRITER_OLDEST` names the oldest format the writer produces ([#247](https://github.com/stephenberry/hdf5-pure/pull/247)).
- `mat::Options::libver` sets the newest HDF5 format a `.mat` file may use, defaulting to `LibVer::V18`. `mat::MatError::CompressionNeedsNewerFormat` reports the one combination that cannot hold ([#247](https://github.com/stephenberry/hdf5-pure/pull/247)).
- `Group::attr_datatypes` and `Dataset::attr_datatypes` give an attribute's on-disk `Datatype`, which `attrs()` normalizes away — the stored width, and the `enum[FALSE, TRUE]` that marks an h5py `np.bool_` as boolean rather than a one-byte integer. Every attribute is reported, including the ones `attrs()` omits for having no `AttrValue`, though an attribute's rank stays unexposed ([#253](https://github.com/stephenberry/hdf5-pure/pull/253)).

### Changed

- **Breaking:** MAT files are written in the HDF5 1.8 format by default, so MATLAB can `load` them; MATLAB used HDF5 1.8.12 before R2021b and cannot open a version 3 superblock. Set `mat::Options::libver` to `LibVer::V110` for the previous format, which compression requires ([#247](https://github.com/stephenberry/hdf5-pure/pull/247)).
- **Breaking:** `mat::Options::default` uses `EmptyMarkerEncoding::DataAsDims`, matching what MATLAB and `matio` write, so an empty value reads back as empty under a plain `isempty` ([#247](https://github.com/stephenberry/hdf5-pure/pull/247)).
- **Breaking:** `LibVer::WRITER_OUTPUT` is now `LibVer::WRITER_DEFAULT`, since the writer no longer emits a single format. Its value is unchanged ([#247](https://github.com/stephenberry/hdf5-pure/pull/247)).
- `FileBuilder::with_create_properties` resets the properties its argument does not carry, so a bound set before the call no longer decides the format of a file whose property list names no version ([#247](https://github.com/stephenberry/hdf5-pure/pull/247)).
- `FileBuilder::write` creates the destination only once the writer has bytes for it, so a refused build leaves an existing file at that path untouched ([#247](https://github.com/stephenberry/hdf5-pure/pull/247)).
- An edit session writes a contiguous dataset's data-layout message in the format of the file it opened, so a `.mat` file edited through `File::open_rw` stays readable by MATLAB ([#247](https://github.com/stephenberry/hdf5-pure/pull/247)).

### Fixed

- An object header holding compact attributes declares how many it has, so `H5Oget_info().num_attrs` agrees with iteration instead of reporting zero — an `h5repack` round trip used to strip every `MATLAB_*` attribute from a `.mat` file without warning ([#247](https://github.com/stephenberry/hdf5-pure/pull/247)).
- A file with a userblock reads whole when it holds an object-header continuation block or dense link storage, which the C library writes and this crate does not; `File::open` used to fail outright on such a file ([#247](https://github.com/stephenberry/hdf5-pure/pull/247)).
- An empty MAT value carries `MATLAB_class` and `MATLAB_empty` and nothing else, matching MATLAB, and both emitters agree on its dimensions — including for an empty `Matrix`, which one of them wrote as a plain zero-element dataset ([#247](https://github.com/stephenberry/hdf5-pure/pull/247)).
- `repack` keeps each attribute's own datatype and shape instead of rebuilding it from an `AttrValue`, which widened every integer and float to 64 bits, turned a variable-length string into a fixed-width one, and flattened a rank-2 attribute to rank 1. Enumeration, compound, bit-field and opaque attributes are now carried across rather than refused; a reference attribute still is ([#241](https://github.com/stephenberry/hdf5-pure/issues/241)).
- An enumeration attribute reaches the caller, decoded through its integer base type as enum dataset data already is; `attrs()` skipped it before, so every `np.bool_` attribute in an h5py-written file — stored as `enum[FALSE, TRUE]` — went missing without a trace. The member names have no `AttrValue` to live in, so the codes are what survives ([#248](https://github.com/stephenberry/hdf5-pure/pull/248)).
- A committed (`H5Tcommit`) datatype resolves to the type it names, so `Dataset::datatype`, `Dataset::read_*`, `attrs()` and `attr_datatypes()` report a named type instead of the zero-width time type its stored reference used to decode as — the shape netCDF-4 and h5py write for a user-defined type ([#254](https://github.com/stephenberry/hdf5-pure/issues/254)).
- `repack` refuses a committed datatype rather than writing an output the C library could not read *any* attributes from, and the new `FormatError::UnsupportedSohmReference` refuses a message stored in the shared-message (SOHM) heap instead of following its heap id as an address ([#254](https://github.com/stephenberry/hdf5-pure/issues/254)).

## [0.33.0] - 2026-08-02

A file whose superblock marks it as held by a writer is refused rather than opened: `File::open`, `open_streaming`, `open_rw`, `open_swmr_writer` and `repack` report the new `Error::FileMarkedInUse`, which is the check `H5Fopen` makes of the same byte — a file a crashed SWMR writer left flagged used to open, and `open_rw` used to edit it in place under a writer the file still recorded ([#245](https://github.com/stephenberry/hdf5-pure/issues/245)). `File::open_swmr` follows such a file instead of refusing it, since that pairing is what the flag exists for, and `File::from_bytes` does not consult the byte at all, so a caller holding the bytes can still read a flagged file on a read-only mount, where the `File::clear_swmr_flag` recovery (the `h5clear -s` equivalent) cannot get the write access it needs. The check applies to version-3 superblocks, which is where the C library applies it, and `open_swmr_writer` now requires one for the same reason libhdf5 does. Two smaller changes come with it: a read-write open validates the superblock before it builds its backing, so refusing a mirrored file no longer reads the whole file first, and a version-1 superblock's status flags and chunk B-tree K are read from the offsets the C library writes them to rather than swapped. Files written by earlier versions still read.

### Added

- `Error::FileMarkedInUse` reports an open refused by the superblock's status-flags byte. Unlike `Error::FileLocked` it outlives the process that set it, so it means a writer is active *or* one exited without closing the file ([#245](https://github.com/stephenberry/hdf5-pure/issues/245)).

### Changed

- **Breaking:** `File::open`, `File::open_streaming`, `File::open_rw`, `File::open_swmr_writer` and `repack` refuse a file whose superblock marks it as held by a writer, matching `H5Fopen`; a file left flagged by a crashed SWMR writer used to open, and `open_rw` used to edit it in place. `File::open_swmr` follows such a file as before, `File::from_bytes` does not check, and `File::clear_swmr_flag` recovers a stale flag ([#245](https://github.com/stephenberry/hdf5-pure/issues/245)).
- **Breaking:** `File::open_swmr_writer` requires a version-3 superblock, as the C library does, rather than version 2 or 3 ([#245](https://github.com/stephenberry/hdf5-pure/issues/245)).
- A read-write open that refuses a file no longer reads the file first: `File::open_rw` validates the superblock through the handle and builds its backing only once nothing can refuse it, so refusing a mirrored file costs a few bounded reads rather than a whole-file copy ([#245](https://github.com/stephenberry/hdf5-pure/issues/245)).

### Fixed

- A version-1 superblock's status flags and chunk B-tree K are read from the offsets the C library writes them to; the two were swapped, so `File::superblock()` reported a v1 file's `indexed_storage_internal_node_k` as its `consistency_flags` and vice versa ([#245](https://github.com/stephenberry/hdf5-pure/issues/245)).

## [0.32.0] - 2026-07-31

`Display` now covers the types that describe what a file holds — `AttrValue`, `Datatype` and its component enums, `MessageType`, `Layout`, `ChunkIndex` and `Filter` — so a message quoting one reads as HDF5 rather than as a Rust value ([#242](https://github.com/stephenberry/hdf5-pure/pull/242)). A name the file records is escaped and truncated wherever it is written, and a member list elided past sixteen. `DType::Other` carries the `Datatype` itself, which is the only view a caller gets of a type nested in a compound field or an array base ([#244](https://github.com/stephenberry/hdf5-pure/pull/244)).

### Added

- `Display` for `AttrValue`, `Datatype`, `DatatypeByteOrder`, `StringPadding`, `CharacterSet`, `ReferenceType`, `MessageType`, `Layout`, `ChunkIndex` and `Filter`. `AttrValue` writes the value — `1.5`, `"metres"`, `[1, 2, 3]` — and elides an array past eight elements, reporting how many it dropped ([#242](https://github.com/stephenberry/hdf5-pure/pull/242)).
- `AttrValue::type_name` gives the name of the type a value holds, such as `f64` or `ascii_string[]`. It names every variant, so a caller that matched on this `#[non_exhaustive]` enum and reached its `_` arm can still report what it received ([#242](https://github.com/stephenberry/hdf5-pure/pull/242)).

### Changed

- **Breaking:** `DType::Other` carries the `Datatype` rather than a string describing it, so a type nested in a compound field or an array base can be matched on. It writes as `other(opaque[3] "rgb")` ([#244](https://github.com/stephenberry/hdf5-pure/pull/244)).
- **Breaking:** `DType::Array` writes its shape as `array<f32, 2x3>` rather than `array<f32, [2, 3]>` ([#242](https://github.com/stephenberry/hdf5-pure/pull/242)).
- **Breaking:** a name a file records — a compound member's, an enum label's, a filter's — is escaped and truncated wherever it is written, and a member list is elided past sixteen ([#242](https://github.com/stephenberry/hdf5-pure/pull/242)).
- **Breaking:** `Error::MissingMessage` names the message — `missing required message: data layout` — instead of its Rust variant, and the unrecognized-chunk-index error reports the raw index-type byte ([#242](https://github.com/stephenberry/hdf5-pure/pull/242)).

## [0.31.0] - 2026-07-30

An attribute now reads back as the `AttrValue` variant it was written from: the dataspace kind decides scalar against array, so a one-element array stays an array, and the charset selects the `Ascii` variants, so `MATLAB_class` reads as `AsciiString` and `MATLAB_fields` as `VarLenAsciiArray` ([#239](https://github.com/stephenberry/hdf5-pure/pull/239)). That fidelity means several variants can carry one logical value, so read through the new accessors — `AttrValue::as_str`, `as_strings`, `as_i64`, `as_u64`, `as_f64`, `to_i64s`, `to_u64s`, `to_f64s` — each of which spans every variant that can hold the shape it names and applies its range rule per element ([#238](https://github.com/stephenberry/hdf5-pure/pull/238)). Two data-correctness fixes come with it: an unsigned array reads as the new `AttrValue::U64Array` rather than an `I64Array` of reinterpreted bits, so a value above `i64::MAX` no longer reads back negative, and `repack` stops re-encoding the attributes it copies — a fixed-width ASCII string used to come out UTF-8 and a variable-length array fixed-width, which is the encoding MATLAB and matio require. Separately, an attribute holding an empty string is written with a one-byte-wide string datatype instead of a zero-size one, which libhdf5 rejects while iterating an object's attributes: a single empty-string attribute made every attribute on that object unreadable to the C library ([#240](https://github.com/stephenberry/hdf5-pure/pull/240)). Widths, true variable-length strings, dataspace rank and fixed-string padding are still not recovered on read, and are tracked in [#241](https://github.com/stephenberry/hdf5-pure/issues/241). Files written by earlier versions still read.

### Added

- `AttrValue::as_str`, `as_strings`, `as_i64`, `as_u64`, `as_f64`, `to_i64s`, `to_u64s` and `to_f64s` read an attribute value without matching on its variant. Each spans every variant that can carry the shape asked for — both string charsets and all four integer widths, scalar or one-element array — and applies its range rule per element, so a value that does not fit reports `None` rather than a wrapped number. The prefix states the cost: `as_*` borrows or copies, `to_*` allocates ([#238](https://github.com/stephenberry/hdf5-pure/pull/238)).
- `AttrValue::U64Array` writes an unsigned 64-bit array attribute, which `I64Array` could not represent above `i64::MAX` ([#238](https://github.com/stephenberry/hdf5-pure/pull/238)).

### Changed

- **Breaking:** an attribute reads back as the `AttrValue` variant it was written from: the dataspace kind decides scalar against array, so a one-element array stays an array, and the charset selects the `Ascii` variants. `MATLAB_class` reads as `AsciiString` rather than `String` and `MATLAB_fields` as `VarLenAsciiArray` rather than `StringArray`. Read values through `AttrValue::as_str`/`as_strings`/`as_i64`/`to_i64s`, which span every shape. Widths are still widened, and a true variable-length string reads as the fixed-width variant of its charset ([#239](https://github.com/stephenberry/hdf5-pure/pull/239)).
- **Breaking:** an unsigned integer array reads as `AttrValue::U64Array` instead of an `I64Array` holding reinterpreted bits, so a value above `i64::MAX` no longer reads back negative ([#239](https://github.com/stephenberry/hdf5-pure/pull/239)).

### Fixed

- An attribute holding an empty string is written with a one-byte-wide string datatype rather than a zero-size one, which libhdf5 rejects while iterating an object's attributes — a single empty-string attribute made every attribute on that object unreadable to the C library ([#240](https://github.com/stephenberry/hdf5-pure/pull/240)).
- `repack` no longer re-encodes an attribute it copies: a variable-length ASCII array stays variable-length, and a fixed-width ASCII string keeps its charset, where both previously came out as UTF-8 fixed-width ([#239](https://github.com/stephenberry/hdf5-pure/pull/239)).
- A MAT file honors `MATLAB_class` and `MATLAB_empty` whichever integer width, charset, or one-element shape its writer chose, rather than reporting an unexpected attribute type or reading the flag as absent ([#238](https://github.com/stephenberry/hdf5-pure/pull/238), [#239](https://github.com/stephenberry/hdf5-pure/pull/239)).

## [0.30.0] - 2026-07-30

`DatasetBuilder::with_lzf` writes h5py's LZF filter (id 32000), a fast lossless compressor h5py reads without any plugin installed, and LZF datasets — including h5py-written ones — can be read, edited in place, and repacked ([#231](https://github.com/stephenberry/hdf5-pure/pull/231)). The MAT serde writer honors `Options::null_policy`, which it previously ignored: `None`, `()`, a unit struct, and `Value::Null` now write MATLAB `struct([])` rather than dropping the field, so MATLAB code can reference it unconditionally — at the cost that a Rust reader relying on `#[serde(default)]` for a non-`Option` field now finds the field present. `NullPolicy::Omit` writes the previous output. Two new options join it: `Options::unit_variant_encoding` writes a fieldless enum variant as its name or as its declaration index, and `Options::empty_sequence_policy` picks `[]` or `{}` for a sequence that turned out to be empty ([#232](https://github.com/stephenberry/hdf5-pure/pull/232)). That writer also collects a flat numeric or complex sequence packed, one element wide, instead of one 56-byte value per element — serializing a `Vec<f64>` cost 7x its own size and a `Vec<ComplexI16>` 14x, both now about 1x, with the same bytes out — and the empty-value paths its two emitters take now agree with each other. On the filter side, requesting a filter that another would displace is refused instead of silently dropped, `FormatError::DecompressionError` is removed in favor of the `FilterError` that shuffle, scale-offset and LZF already reported, and decoding a chunk reserves memory against what its own stream could expand to rather than against a size the file merely declares ([#233](https://github.com/stephenberry/hdf5-pure/issues/233)). Files written by earlier versions still read.

### Added

- `DatasetBuilder::with_lzf` writes h5py's LZF filter (id 32000), a fast lossless compressor h5py reads without any plugin, and LZF datasets — including h5py-written ones — can be read, edited in place, and repacked; combining LZF with deflate is refused ([#231](https://github.com/stephenberry/hdf5-pure/pull/231)).
- `Options::unit_variant_encoding` picks whether a fieldless enum variant is written as its name (`UnitVariantEncoding::Name`, the default and previous behavior) or as its declaration index in a `uint32` (`UnitVariantEncoding::Index`). Serde hands the serializer both, so either is reachable; `Index` suits a reader that already expects the integer, but an index cannot be interpreted without the schema that fixes the ordering, so `Name` is the better default. The index is serde's, counted from zero, so an explicit discriminant (`enum E { A = 5 }`) is not the number that reaches the file.
- `Options::empty_sequence_policy` picks the MATLAB class of a sequence that turned out to be empty: `EmptySequencePolicy::DoubleArray` (the default and previous behavior) writes `[]`, `Cell` writes `{}`. An empty `serialize_seq` carries no element type, so there is nothing to infer the class from; `Cell` is right when the sequence would have held structs.
- `NullPolicy::Omit` selects the pre-0.30 serde behavior of dropping the field.

### Changed

- **Breaking:** requesting a filter that another would displace is refused instead of silently dropped. `with_zfp` alongside `with_shuffle`, `with_deflate` or `with_lzf`, and `with_scale_offset` alongside `with_shuffle`, now fail with a filter error naming both; only the scale-offset/ZFP clash did before ([#233](https://github.com/stephenberry/hdf5-pure/issues/233)).
- **Breaking:** `FormatError::DecompressionError` is removed. A deflate stream that fails to decode reports `FormatError::FilterError`, which shuffle, scale-offset and LZF already used, so "this chunk did not decode" is one match arm rather than two ([#233](https://github.com/stephenberry/hdf5-pure/issues/233)).
- The serde writer collects a flat numeric or complex sequence packed, one element wide, instead of one 56-byte `MatValue` per element. A `Vec<f64>` cost 7x its own size while being serialized and a `Vec<ComplexI16>` cost 14x; both now cost about 1x. A sequence whose elements do not all agree spills to the previous per-element form at the point they diverge, so cell arrays and matrices built from equal-length rows are unaffected. Same bytes out.
- The serde writer gives an empty cell array MATLAB shape `[0, 1]` rather than `[0, 0]`, which is the `[n, 1]` rule it already applies to a non-empty one, and matches what an empty BEVE array converts to. `isempty` holds either way; `size(x, 2)` no longer changes as a list empties out.
- **Breaking:** the serde writer honors `Options::null_policy`, which it previously ignored. `None`, `()`, a unit struct, and `serde_json::Value::Null` now write MATLAB `struct([])` by default rather than dropping the field, so `isfield` reports true and MATLAB code can reference the field unconditionally with `isempty(fieldnames(x))`. Set `NullPolicy::Omit` for the old output.

  Note the reader-side consequence, which cuts both ways. A field that is present no longer needs `#[serde(default)]`, but a reader that *relies* on `#[serde(default)]` for a non-`Option` field now fails on newly written files: the field is present with a struct value rather than missing, so the default is never consulted. `struct([])` reads into `Option<T>`, `Vec<T>`, `serde_json::Value` and `()`, and reports a type error for a bare scalar, `String`, struct or map. Give such a field type `Option<T>`, or write with `NullPolicy::Omit`.

### Fixed

- Decoding a chunk no longer reserves memory against a size the file merely declares: deflate and LZF reserve at most what their own stream could expand to, and scale-offset refuses a chunk claiming more bytes than the chunk holds. A small file declaring an enormous chunk previously drove the allocation before anything had checked the claim ([#233](https://github.com/stephenberry/hdf5-pure/issues/233)).
- An empty `Matrix<T>` serializes under `EmptySequencePolicy::Cell`. The policy was applied where the `Matrix` sentinel's own `data` field is lowered, so an empty matrix became a cell array and the sentinel handler had no vector left to recover the element class from, failing with "Matrix::data must be a Vec<T>, got cell array" for every empty numeric and complex matrix. The policy is now pinned for that field, since it is internal plumbing rather than a sequence the caller wrote; a caller's own empty sequence still follows it.
- An `Options` persisted by an earlier version still deserializes. The two fields added this release carried no serde default, so an older serialized `Options` failed with a missing-field error. The default is now declared on the struct, so future additions stay loadable without anyone having to remember.
- `NullPolicy::Error` refuses a null at the file root, which it previously did not. The root serializer never consulted `null_policy` at all, so a root `None`, `()` or `Value::Null` wrote a zero-variable file even under the one policy whose whole purpose is to report nulls. It now routes through the same lowering as every other slot, and reports the same error. The other two policies are unchanged and now documented rather than incidental: the root names no slot, so a null there is an empty variable *namespace*, and both still write a valid file with no variables, byte-identical to what an empty root map or a fieldless struct writes. Note such a file does not read back as `None`, since the deserializer presents the root as a struct.
- A fieldless enum variant reads back from either encoding. The deserializer accepted only a name, so a file written under the new `UnitVariantEncoding::Index` could not be read by this crate at all; it now resolves an index too, from whatever numeric class carries it, which also lets an index typed in MATLAB (a `double`) be read.
- An empty marker has the same element type from both serde emitters. `to_bytes` wrote a `uint64` zero-element dataset for `struct([])` where `to_bytes_with_options` under the same default options wrote `uint8`, so the two produced different files for one value. Both now write `uint64`, matching the reference library's own empty marker, and the choice is made in one place rather than duplicated. This also changes an empty cell array written through `MatBuilder` under `EmptyMarkerEncoding::ZeroElement` from `uint8` to `uint64`; the class attributes that identify it are unchanged, and the reference library reads either.

## [0.29.0] - 2026-07-28

Dense attributes take on the reference library's geometry: both indexes are multi-level B-trees of 512-byte nodes, and the heap is a doubling table of direct blocks reached through indirect blocks, so a large attribute set grows by adding blocks rather than rounding one up to a power of two. The two attribute-count ceilings go with it, along with the errors that reported them, and the remaining heap-size error now bounds the heap's address space rather than one direct block — the release's only breaking changes ([#195](https://github.com/stephenberry/hdf5-pure/issues/195)). MAT v7.3 files no longer have to be held in memory to be written: `MatBuilder::finish_to` assembles onto any `io::Write`, `MatBuilder::write_blocks` stages a numeric array whose bytes a `DataProducer` supplies one block at a time, `mat::to_file` streams rather than buffering, and `FileBuilder::with_userblock_content` keeps a wrapper format's header reachable on those paths ([#226](https://github.com/stephenberry/hdf5-pure/pull/226)). Chunked datasets are written back to back instead of padded to the host's cache line, which aligned nothing measurable and made the same dataset larger on `aarch64` than on `x86_64` ([#227](https://github.com/stephenberry/hdf5-pure/issues/227)). One soundness fix: a MATLAB matrix shape whose `rows * cols` wraps `usize` is refused at every entry point, where the wrapped product could previously match a short data vector and the writer's transpose then wrote past its allocation ([#230](https://github.com/stephenberry/hdf5-pure/pull/230)). Two dense attributes whose names hash alike are also now indexed in the order the reference library searches, which every earlier version got wrong ([#225](https://github.com/stephenberry/hdf5-pure/issues/225)). Files written by earlier versions still read.

### Added

- An object carries any number of dense (fractal-heap) attributes: both the name index and the huge-object index are now multi-level B-trees of fixed 512-byte nodes, matching what the reference C library emits, instead of one leaf grown to fit ([#195](https://github.com/stephenberry/hdf5-pure/issues/195)).
- Dense attributes are held in a doubling table of direct blocks reached through indirect blocks, the same heap geometry the reference C library uses, so a large attribute set no longer rounds its storage up to a power of two ([#195](https://github.com/stephenberry/hdf5-pure/issues/195)).
- `MatBuilder::finish_to` assembles a `.mat` onto any `io::Write` (`MatBuilder::write` onto a path), as do `mat::to_writer` / `mat::to_writer_with_options` for the serde entry points. Byte-for-byte what the buffered calls produce, on a sink that need not be seekable ([#226](https://github.com/stephenberry/hdf5-pure/pull/226)).
- `MatBuilder::write_blocks` stages a numeric array whose bytes a `DataProducer` supplies one `Block` at a time during the write, so a dataset larger than memory can be written. Uncompressed only, since the layout needs the region's exact size before it writes anything ([#226](https://github.com/stephenberry/hdf5-pure/pull/226)).
- `FileBuilder::with_userblock_content` makes the userblock part of the file the writer emits, so a wrapper format's header survives the streaming output paths that leave nothing to patch afterwards ([#226](https://github.com/stephenberry/hdf5-pure/pull/226)).

### Changed

- `FileBuilder::with_userblock` now refuses a size the format does not define — it must be zero or a power of two of at least 512 — with the new `FormatError::InvalidUserblockSize`, instead of writing a file whose superblock nothing can find ([#226](https://github.com/stephenberry/hdf5-pure/pull/226)).
- **Breaking:** `FormatError::TooManyDenseAttributes` and `FormatError::TooManyHugeDenseAttributes` are removed along with the 61,680- and 43,690-attribute limits that produced them ([#195](https://github.com/stephenberry/hdf5-pure/issues/195)).
- **Breaking:** `FormatError::DenseAttributeHeapTooLarge` now carries only `limit`, and bounds the heap's 40-bit address space rather than a single 2 GiB direct block ([#195](https://github.com/stephenberry/hdf5-pure/issues/195)).
- Every dense attribute set has different bytes: its name index is a tree of 512-byte nodes, and its attributes sit in a doubling table whose blocks start at 1 KiB and grow by adding blocks rather than by rounding one up to a power of two. Files written by earlier versions still read ([#195](https://github.com/stephenberry/hdf5-pure/issues/195)).
- `mat::to_file` and `mat::to_file_with_options` stream to disk rather than building the whole file in memory first. Same bytes ([#226](https://github.com/stephenberry/hdf5-pure/pull/226)).
- Chunks are written back to back instead of padded to the host's cache line, so a chunked dataset no longer occupies more space on `aarch64` than on `x86_64`, and chunk placement no longer varies by target. Files written by earlier versions still read ([#227](https://github.com/stephenberry/hdf5-pure/issues/227)).

### Fixed

- A MATLAB matrix shape whose `rows * cols` wraps `usize` is refused everywhere it can enter: `Matrix::from_row_major` and `Matrix::zeros` panic, while the serde and file-reading paths return an error. Previously the wrapped product could match a short data vector, and the writer's transpose then wrote past its allocation ([#230](https://github.com/stephenberry/hdf5-pure/pull/230)).
- Two dense attributes whose names hash alike are indexed in name order rather than insertion order, so the reference C library can open both by name; written the other way round one of the pair was unfindable by name, while iteration still reported both. A file written by 0.28.0 or earlier is corrected by `repack` ([#225](https://github.com/stephenberry/hdf5-pure/issues/225)).

## [0.28.0] - 2026-07-28

`File::open_rw` picks its own backing. A latest-format file with no userblock is edited in bounded memory rather than through a whole-file mirror, and the mirror is now the fallback for the files the bounded engine cannot edit rather than the default for everything. Nothing about a file's space strategy decides which open a caller reaches for any more, so `File::open_rw_bounded` is deprecated: it survives only as the strict default, now expressible as `MemoryStrategy::Bounded` on `FileAccessProperties`, and `File::edit_backing` reports which backend an open actually resolved to. Two guarantees that used to be silently ignored are refused before any work happens: the SWMR writer will not accept a `Bounded` it cannot honor, and `File::create_with_options` checks a creation/access pair up front rather than leaving a file on disk and returning the reopen's error.

### Added

- `FileAccessProperties::with_memory_strategy` and `MemoryStrategy` say how much memory a read-write open may spend holding the file: `Bounded` refuses a file the bounded engine cannot edit rather than mirroring it, `Auto` falls back to the mirror, and `Mirrored` always mirrors. `File::edit_backing` reports which backend an open resolved to, as an `EditBacking` ([#198](https://github.com/stephenberry/hdf5-pure/issues/198)).

### Changed

- `File::open_rw` now edits a latest-format file with no userblock in **bounded memory** instead of building a whole-file mirror, and falls back to the mirror only for a file the bounded engine cannot edit. Nothing about a file's space strategy decides which open a caller reaches for any more. A large `Dataset::append` is applied in whole-chunk batches on the bounded backing, so a crash mid-call leaves a valid shorter dataset; pass `MemoryStrategy::Mirrored` for the previous unconditional mirror ([#198](https://github.com/stephenberry/hdf5-pure/issues/198)).
- `File::open_rw` refuses a paged file without persisted free space at open rather than at commit, since neither backing can edit one. This includes a paged file with a userblock, whose free-space managers go unseeded for the same reason ([#198](https://github.com/stephenberry/hdf5-pure/issues/198)).
- `File::open_rw` now applies `FileAccessProperties::with_metadata_cache`, which the whole-file mirror ignored, and its reads are served from the file rather than from a snapshot taken at open — visible only to a session sharing a file with a lock-free writer ([#198](https://github.com/stephenberry/hdf5-pure/issues/198)).
- `File::open_swmr_writer_with_options` refuses an explicit `MemoryStrategy::Bounded` instead of silently mirroring; this writer always mirrors, and `Auto` or unset is satisfied by that ([#198](https://github.com/stephenberry/hdf5-pure/issues/198)).
- `File::create_with_options` refuses a creation/access pair it could not then reopen — a paged file with `persist = false`, or a userblock under `MemoryStrategy::Bounded` — before writing anything, instead of leaving a file on disk and returning an error ([#198](https://github.com/stephenberry/hdf5-pure/issues/198)).
- A userblock that is not a whole number of file-space pages now reports `FormatError::UserblockNotPageAligned` naming both sizes, rather than an `InvalidFileSpacePageSize` that called a valid page size invalid ([#198](https://github.com/stephenberry/hdf5-pure/issues/198)).

### Deprecated

- `File::open_rw_bounded` and `open_rw_bounded_with_options`: `File::open_rw` now picks the bounded engine on its own, so these survive only as the strict `MemoryStrategy::Bounded` default. Pass that strategy to `File::open_rw_with_options` to keep the refusal ([#198](https://github.com/stephenberry/hdf5-pure/issues/198)).

## [0.27.0] - 2026-07-27

The two read-write engines converge. `File::open_rw` now commits staged edits to a genuine paged file, and `File::open_rw_bounded` offers the full staged edit surface — `Dataset::write`, attribute edits, `create_*`/`delete`, `copy`, `space_accounting` — while holding only what a commit is building rather than a whole-file mirror. Neither the file's internal space strategy nor the kind of edit being made decides which open a caller reaches for, and `Dataset::append` grows a free-space-persisting file from either one. `Error::BoundedStagedUnsupported` is gone along with the refusals that returned it, the single breaking change here. Separately, MAT v7.3 complex arrays gain integer components across the serde, `Matrix<T>`, and `MatBuilder` surfaces, so a capture that samples as 16-bit integer pairs stores four bytes per sample instead of eight; three defects on the complex path are fixed with it, one of which changes the stored shape of a 1-D complex array written through `to_bytes_with_options`.

### Added

- `File::open_rw` commits staged edits to a genuine paged file (`H5F_FSPACE_STRATEGY_PAGE`), through a commit that keeps each page homogeneous and rewrites the per-page-type free-space managers, so the full edit surface is no longer limited to `File::open_rw_bounded`'s appends. A paged file is still refused unless it persists its free space and has no userblock ([#198](https://github.com/stephenberry/hdf5-pure/issues/198)).
- `File::open_rw_bounded` offers the full staged edit surface — `Dataset::write`, attribute edits, `create_*`/`delete`, `copy`, `commit`, `space_accounting` — at bounded memory: a commit holds only what it is building rather than a whole-file mirror. It still requires a latest-format file with 8-byte offsets and no userblock ([#198](https://github.com/stephenberry/hdf5-pure/issues/198)).
- `Dataset::append` grows a file that persists its free space, including a paged one, from `File::open_rw` as well as `File::open_rw_bounded`; the on-disk free-space managers are re-homed when the file is closed ([#198](https://github.com/stephenberry/hdf5-pure/issues/198)).
- A `Dataset` reached by object reference can append on either read-write open, not only `File::open_rw_bounded`. It is refused once the session stages or commits an edit, because a commit can move the object header the handle names ([#198](https://github.com/stephenberry/hdf5-pure/issues/198)).
- MAT v7.3 complex arrays with integer components: `mat::ComplexI8`/`I16`/`I32`/`I64` and the `ComplexU*` counterparts join `Complex64`/`Complex32` across the serde, `Matrix<T>`, and `MatBuilder::write_complex_*` surfaces, so a capture that samples as 16-bit integer pairs stores four bytes per sample instead of eight. Components are never converted between widths: an `int16` complex dataset deserializes into `ComplexI16` and nothing else, in either direction.

### Changed

- Dropping a read-write `File` without `close` now re-homes the on-disk free-space managers of a persisting file and flushes, matching what `close` does; previously only `File::open_rw_bounded` handles did this. Staged edits are still discarded on drop ([#198](https://github.com/stephenberry/hdf5-pure/issues/198)).
- **Breaking:** `Error::BoundedStagedUnsupported` is removed, along with the refusals that returned it ([#198](https://github.com/stephenberry/hdf5-pure/issues/198)).
- A MAT complex vector written through `to_bytes_with_options` now takes the configured `OneDimensionalMode` like every other 1-D array; it was always a MATLAB row vector before, so existing callers of that path get columns under the default and their stored shape changes.
- A MAT complex dataset whose `MATLAB_class` disagrees with its `{real, imag}` compound is refused instead of decoded, including a `{imag, real}` compound that used to read back swapped.

### Fixed
- A one-element MAT complex array deserializes into a `Vec<Complex*>`, matching the allowance the real numeric path already makes for a one-element numeric array.
- An empty MAT complex array of an integer class reads back as an empty complex array of that class rather than as an untyped empty vector.

## [0.26.0] - 2026-07-27

Attributes lose their size ceiling: one too large for an object-header message selects fractal-heap storage on its own, and one too large even for a managed heap object becomes a *huge* object, so `FormatError::DenseAttributeTooLarge` is gone rather than refusing a shape the reference library writes ([#195](https://github.com/stephenberry/hdf5-pure/issues/195)). Three defects on that path are fixed with it: a variable-length attribute stored in a fractal heap silently lost its values, dense attributes were unreadable in a file with a userblock, and reading many of them was quadratic in their number ([#214](https://github.com/stephenberry/hdf5-pure/pull/214), [#195](https://github.com/stephenberry/hdf5-pure/issues/195)). The property-list types are renamed to say what they stand in for — `FileAccessProperties`, `FileCreateProperties`, `DatasetAccessProperties` — and checking each one's settings against the official group pages caught two properties documented under the wrong class ([#198](https://github.com/stephenberry/hdf5-pure/issues/198)). Breaking, but every break is a one-line call-site edit, and deprecated aliases keep 0.25.0 code compiling for this cycle.

### Added

- An attribute of any size is written rather than refused: one too large for an object-header message selects fractal-heap storage on its own, and one too large for a managed heap object becomes a *huge* object. A name, datatype, or dataspace longer than the 2-byte field describing it is still refused, as the new `FormatError::AttributeFieldTooLong` ([#195](https://github.com/stephenberry/hdf5-pure/issues/195)).
- `FormatError::TooManyHugeDenseAttributes` names the one bound the new huge-object path adds, on how many such attributes a single object may carry ([#195](https://github.com/stephenberry/hdf5-pure/issues/195)).
- `FormatError::UnexpectedHugeObjectBTree` refuses a fractal heap whose huge-objects B-tree is not the record layout this reader decodes, instead of reading an object ID out of another field's bytes ([#195](https://github.com/stephenberry/hdf5-pure/issues/195)).

### Changed

- **Breaking:** `FileAccessOptions`, `FileCreateOptions`, and `DatasetAccessOptions` are renamed `FileAccessProperties`, `FileCreateProperties`, and `DatasetAccessProperties`, so that a type standing in for a whole HDF5 property list says so in its name; `FileBuilder::with_create_options` and `File::access_options` follow suit. Deprecated aliases under the old names keep 0.25.0 code compiling for this cycle, and `H5Pset_libver_bounds` is now documented as the file-access property it is ([#198](https://github.com/stephenberry/hdf5-pure/issues/198)).

### Fixed

- Reading an object with many huge dense attributes now parses the heap's huge-object index once per walk instead of once per attribute, so the read is no longer quadratic in their number (1,600 such attributes drop from ~164 ms to ~75 ms) ([#195](https://github.com/stephenberry/hdf5-pure/issues/195)).
- A variable-length attribute stored in a fractal heap keeps its values. Written into the heap before its global-heap references had addresses, it read back with its values lost, silently ([#214](https://github.com/stephenberry/hdf5-pure/pull/214)).
- Dense (fractal-heap) attributes are readable in a file with a userblock, through `File::open`, `File::open_streaming`, `repack` and `copy` alike. The heap address was taken as an absolute file offset rather than one relative to the base address, so the read failed on a file the reference C library reads correctly ([#214](https://github.com/stephenberry/hdf5-pure/pull/214)).

### Removed

- **Breaking:** `FormatError::DenseAttributeTooLarge` is gone, as no attribute size is refused any more ([#195](https://github.com/stephenberry/hdf5-pure/issues/195)).

## [0.25.0] - 2026-07-26

An API-consolidation release. File properties are now reusable values rather than scattered function variants: one `FileAccessOptions` (the `fapl` analogue) carries cache budgets and the locking policy to every open, and the new `FileCreateOptions` (the `fcpl` analogue) lets a file layout be defined once and applied to any write, including through `File::create_with_options`. Two long-standing gaps fell out of that work — the read-write mirror backend was silently discarding access options, and the bounded backend always locked regardless of policy. Public types the HDF5 format will keep growing are now `#[non_exhaustive]`, so a future datatype class, reference kind, or MATLAB class is an additive change instead of a breaking one, and a test guards each seal against silent removal. Enumerations can finally be built over any integer base type. Two defects from 0.24.0 are fixed: `repack` corrupting a dataset whose datatype contains a variable-length or reference member, and a hang when reading a file from inside a builder closure. The release carries a number of breaking changes, all listed below; most are one-line call-site edits.

### Added

- `FileCreateOptions` collects the file-creation properties (userblock, file-space strategy and page size, library-version bounds) into one reusable value — the `fcpl` analogue — applied with `FileBuilder::with_create_options` or the new `File::create_with_options`, which mirrors `H5Fcreate(name, flags, fcpl, fapl)` and is the first way to reach creation properties from the owned-handle path ([#205](https://github.com/stephenberry/hdf5-pure/issues/205)).
- `FileAccessOptions::with_locking` carries the file-locking policy, and `File::open_rw_with_options` / `open_swmr_writer_with_options` accept the options, so one `fapl` value now serves every open. The mirror backend honors the configured chunk cache (it previously discarded access options), and `open_rw_bounded*` honors the locking policy instead of always locking ([#204](https://github.com/stephenberry/hdf5-pure/issues/204)).
- `EnumTypeBuilder::with_base` builds an enumeration over any integer base type (not just `i32`/`u8`), with `raw_value` for a member given as its raw little-endian bytes and `i64_value` for wider integers ([#208](https://github.com/stephenberry/hdf5-pure/issues/208)).
- Chunked, filtered, and resizable variable-length datasets now write: `DatasetBuilder::with_vlen_strings` accepts `with_chunks`, `with_deflate`, and `with_maxshape`, and `repack` reproduces such a dataset with its chunk geometry, filters, and unlimited dimension intact. Adding one to an existing file through the in-place edit engine is still refused ([#109](https://github.com/stephenberry/hdf5-pure/issues/109)).
- `Dataset::write_staged` overwrites a dataset through its full `DatasetBuilder`, the builder-level counterpart of `Dataset::write` for element kinds that are not `H5Element` ([#148](https://github.com/stephenberry/hdf5-pure/issues/148)).
- `Group::create_group_with` stages a new group configured through a `StagedGroup` closure, so a group's attributes and children land with its creation (`set_attr` cannot reach it, since it needs a group that already resolves). `create_group(name)` is unchanged ([#148](https://github.com/stephenberry/hdf5-pure/issues/148)).

### Fixed

- Reading the same `File` from inside a builder closure (`Group::create_dataset`, `create_group_with`, `Dataset::write_staged`, `append_staged`) no longer hangs the process. The closures now configure a builder off the session lock, so a staged dataset may depend on data already in the file; it sees the file as it was before the call, since staged edits resolve only on `commit` ([#200](https://github.com/stephenberry/hdf5-pure/issues/200)).
- `repack` no longer corrupts a dataset whose datatype *contains* a variable-length member, an object-reference member, or both — such as a compound with a variable-length string field. The embedded addresses were copied verbatim and left pointing into the source file, producing a destination this crate read back without complaint and the reference C library could not read at all; they are now rewritten like top-level ones ([#201](https://github.com/stephenberry/hdf5-pure/issues/201)).

### Changed

- **Breaking:** `EnumTypeBuilder::build` returns `Result<Datatype, FormatError>`; it now refuses a non-integer base type, a member value too wide for the base, and raw bytes whose length disagrees with it, instead of writing a malformed datatype ([#208](https://github.com/stephenberry/hdf5-pure/issues/208)).
- **Breaking:** `EnumMember` is now `#[non_exhaustive]`, which `EnumTypeBuilder`'s arbitrary-base support makes free — build enumerations through the builder rather than a literal ([#208](https://github.com/stephenberry/hdf5-pure/issues/208)).
- **Breaking:** three refusals on the owned write path now report a more specific error: an unaligned SWMR append gives `SwmrAppendUnsupported` instead of a `ChunkedReadError`, an ineligible immediate append gives `AppendInPlaceUnsupported` instead of `AppendUnsupported`, and a missing edit target gives `PathNotFound` when the handle is resolved instead of `AppendUnsupported`/`EditUnsupported` at commit ([#148](https://github.com/stephenberry/hdf5-pure/issues/148)).
- **Breaking:** `AttrValue`, `DType`, `Datatype`, `ReferenceType`, `LibVer`, `Object`, `CompoundMember`, `FileSpaceInfo`, `VerifyResult`, `mat::MatClass`, and the four `mat::opaque` decode structs are now `#[non_exhaustive]`, so a new datatype, reference kind, library-version bound, or MATLAB class no longer breaks callers. Add a `_` arm when matching a read-back value; constructing the existing variants is unaffected, including `Datatype` literals for types this crate has no constructor for ([#206](https://github.com/stephenberry/hdf5-pure/pull/206)).
- **Breaking:** `RepackOptions` is now built only through `new` and `drop_path`, matching `FileAccessOptions`; the `drop` field is private and readable with `RepackOptions::drop_paths` ([#206](https://github.com/stephenberry/hdf5-pure/pull/206)).

### Removed

- **Breaking:** `File::open_rw_with_locking` is gone; the policy is an option on the open — `File::open_rw_with_options(path, FileAccessOptions::new().with_locking(..))` ([#204](https://github.com/stephenberry/hdf5-pure/issues/204)).
- **Breaking:** `Datatype::parse` and `Datatype::serialize` are now crate-internal. Read a dataset's type with `Dataset::datatype` and pass a `Datatype` to `DatasetBuilder::with_dtype`, which encodes it; `Datatype::type_size` stays public ([#206](https://github.com/stephenberry/hdf5-pure/pull/206)).
- **Breaking:** `FormatError::ChunkedVlenStringUnsupported` is gone, as nothing refuses those datasets any more ([#109](https://github.com/stephenberry/hdf5-pure/issues/109)).
- **Breaking:** `AppendWriter`, `SwmrWriter`, and `EditSession`, deprecated since 0.22.0, are gone. Use `File::open_rw` (or `File::open_swmr_writer`) with owned `Dataset` and `Group` handles; `File::open_rw_with_options` with `FileAccessOptions::with_locking` replaces `AppendWriter::open_with_locking` and `File::clear_swmr_flag` replaces `SwmrWriter::clear_swmr_flag`. The former `EditSession` methods map to `Dataset::append`/`append_staged`/`write`/`write_staged`/`set_attr`/`remove_attr`, `Group::create_group`/`create_dataset`/`delete`/`set_attr`, and `File::copy`/`copy_from`/`space_accounting`, with an object staged in an uncommitted batch reachable only through `create_group_with` ([#148](https://github.com/stephenberry/hdf5-pure/issues/148)).

## [0.24.0] - 2026-07-24

Variable-length writes lose their 65,535-element cap: `DatasetBuilder::with_vlen_strings` and `repack` now split across as many global heap collections as they need, and resolving an element is no longer quadratic ([#189](https://github.com/stephenberry/hdf5-pure/issues/189)). Attributes too large for compact or dense storage are now refused by name instead of silently dropped or written unreadable ([#190](https://github.com/stephenberry/hdf5-pure/issues/190), [#191](https://github.com/stephenberry/hdf5-pure/issues/191)). Reads gain bounds: a chunked dataset declaring an impossible per-chunk size is refused rather than allocated, with the new `Dataset::element_size` to size a read up front ([#185](https://github.com/stephenberry/hdf5-pure/issues/185)), and row windows of inner-chunked and variable-length string datasets stream instead of falling back to a whole read ([#183](https://github.com/stephenberry/hdf5-pure/pull/183), [#186](https://github.com/stephenberry/hdf5-pure/pull/186)). Additive minor bump.

### Added

- `OBJECT_HEADER_MESSAGE_MAX` is the largest message a version 2 object header can describe (65,535 bytes), the bound behind the new oversized-message refusals ([#190](https://github.com/stephenberry/hdf5-pure/issues/190)).
- `Dataset::element_size` returns the on-disk byte width of one element (HDF5's `H5Tget_size`), so a caller reading an untrusted file can multiply it by the element count from `shape` to bound a read's allocation before requesting it, rather than trusting the file's declared extent ([#185](https://github.com/stephenberry/hdf5-pure/issues/185)).

### Fixed

- Reading a chunked dataset whose datatype or chunk extent declares an impossible per-chunk logical size (over the 4 GiB format limit, e.g. a fixed-length string element of billions of bytes) is now refused with an `InvalidChunkGeometry` error instead of eagerly allocating the whole declared extent, so a crafted file can no longer drive a multi-gigabyte out-of-memory allocation from a few kilobytes ([#185](https://github.com/stephenberry/hdf5-pure/issues/185)).
- Variable-length datasets and attributes with more than 65,535 elements now write correctly, split across as many global heap collections as they need, so `DatasetBuilder::with_vlen_strings` and `repack` are no longer capped there ([#189](https://github.com/stephenberry/hdf5-pure/issues/189)).
- Resolving a variable-length element now binary-searches its heap collection's directory instead of scanning it, so reading a large variable-length string dataset is no longer quadratic in its element count ([#189](https://github.com/stephenberry/hdf5-pure/issues/189)).
- `Dataset::read_raw_rows` and the typed `read_*_rows` now stream a row window of an inner-chunked dataset by decoding only the chunks the window overlaps, instead of falling back to a whole read, so peak memory scales with the window plus one chunk rather than the dataset ([#183](https://github.com/stephenberry/hdf5-pure/pull/183)).
- `Dataset::read_string_rows` on variable-length strings now resolves only the window's heap references instead of reading and resolving the whole dataset before slicing, so the row-window memory bound holds for every windowed read: peak allocation is the window's references, its text, and the metadata of the heap collections it touches ([#186](https://github.com/stephenberry/hdf5-pure/pull/186)).
- `FileBuilder::write`/`finish` and `repack` now refuse a compact attribute whose object-header message exceeds `OBJECT_HEADER_MESSAGE_MAX` (the new `FormatError::AttributeMessageTooLarge`, naming the attribute) instead of truncating its size field, which silently dropped the attribute or left the file unreadable ([#190](https://github.com/stephenberry/hdf5-pure/issues/190)).
- An attribute too large for dense (fractal-heap) storage is now refused with the new `FormatError::DenseAttributeTooLarge` naming it, instead of being written into a heap that read back empty and aborted an assertion-enabled reference C library; more than 61,680 attributes on one object is likewise refused with `FormatError::TooManyDenseAttributes` ([#191](https://github.com/stephenberry/hdf5-pure/issues/191)).
- Dense attribute sets whose total passes 64 KiB are no longer refused by `EditSession` copies: the bound now tracks each attribute's size, which is what the emitter is actually limited by, so multi-megabyte sets of individually small attributes are written and copied normally. The dense-copy refusal now reports `FormatError` rather than `Error::EditUnsupported`, so it can name the attribute ([#191](https://github.com/stephenberry/hdf5-pure/issues/191)).
- A dense attribute heap larger than 64 KiB now records a maximum direct block size covering the block it actually contains, instead of a fixed 65,536 that its own block exceeded. Such files already read correctly in both libraries, so this changes their bytes without changing their meaning; heaps at or below 64 KiB are byte-for-byte unchanged ([#191](https://github.com/stephenberry/hdf5-pure/issues/191)).

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

[Unreleased]: https://github.com/stephenberry/hdf5-pure/compare/v0.33.0...HEAD
[0.33.0]: https://github.com/stephenberry/hdf5-pure/compare/v0.32.0...v0.33.0
[0.32.0]: https://github.com/stephenberry/hdf5-pure/compare/v0.31.0...v0.32.0
[0.31.0]: https://github.com/stephenberry/hdf5-pure/compare/v0.30.0...v0.31.0
[0.30.0]: https://github.com/stephenberry/hdf5-pure/compare/v0.29.0...v0.30.0
[0.29.0]: https://github.com/stephenberry/hdf5-pure/compare/v0.28.0...v0.29.0
[0.28.0]: https://github.com/stephenberry/hdf5-pure/compare/v0.27.0...v0.28.0
[0.27.0]: https://github.com/stephenberry/hdf5-pure/compare/v0.26.0...v0.27.0
[0.26.0]: https://github.com/stephenberry/hdf5-pure/compare/v0.25.0...v0.26.0
[0.25.0]: https://github.com/stephenberry/hdf5-pure/compare/v0.24.0...v0.25.0
[0.24.0]: https://github.com/stephenberry/hdf5-pure/compare/v0.23.2...v0.24.0
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
