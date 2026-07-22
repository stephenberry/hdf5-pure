# Limitations & Unsupported Features

`hdf5-pure` reads and writes a broad, interoperable subset of the HDF5 format. Where it cannot yet handle something, it returns a **clear typed error** rather than producing a wrong result — every gap below is a deliberate, well-messaged refusal, not a silent misread.

The refusals fall into two kinds:

- **[Deliberately unsupported](#deliberately-unsupported)** — by-design constraints or guards against file formats outside the range `hdf5-pure` models. These are not planned to change.
- **[Planned support](#planned-support)** — features refused *for now*, each tracked by a GitHub issue. The error messages for these read `... not supported yet` / `... cannot be ... yet`.

!!! note "Ordinary errors are not on this page"
    Malformed-file errors (truncated or garbled headers, an address that exceeds the platform's pointer width) and API-contract errors (deleting or copying the root group, conflicting edits in a single commit) are normal runtime errors, not capability limits, so they are not catalogued here.

## Deliberately unsupported

### Reading non-modeled formats

| Refused | Error | Why |
|---|---|---|
| Superblock version > 3 | `FormatError::UnsupportedVersion` | Superblock versions 0–3 are read; no higher version exists in the released format |
| An unrecognized object-header message flagged *must-understand* | `FormatError::UnsupportedMessage` | Refusing is the format-required behavior for a must-understand message a reader does not know |
| File Space Info message version other than 1 | `FormatError::UnsupportedFileSpaceInfoVersion` | Only version 1 is defined for the layouts this crate emits and reads |

These guard against files outside the format-version range `hdf5-pure` models; they are not features to add.

### Compression

| Refused | Error | Why |
|---|---|---|
| A filter whose backend is not compiled in | `FormatError::UnsupportedFilter` | Enable the `deflate` (or `zfp`) Cargo feature — see [Cargo Features](features.md) |
| ZFP outside fixed-rate, ranks 1–4, dtypes `f32`/`f64`/`i32`/`i64` | `FormatError::UnsupportedZfp` | The supported scope of the bundled ZFP codec — see [Compression](../guide/compression.md) |

### Repack faithfulness

`repack` rewrites a file and refuses **lossy filter re-encoding** (lossy float scale-offset, ZFP) rather than silently altering data: only *lossless* integer scale-offset with an undefined fill value can be re-encoded faithfully, since re-compressing lossy data would change the values. (Repack instead copies already-compressed chunks **verbatim** wherever it can, which preserves lossy filters byte-exact without re-encoding.)

### SWMR (single-writer / multiple-reader)

SWMR append requires a **latest-format** file (v2/v3 superblock) and **no userblock**. This mirrors the HDF5 SWMR model, which is only defined for the latest format.

### In-place editing

In-place editing operates on files with **8-byte offsets and lengths** (what the writer emits and what modern files use). Other offset/length widths are not editable in place.

### Bounded-memory read-write

A file opened with [`File::open_rw_bounded`](../guide/editing.md#bounded-memory-appends) supports reads and immediate `Dataset::append` only; the staged edit surface (`write`, attribute edits, `create_*`/`delete`, `copy`, `commit`, `space_accounting`) needs the whole-file mirror and returns `Error::BoundedStagedUnsupported` — open with `File::open_rw` for those. It requires a latest-format file with 8-byte offsets, no userblock, and no persisted free-space managers, and shares the streaming backend's path-resolution limits.

Adding an **object-reference dataset** (`EditSession::create_dataset(...).with_path_references(...)`) resolves a path target against every object this commit places, but only once that object has actually been placed: `commit()` processes groups deepest-first and, within a group, non-reference datasets before reference ones, so a target that is itself still being written when the reference is resolved — an ancestor group, a same-depth sibling group ordered later, a copy destination (or its interior), or a `write_dataset` target — is refused rather than resolved to a stale or wrong address. A target untouched by the commit resolves against the pre-commit file; a path that resolves nowhere at all becomes an undefined reference, matching `FileBuilder`'s resolution convention for the same builder type. This is a permanent scope line (not a `... yet` gap): reproducing the whole-file writer's two-pass dummy/real-address scheme inside `EditSession`'s single-pass commit would be a large rewrite of the core apply loop for a narrow benefit.

### Group creation property list (GCPL)

There is no property-list API for group creation, and none of its settings are configurable — every group `hdf5-pure` writes (including the root group) has exactly one fixed shape: a new-style (v2 object header) group with compact link storage and no stored timestamps. This is equivalent to always creating every group with `obj_track_times = false`, and never switching to old-style (symbol-table) or dense (fractal-heap) link storage, regardless of file version or child count. Unlike the reference library, whose GCPL defaults vary by version, this shape is fixed on purpose: it keeps output byte-for-byte reproducible, which is exactly what makes `hdf5-pure` a good fit for stable snapshot files. See [#131](https://github.com/stephenberry/hdf5-pure/issues/131).

## Planned support

Refused today with a `... yet` message, intended to land. Each row links to its tracking issue.

### In-place editing

| Capability | Tracking |
|---|---|
| Overwrite & copy of **chunked / filtered** datasets | [#101](https://github.com/stephenberry/hdf5-pure/issues/101) |
| **Dense** (fractal-heap) link & attribute storage | [#102](https://github.com/stephenberry/hdf5-pure/issues/102) |
| Editing across **soft / external links** | [#103](https://github.com/stephenberry/hdf5-pure/issues/103) |
| Userblock files, creation-order tracking, shared/SOHM messages, v0/v1 conversion | [#104](https://github.com/stephenberry/hdf5-pure/issues/104) |
| Adding **chunked/extensible variable-length-string** datasets | [#105](https://github.com/stephenberry/hdf5-pure/issues/105) |
| **Cross-file copy** of variable-length / reference / shared data | [#106](https://github.com/stephenberry/hdf5-pure/issues/106) |

### Repack

| Capability | Tracking |
|---|---|
| Repack of **region references**, non-8-byte object references, chunked/filtered/resizable variable-length & reference datasets, and unrecognized filter pipelines (time, contiguous non-string-vlen sequences, and 8-byte object references now repack faithfully) | [#107](https://github.com/stephenberry/hdf5-pure/issues/107) |

### Reading

| Capability | Tracking |
|---|---|
| **Filter-encoded fractal-heap** objects | [#108](https://github.com/stephenberry/hdf5-pure/issues/108) |
| **Virtual (VDS)** datasets | [#111](https://github.com/stephenberry/hdf5-pure/issues/111) |

Virtual datasets are also refused by `repack` (it cannot relocate data living outside the file); that lifts together with VDS read support ([#111](https://github.com/stephenberry/hdf5-pure/issues/111)).

### Writing

| Capability | Tracking |
|---|---|
| **Chunked / filtered / resizable variable-length-string** datasets | [#109](https://github.com/stephenberry/hdf5-pure/issues/109) |

### SWMR

| Capability | Tracking |
|---|---|
| Append to **multi-dimensional** and **filtered** datasets | [#110](https://github.com/stephenberry/hdf5-pure/issues/110) |

This gap is specific to SWMR (concurrent-reader) append. Appending to a **filtered** 1-D unlimited dataset without concurrent readers is already supported via [`EditSession::append_dataset`](../guide/editing.md#appending-to-an-unlimited-dataset) (any length) and [streaming `Dataset::append`](../guide/editing.md#streaming-appends) (whole chunks).
