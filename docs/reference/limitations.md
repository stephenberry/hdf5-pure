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

A file [`File::open_rw` edits bounded](../guide/editing.md#bounded-memory-appends) offers the same edit surface as one it mirrors — reads, immediate `Dataset::append`, and the staged surface (`write`, attribute edits, `create_*`/`delete`, `copy`, `commit`, `space_accounting`) — with a commit whose resident memory is bounded by the edit rather than by the file (`File::copy` excepted: copying an object reads the whole of it into memory). The bounded backing needs a latest-format file with 8-byte offsets and no userblock, and anything else is mirrored instead; reads have the [streaming backend's capabilities](../guide/streaming.md). It grows a file that persists its free space, including a genuine paged file (`H5F_FSPACE_STRATEGY_PAGE`); a paged file that does *not* persist its free space is refused with `Error::EditUnsupported` (recreate it with `persist = true`), whichever backing holds it.

Adding an **object-reference dataset** (`Group::create_dataset` with `with_path_references(...)`) resolves a path target against every object this commit places, but only once that object has actually been placed: `commit()` processes groups deepest-first and, within a group, non-reference datasets before reference ones, so a target that is itself still being written when the reference is resolved — an ancestor group, a same-depth sibling group ordered later, a copy destination (or its interior), or a `Dataset::write` target — is refused rather than resolved to a stale or wrong address. A target untouched by the commit resolves against the pre-commit file; a path that resolves nowhere at all becomes an undefined reference, matching `FileBuilder`'s resolution convention for the same builder type. This is a permanent scope line (not a `... yet` gap): reproducing the whole-file writer's two-pass dummy/real-address scheme inside the editor's single-pass commit would be a large rewrite of the core apply loop for a narrow benefit.

### Object header message size

A version 2 object header describes each message's length in a 2-byte field, so no message it carries may exceed `OBJECT_HEADER_MESSAGE_MAX` (65,535 bytes). The whole-file writer refuses rather than truncating:

- A **compact attribute** past the limit would be refused with `FormatError::AttributeMessageTooLarge`, naming the attribute. The limit is on the *message* — name, datatype, dataspace, and data — not on the element count.
- Any **other** oversized message — most reachably a Link message from a very long dataset or group name — is refused with `FormatError::ObjectHeaderMessageTooLarge`, carrying the message type.

An attribute that would exceed the limit is not refused, though: it selects fractal-heap storage instead, where no such field bounds it. See [dense attribute storage](#dense-attribute-storage) below. `AttributeMessageTooLarge` is therefore a backstop no input reaches today — the writer sends exactly those attributes to a heap — kept because the limit it describes is a real property of the object header. Only `ObjectHeaderMessageTooLarge`, which covers messages with no heap alternative, is reachable on size.

The [in-place editor](../guide/editing.md) enforces the same limit separately, reporting `Error::EditUnsupported`.

### Dense attribute storage

An object stores its attributes in a fractal heap when it has more than eight of them, **or** when any one of them is too large for an object-header message — the same disjunction the reference C library uses, and the reason a single large attribute is written rather than refused.

The writer emits a single root direct block, indexed by B-trees of fixed 512-byte nodes — the node size the reference C library uses for both indexes — which grow internal levels as the record count rises. An attribute serializing past **65,514 bytes** does not fit a heap *managed* object, so it is written as a **huge** object instead: its bytes go outside the managed blocks and a huge-objects B-tree maps a generated ID to them. There is no limit on an individual attribute's size, and none on how many attributes an object may carry. What is still refused:

- An attribute whose **name, datatype, or dataspace** serializes past **65,535 bytes** is refused with `FormatError::AttributeFieldTooLong`, naming the attribute and the field. Each has a 2-byte length field in the attribute message, and huge storage lifts the limit on an attribute's data, not on the fields that describe it.
- Gigabytes of *managed* attributes on one object are refused with `FormatError::DenseAttributeHeapTooLarge`, which needs a direct block past the format's 2 GiB maximum for one.

The *total* is otherwise not limited — the root direct block is sized to the content, so multi-megabyte heaps of individually small attributes are written normally. Note that block is padded up to a power of two, so a large dense attribute set can occupy up to roughly twice its own size on disk.

### Group creation property list (GCPL)

There is no property-list API for group creation, and none of its settings are configurable — every group `hdf5-pure` writes (including the root group) has exactly one fixed shape: a new-style (v2 object header) group with compact link storage and no stored timestamps. This is equivalent to always creating every group with `obj_track_times = false`, and never switching to old-style (symbol-table) or dense (fractal-heap) link storage, regardless of file version or child count. Unlike the reference library, whose GCPL defaults vary by version, this shape is fixed on purpose: it keeps output byte-for-byte reproducible, which is exactly what makes `hdf5-pure` a good fit for stable snapshot files. See [#131](https://github.com/stephenberry/hdf5-pure/issues/131).

## Planned support

Refused today with a `... yet` message, intended to land. Each row links to its tracking issue.

### In-place editing

| Capability | Tracking |
|---|---|
| **Dense** (fractal-heap) link & attribute storage | [#102](https://github.com/stephenberry/hdf5-pure/issues/102) |
| Editing across **soft / external links** | [#103](https://github.com/stephenberry/hdf5-pure/issues/103) |
| Creation-order tracking, shared/SOHM messages, copying a **version-1** object | [#104](https://github.com/stephenberry/hdf5-pure/issues/104) |
| Adding **chunked/extensible variable-length-string** datasets | [#105](https://github.com/stephenberry/hdf5-pure/issues/105) |
| **Cross-file copy** of variable-length / reference / shared data | [#106](https://github.com/stephenberry/hdf5-pure/issues/106) |

### Repack

| Capability | Tracking |
|---|---|
| Repack of **region references**, non-8-byte object references, chunked/filtered/resizable **reference** datasets, and unrecognized filter pipelines (time, variable-length sequences, and 8-byte object references now repack faithfully; chunked, filtered, and resizable variable-length datasets repack as of [#109](https://github.com/stephenberry/hdf5-pure/issues/109)) | [#107](https://github.com/stephenberry/hdf5-pure/issues/107) |

### Reading

| Capability | Tracking |
|---|---|
| **Filter-encoded fractal-heap** objects | [#108](https://github.com/stephenberry/hdf5-pure/issues/108) |
| **Virtual (VDS)** datasets | [#111](https://github.com/stephenberry/hdf5-pure/issues/111) |

Virtual datasets are also refused by `repack` (it cannot relocate data living outside the file); that lifts together with VDS read support ([#111](https://github.com/stephenberry/hdf5-pure/issues/111)).

### Writing

| Capability | Tracking |
|---|---|
| **Append** to a variable-length dataset (writing one resizable is supported; growing it is not) | [#109](https://github.com/stephenberry/hdf5-pure/issues/109) |
| Add a **chunked variable-length** dataset to an *existing* file in place | [#109](https://github.com/stephenberry/hdf5-pure/issues/109) |

Chunked, filtered, and resizable variable-length datasets now write, so the entries that were here for those have gone ([#109](https://github.com/stephenberry/hdf5-pure/issues/109)). Two gaps remain. A resizable one can be created but not grown: `Dataset::append` is typed and `H5Element` covers only numeric scalars, and `append_raw` refuses a variable-length datatype because it cannot encode the heap references. And adding such a dataset to an existing file through the in-place edit engine is refused, since the engine appends into a fixed layout with nowhere to place the heap collections ahead of the chunks.

### SWMR

| Capability | Tracking |
|---|---|
| Append to **multi-dimensional** and **filtered** datasets | [#110](https://github.com/stephenberry/hdf5-pure/issues/110) |

This gap is specific to SWMR (concurrent-reader) append. Appending to a **filtered** 1-D unlimited dataset without concurrent readers is already supported via [`Dataset::append_staged`](../guide/editing.md#appending-to-an-unlimited-dataset) (any length) and [streaming `Dataset::append`](../guide/editing.md#streaming-appends) (whole chunks).
