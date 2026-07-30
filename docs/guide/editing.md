# Editing in Place

`File::open_rw` opens an existing HDF5 file for reading **and** writing, and edits it through owned `Dataset` and `Group` handles that reach every object by name — adding, copying, or deleting objects, or editing attributes, without rewriting the file from scratch. New data and rebuilt object headers are appended at the end of the file and the superblock is repointed last, so the cost is proportional to what changes rather than to the file size, and a failed commit leaves the original file valid.

!!! tip "Runnable example"
    This page mirrors [`examples/edit_in_place.rs`](https://github.com/stephenberry/hdf5-pure/blob/main/examples/edit_in_place.rs). Run it with:

    ```bash
    cargo run --example edit_in_place
    ```

## Choosing a write path

`File::open_rw` is the read-write open, and it picks how to hold the file's bytes from the file itself rather than making you pick a function. A latest-format file with no userblock is edited **bounded**: no whole-file copy is ever built, so memory stays at the [configured caches](streaming.md) plus whatever an edit is building. Anything else — a pre-v2 superblock, or a userblock — falls back to a whole-file **mirror**, `O(file size)`, which is what makes such a file editable at all. Either backing mutates the file the same way: new bytes are appended and a small, fixed set of locations is patched, never a rewrite on commit; see [write paths](../about/architecture.md#write-paths) for the mechanics.

Ask a file which backing it got with `File::edit_backing()`, and demand one with `FileAccessProperties::with_memory_strategy`:

| `MemoryStrategy` | Effect |
| --- | --- |
| unset (the default) | Prefer bounded; fall back to the mirror for a file the bounded engine cannot edit |
| `Bounded` | Never build a mirror; refuse such a file with `Error::EditUnsupported` |
| `Mirrored` | Always build the mirror, whatever the file looks like |

`File::open_rw_bounded` is the deprecated spelling of `Bounded`. The answer from `File::edit_backing()` is an `EditBacking` (`Bounded` or `Mirrored`) rather than the `MemoryStrategy` that was asked for, because `Auto` is a preference between the two backings and never an outcome; `.into()` converts an `EditBacking` back into the `MemoryStrategy` that pins a later reopen to it.

`File::open_swmr_writer` always mirrors, so it accepts `Auto` and `Mirrored` — both satisfied by the mirror — and refuses an explicit `Bounded` rather than quietly not honoring it.

The two backings are the same engine and the same edit vocabulary — reads, immediate `Dataset::append` / `append_raw`, `Dataset::write`, `append_staged`, `set_attr` / `remove_attr`, `Group::create_dataset` / `create_group` / `create_group_with` / `delete`, `File::copy` / `copy_from`, `commit`, and `space_accounting` — with one trade between them. A large `Dataset::append` is one crash-atomic apply on the mirror and several ~1 MiB whole-chunk batches when bounded, so a crash mid-call there leaves a valid shorter dataset. A commit's resident memory follows the backing too: bounded by the edit rather than by the file, with `File::copy` the exception, since copying an object reads the whole of it into memory first.

One caveat inside either backing: the immediate `Dataset::append` is stricter than the staged surface — it needs a latest-format (v2/v3-superblock) file with no userblock, and the refusal (`Error::AppendInPlaceUnsupported`) names `Dataset::append_staged` as the fallback.

The file's [file-space strategy](file-space.md) gates what an edit can do at all, whichever backing holds it:

| File-space strategy | Staged edits + `commit` | Immediate append |
| --- | --- | --- |
| None recorded, or `FsmAggr` / `Aggr` / `None` with `persist = false` | Yes | Yes |
| `FsmAggr` / `Aggr` / `None` with `persist = true` | Yes — freed space is recorded on disk | Yes — managers rewritten at `close` |
| `Page` with `persist = true` | Yes — page-aware commit, per-page-type managers rewritten | Yes — appends stay page-homogeneous |
| `Page` with `persist = false` | No — refused at open | No — refused at open |

The one refusal here is `Error::EditUnsupported` for a paged file that does not persist its free space: neither backing can keep such a file's pages segregated, so `open_rw` refuses it up front rather than letting an edit be staged against it. It fires before any byte of the file changes. (`MemoryStrategy::Mirrored` still opens the file, since that asks for a backing rather than expressing a preference, but a commit through it refuses.) A `Page` / `persist = false` file stays fully readable through every read path and can be rewritten compactly by [repack](repack.md).

A paged commit keeps each page homogeneous — raw data and file metadata never share a page, apart from a chunked dataset's index, which travels with its chunk data (see [File-Space Strategy](file-space.md)) — and page-aligns the end of allocation, so the reference C library reopens the result as a paged file and recovers its free space. Because a free hole belongs to one page type, such a commit appends rather than reusing holes within the commit; the space is recovered by the manager rewrite at the end of it.

For a brand-new file, use [`FileBuilder`](writing.md); to append while readers are live, use the [SWMR writer](swmr.md); to compact a file or drop objects across a reopen, use [repack](repack.md). The [file properties reference](../reference/property-support.md) has the corresponding fcpl/fapl support matrix.

## Staging and committing edits

An edit session is transactional: you stage operations on an open file, then apply them all at once with `commit()`. Nothing on disk changes until `commit()` succeeds.

```rust
use hdf5_pure::{AttrValue, File};

let file = File::open_rw("output.h5").unwrap();
let root = file.root();

root.create_group_with("run2", |g| {
    g.set_attr("kind", AttrValue::AsciiString("trial".into()));
})
.unwrap();
root.create_dataset("run2/signal", |b| {
    b.with_f64_data(&[1.0, 2.0, 3.0]);
})
.unwrap();
file.copy("temperature", "temperature_backup").unwrap(); // H5Ocopy
root.delete("sensors/pressure").unwrap();                // H5Ldelete

file.commit().unwrap(); // apply everything in place
```

After a successful `commit()`, the staged set is cleared and the open file can be reused for further edits.

## Operations

| Method | Effect | HDF5 analog |
| --- | --- | --- |
| `File::open_rw(path)` | Open an existing file for reading and writing | — |
| `Group::create_group(name)` | Stage a new empty group | — |
| `Group::create_group_with(name, build)` | Stage a new group, configured through `build` (attributes, nested objects) | — |
| `Group::create_dataset(name, build)` | Stage a new dataset, configured through a `DatasetBuilder` | — |
| `Dataset::append_staged(build)` | Stage appending elements along axis 0 of an existing chunked, unlimited dataset, via an `AppendBuilder` | `H5Dset_extent` + write |
| `Dataset::append(data)` | Append immediately and durably, no `commit` needed | `H5Dset_extent` + write |
| `Dataset::write(data)` / `write_staged(build)` | Stage a value overwrite of the same datatype and shape | `H5Dwrite` |
| `Group::set_attr(name, value)` / `Dataset::set_attr` | Stage adding or replacing a compact attribute | — |
| `Group::remove_attr(name)` / `Dataset::remove_attr` | Stage removing a compact attribute | — |
| `File::copy(src, dst)` | Stage a deep copy of a dataset or whole group subtree within this file | `H5Ocopy` |
| `File::copy_from(source, src, dst)` | Copy a dataset or subtree out of another open `File` into this one | `H5Ocopy` (across files) |
| `Group::delete(name)` | Stage removing the link at `name` (and, for a group, its whole subtree) | `H5Ldelete` |
| `File::commit()` | Apply all staged operations in place and flush | — |

`create_dataset` hands you the same `DatasetBuilder` used by [`FileBuilder`](writing.md), so you configure the new dataset exactly as you would when creating a file from scratch:

```rust
root.create_dataset("run2/signal", |b| {
    b.with_f64_data(&[1.0, 2.0, 3.0]);
})
.unwrap();
```

`set_attr` needs a group that already resolves, so a group staged in this same batch is not reachable through it — stage those attributes with `create_group_with` instead. A staged object is not resolvable by name until `commit`, so `File::group`/`File::dataset` will not find it before then.

The `create_group_with`/`create_dataset`/`write_staged`/`append_staged` closures configure a builder rather than the file itself, so a closure may read the same `File` — staging a dataset whose contents depend on one already there works. What it reads is the file as it was before the call, since nothing staged resolves until `commit`.

`set_attr` takes an `AttrValue`, fixed-size or variable-length (`AttrValue::VarLenAsciiArray`). `File::root()` names the root group. Attributes are stored compactly in the rebuilt group header; an edit that would exceed the compact-attribute limit, or a group using dense (fractal-heap) attribute storage, is refused before any file bytes change.

`copy` performs a deep copy: fresh copies of every object's data and header are written, internal links and the contiguous data address are repointed to the copies, and a link named by the last component of `dst` is added to its parent group. The original is untouched. `src` must exist and `dst` must not (and may not lie inside `src`). Compact attributes are carried over byte-for-byte — including the latest-format form the C library and h5py write, where a handful of inline attributes are accompanied by an Attribute Info message. Dense (fractal-heap) attribute storage, which an object takes on above 8 attributes or for a single attribute too large for an object-header message, is also reproduced: the source attributes are read out of the source heap and re-emitted into a fresh single-direct-block fractal heap plus B-tree v2 name index in the destination (the copy tracks only the name index, not the creation-order index). An attribute too large even for a managed heap object is re-emitted as a *huge* object, as it was in the source. More attributes than the single B-tree leaf can index is refused by name rather than mis-encoded; the set's *total* size is not limited, since the heap's one block is sized to the content (see [Limitations](../reference/limitations.md#dense-attribute-storage)).

`copy_from` is the same operation **across two open files** — the cross-file form of `H5Ocopy`. The source lives in a separate [`File`](reading.md) reader rather than the file being edited:

```rust
use hdf5_pure::File;

let library = File::open("library.h5").unwrap();
let file = File::open_rw("output.h5").unwrap();
file.copy_from(&library, "calibration", "run2/calibration").unwrap();
file.commit().unwrap();
```

Unlike `copy`, the source subtree is read and validated **eagerly** (the `File` borrow need not outlive the call), so `copy_from` returns a `Result`; the destination still changes only on `commit()`. Because the copy is byte-for-byte verbatim, anything whose stored bytes embed a *source-file* absolute address — which would dangle in another file — is refused up front: variable-length and reference datasets and attributes (whether compact or dense), and any shared header message (a committed datatype, or an SOHM-shared dataspace, fill value, or filter pipeline). The same-file `copy` keeps these forms valid instead, by sharing the source file's global heaps and objects. The `source` must be a buffered file (`File::open` or `File::from_bytes`, not `File::open_streaming`) using 8-byte offsets and no userblock.

## Appending to an unlimited dataset

`Dataset::append_staged` grows an existing **chunked, unlimited** dataset in place along its first (axis-0) dimension, **including filtered** datasets (deflate, shuffle, fletcher32, scale-offset, LZF, and ZFP with the `zfp` feature). It is the general, non-SWMR counterpart to the [SWMR writer](swmr.md), which appends only to *unfiltered*, chunk-aligned datasets. It returns an `AppendBuilder` whose typed and generic methods mirror the writer's; repeated calls concatenate in call order.

```rust
use hdf5_pure::File;

let file = File::open_rw("log.h5").unwrap();
file.dataset("samples")
    .unwrap()
    .append_staged(|a| {
        a.append_i32(&[8, 9, 10, 11]);
    })
    .unwrap();
file.commit().unwrap();
```

Existing chunks stay exactly where they are. Only the newly appended chunks — plus the single trailing partial chunk, when the dataset's current length is not a whole multiple of the chunk length — are compressed and written; every other chunk is carried into the rebuilt index by metadata alone. So an append does not rewrite existing data and the file does not grow by the whole dataset each time. Appends of any length are allowed, and the datatype, fill value, filter pipeline, and attributes are preserved.

Like every staged edit, an append commits by writing the new chunks and a rebuilt index at end-of-file and repointing the superblock last (under the file's exclusive lock), so a crash leaves either the original dataset or the fully grown one, never a torn state. It sets no SWMR flag.

### Eligibility

The first release supports the Extensible-Array chunk index — the index the reference C library and h5py select for a single unlimited dimension under the latest format, and the one this crate writes for every unlimited dataset — with rank-1 datasets that have a single hard link. A dataset that is not chunked, not unlimited along axis 0, not Extensible-Array indexed, higher than rank 1, uses a filter this engine cannot re-encode, or has a sparse chunk grid is refused with `Error::AppendUnsupported` before any file bytes change. Check eligibility up front with the read-side accessors [`is_chunked`, `maxshape`, `chunk_shape`, and `filters`](reading.md#chunking-filters-and-append-eligibility) rather than relying on the refusal error.

Element types are checked, never coerced: each typed `append_*` call records the datatype it implies, and `commit` refuses a mismatch against the dataset's on-disk datatype — including a mix of element types in one builder. `append_raw` appends already-little-endian element bytes verbatim; its length must be a whole multiple of the element size and the dataset's datatype must be little-endian.

!!! tip "Runnable example"
    This section mirrors [`examples/append_dataset.rs`](https://github.com/stephenberry/hdf5-pure/blob/main/examples/append_dataset.rs). Run it with `cargo run --example append_dataset`.

### Streaming appends

`append_staged` rebuilds the dataset's chunk index and relocates its header on every `commit` (and each new `File::open_rw` re-reads the metadata it needs, or the whole file when it falls back to the mirror), which is the right trade for a one-off append composed alongside other edits, but not for a high-frequency append loop. For that, open the file **once** with `File::open_rw` and append many times through a `Dataset` handle, growing the Extensible-Array index *in place* — so each append costs `O(appended bytes)` plus amortized `O(1)` index overhead, with no whole-file re-read and no index rebuild.

```rust
use hdf5_pure::File;

let file = File::open_rw("log.h5").unwrap();
let mut samples = file.dataset("samples").unwrap();
samples.append(&[8i32, 9, 10, 11]).unwrap();
samples.append(&[12i32, 13]).unwrap(); // unfiltered: any length
file.close().unwrap();
```

One open file reaches every dataset by name, takes an exclusive file lock for its lifetime, and sets no SWMR flag. **Every `append` is crash-atomic**: writes are ordered child-before-parent with `fsync` barriers and the dataspace dimension is published last as the single commit point, so a crash between appends leaves either the previous length or the new one — never a torn or lost view.

That atomicity is why filtered and unfiltered datasets have different length rules. An **unfiltered** append may be **any length**: when the current length is not chunk-aligned, the trailing partial chunk is rewritten and its index element — a single chunk address — is repointed with one atomic write. A **filtered** append must be **chunk-aligned** (the current length and the appended length both whole multiples of the chunk length), because a filtered index element is a multi-field record whose in-place repoint is not power-loss atomic; a filtered append therefore only ever inserts new chunks. For a non-chunk-aligned filtered append, use `Dataset::append_staged`, which rebuilds the index and repoints the superblock last (fully atomic).

The remaining eligibility rules match `Dataset::append_staged` (chunked, unlimited axis 0, Extensible-Array index, rank 1, a re-encodable filter pipeline), plus the file-level gates in [the tables above](#choosing-a-write-path), with one difference: because it grows the index in place rather than rebuilding it, the index must already be allocated. This crate allocates it eagerly, so an empty dataset it wrote can be grown from the first append; an empty dataset the C library created without any initial data defers its index and is refused — make that first append with `Dataset::append_staged` (which materializes the index), or create the dataset with initial data. The dead bytes left when an unfiltered partial chunk is relocated are reclaimed by [repack](repack.md) rather than reused within the session in this release. This is the throughput-oriented counterpart to `Dataset::append_staged` and the filter-capable counterpart to the [SWMR writer](swmr.md).

!!! tip "Runnable example"
    This section mirrors [`examples/append_streaming.rs`](https://github.com/stephenberry/hdf5-pure/blob/main/examples/append_streaming.rs). Run it with `cargo run --example append_streaming`.

### Bounded-memory appends

Appending to a large file needs no special entry point: `File::open_rw` already edits a latest-format file bounded, the read-write sibling of [`open_streaming`](streaming.md). Reads are served by positioned I/O with the streaming backend's capabilities, and `Dataset::append` runs the same crash-atomic in-place engine as the mirror, reading and patching only bounded windows (the object header, the extensible-array blocks it touches, and the trailing chunk). A large append is applied in whole-chunk batches, each crash-atomic on its own, so peak memory stays at the configured caches plus a few chunks — independent of the file size and of how much one call appends.

```rust
use hdf5_pure::{EditBacking, File, FileAccessProperties, MemoryStrategy};

// Bounded because the file allows it; add the hint to make it a requirement
// rather than a preference.
let file = File::open_rw_with_options(
    "huge-log.h5",
    FileAccessProperties::new().with_memory_strategy(MemoryStrategy::Bounded),
).unwrap();
assert_eq!(file.edit_backing(), Some(EditBacking::Bounded));
let mut samples = file.dataset("samples").unwrap();
samples.append(&[8i32, 9, 10, 11]).unwrap();
file.close().unwrap();
```

Bounded editing **does** grow a file that persists its free space — including a genuine paged file (`H5F_FSPACE_STRATEGY_PAGE`) — seeding the on-disk free-space managers on open and rewriting them at `File::close`, with paged appends kept page-homogeneous (raw and metadata in separate pages). See [File-Space Strategy](file-space.md) for the paged details. Memory budgets are set with the same `FileAccessProperties` as the streaming reader; cached metadata windows touched by an append are invalidated automatically, so reads through the same file never observe stale bytes.

## How it works

`commit()` appends each new dataset (its data blob and object header) and each new group, then appends rewritten object headers for every touched group and its ancestors up to the root (omitting any deleted links), and finally repoints the superblock at the new root.

The appended data is `fsync`ed before the root is repointed, so the "repoint last" guarantee is real: if the process or machine fails during a commit, the original file is still intact and readable, because the superblock still points at the old root. The cost of a commit scales with the size of the edit, not the size of the file.

!!! warning "All-or-nothing safety"
    Every check runs before the first byte is written. On any `Error::EditUnsupported`, the file on disk is left untouched. This makes editing safe to attempt: an unsupported edit fails cleanly rather than producing a partially modified or corrupt file.

## Supported targets and formats

Contiguous and chunked datasets (with any filter the whole-file writer supports) and compact-link groups are supported. The editor works across every on-disk format the reference HDF5 C library and h5py produce:

- Version 0, 1, 2, and 3 superblocks.
- Single- and multi-chunk object headers. A multi-chunk header is collapsed into a single chunk on rewrite.
- A version 0/1 symbol-table group on the edited path is converted to the latest compact-link format. Adding and deleting are supported on these older files; copying a version-1 object is not.

Rather than silently degrade a file, the editor refuses anything it cannot reproduce faithfully, returning `Error::EditUnsupported`:

- A file whose superblock is not located at its base address — a relocated or malformed userblock layout. (A canonical userblock, such as a MATLAB v7.3 `.mat` file's 512-byte userblock, is supported: addresses are read and written relative to the base and the userblock bytes are preserved.)
- Dense-storage headers on the edited path.
- Copying an existing version-1 object.
- Across files (`copy_from`): variable-length or reference datasets and attributes, any shared (committed/SOHM) header message, and a streaming source file — none of which can be reproduced verbatim in another file.

See [`Error::EditUnsupported`](../reference/data-types.md) for the full set of refusals.

## Space reuse and truncation

Within a session, the space a deletion frees is reused for later writes in the same commit, so add/delete churn stays bounded instead of only ever growing the file. If a freed run reaches the end of the file, the file is truncated.

Contiguous and chunked datasets (chunk index plus chunk data) and whole group subtrees are reclaimed. Reclaim is best-effort: an object whose blocks cannot be enumerated exhaustively (variable-length global-heap storage, dense attribute or link heaps, a version 2 B-tree chunk index) is left as dead bytes rather than risk freeing a region still in use.

!!! note "Cross-session reuse and guaranteed compaction"
    By default, freed space is reused only within the open session and forgotten on close. For a file created with `H5Pset_file_space_strategy(persist = true)`, freed space is recorded on disk and survives reopen; see [File-space strategy](file-space.md). For a guaranteed shrink that rewrites the whole file compact across a reopen, see [Reclaiming space with repack](repack.md).

## Verifying edits

Reopen the file with [`File::open`](reading.md) to confirm the edits landed:

```rust
use hdf5_pure::File;

let file = File::open("output.h5").unwrap();
let signal = file.dataset("run2/signal").unwrap().read_f64().unwrap();
let backup = file.dataset("temperature_backup").unwrap().read_f64().unwrap();
let run2_attrs = file.group("run2").unwrap().attrs().unwrap();

assert_eq!(signal, vec![1.0, 2.0, 3.0]);
assert_eq!(backup, vec![22.5, 23.1, 21.8]);
assert!(file.dataset("sensors/pressure").is_err());
```

For background on the append-and-repoint design, see the [architecture overview](../about/architecture.md).
