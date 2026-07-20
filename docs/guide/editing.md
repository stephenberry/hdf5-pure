# Editing in Place

`EditSession` opens an existing HDF5 file and adds, copies, or deletes objects, or edits group attributes, without reading the whole file in and rewriting it. New data and rebuilt object headers are appended at the end of the file and the superblock is repointed last, so the cost is proportional to what changes rather than to the file size, and a failed commit leaves the original file valid.

!!! warning "Deprecated"
    `EditSession` is superseded by the owned-handle API and will be removed in a later release. Open a file for reading **and** writing with `File::open_rw` and edit it through owned `Dataset` and `Group` handles that reach every object by name: `Dataset::append`/`write`/`append_staged`, `Group::create_dataset`/`create_group`/`delete`/`set_attr`, and `File::copy`/`copy_from`/`commit`. One open file both writes and reads back, with no separate session type.

!!! tip "Runnable example"
    This page mirrors [`examples/edit_in_place.rs`](https://github.com/stephenberry/hdf5-pure/blob/main/examples/edit_in_place.rs). Run it with:

    ```bash
    cargo run --example edit_in_place
    ```

## Staging and committing edits

An edit session is transactional: you stage operations on an open file, then apply them all at once with `commit()`. Nothing on disk changes until `commit()` succeeds.

```rust
use hdf5_pure::{AttrValue, EditSession};

let mut session = EditSession::open("output.h5").unwrap();

session.create_group("run2");
session.set_group_attr("run2", "kind", AttrValue::AsciiString("trial".into()));
session.create_dataset("run2/signal").with_f64_data(&[1.0, 2.0, 3.0]);
session.copy("temperature", "temperature_backup"); // H5Ocopy
session.delete("sensors/pressure");                // H5Ldelete

session.commit().unwrap(); // apply everything in place
```

After a successful `commit()`, the staged set is cleared and the session can be reused for further edits.

## Operations

| Method | Effect | HDF5 analog |
| --- | --- | --- |
| `EditSession::open(path)` | Open an existing file for editing | — |
| `create_group(path)` | Stage a new empty group; its parent must exist or be created in the same session | — |
| `create_dataset(path)` | Stage a new dataset and return a `DatasetBuilder` to configure data, shape, and attributes | — |
| `append_dataset(path)` | Stage appending elements along axis 0 of an existing chunked, unlimited dataset; returns an `AppendBuilder` | `H5Dset_extent` + write |
| `set_group_attr(path, name, value)` | Stage adding or replacing a compact group attribute | — |
| `remove_group_attr(path, name)` | Stage removing a compact group attribute | — |
| `copy(src, dst)` | Stage a deep copy of a dataset or whole group subtree within this file | `H5Ocopy` |
| `copy_from(source, src, dst)` | Copy a dataset or subtree out of another open `File` into this one | `H5Ocopy` (across files) |
| `delete(path)` | Stage removing the link at `path` (and, for a group, its whole subtree) | `H5Ldelete` |
| `commit()` | Apply all staged operations in place and flush | — |

`create_dataset` returns the same `DatasetBuilder` used by [`FileBuilder`](writing.md), so you configure the new dataset exactly as you would when creating a file from scratch:

```rust
session.create_dataset("run2/signal").with_f64_data(&[1.0, 2.0, 3.0]);
```

`set_group_attr` takes an `AttrValue`, fixed-size or variable-length (`AttrValue::VarLenAsciiArray`). The group it names may already exist or may be created earlier in the same session; `""` or `"/"` names the root group. Attributes are stored compactly in the rebuilt group header; an edit that would exceed the compact-attribute limit, or a group using dense (fractal-heap) attribute storage, is refused before any file bytes change.

`copy` performs a deep copy: fresh copies of every object's data and header are written, internal links and the contiguous data address are repointed to the copies, and a link named by the last component of `dst` is added to its parent group. The original is untouched. `src` must exist and `dst` must not (and may not lie inside `src`). Compact attributes are carried over byte-for-byte — including the latest-format form the C library and h5py write, where a handful of inline attributes are accompanied by an Attribute Info message. Dense (fractal-heap) attribute storage, which appears above 8 attributes, is also reproduced: the source attributes are read out of the source heap and re-emitted into a fresh single-direct-block fractal heap plus B-tree v2 name index in the destination (the copy tracks only the name index, not the creation-order index). An attribute set too large for that single direct block (one that would need fractal-heap indirect blocks) is refused by name rather than mis-encoded.

`copy_from` is the same operation **across two open files** — the cross-file form of `H5Ocopy`. The source lives in a separate [`File`](reading.md) reader rather than the file being edited:

```rust
use hdf5_pure::{EditSession, File};

let library = File::open("library.h5").unwrap();
let mut session = EditSession::open("output.h5").unwrap();
session.copy_from(&library, "calibration", "run2/calibration").unwrap();
session.commit().unwrap();
```

Unlike `copy`, the source subtree is read and validated **eagerly** (the `File` borrow need not outlive the call), so `copy_from` returns a `Result`; the destination still changes only on `commit()`. Because the copy is byte-for-byte verbatim, anything whose stored bytes embed a *source-file* absolute address — which would dangle in another file — is refused up front: variable-length and reference datasets and attributes (whether compact or dense), and any shared header message (a committed datatype, or an SOHM-shared dataspace, fill value, or filter pipeline). The same-file `copy` keeps these forms valid instead, by sharing the source file's global heaps and objects. The `source` must be a buffered file (`File::open` or `File::from_bytes`, not `File::open_streaming`) using 8-byte offsets and no userblock.

## Appending to an unlimited dataset

`append_dataset` grows an existing **chunked, unlimited** dataset in place along its first (axis-0) dimension, **including filtered** datasets (deflate, shuffle, fletcher32, scale-offset, and ZFP with the `zfp` feature). It is the general, non-SWMR counterpart to the [SWMR writer](swmr.md), which appends only to *unfiltered*, chunk-aligned datasets. It returns an `AppendBuilder` whose typed and generic methods mirror the writer's; repeated calls concatenate in call order.

```rust
use hdf5_pure::EditSession;

let mut session = EditSession::open("log.h5").unwrap();
session.append_dataset("samples").append_i32(&[8, 9, 10, 11]);
session.commit().unwrap();
```

Existing chunks stay exactly where they are. Only the newly appended chunks — plus the single trailing partial chunk, when the dataset's current length is not a whole multiple of the chunk length — are compressed and written; every other chunk is carried into the rebuilt index by metadata alone. So an append does not rewrite existing data and the file does not grow by the whole dataset each time. Appends of any length are allowed, and the datatype, fill value, filter pipeline, and attributes are preserved.

Like every `EditSession` edit, an append commits by writing the new chunks and a rebuilt index at end-of-file and repointing the superblock last (under the session's exclusive lock), so a crash leaves either the original dataset or the fully grown one, never a torn state. It sets no SWMR flag.

### Eligibility

The first release supports the Extensible-Array chunk index — the index the reference C library and h5py select for a single unlimited dimension under the latest format, and the one this crate writes for every unlimited dataset — with rank-1 datasets that have a single hard link. A dataset that is not chunked, not unlimited along axis 0, not Extensible-Array indexed, higher than rank 1, uses a filter this engine cannot re-encode, or has a sparse chunk grid is refused with `Error::AppendUnsupported` before any file bytes change. Check eligibility up front with the read-side accessors [`is_chunked`, `maxshape`, `chunk_shape`, and `filters`](reading.md#chunking-filters-and-append-eligibility) rather than relying on the refusal error.

Element types are checked, never coerced: each typed `append_*` call records the datatype it implies, and `commit` refuses a mismatch against the dataset's on-disk datatype — including a mix of element types in one builder. `append_raw` appends already-little-endian element bytes verbatim; its length must be a whole multiple of the element size and the dataset's datatype must be little-endian.

!!! tip "Runnable example"
    This section mirrors [`examples/append_dataset.rs`](https://github.com/stephenberry/hdf5-pure/blob/main/examples/append_dataset.rs). Run it with `cargo run --example append_dataset`.

### Streaming appends

`append_dataset` re-reads the whole file and rebuilds the chunk index on every `commit`, which is the right trade for a one-off append composed alongside other edits, but not for a high-frequency append loop. For that, open the file **once** with `File::open_rw` and append many times through a `Dataset` handle, growing the Extensible-Array index *in place* — so each append costs `O(appended bytes)` plus amortized `O(1)` index overhead, with no whole-file re-read and no index rebuild.

```rust
use hdf5_pure::File;

let file = File::open_rw("log.h5").unwrap();
let mut samples = file.dataset("samples").unwrap();
samples.append(&[8i32, 9, 10, 11]).unwrap();
samples.append(&[12i32, 13]).unwrap(); // unfiltered: any length
file.close().unwrap();
```

One open file reaches every dataset by name, takes an exclusive file lock for its lifetime, and sets no SWMR flag. **Every `append` is crash-atomic**: writes are ordered child-before-parent with `fsync` barriers and the dataspace dimension is published last as the single commit point, so a crash between appends leaves either the previous length or the new one — never a torn or lost view.

That atomicity is why filtered and unfiltered datasets have different length rules. An **unfiltered** append may be **any length**: when the current length is not chunk-aligned, the trailing partial chunk is rewritten and its index element — a single chunk address — is repointed with one atomic write. A **filtered** append must be **chunk-aligned** (the current length and the appended length both whole multiples of the chunk length), because a filtered index element is a multi-field record whose in-place repoint is not power-loss atomic; a filtered append therefore only ever inserts new chunks. For a non-chunk-aligned filtered append, use `append_dataset`, which rebuilds the index and repoints the superblock last (fully atomic).

The remaining eligibility rules match `append_dataset` (chunked, unlimited axis 0, Extensible-Array index, rank 1, a re-encodable filter pipeline), with one difference: because it grows the index in place rather than rebuilding it, the index must already be allocated. This crate allocates it eagerly, so an empty dataset it wrote can be grown from the first append; an empty dataset the C library created without any initial data defers its index and is refused — make that first append with `append_dataset` (which materializes the index), or create the dataset with initial data. The dead bytes left when an unfiltered partial chunk is relocated are reclaimed by [repack](repack.md) rather than reused within the session in this release. This is the throughput-oriented counterpart to `append_dataset` and the filter-capable counterpart to the [SWMR writer](swmr.md).

!!! note "Deprecated: `AppendWriter`"
    The standalone `AppendWriter` type provided this same in-place streaming append before the owned-handle API existed. It is now **deprecated** in favor of `File::open_rw` + `Dataset::append` — identical mechanics, rules, and crash-atomicity — and will be removed in a later release. The one behavior without a `File::open_rw` equivalent yet is opening with file locking disabled (`AppendWriter::open_with_locking`).

!!! tip "Runnable example"
    This section mirrors [`examples/append_streaming.rs`](https://github.com/stephenberry/hdf5-pure/blob/main/examples/append_streaming.rs). Run it with `cargo run --example append_streaming`.

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

Rather than silently degrade a file, `EditSession` refuses anything it cannot reproduce faithfully, returning `Error::EditUnsupported`:

- A userblock or non-zero base address.
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
