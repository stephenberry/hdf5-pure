# Editing in Place

`EditSession` opens an existing HDF5 file and adds, copies, or deletes objects, or edits group attributes, without reading the whole file in and rewriting it. New data and rebuilt object headers are appended at the end of the file and the superblock is repointed last, so the cost is proportional to what changes rather than to the file size, and a failed commit leaves the original file valid.

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
| `set_group_attr(path, name, value)` | Stage adding or replacing a compact group attribute | — |
| `remove_group_attr(path, name)` | Stage removing a compact group attribute | — |
| `copy(src, dst)` | Stage a deep copy of a dataset or whole group subtree | `H5Ocopy` |
| `delete(path)` | Stage removing the link at `path` (and, for a group, its whole subtree) | `H5Ldelete` |
| `commit()` | Apply all staged operations in place and flush | — |

`create_dataset` returns the same `DatasetBuilder` used by [`FileBuilder`](writing.md), so you configure the new dataset exactly as you would when creating a file from scratch:

```rust
session.create_dataset("run2/signal").with_f64_data(&[1.0, 2.0, 3.0]);
```

`set_group_attr` takes an `AttrValue`. The group it names may already exist or may be created earlier in the same session; `""` or `"/"` names the root group. Attributes are stored compactly in the rebuilt group header, and variable-length attributes or edits that would exceed the compact-attribute limit are refused before any file bytes change.

`copy` performs a deep copy: fresh copies of every object's data and header are written, internal links and the contiguous data address are repointed to the copies, and a link named by the last component of `dst` is added to its parent group. The original is untouched. `src` must exist and `dst` must not (and may not lie inside `src`).

## How it works

`commit()` appends each new dataset (its data blob and object header) and each new group, then appends rewritten object headers for every touched group and its ancestors up to the root (omitting any deleted links), and finally repoints the superblock at the new root.

The appended data is `fsync`ed before the root is repointed, so the "repoint last" guarantee is real: if the process or machine fails during a commit, the original file is still intact and readable, because the superblock still points at the old root. The cost of a commit scales with the size of the edit, not the size of the file.

!!! warning "All-or-nothing safety"
    Every check runs before the first byte is written. On any `Error::EditUnsupported`, the file on disk is left untouched. This makes editing safe to attempt: an unsupported edit fails cleanly rather than producing a partially modified or corrupt file.

## Supported targets and formats

Contiguous, unfiltered datasets and compact-link groups are supported. The editor works across every on-disk format the reference HDF5 C library and h5py produce:

- Version 0, 1, 2, and 3 superblocks.
- Single- and multi-chunk object headers. A multi-chunk header is collapsed into a single chunk on rewrite.
- A version 0/1 symbol-table group on the edited path is converted to the latest compact-link format. Adding and deleting are supported on these older files; copying a version-1 object is not.

Rather than silently degrade a file, `EditSession` refuses anything it cannot reproduce faithfully, returning `Error::EditUnsupported`:

- A userblock or non-zero base address.
- Chunked or compressed dataset additions.
- Dense-storage headers on the edited path.
- Copying an existing version-1 object.

See [`Error::EditUnsupported`](../reference/data-types.md) for the full set of refusals.

## Space reuse and truncation

Within a session, the space a deletion frees is reused for later writes in the same commit, so add/delete churn stays bounded instead of only ever growing the file. If a freed run reaches the end of the file, the file is truncated.

Reclaim is best-effort: an object whose blocks cannot be enumerated exhaustively (chunked or variable-length storage, dense attribute or link heaps) is left as dead bytes rather than risk freeing a region still in use.

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
