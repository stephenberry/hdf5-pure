# Reclaiming Space (repack)

`repack` rewrites a whole HDF5 file into a fresh, compact copy, optionally dropping objects on the way. It is the guaranteed-shrink answer to a fundamental limitation of in-place editing: deleting an object cannot always return its bytes to the operating system.

!!! tip "Runnable example"
    This page is backed by [`examples/repack.rs`](https://github.com/stephenberry/hdf5-pure/blob/main/examples/repack.rs). Run it with:

    ```bash
    cargo run --example repack
    ```

## Why a delete cannot always shrink a file

Deleting an object inside an [`EditSession`](editing.md) reuses the freed space *within that session*, and the file is truncated when the freed bytes happen to reach the very end. But a single delete-then-close cannot shrink a file whose freed region sits in the middle: an HDF5 file is a single address space, and a hole in the middle cannot be removed by truncating the tail. This is the same reason the HDF5 C library ships a separate `h5repack` tool rather than relying on deletion alone.

`repack` solves this by reading every surviving object and rewriting the whole file from scratch through [`FileBuilder`](writing.md), so the result has no dead space and is strictly smaller when objects are dropped.

## Basic usage

`repack(src, dst, &RepackOptions)` reads every object of `src` not excluded by the options and writes them into a fresh, compact file at `dst`.

```rust
use hdf5_pure::{repack, RepackOptions};

// Pure compaction copy: drop nothing, just remove dead space.
repack("input.h5", "compact.h5", &RepackOptions::new()).unwrap();
```

## Dropping objects

`RepackOptions::new()` starts from a pure-compaction copy. `RepackOptions::drop_path(path)` adds a path to omit from the output and is chainable. Dropping a group drops its whole subtree.

```rust
use hdf5_pure::{repack, RepackOptions};

// Drop a dataset and a whole group subtree, then write a fresh, compact file.
let options = RepackOptions::new()
    .drop_path("scratch")
    .drop_path("runs/aborted");
repack("input.h5", "compact.h5", &options).unwrap();
```

Leading and trailing slashes in a drop path are ignored, so `"grp/old"` and `"/grp/old"` are equivalent.

!!! warning "Every drop path must exist"
    A drop path that does not match any object in the source fails the repack rather than being silently ignored — a no-op drop is treated as a mistake. The error is reported as `Error::RepackUnsupported`, and no output file is written.

## The fidelity guarantee

`repack` never silently degrades data. Every surviving object is reproduced byte-for-byte — datatype, shape, max-shape, chunking, supported filters, raw element data, and attributes — or the whole operation fails with `Error::RepackUnsupported` naming the object and the reason. It refuses rather than approximate.

The operation is all-or-nothing: the entire source is validated and staged in memory before the first byte is committed, so on any failure nothing is written to `dst` and no partial output file is left behind.

### What it reproduces

| Aspect | Supported |
| --- | --- |
| Datatypes | fixed-point, floating-point, fixed-length string, bit-field, opaque, compound, enumeration, array |
| Layout | contiguous / compact or chunked |
| Filters | deflate, shuffle, fletcher32, and/or lossless integer scale-offset |
| Structure | group hierarchy of arbitrary depth |
| Attributes | numbers, fixed- and variable-length strings and their arrays, on datasets, groups, and root |
| File-space strategy | the source's strategy, page size, and threshold (carried forward as non-persistent) |

A repacked file has no free space to persist, so even when the source recorded a persistent file-space strategy the compact output carries that strategy forward as non-persistent. See [File-space strategy](file-space.md) for what that controls.

!!! note "Lossless filters only"
    `repack` reads each dataset's *decompressed* bytes and re-applies its filters. It can therefore reproduce only **lossless** filters, where the re-encoded chunks decompress to the exact same bytes. This includes deflate, shuffle, fletcher32, and lossless integer scale-offset. See [Compression](compression.md) for the full filter list.

### What it refuses (by name)

These are reported as `Error::RepackUnsupported` naming the object, never silently dropped or degraded:

| Refused | Reason |
| --- | --- |
| variable-length datatypes | not reproducible faithfully yet |
| time datatypes | byte order is not modelled |
| reference datatypes | stored absolute addresses would go stale on rewrite |
| virtual and external data layouts | not reproducible by rewriting |
| lossy filters: float D-scale scale-offset and ZFP | re-encoding is not guaranteed idempotent |
| SZIP filter | this crate cannot write it |
| an attribute the reader cannot decode | e.g. an enumeration, compound, or boolean attribute |

## Verifying the result

After a repack, the surviving objects open exactly as before and the dropped objects are gone. Adapting the example:

```rust
use hdf5_pure::File;

let file = File::open("compact.h5").unwrap();
let keep = file.dataset("keep").unwrap().read_f64().unwrap();
assert_eq!(keep, vec![1.0, 2.0, 3.0]);

// Dropped objects are absent.
assert!(file.dataset("scratch").is_err());
assert!(file.group("runs").is_err());
```

## Repack vs. in-place editing

| | `EditSession` delete | `repack` |
| --- | --- | --- |
| Reclaims space mid-session | Yes (reused for later writes) | n/a |
| Shrinks a closed file | Only if freed bytes reach the end | Always |
| Spans a reopen | No | Yes (writes a new file) |
| Output | edits the same file | a fresh file at `dst` |

For incremental edits where add/delete churn stays bounded, prefer an [`EditSession`](editing.md). For guaranteed compaction across a reopen, or to drop objects and reclaim their space unconditionally, use `repack`.
