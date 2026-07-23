# Streaming Large Files

This page covers reading HDF5 files that are too large to buffer in memory. `File::open` loads the whole file into RAM; `File::open_streaming` fetches metadata and dataset chunks from disk on demand, so peak memory tracks the data you actually read rather than the size of the file.

## Why stream

`File::open(path)` reads the entire file into memory before you touch any dataset. That is simple and fast for files that comfortably fit in RAM, but it does not scale to files that exceed available memory, for example a multi-gigabyte file produced on a 32-bit host where it exceeds the address space.

`File::open_streaming(path)` opens the same file with a lazy backing store. It fetches metadata and dataset chunks from the file as they are needed instead of buffering it whole, so it never holds the entire file in memory at once. Peak memory tracks what you actually read: one dataset, decompressed, with its chunks fetched on demand, plus the metadata being parsed.

```rust
use hdf5_pure::File;

let file = File::open_streaming("huge.h5").unwrap();
let ds = file.dataset("signal").unwrap();
let values = ds.read_f64().unwrap();  // only this dataset's chunks are read
```

The reading API is identical to `File::open`; only the backing store differs. Everything you can do with an in-memory file (see [Reading datasets](reading.md)) applies here too.

!!! note
    `open_streaming` requires the `std` filesystem and is therefore unavailable in `no_std` builds. See [Features](../reference/features.md) for the feature matrix.

## What streaming supports

Dataset reads are fully supported across every storage layout:

| Layout | Supported when streaming |
| --- | --- |
| Contiguous | Yes |
| Compact | Yes |
| Chunked (B-tree v1 index) | Yes |
| Chunked (fixed array index) | Yes |
| Chunked (extensible array index) | Yes |

!!! warning
    Two limits apply to the streaming backend that in-memory reading does not have:

    - Only latest-format (v2) groups resolve along a path. A v1 symbol-table group is rejected.
    - Reading attributes is not yet supported.

    If a file uses v1 groups or you need attributes, open it with `File::open` instead.

Streaming opens are read-only. To **append** to a file with the same bounded-memory discipline, open it with [`File::open_rw_bounded`](editing.md#bounded-memory-appends) — the read-write sibling of `open_streaming`, sharing this backend's read capabilities and the `FileAccessOptions` cache budgets below.

## Reading a large dataset a window at a time

The whole-dataset reads above materialize one dataset in full. When a single dataset is itself too large to hold decompressed, read it in **row windows**: `read_raw_rows(start, count)` and the typed `read_f64_rows` / … / `read_string_rows` decode only the leading-dimension rows `[start, start + count)`, touching only the chunks that window overlaps. Peak memory then tracks the window (plus one chunk), not the dataset.

```rust
use hdf5_pure::File;

let file = File::open_streaming("huge.h5").unwrap();
let ds = file.dataset("signal").unwrap();
let rows = ds.shape().unwrap()[0];

// Process a million rows at a time, never holding the whole dataset.
for start in (0..rows).step_by(1_000_000) {
    let window = ds.read_f64_rows(start, 1_000_000).unwrap();
    // ... process `window` ...
    let _ = window;
}
```

The window is clamped to the dataset, so the final short window needs no special-casing. See [Reading a row window](reading.md#reading-a-row-window) for the full method list and the fallback for variable-length string datasets.

## Tuning retained memory

`File::open_streaming_with_options(path, FileAccessOptions)` bounds the memory the streaming backend retains. `FileAccessOptions::new()` returns the crate's default access behavior; you layer on two independent caches with its builder methods.

`MetadataCacheConfig` mirrors the memory-budget role of HDF5's `H5Pset_mdc_config`: it caps the bytes retained for parsed metadata reads. `MetadataCacheConfig::new(max_bytes)` sets the total byte budget, and `.with_max_entry_bytes(...)` caps the size of any single cached metadata read so one large heap or index block cannot monopolize the cache.

`ChunkCacheConfig` mirrors the raw-data chunk-cache settings from `H5Pset_cache`. `ChunkCacheConfig::from_h5p_cache(rdcc_nslots, rdcc_nbytes)` builds one directly from the familiar HDF5 slot count and byte budget. It controls decompressed chunk data and whether parsed chunk indexes are retained between repeated reads of the same dataset.

```rust
use hdf5_pure::{ChunkCacheConfig, File, FileAccessOptions, MetadataCacheConfig};

let options = FileAccessOptions::new()
    .with_metadata_cache(MetadataCacheConfig::new(8 * 1024 * 1024).with_max_entry_bytes(64 * 1024))
    .with_chunk_cache(ChunkCacheConfig::from_h5p_cache(521, 256 * 1024));
let file = File::open_streaming_with_options("huge.h5", options).unwrap();
```

The chunk cache configured here is the file-wide default; it applies to every dataset opened from this file. The metadata cache only affects streaming opens, since an in-memory open already holds the whole file in one buffer.

| Config | HDF5 analogue | Controls |
| --- | --- | --- |
| `MetadataCacheConfig` | `H5Pset_mdc_config` (memory budget) | Bytes retained for parsed metadata reads |
| `ChunkCacheConfig` (file-wide) | `H5Pset_cache` raw-data settings | Decompressed chunk bytes and retained chunk indexes, as the default for all datasets |
| `ChunkCacheConfig` (per dataset) | `H5Pset_chunk_cache` | Same, overridden for one dataset |

### Per-dataset overrides

To override the chunk cache for a single dataset, open it with `dataset_with_options(name, DatasetAccessOptions)`. This is the analogue of HDF5's per-dataset access property list (`H5Pset_chunk_cache`). The override replaces the file-wide default for that one dataset; other datasets keep the default. A dataset that is read once front-to-back, for instance, gains nothing from caching its decompressed chunks, so you can disable the cache with `ChunkCacheConfig::disabled()`:

```rust
use hdf5_pure::{ChunkCacheConfig, DatasetAccessOptions, File};

let file = File::open("data.h5").unwrap();
// This dataset is read once front-to-back: skip caching its decompressed chunks.
let dapl = DatasetAccessOptions::new().with_chunk_cache(ChunkCacheConfig::disabled());
let ds = file.dataset_with_options("scan", dapl).unwrap();
let values = ds.read_f64().unwrap();
```

`dataset_with_options` is available on both `File` and `Group`. `Dataset::chunk_cache_config()` reports the effective `ChunkCacheConfig` for an opened dataset (the analogue of `H5Pget_chunk_cache`): the per-dataset override when one was supplied, otherwise the file-wide default.

!!! tip
    `DatasetAccessOptions::new()` inherits every file-wide access default, so you only set what you want to change.

## Confirming cache behavior

To confirm a cache is behaving as configured, `Dataset::chunk_cache_stats()` returns a read-only `ChunkCacheStats` snapshot taken after a read. It reports whether the parsed index is loaded (`index_loaded()`), how many decompressed chunks are retained (`cached_chunks()`), and how many bytes of chunk data are retained (`cached_bytes()`).

```rust
use hdf5_pure::File;

let file = File::open("data.h5").unwrap();
let ds = file.dataset("signal").unwrap();
let _ = ds.read_f64().unwrap();
let stats = ds.chunk_cache_stats();
// "signal" here is a chunked dataset, so chunks are retained for reuse;
// a contiguous or compact dataset has no chunk cache and reports zero.
assert!(stats.cached_chunks() > 0);
```

The counts are a point-in-time view and change as further reads populate or evict chunks. A disabled cache, or one over its byte or slot budget, reports fewer or no retained chunks.

## Related topics

- [Reading datasets](reading.md) for the dataset read API that is shared between in-memory and streaming opens.
- [Variable-length strings](vlen-strings.md) for reading string datasets.
- [Features](../reference/features.md) for the `std` feature requirement.
