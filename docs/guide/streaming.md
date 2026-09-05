# Streaming Large Files

This page covers working with HDF5 files that are too large to buffer in memory, in both directions. Reading comes first: `File::open` loads the whole file into RAM, while `File::open_streaming` fetches metadata and dataset chunks from disk on demand, so peak memory tracks the data you actually read rather than the size of the file. [Writing without buffering](#writing-without-buffering) covers the other direction.

## Why stream

`File::open(path)` reads the entire file into memory before you touch any dataset. That is simple and fast for files that comfortably fit in RAM, but it does not scale to files that exceed available memory, for example a multi-gigabyte file produced on a 32-bit host where it exceeds the address space.

`File::open_streaming(path)` opens the same file with a lazy backing store. It fetches metadata and dataset chunks from the file as they are needed instead of buffering it whole, so it never holds the entire file in memory at once. Peak memory tracks what you actually read: one dataset, decompressed, with its chunks fetched on demand, plus the metadata being parsed.

Chunks that lie next to each other on disk are fetched together, in reads of at most 256 KiB. This is what makes a file written a row at a time readable at a sensible speed: such a file carries one small chunk per row, so fetching them one at a time costs thousands of reads where a handful would do. A recording of 73 datasets over 280 rows holds 20,440 chunks of 32 bytes, which the streaming reader fetches in 73 reads. A read never fetches bytes outside the chunks it needs, so what this trades is read count, not read volume. A chunk larger than 256 KiB is read on its own, as it was before.

```rust
use hdf5_pure::File;

let file = File::open_streaming("huge.h5").unwrap();
let ds = file.dataset("signal").unwrap();
let values = ds.read_f64().unwrap();  // only this dataset's chunks are read
```

The reading API is identical to `File::open`; only the backing store differs. Everything you can do with an in-memory file (see [Reading datasets](reading.md)) applies here too.

!!! note
    `open_streaming` requires the `std` filesystem and is therefore unavailable in `no_std` builds. See [Features](../reference/features.md) for the feature matrix.

## Streaming from something that is not a path

`File::from_source(source)` opens the same lazy backing store over anything that implements `Source`, for a caller whose bytes are not a file: an object store addressed by HTTP range request, a WebAssembly guest that receives byte ranges from its host and has no filesystem at all, a decrypting layer.

`Source` asks for two answers — the total length, and the bytes at an absolute offset — which is all the reader ever asks of a file:

```rust
use hdf5_pure::{File, FormatError, Source};

struct Remote {
    len: u64,
}

impl Source for Remote {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        // However the bytes arrive: a range request, a host call, a decrypting
        // layer. Fill the whole request or fail — a short read is an error.
        let bytes = fetch(offset, buf.len()).map_err(FormatError::Source)?;
        buf.copy_from_slice(&bytes);
        Ok(())
    }
}

let file = File::from_source(Remote { len: 1 << 30 })?;
let values = file.dataset("signal")?.read_f64()?;
```

A `Read + Seek` needs no implementation of its own: `ReadSeekSource` wraps one. (Doing that to a `std::fs::File` is exactly `File::open_streaming`, which wraps it for you.)

Two things are worth knowing before pointing this at a remote store:

- **Turn the metadata cache on**, through `File::from_source_with_options`. It is off by default, and without one every read a parser makes is a round trip. See `MetadataCacheConfig`.
- **A file the superblock marks as held by a writer is refused here too**, with `Error::FileMarkedInUse`. Some of the recoveries a path open would name do not apply: `File::open_swmr` and `File::clear_swmr_flag` both need a path. What a source can take is `FileAccessProperties::with_write_mark_policy(WriteMarkPolicy::AllowSnapshot)`, which reads a file marked by a non-SWMR writer that has flushed (see [Reading](reading.md)), or `File::from_bytes` with the whole file in memory. The refusal says which.

## What streaming supports

Dataset reads are fully supported across every storage layout:

| Layout | Supported when streaming |
| --- | --- |
| Contiguous | Yes |
| Compact | Yes |
| Chunked (B-tree v1 index) | Yes |
| Chunked (fixed array index) | Yes |
| Chunked (extensible array index) | Yes |

!!! note
    The streaming backend resolves both group forms (v2 and v1 symbol-table) and reads compact, dense, shared, and variable-length attributes, same as `File::open`. The remaining differences:

    - `File::as_bytes` returns an empty slice (there is no whole-file buffer), and `persisted_free_space` returns no regions (the free-space-manager blocks are not loaded).
    - A streaming file cannot be the **source** of a cross-file [`copy_from`](editing.md) — that copy requires a buffered source.
    - Chunk decompression is sequential.

Streaming opens are read-only. To **append** to a file with the same bounded-memory discipline, open it with [`File::open_rw`](editing.md#bounded-memory-appends), which edits a latest-format file bounded — the read-write sibling of `open_streaming`, sharing this backend's read capabilities and the `FileAccessProperties` cache budgets below.

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

The window is clamped to the dataset, so the final short window needs no special-casing. See [Reading a row window](reading.md#reading-a-row-window) for the full method list.

A window need not be rows. `read_raw_region(start, count)` and the typed `read_*_region` forms read a box along every dimension — a tile, a frame, a band along an axis that is not the first — touching only the chunks the box meets, under the same bound. A dataset whose leading axis has length one, as netCDF writers record a single time step, streams along its next axis this way where a row window would be the whole plane. See [Reading a region](reading.md#reading-a-region).

## Tuning retained memory

`File::open_streaming_with_options(path, FileAccessProperties)` bounds the memory the streaming backend retains. `FileAccessProperties::new()` returns the crate's default access behavior; you layer on two independent caches with its builder methods.

`MetadataCacheConfig` mirrors the memory-budget role of HDF5's `H5Pset_mdc_config`: it caps the bytes retained for parsed metadata reads. `MetadataCacheConfig::new(max_bytes)` sets the total byte budget, and `.with_max_entry_bytes(...)` caps the size of any single cached metadata read so one large heap or index block cannot monopolize the cache. It is disabled unless you ask for it, and worth asking for on a file holding many datasets: opening one walks the root group's link storage, and every dataset repeats that walk. Entries are indexed rather than searched, so raising the budget costs memory and not lookup time.

`ChunkCacheConfig` mirrors the raw-data chunk-cache settings from `H5Pset_cache`. `ChunkCacheConfig::from_h5p_cache(rdcc_nslots, rdcc_nbytes)` builds one directly from the familiar HDF5 slot count and byte budget. It controls decompressed chunk data and whether parsed chunk indexes are retained between repeated reads of the same dataset.

```rust
use hdf5_pure::{ChunkCacheConfig, File, FileAccessProperties, MetadataCacheConfig};

let access = FileAccessProperties::new()
    .with_metadata_cache(MetadataCacheConfig::new(8 * 1024 * 1024).with_max_entry_bytes(64 * 1024))
    .with_chunk_cache(ChunkCacheConfig::from_h5p_cache(521, 256 * 1024));
let file = File::open_streaming_with_options("huge.h5", access).unwrap();
```

The chunk cache configured here is the file-wide default; it applies to every dataset opened from this file. The metadata cache only affects streaming opens, since an in-memory open already holds the whole file in one buffer.

| Config | HDF5 analogue | Controls |
| --- | --- | --- |
| `MetadataCacheConfig` | `H5Pset_mdc_config` (memory budget) | Bytes retained for parsed metadata reads |
| `ChunkCacheConfig` (file-wide) | `H5Pset_cache` raw-data settings | Decompressed chunk bytes and retained chunk indexes, as the default for all datasets |
| `ChunkCacheConfig` (per dataset) | `H5Pset_chunk_cache` | Same, overridden for one dataset |

### Per-dataset overrides

To override the chunk cache for a single dataset, open it with `dataset_with_options(name, DatasetAccessProperties)`. This is the analogue of HDF5's per-dataset access property list (`H5Pset_chunk_cache`). The override replaces the file-wide default for that one dataset; other datasets keep the default. A dataset that is read once front-to-back, for instance, gains nothing from caching its decompressed chunks, so you can disable the cache with `ChunkCacheConfig::disabled()`:

```rust
use hdf5_pure::{ChunkCacheConfig, DatasetAccessProperties, File};

let file = File::open("data.h5").unwrap();
// This dataset is read once front-to-back: skip caching its decompressed chunks.
let dapl = DatasetAccessProperties::new().with_chunk_cache(ChunkCacheConfig::disabled());
let ds = file.dataset_with_options("scan", dapl).unwrap();
let values = ds.read_f64().unwrap();
```

`dataset_with_options` is available on both `File` and `Group`. `Dataset::chunk_cache_config()` reports the effective `ChunkCacheConfig` for an opened dataset (the analogue of `H5Pget_chunk_cache`): the per-dataset override when one was supplied, otherwise the file-wide default.

!!! tip
    `DatasetAccessProperties::new()` inherits every file-wide access default, so you only set what you want to change.

## Confirming cache behavior

### The chunk cache

To confirm a cache is behaving as configured, `Dataset::chunk_cache_stats()` returns a read-only `ChunkCacheStats` snapshot. Occupancy is a point-in-time view: whether the parsed index is loaded (`index_loaded()`), how many decompressed chunks are retained (`cached_chunks()`), and how many bytes they occupy (`cached_bytes()`). The counters beside it are cumulative since the handle was opened, or since the last `reset_chunk_cache_stats()`.

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

**Read `rejections()` and `evictions()` together, not one or the other.** Which of the two moves depends on how the dataset is read, and each is structurally zero on the other's path:

| read | budget signal | stays zero |
|---|---|---|
| whole (`read_f64` and friends) | `rejections()` | `evictions()` |
| row window (`read_*_rows`) | `evictions()` | `rejections()` |

A whole read visits each of its chunks exactly once, so once it has filled the cache it stops offering rather than giving back chunks it has already placed or been served for chunks it will not ask for again: a dataset eight times the budget reports seven eighths of its chunks rejected and no evictions at all. Reading it again hits the retained chunks and rejects the rest, so a settled cache costs nothing to keep. A row window keeps the plain LRU rule, because the chunk its successor needs is the one it finished on. Either counter climbing while `hit_rate()` stays low is the same finding.

```rust
use hdf5_pure::File;

let file = File::open("data.h5").unwrap();
let ds = file.dataset("signal").unwrap();

// The reads that fill a cache miss by definition, so a rate measured over the
// whole run charges the steady state for the warm-up.
let _ = ds.read_f64().unwrap();
ds.reset_chunk_cache_stats();
let _ = ds.read_f64().unwrap();

let stats = ds.chunk_cache_stats();
if stats.rejections() > 0 || stats.evictions() > 0 {
    // The working set did not fit: raise max_slots, max_bytes, or both.
}
if stats.oversize_chunks() > 0 {
    // One chunk is larger than the whole byte budget; more slots will not help.
}
println!("{:?} over {} lookups", stats.hit_rate(), stats.lookups());
```

Two figures need a word of care. `hit_rate()` returns `None` before any lookup rather than `0.0`, which is what a cache that missed everything — and a cache that was disabled — both report; read it beside `chunk_cache_config()`. And `invalidations()` is session-wide rather than per handle: a commit through *any* handle advances the file's content revision, so every dataset handle drops its chunks. A figure approaching `misses()` means the session is rewriting what it caches, and a larger budget will not change that.

### The metadata cache

A cache budget is a number chosen before a single read has happened. `File::metadata_cache_stats` reports what it bought, so the next number is measured rather than guessed. It is the counterpart of HDF5's `H5Fget_mdc_hit_rate` and `H5Fget_mdc_size`, and returns `None` when the open has no metadata cache to report on.

```rust
use hdf5_pure::{File, FileAccessProperties, MetadataCacheConfig};

let access = FileAccessProperties::new().with_metadata_cache(MetadataCacheConfig::new(8 << 20));
let file = File::open_streaming_with_options("huge.h5", access).unwrap();

// The reads that fill a cache miss by definition, so a rate measured over the
// whole run charges the steady state for the warm-up.
for name in file.root().datasets().unwrap() {
    let _ = file.dataset(&name).unwrap().read_raw().unwrap();
}
file.reset_metadata_cache_stats();

for name in file.root().datasets().unwrap() {
    let _ = file.dataset(&name).unwrap().read_raw().unwrap();
}
let stats = file.metadata_cache_stats().unwrap();
println!("{:?} over {} reads, {} of {} bytes held", stats.hit_rate(), stats.reads(), stats.bytes(), 8 << 20);
```

Which figure to read depends on what you are asking:

- **`hit_rate`** is the headline: the fraction of eligible reads served without touching the file. `None` means no eligible read has happened yet, which is not the same as a cache that has missed everything.
- **`evictions`** is what says a larger budget would help. A disappointing hit rate *with* evictions is a budget too small for the working set; the same hit rate with none is a workload that does not revisit metadata, and raising the budget will not change it.
- **`oversize_reads`** counts reads turned away, for exceeding `max_entry_bytes` or the budget itself, before the cache saw them. A file with large fractal heaps or index blocks can be missing from the cache entirely for this reason while the hit rate looks healthy.
- **`invalidations`** applies to a read-write session: entries dropped because a write overlapped them. Approaching the miss count, it means the session is rewriting the metadata it is caching.

`reset_metadata_cache_stats` clears the counters and evicts nothing, so occupancy carries across it.

Set the budget generously; the [property-support reference](../reference/property-support.md#the-metadata-cache-h5pset_mdc_config) covers why `H5AC_cache_config_t`'s adaptive-resize policy is not modeled, field by field.

## Writing without buffering

`FileBuilder::finish()` returns the assembled file, so writing an N-byte file costs N bytes of output on top of whatever the data already cost. `FileBuilder::finish_to(w)` writes the same bytes onto any `io::Write` instead, and `FileBuilder::write(path)` is `finish_to` onto a file.

```rust
use hdf5_pure::FileBuilder;

let mut builder = FileBuilder::new();
builder.create_dataset("x").with_f64_data(&[1.0, 2.0, 3.0]);

let mut sink: Vec<u8> = Vec::new();
builder.finish_to(&mut sink).unwrap();
```

This works because the writer computes every object's address — object headers, data blocks, indexes — *before* it emits a byte, then writes the file in ascending-address order. It never seeks back to patch an address, which is what a backpatching writer would have to do. So the destination can be anything that accepts bytes forward-only: a socket, a pipe, a compressing wrapper, a hash. Both are the same internal assembly pass against different sinks — a `Vec<u8>` for `finish`, the caller's writer for `finish_to` — so they are byte-identical by construction rather than by agreement.

!!! warning
    A failure partway through leaves whatever was already written on the sink. With a non-seekable destination there is nothing to roll back, so if you need all-or-nothing, write to a temporary path and rename on success.

### Userblock content

A file with a userblock needs its header bytes to be part of what the writer emits, since a streaming write has nothing left to patch by the time it returns. `with_userblock_content` supplies them up front:

```rust
use hdf5_pure::FileBuilder;

let mut builder = FileBuilder::new();
builder.with_userblock(512);
builder.with_userblock_content(b"my wrapper format's header");
builder.create_dataset("x").with_f64_data(&[1.0]);
builder.write("wrapped.h5").unwrap();
```

The rest of the region stays zero-filled, and content longer than the userblock is refused with `FormatError::UserblockContentTooLarge` rather than allowed to displace the superblock. This is how the MATLAB v7.3 writer emits its 512-byte header (see [MATLAB interop](../interop/matlab.md#userblock-header)).

### Datasets that never become resident

`finish_to` removes the assembled file from peak memory, but not the data: `with_f64_data(&values)` still copies the slice into the builder. Two paths avoid that too.

**Repacking** an existing file streams each chunk from the source to the destination, verbatim and one at a time, without decoding or re-encoding it. See [Repacking](repack.md).

**Producing** a dataset's bytes at write time is available on the MATLAB writer as `MatBuilder::write_blocks`, which takes a `DataProducer` the writer calls once per block during emission. Layout works from the shape alone, so the producer is never called before the write begins. Paired with `MatBuilder::finish_to`, a `.mat` of any size is written in about one block of memory. See [writing more data than fits in memory](../interop/matlab.md#writing-more-data-than-fits-in-memory) for the full API, including why it is uncompressed-only and which array shape an acquisition should ask for.

## Related topics

- [Reading datasets](reading.md) for the dataset read API that is shared between in-memory and streaming opens.
- [Writing files](writing.md) for the `FileBuilder` workflow these output paths finish.
- [Variable-length strings](vlen-strings.md) for reading string datasets.
- [Features](../reference/features.md) for the `std` feature requirement.
