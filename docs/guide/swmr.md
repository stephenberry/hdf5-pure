# SWMR (Single Writer, Multiple Readers)

SWMR lets a single process append to an unlimited dataset in place while other processes read it concurrently, and it interoperates with the reference HDF5 C library and h5py in both directions. This page covers how to lay out a SWMR-capable dataset, append to it durably, follow it from a reader, and recover a file left flagged by a writer that exited uncleanly.

!!! tip "Runnable example"
    A complete single-process demonstration lives in [`examples/swmr.rs`](https://github.com/stephenberry/hdf5-pure/blob/main/examples/swmr.rs). Run it with:

    ```bash
    cargo run --example swmr
    ```

    It writes and then reads in one process to show the mechanics; in practice the writer and reader are separate processes.

## How it works

The writer appends chunks and flushes them in dependency order, child structures before the parent metadata that references them, with a durability barrier after each step. The dataset's authoritative size (its dataspace dimension) is published last, as the single commit point: before it a reader sees the old length, after it the new one, and never a torn view. A reader therefore only ever observes a consistent prefix of the data. To pick up newly appended data, a reader re-reads with [`refresh()`](#following-a-growing-file).

Because the on-disk format is standard HDF5, a reader opened by this crate can follow a file being written by the reference C library or h5py in SWMR mode, and vice versa.

## Laying out the dataset

A SWMR-capable dataset must have one unlimited dimension and be chunked. The latest format indexes such a dataset with an Extensible Array, which is selected automatically. Create it with the usual [writing](writing.md) builder: set the initial extent with `with_shape`, mark the dimension unlimited with `with_maxshape(&[u64::MAX])`, and pick a chunk shape with `with_chunks`.

```rust
use hdf5_pure::FileBuilder;

let mut builder = FileBuilder::new();
builder
    .create_dataset("log")
    .with_i32_data(&[0, 1, 2])   // initial rows
    .with_shape(&[3])
    .with_maxshape(&[u64::MAX])  // one unlimited dimension
    .with_chunks(&[1]);
builder.write("stream.h5").unwrap();
```

## Appending in place

Open the existing file with `SwmrWriter::open` and append with one of the typed helpers. Each append call flushes durably, leaving the file valid for any concurrent reader throughout.

```rust
use hdf5_pure::SwmrWriter;

let mut writer = SwmrWriter::open("stream.h5").unwrap();
writer.append_i32("log", &[3, 4, 5]).unwrap();
writer.append_i32("log", &[6, 7]).unwrap();
writer.close().unwrap(); // clears the SWMR flag; or just drop the writer
```

`close()` clears the file's SWMR-write flag and flushes, marking the file cleanly closed. Prefer calling it over relying on `Drop`, so the rare flush error surfaces; dropping the writer also clears the flag.

The append helpers are:

| Method | Appends |
| --- | --- |
| `append_i32(dataset, &[i32])` | `i32` values |
| `append_f64(dataset, &[f64])` | `f64` values |
| `append_raw(dataset, &[u8])` | raw little-endian element bytes |

The typed helpers encode their values as little-endian bytes and forward to `append_raw`. With any of them, the appended length must be a whole number of chunks and the dataset's current length must already be chunk-aligned.

!!! warning "Appends must be chunk-aligned"
    Both the dataset's current length and each appended length must be multiples of the chunk length. An append that is not chunk-aligned, or whose byte length is not a whole number of elements, returns an error and publishes nothing, so a reader still sees the prior consistent prefix. After such an error the writer should be dropped rather than reused, because its in-memory mirror may have advanced past what reached disk.

## Following a growing file

Open the file for reading with `File::open_swmr`, which retains a live filesystem handle so the reader can re-read appended data. The initial view is a consistent snapshot; call `refresh()` to advance to a newer one after the writer appends.

```rust
use hdf5_pure::File;

let mut file = File::open_swmr("stream.h5").unwrap();
let n = file.dataset("log").unwrap().shape().unwrap()[0];
// ... later, after the writer appends ...
file.refresh().unwrap();                 // re-read appended data
let ds = file.dataset("log").unwrap();
println!("now {} rows", ds.shape().unwrap()[0]);
```

`refresh()` is the SWMR reader's refresh primitive, analogous to the C library's `H5Drefresh` and h5py's `Dataset.refresh()`. After it returns, newly fetched `Dataset` and `Group` handles observe the appended chunks and extended dimensions. Existing handles borrow `&self`, so they must be dropped before calling `refresh()`; re-fetch them afterward, as shown above. See [reading](reading.md) for the dataset access APIs used here.

!!! note "Refresh cost"
    Each `refresh()` re-reads the entire file from disk (`O(file size)`) and re-validates the superblock checksum; a transient parse failure from catching a writer mid-flush is retried a bounded number of times. When following a large, steadily growing log, budget refresh frequency accordingly. `refresh()` returns an error if the file was not opened with `File::open_swmr` (the in-memory `File::from_bytes` path cannot refresh).

## Recovering a flagged file

While a `SwmrWriter` is open, the file's superblock carries an active-SWMR-writer flag (matching the reference C library and h5py) so concurrent readers may open it accordingly. `close()` or dropping the writer clears it. If a writer process exits without a clean close, the file is left flagged. Recover it with the h5clear equivalent:

```rust
use hdf5_pure::SwmrWriter;

SwmrWriter::clear_swmr_flag("stream.h5").unwrap();
```

`clear_swmr_flag` is safe to call on a file whose flag is already clear.

## Supported subset and requirements

SWMR append supports the following subset, distinct from the general [editing](editing.md) and writing paths:

| Requirement | Detail |
| --- | --- |
| Dimensionality | exactly one unlimited dimension |
| Storage | chunked (Extensible Array index, latest format) |
| Filters | unfiltered (no compression on the appended dataset) |
| Append granularity | chunk-aligned appends |
| File layout | no userblock (zero base address); latest-format v2/v3 superblock |
| Growth | unbounded |
| Build | requires `std` (the default); the in-memory/WASM path cannot refresh |

`SwmrWriter::open` rejects files that fall outside this subset, returning `Error::SwmrAppendUnsupported` for a non-latest-format superblock or a userblock file before performing any mutating write.
