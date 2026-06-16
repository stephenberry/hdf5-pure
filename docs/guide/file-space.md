# File-Space Strategy

A written HDF5 file can record how it manages free space: which allocation strategy it uses, whether freed regions are tracked across closes, and the page size used for paged allocation. This page covers `FileBuilder::with_file_space_strategy` and `with_file_space_page_size`, which mirror HDF5's `H5Pset_file_space_strategy` and `H5Pset_file_space_page_size`, and how the choice interacts with [editing](editing.md) and [repacking](repack.md).

!!! tip
    A runnable example lives at [examples/file_space.rs](https://github.com/stephenberry/hdf5-pure/blob/main/examples/file_space.rs). Run it with:

    ```bash
    cargo run --example file_space
    ```

## Recording a strategy

`FileBuilder::with_file_space_strategy(strategy, persist, threshold)` records the file-space management strategy, whether free space is persisted across closes, and the smallest free section the managers will track. `with_file_space_page_size(size)` sets the file-space page size used for paged allocation. Both are recorded in the file's superblock extension, so the reference HDF5 C library and a later reopen observe the choice.

```rust
use hdf5_pure::{File, FileBuilder, FileSpaceStrategy};

let mut b = FileBuilder::new();
b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
b.with_file_space_strategy(FileSpaceStrategy::Page, false, 1) // strategy, persist, threshold
    .with_file_space_page_size(8192);
b.write("out.h5").unwrap();

assert_eq!(
    File::open("out.h5").unwrap().file_space_strategy(),
    Some(FileSpaceStrategy::Page)
);
```

## Strategy variants

The `FileSpaceStrategy` enum mirrors HDF5's `H5F_fspace_strategy_t`:

| Variant | HDF5 constant | Behavior |
|---|---|---|
| `FileSpaceStrategy::FsmAggr` | `H5F_FSPACE_STRATEGY_FSM_AGGR` | Free-space managers, aggregators, and the virtual file driver — the HDF5 default. |
| `FileSpaceStrategy::Page` | `H5F_FSPACE_STRATEGY_PAGE` | Paged aggregation backed by free-space managers. |
| `FileSpaceStrategy::Aggr` | `H5F_FSPACE_STRATEGY_AGGR` | Aggregators and the virtual file driver only, no free-space managers. |
| `FileSpaceStrategy::None` | `H5F_FSPACE_STRATEGY_NONE` | No free-space tracking; allocation only ever appends. |

## The threshold parameter

The `threshold` argument is the smallest free-space section, in bytes, that the free-space managers will track. Sections below this size are not recorded. The C library's default is `1` (every freed section is eligible for tracking).

## Reading the strategy back

The strategy lives in a superblock-extension message (a standalone object header the superblock points at), so it survives a reopen. `File::file_space_strategy()` returns the recorded `FileSpaceStrategy`, and `File::file_space_info()` returns the full `FileSpaceInfo` record (persist flag, threshold, page size, and the free-space manager addresses).

```rust
use hdf5_pure::File;

let file = File::open("out.h5").unwrap();
if let Some(info) = file.file_space_info() {
    println!("strategy: {:?}", info.strategy);
    println!("persist:  {}", info.persist);
    println!("threshold: {}", info.threshold);
    println!("page size: {}", info.page_size);
}
```

`file_space_strategy()` (and `file_space_info()`) return `None` when the file records no strategy, which is what the C library also writes when the default is left in place.

## Persisting free space across sessions

Passing `persist = true` records that freed space should be tracked on disk across closes. A brand-new file has nothing to track, so this only records the intent. When a later [`EditSession`](editing.md) frees a region — for example by deleting a dataset — the freed region is recorded in on-disk free-space-manager blocks (`FSHD`/`FSSE`) rather than discarded. Later sessions, both this crate's and the reference C library's, recover and reuse those regions.

`File::persisted_free_space()` returns the tracked free regions as `(address, length)` pairs sorted by address, and `EditSession::open` seeds its free list from them so reuse spans sessions rather than just the open session.

```rust
use hdf5_pure::{EditSession, File, FileBuilder, FileSpaceStrategy};

let mut builder = FileBuilder::new();
builder.create_dataset("keep").with_i32_data(&[1, 2, 3]);
builder
    .with_file_space_strategy(FileSpaceStrategy::Page, true, 1) // persist = true
    .with_file_space_page_size(4096);
builder.write("managed.h5").unwrap();

// Create then delete a dataset; the freed region is recorded on disk.
let mut session = EditSession::open("managed.h5").unwrap();
session.create_dataset("scratch").with_f64_data(&vec![0.0; 4096]);
session.commit().unwrap();

let mut session = EditSession::open("managed.h5").unwrap();
session.delete("scratch");
session.commit().unwrap();

// Later opens can read the persisted free regions.
let file = File::open("managed.h5").unwrap();
let free = file.persisted_free_space();
let total_free: u64 = free.iter().map(|&(_, len)| len).sum();
println!("persisted free regions: {}", free.len());
println!("total persisted free bytes: {total_free}");
```

!!! note
    `persisted_free_space()` is empty when the file does not persist free space, and for the streaming backend (which does not load the manager blocks). The addresses are file offsets relative to the base address; [reading data](reading.md) is unaffected by the presence or absence of these managers.

When free space is not persisted, an `EditSession` still reuses space within a single session but does not carry a free list across closes. If churn has left a file with unused gaps and you want to reclaim them outright, [repacking](repack.md) rewrites the file compactly.
