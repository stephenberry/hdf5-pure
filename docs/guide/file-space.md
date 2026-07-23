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

The `threshold` argument is the smallest free-space section, in bytes, that the free-space managers are asked to track; the C library's default is `1` (every freed section is eligible). It is recorded in the file and round-trips through the reference C library. In this crate the value is currently **advisory**: the paged writer and the bounded editor track every page tail and freed section regardless of the recorded threshold, so a `threshold > 1` is preserved on disk but does not change which sections this crate records.

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

## Paged files (`FileSpaceStrategy::Page`)

`FileSpaceStrategy::Page` writes a **genuine paged file**, not just a recorded label. Every allocation is aligned to the page size set by `with_file_space_page_size` (default 4096; it must be a power of two `>= 512`, and any userblock must be a whole number of pages), metadata and raw data are kept in separate pages, and each page's free tail is tracked in a per-page-type free-space manager. The reference HDF5 C library reads the result as a paged file, parses the managers (`H5Fget_freespace` matches the tracked total), reads every dataset, and re-paginates the file when it writes to it.

```rust
use hdf5_pure::{FileBuilder, FileSpaceStrategy};

let mut b = FileBuilder::new();
b.create_dataset("samples")
    .with_i32_data(&(0..1000).collect::<Vec<i32>>())
    .with_shape(&[1000])
    .with_maxshape(&[u64::MAX])
    .with_chunks(&[256]);
b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0) // persist so it can be grown later
    .with_file_space_page_size(4096);
b.write("paged.h5").unwrap();
```

A paged file is grown in place only through [`File::open_rw_bounded`](editing.md#bounded-memory-appends), which keeps each new page homogeneous (raw and metadata never share a page) and rewrites the per-page-type managers at `File::close`. Growing one through the whole-file editor (`File::open_rw`, or the deprecated `EditSession`) is refused, because a whole-file commit would collapse the per-page-type managers into a single non-paged manager and lose the page alignment. A paged file created **without** `persist = true` has no on-disk record of which pages hold metadata versus raw data, so it cannot be grown at all — recreate it with `persist = true` if you need to append to it later.

## Persisting free space across sessions

Passing `persist = true` records that freed space should be tracked on disk across closes. For the non-paged strategies (`FsmAggr`, `Aggr`, `None`) a brand-new file has no free space, so this initially only records the intent; a genuine paged file (above) already tracks its page-tail free space from creation. When a later edit frees a region — for example by deleting a dataset — the freed region is recorded in on-disk free-space-manager blocks (`FSHD`/`FSSE`) rather than discarded, and a later session, this crate's or the reference C library's, recovers and reuses it.

`File::persisted_free_space()` returns the tracked free regions as `(address, length)` pairs sorted by address, and a read-write session seeds its free list from them so reuse spans sessions rather than just the open session.

```rust
use hdf5_pure::{File, FileBuilder, FileSpaceStrategy};

let mut builder = FileBuilder::new();
builder.create_dataset("keep").with_i32_data(&[1, 2, 3]);
builder.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1); // persist = true
builder.write("managed.h5").unwrap();

// Create then delete a dataset through the owned-handle editor; the freed
// region is recorded on disk.
let file = File::open_rw("managed.h5").unwrap();
file.root()
    .create_dataset("scratch", |b| {
        b.with_f64_data(&vec![0.0; 4096]);
    })
    .unwrap();
file.commit().unwrap();
file.root().delete("scratch").unwrap();
file.commit().unwrap();
file.close().unwrap();

// Later opens can read the persisted free regions.
let file = File::open("managed.h5").unwrap();
let free = file.persisted_free_space();
let total_free: u64 = free.iter().map(|&(_, len)| len).sum();
println!("persisted free regions: {}", free.len());
println!("total persisted free bytes: {total_free}");
```

!!! note
    `persisted_free_space()` is empty when the file does not persist free space, and for the streaming backend (which does not load the manager blocks). The addresses are file offsets relative to the base address; [reading data](reading.md) is unaffected by the presence or absence of these managers.

When free space is not persisted, a read-write session still reuses space within a single session but does not carry a free list across closes. If churn has left a file with unused gaps and you want to reclaim them outright, [repacking](repack.md) rewrites the file compactly.
