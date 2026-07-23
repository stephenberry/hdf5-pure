# File Property Support (`fapl` / `fcpl`)

HDF5 configures a file through two property lists: a **file-creation property list** (`fcpl`, passed to `H5Fcreate`) and a **file-access property list** (`fapl`, passed to `H5Fcreate`/`H5Fopen`). `hdf5-pure` does not expose property-list handles. Instead, creation properties are methods on [`FileBuilder`](../guide/writing.md) and access properties are open-mode constructors on `File` plus the `FileAccessOptions` / `DatasetAccessOptions` structs and the `FileLocking` enum.

This page is the consolidated map from each HDF5 property to what the crate supports. For the file-space details behind the first table, see [File-Space Strategy](../guide/file-space.md); for the read-write access modes, see [Editing in Place](../guide/editing.md) and [Streaming Large Files](../guide/streaming.md).

## Status legend

| Status | Meaning |
|---|---|
| **Genuine** | The property changes the on-disk result exactly as HDF5 specifies, verified against the reference C library. |
| **Recorded** | The value is written to the file and round-trips through the C library, but does not change the layout — correct for a fresh file, which has no free space to manage yet. |
| **Read-only** | Honored when reading a file; no write-side effect. |
| **Assertion** | Validated on write, but cannot change the emitted format. |
| **Unsupported** | No equivalent; the property is absent, or a file requiring it is refused up front. |
| **N/A** | Not meaningful for the on-disk format this crate emits. |

## File-creation properties (`fcpl`)

Set through [`FileBuilder`](../guide/writing.md) before `write` / `finish`.

| HDF5 property (C API) | `hdf5-pure` | Status | Behavior |
|---|---|---|---|
| `H5Pset_file_space_strategy(PAGE, …)` | `with_file_space_strategy(FileSpaceStrategy::Page, …)` | **Genuine** | Real page-aligned allocation: metadata and raw data occupy separate pages, and each page's free tail is tracked in a per-page-type `FSHD`/`FSSE` manager. The C library reads it as paged and `H5Fget_freespace` matches the tracked total. |
| `H5Pset_file_space_strategy(FSM_AGGR / AGGR / NONE, …)` | `…(FsmAggr / Aggr / None, …)` | **Recorded** | Strategy stored in the superblock extension; the layout stays sequential. Freed regions become tracked once a read-write session deletes an object. |
| `persist` flag | 2nd argument of `with_file_space_strategy` | **Genuine** (paged) / **Recorded** (non-paged) | Paged: per-page-type managers are written from creation. Non-paged: records intent; managers appear after a later delete. |
| `threshold` | 3rd argument of `with_file_space_strategy` | **Recorded (advisory)** | Round-trips through the C library, but the crate currently tracks every page tail / freed section regardless of it. |
| `H5Pset_file_space_page_size` | `with_file_space_page_size` | **Genuine** (paged) / **Recorded** (non-paged) | Under `Page` it is the alignment quantum (default 4096; must be a power of two `>= 512`). Under other strategies it is recorded but inert. |
| `H5Pset_userblock` | `with_userblock` | **Genuine** | Reserves a zero-filled prefix; all addresses are base-relative. The HDF5 "power of two `>= 512`" rule is **not** validated on the non-paged path (a bad size is accepted); under `Page` the userblock must be a whole number of pages or the write is refused. |
| `H5Pset_libver_bounds` | `with_libver_bounds` | **Assertion** | The writer always emits the v3 (HDF5 1.10) superblock; the bound is an accept/reject check, not a format selector. |
| `H5Pset_fill_value` / `H5Pset_fill_time` (dcpl) | `DatasetBuilder::with_fill_value` | **Genuine** (per dataset) | Encodes the fill value in a v3 Fill Value message; `Dataset::fill_value` reads it back, from this crate's files and the C library's. |
| `H5Pset_obj_track_times` (ocpl) | none | **Unsupported** | Objects are always written with times untracked (equivalent to `false`); there is no way to enable tracking. |
| `H5Pset_sym_k` / `H5Pset_istore_k` | none | **N/A** | The v3 superblock omits these fields; groups are always new-style (link messages + v2 object headers). |
| `H5Pset_link_phase_change` / `H5Pset_est_link_info` (gcpl) | none | **Unsupported** | Group Info is written minimal, so the C library's defaults (max-compact 8, min-dense 6) apply; the thresholds are not tunable. |

## File-access properties (`fapl`)

Selected through the `File` open-mode constructor, with memory budgets and locking set through `FileAccessOptions` / `DatasetAccessOptions` / `FileLocking`.

| HDF5 property / driver | `hdf5-pure` | Status | Behavior |
|---|---|---|---|
| `H5Fopen(RDONLY)`, default sec2 | `File::open` | **Genuine (read-only)** | Whole-file buffered read; takes no lock. |
| positioned / on-demand reads | `File::open_streaming` | **Genuine (read-only, bounded)** | Fetches metadata and chunks on demand; peak memory near one chunk. |
| `H5Pset_fapl_core` / `H5Pset_file_image` | `File::from_bytes` / `FileBuilder::finish` | **Genuine** | Read an in-memory file image, or build one into a `Vec<u8>`. |
| `H5Fopen(RDWR)` | `File::open_rw` | **Genuine (read-write, O(file))** | Whole-file in-memory mirror; reads, appends, and staged edits + `commit`. On a paged file, reads work but commits and appends refuse (grow it via `open_rw_bounded`). |
| `H5Fopen(RDWR)`, bounded memory | `File::open_rw_bounded` | **Genuine (read-write, bounded)** | Bounded reads + immediate crash-atomic `Dataset::append`. The path for growing a file that persists its free space, **including a paged file**. The staged edit surface returns `Error::BoundedStagedUnsupported`. |
| `H5F_ACC_SWMR_READ` / `H5F_ACC_SWMR_WRITE` | `File::open_swmr` / `open_swmr_writer` | **Genuine** | No OS lock; the writer raises the superblock SWMR-write flag and appends only. |
| `H5Pset_file_locking` + `HDF5_USE_FILE_LOCKING` | `FileLocking`, `File::open_rw_with_locking` | **Genuine** | Exclusive advisory lock on the edit path (non-blocking → `Error::FileLocked`); the env var override recognizes the same values as the C library. Readers and the SWMR writer take no lock by design. |
| `H5Fget_libver_bounds` (read) | `File::libver_bound` | **Read-only** | Reports the low library-version bound the superblock version implies. |
| `H5Pset_cache` (rdcc) / `H5Pset_chunk_cache` | `FileAccessOptions::with_chunk_cache`, `DatasetAccessOptions::with_chunk_cache` | **Read-only** | A decompressed-chunk + parsed-index LRU (default 1 MiB / 16 slots). No write coalescing (a mutation clears it) and no `rdcc_w0` preemption policy. |
| `H5Pset_mdc_config` | `FileAccessOptions::with_metadata_cache` | **Read-only (partial)** | A byte budget for a metadata-read LRU on the streaming and bounded backends; default off. Only the memory-budget portion of `H5AC_cache_config_t`, none of the adaptive-resize/flush policy. |
| `H5Pset_page_buffer_size` | none | **Unsupported** | There is no page buffer and no write buffering; paging is a layout-time concern only. |
| `H5Pset_fapl_family` / `split` / `multi` / `mpio` / `direct` / `ros3` / `log` | none | **Unsupported** | Only two implicit drivers exist: an in-memory buffer and `Read + Seek` positioned I/O. A multi-file, parallel, or remote-object file is not opened. |

## Compliance and known limits

The paged and persistent-free-space paths are exercised by C-library crosschecks (`tests/file_space_crosscheck.rs`, `tests/bounded_append_crosscheck.rs`): the reference library recovers the strategy, `H5Fget_freespace` equals the crate's tracked total exactly, and the C library reopens a paged file read-write and re-paginates it. The crate also reads and bounded-mutates genuine C-created paged and persisted files. Compliance here means page **alignment** and structural validity, not byte-for-byte reproduction of the C allocator's intra-page packing.

Current limits worth knowing:

- **Paged mutation goes through `File::open_rw_bounded` only.** The whole-file editor (`File::open_rw` / the deprecated `EditSession`) opens and reads a paged file but refuses to commit edits or append to it, and a paged file created **without** `persist = true` cannot be grown at all — recreate it with `persist = true`.
- **Free space is under-reported, never over-reported.** A final metadata-page tail and the old bytes of a relocated partial chunk are left untracked, so `H5Fget_freespace` can read slightly low. The file stays valid.
- **`threshold` is advisory** and **the non-paged `userblock` size is not validated** (see the tables above).
- Only **File Space Info message version 1** is emitted and read.

See [Limitations](limitations.md) for the full catalog of deliberate refusals.
