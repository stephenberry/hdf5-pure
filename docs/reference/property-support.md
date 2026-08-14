# File Property Support (`fapl` / `fcpl` / `dapl`)

HDF5 configures a file through property lists: a **file-creation property list** (`fcpl`, passed to `H5Fcreate`), a **file-access property list** (`fapl`, passed to `H5Fcreate`/`H5Fopen`), and a **dataset-access property list** (`dapl`, passed to `H5Dopen`). `hdf5-pure` does not expose property-list *handles*, but it does model all three as plain reusable values: `FileCreateProperties`, `FileAccessProperties`, and `DatasetAccessProperties`. Build one once and pass it to `FileBuilder::with_create_properties` / `File::create_with_options`, to any `*_with_options` open, or to `File::dataset_with_options`. The equivalent `FileBuilder` methods set the same creation properties one at a time.

## How these types are named

A `*Properties` type stands in for **one whole HDF5 property list**, so every setting on it is looked up on this page — including the few that have no C counterpart at all (`with_memory_strategy`, `with_sync_policy`), which the tables mark as such. Three types carry that suffix: `FileCreateProperties` (`fcpl`), `FileAccessProperties` (`fapl`), and `DatasetAccessProperties` (`dapl`). Where a setting belongs to a *different* HDF5 class than the table it appears in, that table labels it with its real class in parentheses.

"Stands in for" is not "covers": each of the three models a **subset** of its list's properties, and the tables below give the whole picture by listing what is unsupported as well as what works.

The suffix is a positive claim only. A `*Options` or `*Config` type makes **no** claim either way, and some of them do map to C properties: `ChunkCacheConfig` carries the `rdcc_*` values that both `H5Pset_cache` (an `fapl` property) and `H5Pset_chunk_cache` (a `dapl` property) take, so it belongs to no single list; `MetadataCacheConfig` is the analogue of the `H5AC_cache_config_t` *struct* that one `fapl` property accepts, not of a list; and `RepackOptions`, `VlenStringReadOptions`, and `mat::Options` have no HDF5 counterpart at all.

The `*_with_options` constructors keep their names. That suffix marks the explicit-configuration variant of an open, in the ordinary Rust sense of [`OpenOptions`](https://doc.rust-lang.org/std/fs/struct.OpenOptions.html); it does not name a type, and the type is visible in the signature.

This page is the consolidated map from each HDF5 property to what the crate supports. For the file-space details behind the first table, see [File-Space Strategy](../guide/file-space.md); for the read-write access modes, see [Editing in Place](../guide/editing.md) and [Streaming Large Files](../guide/streaming.md).

## Status legend

| Status | Meaning |
|---|---|
| **Genuine** | The property changes the on-disk result exactly as HDF5 specifies, verified against the reference C library. |
| **Recorded** | The value is written to the file and round-trips through the C library, but does not change the layout — correct for a fresh file, which has no free space to manage yet. |
| **Read-only** | Honored when reading a file; no write-side effect. |
| **Behavioral** | Changes how the crate operates on the file — memory, durability — without changing a byte of the result. There is nothing on disk to verify against the C library. |
| **Assertion** | Validated on write, but cannot change the emitted format. |
| **Unsupported** | No equivalent; the property is absent, or a file requiring it is refused up front. |
| **N/A** | Not meaningful for the on-disk format this crate emits. |

## File-creation properties (`fcpl`)

Set through [`FileBuilder`](../guide/writing.md) before `write` / `finish`, individually or all at once with `FileBuilder::with_create_properties(FileCreateProperties)`. `File::create_with_options(path, fcpl, fapl)` applies them on the owned-handle path, mirroring `H5Fcreate(name, flags, fcpl_id, fapl_id)`. Because it returns an open read-write handle, a creation/access pair the reopen would refuse — a paged file with `persist = false`, or a userblock under `MemoryStrategy::Bounded` — fails before anything is written.

| HDF5 property (C API) | `hdf5-pure` | Status | Behavior |
|---|---|---|---|
| `H5Pset_file_space_strategy(PAGE, …)` | `with_file_space_strategy(FileSpaceStrategy::Page, …)` | **Genuine** | Real page-aligned allocation: metadata and raw data occupy separate pages, and each page's free tail is tracked in a per-page-type `FSHD`/`FSSE` manager. The C library reads it as paged and `H5Fget_freespace` matches the tracked total. |
| `H5Pset_file_space_strategy(FSM_AGGR / AGGR / NONE, …)` | `…(FsmAggr / Aggr / None, …)` | **Recorded** | Strategy stored in the superblock extension; the layout stays sequential. Freed regions become tracked once a read-write session deletes an object. |
| `persist` flag | 2nd argument of `with_file_space_strategy` | **Genuine** (paged) / **Recorded** (non-paged) | Paged: per-page-type managers are written from creation. Non-paged: records intent; managers appear after a later delete. |
| `threshold` | 3rd argument of `with_file_space_strategy` | **Recorded (advisory)** | Round-trips through the C library, but the crate currently tracks every page tail / freed section regardless of it. |
| `H5Pset_file_space_page_size` | `with_file_space_page_size` | **Genuine** (paged) / **Recorded** (non-paged) | Under `Page` it is the alignment quantum (default 4096; must be a power of two `>= 512`). Under other strategies it is recorded but inert. |
| `H5Pset_userblock` | `with_userblock` | **Genuine** | Reserves a zero-filled prefix; all addresses are base-relative. The HDF5 "zero, or a power of two `>= 512`" rule is validated at write time (`FormatError::InvalidUserblockSize`), since the size *is* the superblock's base address and readers scan only the doubling sequence for the signature. Under `Page` the userblock must additionally be a whole number of pages. Contents come from `FileBuilder::with_userblock_content`, which every output path emits. |
| `H5Pset_libver_bounds` (fapl) | `with_libver_bounds` | **Assertion** | The writer always emits the v3 (HDF5 1.10) superblock; the bound is an accept/reject check, not a format selector. HDF5 classes this as a *file-access* property; it sits on `FileCreateProperties` because this crate checks the bound at write time. |
| `H5Pset_fill_value` / `H5Pset_fill_time` (dcpl) | `DatasetBuilder::with_fill_value` | **Genuine** (per dataset) | Encodes the fill value in a v3 Fill Value message; `Dataset::fill_value` reads it back, from this crate's files and the C library's. |
| `H5Pset_obj_track_times` (ocpl) | none | **Unsupported** | Objects are always written with times untracked (equivalent to `false`); there is no way to enable tracking. |
| `H5Pset_sym_k` / `H5Pset_istore_k` | none | **N/A** | The v3 superblock omits these fields; groups are always new-style (link messages + v2 object headers). |
| `H5Pset_link_phase_change` / `H5Pset_est_link_info` (gcpl) | none | **Unsupported** | Group Info is written minimal, so the C library's defaults (max-compact 8, min-dense 6) apply; the thresholds are not tunable. |

## File-access properties (`fapl`)

Selected through the `File` open-mode constructor, with memory budgets and locking set through `FileAccessProperties` / `DatasetAccessProperties` / `FileLocking`.

| HDF5 property / driver | `hdf5-pure` | Status | Behavior |
|---|---|---|---|
| `H5Fopen(RDONLY)`, default sec2 | `File::open` | **Genuine (read-only)** | Whole-file buffered read; takes no lock. |
| positioned / on-demand reads | `File::open_streaming` | **Genuine (read-only, bounded)** | Fetches metadata and chunks on demand; peak memory near one chunk. |
| `H5Pset_fapl_core` / `H5Pset_file_image` | `File::from_bytes` / `FileBuilder::finish` | **Genuine** | Read an in-memory file image, or build one into a `Vec<u8>`. |
| `H5Fopen(RDWR)` | `File::open_rw` / `open_rw_with_options` | **Genuine (read-write)** | Reads, appends, and staged edits + `commit`. Holds a latest-format file with no userblock in bounded memory and anything else in a whole-file mirror, picked from the file. Commits to a paged file through a page-aware tail; a paged file without persisted free space is refused at open. |
| `H5Fopen(RDWR)`, bounded memory | `File::open_rw_bounded` (deprecated) | **Genuine (read-write, bounded)** | Superseded by `File::open_rw`, which now picks the bounded backing itself. Kept as the strict `MemoryStrategy::Bounded` default: it refuses a file the bounded engine cannot edit instead of mirroring it. |
| no C counterpart | `FileAccessProperties::with_memory_strategy`, `MemoryStrategy` | **Behavioral** | How much memory a read-write open may spend holding the file, overriding the dispatch above. `Bounded` refuses a file the bounded engine cannot edit — a pre-v2 superblock or a userblock — rather than mirroring it; `Auto` (what `open_rw` uses unset) falls back to the mirror instead; `Mirrored` always mirrors. A paged file without persisted free space is refused under both preferences, because the mirror cannot commit it either. `File::edit_backing` reports which backend an open resolved to, as an `EditBacking` (`Bounded` or `Mirrored`) — a separate type because `Auto` is a preference between the two and never an outcome. The C library has no analogue: `H5Fopen` picks its own caching with no caller-visible memory contract. |
| `H5F_ACC_SWMR_READ` / `H5F_ACC_SWMR_WRITE` | `File::open_swmr` / `open_swmr_writer` (`*_with_options`) | **Genuine** | No OS lock; the writer raises the superblock SWMR-write flag (v3 superblock required, as in the C library) and appends only. The writer always mirrors, so it accepts `MemoryStrategy::Auto` and `Mirrored` and refuses an explicit `Bounded`. |
| superblock status flags (`h5clear -s`) | checked by `File::open`, `open_streaming`, `open_rw`, `open_swmr`, `open_swmr_writer`; cleared by `File::clear_swmr_flag` | **Genuine** | A file the byte marks as held by a writer is refused with `Error::FileMarkedInUse`, as `H5Fopen` refuses it; `File::open_swmr` follows it instead, refusing only a half-set mark. Version-3 superblocks only, which is where the C library checks. `File::from_bytes` does not consult the byte (and neither, therefore, does `mat::from_file`), so a caller holding the bytes can still read a flagged file. |
| `H5Pset_file_locking` + `HDF5_USE_FILE_LOCKING` | `FileAccessProperties::with_locking`, `FileLocking` | **Genuine** | Exclusive advisory lock on both read-write paths, mirror and bounded (non-blocking → `Error::FileLocked`); the env var override recognizes the same values as the C library. Readers and the SWMR writer take no lock by design and ignore the setting. |
| `H5Fget_libver_bounds` (read) | `File::libver_bound` | **Read-only** | Reports the low library-version bound the superblock version implies. |
| `H5Pset_cache` (rdcc) / `H5Pset_chunk_cache` (dapl) | `FileAccessProperties::with_chunk_cache`, `DatasetAccessProperties::with_chunk_cache` | **Genuine (all backends)** | A decompressed-chunk + parsed-index LRU (default 1 MiB / 16 slots). No write coalescing (a mutation clears it) and no `rdcc_w0` preemption policy. |
| `H5Pset_mdc_config` | `FileAccessProperties::with_metadata_cache` | **Read-only (partial)** | A byte budget for a metadata-read LRU on the streaming and bounded backends; default off. Only the memory-budget portion of `H5AC_cache_config_t`, none of the adaptive-resize/flush policy. |
| no C counterpart | `FileAccessProperties::with_sync_policy`, `SyncPolicy`, `File::sync` | **Behavioral** | Who issues the `fsync`s. The default `Always` forces durability at every point the write paths define one; `OnClose` issues none during the session — the cadence is the application's, through `File::sync` — and one at `close` or drop, which write past the point any caller could order them. Writes still reach the operating system as they are made either way, so what moves to the caller is power-loss durability within the session, not same-machine visibility — see [Choosing the fsync cadence](../guide/editing.md#choosing-the-fsync-cadence) for the cost in full. The C library has no property for this because it never `fsync`s at all: the default `sec2` driver installs no flush callback, so `H5Fflush` drains libhdf5's caches with `write` and stops. |
| `H5Pset_page_buffer_size` | none | **Unsupported** | There is no page buffer and no write buffering; paging is a layout-time concern only. |
| `H5Pset_fapl_family` / `split` / `multi` / `mpio` / `direct` / `ros3` / `log` | none | **Unsupported** | Only two implicit drivers exist: an in-memory buffer and `Read + Seek` positioned I/O. A multi-file, parallel, or remote-object file is not opened. |

## Dataset-access properties (`dapl`)

Passed to `File::dataset_with_options` / `Group::dataset_with_options` as a `DatasetAccessProperties`, overriding the file-wide access defaults for one dataset.

| HDF5 property (C API) | `hdf5-pure` | Status | Behavior |
|---|---|---|---|
| `H5Pset_chunk_cache` | `DatasetAccessProperties::with_chunk_cache` | **Genuine** | Overrides the file-wide chunk cache for this dataset only. Unset means inherit, matching the `H5D_CHUNK_CACHE_*_DEFAULT` sentinels. `rdcc_nslots` and `rdcc_nbytes` map directly; `rdcc_w0` is not modeled, as eviction is strict LRU. |
| `H5Pset_efile_prefix` / `H5Pset_virtual_prefix` | none | **Unsupported** | External and virtual datasets are not resolved, so there is no prefix to set. |
| `H5Pset_virtual_view` / `H5Pset_virtual_printf_gap` | none | **Unsupported** | Virtual datasets are not supported. |
| `H5Pset_append_flush` | none | **Unsupported** | No append callback or per-boundary flush; `Dataset::append` flushes on its own schedule. |

## Compliance and known limits

The paged and persistent-free-space paths are exercised by C-library crosschecks (`tests/file_space_crosscheck.rs`, `tests/bounded_append_crosscheck.rs`): the reference library recovers the strategy, `H5Fget_freespace` equals the crate's tracked total exactly, and the C library reopens a paged file read-write and re-paginates it. The crate also reads and bounded-mutates genuine C-created paged and persisted files. Compliance here means page **alignment** and structural validity, not byte-for-byte reproduction of the C allocator's intra-page packing.

Current limits worth knowing:

- **A paged file must persist its free space to be mutated.** Both editors grow a paged `persist = true` file, keeping pages homogeneous and the end of allocation page-aligned; a paged file created **without** `persist = true`, or one carrying a userblock, has no usable record of which pages hold metadata versus raw data and cannot be grown at all — recreate it with `persist = true` and no userblock.
- **Free space is under-reported, never over-reported.** A final metadata-page tail and the old bytes of a relocated partial chunk are left untracked, so `H5Fget_freespace` can read slightly low. The file stays valid.
- **`threshold` is advisory** (see the tables above).
- Only **File Space Info message version 1** is emitted and read.

See [Limitations](limitations.md) for the full catalog of deliberate refusals.
