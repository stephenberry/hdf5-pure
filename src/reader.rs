//! Reading API: File, Dataset, and Group handles for reading HDF5 files.

use std::borrow::Cow;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::edit::{
    AppendBuilder, AppendGeometry, AppendTarget, EditBacking, MemoryStrategy, SpaceAccounting,
    SyncPolicy, WriteEngine,
};
use crate::element::H5Element;
use crate::type_builders::{DatasetBuilder, VL_REF_SIZE};

use crate::appender::BufferedAppender;
use crate::attribute::{extract_attributes_full, extract_attributes_full_from_source};
use crate::chunk_cache::{CachePass, ChunkCache, ChunkCacheConfig, ChunkCacheStats};
use crate::compound::CompoundType;
use crate::convert::TryToUsize;
use crate::data_layout::DataLayout;
use crate::data_read;
use crate::dataspace::Dataspace;
use crate::datatype::{Datatype, ReferenceType};
use crate::error::{Error, FormatError};
use crate::file_create_properties::FileCreateProperties;
use crate::file_lock::{self, FileLocking, OpenIntent};
use crate::file_space_info::{FileSpaceInfo, FileSpaceStrategy};
use crate::fill_value::FillPattern;
use crate::filter_pipeline::FilterPipeline;
use crate::free_space_manager;
use crate::group_v1::GroupEntry;
use crate::group_v2;
use crate::layout_info::{Chunk, ChunkIndex, Filter, Layout};
use crate::libver::LibVer;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::read_spec::RawReadSpec;
use crate::shared_message::{self, BufferedResolver, SharedResolver, SourceResolver};
use crate::signature;
use crate::source::{
    BaseOffsetSource, BytesSource, MetadataCacheConfig, MetadataCachingSource, ReadSeekSource,
    Source, frame,
};
use crate::superblock::Superblock;
use crate::vl_data::{self, VlenStringReadOptions};

use crate::types::{AttrValue, DType, attrs_to_map, classify_datatype};

// ---------------------------------------------------------------------------
// File
// ---------------------------------------------------------------------------

/// Backing store for a [`File`]: either the whole file buffered in memory, or a
/// lazy [`Source`] that reads regions on demand (see [`File::open_streaming`]).
enum Backend {
    InMemory(Vec<u8>),
    Streaming(Box<dyn Source + Send + Sync>),
    /// A read-write file opened with [`File::open_rw`]: a [`WriteEngine`] (exclusive OS lock + staged
    /// edit queues + append geometry cache) behind a lock, so owned handles can
    /// both read and mutate in place. Handle write methods route to the engine,
    /// and `File::commit` applies staged structural edits.
    ///
    /// Either backing — a whole-file mirror, or positioned I/O against the
    /// handle — appears here as the same `WriteEngine`; which one an open
    /// resolved to is the engine's own business rather than the backend's
    /// (issue #198). Reads
    /// borrow the mirror's slice when there is one and go through the image's
    /// `Source` otherwise; see [`with_engine`](FileInner::with_engine). Boxed to
    /// keep the `Backend` enum small (a `WriteEngine` is far larger than the
    /// other variants).
    Edit(Box<Mutex<WriteEngine>>),
}

/// A borrowed `Source` view over a [`File`]'s backend, used by the
/// streaming-capable read paths so one call site serves both backends.
pub(crate) enum SourceView<'a> {
    Mem(&'a [u8]),
    Stream(&'a (dyn Source + Send + Sync)),
}

impl Source for SourceView<'_> {
    fn len(&self) -> u64 {
        match self {
            SourceView::Mem(b) => b.len() as u64,
            SourceView::Stream(s) => s.len(),
        }
    }
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        match self {
            SourceView::Mem(b) => BytesSource::new(*b).read_at(offset, buf),
            SourceView::Stream(s) => s.read_at(offset, buf),
        }
    }

    fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        match self {
            SourceView::Mem(b) => BytesSource::new(*b).read_metadata_at(offset, len),
            SourceView::Stream(s) => s.read_metadata_at(offset, len),
        }
    }
}

/// File-access properties applied when opening an HDF5 file.
///
/// This is the `hdf5-pure` analogue of an HDF5 **file access property list**
/// (`fapl`): one value carrying every access-time setting, built once and passed
/// to whichever open a caller reaches for, exactly as a `fapl` is handed to
/// `H5Fopen`. Every `*_with_options` constructor on [`File`] accepts it, so a
/// read path and a read-write path can share one configuration.
///
/// The `Properties` suffix means the type stands in for one whole HDF5 property
/// list, so every setting on it has a C counterpart to look up. It is a stand-in
/// and not a port: a plain `Copy` value, with no handle to create or close, no
/// runtime property registry, and no setter that can fail. `fapl` and each
/// `H5Pset_*` it models are doc aliases, so a search for either lands here.
///
/// - The metadata cache (`H5Pset_mdc_config`) applies to the streaming and
///   bounded backends; an in-memory open already holds the whole file in one
///   buffer.
/// - The chunk cache (`H5Pset_cache`) is the file-wide default for datasets
///   opened from any backend, overridable per dataset with
///   [`DatasetAccessProperties`].
/// - The locking policy (`H5Pset_file_locking`) applies to the read-write opens.
///   Readers and the SWMR writer take no lock by design, so they ignore it.
///
/// See the [property-support reference] for the full property-by-property map.
///
/// [property-support reference]: https://github.com/stephenberry/hdf5-pure/blob/main/docs/reference/property-support.md
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[doc(alias = "fapl")]
pub struct FileAccessProperties {
    metadata_cache: MetadataCacheConfig,
    chunk_cache: ChunkCacheConfig,
    locking: FileLocking,
    memory_strategy: Option<MemoryStrategy>,
    libver_bounds: Option<(LibVer, LibVer)>,
    sync_policy: SyncPolicy,
}

impl FileAccessProperties {
    /// A value carrying the crate's default access behavior.
    pub const fn new() -> Self {
        Self {
            metadata_cache: MetadataCacheConfig::disabled(),
            chunk_cache: ChunkCacheConfig::new(),
            locking: FileLocking::Enabled,
            memory_strategy: None,
            libver_bounds: None,
            sync_policy: SyncPolicy::Always,
        }
    }

    /// Configure the bounded streaming metadata cache.
    #[doc(alias = "H5Pset_mdc_config")]
    pub const fn with_metadata_cache(mut self, metadata_cache: MetadataCacheConfig) -> Self {
        self.metadata_cache = metadata_cache;
        self
    }

    /// Configure the per-dataset raw chunk cache used by datasets opened from
    /// this file. This is the `H5Pset_cache`-style file-wide default.
    #[doc(alias = "H5Pset_cache")]
    pub const fn with_chunk_cache(mut self, chunk_cache: ChunkCacheConfig) -> Self {
        self.chunk_cache = chunk_cache;
        self
    }

    /// Set the OS advisory file-locking policy for the read-write opens.
    ///
    /// Defaults to [`FileLocking::Enabled`]. Use [`FileLocking::Disabled`] only
    /// when an external mechanism already guarantees single-writer access, or
    /// [`FileLocking::BestEffort`] on a filesystem (such as some network mounts)
    /// where the OS lock is unavailable. Setting `HDF5_USE_FILE_LOCKING` in the
    /// environment overrides this, as in the C library.
    ///
    /// Readers and [`File::open_swmr_writer`] take no lock by design and ignore
    /// this.
    #[doc(alias = "H5Pset_file_locking")]
    pub const fn with_locking(mut self, locking: FileLocking) -> Self {
        self.locking = locking;
        self
    }

    /// Set how much memory a read-write open may use to hold the file.
    ///
    /// Unset by default, which lets the entry point choose:
    /// [`File::open_rw`] uses [`MemoryStrategy::Auto`], preferring the bounded
    /// engine and falling back to the whole-file mirror for a file it cannot
    /// edit. Setting this overrides that default, in either direction, so
    /// [`MemoryStrategy::Bounded`] refuses such a file rather than quietly
    /// spending `O(file size)` memory on a caller who asked not to;
    /// [`MemoryStrategy::Mirrored`] takes
    /// the whole-file mirror unconditionally, as `open_rw` did before it learned
    /// to dispatch.
    ///
    /// The read-only opens ignore this: they build no editing session at all, and
    /// their own names say what memory they spend. [`File::open_swmr_writer`]
    /// does build one, and always mirrors: it accepts
    /// [`MemoryStrategy::Auto`] and [`MemoryStrategy::Mirrored`], both of which
    /// the mirror satisfies, and refuses an explicit [`MemoryStrategy::Bounded`]
    /// with [`Error::EditUnsupported`] rather than quietly not honoring it. Ask a
    /// `File` which backend it resolved to with [`File::edit_backing`].
    pub const fn with_memory_strategy(mut self, memory_strategy: MemoryStrategy) -> Self {
        self.memory_strategy = Some(memory_strategy);
        self
    }

    /// Constrain the on-disk format an editing session may write, mirroring
    /// HDF5's `H5Pset_libver_bounds` — which the C library classes as a *file
    /// access* property for exactly this reason: it governs what a later write
    /// to an existing file is allowed to add.
    ///
    /// Unset by default, which keeps [`File::open_rw`] adding whatever the
    /// content needs. That default is what lets a file the C library wrote under
    /// its own bounds be edited at all, but it means a session can add content
    /// only a newer library can read *without changing the superblock*, and the
    /// caller has no way to see it happen: adding a chunked, filtered, or
    /// resizable dataset to an HDF5 1.8 file needs the version 4 data-layout
    /// message and a 1.10 chunk index, since this crate does not write the
    /// version 1 B-tree index that 1.8 used.
    ///
    /// Setting a `high` below [`LibVer::V110`] refuses that addition with
    /// [`FormatError::LibverTooOldForContent`](crate::FormatError::LibverTooOldForContent)
    /// at [`File::commit`] instead — the same refusal
    /// [`FileBuilder::with_libver_bounds`](crate::FileBuilder::with_libver_bounds)
    /// gives when writing a whole file, so a `.mat` bounded to 1.8 for MATLAB
    /// stays loadable by MATLAB after an edit.
    ///
    /// The read-only opens ignore this: they write nothing. [`File::open_swmr_writer`]
    /// requires a version 3 superblock, so it refuses a `high` below
    /// [`LibVer::V110`] up front rather than accepting a bound it cannot honor.
    #[doc(alias = "H5Pset_libver_bounds")]
    pub const fn with_libver_bounds(mut self, low: LibVer, high: LibVer) -> Self {
        self.libver_bounds = Some((low, high));
        self
    }

    /// Choose who owns this session's `fsync` cadence — this crate, or the
    /// application through [`File::sync`].
    ///
    /// Defaults to [`SyncPolicy::Always`]: every commit and every immediate
    /// [`Dataset::append`](crate::Dataset::append) forces its writes to durable
    /// storage before returning. [`SyncPolicy::OnClose`] issues no `fsync` at all,
    /// which is what the reference C library does; the writes still reach the
    /// operating system as they are made, so only power-loss durability moves to
    /// the caller.
    ///
    /// The read-only opens ignore this: they write nothing.
    pub const fn with_sync_policy(mut self, sync_policy: SyncPolicy) -> Self {
        self.sync_policy = sync_policy;
        self
    }

    /// Return the configured streaming metadata cache.
    pub const fn metadata_cache(&self) -> MetadataCacheConfig {
        self.metadata_cache
    }

    /// Return the configured library-version bounds, or `None` when an editing
    /// session may write whatever its content needs.
    pub const fn libver_bounds(&self) -> Option<(LibVer, LibVer)> {
        self.libver_bounds
    }

    /// Return the configured per-dataset chunk cache.
    pub const fn chunk_cache(&self) -> ChunkCacheConfig {
        self.chunk_cache
    }

    /// Return the configured file-locking policy.
    pub const fn locking(&self) -> FileLocking {
        self.locking
    }

    /// Return the configured memory strategy, or `None` when none was asked for
    /// and the entry point's own default applies. This is what was *requested*;
    /// for which backend an open resolved to, see [`File::edit_backing`].
    ///
    /// The `Option` distinguishes "no preference stated" from an explicit
    /// [`MemoryStrategy::Auto`], which is what lets an entry point supply its own
    /// default without overriding a caller who asked for one; `None` resolves to
    /// [`MemoryStrategy::Auto`], the only default any entry point now supplies
    /// rather than as a second break on this accessor.
    pub const fn memory_strategy(&self) -> Option<MemoryStrategy> {
        self.memory_strategy
    }

    /// Return the configured `fsync` policy.
    pub const fn sync_policy(&self) -> SyncPolicy {
        self.sync_policy
    }
}

/// Dataset-access properties applied when opening a single dataset.
///
/// This is the `hdf5-pure` analogue of an HDF5 **dataset access property list**
/// (`dapl`). Its chunk cache corresponds to `H5Pset_chunk_cache`: it overrides,
/// for this one dataset, the file-wide chunk-cache default configured with
/// [`FileAccessProperties::with_chunk_cache`] (the `H5Pset_cache` analogue). When
/// left unset, the dataset inherits that file-wide default — matching the `dapl`
/// default sentinels (`H5D_CHUNK_CACHE_*_DEFAULT`), which also mean "use the
/// file's setting".
///
/// The `Properties` suffix means the type stands in for one whole HDF5 property
/// list, so every setting on it has a C counterpart to look up. It is a stand-in
/// and not a port: a plain `Copy` value, with no handle to create or close, no
/// runtime property registry, and no setter that can fail. `dapl` and each
/// `H5Pset_*` it models are doc aliases, so a search for either lands here.
/// The chunk cache is the one `dapl` property modeled; see the
/// [property-support reference] for the rest.
///
/// [`ChunkCacheConfig`] maps `H5Pset_chunk_cache`'s `rdcc_nslots` and
/// `rdcc_nbytes`; its `rdcc_w0` preemption policy is not modeled, because this
/// read cache uses strict LRU eviction (as noted on
/// [`ChunkCacheConfig::from_h5p_cache`]).
///
/// Pass it to [`File::dataset_with_options`] or [`Group::dataset_with_options`].
///
/// [property-support reference]: https://github.com/stephenberry/hdf5-pure/blob/main/docs/reference/property-support.md
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[doc(alias = "dapl")]
pub struct DatasetAccessProperties {
    chunk_cache: Option<ChunkCacheConfig>,
}

impl DatasetAccessProperties {
    /// A value that inherits every file-wide access default.
    pub const fn new() -> Self {
        Self { chunk_cache: None }
    }

    /// Override the raw chunk cache for this one dataset, ignoring the file-wide
    /// default. This is the `H5Pset_chunk_cache` analogue.
    #[doc(alias = "H5Pset_chunk_cache")]
    pub const fn with_chunk_cache(mut self, chunk_cache: ChunkCacheConfig) -> Self {
        self.chunk_cache = Some(chunk_cache);
        self
    }

    /// Return the chunk-cache override, or `None` when the dataset inherits the
    /// file-wide default.
    pub const fn chunk_cache(&self) -> Option<ChunkCacheConfig> {
        self.chunk_cache
    }

    /// Resolve the effective chunk-cache config: the per-dataset override if one
    /// was set, otherwise the file-wide `default`.
    const fn resolved_chunk_cache(&self, default: ChunkCacheConfig) -> ChunkCacheConfig {
        match self.chunk_cache {
            Some(config) => config,
            None => default,
        }
    }
}

/// Test whether a file looks like an HDF5 file, without reading it whole.
///
/// This is the spelling of the C library's `H5Fis_accessible` /
/// `H5Fis_hdf5`: it opens the file and scans only the 8-byte candidate windows
/// where the HDF5 signature is permitted (offsets 0, 512, 1024, 2048, …), so it
/// never buffers the whole file. Returns:
///
/// - `Ok(true)` — the HDF5 signature was found,
/// - `Ok(false)` — the file opened but has no HDF5 signature,
/// - `Err(..)` — the file could not be opened (missing, permissions, …).
///
/// It validates only the signature, not the rest of the format; a truncated or
/// corrupt file past the signature still reports `true`. Use [`File::open`] to
/// fully parse and validate.
pub fn is_hdf5<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<bool> {
    let handle = std::fs::File::open(path)?;
    let source = ReadSeekSource::new(handle).map_err(std::io::Error::other)?;
    match signature::find_signature_in(&source) {
        Ok(_) => Ok(true),
        Err(FormatError::SignatureNotFound) => Ok(false),
        Err(e) => Err(std::io::Error::other(e)),
    }
}

/// Test whether an in-memory buffer begins (at a permitted offset) with the
/// HDF5 signature. The buffer-backed counterpart of [`is_hdf5`].
pub fn is_hdf5_bytes(data: &[u8]) -> bool {
    signature::find_signature(data).is_ok()
}

/// An open HDF5 file for reading.
struct FileInner {
    backend: Backend,
    superblock: Superblock,
    /// Byte offset to add to all relative addresses (= original base_address).
    addr_offset: u64,
    /// Live file handle, retained only when the file was opened with
    /// [`File::open_swmr`] so [`File::refresh`] can re-read appended data.
    handle: Option<std::fs::File>,
    /// File Space Info parsed from the superblock extension, if the file records
    /// one. Best-effort: a malformed or unreadable extension leaves this `None`
    /// rather than failing the open.
    file_space_info: Option<FileSpaceInfo>,
    access_properties: FileAccessProperties,
    /// Set by [`File::close`] to seal a read-write file: after it, a write
    /// through any surviving [`Dataset`]/[`Group`] handle or [`File`] clone
    /// returns [`Error::FileClosed`]. Reads still work. Only ever set on a
    /// `Backend::Edit` file.
    closed: AtomicBool,
    /// True for a file opened with [`File::open_swmr_writer`]: no OS lock is held,
    /// the superblock's SWMR-write flag is raised, only immediate
    /// [`Dataset::append`] is permitted (the staged surface is refused), and the
    /// flag is cleared on [`File::close`] / `Drop`. `false` for every other file.
    swmr_write: bool,
}

impl Drop for FileInner {
    /// Best-effort cleanup for a writer dropped without an explicit
    /// [`File::close`], running only when the last `Arc<FileInner>` clone drops;
    /// a clean `close` already did this work and set `closed`, so this is
    /// idempotent and skipped in that case.
    ///
    /// - A SWMR writer clears the superblock's SWMR-write flag (mirroring
    ///   `File::close`).
    /// - A read-write file that persists its free space rewrites its on-disk
    ///   free-space managers into canonical shape (issue #173), so a
    ///   dropped-without-`close` handle leaves the same file a clean `close`
    ///   would (a no-op unless an immediate append grew the file past them). A
    ///   true crash (`SIGKILL`, power loss) skips `drop` entirely; the appended
    ///   data is still durable, under the default
    ///   [`SyncPolicy::Always`](crate::SyncPolicy).
    ///
    /// Staged edits are *not* committed here: dropping a handle discards them,
    /// which is what `close` exists to distinguish.
    fn drop(&mut self) {
        if self.closed.load(Ordering::Acquire) {
            return;
        }
        let Backend::Edit(m) = &self.backend else {
            return;
        };
        let mut session = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        // Both branches write, and this is the last moment anything can order
        // those writes: the handle is gone once this returns, so `File::sync` is
        // not an option the caller still has. The barrier is therefore forced
        // rather than left to the session's `SyncPolicy` — see
        // [`SyncPolicy::OnClose`](crate::SyncPolicy::OnClose).
        if self.swmr_write {
            let _ = session.set_consistency_flags(0);
            let _ = session.force_sync();
            return;
        }
        let _ = session.finalize_persist();
        let _ = session.force_sync();
    }
}

impl FileInner {
    /// Open an HDF5 file from a filesystem path.
    ///
    /// Reads the file into memory once. To follow a file that a concurrent
    /// single writer is appending to (SWMR), use [`File::open_swmr`] instead.
    /// To read a file larger than memory (e.g. on a 32-bit host) without
    /// buffering it, use [`File::open_streaming`].
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        Self::open_with_options(path, FileAccessProperties::new())
    }

    /// Open an HDF5 file from a filesystem path with explicit access properties.
    ///
    /// Like [`open`](Self::open), this buffers the whole file in memory. Use
    /// [`open_streaming_with_options`](Self::open_streaming_with_options) when
    /// the metadata cache budget should apply to lazy metadata reads.
    pub fn open_with_options<P: AsRef<std::path::Path>>(
        path: P,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        let bytes = std::fs::read(path.as_ref()).map_err(Error::Io)?;
        let inner = Self::from_bytes_with_options(bytes, properties)?;
        // The status-flag check belongs to the *path* opens, not to
        // `from_bytes_with_options` (issue #245). A caller who already holds the
        // bytes has taken its own snapshot: there is no live file to coordinate
        // over, and the recovery this refusal would name — `clear_swmr_flag`,
        // which needs write access to a path — is not available to it either.
        // This is a deliberate divergence from the C library, which checks under
        // its in-memory core driver too.
        file_lock::check_status_flags(&inner.superblock, OpenIntent::Read, path.as_ref())?;
        Ok(inner)
    }

    /// Open an HDF5 file for **streaming** reads, fetching regions on demand from
    /// the file instead of buffering it whole.
    ///
    /// This lets a host read a file larger than its address space — the original
    /// motivation being 32-bit targets reading multi-gigabyte files (issue #27).
    /// Metadata and dataset chunks are read through a `ReadSeekSource`, so peak
    /// memory stays close to one chunk plus the metadata being parsed. Chunks
    /// that sit next to each other on disk are fetched together, in reads of at
    /// most 256 KiB (a larger chunk is read on its own), which is what makes a
    /// file written a row at a time — thousands of chunks of a few dozen bytes
    /// — read at a sensible speed. See [`crate::chunk_span`].
    ///
    /// Reads match the buffered [`File::open`]: every storage layout and chunk
    /// index type, both group forms (v2 and v1 symbol-table), and compact,
    /// dense, shared, and variable-length attributes. What differs:
    /// [`as_bytes`](Self::as_bytes) returns an empty slice (there is no
    /// whole-file buffer), [`persisted_free_space`](Self::persisted_free_space)
    /// returns no regions, a streaming file cannot be the *source* of a
    /// cross-file copy, and chunk decompression is sequential (the `parallel`
    /// feature accelerates only buffered reads).
    pub fn open_streaming<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        Self::open_streaming_with_options(path, FileAccessProperties::new())
    }

    /// Open an HDF5 file for streaming reads with explicit access properties.
    pub fn open_streaming_with_options<P: AsRef<std::path::Path>>(
        path: P,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        let handle = std::fs::File::open(path.as_ref()).map_err(Error::Io)?;
        let source = ReadSeekSource::new(handle).map_err(Error::Format)?;
        let source: Box<dyn Source + Send + Sync> = if properties.metadata_cache.is_enabled() {
            Box::new(MetadataCachingSource::new(
                source,
                properties.metadata_cache,
            ))
        } else {
            Box::new(source)
        };
        let (superblock, addr_offset) = Self::parse_superblock_source(source.as_ref())?;
        file_lock::check_status_flags(&superblock, OpenIntent::Read, path.as_ref())?;
        Ok(Self::from_parts(
            Backend::Streaming(source),
            superblock,
            addr_offset,
            None,
            properties,
        ))
    }

    /// Open an HDF5 file for SWMR (single-writer/multiple-reader) reading.
    ///
    /// Like [`File::open`], but retains a live handle to the file so that
    /// [`File::refresh`] can re-read data appended by a concurrent writer
    /// (whether produced by this crate's append writer, the reference HDF5 C
    /// library, or h5py in SWMR mode). The initial view is a consistent
    /// snapshot; call [`File::refresh`] to advance to a newer one.
    ///
    /// Only the `std` build supports this (it requires a live filesystem
    /// handle); the in-memory [`File::from_bytes`] path cannot refresh.
    pub fn open_swmr<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        Self::open_swmr_with_options(path, FileAccessProperties::new())
    }

    /// Open an HDF5 file for SWMR reading with explicit access properties.
    ///
    /// SWMR reads currently keep an in-memory mirror for refresh semantics, so
    /// only the per-dataset chunk-cache settings affect this backend.
    pub fn open_swmr_with_options<P: AsRef<std::path::Path>>(
        path: P,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        let mut handle = std::fs::File::open(path.as_ref()).map_err(Error::Io)?;
        let mut data = Vec::new();
        handle.read_to_end(&mut data).map_err(Error::Io)?;
        let (superblock, addr_offset) = Self::parse_superblock(&data)?;
        file_lock::check_status_flags(&superblock, OpenIntent::SwmrRead, path.as_ref())?;
        Ok(Self::from_parts(
            Backend::InMemory(data),
            superblock,
            addr_offset,
            Some(handle),
            properties,
        ))
    }

    /// Open an HDF5 file from an in-memory byte vector.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self, Error> {
        Self::from_bytes_with_options(data, FileAccessProperties::new())
    }

    /// Open an HDF5 file from an in-memory byte vector with explicit access properties.
    pub fn from_bytes_with_options(
        data: Vec<u8>,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        let (superblock, addr_offset) = Self::parse_superblock(&data)?;
        Ok(Self::from_parts(
            Backend::InMemory(data),
            superblock,
            addr_offset,
            None,
            properties,
        ))
    }

    /// Open an existing HDF5 file for reading **and** in-place editing, applying
    /// `properties` (its [`FileLocking`] policy governs the OS file lock held for
    /// the file's life, and its chunk cache is the file-wide default).
    fn open_rw<P: AsRef<std::path::Path>>(
        path: P,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        Self::open_rw_with_default(path, properties, MemoryStrategy::Auto)
    }

    /// Open read-write under the properties' memory strategy, falling back to
    /// `default` when the caller expressed none. The two public read-write entry
    /// [`File::open_rw`] passes [`MemoryStrategy::Auto`]: prefer the bounded
    /// engine, but take the mirror for a file the bounded engine cannot edit
    /// (issue #198, step 4). A caller who states a strategy overrides it.
    fn open_rw_with_default<P: AsRef<std::path::Path>>(
        path: P,
        properties: FileAccessProperties,
        default: MemoryStrategy,
    ) -> Result<Self, Error> {
        let session = WriteEngine::open_rw_with_strategy(
            path.as_ref(),
            properties.metadata_cache,
            properties.locking,
            properties.memory_strategy.unwrap_or(default),
        )?;
        Self::from_rw_session(session, properties)
    }

    /// Wrap an opened [`WriteEngine`] as a read-write [`Backend::Edit`] file.
    fn from_rw_session(
        mut session: WriteEngine,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        // The one funnel every read-write session passes through, so the fapl's
        // format bound and `fsync` cadence reach the engine no matter which entry
        // point opened it — the SWMR writer included, which
        // `WriteEngine::open_swmr_writer` says why.
        session.set_libver_bounds(properties.libver_bounds)?;
        session.set_sync_policy(properties.sync_policy);
        // The engine parsed and normalized this at open; take it rather than
        // re-parsing, so the image need not be able to hand out a slice.
        let superblock = session.superblock().clone();
        let addr_offset = superblock.base_address;
        Ok(Self::from_parts(
            Backend::Edit(Box::new(Mutex::new(session))),
            superblock,
            addr_offset,
            None,
            properties,
        ))
    }

    /// Open for SWMR writing: no OS lock, superblock SWMR-write flag raised.
    fn open_swmr_writer<P: AsRef<std::path::Path>>(
        path: P,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        // The SWMR writer always mirrors. `Auto` and unset are *satisfied* by
        // that — they ask for the bounded engine where it applies and accept the
        // mirror where it does not — but `Bounded` is a guarantee, and honoring a
        // guarantee by ignoring it is how a caller ends up spending `O(file size)`
        // memory it asked not to. Refusing is also the permissive direction to be
        // wrong in: if this writer ever runs bounded, the refusal stops firing,
        // which breaks nobody.
        if properties.memory_strategy == Some(MemoryStrategy::Bounded) {
            return Err(Error::EditUnsupported(
                "the SWMR writer always holds the file in a whole-file mirror; leave \
                 MemoryStrategy unset, or pass MemoryStrategy::Auto or MemoryStrategy::Mirrored, \
                 to open it",
            ));
        }
        // A library-version bound below 1.10 is the same shape of unhonorable
        // guarantee. SWMR needs a version 3 superblock — neither library reads
        // the SWMR-write flag back on an older one — so a caller asking for the
        // 1.8 format here is asking for a file this writer cannot produce.
        if let Some((low, high)) = properties.libver_bounds
            && LibVer::resolve_writable(Some((low, high))).map_err(Error::Format)? < LibVer::V110
        {
            return Err(Error::EditUnsupported(
                "the SWMR writer requires a version 3 superblock, which is the v1.10 format; \
                 raise the FileAccessProperties library-version bound to open it",
            ));
        }
        let session = WriteEngine::open_swmr_writer(path, properties.sync_policy)?;
        let mut inner = Self::from_rw_session(session, properties)?;
        inner.swmr_write = true;
        Ok(inner)
    }

    /// After the caller has confirmed a [`Backend::Edit`] backend, gate the
    /// mutation: refuse a sealed file with [`Error::FileClosed`], and in
    /// SWMR-writer mode refuse a staged edit (`staged = true`) with
    /// [`Error::SwmrStagedUnsupported`] — only immediate appends are allowed.
    fn check_mutable(&self, staged: bool) -> Result<(), Error> {
        if self.closed.load(Ordering::Acquire) {
            return Err(Error::FileClosed);
        }
        if staged && self.swmr_write {
            return Err(Error::SwmrStagedUnsupported);
        }
        Ok(())
    }

    /// Gate a staged edit *without* taking the session lock: the backend must
    /// offer the staged surface, and the file must still be mutable.
    ///
    /// This is the same gate the locking helpers apply before locking, split out
    /// so a public method taking a user closure can report a read-only or sealed
    /// file up front, run the closure with no lock held, and take the lock only
    /// to record the result (issue #200).
    fn check_staged_writable(&self) -> Result<(), Error> {
        match &self.backend {
            Backend::Edit(_) => self.check_mutable(true),
            _ => Err(Error::ReadOnly),
        }
    }

    /// A `Source` view over the backend, for the streaming-capable paths.
    pub(crate) fn source(&self) -> SourceView<'_> {
        match &self.backend {
            Backend::InMemory(v) => SourceView::Mem(v),
            Backend::Streaming(s) => SourceView::Stream(s.as_ref()),
            // A mirror or bounded file's bytes live behind a lock and cannot be
            // lent out as a borrowed view; the read paths that reach every
            // backend go through [`with_source`](Self::with_source) instead.
            Backend::Edit(_) => SourceView::Mem(&[]),
        }
    }

    /// Run `f` with a random-access view of this file's bytes, taking the
    /// write-engine lock when the backend requires one. Unlike
    /// [`source`](Self::source) — which cannot lend a borrowed view out of a
    /// lock and returns an empty view for the mirror and bounded backends —
    /// this serves every backend, so it is the dispatch for read paths (heap
    /// reads for variable-length data, chunk enumeration) that must also work
    /// on a read-write file. `f` must not re-enter this file's backend (the
    /// engine lock is held while it runs).
    pub(crate) fn with_source<R>(&self, f: impl FnOnce(&dyn Source) -> R) -> R {
        match &self.backend {
            Backend::InMemory(v) => f(&BytesSource::new(v.as_slice())),
            Backend::Streaming(s) => f(s.as_ref()),
            Backend::Edit(m) => {
                let core = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                f(core.image())
            }
        }
    }

    /// Run a read against a read-write session's file image, choosing the form
    /// its backing can serve: `on_slice` when the session holds the whole file
    /// in memory, so a slice-walking parser borrows the bytes instead of copying
    /// them, and `on_source` otherwise.
    ///
    /// Both closures must compute the same thing. The pair exists because a
    /// mirror can hand out a whole-file slice and a file-backed image cannot,
    /// not because the two backings answer differently (issue #198).
    fn with_engine<R>(
        engine: &Mutex<WriteEngine>,
        on_slice: impl FnOnce(&[u8]) -> R,
        on_source: impl FnOnce(&dyn Source) -> R,
    ) -> R {
        let core = engine
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match core.image_slice() {
            Some(data) => on_slice(data),
            None => on_source(core.image()),
        }
    }

    /// Parse the superblock from `data`, returning it (with `root_group_address`
    /// normalized to an absolute offset) and the base-address offset.
    fn parse_superblock(data: &[u8]) -> Result<(Superblock, u64), Error> {
        let sig_offset = signature::find_signature(data)?;
        let mut superblock = Superblock::parse(data, sig_offset)?;
        let addr_offset = superblock.base_address;
        // Normalize root_group_address to absolute so resolve_path_any works.
        superblock.root_group_address = superblock
            .root_group_address
            .checked_add(addr_offset)
            .ok_or(FormatError::OffsetOverflow {
                offset: superblock.root_group_address,
                length: addr_offset,
            })?;
        debug_assert!(superblock.root_group_address >= addr_offset);
        Ok((superblock, addr_offset))
    }

    /// Streaming counterpart of [`parse_superblock`]: locate and parse the
    /// superblock by reading only small windows from the source.
    fn parse_superblock_source<S: Source + ?Sized>(source: &S) -> Result<(Superblock, u64), Error> {
        let sig_offset = signature::find_signature_in(source)?;
        let mut superblock = Superblock::parse_from_source(source, sig_offset)?;
        let addr_offset = superblock.base_address;
        superblock.root_group_address = superblock
            .root_group_address
            .checked_add(addr_offset)
            .ok_or(FormatError::OffsetOverflow {
                offset: superblock.root_group_address,
                length: addr_offset,
            })?;
        debug_assert!(superblock.root_group_address >= addr_offset);
        Ok((superblock, addr_offset))
    }

    /// Assemble a [`File`] from parsed parts, then load the File Space Info from
    /// the superblock extension (best-effort, so a bad extension never fails the
    /// open).
    fn from_parts(
        backend: Backend,
        superblock: Superblock,
        addr_offset: u64,
        handle: Option<std::fs::File>,
        access_properties: FileAccessProperties,
    ) -> Self {
        let mut file = FileInner {
            backend,
            superblock,
            addr_offset,
            handle,
            file_space_info: None,
            access_properties,
            closed: AtomicBool::new(false),
            swmr_write: false,
        };
        file.file_space_info = file.read_file_space_info();
        file
    }

    /// Parse the File Space Info message from the superblock extension, if the
    /// file records one and it can be read. Best-effort: any failure (no
    /// extension, unreadable object header, malformed message) yields `None`.
    fn read_file_space_info(&self) -> Option<FileSpaceInfo> {
        let rel = self.superblock.superblock_extension_address?;
        if rel == u64::MAX {
            return None;
        }
        let abs = self.addr_offset.checked_add(rel)?;
        let header = self.parse_header(abs).ok()?;
        let msg = header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::FileSpaceInfo)?;
        FileSpaceInfo::parse(
            &msg.data,
            self.superblock.offset_size,
            self.superblock.length_size,
        )
        .ok()
    }

    /// Re-read the file from disk to pick up data appended by a concurrent
    /// writer, then re-parse the superblock.
    ///
    /// This is the SWMR reader's refresh primitive (analogous to the C library's
    /// `H5Drefresh` / h5py's `Dataset.refresh()`): after it returns, newly
    /// fetched [`Dataset`]/[`Group`] handles observe the writer's appended
    /// chunks and extended dimensions, because they re-parse object headers at
    /// their (stable) addresses against the refreshed bytes. Existing handles
    /// borrow `&self`, so they must be dropped before calling this; re-fetch
    /// them afterward.
    ///
    /// Returns [`Error::SwmrUnsupported`] if the file was not opened with
    /// [`File::open_swmr`]. The superblock is checksum-validated on every
    /// re-read; a transient parse failure (a writer caught mid-flush) is
    /// retried a bounded number of times before being surfaced.
    ///
    /// Cost: each call re-reads the entire file from disk (`O(file size)`).
    /// That keeps the implementation simple and correct, but when following a
    /// large, steadily growing log it is the cost paid per refresh; budget
    /// refresh frequency accordingly.
    pub fn refresh(&mut self) -> Result<(), Error> {
        let handle = self.handle.as_mut().ok_or(Error::SwmrUnsupported)?;

        // A writer only appends (the file grows) and updates a few fixed-size,
        // individually checksummed structures in place (superblock EOF, object
        // header dimensions, array header counts). Re-reading the whole file and
        // re-validating the superblock checksum yields a consistent view; if the
        // superblock is caught mid-update, retry.
        const MAX_ATTEMPTS: u32 = 100;
        let mut last_err = None;
        for attempt in 0..MAX_ATTEMPTS {
            let mut data = Vec::new();
            handle.seek(SeekFrom::Start(0)).map_err(Error::Io)?;
            handle.read_to_end(&mut data).map_err(Error::Io)?;
            match Self::parse_superblock(&data) {
                Ok((superblock, addr_offset)) => {
                    self.backend = Backend::InMemory(data);
                    self.superblock = superblock;
                    self.addr_offset = addr_offset;
                    self.file_space_info = self.read_file_space_info();
                    return Ok(());
                }
                Err(e) => {
                    last_err = Some(e);
                    // Brief backoff before re-reading; the writer's in-place
                    // updates are tiny, so a short pause clears the window. Skip
                    // it on the final attempt, where there is no re-read to come.
                    if attempt + 1 < MAX_ATTEMPTS {
                        std::thread::sleep(std::time::Duration::from_micros(
                            50 * (attempt + 1) as u64,
                        ));
                    }
                }
            }
        }
        // The loop always runs at least once and only reaches here via the
        // `Err` arm, so `last_err` is always `Some`; surface the real error.
        Err(last_err.expect("refresh retried at least once before failing"))
    }

    /// Resolve a path to an object-header address, dispatching on the backend.
    fn resolve_path(&self, path: &str) -> Result<u64, Error> {
        Ok(match &self.backend {
            Backend::InMemory(v) => group_v2::resolve_path_any(v, &self.superblock, path)?,
            Backend::Streaming(s) => {
                group_v2::resolve_path_any_from_source(s.as_ref(), &self.superblock, path)?
            }
            // A staged commit can relocate the object tree's root, so this
            // file's cached superblock may name a stale one; resolve against the
            // session's own superblock, which the commit updates.
            Backend::Edit(m) => {
                let core = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                let sb = core.superblock().clone();
                match core.image_slice() {
                    Some(data) => group_v2::resolve_path_any(data, &sb, path)?,
                    None => group_v2::resolve_path_any_from_source(core.image(), &sb, path)?,
                }
            }
        })
    }

    /// The current root-group address (base-adjusted, absolute). For a read-write
    /// [`Backend::Edit`] file a prior relocating commit can have moved the
    /// root, so take the session's own superblock, which the commit updates;
    /// other backends use this file's cached one.
    fn mirror_root_address(&self) -> u64 {
        if let Backend::Edit(m) = &self.backend {
            let core = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            return core.superblock().root_group_address;
        }
        self.superblock.root_group_address
    }

    /// Returns the raw file bytes for an in-memory file, or an empty slice for a
    /// streaming file (which has no whole-file buffer).
    pub fn as_bytes(&self) -> &[u8] {
        match &self.backend {
            Backend::InMemory(v) => v,
            // A streaming, mirror, or bounded file has no borrowable whole-file
            // buffer.
            Backend::Streaming(_) | Backend::Edit(_) => &[],
        }
    }

    /// Return the access properties used when opening this file.
    pub const fn access_properties(&self) -> FileAccessProperties {
        self.access_properties
    }

    /// The backend this file's editing session resolved to, or `None` when there
    /// is no editing session to ask. Always [`EditBacking::Mirrored`] for the
    /// SWMR writer, which builds a session but does not dispatch on the strategy.
    fn edit_backing(&self) -> Option<EditBacking> {
        match &self.backend {
            Backend::Edit(m) => Some(m.lock().unwrap_or_else(|e| e.into_inner()).edit_backing()),
            _ => None,
        }
    }

    /// Returns a reference to the parsed superblock.
    pub fn superblock(&self) -> &Superblock {
        &self.superblock
    }

    /// The whole-file byte image when this file is buffered in memory
    /// ([`open`](Self::open) / [`from_bytes`](Self::from_bytes)); `None` for a
    /// streaming file ([`open_streaming`](Self::open_streaming)). Cross-file
    /// object copy ([`File::copy_from`](crate::File::copy_from)) uses this to read
    /// source objects by absolute address.
    pub(crate) fn in_memory_image(&self) -> Option<&[u8]> {
        match &self.backend {
            Backend::InMemory(data) => Some(data),
            Backend::Streaming(_) | Backend::Edit(_) => None,
        }
    }

    /// The base address (`H5F` superblock base address), i.e. the byte offset
    /// added to every stored relative address. Zero for a file with no
    /// userblock.
    pub(crate) fn base_address(&self) -> u64 {
        self.addr_offset
    }

    /// The file-space management strategy this file records in its superblock
    /// extension (set with `H5Pset_file_space_strategy`), or `None` if the file
    /// records none — the default, which the C library also writes as "no
    /// message". See [`file_space_info`](Self::file_space_info) for the full
    /// record (persist flag, threshold, page size).
    pub fn file_space_strategy(&self) -> Option<FileSpaceStrategy> {
        self.file_space_info.as_ref().map(|info| info.strategy)
    }

    /// The full [`FileSpaceInfo`] recorded in this file's superblock extension,
    /// if present and readable.
    pub fn file_space_info(&self) -> Option<&FileSpaceInfo> {
        self.file_space_info.as_ref()
    }

    /// The free regions a file persists on disk in its free-space managers (when
    /// written with `H5Pset_file_space_strategy(..., persist = true)`), as
    /// `(address, length)` pairs sorted by address.
    ///
    /// Empty when the file does not persist free space, or for the streaming
    /// backend (which does not load the manager blocks). The addresses are file
    /// offsets (relative to the base address); reading data is unaffected by the
    /// presence or absence of these managers.
    pub fn persisted_free_space(&self) -> Vec<(u64, u64)> {
        let Some(info) = &self.file_space_info else {
            return Vec::new();
        };
        if !info.persist {
            return Vec::new();
        }
        let Backend::InMemory(data) = &self.backend else {
            return Vec::new();
        };
        let mut sections = free_space_manager::read_persisted_sections(
            data,
            &info.manager_addrs,
            self.addr_offset,
            self.superblock.offset_size,
        )
        .unwrap_or_default();
        sections.sort_by_key(|s| s.addr);
        sections.into_iter().map(|s| (s.addr, s.size)).collect()
    }

    /// The size of the underlying file in bytes (the HDF5 `H5Fget_filesize`).
    ///
    /// This is the total byte length of the backing store — for a streaming
    /// file the length reported by its source, for an in-memory file the length
    /// of its buffer. It includes any userblock prefix and trailing bytes, so it
    /// may exceed the superblock's logical end-of-file address; compare against
    /// `Superblock::eof_address` (reachable via
    /// [`File::superblock`]) to detect appended or unaccounted tail bytes.
    pub fn file_size(&self) -> u64 {
        match &self.backend {
            Backend::Edit(m) => {
                let core = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                core.image().len()
            }
            _ => self.source().len(),
        }
    }

    /// The minimum library version required to read this file, derived from its
    /// superblock version (the *low bound* of HDF5's `H5Fget_libver_bounds`).
    ///
    /// A version 3 superblock, for example, reports [`LibVer::V110`] because it
    /// was introduced in HDF5 1.10.
    pub fn libver_bound(&self) -> LibVer {
        LibVer::from_superblock_version(self.superblock.version)
    }

    fn parse_header(&self, address: u64) -> Result<ObjectHeader, FormatError> {
        let os = self.superblock.offset_size;
        let ls = self.superblock.length_size;
        match &self.backend {
            Backend::InMemory(v) => {
                ObjectHeader::parse_with_base(v, address.to_usize()?, os, ls, self.addr_offset)
            }
            Backend::Streaming(s) => {
                ObjectHeader::parse_from_source(s.as_ref(), address, os, ls, self.addr_offset)
            }
            Backend::Edit(m) => Self::with_engine(
                m,
                |d| ObjectHeader::parse_with_base(d, address.to_usize()?, os, ls, self.addr_offset),
                |s| ObjectHeader::parse_from_source(s, address, os, ls, self.addr_offset),
            ),
        }
    }

    /// Resolve a base-relative object-header address (the value stored in an
    /// HDF5 `H5R_OBJECT` reference element) to the [`Object`] it points at.
    ///
    /// The stored address is relative to the superblock base address, so any
    /// MAT-file userblock is accounted for here. A null (`0`) or undefined
    /// (`HADDR_UNDEF`) address, or one whose object header is neither a dataset
    /// nor a group, yields [`FormatError::InvalidObjectReference`].
    fn object_at_relative(file: &Arc<FileInner>, rel_addr: u64) -> Result<Object, Error> {
        // HADDR_UNDEF and the null address never name a real object. (Relative
        // address 0 is where the superblock sits, not an object header.)
        if rel_addr == u64::MAX || rel_addr == 0 {
            return Err(FormatError::InvalidObjectReference(rel_addr).into());
        }
        let abs = rel_addr
            .checked_add(file.addr_offset)
            .ok_or(FormatError::InvalidObjectReference(rel_addr))?;
        let hdr = file.parse_header(abs)?;
        if has_message(&hdr, MessageType::DataLayout) {
            let chunk_cache = DatasetAccessProperties::new()
                .resolved_chunk_cache(file.access_properties.chunk_cache);
            Ok(Object::Dataset(Box::new(Dataset {
                file: file.clone(),
                address: abs,
                header: hdr,
                chunk_cache: ChunkCache::with_config(chunk_cache),
                chunk_cache_config: chunk_cache,
                path: None,
            })))
        } else if is_group(&hdr) {
            Ok(Object::Group(Group {
                file: file.clone(),
                address: abs,
                path: None,
            }))
        } else {
            Err(FormatError::InvalidObjectReference(rel_addr).into())
        }
    }

    fn offset_size(&self) -> u8 {
        self.superblock.offset_size
    }

    fn length_size(&self) -> u8 {
        self.superblock.length_size
    }

    /// Resolve the children of a group object header, dispatching on the backend
    /// and converting link addresses to absolute.
    fn group_children(&self, hdr: &ObjectHeader) -> Result<Vec<GroupEntry>, Error> {
        let (os, ls, base) = (self.offset_size(), self.length_size(), self.addr_offset);
        let mut entries = match &self.backend {
            Backend::InMemory(v) => group_v2::resolve_group_entries(v, hdr, os, ls, base),
            Backend::Streaming(s) => {
                group_v2::resolve_group_entries_from_source(s.as_ref(), hdr, os, ls, base)
            }
            Backend::Edit(m) => Self::with_engine(
                m,
                |d| group_v2::resolve_group_entries(d, hdr, os, ls, base),
                |s| group_v2::resolve_group_entries_from_source(s, hdr, os, ls, base),
            ),
        }
        .map_err(Error::Format)?;
        for entry in &mut entries {
            // The stored address is relative to the base address; normalize to an
            // absolute file offset. A crafted entry (e.g. the HADDR_UNDEF sentinel)
            // must not wrap or panic.
            entry.object_header_address = entry.object_header_address.checked_add(base).ok_or(
                FormatError::OffsetOverflow {
                    offset: entry.object_header_address,
                    length: base,
                },
            )?;
        }
        Ok(entries)
    }

    /// The base-adjusted object-header address of the child named `name`, or
    /// `None` if the group has no such child.
    ///
    /// The by-name counterpart of [`group_children`](Self::group_children), and
    /// the one to reach for when a single child is wanted: it stops at the match
    /// rather than building an entry, and an owned name, for every other child
    /// of the group (issue #228).
    fn group_child(&self, group_address: u64, name: &str) -> Result<Option<u64>, Error> {
        let (os, ls, base) = (self.offset_size(), self.length_size(), self.addr_offset);
        let addr = group_address;
        let stored = match &self.backend {
            Backend::InMemory(v) => group_v2::find_child_address(v, addr, os, ls, base, name),
            Backend::Streaming(s) => {
                group_v2::find_child_address_from_source(s.as_ref(), addr, os, ls, base, name)
            }
            Backend::Edit(m) => Self::with_engine(
                m,
                |d| group_v2::find_child_address(d, addr, os, ls, base, name),
                |s| group_v2::find_child_address_from_source(s, addr, os, ls, base, name),
            ),
        }
        .map_err(Error::Format)?;
        // The stored address is relative to the base address; normalize to an
        // absolute file offset, refusing the wrap a crafted entry (e.g. the
        // HADDR_UNDEF sentinel) would otherwise cause — as `group_children` does.
        stored
            .map(|addr| {
                addr.checked_add(base)
                    .ok_or(FormatError::OffsetOverflow {
                        offset: addr,
                        length: base,
                    })
                    .map_err(Error::Format)
            })
            .transpose()
    }

    /// Read all attributes attached to an object header, dispatching on the
    /// backend.
    fn attrs_of(&self, hdr: &ObjectHeader) -> Result<HashMap<String, AttrValue>, Error> {
        let (os, ls, base) = (self.offset_size(), self.length_size(), self.addr_offset);
        let attr_msgs = self.attr_messages_of(hdr)?;
        match &self.backend {
            Backend::Edit(m) => Ok(Self::with_engine(
                m,
                |d| attrs_to_map(&attr_msgs, &BytesSource::new(d), os, ls, base),
                |s| attrs_to_map(&attr_msgs, s, os, ls, base),
            )),
            _ => Ok(attrs_to_map(&attr_msgs, &self.source(), os, ls, base)),
        }
    }

    /// The content of a header message, following the reference when the record
    /// marks the message *shared*.
    ///
    /// A shared record's body is not the message: it is an address, and a
    /// committed (`H5Tcommit`) datatype is stored exactly that way. Decoding the
    /// body directly turns a named `H5T_STD_I32LE` into a zero-width time type
    /// with no error anywhere, so every read of a message that HDF5 permits to be
    /// shared — datatype, dataspace, fill value, filter pipeline — goes through
    /// here. The borrowed case allocates nothing, which is every message this
    /// crate writes and nearly every one it reads.
    fn message_body<'m>(
        &self,
        msg: &'m crate::object_header::HeaderMessage,
    ) -> Result<Cow<'m, [u8]>, Error> {
        if !shared_message::is_shared(msg.flags) {
            return Ok(Cow::Borrowed(&msg.data));
        }
        let (os, ls, base) = (self.offset_size(), self.length_size(), self.addr_offset);
        // A shared reference stores its address relative to the base address, so
        // frame the file at `base` exactly as [`Self::attr_messages_of`] does.
        let resolved = match &self.backend {
            Backend::InMemory(v) => {
                BufferedResolver::new(frame(v, base)?, os, ls).resolve(&msg.data, msg.msg_type)
            }
            Backend::Streaming(s) if base == 0 => {
                SourceResolver::new(s.as_ref(), os, ls).resolve(&msg.data, msg.msg_type)
            }
            Backend::Streaming(s) => SourceResolver::new(
                &BaseOffsetSource {
                    inner: s.as_ref(),
                    base,
                },
                os,
                ls,
            )
            .resolve(&msg.data, msg.msg_type),
            Backend::Edit(m) => Self::with_engine(
                m,
                |d| BufferedResolver::new(frame(d, base)?, os, ls).resolve(&msg.data, msg.msg_type),
                |s| {
                    if base == 0 {
                        SourceResolver::new(s, os, ls).resolve(&msg.data, msg.msg_type)
                    } else {
                        SourceResolver::new(&BaseOffsetSource { inner: s, base }, os, ls)
                            .resolve(&msg.data, msg.msg_type)
                    }
                },
            ),
        }?;
        Ok(Cow::Owned(resolved))
    }

    /// The object-header address a *shared* header message names, or `None` when
    /// the record carries its own content.
    ///
    /// [`Self::message_body`] answers what the message says; this answers which
    /// object says it. A rewrite needs both: the content to reproduce the type,
    /// and the address to tell which users share one committed object rather than
    /// each naming a type of their own.
    pub(crate) fn shared_target_address(
        &self,
        msg: &crate::object_header::HeaderMessage,
    ) -> Result<Option<u64>, Error> {
        if !shared_message::is_shared(msg.flags) {
            return Ok(None);
        }
        let reference =
            shared_message::parse_shared_ref(&msg.data, self.offset_size(), self.length_size())?;
        match reference.location {
            shared_message::SharedLocation::ObjectHeader(addr) => Ok(Some(addr)),
            shared_message::SharedLocation::SohmHeap(_) => {
                Err(Error::Format(FormatError::UnsupportedSohmReference))
            }
        }
    }

    /// Extract every attribute message attached to an object header (compact,
    /// shared, and dense storage), dispatching on the backend.
    pub(crate) fn attr_messages_of(
        &self,
        hdr: &ObjectHeader,
    ) -> Result<Vec<crate::attribute::AttributeMessage>, Error> {
        let (os, ls) = (self.offset_size(), self.length_size());
        // Compact attributes come out of `hdr`, but the two addresses this walk
        // follows are read from message bodies and so are stored relative to the
        // base address: the Attribute Info message's fractal-heap address, and a
        // shared attribute's message address. Frame the file at `base` exactly as
        // [`Self::read_dataset_raw`] does, so both index it directly. For a plain
        // file (`base == 0`) this is the identity; without it, a userblock file's
        // dense attributes are looked for one userblock too early.
        let base = self.addr_offset;
        match &self.backend {
            Backend::InMemory(v) => Ok(extract_attributes_full(frame(v, base)?, hdr, os, ls)?),
            Backend::Streaming(s) if base == 0 => Ok(extract_attributes_full_from_source(
                s.as_ref(),
                hdr,
                os,
                ls,
            )?),
            Backend::Streaming(s) => {
                let framed = BaseOffsetSource {
                    inner: s.as_ref(),
                    base,
                };
                Ok(extract_attributes_full_from_source(&framed, hdr, os, ls)?)
            }
            Backend::Edit(m) => Self::with_engine(
                m,
                |d| Ok(extract_attributes_full(frame(d, base)?, hdr, os, ls)?),
                |s| {
                    if base == 0 {
                        Ok(extract_attributes_full_from_source(s, hdr, os, ls)?)
                    } else {
                        let framed = BaseOffsetSource { inner: s, base };
                        Ok(extract_attributes_full_from_source(&framed, hdr, os, ls)?)
                    }
                },
            ),
        }
    }

    /// Read a dataset's raw bytes for the given layout, dispatching on the backend.
    fn read_dataset_raw(
        &self,
        spec: RawReadSpec<'_>,
        cache: &ChunkCache,
    ) -> Result<Vec<u8>, FormatError> {
        let (os, ls) = (self.offset_size(), self.length_size());
        // Every on-disk address in `dl` — the contiguous data address, the chunk
        // index root, and (followed deeper in the chunked reader) every B-tree /
        // fixed-array / extensible-array node and chunk-data address — is stored
        // relative to the base address. Present the payload reader a base-relative
        // view of the file so all of them index it directly: slice the in-memory
        // buffer at `base`, or wrap the streaming source to add `base` to each
        // read. For a plain file (`base == 0`) this is the identity.
        let base = self.addr_offset;
        match &self.backend {
            Backend::InMemory(v) => {
                data_read::read_raw_data_cached(frame(v, base)?, spec, os, ls, cache)
            }
            Backend::Streaming(s) if base == 0 => {
                data_read::read_raw_data_cached_from_source(s.as_ref(), spec, os, ls, cache)
            }
            Backend::Streaming(s) => {
                let framed = BaseOffsetSource {
                    inner: s.as_ref(),
                    base,
                };
                data_read::read_raw_data_cached_from_source(&framed, spec, os, ls, cache)
            }
            Backend::Edit(m) => Self::with_engine(
                m,
                |data| {
                    let frame = if base == 0 {
                        data
                    } else {
                        let start = base.to_usize()?;
                        data.get(start..).ok_or(FormatError::UnexpectedEof {
                            expected: start,
                            available: data.len(),
                        })?
                    };
                    data_read::read_raw_data_cached(frame, spec, os, ls, cache)
                },
                |s| {
                    let framed = BaseOffsetSource { inner: s, base };
                    data_read::read_raw_data_cached_from_source(&framed, spec, os, ls, cache)
                },
            ),
        }
    }

    /// Windowed counterpart of [`read_dataset_raw`](Self::read_dataset_raw): read
    /// the raw element bytes of the row window `[start_row, start_row + num_rows)`,
    /// touching only the storage it overlaps. Reads through the same base-framed
    /// `Source`, so on-disk addresses resolve the same way. The caller clamps
    /// the window to the dataset.
    fn read_dataset_raw_rows(
        &self,
        spec: RawReadSpec<'_>,
        cache: &ChunkCache,
        pass: CachePass,
        start_row: u64,
        num_rows: u64,
    ) -> Result<Vec<u8>, FormatError> {
        let (os, ls) = (self.offset_size(), self.length_size());
        let (dl, ds, dt) = (spec.layout, spec.dataspace, spec.datatype);
        let elem_size = dt.element_size_usize()?;
        // Elements per row (product of inner dims; 1 when 0-D or 1-D). Checked so
        // a crafted dataspace whose inner dims overflow `usize` errors instead of
        // panicking (debug) or wrapping (release).
        let row_elems: usize = ds.dimensions.iter().skip(1).try_fold(1usize, |acc, &d| {
            acc.checked_mul(d.to_usize()?)
                .ok_or(FormatError::OffsetOverflow {
                    offset: acc as u64,
                    length: d,
                })
        })?;
        let row_bytes =
            row_elems
                .checked_mul(elem_size.get())
                .ok_or(FormatError::OffsetOverflow {
                    offset: row_elems as u64,
                    length: elem_size.get() as u64,
                })?;

        // Compact data is inline in the layout message — no I/O, no framing.
        if let DataLayout::Compact { data } = dl {
            let start = start_row.to_usize()?.checked_mul(row_bytes);
            let len = num_rows.to_usize()?.checked_mul(row_bytes);
            let (Some(start), Some(len)) = (start, len) else {
                return Err(FormatError::OffsetOverflow {
                    offset: start_row,
                    length: row_bytes as u64,
                });
            };
            let end = start.checked_add(len).ok_or(FormatError::OffsetOverflow {
                offset: start as u64,
                length: len as u64,
            })?;
            return data
                .get(start..end)
                .map(<[u8]>::to_vec)
                .ok_or(FormatError::DataSizeMismatch {
                    expected: end,
                    actual: data.len(),
                });
        }

        let base = self.addr_offset;
        match &self.backend {
            Backend::InMemory(v) => {
                let frame = if base == 0 {
                    v.as_slice()
                } else {
                    let start = base.to_usize()?;
                    v.get(start..).ok_or(FormatError::UnexpectedEof {
                        expected: start,
                        available: v.len(),
                    })?
                };
                read_rows_framed(
                    &BytesSource::new(frame),
                    spec,
                    os,
                    ls,
                    cache,
                    pass,
                    start_row,
                    num_rows,
                    row_bytes,
                )
            }
            Backend::Streaming(s) if base == 0 => read_rows_framed(
                s.as_ref(),
                spec,
                os,
                ls,
                cache,
                pass,
                start_row,
                num_rows,
                row_bytes,
            ),
            Backend::Streaming(s) => {
                let framed = BaseOffsetSource {
                    inner: s.as_ref(),
                    base,
                };
                read_rows_framed(
                    &framed, spec, os, ls, cache, pass, start_row, num_rows, row_bytes,
                )
            }
            Backend::Edit(m) => Self::with_engine(
                m,
                |data| {
                    let frame = if base == 0 {
                        data
                    } else {
                        let start = base.to_usize()?;
                        data.get(start..).ok_or(FormatError::UnexpectedEof {
                            expected: start,
                            available: data.len(),
                        })?
                    };
                    read_rows_framed(
                        &BytesSource::new(frame),
                        spec,
                        os,
                        ls,
                        cache,
                        pass,
                        start_row,
                        num_rows,
                        row_bytes,
                    )
                },
                |s| {
                    let framed = BaseOffsetSource { inner: s, base };
                    read_rows_framed(
                        &framed, spec, os, ls, cache, pass, start_row, num_rows, row_bytes,
                    )
                },
            ),
        }
    }
}

/// Read a row window through an already base-framed `Source`. Contiguous
/// layouts are one bounded sub-read; chunked layouts use the windowed chunk
/// reader (only the rank-0 crafted-file corner falls back to a whole read
/// plus slice).
fn read_rows_framed<S: Source + ?Sized>(
    source: &S,
    spec: RawReadSpec<'_>,
    os: u8,
    ls: u8,
    cache: &ChunkCache,
    pass: CachePass,
    start_row: u64,
    num_rows: u64,
    row_bytes: usize,
) -> Result<Vec<u8>, FormatError> {
    let (dl, fill) = (spec.layout, spec.fill);
    // A zero-row window reads nothing, uniformly across the *supported* layouts.
    // A `Virtual` layout is unsupported and must still error like `read_raw`
    // does, so it is excluded here and falls through to the match.
    if num_rows == 0 && !matches!(dl, DataLayout::Virtual { .. }) {
        return Ok(Vec::new());
    }
    match dl {
        DataLayout::Compact { .. } => unreachable!("compact is handled before framing"),
        DataLayout::Contiguous { address, size } => {
            // Unallocated storage: the window reads as the fill value, the same
            // answer the whole-dataset readers give for it.
            let Some(addr) = *address else {
                let len = num_rows.to_usize()?.saturating_mul(row_bytes);
                return fill.buffer(len);
            };
            let start =
                start_row
                    .checked_mul(row_bytes as u64)
                    .ok_or(FormatError::OffsetOverflow {
                        offset: start_row,
                        length: row_bytes as u64,
                    })?;
            let len =
                num_rows
                    .to_usize()?
                    .checked_mul(row_bytes)
                    .ok_or(FormatError::OffsetOverflow {
                        offset: num_rows,
                        length: row_bytes as u64,
                    })?;
            // Never read past the dataset's own contiguous storage.
            if start.saturating_add(len as u64) > *size {
                return Err(FormatError::DataSizeMismatch {
                    expected: start.to_usize()?.saturating_add(len),
                    actual: (*size).to_usize()?,
                });
            }
            let off = addr.checked_add(start).ok_or(FormatError::OffsetOverflow {
                offset: addr,
                length: start,
            })?;
            source.read_exact_at(off, len)
        }
        DataLayout::Chunked { .. } => {
            match crate::chunked_read::read_chunked_rows_from_source(
                source, spec, os, ls, cache, pass, start_row, num_rows,
            )? {
                Some(bytes) => Ok(bytes),
                // Rank-0 chunked (a crafted-file corner): fall back to a whole
                // read, then slice.
                None => {
                    let full =
                        data_read::read_raw_data_cached_from_source(source, spec, os, ls, cache)?;
                    let start = start_row.to_usize()? * row_bytes;
                    let len = num_rows.to_usize()? * row_bytes;
                    full.get(start..start + len).map(<[u8]>::to_vec).ok_or(
                        FormatError::DataSizeMismatch {
                            expected: start + len,
                            actual: full.len(),
                        },
                    )
                }
            }
        }
        DataLayout::Virtual { .. } => Err(FormatError::UnsupportedVirtualLayout),
    }
}

impl std::fmt::Debug for FileInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("File")
            .field("size", &self.file_size())
            .field("superblock_version", &self.superblock.version)
            .finish()
    }
}

/// An open HDF5 file.
///
/// A `File` is an owned, cheaply cloneable handle to an open file: cloning it (or
/// deriving a [`Dataset`]/[`Group`] from it) shares one underlying open file
/// rather than re-reading it. Object handles returned by [`dataset`](Self::dataset),
/// [`group`](Self::group), and [`root`](Self::root) are **owned** — they keep the
/// file open for as long as they live and carry no borrow of the `File`, so they
/// can be stored in a struct, cached, and moved across threads.
#[derive(Clone)]
pub struct File {
    inner: Arc<FileInner>,
}

impl std::fmt::Debug for File {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&*self.inner, f)
    }
}

impl File {
    /// Open an HDF5 file from a filesystem path.
    ///
    /// Reads the file into memory once. To follow a file that a concurrent
    /// single writer is appending to (SWMR), use [`File::open_swmr`] instead.
    /// To read a file larger than memory (e.g. on a 32-bit host) without
    /// buffering it, use [`File::open_streaming`].
    ///
    /// A file whose superblock marks it as held by a writer is refused with
    /// [`Error::FileMarkedInUse`](crate::Error::FileMarkedInUse) — the check
    /// `H5Fopen` makes of the same byte. That means a live writer or one that
    /// exited without closing the file; clear a stale flag with
    /// [`clear_swmr_flag`](Self::clear_swmr_flag), and follow a live SWMR writer
    /// with [`open_swmr`](Self::open_swmr). [`from_bytes`](Self::from_bytes) does
    /// not check, since its caller already holds the bytes — which is also the
    /// way to read a flagged file on a read-only mount, where clearing the flag
    /// would need write access.
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::open(path)?),
        })
    }

    /// Open an HDF5 file from a filesystem path with explicit access properties.
    pub fn open_with_options<P: AsRef<std::path::Path>>(
        path: P,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::open_with_options(path, properties)?),
        })
    }

    /// Open an HDF5 file for **streaming** reads, fetching regions on demand from
    /// the file instead of buffering it whole.
    ///
    /// This lets a host read a file larger than its address space. Metadata and
    /// dataset chunks are read through a `ReadSeekSource`, so peak memory stays
    /// close to one chunk plus the metadata being parsed; chunks adjacent on
    /// disk are fetched together, in reads of at most 256 KiB, and a chunk
    /// larger than that is read on its own. Attribute reading and v1
    /// symbol-table groups on the resolved path are not yet supported on this
    /// backend.
    ///
    /// Like [`open`](Self::open), this refuses a file whose superblock marks it
    /// as held by a writer.
    pub fn open_streaming<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::open_streaming(path)?),
        })
    }

    /// Open an HDF5 file for streaming reads with explicit access properties.
    pub fn open_streaming_with_options<P: AsRef<std::path::Path>>(
        path: P,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::open_streaming_with_options(path, properties)?),
        })
    }

    /// Open an HDF5 file for SWMR (single-writer/multiple-reader) reading.
    ///
    /// Like [`File::open`], but retains a live handle to the file so that
    /// [`File::refresh`] can re-read data appended by a concurrent writer.
    ///
    /// This is the open that *follows* a file marked as held by a SWMR writer,
    /// where [`open`](Self::open) refuses one. Only a half-set mark is refused
    /// here, with [`Error::FileMarkedInUse`](crate::Error::FileMarkedInUse):
    /// either bit without the other. Write access alone is what a plain
    /// (non-SWMR) writer leaves, and there is no protocol for following a writer
    /// that is not publishing consistent prefixes; the SWMR bit alone is a state
    /// no writer produces. Both bits is the live SWMR writer this exists to
    /// follow, and neither is a quiescent file.
    #[doc(alias = "H5F_ACC_SWMR_READ")]
    pub fn open_swmr<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::open_swmr(path)?),
        })
    }

    /// Open an HDF5 file for SWMR reading with explicit access properties.
    pub fn open_swmr_with_options<P: AsRef<std::path::Path>>(
        path: P,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::open_swmr_with_options(path, properties)?),
        })
    }

    /// Open an HDF5 file from an in-memory byte vector.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::from_bytes(data)?),
        })
    }

    /// Open an HDF5 file from an in-memory byte vector with explicit access properties.
    pub fn from_bytes_with_options(
        data: Vec<u8>,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::from_bytes_with_options(data, properties)?),
        })
    }

    /// Open an existing HDF5 file for reading **and** in-place editing.
    ///
    /// Unlike [`open`](Self::open) (read-only, buffered), this takes an exclusive
    /// OS file lock held for the file's life and lets owned handles modify the
    /// file — immediate [`Dataset::append`]s, plus [`Dataset::write`]/`set_attr`,
    /// [`Group::create_dataset`]/`create_group`/`delete`/`set_attr`, and
    /// [`copy`](Self::copy)/[`copy_from`](Self::copy_from) staged until
    /// [`commit`](Self::commit). The file must use 8-byte offsets and lengths and
    /// keep its superblock at its base address (a canonical userblock, as in a
    /// MATLAB `.mat` file, is supported); anything else is refused with
    /// [`Error::EditUnsupported`](crate::Error::EditUnsupported).
    ///
    /// The fast immediate [`Dataset::append`] additionally requires a
    /// latest-format (version-2/3) file with no userblock and an
    /// Extensible-Array-indexed dataset; [`Dataset::append_staged`] covers the
    /// general case.
    ///
    /// Two things can turn this open away because another writer holds the file:
    /// the exclusive OS lock, reported as
    /// [`Error::FileLocked`](crate::Error::FileLocked), and the superblock's
    /// status-flags byte, reported as
    /// [`Error::FileMarkedInUse`](crate::Error::FileMarkedInUse). The second
    /// covers what the first cannot — a SWMR writer takes no lock, and a writer
    /// that exited without closing the file leaves the flag behind; recover a
    /// stale one with [`clear_swmr_flag`](Self::clear_swmr_flag).
    ///
    /// # Memory
    ///
    /// This picks its backing from the file rather than making the caller pick a
    /// function (issue #198): a latest-format file with no userblock is edited
    /// **bounded**, holding only the metadata being parsed plus the configured
    /// caches plus what an edit is building, so resident memory does not scale
    /// with the file; anything else falls back to a whole-file in-memory mirror,
    /// which is what makes a pre-v2 or userblock file editable at all. The two
    /// backings are the same engine over different storage and offer the same
    /// edit surface, differing in one trade: the bounded one applies a large
    /// immediate append in whole-chunk batches, each crash-atomic on its own, so
    /// a crash mid-call leaves a valid shorter dataset rather than none of the
    /// append. Ask a file which it got with
    /// [`edit_backing`](Self::edit_backing), and demand one with
    /// [`FileAccessProperties::with_memory_strategy`] —
    /// [`MemoryStrategy::Mirrored`] restores the unconditional mirror this
    /// entry point used before it learned to dispatch.
    #[doc(alias = "H5Fopen")]
    pub fn open_rw<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        Self::open_rw_with_options(path, FileAccessProperties::new())
    }

    /// Open an existing file for reading and in-place editing with explicit
    /// access properties — see [`open_rw`](Self::open_rw).
    ///
    /// The properties carry the locking policy (the `H5Pset_file_locking` analogue,
    /// [`FileAccessProperties::with_locking`]), the memory strategy
    /// ([`FileAccessProperties::with_memory_strategy`], which overrides the
    /// dispatch described on [`open_rw`](Self::open_rw)), the `fsync` cadence
    /// ([`FileAccessProperties::with_sync_policy`]), the metadata cache used by
    /// the bounded backing, and the file-wide chunk-cache default applied to
    /// datasets opened from this file. Because one [`FileAccessProperties`] value
    /// serves every open, the same configuration can be shared with a read path.
    pub fn open_rw_with_options<P: AsRef<std::path::Path>>(
        path: P,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::open_rw(path, properties)?),
        })
    }

    /// Open exactly as [`open_rw`](Self::open_rw) does, but behind an image that
    /// withholds its whole-file slice, so every read takes the `Source` path
    /// rather than the slice fast path.
    ///
    /// Each read this file serves has two forms (see `with_engine`), and only
    /// the slice form runs in production until a mirrorless backing lands
    /// (issue #198). Opening the same file both ways and comparing is what
    /// holds the other form to the same answers in the meantime.
    #[cfg(test)]
    pub(crate) fn open_rw_source_only(path: &std::path::Path) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::from_rw_session(
                WriteEngine::open_source_only(path)?,
                FileAccessProperties::new(),
            )?),
        })
    }

    /// Open an existing file for **SWMR** (single-writer/multiple-reader)
    /// appending: take **no** OS lock (so concurrent readers, and Windows'
    /// mandatory locks, are never blocked) and raise the superblock's SWMR-write
    /// flag so a reader may attach with [`File::open_swmr`], the C library's
    /// `H5F_ACC_SWMR_READ`, or h5py `swmr=True`.
    ///
    /// Only immediate [`Dataset::append`] is permitted, and only over the SWMR
    /// subset — an **unfiltered**, chunk-aligned append, so a concurrent reader
    /// only ever observes a consistent prefix; a filtered or non-chunk-aligned
    /// append returns [`Error::SwmrAppendUnsupported`](crate::Error::SwmrAppendUnsupported).
    /// The staged edit surface (`write`/`set_attr`/`create_*`/`delete`/`copy`/
    /// `commit`) returns
    /// [`Error::SwmrStagedUnsupported`](crate::Error::SwmrStagedUnsupported).
    /// [`close`](Self::close) clears the SWMR-write flag; a writer that exits
    /// without a clean close leaves it set — recover with
    /// [`clear_swmr_flag`](Self::clear_swmr_flag). While the flag stands, this
    /// open is refused with
    /// [`Error::FileMarkedInUse`](crate::Error::FileMarkedInUse), which is what
    /// keeps a second writer off a file SWMR gives only one (no OS lock is held
    /// to do it).
    ///
    /// Requires a latest-format (version-3 superblock) file with no userblock
    /// and no persisted free-space; other files are refused with
    /// [`Error::SwmrAppendUnsupported`](crate::Error::SwmrAppendUnsupported).
    /// The version-3 requirement is the C library's: neither library reads the
    /// SWMR-write flag back on an older superblock, so raising one there would
    /// announce the writer to nobody.
    #[doc(alias = "H5F_ACC_SWMR_WRITE")]
    pub fn open_swmr_writer<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        Self::open_swmr_writer_with_options(path, FileAccessProperties::new())
    }

    /// Open for SWMR appending with explicit access properties — see
    /// [`open_swmr_writer`](Self::open_swmr_writer).
    ///
    /// The properties' chunk cache is the file-wide default for datasets opened
    /// from this file. Its locking policy is ignored, which costs the caller
    /// nothing: SWMR takes no OS lock by design, which is stronger than any
    /// locking a caller could ask for. Its memory strategy is *not* ignored the
    /// same way — this writer always mirrors, so an explicit
    /// [`MemoryStrategy::Bounded`] is a guarantee it cannot meet and is refused
    /// with [`Error::EditUnsupported`]; [`MemoryStrategy::Auto`] and
    /// [`MemoryStrategy::Mirrored`] are both satisfied by the mirror.
    ///
    /// Its [`SyncPolicy`](crate::SyncPolicy) applies here as to any other
    /// read-write session, the SWMR-write flag included; a reader on this
    /// machine is unaffected either way, since the barriers carry the write
    /// order across power loss rather than across processes.
    pub fn open_swmr_writer_with_options<P: AsRef<std::path::Path>>(
        path: P,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::open_swmr_writer(path, properties)?),
        })
    }

    /// Clear a stale SWMR-write flag left in `path` by a writer that exited
    /// without a clean [`close`](Self::close) — the `h5clear -s` equivalent, for
    /// recovering a file that both this crate and the reference C library
    /// otherwise refuse to open ([`Error::FileMarkedInUse`](crate::Error::FileMarkedInUse)).
    /// A no-op if the flag is already clear.
    ///
    /// It takes the exclusive OS lock first, so it cannot clear the flag out
    /// from under a *live* [`open_rw`](Self::open_rw) writer. A live SWMR writer
    /// holds no lock, so make sure it is really gone: clearing the flag under
    /// one leaves its readers with no record that it is publishing.
    pub fn clear_swmr_flag<P: AsRef<std::path::Path>>(path: P) -> Result<(), Error> {
        crate::file_lock::clear_swmr_flag_at(path.as_ref())
    }

    /// Create a new, empty HDF5 file at `path` and open it for reading and
    /// writing, so its contents can be built entirely through owned handles
    /// ([`Group::create_dataset`]/[`create_group`](Group::create_group), then
    /// [`commit`](Self::commit)).
    ///
    /// Overwrites any existing file at `path`. For an all-at-once write, use
    /// [`FileBuilder`](crate::FileBuilder) instead.
    #[doc(alias = "H5Fcreate")]
    pub fn create<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        Self::create_with_options(
            path,
            FileCreateProperties::new(),
            FileAccessProperties::new(),
        )
    }

    /// Create a new, empty HDF5 file with explicit creation and access properties,
    /// then open it for reading and writing — see [`create`](Self::create).
    ///
    /// Mirrors `H5Fcreate(name, flags, fcpl_id, fapl_id)`: `create` carries the
    /// creation properties recorded in the new file (userblock, file-space
    /// strategy, library-version bounds), and `access` the properties governing
    /// the handle returned (locking policy, `fsync` cadence, chunk cache). Both are values, so a
    /// layout defined once can be reused across every file an application writes.
    ///
    /// A creation property is validated as the file is written, so an invalid
    /// userblock or page size surfaces here rather than when the properties were
    /// built. A file created with [`FileSpaceStrategy::Page`] can be grown
    /// through either editor, by an immediate [`Dataset::append`] or a staged
    /// commit, provided it also persists its free space (issue #198).
    pub fn create_with_options<P: AsRef<std::path::Path>>(
        path: P,
        create: FileCreateProperties,
        access: FileAccessProperties,
    ) -> Result<Self, Error> {
        // Refuse a pair the reopen below would refuse, before anything is
        // written: this call promises a file *and* an open handle, and half of
        // that is worse than neither.
        if let Some(reason) = crate::edit::create_would_refuse_reopen(&create, &access) {
            return Err(Error::EditUnsupported(reason));
        }
        let mut builder = crate::writer::FileBuilder::new();
        builder.with_create_properties(create);
        let bytes = builder.finish()?;
        std::fs::write(path.as_ref(), bytes).map_err(Error::Io)?;
        Self::open_rw_with_options(path, access)
    }

    /// Apply all staged structural edits made through this file's handles —
    /// [`Dataset::write`]/`set_attr`/`remove_attr` and
    /// [`Group::create_group`]/`delete` — as one transaction. Immediate
    /// [`Dataset::append`]s need no commit.
    ///
    /// Requires a read-write file ([`File::open_rw`]); a read-only file returns
    /// [`Error::ReadOnly`](crate::Error::ReadOnly). A commit that relocates
    /// objects invalidates outstanding handles — re-fetch any you keep using.
    ///
    /// The commit is durable when it returns, under the default
    /// [`SyncPolicy::Always`]; under
    /// [`SyncPolicy::OnClose`](crate::SyncPolicy::OnClose) it has reached the
    /// operating system and waits for a [`sync`](Self::sync).
    pub fn commit(&self) -> Result<(), Error> {
        self.with_mirror_session(true, |session| session.commit())
    }

    /// Copy the object at `src` to `dst` within this file (the in-file
    /// `H5Ocopy`), staged until [`commit`](Self::commit).
    ///
    /// Requires a read-write file ([`File::open_rw`]); a read-only file returns
    /// [`Error::ReadOnly`](crate::Error::ReadOnly).
    pub fn copy(&self, src: &str, dst: &str) -> Result<(), Error> {
        self.with_mirror_session(true, |session| {
            session.copy(&normalize_path(src), &normalize_path(dst))
        })
    }

    /// Copy the object at `src` in `source` — a separate, buffered read-only
    /// file — into this file at `dst`: the cross-file `H5Ocopy`, staged until
    /// [`commit`](Self::commit).
    ///
    /// `source` must be a buffered file ([`File::open`] or [`File::from_bytes`],
    /// not [`File::open_streaming`]) that uses 8-byte offsets and has no
    /// userblock; anything else is refused with
    /// [`Error::EditUnsupported`](crate::Error::EditUnsupported). The source
    /// subtree is read and validated eagerly, so `source` need not outlive this
    /// call. Requires a read-write destination ([`File::open_rw`]); a read-only
    /// one returns [`Error::ReadOnly`](crate::Error::ReadOnly).
    pub fn copy_from(&self, source: &File, src: &str, dst: &str) -> Result<(), Error> {
        self.with_mirror_session(true, |session| session.copy_from(source, src, dst))
    }

    /// Report whether this file has structural edits staged but not yet applied
    /// by [`commit`](Self::commit) — [`Dataset::write`]/`set_attr`/`remove_attr`,
    /// [`Dataset::append_staged`], [`Group::create_group`]/`create_dataset`/
    /// `delete`/`set_attr`/`remove_attr`, and [`copy`](Self::copy)/
    /// [`copy_from`](Self::copy_from). Immediate [`Dataset::append`]s are never
    /// staged and do not count. Always `false` for a read-only file.
    pub fn has_staged_edits(&self) -> bool {
        match &self.inner.backend {
            Backend::Edit(m) => {
                let session = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                session.has_staged_edits()
            }
            _ => false,
        }
    }

    /// Report this read-write file's live space usage as a [`SpaceAccounting`] —
    /// the current logical size, total reusable free bytes, and reusable free
    /// regions. It reflects committed state plus immediate in-place appends, not
    /// edits still staged for [`commit`](Self::commit).
    ///
    /// Requires a read-write file ([`File::open_rw`]); a read-only file returns
    /// [`Error::ReadOnly`](crate::Error::ReadOnly).
    pub fn space_accounting(&self) -> Result<SpaceAccounting, Error> {
        match &self.inner.backend {
            Backend::Edit(m) => {
                let session = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                Ok(session.space_accounting())
            }
            _ => Err(Error::ReadOnly),
        }
    }

    /// Force everything written to this file so far to durable storage — the
    /// `fsync` the application issues at its own cadence under
    /// [`SyncPolicy::OnClose`], and a redundant one under the default
    /// [`SyncPolicy::Always`]. A SWMR-writer file syncs the same way.
    ///
    /// This is a durability barrier, not a flush: it writes nothing itself.
    /// Staged edits are not applied ([`commit`](Self::commit) does that, and a
    /// `sync` before one makes only the *previous* state durable), and elements
    /// held by a live [`BufferedAppender`](crate::BufferedAppender) have not
    /// reached the file at all — flush it first.
    ///
    /// There is no need to call it before [`close`](Self::close): `close` — and
    /// dropping the last handle — issues its own barrier under every policy,
    /// because both write and both destroy the handle that would have ordered
    /// those writes. This is the mid-session checkpoint, not the closing one.
    ///
    /// Requires a read-write file ([`File::open_rw`]); a read-only file returns
    /// [`Error::ReadOnly`](crate::Error::ReadOnly), and a sealed one
    /// [`Error::FileClosed`](crate::Error::FileClosed) — a closed file has
    /// already been synced.
    #[doc(alias = "fsync")]
    pub fn sync(&self) -> Result<(), Error> {
        self.with_mirror_session(false, |session| session.force_sync())
    }

    /// Commit any staged edits and seal this file. The exclusive OS lock is
    /// released once the last handle derived from this file is also dropped.
    ///
    /// After `close`, a write through any surviving [`Dataset`]/[`Group`] handle
    /// or [`File`] clone returns [`Error::FileClosed`](crate::Error::FileClosed);
    /// reads still work.
    pub fn close(self) -> Result<(), Error> {
        if matches!(self.inner.backend, Backend::Edit(_)) {
            if self.inner.swmr_write {
                // SWMR mode stages nothing (the staged surface is refused), so do
                // not commit — clear the SWMR-write flag and flush, marking the
                // file cleanly closed for any concurrent reader.
                if let Backend::Edit(m) = &self.inner.backend {
                    let mut session = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                    session.set_consistency_flags(0)?;
                    // Forced, not left to the policy: this call consumes the
                    // handle, so nothing the caller still holds could order the
                    // flag write. A file left flagged is refused by every later
                    // open until `clear_swmr_flag`, which makes it the write
                    // here whose loss costs availability rather than freshness.
                    session.force_sync()?;
                }
            } else {
                self.commit()?;
                // Immediate appends grow the file past any persisted free-space
                // managers without running a commit tail, so re-home them here.
                // A no-op unless this session left them stale.
                // The barrier covering both: the commit above, and the file
                // length that finalize's manager rewrite changed. Forced under
                // every policy — `close` consumes the handle, so this is the last
                // point at which either write can be ordered at all.
                self.with_mirror_session(false, |session| {
                    session.finalize_persist()?;
                    session.force_sync()
                })?;
            }
            self.inner.closed.store(true, Ordering::Release);
        }
        Ok(())
    }

    /// Run `f` with the locked write session of a read-write file. `staged`
    /// distinguishes an edit applied by [`commit`](Self::commit) from an immediate
    /// one. Returns [`Error::ReadOnly`](crate::Error::ReadOnly) for a read-only
    /// file, [`Error::FileClosed`](crate::Error::FileClosed) once the file is
    /// sealed by [`close`](Self::close), and
    /// [`Error::SwmrStagedUnsupported`](crate::Error::SwmrStagedUnsupported) for a
    /// staged edit on a SWMR-writer file.
    fn with_mirror_session<R>(
        &self,
        staged: bool,
        f: impl FnOnce(&mut WriteEngine) -> Result<R, Error>,
    ) -> Result<R, Error> {
        let Backend::Edit(m) = &self.inner.backend else {
            return Err(Error::ReadOnly);
        };
        self.inner.check_mutable(staged)?;
        let mut session = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        f(&mut session)
    }

    /// Returns an owned handle to the root group.
    pub fn root(&self) -> Group {
        Group {
            // A relocating commit on a read-write file can move the root, so
            // resolve it from the live mirror rather than the cached superblock.
            address: self.inner.mirror_root_address(),
            file: self.inner.clone(),
            path: Some(String::new()),
        }
    }

    /// Resolve a path and return an owned [`Dataset`] handle.
    ///
    /// The dataset uses the file-wide chunk-cache default (configured with
    /// [`FileAccessProperties::with_chunk_cache`]). To override the cache for this
    /// one dataset, use [`dataset_with_options`](Self::dataset_with_options).
    pub fn dataset(&self, path: &str) -> Result<Dataset, Error> {
        self.dataset_with_options(path, DatasetAccessProperties::new())
    }

    /// Resolve a path and return an owned [`Dataset`] handle, applying per-dataset
    /// [`DatasetAccessProperties`] that override file-wide access defaults.
    ///
    /// This is the dataset-open-with-access-property-list path (HDF5's `dapl`):
    /// the properties' chunk cache corresponds to `H5Pset_chunk_cache` and takes
    /// precedence, for this dataset only, over the `H5Pset_cache`-style
    /// file-wide default.
    pub fn dataset_with_options(
        &self,
        path: &str,
        properties: DatasetAccessProperties,
    ) -> Result<Dataset, Error> {
        let addr = self.inner.resolve_path(path)?;
        let hdr = self.inner.parse_header(addr)?;
        if !has_message(&hdr, MessageType::DataLayout) {
            return Err(Error::NotADataset(path.to_string()));
        }
        let chunk_cache = properties.resolved_chunk_cache(self.inner.access_properties.chunk_cache);
        Ok(Dataset {
            file: self.inner.clone(),
            address: addr,
            header: hdr,
            chunk_cache: ChunkCache::with_config(chunk_cache),
            chunk_cache_config: chunk_cache,
            path: Some(normalize_path(path)),
        })
    }

    /// Resolve a path and return an owned [`Group`] handle.
    pub fn group(&self, path: &str) -> Result<Group, Error> {
        let addr = self.inner.resolve_path(path)?;
        Ok(Group {
            file: self.inner.clone(),
            address: addr,
            path: Some(normalize_path(path)),
        })
    }

    /// Re-read the file from disk to pick up data appended by a concurrent
    /// writer, then re-parse the superblock.
    ///
    /// This is the SWMR reader's refresh primitive. Returns
    /// [`Error::SwmrUnsupported`] if the file was not opened with
    /// [`File::open_swmr`], and [`Error::HandlesOutstanding`] if any owned
    /// [`Dataset`]/[`Group`] handle (or a clone of this `File`) is still alive —
    /// drop them before refreshing, then re-fetch them afterward, since they
    /// observe the new bytes only when re-derived from the refreshed file.
    pub fn refresh(&mut self) -> Result<(), Error> {
        let inner = Arc::get_mut(&mut self.inner).ok_or(Error::HandlesOutstanding)?;
        inner.refresh()
    }

    // --- delegating value getters (forward to the shared inner state) ---

    /// Returns the raw file bytes for an in-memory file, or an empty slice for a
    /// streaming file (which has no whole-file buffer).
    pub fn as_bytes(&self) -> &[u8] {
        self.inner.as_bytes()
    }

    /// Return the access properties used when opening this file.
    pub fn access_properties(&self) -> FileAccessProperties {
        self.inner.access_properties()
    }

    /// Which backend this file's read-write session resolved to:
    /// [`EditBacking::Bounded`] when it reads through a handle, or
    /// [`EditBacking::Mirrored`] when it holds a whole-file image.
    ///
    /// This is how a caller who opened with [`MemoryStrategy::Auto`] finds out
    /// whether the fallback was taken, and so whether memory scales with the
    /// file. A file with no editing session — a read-only open, a streaming open
    /// — reports `None`.
    ///
    /// The answer is an [`EditBacking`] rather than the [`MemoryStrategy`] that
    /// was asked for, because `Auto` is a preference between the two backends and
    /// not an outcome either can report; `.into()` converts back when a later
    /// reopen should be pinned to what this one got.
    pub fn edit_backing(&self) -> Option<EditBacking> {
        self.inner.edit_backing()
    }

    /// Returns a reference to the parsed superblock.
    pub fn superblock(&self) -> &Superblock {
        self.inner.superblock()
    }

    /// The file-space management strategy this file records in its superblock
    /// extension, or `None` if it records none.
    pub fn file_space_strategy(&self) -> Option<FileSpaceStrategy> {
        self.inner.file_space_strategy()
    }

    /// The full [`FileSpaceInfo`] recorded in this file's superblock extension,
    /// if present and readable.
    pub fn file_space_info(&self) -> Option<&FileSpaceInfo> {
        self.inner.file_space_info()
    }

    /// The free regions a file persists on disk in its free-space managers, as
    /// `(address, length)` pairs sorted by address.
    pub fn persisted_free_space(&self) -> Vec<(u64, u64)> {
        self.inner.persisted_free_space()
    }

    /// The size of the underlying file in bytes (the HDF5 `H5Fget_filesize`).
    pub fn file_size(&self) -> u64 {
        self.inner.file_size()
    }

    /// The minimum library version required to read this file, derived from its
    /// superblock version (the *low bound* of HDF5's `H5Fget_libver_bounds`).
    pub fn libver_bound(&self) -> LibVer {
        self.inner.libver_bound()
    }

    /// A `Source` view over the backend, for the streaming-capable paths.
    pub(crate) fn source(&self) -> SourceView<'_> {
        self.inner.source()
    }

    /// The whole-file byte image when this file is buffered in memory; `None`
    /// for a streaming file. Used by cross-file object copy.
    pub(crate) fn in_memory_image(&self) -> Option<&[u8]> {
        self.inner.in_memory_image()
    }

    /// The base address (superblock base address) added to every stored relative
    /// address. Zero for a file with no userblock.
    pub(crate) fn base_address(&self) -> u64 {
        self.inner.base_address()
    }
}

// ---------------------------------------------------------------------------
// Object reference target
// ---------------------------------------------------------------------------

/// The resolved target of an HDF5 object reference (`H5R_OBJECT`): either a
/// group or a dataset.
///
/// Produced by [`Dataset::dereference`]. MATLAB `.mat` files use object
/// references pervasively — a cell array stores one reference per element, and
/// the `#subsystem#` machinery references its payloads — so resolving a
/// reference to the group or dataset it names is the foundation for reading
/// those structures.
///
/// The [`Dataset`](Object::Dataset) handle is boxed: it carries a parsed object
/// header and is much larger than a [`Group`](Object::Group) handle, so boxing
/// keeps `Object` (and a `Vec<Object>`) compact without a size disparity. The
/// `Box` derefs transparently, so `&obj_dataset` is usable wherever a
/// `&Dataset` is expected.
///
/// Non-exhaustive: a reference can name an object kind this crate does not yet
/// resolve — a committed (named) datatype is refused with
/// [`FormatError::InvalidObjectReference`](crate::FormatError::InvalidObjectReference)
/// today — so match with a `_` arm.
#[non_exhaustive]
pub enum Object {
    /// The reference points at a group's object header.
    Group(Group),
    /// The reference points at a dataset's object header.
    Dataset(Box<Dataset>),
}

impl std::fmt::Debug for Object {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Object::Group(_) => f.write_str("Object::Group"),
            Object::Dataset(_) => f.write_str("Object::Dataset"),
        }
    }
}

// ---------------------------------------------------------------------------
// Group handle
// ---------------------------------------------------------------------------

/// A group that exists only as a staged edit, handed to
/// [`Group::create_group_with`]'s closure so attributes can be set on a group
/// that is not yet committed (and so has no resolvable header to hang a
/// [`Group`] handle off).
///
/// Every method stages; nothing is written until [`File::commit`], and a staged
/// object is not resolvable by name until then.
///
/// The closure holding this records into a buffer rather than into the file's
/// writable session, so nothing is locked while it runs. The recorded operations
/// are applied together when it returns, which is also why a staged object is not
/// resolvable until [`File::commit`].
pub struct StagedGroup<'a> {
    ops: &'a mut Vec<StagedOp>,
    path: String,
}

impl StagedGroup<'_> {
    /// Stage an attribute on this group, applied with its creation on
    /// [`File::commit`].
    pub fn set_attr(&mut self, name: &str, value: AttrValue) -> &mut Self {
        self.ops.push(StagedOp::SetGroupAttr {
            path: self.path.clone(),
            name: name.to_string(),
            value,
        });
        self
    }

    /// Stage an empty subgroup of this group.
    ///
    /// To configure it in the same commit, use
    /// [`create_group_with`](Self::create_group_with).
    pub fn create_group(&mut self, name: &str) -> &mut Self {
        self.create_group_with(name, |_| {})
    }

    /// Stage a subgroup of this group, configured through `build`.
    pub fn create_group_with(
        &mut self,
        name: &str,
        build: impl FnOnce(&mut StagedGroup<'_>),
    ) -> &mut Self {
        let child = format!("{}/{}", self.path, name);
        self.ops.push(StagedOp::CreateGroup(child.clone()));
        let mut staged = StagedGroup {
            ops: &mut *self.ops,
            path: child,
        };
        build(&mut staged);
        self
    }

    /// Stage a dataset in this group, configured through `build`.
    pub fn create_dataset(
        &mut self,
        name: &str,
        build: impl FnOnce(&mut DatasetBuilder),
    ) -> &mut Self {
        let mut builder = DatasetBuilder::new(name);
        build(&mut builder);
        self.ops.push(StagedOp::CreateDataset {
            path: format!("{}/{}", self.path, name),
            builder: Box::new(builder),
        });
        self
    }
}

/// One edit recorded by a [`StagedGroup`] closure, replayed onto the writable
/// session after the closure returns.
///
/// The indirection is what keeps user code off the session lock: the closure
/// touches only this buffer, so calling back into the same [`File`] from inside
/// it is at worst wrongly ordered rather than a deadlock (issue #200).
enum StagedOp {
    CreateGroup(String),
    SetGroupAttr {
        path: String,
        name: String,
        value: AttrValue,
    },
    CreateDataset {
        path: String,
        /// Boxed because a `DatasetBuilder` dwarfs the other variants, and a
        /// closure staging many groups would otherwise pay its size per entry.
        builder: Box<DatasetBuilder>,
    },
}

impl StagedOp {
    /// Record this edit on the session. Applied in the order the closure made
    /// the calls, so a group is always staged before its own attributes and
    /// children.
    fn apply(self, session: &mut WriteEngine) -> Result<(), Error> {
        match self {
            StagedOp::CreateGroup(path) => session.create_group(&path),
            StagedOp::SetGroupAttr { path, name, value } => {
                session.set_group_attr(&path, &name, value)
            }
            StagedOp::CreateDataset { path, builder } => {
                session.stage_created_dataset(&path, *builder)
            }
        }
    }
}

/// An owned handle to an HDF5 group.
pub struct Group {
    file: Arc<FileInner>,
    address: u64,
    /// Root-relative path of this group (e.g. `""` for the root, `"a/b"`), used
    /// to address the group and its children for write operations on a
    /// read-write file. `None` for a group reached by object reference
    /// ([`Dataset::dereference`]), which has no resolvable path.
    path: Option<String>,
}

impl Group {
    /// Address of this group's object header (base-adjusted, file-absolute).
    /// Used to resolve object references that point at this group.
    pub(crate) fn header_address(&self) -> u64 {
        self.address
    }

    /// List the names of datasets in this group.
    ///
    /// To read from the datasets themselves, prefer
    /// [`iter_datasets`](Self::iter_datasets): it hands back opened handles for
    /// the cost of this call, where opening each name separately re-walks the
    /// group once per member.
    pub fn datasets(&self) -> Result<Vec<String>, Error> {
        let entries = self.children()?;
        let mut names = Vec::new();
        for entry in &entries {
            let hdr = self.file.parse_header(entry.object_header_address)?;
            if has_message(&hdr, MessageType::DataLayout) {
                names.push(entry.name.clone());
            }
        }
        Ok(names)
    }

    /// Open every dataset in this group, each paired with its name.
    ///
    /// This is the walk to reach for when the members themselves are what you
    /// want — their attributes, shapes or data — rather than a list of names.
    /// [`datasets`](Self::datasets) already parses every child's object header to
    /// tell a dataset from a group, and then keeps only the name, so following it
    /// with a [`dataset`](Self::dataset) call per entry re-walks the group's link
    /// structure and re-parses that same header. This keeps what it read, and
    /// costs one enumeration of the group rather than one per member.
    ///
    /// **This walk is for taking every member, or nearly every one.** Telling a
    /// dataset from a group means parsing its header, so the whole group is
    /// enumerated and every child's header parsed before the iterator is
    /// returned — breaking out early saves nothing, and reaching one known member
    /// this way costs far more than [`dataset`](Self::dataset) does. Only the
    /// handle construction is deferred to each step, and that is not where the
    /// cost is.
    ///
    /// The headers of the members are held for the length of the walk, since each
    /// one is what its handle is built from. That is bounded by the group being
    /// walked rather than by the file, but it is proportional to the group: a
    /// header carries a compact dataset's data and its compact attributes inline,
    /// so a large group of such datasets is a large allocation.
    ///
    /// Each dataset gets the file-wide chunk-cache default; to override the cache
    /// for one, open it by name with
    /// [`dataset_with_options`](Self::dataset_with_options).
    ///
    /// Members arrive in the order the group's link structure yields them — the
    /// same order [`datasets`](Self::datasets) reports, which is not necessarily
    /// sorted. Each handle is a snapshot taken when the iterator was built, so a
    /// [`File::commit`] that runs mid-walk is not reflected in the members still
    /// to come; re-open the group to see past it.
    ///
    /// ```no_run
    /// # use hdf5_pure::File;
    /// # fn main() -> Result<(), hdf5_pure::Error> {
    /// let file = File::open("runs.h5")?;
    /// for (name, dataset) in file.root().iter_datasets()? {
    ///     println!("{name}: {:?}", dataset.shape());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn iter_datasets(
        &self,
    ) -> Result<impl ExactSizeIterator<Item = (String, Dataset)> + use<>, Error> {
        let mut members = Vec::new();
        for entry in self.children()? {
            let hdr = self.file.parse_header(entry.object_header_address)?;
            if has_message(&hdr, MessageType::DataLayout) {
                members.push((entry, hdr));
            }
        }
        let file = Arc::clone(&self.file);
        let parent = self.path.clone();
        let chunk_cache = DatasetAccessProperties::new()
            .resolved_chunk_cache(self.file.access_properties.chunk_cache);
        Ok(members.into_iter().map(move |(entry, header)| {
            let path = child_path_of(parent.as_deref(), &entry.name);
            let dataset = Dataset {
                file: Arc::clone(&file),
                address: entry.object_header_address,
                header,
                chunk_cache: ChunkCache::with_config(chunk_cache),
                chunk_cache_config: chunk_cache,
                path,
            };
            (entry.name, dataset)
        }))
    }

    /// The names of children that are committed (`H5Tcommit`) datatype objects:
    /// an object header carrying a datatype and neither data nor links.
    ///
    /// Such an object is the third kind HDF5 links into a group, and it appears
    /// in neither [`datasets`](Self::datasets) nor [`groups`](Self::groups) — so a
    /// walk that asks only for those two passes over one without noticing. Read
    /// the type itself with [`named_datatype`](Self::named_datatype).
    pub fn named_datatypes(&self) -> Result<Vec<String>, Error> {
        let entries = self.children()?;
        let mut names = Vec::new();
        for entry in &entries {
            let hdr = self.file.parse_header(entry.object_header_address)?;
            if has_message(&hdr, MessageType::Datatype)
                && !has_message(&hdr, MessageType::DataLayout)
                && !is_group(&hdr)
            {
                names.push(entry.name.clone());
            }
        }
        Ok(names)
    }

    /// The datatype a committed (`H5Tcommit`) child object holds.
    ///
    /// `name` must be one [`named_datatypes`](Self::named_datatypes) returned;
    /// any other name fails with [`FormatError::PathNotFound`], and a child that
    /// is not a datatype object fails for want of a datatype message.
    pub fn named_datatype(&self, name: &str) -> Result<Datatype, Error> {
        Ok(self.named_datatype_at(name)?.0)
    }

    /// How many things reference the committed (`H5Tcommit`) datatype `name`:
    /// its hard links, plus every dataset and attribute that names it.
    ///
    /// This is HDF5's own object reference count (`H5Oget_info`'s `rc`), and what
    /// says whether unlinking the name would destroy the type or merely stop it
    /// being reachable by that name. A header that stores no count has exactly
    /// one reference, which is what the format means by omitting the message.
    pub fn named_datatype_references(&self, name: &str) -> Result<u32, Error> {
        let entry = self.child_entry(name)?;
        let hdr = self.file.parse_header(entry.object_header_address)?;
        let Ok(msg) = find_message(&hdr, MessageType::ObjectReferenceCount) else {
            return Ok(1);
        };
        // version(1) + count(4).
        let body = self.file.message_body(msg)?;
        if body.len() < 5 {
            return Err(Error::Format(FormatError::UnexpectedEof {
                expected: 5,
                available: body.len(),
            }));
        }
        Ok(u32::from_le_bytes([body[1], body[2], body[3], body[4]]))
    }

    /// The link entry for a child of this group, by name.
    fn child_entry(&self, name: &str) -> Result<GroupEntry, Error> {
        self.children()?
            .into_iter()
            .find(|e| e.name == name)
            .ok_or_else(|| Error::Format(FormatError::PathNotFound(name.to_string())))
    }

    /// The datatype a committed child object holds, and the address of the object
    /// header holding it.
    ///
    /// The address is the identity every user of the type shares: two datasets
    /// naming the same address name one type, and reproducing that requires
    /// matching them up by address rather than by what the type decodes to.
    pub(crate) fn named_datatype_at(&self, name: &str) -> Result<(Datatype, u64), Error> {
        let entry = self.child_entry(name)?;
        let hdr = self.file.parse_header(entry.object_header_address)?;
        let msg = find_message(&hdr, MessageType::Datatype)?;
        let (dt, _) = Datatype::parse(&self.file.message_body(msg)?)?;
        Ok((dt, entry.object_header_address))
    }

    /// List the names of subgroups in this group.
    ///
    /// To descend into the subgroups themselves, prefer
    /// [`iter_groups`](Self::iter_groups), which hands back opened handles for
    /// the cost of this call.
    pub fn groups(&self) -> Result<Vec<String>, Error> {
        let entries = self.children()?;
        let mut names = Vec::new();
        for entry in &entries {
            let hdr = self.file.parse_header(entry.object_header_address)?;
            if is_group(&hdr) {
                names.push(entry.name.clone());
            }
        }
        Ok(names)
    }

    /// Open every subgroup of this group, each paired with its name.
    ///
    /// The counterpart to [`iter_datasets`](Self::iter_datasets), and the way to
    /// recurse without paying a [`group`](Self::group) lookup per child: that
    /// lookup re-walks this group's link structure, which a walk of the whole
    /// tree would otherwise repeat once per subgroup.
    ///
    /// As with [`iter_datasets`](Self::iter_datasets), the whole group is
    /// enumerated and classified before the iterator is returned, so this is the
    /// walk for taking every subgroup rather than for reaching one — breaking out
    /// early saves nothing. A [`Group`] handle carries no parsed header, so
    /// unlike `iter_datasets` this holds none of them.
    ///
    /// Members arrive in the order the group's link structure yields them — the
    /// same order [`groups`](Self::groups) reports, which is not necessarily
    /// sorted.
    ///
    /// ```no_run
    /// # use hdf5_pure::{Error, Group};
    /// fn total_datasets(group: &Group) -> Result<usize, Error> {
    ///     let mut n = group.datasets()?.len();
    ///     for (_, child) in group.iter_groups()? {
    ///         n += total_datasets(&child)?;
    ///     }
    ///     Ok(n)
    /// }
    /// ```
    pub fn iter_groups(
        &self,
    ) -> Result<impl ExactSizeIterator<Item = (String, Group)> + use<>, Error> {
        let mut members = Vec::new();
        for entry in self.children()? {
            // A `Group` handle carries no parsed header, so the header that
            // classified this child is dropped here rather than held for the
            // length of the walk.
            if is_group(&self.file.parse_header(entry.object_header_address)?) {
                members.push(entry);
            }
        }
        let file = Arc::clone(&self.file);
        let parent = self.path.clone();
        Ok(members.into_iter().map(move |entry| {
            let path = child_path_of(parent.as_deref(), &entry.name);
            let group = Group {
                file: Arc::clone(&file),
                address: entry.object_header_address,
                path,
            };
            (entry.name, group)
        }))
    }

    /// Read all attributes of this group.
    ///
    /// Each value takes the [`AttrValue`] variant that describes its on-disk
    /// encoding, so the variant reflects the charset and dataspace its writer
    /// chose rather than the shape of the data alone: a one-element array stays
    /// an array, and an ASCII string does not arrive as a UTF-8
    /// [`String`](AttrValue::String). Prefer the accessors — [`AttrValue::as_str`],
    /// [`as_strings`](AttrValue::as_strings), [`as_i64`](AttrValue::as_i64) and
    /// the rest — over matching on the variant, unless the encoding is the thing
    /// you care about. **The variant may become more specific in a future
    /// release** as `AttrValue` grows narrower ones (fixed widths, variable-length
    /// strings), and a `_` arm is required regardless because the enum is
    /// `#[non_exhaustive]`.
    ///
    /// An attribute whose datatype has no `AttrValue` representation is omitted
    /// from the map rather than reported as an error. Read
    /// [`attr_datatypes`](Self::attr_datatypes) to see it.
    pub fn attrs(&self) -> Result<HashMap<String, AttrValue>, Error> {
        let hdr = self.file.parse_header(self.address)?;
        self.file.attrs_of(&hdr)
    }

    /// The exact on-disk [`Datatype`] of every attribute on this group, keyed by
    /// name — including compound field offsets, integer widths and enumeration
    /// members.
    ///
    /// This is the type channel to [`attrs`](Self::attrs)'s value channel, the
    /// pair a dataset already has in [`Dataset::datatype`] and its `read_*`
    /// methods. An [`AttrValue`] is a deliberately lossy view of the value, so an
    /// attribute's stored width, string padding and enumeration members are
    /// recoverable only from here. Its *rank* is not: that lives in the
    /// dataspace, which nothing public exposes, so a rank-2 attribute still
    /// reads as a flat `AttrValue` array with no way to recover its shape.
    ///
    /// **Every attribute message is reported, including the ones `attrs` omits**
    /// because no `AttrValue` can carry them, so a name missing from that map can
    /// be told from one the object does not have.
    ///
    /// A **committed** datatype — one created with `H5Tcommit`, what netCDF-4
    /// writes for a user-defined type and what h5py writes for
    /// `f["t"] = np.dtype(...)` — is stored as a reference to the type's own
    /// object header rather than inline, and is resolved to the type it names.
    /// What it does *not* carry is the name: two attributes sharing `/mytype`
    /// report the same [`Datatype`] as one that spells it out inline.
    ///
    /// A boolean attribute is the case that needs both channels. The C library
    /// gives `H5T_NATIVE_HBOOL` — what h5py writes for every `np.bool_` — a
    /// [`Datatype::Enumeration`] of `FALSE` and `TRUE` over an 8-bit base, and
    /// `attrs` decodes it through that base, so the value arrives as `0` or `1`
    /// and only the datatype records that it was a bool.
    pub fn attr_datatypes(&self) -> Result<HashMap<String, Datatype>, Error> {
        Ok(self
            .attr_messages()?
            .into_iter()
            .map(|a| (a.name, a.datatype))
            .collect())
    }

    /// Every attribute message on this group as it is encoded on disk, in the
    /// order the header holds them.
    ///
    /// [`attrs`](Self::attrs) decodes each into an [`AttrValue`], which loses the
    /// encoding; this keeps it. Repack copies from here so an attribute survives
    /// a rewrite unchanged, and falls back to the decoded map only where the
    /// bytes are not position-independent.
    pub(crate) fn attr_messages(&self) -> Result<Vec<crate::attribute::AttributeMessage>, Error> {
        let hdr = self.file.parse_header(self.address)?;
        self.file.attr_messages_of(&hdr)
    }

    /// Get a dataset within this group by name.
    ///
    /// The dataset uses the file-wide chunk-cache default. To override the cache
    /// for this one dataset, use
    /// [`dataset_with_options`](Self::dataset_with_options).
    pub fn dataset(&self, name: &str) -> Result<Dataset, Error> {
        self.dataset_with_options(name, DatasetAccessProperties::new())
    }

    /// Get a dataset within this group by name, applying per-dataset
    /// [`DatasetAccessProperties`] that override file-wide access defaults (HDF5's
    /// `dapl`; see `H5Pset_chunk_cache`).
    pub fn dataset_with_options(
        &self,
        name: &str,
        properties: DatasetAccessProperties,
    ) -> Result<Dataset, Error> {
        let address = self
            .child_address(name)?
            .ok_or_else(|| Error::Format(FormatError::PathNotFound(name.to_string())))?;
        let hdr = self.file.parse_header(address)?;
        if !has_message(&hdr, MessageType::DataLayout) {
            return Err(Error::NotADataset(name.to_string()));
        }
        let chunk_cache = properties.resolved_chunk_cache(self.file.access_properties.chunk_cache);
        Ok(Dataset {
            file: self.file.clone(),
            address,
            header: hdr,
            chunk_cache: ChunkCache::with_config(chunk_cache),
            chunk_cache_config: chunk_cache,
            path: self.child_path(name),
        })
    }

    /// Get a subgroup within this group by name.
    pub fn group(&self, name: &str) -> Result<Group, Error> {
        let address = self
            .child_address(name)?
            .ok_or_else(|| Error::Format(FormatError::PathNotFound(name.to_string())))?;
        Ok(Group {
            file: self.file.clone(),
            address,
            path: self.child_path(name),
        })
    }

    /// The object-header address of this group's child named `name`.
    ///
    /// The by-name form of [`children`](Self::children): it reads the group's
    /// links without building one entry per child, which is what makes opening
    /// each member of a large group in turn cost the group once rather than once
    /// per member (issue #228).
    fn child_address(&self, name: &str) -> Result<Option<u64>, Error> {
        self.file.group_child(self.address, name)
    }

    /// The root-relative path of a child named `name`, or `None` if this group
    /// itself has no resolvable path (reached by object reference).
    fn child_path(&self, name: &str) -> Option<String> {
        child_path_of(self.path.as_deref(), name)
    }

    /// Create an empty subgroup `name` within this group, staged until
    /// [`File::commit`].
    ///
    /// To give the new group attributes or children in the same commit, use
    /// [`create_group_with`](Self::create_group_with).
    ///
    /// Requires a read-write file ([`File::open_rw`]), else
    /// [`Error::ReadOnly`](crate::Error::ReadOnly).
    ///
    /// ```no_run
    /// # use hdf5_pure::File;
    /// # fn main() -> Result<(), hdf5_pure::Error> {
    /// let file = File::open_rw("runs.h5")?;
    /// file.root().create_group("run2")?;
    /// file.commit()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn create_group(&self, name: &str) -> Result<(), Error> {
        self.create_group_with(name, |_| {})
    }

    /// Create a subgroup `name` within this group, configuring it through
    /// `build` (attributes, nested groups and datasets), staged until
    /// [`File::commit`].
    ///
    /// The closure exists because [`set_attr`](Self::set_attr) needs a group
    /// that already *resolves*, so it cannot reach a group that is itself still
    /// staged; this can, and the creation and its attributes land in one commit.
    /// For a plain empty group use [`create_group`](Self::create_group).
    ///
    /// The closure records into a buffer rather than into the file itself, and
    /// nothing it stages resolves until [`File::commit`], so reading the same
    /// [`File`] from inside it sees the file as it was before this call.
    ///
    /// Requires a read-write file ([`File::open_rw`]), else
    /// [`Error::ReadOnly`](crate::Error::ReadOnly).
    ///
    /// ```no_run
    /// # use hdf5_pure::{AttrValue, File};
    /// # fn main() -> Result<(), hdf5_pure::Error> {
    /// let file = File::open_rw("runs.h5")?;
    /// file.root().create_group_with("run2", |g| {
    ///     g.set_attr("count", AttrValue::I64(7));
    ///     g.set_attr("label", AttrValue::String("second".into()));
    /// })?;
    /// file.commit()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn create_group_with(
        &self,
        name: &str,
        build: impl FnOnce(&mut StagedGroup<'_>),
    ) -> Result<(), Error> {
        let child = self.child_edit_path(name)?;
        let mut ops = vec![StagedOp::CreateGroup(child.clone())];
        build(&mut StagedGroup {
            ops: &mut ops,
            path: child,
        });
        self.apply_staged(ops)
    }

    /// Create a dataset `name` within this group, configuring it through `build`
    /// (shape, data, chunks, filters, …), staged until [`File::commit`].
    ///
    /// As with [`create_group_with`](Self::create_group_with), the closure
    /// configures a builder rather than the file, so it may read the same
    /// [`File`] — it will see the file as it was before this call.
    ///
    /// Requires a read-write file ([`File::open_rw`]), else
    /// [`Error::ReadOnly`](crate::Error::ReadOnly).
    pub fn create_dataset(
        &self,
        name: &str,
        build: impl FnOnce(&mut DatasetBuilder),
    ) -> Result<(), Error> {
        let child = self.child_edit_path(name)?;
        let mut builder = DatasetBuilder::new(name);
        build(&mut builder);
        self.apply_staged(vec![StagedOp::CreateDataset {
            path: child,
            builder: Box::new(builder),
        }])
    }

    /// Delete the object named `name` from this group, staged until
    /// [`File::commit`]. See [`create_group`](Self::create_group) for the
    /// file-mode rules.
    pub fn delete(&self, name: &str) -> Result<(), Error> {
        self.with_child_session(name, |session, child| session.delete(child))
    }

    /// Add or update a compact attribute on this group, staged until
    /// [`File::commit`]. Use [`remove_attr`](Self::remove_attr) to remove one.
    /// The [`root`](File::root) group's attributes are edited the same way.
    ///
    /// Requires a read-write file ([`File::open_rw`]), else
    /// [`Error::ReadOnly`](crate::Error::ReadOnly). An attribute set too large
    /// for compact storage, or a group using dense (fractal-heap) attribute
    /// storage, is refused on [`File::commit`].
    pub fn set_attr(&self, name: &str, value: AttrValue) -> Result<(), Error> {
        self.with_own_session(|session, path| session.set_group_attr(path, name, value))
    }

    /// Remove a compact attribute from this group, staged until [`File::commit`].
    /// See [`set_attr`](Self::set_attr) for the file-mode rules.
    pub fn remove_attr(&self, name: &str) -> Result<(), Error> {
        self.with_own_session(|session, path| session.remove_group_attr(path, name))
    }

    /// Run `f` with the writable session and the root-relative path of child
    /// `name`. Returns [`Error::ReadOnly`](crate::Error::ReadOnly) if the file is
    /// read-only or this group has no resolvable path.
    fn with_child_session<R>(
        &self,
        name: &str,
        f: impl FnOnce(&mut WriteEngine, &str) -> Result<R, Error>,
    ) -> Result<R, Error> {
        let Backend::Edit(m) = &self.file.backend else {
            return Err(Error::ReadOnly);
        };
        self.file.check_mutable(true)?;
        let child = self.child_path(name).ok_or(Error::ReadOnly)?;
        let mut session = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        f(&mut session, &child)
    }

    /// Validate that this group can stage an edit to child `name` and return the
    /// child's root-relative path, *without* taking the session lock.
    ///
    /// Paired with [`apply_staged`](Self::apply_staged): the checks run first so
    /// a read-only or sealed file is reported before any user closure runs, the
    /// closure then runs unlocked, and the lock is taken only to record what it
    /// built (issue #200).
    fn child_edit_path(&self, name: &str) -> Result<String, Error> {
        self.file.check_staged_writable()?;
        self.child_path(name).ok_or(Error::ReadOnly)
    }

    /// Record already-built edits on the writable session, holding the lock only
    /// for the duration of the replay.
    ///
    /// The file is re-checked here because the closure that produced `ops` ran
    /// unlocked and could have closed the file in the meantime; staging into a
    /// sealed file would otherwise be silently accepted and then dropped.
    fn apply_staged(&self, ops: Vec<StagedOp>) -> Result<(), Error> {
        self.file.check_staged_writable()?;
        let Backend::Edit(m) = &self.file.backend else {
            // `check_staged_writable` accepts only a mirror backend, and a
            // file's backend is fixed for its lifetime.
            return Err(Error::ReadOnly);
        };
        let mut session = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        for op in ops {
            op.apply(&mut session)?;
        }
        Ok(())
    }

    /// Run `f` with the writable session and this group's *own* root-relative
    /// path (for attribute edits, which act on the group itself rather than a
    /// child). Returns [`Error::ReadOnly`](crate::Error::ReadOnly) if the file is
    /// read-only or this group has no resolvable path, and
    /// [`Error::FileClosed`](crate::Error::FileClosed) once the file is sealed.
    fn with_own_session<R>(
        &self,
        f: impl FnOnce(&mut WriteEngine, &str) -> Result<R, Error>,
    ) -> Result<R, Error> {
        let Backend::Edit(m) = &self.file.backend else {
            return Err(Error::ReadOnly);
        };
        self.file.check_mutable(true)?;
        let path = self.path.clone().ok_or(Error::ReadOnly)?;
        let mut session = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        f(&mut session, &path)
    }

    fn children(&self) -> Result<Vec<GroupEntry>, Error> {
        let hdr = self.file.parse_header(self.address)?;
        self.file.group_children(&hdr)
    }
}

// ---------------------------------------------------------------------------
// Dataset handle
// ---------------------------------------------------------------------------

/// An owned handle to an HDF5 dataset.
pub struct Dataset {
    file: Arc<FileInner>,
    /// Address of this dataset's object header (base-adjusted, file-absolute).
    /// Used to resolve object references that point at this dataset.
    address: u64,
    header: ObjectHeader,
    // Held per-dataset: the chunk index is keyed only by chunk coordinate, so
    // a file-level cache would alias chunk addresses across datasets.
    chunk_cache: ChunkCache,
    // The effective chunk-cache config for this dataset: the file-wide default
    // or a per-dataset DAPL override. Reported by `chunk_cache_config`.
    chunk_cache_config: ChunkCacheConfig,
    /// Root-relative path of this dataset, used to address it for write
    /// operations on a read-write file. `None` for a dataset reached by object
    /// reference ([`Dataset::dereference`]), which has no resolvable path.
    path: Option<String>,
}

impl std::fmt::Debug for Dataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dataset")
            .field("messages", &self.header.messages.len())
            .finish()
    }
}

/// How many stored bytes a typed whole-dataset read holds beside its output,
/// before rounding the window up to whole chunk bands.
///
/// The decoded values are the caller's and there is no bound to put on them;
/// what this bounds is the *stored* copy standing next to them, which used to be
/// the whole dataset over again (issue #289). A mebibyte is small against any
/// dataset large enough for that to matter, and large enough that a sweep costs
/// reads in the tens rather than the thousands. A dataset that fits inside one
/// window is read whole, exactly as before.
const TYPED_READ_WINDOW_BYTES: u64 = 1 << 20;

/// How many values of the requested type a whole-dataset read produces per
/// stored element, which is what its output buffer is reserved at.
#[derive(Clone, Copy)]
enum OutputSize {
    /// One value per stored element — every numeric decoder.
    PerElement,
    /// One value per stored *byte* — [`Dataset::read_i8`], which reinterprets
    /// bytes rather than decoding elements, and so yields one per byte of a
    /// dataset whose elements are wider than one.
    PerByte,
}

/// How many leading-dimension rows a typed whole-dataset read decodes at a time.
///
/// [`TYPED_READ_WINDOW_BYTES`] of stored bytes, rounded *down* to whole chunk
/// bands so that no chunk is ever decoded for two windows, and never fewer than
/// one row — or one band, when a single band is already over budget, since a
/// window narrower than that would decode the same chunks again.
///
/// A dataset whose rows have no bytes (a zero inner dimension) has no elements
/// to read at all: it answers `NonZeroU64::MAX`, which the caller reads as "one
/// window covers it".
///
/// The answer is a `NonZeroU64` because the sweep advances by it: a window of no
/// rows would leave that loop running forever rather than returning something
/// wrong, and a test cannot report the difference.
fn typed_window_rows(
    dl: &DataLayout,
    ds: &Dataspace,
    elem_size: NonZeroUsize,
) -> Result<NonZeroU64, FormatError> {
    let mut row_bytes = elem_size.get() as u64;
    for &d in ds.dimensions.iter().skip(1) {
        row_bytes = row_bytes
            .checked_mul(d)
            .ok_or(FormatError::OffsetOverflow {
                offset: row_bytes,
                length: d,
            })?;
    }
    if row_bytes == 0 {
        return Ok(NonZeroU64::MAX);
    }

    let mut rows = (TYPED_READ_WINDOW_BYTES / row_bytes).max(1);
    if let DataLayout::Chunked {
        chunk_dimensions, ..
    } = dl
    {
        // A chunked layout message carries rank + 1 dimensions, the last being
        // the element size, so the first is the leading dimension's chunk extent
        // for every layout version this crate parses.
        if let Some(band) = chunk_dimensions
            .first()
            .map(|&d| u64::from(d))
            .filter(|&d| d > 0)
        {
            rows = if rows < band {
                band
            } else {
                rows - rows % band
            };
        }
    }
    // At least one row always, and at least one whole band when a band applied:
    // that arm runs only when `rows >= band`, so the remainder it subtracts
    // leaves a band standing.
    Ok(NonZeroU64::new(rows).unwrap_or(NonZeroU64::MIN))
}

impl Dataset {
    /// Address of this dataset's object header (base-adjusted, file-absolute).
    /// Used to resolve object references that point at this dataset.
    pub(crate) fn header_address(&self) -> u64 {
        self.address
    }

    /// Append `data` to this dataset in place, growing it along its first
    /// (unlimited) dimension, and refresh this handle so subsequent reads observe
    /// the new length.
    ///
    /// The file must have been opened for writing with [`File::open_rw`];
    /// a read-only file returns
    /// [`Error::ReadOnly`](crate::Error::ReadOnly). The target must be a chunked,
    /// rank-1, unlimited, Extensible-Array-indexed dataset, and a filtered one
    /// must already be a whole number of chunks long — growing a trailing chunk
    /// a reader can see is not power-loss atomic, where an unfiltered dataset
    /// may be any length. The *appended* length is unconstrained either way;
    /// anything else returns
    /// [`Error::AppendInPlaceUnsupported`](crate::Error::AppendInPlaceUnsupported).
    /// The append is immediate and crash-atomic (no `commit` needed) — under the
    /// default [`SyncPolicy::Always`]. Under
    /// [`SyncPolicy::OnClose`](crate::SyncPolicy::OnClose) the same writes are made
    /// in the same order without the `fsync` barriers between them, so the
    /// append is still immediate and still crash-atomic against *this process*
    /// failing, but ordering it against power loss is the caller's, through
    /// [`File::sync`].
    ///
    /// A handle reached by object reference ([`dereference`](Self::dereference))
    /// has no resolvable path, so it names its dataset by the object-header
    /// address it was reached through and can append like any other — until the
    /// session stages or commits an edit. A commit can move that header, and the
    /// bytes it vacates still parse as the dataset they were, so an append
    /// against the old address would land in a header nothing points at. Rather
    /// than do that silently, such an append is refused once edits are staged or
    /// a commit has run; re-open the dataset by path to keep appending. A
    /// path-named handle is unaffected, because the path is resolved afresh every
    /// time.
    pub fn append<T: H5Element>(&mut self, data: &[T]) -> Result<(), Error> {
        let g = self.append_geometry()?;
        self.append_batches(g, data.len() as u64, |b, r| {
            b.append(&data[r]);
        })
    }

    /// Append raw little-endian element bytes to this dataset in place. Prefer
    /// [`append`](Self::append) when the element type is known; see it for the
    /// file-mode and eligibility rules.
    pub fn append_raw(&mut self, bytes: &[u8]) -> Result<(), Error> {
        let g = self.append_geometry()?;
        let es = g.element_size;
        // Whole-element length is checked before any batch applies, so the
        // refusal is atomic (the per-batch validation would only reject the
        // final, short batch after earlier ones had durably committed).
        if bytes.len() % es != 0 {
            return Err(Error::AppendInPlaceUnsupported(
                "appended byte length is not a whole number of elements",
            ));
        }
        let total = (bytes.len() / es) as u64;
        self.append_batches(g, total, |b, r| {
            b.append_raw(&bytes[r.start * es.get()..r.end * es.get()]);
        })
    }

    /// How an append names this dataset to the session: by path when the handle
    /// has one, so the session can check the target against its own staged
    /// edits, and otherwise by the object-header address the handle was reached
    /// through — which is what lets a handle obtained by object reference append
    /// at all.
    fn append_target(&self) -> AppendTarget<'_> {
        match &self.path {
            Some(path) => AppendTarget::Path(path),
            None => AppendTarget::Header(self.address),
        }
    }

    /// A [`BufferedAppender`] over this dataset: appended elements are held in
    /// memory and written a whole chunk at a time, so a caller appending less
    /// than a chunk per call writes to the file once per chunk instead of once
    /// per call — and can append to a *filtered* dataset by any length, which
    /// [`append`](Self::append) refuses.
    ///
    /// Every eligibility rule [`append`](Self::append) applies is applied here,
    /// so an ineligible dataset is reported now rather than on the first write.
    /// Buffered elements are not in the file until the appender flushes; see
    /// [`BufferedAppender`] for the full bargain.
    pub fn buffered_appender(&mut self) -> Result<BufferedAppender<'_>, Error> {
        BufferedAppender::new(self)
    }

    /// Register a live `BufferedAppender` on this dataset with the session, so a
    /// staged edit that would stop it from flushing is refused at the call that
    /// creates the conflict rather than in the appender's `Drop`.
    pub(crate) fn claim_for_appender(&self, needs_commit: bool) -> Result<u64, Error> {
        let Backend::Edit(m) = &self.file.backend else {
            return Err(Error::ReadOnly);
        };
        m.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .claim_for_appender(self.path.as_deref(), needs_commit)
    }

    /// Record whether the live appender still owes a staged realignment.
    pub(crate) fn set_appender_needs_commit(&self, token: u64, needs_commit: bool) {
        if let Backend::Edit(m) = &self.file.backend {
            m.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .set_appender_needs_commit(token, needs_commit);
        }
    }

    /// Release the claim taken by [`claim_for_appender`](Self::claim_for_appender).
    pub(crate) fn release_appender_claim(&self, token: u64) {
        if let Backend::Edit(m) = &self.file.backend {
            m.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .release_appender_claim(token);
        }
    }

    /// Whether this dataset's session is the SWMR writer, whose append rules are
    /// a strict subset of the ordinary ones. `false` for a read-only file, which
    /// has no session to ask.
    pub(crate) fn session_is_swmr(&self) -> bool {
        match &self.file.backend {
            Backend::Edit(m) => m
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .is_swmr(),
            _ => false,
        }
    }

    /// Immediate in-place append of an already-gathered builder, used by
    /// [`BufferedAppender`], whose bytes are materialized in its buffer before
    /// it decides how many of them to write. `append_batches` exists for the
    /// opposite case — a caller whose bytes are cheaper to build per batch — so
    /// this hands the engine one builder and lets it batch the plan.
    pub(crate) fn append_prebuilt(&mut self, b: &AppendBuilder) -> Result<(), Error> {
        let Backend::Edit(m) = &self.file.backend else {
            return Err(Error::ReadOnly);
        };
        self.file.check_mutable(false)?;
        {
            let mut engine = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            engine.append_inplace_gathered(self.append_target(), b, 4)?;
        }
        self.header = self.file.parse_header(self.address)?;
        // Same staleness rule as `append_batches`: the append extended the chunk
        // index this handle may have cached.
        self.chunk_cache.clear();
        Ok(())
    }

    /// Stage an append and commit it in the same lock, used by
    /// [`BufferedAppender`] for the one case in-place growth cannot serve: a
    /// filtered dataset whose trailing chunk is partial. Refused while the
    /// session holds unrelated staged edits, which this commit would otherwise
    /// publish as a side effect of an append.
    pub(crate) fn append_staged_committed(&mut self, b: AppendBuilder) -> Result<(), Error> {
        // A path-less handle (reached by object reference) cannot be named to the
        // staging surface at all. Say that, rather than the `ReadOnly` that
        // `check_staged_edit` reports for the same condition — the file is not
        // read-only, and telling the caller to reopen it read-write is advice
        // they have already taken.
        let path = self.path.clone().ok_or(Error::AppendInPlaceUnsupported(
            "this dataset handle was reached by object reference and has no path, so the \
                 staged rewrite that grows a filtered dataset's partial trailing chunk cannot \
                 name it; re-open the dataset by path",
        ))?;
        self.check_staged_edit()?;
        let Backend::Edit(m) = &self.file.backend else {
            return Err(Error::ReadOnly);
        };
        self.file.check_mutable(true)?;
        {
            let mut engine = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if engine.has_staged_edits() {
                // The same variant the engine's own two staged-conflict refusals
                // use (`append_prepare`), so a caller catching that one to fall
                // back on `append_staged` catches this one too.
                return Err(Error::AppendInPlaceUnsupported(
                    "a buffered append onto a filtered dataset with a partial trailing chunk \
                     must commit, and this session holds other staged edits; commit the staged \
                     edits before appending, or use Dataset::append_staged",
                ));
            }
            // Suspend this appender's own claim: it is the reason no other
            // edit may be staged, and it must not refuse its own realignment.
            engine.within_appender_commit(|e| {
                e.stage_dataset_append(&path, b)?;
                e.commit()
            })?;
        }
        // A commit rewrites and *relocates* object headers, so unlike every other
        // `with_session_mut` caller this handle's cached address is stale as well
        // as its header. Re-resolve by path before touching either; parsing the
        // vacated address would read whatever the commit left there.
        self.address = self.file.resolve_path(&path)?;
        self.header = self.file.parse_header(self.address)?;
        self.chunk_cache.clear();
        Ok(())
    }

    /// Fetch (locating on first use) this dataset's append geometry from the
    /// write session, which also applies every refusal that does not depend on
    /// the bytes being appended.
    pub(crate) fn append_geometry(&self) -> Result<AppendGeometry, Error> {
        let Backend::Edit(m) = &self.file.backend else {
            return Err(Error::ReadOnly);
        };
        self.file.check_mutable(false)?;
        let mut engine = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        engine.append_geometry(self.append_target())
    }

    /// Immediate in-place append, driven batch by batch. The call is split into
    /// aligned batches — the trailing partial chunk is filled first, then
    /// whole-chunk batches under the session's byte budget — and `fill` builds
    /// each batch's bytes on demand, so a bounded session's peak memory holds
    /// one batch rather than the whole call. A session that keeps the whole file
    /// resident reports one unbounded batch, so the call stays a single
    /// crash-atomic apply there.
    ///
    /// Every predictable refusal (wrong datatype, ineligible dataset,
    /// non-chunk-aligned filtered append) is raised before the first batch is
    /// applied. The cached header and chunk cache are then refreshed so later
    /// reads on this handle observe the new length.
    fn append_batches(
        &mut self,
        g: AppendGeometry,
        total_elems: u64,
        fill: impl Fn(&mut AppendBuilder, std::ops::Range<usize>),
    ) -> Result<(), Error> {
        let Backend::Edit(m) = &self.file.backend else {
            return Err(Error::ReadOnly);
        };
        // Atomic refusal before any batch: a filtered append must start
        // chunk-aligned (the engine re-checks per batch as a backstop). The
        // appended length is unconstrained — an unaligned remainder is always the
        // last batch, and its chunk is a fresh element no reader can see yet.
        if g.filtered && g.current_dim % g.chunk_elems != 0 {
            return Err(Error::AppendInPlaceUnsupported(
                "a filtered dataset whose length is not a whole multiple of the chunk length \
                 cannot be appended in place: growing its trailing partial chunk would repoint \
                 an index element a reader can already see. Use Dataset::append_staged, or a \
                 BufferedAppender, which keeps the on-disk length chunk-aligned",
            ));
        }
        let mut dim = g.current_dim;
        let mut done = 0u64;
        loop {
            // An empty append still runs one (empty) engine call, so datatype
            // validation happens whether or not there are elements.
            self.file.check_mutable(false)?;
            let to_boundary = (g.chunk_elems - dim % g.chunk_elems) % g.chunk_elems;
            let take = (total_elems - done).min(to_boundary.saturating_add(g.full_batch_elems));
            let mut b = AppendBuilder::new();
            fill(&mut b, done.to_usize()?..(done + take).to_usize()?);
            {
                let mut engine = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                engine.append_inplace_gathered(self.append_target(), &b, 4)?;
            }
            dim += take;
            done += take;
            if done >= total_elems {
                break;
            }
        }
        self.header = self.file.parse_header(self.address)?;
        // Same staleness rule as `with_session_mut`: the append repointed or
        // extended the chunk index this handle may have cached.
        self.chunk_cache.clear();
        Ok(())
    }

    /// Overwrite this dataset's values, staged until [`File::commit`]. The new
    /// data must match the dataset's existing shape and datatype.
    ///
    /// The file must have been opened with [`File::open_rw`], else
    /// [`Error::ReadOnly`](crate::Error::ReadOnly). Unlike [`append`](Self::append)
    /// (immediate), this is a staged edit applied on [`File::commit`].
    pub fn write<T: H5Element>(&mut self, data: &[T]) -> Result<(), Error> {
        // Build off the lock — `H5Element` is user-implementable, so
        // `write_into` is potentially user code (see `write_staged`).
        self.check_staged_edit()?;
        let mut builder = DatasetBuilder::new("");
        T::write_into(&mut builder, data);
        self.with_session_mut(true, |session, path| {
            session.stage_dataset_write(path, builder)
        })
    }

    /// Overwrite this dataset's values through its full [`DatasetBuilder`],
    /// staged until [`File::commit`] — the builder-level counterpart of
    /// [`write`](Self::write), for element kinds that are not [`H5Element`]
    /// (variable-length strings, raw bytes with an explicit datatype).
    ///
    /// The replacement must match the on-disk datatype and shape exactly; a
    /// reshape or retype is refused on [`File::commit`].
    ///
    /// The file must have been opened with [`File::open_rw`], else
    /// [`Error::ReadOnly`](crate::Error::ReadOnly).
    ///
    /// ```no_run
    /// # use hdf5_pure::File;
    /// # fn main() -> Result<(), hdf5_pure::Error> {
    /// let file = File::open_rw("labels.h5")?;
    /// let mut ds = file.dataset("names")?;
    /// ds.write_staged(|b| {
    ///     b.with_vlen_strings(&["ada", "grace", "katherine"]);
    /// })?;
    /// file.commit()?;
    /// # Ok(())
    /// # }
    /// ```
    /// The closure configures a standalone builder, not the file, so it may read
    /// the same [`File`]; nothing it stages resolves until [`File::commit`].
    pub fn write_staged(&mut self, build: impl FnOnce(&mut DatasetBuilder)) -> Result<(), Error> {
        // Report a read-only, sealed, or unaddressable dataset before running the
        // closure, then run it with no lock held; `stage_dataset_write` names the
        // builder from the dataset's path (issue #200).
        self.check_staged_edit()?;
        let mut builder = DatasetBuilder::new("");
        build(&mut builder);
        self.with_session_mut(true, |session, path| {
            session.stage_dataset_write(path, builder)
        })
    }

    /// Stage an append to this dataset applied on [`File::commit`] — the staged,
    /// index-rebuilding counterpart of the immediate [`append`](Self::append).
    ///
    /// Unlike [`append`](Self::append) (immediate, amortized `O(1)`,
    /// Extensible-Array only, and refused on a filtered dataset whose length is
    /// not already a whole number of chunks), this rebuilds the chunk index on
    /// commit and so also grows **filtered** datasets from any length (the
    /// trailing partial chunk is rewritten) and datasets whose
    /// Extensible-Array index is not yet allocated. Configure the appended
    /// elements through `build` on the [`AppendBuilder`]; repeated calls within
    /// the builder concatenate in order. The dataset must be chunked, unlimited
    /// along axis 0, Extensible-Array indexed, rank 1, use a re-encodable filter
    /// pipeline, and have a single hard link, otherwise
    /// [`Error::AppendUnsupported`](crate::Error::AppendUnsupported) is returned
    /// on [`File::commit`].
    ///
    /// The file must have been opened with [`File::open_rw`], else
    /// [`Error::ReadOnly`](crate::Error::ReadOnly).
    /// The closure configures a standalone builder, not the file, so it may read
    /// the same [`File`]; nothing it stages resolves until [`File::commit`].
    pub fn append_staged(&mut self, build: impl FnOnce(&mut AppendBuilder)) -> Result<(), Error> {
        self.check_staged_edit()?;
        let mut builder = AppendBuilder::new();
        build(&mut builder);
        self.with_session_mut(true, |session, path| {
            session.stage_dataset_append(path, builder)
        })
    }

    /// Add or update a compact attribute on this dataset, staged until
    /// [`File::commit`]. Use [`remove_attr`](Self::remove_attr) to remove one.
    ///
    /// The file must have been opened with [`File::open_rw`], else
    /// [`Error::ReadOnly`](crate::Error::ReadOnly).
    pub fn set_attr(&mut self, name: &str, value: AttrValue) -> Result<(), Error> {
        self.with_session_mut(true, |session, path| {
            session.set_dataset_attr(path, name, value)
        })
    }

    /// Remove a compact attribute from this dataset, staged until
    /// [`File::commit`]. See [`set_attr`](Self::set_attr) for the file-mode rules.
    pub fn remove_attr(&mut self, name: &str) -> Result<(), Error> {
        self.with_session_mut(true, |session, path| {
            session.remove_dataset_attr(path, name)
        })
    }

    /// Gate a staged edit on this dataset *without* taking the session lock:
    /// the file must accept staged edits, and this handle must have a resolvable
    /// path.
    ///
    /// The path check belongs here rather than only in
    /// [`with_session_mut`](Self::with_session_mut) so that every reason to
    /// refuse is reported *before* a user closure runs, not after. A handle
    /// reached by object reference ([`dereference`](Self::dereference)) has no
    /// path, and would otherwise have its closure run and its result discarded.
    fn check_staged_edit(&self) -> Result<(), Error> {
        self.file.check_staged_writable()?;
        if self.path.is_none() {
            return Err(Error::ReadOnly);
        }
        Ok(())
    }

    /// Run `f` with the writable session and this dataset's path, then refresh
    /// the cached header so a later read on this handle reflects any immediate
    /// change (e.g. an append's new dimension). Returns
    /// [`Error::ReadOnly`](crate::Error::ReadOnly) if the file is read-only or the
    /// handle has no resolvable path (reached by object reference).
    fn with_session_mut<R>(
        &mut self,
        staged: bool,
        f: impl FnOnce(&mut WriteEngine, &str) -> Result<R, Error>,
    ) -> Result<R, Error> {
        let Backend::Edit(m) = &self.file.backend else {
            return Err(Error::ReadOnly);
        };
        self.file.check_mutable(staged)?;
        let path = self.path.clone().ok_or(Error::ReadOnly)?;
        let out = {
            let mut session = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            f(&mut session, &path)?
        };
        self.header = self.file.parse_header(self.address)?;
        // An append relocates the trailing chunk and grows the chunk index, so
        // this handle's cached index and retained chunks are stale; drop them
        // so the next read re-walks the live index.
        self.chunk_cache.clear();
        Ok(out)
    }

    /// The effective raw chunk-cache configuration for this dataset.
    ///
    /// This reflects the per-dataset [`DatasetAccessProperties`] override when one
    /// was supplied to [`File::dataset_with_options`] /
    /// [`Group::dataset_with_options`], otherwise the file-wide default. It is
    /// the read-side analogue of HDF5's `H5Pget_chunk_cache`.
    pub const fn chunk_cache_config(&self) -> ChunkCacheConfig {
        self.chunk_cache_config
    }

    /// A point-in-time snapshot of this dataset handle's chunk-cache occupancy.
    ///
    /// Lets callers confirm a chunk-cache configuration (set with
    /// [`FileAccessProperties::with_chunk_cache`]) is taking effect: after a
    /// chunked read, an enabled cache reports a loaded index and retained
    /// chunks; a disabled one (or one over its budget) reports fewer or none.
    /// The cache is per-handle, so a freshly opened [`Dataset`] reports an empty
    /// snapshot until its first read.
    pub fn chunk_cache_stats(&self) -> ChunkCacheStats {
        self.chunk_cache.stats()
    }

    /// Returns the shape (dimensions) of the dataset.
    pub fn shape(&self) -> Result<Vec<u64>, Error> {
        let ds = self.dataspace()?;
        Ok(ds.dimensions.clone())
    }

    /// The dataset's maximum dimensions, when it is extensible. An unlimited
    /// dimension is reported as `u64::MAX`. Returns `Ok(None)` for a fixed-shape
    /// dataset (no maximum-dimensions record, or one equal to the current shape).
    ///
    /// Together with [`is_chunked`](Self::is_chunked) and
    /// [`chunk_shape`](Self::chunk_shape), this lets a caller check up front
    /// whether a dataset is eligible for
    /// [`Dataset::append_staged`](crate::Dataset::append_staged)
    /// (which requires a chunked dataset whose first maximum dimension is
    /// `u64::MAX`) instead of relying on the append's refusal error.
    pub fn maxshape(&self) -> Result<Option<Vec<u64>>, Error> {
        let ds = self.dataspace()?;
        match &ds.max_dimensions {
            Some(md) if *md != ds.dimensions => Ok(Some(md.clone())),
            _ => Ok(None),
        }
    }

    /// Whether the dataset uses chunked storage (as opposed to contiguous or
    /// compact). Filtered datasets are always chunked. Returns `false` for a
    /// dataset with no data-layout message or a non-chunked layout.
    pub fn is_chunked(&self) -> bool {
        matches!(self.data_layout(), Ok(DataLayout::Chunked { .. }))
    }

    /// The dataset's chunk dimensions (one per dataset rank), or `Ok(None)` when
    /// the dataset is not chunked. The element-size dimension the on-disk layout
    /// appends is stripped, so the result lines up with
    /// [`shape`](Self::shape) / [`maxshape`](Self::maxshape).
    pub fn chunk_shape(&self) -> Result<Option<Vec<u64>>, Error> {
        let DataLayout::Chunked {
            chunk_dimensions, ..
        } = self.data_layout()?
        else {
            return Ok(None);
        };
        let rank = self.dataspace()?.dimensions.len();
        if chunk_dimensions.len() <= rank {
            return Ok(None);
        }
        Ok(Some(
            chunk_dimensions[..rank]
                .iter()
                .map(|&c| u64::from(c))
                .collect(),
        ))
    }

    /// The HDF5 filter IDs applied to this dataset's chunks, in pipeline
    /// (application) order, or an empty vector when the dataset is unfiltered.
    /// The IDs are the registered HDF5 filter numbers — e.g. 1 = deflate,
    /// 2 = shuffle, 3 = fletcher32, 6 = scale-offset — so a caller can inspect
    /// the pipeline without decoding a chunk.
    pub fn filters(&self) -> Vec<u16> {
        self.filter_pipeline_parsed()
            .map(|p| p.filters.iter().map(|f| f.filter_id).collect())
            .unwrap_or_default()
    }

    /// How and where this dataset's raw data is stored: compact, contiguous,
    /// chunked, or virtual.
    ///
    /// The structured companion to [`is_chunked`](Self::is_chunked) and
    /// [`chunk_shape`](Self::chunk_shape), which it subsumes: one call that
    /// classifies the layout and, for a [`Layout::Contiguous`] dataset, gives the
    /// absolute address and byte size to seek to, or for a [`Layout::Chunked`]
    /// dataset the chunk shape and [`ChunkIndex`] kind. This parses only the
    /// data-layout message; it never walks the chunk index or reads any data —
    /// use [`chunks`](Self::chunks) for per-chunk locations. The curated analogue
    /// of `H5Pget_layout`.
    ///
    /// Returns `Err` if the dataset has no data-layout message, if it cannot be
    /// parsed, or if a chunked dataset uses an index kind this crate does not
    /// recognize.
    pub fn layout(&self) -> Result<Layout, Error> {
        Ok(match self.data_layout()? {
            DataLayout::Compact { data } => Layout::Compact {
                size: data.len() as u64,
            },
            DataLayout::Contiguous { address, size } => Layout::Contiguous {
                address: self.absolute_address(address)?,
                size,
            },
            DataLayout::Chunked {
                version,
                chunk_index_type,
                ..
            } => Layout::Chunked {
                // Reuse `chunk_shape` so the two accessors can never disagree on
                // how the element-size dimension is stripped.
                chunk_shape: self.chunk_shape()?.unwrap_or_default(),
                index: ChunkIndex::from_layout(version, chunk_index_type)?,
            },
            DataLayout::Virtual { .. } => Layout::Virtual,
        })
    }

    /// The [`ChunkIndex`] kind of this chunked dataset, or `Ok(None)` when the
    /// dataset is not chunked.
    ///
    /// A convenience shortcut for the `index` of [`Layout::Chunked`], for the
    /// common up-front append-eligibility check
    /// ([`ChunkIndex::supports_inplace_append`]). Complements
    /// [`maxshape`](Self::maxshape) and [`chunk_shape`](Self::chunk_shape).
    ///
    /// Returns `Err` if the data-layout message is missing or cannot be parsed,
    /// or if a chunked dataset uses an index kind this crate does not recognize.
    pub fn chunk_index(&self) -> Result<Option<ChunkIndex>, Error> {
        match self.data_layout()? {
            DataLayout::Chunked {
                version,
                chunk_index_type,
                ..
            } => Ok(Some(ChunkIndex::from_layout(version, chunk_index_type)?)),
            _ => Ok(None),
        }
    }

    /// Enumerate every allocated chunk of this chunked dataset — one [`Chunk`]
    /// (logical offset, absolute file address, on-disk stored size, filter mask)
    /// per chunk, in index order.
    ///
    /// This reads only the chunk index, not the chunk data, so a caller can seek
    /// to and decode chunks one at a time without materializing the whole
    /// dataset. The curated analogue of `H5Dget_num_chunks` + `H5Dget_chunk_info`
    /// (`chunks()?.len()` is the chunk count).
    ///
    /// Returns `Ok(vec![])` for a chunked dataset whose storage has not been
    /// allocated yet (including a not-yet-written dataset that will use a
    /// [`ChunkIndex::BTreeV2`] index). Returns `Err` if the dataset is not chunked
    /// (check [`layout`](Self::layout) or [`is_chunked`](Self::is_chunked) first),
    /// or if its allocated storage is indexed by a [`ChunkIndex::BTreeV2`] index,
    /// which has no enumerator yet.
    pub fn chunks(&self) -> Result<Vec<Chunk>, Error> {
        let rank = self.dataspace()?.dimensions.len();
        Ok(self
            .raw_chunks()?
            .into_iter()
            .map(|c| Chunk {
                offset: c.offsets.into_iter().take(rank).collect(),
                address: c.address,
                storage_size: u64::from(c.chunk_size),
                filter_mask: c.filter_mask,
            })
            .collect())
    }

    /// This dataset's filter pipeline as an ordered list of [`Filter`]s — each
    /// with its identifier, optional name, optional/mandatory flag, and client
    /// data — or an empty vector when the dataset is unfiltered.
    ///
    /// The detailed companion to [`filters`](Self::filters), which returns just
    /// the identifiers. Filters are listed in application (write) order — the
    /// on-disk pipeline order, matching [`filters`](Self::filters); a reader
    /// inverts them in the *reverse* of this order to decode a chunk. The curated
    /// analogue of `H5Pget_nfilters` + `H5Pget_filter2`.
    pub fn filter_pipeline(&self) -> Vec<Filter> {
        self.filter_pipeline_parsed()
            .map(|p| {
                p.filters
                    .into_iter()
                    .map(|f| Filter {
                        id: f.filter_id,
                        name: f.name,
                        is_optional: f.flags & 0x1 != 0,
                        client_data: f.client_data,
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Shift a base-relative on-disk address to an absolute file offset using the
    /// superblock base address (`addr_offset`). A no-op for the common
    /// base-zero file. Returns `Ok(None)` for an unallocated (undefined) address.
    fn absolute_address(&self, address: Option<u64>) -> Result<Option<u64>, Error> {
        match address {
            Some(rel) => Ok(Some(rel.checked_add(self.file.addr_offset).ok_or(
                crate::error::FormatError::OffsetOverflow {
                    offset: rel,
                    length: 0,
                },
            )?)),
            None => Ok(None),
        }
    }

    /// Returns the simplified datatype of the dataset.
    pub fn dtype(&self) -> Result<DType, Error> {
        let dt = self.datatype()?;
        Ok(classify_datatype(&dt))
    }

    /// The size in bytes of one on-disk element of this dataset's datatype —
    /// HDF5's datatype storage size (`H5Tget_size`).
    ///
    /// This is the byte width of a single stored element: 8 for `f64`, the
    /// declared length for a fixed-length string, the record size for a compound
    /// type, or the reference/descriptor size for a variable-length type (whose
    /// payload lives separately in the file's global heaps).
    ///
    /// Multiplied by the element count from [`shape`](Self::shape), it is the
    /// exact number of raw bytes a full [`read_raw`](Self::read_raw)
    /// materializes. A caller reading an untrusted file can use it to bound that
    /// allocation up front rather than trusting the file's declared extent: a
    /// dataset can name a small element count yet a per-element size of billions
    /// of bytes, so the product — not the count alone — is what a read allocates.
    pub fn element_size(&self) -> Result<u64, Error> {
        Ok(u64::from(self.datatype()?.type_size()))
    }

    /// The raw bytes of this dataset's user-defined fill value, encoded in its
    /// datatype, or `None` when no user-defined fill value is set (the library
    /// default or an explicitly undefined fill). Reads whichever Fill Value
    /// message the header carries — the current `0x0005` (versions 1/2/3) or the
    /// legacy `0x0004` — so files from this crate, the reference C library, and
    /// h5py are all handled.
    pub(crate) fn defined_fill_bytes(&self) -> Result<Option<Vec<u8>>, Error> {
        let msg = self
            .header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::FillValue)
            .or_else(|| {
                self.header
                    .messages
                    .iter()
                    .find(|m| m.msg_type == MessageType::FillValueOld)
            });
        match msg {
            Some(m) => Ok(crate::fill_value::parse_defined_fill_value(
                m.msg_type,
                &self.file.message_body(m)?,
            )?),
            None => Ok(None),
        }
    }

    /// The fill bytes that *unallocated storage reads as* — which is not the
    /// same question [`defined_fill_bytes`](Self::defined_fill_bytes) answers.
    ///
    /// A dataset may declare a fill value and also declare, through the Fill
    /// Value Write Time, that the library never writes it
    /// (`H5D_FILL_TIME_NEVER`). Its unallocated storage then has no defined
    /// contents — the C library leaves the read buffer untouched — so this
    /// returns `None` and the region reads as deterministic zeros rather than as
    /// a value nothing ever put there. `fill_value` still reports the declared
    /// value, because it *is* declared; see [`fill_value_is_written`].
    fn fill_bytes(&self) -> Result<Option<Vec<u8>>, Error> {
        let msg = self
            .header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::FillValue)
            .or_else(|| {
                self.header
                    .messages
                    .iter()
                    .find(|m| m.msg_type == MessageType::FillValueOld)
            });
        let Some(m) = msg else {
            return Ok(None);
        };
        let body = self.file.message_body(m)?;
        if !crate::fill_value::fill_value_is_written(m.msg_type, &body)? {
            return Ok(None);
        }
        Ok(crate::fill_value::parse_defined_fill_value(
            m.msg_type, &body,
        )?)
    }

    /// Read the whole dataset with `decode`, sweeping it a row window at a time.
    ///
    /// A typed whole-dataset read used to be [`read_raw`](Self::read_raw)
    /// followed by a decode of the entire buffer, which held the stored bytes
    /// and the decoded values at the same time and so peaked at twice the
    /// dataset — a caller reading a 4 GiB array needed 8 GiB (issue #289).
    /// Decoding a window at a time leaves one window of stored bytes beside the
    /// output instead of a whole second copy of it, and the bytes are identical
    /// either way: a window returns exactly the rows [`read_raw`](Self::read_raw)
    /// would have put there.
    ///
    /// The output buffer is reserved once, at the size the whole dataset decodes
    /// to, so no window reallocates it — a growth step would put a second copy of
    /// the output alongside the first and give back what the windowing saved.
    ///
    /// A dataset that fits in one window — including one with no rows at all — is
    /// read whole. The empty case still runs `decode`, because a decoder is also
    /// what reports a datatype it cannot read, and a zero-element string dataset
    /// must go on failing a numeric read rather than answering with an empty
    /// vector.
    fn read_whole_typed<T, F>(&self, out_size: OutputSize, decode: F) -> Result<Vec<T>, Error>
    where
        F: Fn(&[u8], &Datatype, &mut Vec<T>) -> Result<(), FormatError>,
    {
        let dt = self.datatype()?;
        let ds = self.dataspace()?;
        let dl = self.data_layout()?;
        let pipeline = self.filter_pipeline_parsed();
        // See `read_raw`: an unparseable fill value message is carried into the
        // read rather than failing it up front.
        let fill_bytes = self.fill_bytes();
        let elem_size = dt.element_size_usize()?;
        let fill = match &fill_bytes {
            Ok(b) => FillPattern::new(b.as_deref(), elem_size),
            Err(_) => FillPattern::UNKNOWN,
        };
        let spec = RawReadSpec {
            layout: &dl,
            dataspace: &ds,
            datatype: &dt,
            pipeline: pipeline.as_ref(),
            fill,
        };

        // What a whole read checks before it reads a byte, and what a sweep would
        // otherwise skip: a compact or contiguous layout whose declared size
        // disagrees with the dataspace is refused. Reading in windows must not
        // turn that into a check that fires only on datasets small enough to be
        // read whole.
        let stored = spec.stored_byte_len()?;

        // A window is cut by the *stored* element width, while a decoder slices
        // what it is handed by the width of the type it decodes — the base type,
        // for an enumeration. Those are the same width for every valid file, an
        // enumeration's size being its base's. A crafted file where they differ
        // must not get one verdict from a sweep and another from a whole read, so
        // it is read whole.
        let decoded_width = data_read::effective_numeric(&dt).type_size();

        let mut out = Vec::new();
        let n0 = ds.dimensions.first().copied().unwrap_or(1);
        let rows = typed_window_rows(&dl, &ds, elem_size)?.get();
        if n0 <= rows || decoded_width != dt.type_size() {
            // No reservation here: `decode` sizes the output from the bytes it
            // was handed, which is exact.
            let raw = self.file.read_dataset_raw(spec, &self.chunk_cache)?;
            decode(&raw, &dt, &mut out)?;
            return Ok(out);
        }

        let values = match out_size {
            OutputSize::PerElement => ds.num_elements().to_usize()?,
            OutputSize::PerByte => stored,
        };

        // One pass for the whole sweep, not one per window: the sweep visits each
        // chunk exactly once, so a window that offered its chunks to a cache
        // already full would copy and evict with no later reader for either. The
        // cache ends up holding what a whole read would have left it — the
        // chunks reached first. See [`CachePass`].
        let pass = self.chunk_cache.begin_pass();
        let mut start = 0;
        while start < n0 {
            let count = rows.min(n0 - start);
            let raw =
                self.file
                    .read_dataset_raw_rows(spec, &self.chunk_cache, pass, start, count)?;
            if start == 0 {
                // Reserved once, and only after a window has come back. This size
                // comes from the file: sizing an allocation from it before
                // reading anything lets a dataspace claiming a terabyte ask for a
                // terabyte over a file that cannot serve one row. `try_reserve`
                // for the same reason — a file-derived capacity that cannot be
                // had is an answer this reader owes its caller, not a panic.
                out.try_reserve(values)
                    .map_err(|_| FormatError::ValueTooLargeForPlatform {
                        value: values as u64,
                        target: "one allocation",
                    })?;
            }
            decode(&raw, &dt, &mut out)?;
            start += count;
        }
        Ok(out)
    }

    /// Read all data as `f64` values.
    ///
    /// This and the other typed whole-dataset readers decode a row window at a
    /// time, so the memory standing beside the returned `Vec` is one window of
    /// stored bytes — on the order of a mebibyte — rather than a second copy of
    /// the dataset. Reading a 4 GiB array costs about 4 GiB, not 8.
    ///
    /// The values are what [`read_raw`](Self::read_raw) returns, decoded: a
    /// dataset stored as a narrower or wider type is converted, so pick the
    /// reader that matches the stored type for a lossless read.
    pub fn read_f64(&self) -> Result<Vec<f64>, Error> {
        self.read_whole_typed(OutputSize::PerElement, data_read::read_as_f64_into)
    }

    /// Read all data as `f32` values.
    pub fn read_f32(&self) -> Result<Vec<f32>, Error> {
        self.read_whole_typed(OutputSize::PerElement, data_read::read_as_f32_into)
    }

    /// Read all data as `i32` values.
    pub fn read_i32(&self) -> Result<Vec<i32>, Error> {
        self.read_whole_typed(OutputSize::PerElement, data_read::read_as_i32_into)
    }

    /// Read all data as `i64` values.
    pub fn read_i64(&self) -> Result<Vec<i64>, Error> {
        self.read_whole_typed(OutputSize::PerElement, data_read::read_as_i64_into)
    }

    /// Read all data as `u64` values.
    pub fn read_u64(&self) -> Result<Vec<u64>, Error> {
        self.read_whole_typed(OutputSize::PerElement, data_read::read_as_u64_into)
    }

    /// Read all data as `u8` values.
    pub fn read_u8(&self) -> Result<Vec<u8>, Error> {
        self.read_raw()
    }

    /// Read all data as `i8` values.
    pub fn read_i8(&self) -> Result<Vec<i8>, Error> {
        self.read_whole_typed(OutputSize::PerByte, |raw, _dt, out| {
            #[expect(
                clippy::cast_possible_wrap,
                reason = "read_i8 reinterprets each stored byte as the signed i8 the caller requested"
            )]
            out.extend(raw.iter().map(|&b| b as i8));
            Ok(())
        })
    }

    /// Read all data as `i16` values.
    pub fn read_i16(&self) -> Result<Vec<i16>, Error> {
        self.read_whole_typed(OutputSize::PerElement, data_read::read_as_i16_into)
    }

    /// Read all data as `u16` values.
    pub fn read_u16(&self) -> Result<Vec<u16>, Error> {
        self.read_whole_typed(OutputSize::PerElement, data_read::read_as_u16_into)
    }

    /// Read all data as `u32` values.
    pub fn read_u32(&self) -> Result<Vec<u32>, Error> {
        self.read_whole_typed(OutputSize::PerElement, data_read::read_as_u32_into)
    }

    /// Read all data as `String` values.
    ///
    /// Fixed-length and variable-length HDF5 string datasets are both
    /// supported. Use [`read_vlen_strings`](Self::read_vlen_strings) when
    /// variable-length allocation limits are required.
    pub fn read_string(&self) -> Result<Vec<String>, Error> {
        let dt = self.datatype()?;
        if vl_data::is_vlen_string_datatype(&dt) {
            self.read_vlen_strings(VlenStringReadOptions::default())
        } else {
            let raw = self.read_raw()?;
            Ok(data_read::read_as_strings(&raw, &dt)?)
        }
    }

    /// Return the total bytes referenced by this VL string dataset.
    ///
    /// This is the payload equivalent of HDF5's `H5Dvlen_get_buf_size`: it
    /// excludes `Vec<String>` and `String` allocation metadata.
    pub fn vlen_string_payload_size(&self) -> Result<u64, Error> {
        let datatype = self.datatype()?;
        if !vl_data::is_vlen_string_datatype(&datatype) {
            return Err(FormatError::TypeMismatch {
                expected: "VariableLength string",
                actual: "non-VariableLength string",
            }
            .into());
        }
        let dataspace = self.dataspace()?;
        let raw = self.read_raw()?;
        Ok(vl_data::vlen_string_payload_size(
            &raw,
            dataspace.num_elements(),
            self.file.offset_size(),
        )?)
    }

    /// Read a VL string dataset with explicit allocation limits.
    ///
    /// Both limits are checked before any string payload is materialized.
    pub fn read_vlen_strings(&self, options: VlenStringReadOptions) -> Result<Vec<String>, Error> {
        let mut strings = Vec::new();
        self.visit_vlen_strings(options, |string| strings.push(string.to_owned()))?;
        Ok(strings)
    }

    /// Visit a VL string dataset one element at a time.
    ///
    /// The string slice passed to `visitor` is valid only for the duration of
    /// that callback. This avoids retaining all decoded string payloads at once.
    ///
    /// On a read-write file ([`File::open_rw`]) the
    /// visitor runs while the file's engine lock is held, so it must not read
    /// or write through this file (or a clone / handle of it) — doing so
    /// deadlocks. Collect values and act on them after the call instead.
    pub fn visit_vlen_strings<F>(
        &self,
        options: VlenStringReadOptions,
        visitor: F,
    ) -> Result<(), Error>
    where
        F: FnMut(&str),
    {
        let datatype = self.datatype()?;
        if !vl_data::is_vlen_string_datatype(&datatype) {
            return Err(FormatError::TypeMismatch {
                expected: "VariableLength string",
                actual: "non-VariableLength string",
            }
            .into());
        }
        let dataspace = self.dataspace()?;
        if let Some(limit) = options.max_elements()
            && dataspace.num_elements() > limit as u64
        {
            return Err(FormatError::VariableLengthElementLimitExceeded {
                limit,
                actual: dataspace.num_elements(),
            }
            .into());
        }
        let raw = self.read_raw()?;
        self.file.with_source(|source| {
            Ok(vl_data::visit_vl_strings_from_source(
                source,
                &raw,
                dataspace.num_elements(),
                self.file.offset_size(),
                self.file.length_size(),
                self.file.addr_offset,
                options,
                visitor,
            )?)
        })
    }

    /// Read a VL string dataset's exact heap bytes, preserving the
    /// null-vs-empty distinction and never lossily decoding.
    ///
    /// Unlike [`read_vlen_strings`](Self::read_vlen_strings), which returns
    /// `String`s via `from_utf8_lossy` and so cannot reproduce embedded NULs or
    /// non-UTF-8 payloads, this yields each element's raw bytes (or a null
    /// marker). It underpins faithful rewriting (e.g. repack) of VL strings.
    pub(crate) fn read_vlen_string_bytes(
        &self,
        options: VlenStringReadOptions,
    ) -> Result<Vec<vl_data::VlByteObject>, Error> {
        let datatype = self.datatype()?;
        if !vl_data::is_vlen_string_datatype(&datatype) {
            return Err(FormatError::TypeMismatch {
                expected: "VariableLength string",
                actual: "non-VariableLength string",
            }
            .into());
        }
        let dataspace = self.dataspace()?;
        if let Some(limit) = options.max_elements()
            && dataspace.num_elements() > limit as u64
        {
            return Err(FormatError::VariableLengthElementLimitExceeded {
                limit,
                actual: dataspace.num_elements(),
            }
            .into());
        }
        let raw = self.read_raw()?;
        self.file.with_source(|source| {
            Ok(vl_data::read_vl_byte_objects_from_source(
                source,
                &raw,
                dataspace.num_elements(),
                self.file.offset_size(),
                self.file.length_size(),
                self.file.addr_offset,
                1, // a VL string's base type is a single byte
                options,
            )?)
        })
    }

    /// Read every element of a *non-string* variable-length (sequence) dataset as
    /// its exact heap bytes, alongside the base-type element size in bytes.
    ///
    /// Each element's heap object holds `length * element_size` bytes, where
    /// `length` is the stored element count and `element_size` is the byte width
    /// of the sequence's base type. Returning the raw bytes (not decoded values)
    /// keeps a faithful rewrite (repack) byte-exact for any base type whose bytes
    /// carry no embedded heap or file addresses. Errors with a
    /// [`TypeMismatch`](crate::FormatError::TypeMismatch) if the datatype is not a
    /// non-string VL datatype.
    pub(crate) fn read_vlen_sequence_bytes(
        &self,
        options: VlenStringReadOptions,
    ) -> Result<(Vec<vl_data::VlByteObject>, usize), Error> {
        let datatype = self.datatype()?;
        let Datatype::VariableLength { base_type, .. } = &datatype else {
            return Err(FormatError::TypeMismatch {
                expected: "non-string VariableLength",
                actual: "non-VariableLength",
            }
            .into());
        };
        if vl_data::is_vlen_string_datatype(&datatype) {
            return Err(FormatError::TypeMismatch {
                expected: "non-string VariableLength",
                actual: "VariableLength string",
            }
            .into());
        }
        let element_size = base_type.type_size() as usize;
        if element_size == 0 {
            return Err(
                FormatError::VlDataError("non-string VL base type has zero size".into()).into(),
            );
        }
        let dataspace = self.dataspace()?;
        if let Some(limit) = options.max_elements()
            && dataspace.num_elements() > limit as u64
        {
            return Err(FormatError::VariableLengthElementLimitExceeded {
                limit,
                actual: dataspace.num_elements(),
            }
            .into());
        }
        let raw = self.read_raw()?;
        let objects = self.file.with_source(|source| {
            vl_data::read_vl_byte_objects_from_source(
                source,
                &raw,
                dataspace.num_elements(),
                self.file.offset_size(),
                self.file.length_size(),
                self.file.addr_offset,
                element_size,
                options,
            )
        })?;
        Ok((objects, element_size))
    }

    /// Read a dataset whose datatype *contains* variable-length references
    /// without being variable-length itself — a compound with a VL member, or an
    /// array of them (issue #201).
    ///
    /// Returns everything a rewrite needs: the element bytes, where each embedded
    /// reference sits within them, and the heap payload each one names. That lets
    /// the writer re-stage the payloads into a new file's global heap and rewrite
    /// the references in place, which is what keeps a rewrite from carrying the
    /// source file's heap addresses into the destination.
    ///
    /// The references are resolved one slot at a time, so `options`' limits apply
    /// per slot rather than across the whole dataset.
    pub(crate) fn read_embedded_vlen_bytes(
        &self,
        slots: &[vl_data::EmbeddedVlSlot],
        options: VlenStringReadOptions,
    ) -> Result<vl_data::EmbeddedVlData, Error> {
        let stride = self.datatype()?.type_size() as usize;
        let dataspace = self.dataspace()?;
        let n = dataspace.num_elements();
        if let Some(limit) = options.max_elements()
            && n > limit as u64
        {
            return Err(
                FormatError::VariableLengthElementLimitExceeded { limit, actual: n }.into(),
            );
        }

        // A zero-element dataset owns no element bytes, so there is no storage
        // to visit: skip the read rather than open a dataset the C library
        // left unallocated only to receive the same empty buffer back.
        let raw = if n == 0 { Vec::new() } else { self.read_raw()? };
        let n_usize = n.to_usize()?;
        let needed = n_usize
            .checked_mul(stride)
            .ok_or(FormatError::OffsetOverflow {
                offset: n,
                length: stride as u64,
            })?;
        if raw.len() < needed {
            return Err(FormatError::UnexpectedEof {
                expected: needed,
                available: raw.len(),
            }
            .into());
        }

        let mut offsets = Vec::with_capacity(n_usize * slots.len());
        let mut objects = Vec::with_capacity(n_usize * slots.len());
        for slot in slots {
            // Gather this slot's reference from every element into a dense buffer,
            // which is the shape the shared VL reader consumes. Each slot has its
            // own base-type width, so they are resolved a slot at a time rather
            // than in one pass.
            let mut dense = Vec::with_capacity(n_usize * VL_REF_SIZE);
            for e in 0..n_usize {
                let at = e * stride + slot.byte_offset;
                dense.extend_from_slice(&raw[at..at + VL_REF_SIZE]);
                offsets.push(at);
            }
            let resolved = self.file.with_source(|source| {
                vl_data::read_vl_byte_objects_from_source(
                    source,
                    &dense,
                    n,
                    self.file.offset_size(),
                    self.file.length_size(),
                    self.file.addr_offset,
                    slot.element_size,
                    options,
                )
            })?;
            objects.extend(resolved);
        }
        Ok(vl_data::EmbeddedVlData {
            raw,
            offsets,
            objects,
        })
    }

    /// Read all attributes of this dataset.
    ///
    /// The variant of each value describes its on-disk encoding; see
    /// [`Group::attrs`] for what that means for matching on it, and prefer the
    /// [`AttrValue`] accessors.
    pub fn attrs(&self) -> Result<HashMap<String, AttrValue>, Error> {
        self.file.attrs_of(&self.header)
    }

    /// The exact on-disk [`Datatype`] of every attribute on this dataset, keyed
    /// by name.
    ///
    /// See [`Group::attr_datatypes`] for what this channel carries that
    /// [`attrs`](Self::attrs) cannot, including how a boolean attribute is
    /// recognized. Note that this describes the *attributes*, not the dataset's
    /// own element type — that is [`datatype`](Self::datatype).
    pub fn attr_datatypes(&self) -> Result<HashMap<String, Datatype>, Error> {
        Ok(self
            .attr_messages()?
            .into_iter()
            .map(|a| (a.name, a.datatype))
            .collect())
    }

    /// Every attribute message on this dataset as it is encoded on disk, in the
    /// order the header holds them.
    ///
    /// See [`Group::attr_messages`] for why repack reads these rather than the
    /// decoded map.
    pub(crate) fn attr_messages(&self) -> Result<Vec<crate::attribute::AttributeMessage>, Error> {
        self.file.attr_messages_of(&self.header)
    }

    /// Returns the exact HDF5 datatype, including compound field offsets and
    /// total record size.
    ///
    /// A committed (`H5Tcommit`) element type — what netCDF-4 writes for a
    /// user-defined type, and what h5py writes for
    /// `create_dataset(..., dtype=f["t"])` — is stored as a reference to the
    /// datatype's own object header and is resolved to the type it names.
    pub fn datatype(&self) -> Result<Datatype, Error> {
        let msg = find_message(&self.header, MessageType::Datatype)?;
        let (dt, _) = Datatype::parse(&self.file.message_body(msg)?)?;
        Ok(dt)
    }

    /// The object-header address of this dataset's committed (shared) element
    /// type, or `None` when the type is written in the dataset's own header.
    ///
    /// [`datatype`](Self::datatype) resolves it either way, so this is for
    /// callers that must *reproduce* the dataset: writing the resolved type back
    /// inline loses the link every C-library reader reports by name, and the
    /// address is what says which committed object to name instead.
    pub(crate) fn committed_datatype_address(&self) -> Result<Option<u64>, Error> {
        let msg = find_message(&self.header, MessageType::Datatype)?;
        self.file.shared_target_address(msg)
    }

    pub(crate) fn dataspace(&self) -> Result<Dataspace, Error> {
        let msg = find_message(&self.header, MessageType::Dataspace)?;
        Ok(Dataspace::parse(
            &self.file.message_body(msg)?,
            self.file.length_size(),
        )?)
    }

    pub(crate) fn data_layout(&self) -> Result<DataLayout, Error> {
        let msg = find_message(&self.header, MessageType::DataLayout)?;
        Ok(DataLayout::parse(
            &msg.data,
            self.file.offset_size(),
            self.file.length_size(),
        )?)
    }

    pub(crate) fn filter_pipeline_parsed(&self) -> Option<FilterPipeline> {
        let msg = self
            .header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::FilterPipeline)?;
        let body = self.file.message_body(msg).ok()?;
        FilterPipeline::parse(&body).ok()
    }

    /// The raw, still-compressed on-disk bytes of every allocated chunk of this
    /// chunked dataset, with each chunk's `(address, on-disk size, filter mask,
    /// logical offset)` — the same `ChunkInfo`s the chunked reader walks before
    /// decompressing. Used by repack to copy compressed chunks verbatim without
    /// ever decoding them.
    ///
    /// Returns `Err` if the layout is not chunked. Returns `Ok(vec![])` for an
    /// empty / never-allocated chunked dataset (no index address). Covers every
    /// index type the reader supports (v3 B-tree and v4 single-chunk, implicit,
    /// fixed-array, and extensible-array).
    pub(crate) fn raw_chunks(&self) -> Result<Vec<crate::chunked_read::ChunkInfo>, Error> {
        let DataLayout::Chunked {
            chunk_dimensions,
            btree_address,
            version,
            chunk_index_type,
            single_chunk_filtered_size,
            single_chunk_filter_mask,
        } = self.data_layout()?
        else {
            return Err(Error::Format(crate::error::FormatError::ChunkedReadError(
                "chunk enumeration requires a chunked dataset".into(),
            )));
        };
        // An undefined index address means no storage is allocated yet.
        let Some(addr) = btree_address else {
            return Ok(Vec::new());
        };
        let dataspace = self.dataspace()?;
        let elem_size = self.datatype()?.element_size_usize()?;
        let base = self.file.addr_offset;
        // The chunk index — its root at `addr` and every internal node — stores
        // addresses relative to the base address. Walk it through a base-relative
        // view so those resolve, then shift each returned chunk address back to an
        // absolute file offset, since callers (repack) read the chunk bytes from
        // the full file source.
        self.file.with_source(|source| {
            if base == 0 {
                return Ok(crate::chunked_read::collect_chunks_for_layout_from_source(
                    source,
                    version,
                    chunk_index_type,
                    addr,
                    single_chunk_filtered_size,
                    single_chunk_filter_mask,
                    &chunk_dimensions,
                    &dataspace,
                    elem_size,
                    self.file.offset_size(),
                    self.file.length_size(),
                )?);
            }
            let framed = BaseOffsetSource {
                inner: source,
                base,
            };
            let mut chunks = crate::chunked_read::collect_chunks_for_layout_from_source(
                &framed,
                version,
                chunk_index_type,
                addr,
                single_chunk_filtered_size,
                single_chunk_filter_mask,
                &chunk_dimensions,
                &dataspace,
                elem_size,
                self.file.offset_size(),
                self.file.length_size(),
            )?;
            for c in &mut chunks {
                c.address = c.address.checked_add(base).ok_or(
                    crate::error::FormatError::OffsetOverflow {
                        offset: c.address,
                        length: 0,
                    },
                )?;
            }
            Ok(chunks)
        })
    }

    /// The raw `FilterPipeline` message bytes from this dataset's object header,
    /// if it has one. Repack reuses this verbatim so that every filter — including
    /// ones this crate cannot itself apply (ZFP, SZIP, unknown) — is reproduced
    /// byte-for-byte in the repacked file's pipeline message.
    pub(crate) fn filter_pipeline_message_bytes(&self) -> Option<Vec<u8>> {
        let msg = self
            .header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::FilterPipeline)?;
        // A shared pipeline message's record body is an address, not a pipeline;
        // its resolved content is what a copy must carry. The content itself is
        // position-independent, so copying it verbatim stays faithful.
        Some(self.file.message_body(msg).ok()?.into_owned())
    }

    /// Read the dataset's exact unfiltered element bytes.
    ///
    /// For compound datasets this preserves all file padding and uses the
    /// offsets reported by [`datatype`](Self::datatype).
    pub fn read_raw(&self) -> Result<Vec<u8>, Error> {
        let dt = self.datatype()?;
        let ds = self.dataspace()?;
        let dl = self.data_layout()?;
        // The data layout's on-disk addresses are left base-relative here;
        // `read_dataset_raw` applies the base address centrally (for both
        // contiguous and chunked layouts) by reading from a base-relative view of
        // the file.
        let pipeline = self.filter_pipeline_parsed();
        // A fill value message this parser cannot read does not, by itself,
        // make the dataset unreadable: it only decides what *unallocated*
        // storage looks like. Carry the uncertainty into the read and let it
        // fail there, and only there. `Dataset::fill_value` still reports the
        // parse error to a caller asking about the value.
        let fill_bytes = self.fill_bytes();
        let fill = match &fill_bytes {
            Ok(b) => FillPattern::new(b.as_deref(), dt.element_size_usize()?),
            Err(_) => FillPattern::UNKNOWN,
        };
        let spec = RawReadSpec {
            layout: &dl,
            dataspace: &ds,
            datatype: &dt,
            pipeline: pipeline.as_ref(),
            fill,
        };
        Ok(self.file.read_dataset_raw(spec, &self.chunk_cache)?)
    }

    /// Read the raw element bytes of the row window `[start_row, start_row + num_rows)`
    /// — a range along the first dimension.
    ///
    /// The windowed companion to [`read_raw`](Self::read_raw): only the storage the
    /// window overlaps is read — a bounded sub-read for compact and contiguous
    /// layouts, just the overlapping chunks for chunked layouts — so peak memory
    /// scales with the window, not the dataset. Use it to stream a large dataset a
    /// fixed number of rows at a time.
    ///
    /// Each row keeps its full inner shape, and the bytes match what
    /// [`read_raw`](Self::read_raw) produces for those rows, so the typed
    /// `read_*_rows` helpers decode a window like their whole-dataset forms. The
    /// window is clamped to the first dimension: a read past the end returns only
    /// the rows that exist, and a 0-D scalar is one row. A window covering every
    /// row delegates to [`read_raw`](Self::read_raw), so a full-range window never
    /// costs more than a whole read. Variable-length string
    /// bytes are heap references, not text — use
    /// [`read_string_rows`](Self::read_string_rows).
    pub fn read_raw_rows(&self, start_row: u64, num_rows: u64) -> Result<Vec<u8>, Error> {
        let dt = self.datatype()?;
        let ds = self.dataspace()?;
        let dl = self.data_layout()?;

        let n0 = ds.dimensions.first().copied().unwrap_or(1);
        let start = start_row.min(n0);
        let count = num_rows.min(n0 - start);

        // A window covering every row is exactly a whole read: delegate, so it
        // never costs a window-shaped copy on top of one.
        // See `read_raw`: an unparseable fill value message is carried into the
        // read rather than failing it up front.
        let parsed_fill = self.fill_bytes();
        let fill = match &parsed_fill {
            Ok(b) => FillPattern::new(b.as_deref(), dt.element_size_usize()?),
            Err(_) => FillPattern::UNKNOWN,
        };

        let pipeline = self.filter_pipeline_parsed();
        let spec = RawReadSpec {
            layout: &dl,
            dataspace: &ds,
            datatype: &dt,
            pipeline: pipeline.as_ref(),
            fill,
        };

        if start == 0 && count == n0 {
            return Ok(self.file.read_dataset_raw(spec, &self.chunk_cache)?);
        }

        // A lone window's successor is the adjacent one, and the chunk they share
        // is the one this read finishes on; `CachePass::LRU` is what retains it.
        Ok(self.file.read_dataset_raw_rows(
            spec,
            &self.chunk_cache,
            CachePass::LRU,
            start,
            count,
        )?)
    }

    /// Windowed [`read_f64`](Self::read_f64) — decodes only the row window.
    pub fn read_f64_rows(&self, start_row: u64, num_rows: u64) -> Result<Vec<f64>, Error> {
        let raw = self.read_raw_rows(start_row, num_rows)?;
        Ok(data_read::read_as_f64(&raw, &self.datatype()?)?)
    }

    /// Windowed [`read_f32`](Self::read_f32) — decodes only the row window.
    pub fn read_f32_rows(&self, start_row: u64, num_rows: u64) -> Result<Vec<f32>, Error> {
        let raw = self.read_raw_rows(start_row, num_rows)?;
        Ok(data_read::read_as_f32(&raw, &self.datatype()?)?)
    }

    /// Windowed [`read_i8`](Self::read_i8) — decodes only the row window.
    #[expect(
        clippy::cast_possible_wrap,
        reason = "read_i8 reinterprets each stored byte as the signed i8 the caller requested"
    )]
    pub fn read_i8_rows(&self, start_row: u64, num_rows: u64) -> Result<Vec<i8>, Error> {
        let raw = self.read_raw_rows(start_row, num_rows)?;
        Ok(raw.iter().map(|&b| b as i8).collect())
    }

    /// Windowed [`read_i16`](Self::read_i16) — decodes only the row window.
    pub fn read_i16_rows(&self, start_row: u64, num_rows: u64) -> Result<Vec<i16>, Error> {
        let raw = self.read_raw_rows(start_row, num_rows)?;
        Ok(data_read::read_as_i16(&raw, &self.datatype()?)?)
    }

    /// Windowed [`read_i32`](Self::read_i32) — decodes only the row window.
    pub fn read_i32_rows(&self, start_row: u64, num_rows: u64) -> Result<Vec<i32>, Error> {
        let raw = self.read_raw_rows(start_row, num_rows)?;
        Ok(data_read::read_as_i32(&raw, &self.datatype()?)?)
    }

    /// Windowed [`read_i64`](Self::read_i64) — decodes only the row window.
    pub fn read_i64_rows(&self, start_row: u64, num_rows: u64) -> Result<Vec<i64>, Error> {
        let raw = self.read_raw_rows(start_row, num_rows)?;
        Ok(data_read::read_as_i64(&raw, &self.datatype()?)?)
    }

    /// Windowed [`read_u8`](Self::read_u8) — reads only the row window.
    pub fn read_u8_rows(&self, start_row: u64, num_rows: u64) -> Result<Vec<u8>, Error> {
        self.read_raw_rows(start_row, num_rows)
    }

    /// Windowed [`read_u16`](Self::read_u16) — decodes only the row window.
    pub fn read_u16_rows(&self, start_row: u64, num_rows: u64) -> Result<Vec<u16>, Error> {
        let raw = self.read_raw_rows(start_row, num_rows)?;
        Ok(data_read::read_as_u16(&raw, &self.datatype()?)?)
    }

    /// Windowed [`read_u32`](Self::read_u32) — decodes only the row window.
    pub fn read_u32_rows(&self, start_row: u64, num_rows: u64) -> Result<Vec<u32>, Error> {
        let raw = self.read_raw_rows(start_row, num_rows)?;
        Ok(data_read::read_as_u32(&raw, &self.datatype()?)?)
    }

    /// Windowed [`read_u64`](Self::read_u64) — decodes only the row window.
    pub fn read_u64_rows(&self, start_row: u64, num_rows: u64) -> Result<Vec<u64>, Error> {
        let raw = self.read_raw_rows(start_row, num_rows)?;
        Ok(data_read::read_as_u64(&raw, &self.datatype()?)?)
    }

    /// Windowed [`read_string`](Self::read_string).
    ///
    /// Fixed-length strings decode straight from the window. Variable-length
    /// strings resolve only the window's heap references, so the window memory
    /// bound holds for them too: peak allocation is the window's references,
    /// its text, and the metadata of the heap collections it touches.
    pub fn read_string_rows(&self, start_row: u64, num_rows: u64) -> Result<Vec<String>, Error> {
        let dt = self.datatype()?;
        if vl_data::is_vlen_string_datatype(&dt) {
            // The window's heap references, read memory-bounded like any other
            // fixed-size element (4-byte length + collection address + 4-byte
            // object index), one row spanning its inner dimensions. Resolving
            // only those against the global heap keeps the bound — the same
            // resolution `read_string` runs over the whole dataset's references.
            let raw = self.read_raw_rows(start_row, num_rows)?;
            let ref_size = 4 + self.file.offset_size() as usize + 4;
            let num_elements = (raw.len() / ref_size) as u64;
            let mut strings = Vec::new();
            self.file.with_source(|source| -> Result<(), Error> {
                Ok(vl_data::visit_vl_strings_from_source(
                    source,
                    &raw,
                    num_elements,
                    self.file.offset_size(),
                    self.file.length_size(),
                    self.file.addr_offset,
                    VlenStringReadOptions::default(),
                    |string| strings.push(String::from(string)),
                )?)
            })?;
            return Ok(strings);
        }
        let raw = self.read_raw_rows(start_row, num_rows)?;
        Ok(data_read::read_as_strings(&raw, &dt)?)
    }

    /// Interpret this dataset as an array of HDF5 object references
    /// (`H5R_OBJECT`) and resolve each, in storage order, to the [`Object`] it
    /// points at.
    ///
    /// MATLAB cell arrays and the `#subsystem#` machinery store their members
    /// this way: the dataset holds one object-header address per element, each
    /// naming an object elsewhere in the file (conventionally under the hidden
    /// `#refs#` group).
    ///
    /// # Errors
    ///
    /// - [`FormatError::TypeMismatch`] if this dataset's datatype is not an
    ///   object reference.
    /// - [`FormatError::InvalidObjectReference`] if an element is a null or
    ///   undefined reference, or does not point at a group or dataset.
    pub fn dereference(&self) -> Result<Vec<Object>, Error> {
        let dt = self.datatype()?;
        if !matches!(
            dt,
            Datatype::Reference {
                ref_type: ReferenceType::Object,
                ..
            }
        ) {
            return Err(FormatError::TypeMismatch {
                expected: "object reference",
                actual: "non-reference datatype",
            }
            .into());
        }
        // An object reference stores an 8-byte object-header address. Refuse a
        // sub-address-width element rather than read a truncated address.
        let elem_size = dt.type_size().to_usize()?;
        if elem_size < 8 {
            return Err(FormatError::TypeMismatch {
                expected: "8-byte object reference",
                actual: "object reference narrower than 8 bytes",
            }
            .into());
        }
        let raw = self.read_raw()?;
        if raw.is_empty() {
            return Ok(Vec::new());
        }
        if !raw.len().is_multiple_of(elem_size) {
            return Err(FormatError::DataSizeMismatch {
                expected: elem_size,
                actual: raw.len(),
            }
            .into());
        }
        let mut out = Vec::with_capacity(raw.len() / elem_size);
        for chunk in raw.chunks_exact(elem_size) {
            let addr = u64::from_le_bytes(chunk[..8].try_into().expect("chunk has >= 8 bytes"));
            out.push(FileInner::object_at_relative(&self.file, addr)?);
        }
        Ok(out)
    }

    /// Decode all elements of a compound dataset field by field.
    ///
    /// Built-in implementations support numeric tuples with one through twelve
    /// fields. Decoding uses the file's field offsets rather than Rust's tuple
    /// memory layout, so padded compound records are supported safely.
    pub fn read_compound<T: CompoundType>(&self) -> Result<Vec<T>, Error> {
        let datatype = self.datatype()?;
        let element_size = datatype.element_size_usize()?;
        if !matches!(datatype, Datatype::Compound { .. }) {
            return Err(FormatError::TypeMismatch {
                expected: "Compound",
                actual: "non-Compound",
            }
            .into());
        }
        let raw = self.read_raw()?;
        if !raw.len().is_multiple_of(element_size.get()) {
            return Err(FormatError::DataSizeMismatch {
                expected: element_size.get(),
                actual: raw.len(),
            }
            .into());
        }
        raw.chunks_exact(element_size.get())
            .map(|bytes| T::decode(&datatype, bytes).map_err(Error::from))
            .collect()
    }

    /// Verify this dataset against its stored provenance hash.
    ///
    /// Recomputes the SHA-256 of the dataset's raw bytes and compares it with
    /// the `_provenance_sha256` attribute written by
    /// [`DatasetBuilder::with_provenance`](crate::DatasetBuilder::with_provenance).
    /// Returns [`VerifyResult::NoHash`](crate::VerifyResult::NoHash) when the
    /// dataset carries no provenance hash, so a missing hash is distinguishable
    /// from an actual mismatch.
    #[cfg(feature = "provenance")]
    pub fn verify_provenance(&self) -> Result<crate::provenance::VerifyResult, Error> {
        use crate::provenance::{ATTR_SHA256, VerifyResult, sha256_hex};

        let attrs = self.attrs()?;
        let stored = match attrs.get(ATTR_SHA256).and_then(AttrValue::as_str) {
            Some(s) => s.trim_end_matches('\0').to_string(),
            None => return Ok(VerifyResult::NoHash),
        };

        let computed = sha256_hex(&self.read_raw()?);
        if computed == stored {
            Ok(VerifyResult::Ok)
        } else {
            Ok(VerifyResult::Mismatch { stored, computed })
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn find_message(
    header: &ObjectHeader,
    msg_type: MessageType,
) -> Result<&crate::object_header::HeaderMessage, Error> {
    header
        .messages
        .iter()
        .find(|m| m.msg_type == msg_type)
        .ok_or(Error::MissingMessage(msg_type))
}

/// Normalize a user-supplied object path to the root-relative form the write
/// session addresses by: strip any leading/trailing `/` so `"/a/b"` and `"a/b"`
/// name the same object.
fn normalize_path(path: &str) -> String {
    path.trim_matches('/').to_string()
}

fn has_message(header: &ObjectHeader, msg_type: MessageType) -> bool {
    header.messages.iter().any(|m| m.msg_type == msg_type)
}

fn is_group(header: &ObjectHeader) -> bool {
    header.messages.iter().any(|m| {
        m.msg_type == MessageType::LinkInfo
            || m.msg_type == MessageType::Link
            || m.msg_type == MessageType::SymbolTable
    })
}

/// The root-relative path of a child named `name` under `parent`, or `None` if
/// the parent has no resolvable path (reached by object reference).
///
/// Free-standing rather than a method on [`Group`] so the member iterators can
/// build child paths from a closure that outlives the borrow of the group they
/// came from.
fn child_path_of(parent: Option<&str>, name: &str) -> Option<String> {
    parent.map(|p| {
        if p.is_empty() {
            name.to_string()
        } else {
            format!("{p}/{name}")
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FileBuilder;
    use std::sync::atomic::AtomicUsize;

    /// Read everything a read-write file can serve through the paired read
    /// paths, as comparable text.
    ///
    /// Each entry exercises a different `with_engine` call site: path
    /// resolution, object-header parsing, group listing, attribute reads (both
    /// the compact and the dense form), a whole-dataset read, and a row-range
    /// read. Errors are formatted rather than unwrapped so that a *divergence in
    /// which error* is reported also fails the comparison.
    fn read_everything(file: &File) -> Vec<String> {
        let mut out = Vec::new();
        out.push(format!("root groups: {:?}", file.root().groups()));
        out.push(format!("root datasets: {:?}", file.root().datasets()));
        out.push(format!("root attrs: {:?}", sorted(file.root().attrs())));

        for path in ["plain", "g/nested", "many_attrs", "missing", "g/missing"] {
            match file.dataset(path) {
                Ok(ds) => {
                    out.push(format!("{path}: shape {:?}", ds.shape()));
                    out.push(format!("{path}: attrs {:?}", sorted(ds.attrs())));
                    out.push(format!("{path}: all {:?}", ds.read_i32()));
                    out.push(format!("{path}: rows {:?}", ds.read_i32_rows(1, 2)));
                    out.push(format!("{path}: raw rows {:?}", ds.read_raw_rows(0, 1)));
                }
                Err(e) => out.push(format!("{path}: error {e}")),
            }
        }
        out
    }

    /// Attribute maps compare only after ordering; `HashMap`'s `Debug` is not
    /// deterministic, and an ordering difference here would be noise rather
    /// than the divergence this is looking for.
    fn sorted(attrs: Result<HashMap<String, AttrValue>, Error>) -> Vec<String> {
        match attrs {
            Ok(map) => {
                let mut v: Vec<String> =
                    map.iter().map(|(k, val)| format!("{k}={val:?}")).collect();
                v.sort();
                v
            }
            Err(e) => vec![format!("error {e}")],
        }
    }

    /// Drive `bytes` down both forms of every read a read-write file serves and
    /// require identical answers.
    ///
    /// Every read has a slice form (walking the whole-file mirror) and a
    /// `Source` form, and until a mirrorless backing lands (issue #198) only the
    /// slice form ever runs. This makes the other form reachable now, so it
    /// cannot quietly drift as its twin is edited.
    fn assert_both_read_paths_agree(bytes: &[u8], what: &str) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("both.h5");
        std::fs::write(&path, bytes).unwrap();

        // One session at a time: `open_rw` takes an exclusive lock, and holding
        // two over one path fails outright where OS locks are mandatory.
        let via_mirror = {
            let f = File::open_rw(&path).unwrap();
            read_everything(&f)
        };
        let via_source = {
            let f = File::open_rw_source_only(&path).unwrap();
            read_everything(&f)
        };

        assert_eq!(
            via_mirror.len(),
            via_source.len(),
            "{what}: the two read paths produced different numbers of results"
        );
        for (m, s) in via_mirror.iter().zip(&via_source) {
            assert_eq!(m, s, "{what}: slice and Source read paths disagree");
        }
        // Guard the guard: a helper that read nothing would make the comparison
        // vacuous, and a file whose datasets all failed to open would too.
        assert!(
            via_mirror.iter().any(|r| r.contains("all Ok(")),
            "{what}: no dataset read succeeded, so this compared nothing"
        );
    }

    /// A file exercising each paired read: a plain dataset, a nested one behind
    /// a group (path resolution), and one carrying enough attributes to force
    /// the dense (fractal-heap) attribute layout rather than compact messages.
    fn both_paths_file_bytes(userblock: Option<u64>) -> Vec<u8> {
        let mut b = FileBuilder::new();
        if let Some(ub) = userblock {
            b.with_userblock(ub);
        }
        b.create_dataset("plain")
            .with_i32_data(&(0..24).collect::<Vec<i32>>())
            .with_shape(&[6, 4])
            .set_attr("units", AttrValue::String("m".into()));
        // Well past the eight-attribute compact limit, so the header converts to
        // the dense layout and the dense extraction path is the one that runs.
        {
            let ds = b
                .create_dataset("many_attrs")
                .with_i32_data(&(0..8).collect::<Vec<i32>>());
            for i in 0..24 {
                ds.set_attr(&format!("attr_{i:02}"), AttrValue::I64(i));
            }
        }
        let mut g = b.create_group("g");
        g.create_dataset("nested")
            .with_i32_data(&(100..112).collect::<Vec<i32>>())
            .with_shape(&[3, 4]);
        b.add_group(g.finish());
        b.finish().unwrap()
    }

    #[test]
    fn both_read_paths_agree() {
        assert_both_read_paths_agree(&both_paths_file_bytes(None), "no userblock");
    }

    /// The userblock case is the one where the two forms are built differently:
    /// the slice form reframes by slicing at the base address, the `Source` form
    /// wraps in a `BaseOffsetSource`. A file with a nonzero base is the only way
    /// to compare them.
    #[test]
    fn both_read_paths_agree_with_a_userblock() {
        assert_both_read_paths_agree(&both_paths_file_bytes(Some(512)), "512-byte userblock");
    }

    /// One 256-element i32 dataset, chunked into 32-element chunks, in memory.
    fn chunked_file_bytes() -> Vec<u8> {
        let data: Vec<i32> = (0..256).collect();
        let mut b = FileBuilder::new();
        b.create_dataset("chunked")
            .with_i32_data(&data)
            .with_shape(&[256])
            .with_chunks(&[32]);
        b.finish().unwrap()
    }

    // The DAPL override must drive the *live* `ChunkCache`, not merely the value
    // reported by `chunk_cache_config()`. These assertions reach the crate's
    // `#[cfg(test)]` cache introspection (unavailable to integration tests), so
    // they fail if the resolved config ever stops flowing into the real cache.

    #[test]
    fn enabled_override_populates_live_cache_over_disabled_file_default() {
        let file = File::from_bytes_with_options(
            chunked_file_bytes(),
            FileAccessProperties::new().with_chunk_cache(ChunkCacheConfig::disabled()),
        )
        .unwrap();

        let ds = file
            .dataset_with_options(
                "chunked",
                DatasetAccessProperties::new().with_chunk_cache(ChunkCacheConfig::new()),
            )
            .unwrap();
        assert_eq!(ds.read_i32().unwrap(), (0..256).collect::<Vec<i32>>());

        // The enabled override built the chunk index and retained chunks; the
        // disabled file default would have left both empty.
        assert!(ds.chunk_cache_stats().index_loaded());
        assert!(ds.chunk_cache_stats().cached_chunks() > 0);
    }

    #[test]
    fn disabled_override_suppresses_live_cache_over_enabled_file_default() {
        let file = File::from_bytes_with_options(
            chunked_file_bytes(),
            FileAccessProperties::new().with_chunk_cache(ChunkCacheConfig::new()),
        )
        .unwrap();

        let ds = file
            .dataset_with_options(
                "chunked",
                DatasetAccessProperties::new().with_chunk_cache(ChunkCacheConfig::disabled()),
            )
            .unwrap();
        assert_eq!(ds.read_i32().unwrap(), (0..256).collect::<Vec<i32>>());

        // The disabled override suppressed the index and chunk retention; the
        // enabled file default would have populated both.
        assert!(!ds.chunk_cache_stats().index_loaded());
        assert_eq!(ds.chunk_cache_stats().cached_chunks(), 0);
    }

    /// A group child whose stored (base-relative) object-header address overflows
    /// `u64` once the base address is added must be rejected, not wrapped or
    /// panicked on. Reaching this needs a nonzero base address, so the file
    /// carries a userblock; the child link's stored address is then rewritten to
    /// `HADDR_UNDEF` (all ones) so `group_children`'s normalization overflows.
    #[test]
    fn group_child_address_base_overflow_is_rejected() {
        const UB: u64 = 512;
        let mut b = FileBuilder::new();
        b.with_userblock(UB);
        let mut child = b.create_group("child");
        child.create_dataset("inner").with_i32_data(&[1, 2, 3]);
        b.add_group(child.finish());
        let mut bytes = b.finish().unwrap();

        // Baseline: the file reads and the subgroup is listed.
        let file = File::from_bytes(bytes.clone()).unwrap();
        assert_eq!(file.root().groups().unwrap(), vec!["child".to_string()]);

        // Rewrite the child's stored object-header address to HADDR_UNDEF. It is
        // stored base-relative (absolute minus the userblock base) and, for this
        // single-child file, appears exactly once in the bytes. The link lives in
        // the root object header's chunk-0.
        let stored = file.root().group("child").unwrap().address - UB;
        let needle = stored.to_le_bytes();
        let matches: Vec<usize> = bytes
            .windows(8)
            .enumerate()
            .filter(|(_, w)| *w == needle)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            matches.len(),
            1,
            "stored child address {stored:#x} was not uniquely locatable: {matches:?}"
        );
        bytes[matches[0]..matches[0] + 8].copy_from_slice(&u64::MAX.to_le_bytes());

        // The v2 object header is checksum-protected, so a real crafted file would
        // carry a matching checksum; recompute the root header's over the edited
        // bytes so parsing reaches the address normalization rather than failing on
        // the checksum first. Mirrors the chunk-0 extent from `parse_v2`.
        #[cfg(feature = "checksum")]
        {
            let root_addr = file.root().address as usize;
            assert_eq!(&bytes[root_addr..root_addr + 4], b"OHDR");
            let flags = bytes[root_addr + 5];
            let mut pos = root_addr + 6;
            if flags & 0x20 != 0 {
                pos += 16;
            }
            if flags & 0x10 != 0 {
                pos += 4;
            }
            let width = 1usize << (flags & 0x03);
            let chunk0 = (0..width).fold(0usize, |acc, i| {
                acc | ((bytes[pos + i] as usize) << (8 * i))
            });
            pos += width;
            let chunk0_end = pos + chunk0;
            assert!(
                matches[0] < chunk0_end,
                "patched link address is outside the root header's chunk-0"
            );
            let cs = crate::checksum::jenkins_lookup3(&bytes[root_addr..chunk0_end]);
            bytes[chunk0_end..chunk0_end + 4].copy_from_slice(&cs.to_le_bytes());
        }

        // Iterating the root now normalizes `u64::MAX + base` and must surface the
        // overflow as a format error rather than panicking or wrapping.
        let file = File::from_bytes(bytes).unwrap();
        match file.root().groups() {
            Err(Error::Format(FormatError::OffsetOverflow { offset, length })) => {
                assert_eq!(offset, u64::MAX);
                assert_eq!(length, UB);
            }
            other => panic!("expected group-child address overflow, got {other:?}"),
        }
    }

    /// Recompute a version-2 object header's checksum over its chunk 0, after a
    /// test has edited a message inside it.
    ///
    /// A crafted file a reader must survive carries a *valid* checksum — an
    /// attacker recomputes it — so a test that edits a header and leaves the old
    /// one measures the checksum rather than the thing it meant to.
    #[cfg(feature = "checksum")]
    fn refresh_v2_header_checksum(bytes: &mut [u8], header_addr: usize) -> std::ops::Range<usize> {
        assert_eq!(&bytes[header_addr..header_addr + 4], b"OHDR");
        let flags = bytes[header_addr + 5];
        let mut pos = header_addr + 6;
        if flags & 0x20 != 0 {
            pos += 16;
        }
        if flags & 0x10 != 0 {
            pos += 4;
        }
        let width = 1usize << (flags & 0x03);
        let chunk0 = (0..width).fold(0usize, |acc, i| {
            acc | ((bytes[pos + i] as usize) << (8 * i))
        });
        pos += width;
        let chunk0_end = pos + chunk0;
        let cs = crate::checksum::jenkins_lookup3(&bytes[header_addr..chunk0_end]);
        bytes[chunk0_end..chunk0_end + 4].copy_from_slice(&cs.to_le_bytes());
        header_addr..chunk0_end
    }

    /// A contiguous layout whose declared size disagrees with its dataspace is
    /// refused, whatever the dataset's size and whichever read asks.
    ///
    /// The whole-dataset readers have always refused it. A typed read now takes a
    /// large dataset a row window at a time, and a window only ever checks that
    /// its *own* rows are inside the declared storage — so without the shared
    /// check this refusal would have applied to small datasets, which are still
    /// read whole, and not to large ones. A validation that fires depending on
    /// the size of the input is the kind that surfaces years later as an
    /// inconsistent bug report, which is why both sizes are here.
    #[cfg(feature = "checksum")]
    #[test]
    fn a_layout_size_disagreeing_with_the_dataspace_is_refused_at_every_dataset_size() {
        // One dataset below the typed read's window budget and read whole; one
        // above it and swept.
        for n in [1000usize, 200_000] {
            let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let mut b = crate::writer::FileBuilder::new();
            b.create_dataset("t")
                .with_f64_data(&data)
                .with_shape(&[n as u64]);
            let mut bytes = b.finish().unwrap();
            // Taken before the edit: an edited header fails its checksum, and the
            // address is needed to recompute it.
            let header_addr = {
                let file = File::from_bytes(bytes.clone()).unwrap();
                file.dataset("t").unwrap().header_address() as usize
            };

            // The layout message's size field: the dataset's byte length, which
            // appears once in the file. The assertion is the fixture's own guard —
            // patching some other field would test nothing in particular.
            let declared = (n * 8) as u64;
            let needle = declared.to_le_bytes();
            let at: Vec<usize> = bytes
                .windows(8)
                .enumerate()
                .filter(|(_, w)| *w == needle)
                .map(|(i, _)| i)
                .collect();
            assert_eq!(
                at.len(),
                1,
                "the stored size {declared} was not uniquely locatable in a {n}-element file: {at:?}"
            );
            bytes[at[0]..at[0] + 8].copy_from_slice(&(declared * 2).to_le_bytes());

            let chunk0 = refresh_v2_header_checksum(&mut bytes, header_addr);
            assert!(
                chunk0.contains(&at[0]),
                "the patched size is outside the header chunk whose checksum was refreshed"
            );

            let file = File::from_bytes(bytes).unwrap();
            let ds = file.dataset("t").unwrap();
            let expected = FormatError::DataSizeMismatch {
                expected: n * 8,
                actual: n * 16,
            };
            for (what, err) in [
                ("read_raw", ds.read_raw().unwrap_err()),
                ("read_f64", ds.read_f64().unwrap_err()),
                ("read_i32", ds.read_i32().unwrap_err()),
            ] {
                match err {
                    Error::Format(got) => assert_eq!(
                        format!("{got:?}"),
                        format!("{expected:?}"),
                        "{what} over {n} elements reported the wrong mismatch"
                    ),
                    other => {
                        panic!("{what} over {n} elements: expected a format error, got {other:?}")
                    }
                }
            }
        }
    }

    /// A zero-row window returns `Ok(empty)` uniformly across layouts, including
    /// over unallocated storage. Unallocated storage now reads as the fill value
    /// rather than erroring, so this no longer guards a cross-layout divergence
    /// in the error; what it still pins is that a window of no rows is an *empty*
    /// buffer and not a zero-length fill, which is what a caller iterating past
    /// the end of a dataset sees.
    #[test]
    fn read_rows_framed_zero_row_window_is_ok_even_when_unallocated() {
        let dl = DataLayout::Contiguous {
            address: None,
            size: 0,
        };
        let ds = Dataspace {
            space_type: crate::dataspace::DataspaceType::Simple,
            rank: 1,
            dimensions: vec![0],
            max_dimensions: None,
        };
        let dt = Datatype::FixedPoint {
            size: 8,
            byte_order: crate::datatype::DatatypeByteOrder::LittleEndian,
            signed: false,
            bit_offset: 0,
            bit_precision: 64,
        };
        let cache = ChunkCache::new();
        let out = read_rows_framed(
            &BytesSource::new(b""),
            RawReadSpec::plain(&dl, &ds, &dt),
            8,
            8,
            &cache,
            CachePass::LRU,
            0,
            0,
            8,
        )
        .expect("a zero-row window must be Ok(empty)");
        assert!(out.is_empty());

        // A Virtual layout is unsupported and must still error for a zero-row
        // window, matching `read_raw`, rather than being swallowed by the early
        // return.
        let virtual_dl = DataLayout::Virtual { version: 4 };
        let err = read_rows_framed(
            &BytesSource::new(b""),
            RawReadSpec::plain(&virtual_dl, &ds, &dt),
            8,
            8,
            &cache,
            CachePass::LRU,
            0,
            0,
            8,
        )
        .expect_err("a virtual layout must error even for a zero-row window");
        assert!(
            matches!(err, FormatError::UnsupportedVirtualLayout),
            "expected UnsupportedVirtualLayout, got {err:?}"
        );
    }

    /// The window a typed whole-dataset read sweeps in is a budget in *stored
    /// bytes* resolved against the dataset's own geometry, and a chunked dataset
    /// adds a second rule on top: whole chunk bands.
    ///
    /// A window that ended mid-band would make the next window decode the band
    /// again — the cost windowing exists to avoid — so the budget is rounded down
    /// to a multiple of the band, and *up* to one whole band when even one band
    /// is over budget. The three ways a window can come out are one rule with
    /// different inputs, which is why they are asserted together.
    #[test]
    fn a_typed_read_windows_in_whole_chunk_bands() {
        const BUDGET: u64 = TYPED_READ_WINDOW_BYTES;
        let ds1 = |dims: &[u64]| Dataspace {
            space_type: crate::dataspace::DataspaceType::Simple,
            rank: dims.len() as u8,
            dimensions: dims.to_vec(),
            max_dimensions: None,
        };
        let contiguous = DataLayout::Contiguous {
            address: Some(0),
            size: 0,
        };
        let chunked = |band: u32| DataLayout::Chunked {
            chunk_dimensions: vec![band, 8],
            btree_address: Some(0),
            version: 3,
            chunk_index_type: None,
            single_chunk_filtered_size: None,
            single_chunk_filter_mask: None,
        };
        let elem = NonZeroUsize::new(8).unwrap();

        // Unchunked: the budget, divided by the row width, exactly.
        assert_eq!(
            typed_window_rows(&contiguous, &ds1(&[1 << 20]), elem)
                .unwrap()
                .get(),
            BUDGET / 8
        );

        // A band that divides the budget takes it unchanged; one that does not
        // is rounded down to a whole number of bands, never up.
        assert_eq!(
            typed_window_rows(&chunked(512), &ds1(&[1 << 20]), elem)
                .unwrap()
                .get(),
            BUDGET / 8
        );
        let rows = typed_window_rows(&chunked(300), &ds1(&[1 << 20]), elem)
            .unwrap()
            .get();
        assert_eq!(rows % 300, 0, "a window must end on a chunk band");
        assert!(
            rows <= BUDGET / 8 && rows > BUDGET / 8 - 300,
            "a window must be the largest whole number of bands within the \
             budget, not a smaller one: {rows} rows against {} in budget",
            BUDGET / 8
        );

        // One band over budget: the window is that band, since a narrower one
        // would decode it twice.
        assert_eq!(
            typed_window_rows(&chunked(1 << 20), &ds1(&[1 << 21]), elem)
                .unwrap()
                .get(),
            1 << 20
        );

        // Rows wider than the whole budget: one row, which is the least a window
        // can be and still make progress.
        assert_eq!(
            typed_window_rows(&contiguous, &ds1(&[4, 1 << 20]), elem)
                .unwrap()
                .get(),
            1
        );

        // Rank 0. A chunked layout message carries rank + 1 dimensions, so a
        // scalar's only entry is the element-size trailer and the "band" read
        // out of it is not a band at all. Nothing rests on it: a scalar has one
        // row, so `n0 <= rows` sends it to the whole read whatever this says.
        // Asserted so that a later reading of `first()` as the leading extent
        // has to account for this case rather than discover it.
        let scalar = Dataspace {
            space_type: crate::dataspace::DataspaceType::Scalar,
            rank: 0,
            dimensions: Vec::new(),
            max_dimensions: None,
        };
        assert!(typed_window_rows(&chunked(8), &scalar, elem).unwrap().get() >= 1);

        // A zero inner dimension makes a row zero bytes wide, and the dataset
        // has no elements at all: one window covers it, and nothing divides by
        // zero on the way there.
        assert_eq!(
            typed_window_rows(&contiguous, &ds1(&[4, 0]), elem)
                .unwrap()
                .get(),
            u64::MAX
        );
    }

    /// The two channels a caller has for one object's attributes.
    struct AttrChannels {
        owner: &'static str,
        /// What `attrs` decoded — the values, lossily.
        values: HashMap<String, AttrValue>,
        /// What `attr_datatypes` reported — the encodings, exactly.
        datatypes: HashMap<String, Datatype>,
    }

    /// One written file: its bytes, and both channels read back from each owner.
    struct AttrFile {
        /// The file as written, so a test can assert which storage form it got.
        bytes: Vec<u8>,
        owners: Vec<AttrChannels>,
    }

    /// Put the same attributes on the root group and on a dataset, write the
    /// file, and read both channels back from each owner.
    ///
    /// Both owners, because `Group` and `Dataset` reach their attribute messages
    /// by different routes: one parses an object header by address, the other
    /// already holds one.
    fn attr_channels(
        values: &[(&str, AttrValue)],
        verbatim: &[crate::attribute::AttributeMessage],
    ) -> AttrFile {
        let mut b = FileBuilder::new();
        for (name, value) in values {
            b.set_attr(name, value.clone());
        }
        for message in verbatim {
            b.set_attr_verbatim(message.clone());
        }
        {
            let ds = b.create_dataset("data").with_f64_data(&[1.0]);
            for (name, value) in values {
                ds.set_attr(name, value.clone());
            }
            for message in verbatim {
                ds.set_attr_verbatim(message.clone());
            }
        }
        let bytes = b.finish().unwrap();
        let file = File::from_bytes(bytes.clone()).unwrap();
        let root = file.root();
        let dataset = file.dataset("data").unwrap();
        AttrFile {
            bytes,
            owners: vec![
                AttrChannels {
                    owner: "root group",
                    values: root.attrs().unwrap(),
                    datatypes: root.attr_datatypes().unwrap(),
                },
                AttrChannels {
                    owner: "dataset",
                    values: dataset.attrs().unwrap(),
                    datatypes: dataset.attr_datatypes().unwrap(),
                },
            ],
        }
    }

    /// The datatype channel carries the width the value channel widens away: one
    /// attribute, read as a 64-bit value and as the 4-byte type it is stored as.
    ///
    /// The pair is the point. `AttrValue` is documented as lossy, so a caller
    /// that needs the encoding — to map it onto a typed column, say — needs
    /// somewhere else to read it, and for an attribute there was nowhere (#248).
    #[test]
    fn attr_datatypes_reports_the_width_attrs_widens() {
        for c in attr_channels(&[("count", AttrValue::I32(-7))], &[]).owners {
            assert_eq!(
                c.values.get("count"),
                Some(&AttrValue::I64(-7)),
                "{}: the value channel widens every integer to 64 bits",
                c.owner
            );
            let Some(Datatype::FixedPoint { size, signed, .. }) = c.datatypes.get("count") else {
                panic!(
                    "{}: expected a fixed-point datatype, got {:?}",
                    c.owner,
                    c.datatypes.get("count")
                );
            };
            assert_eq!(
                (*size, *signed),
                (4, true),
                "{}: the datatype channel must report the width on disk",
                c.owner
            );
        }
    }

    /// Every attribute message is reported, including one `attrs` drops because
    /// no `AttrValue` can carry it.
    ///
    /// That is what lets a caller tell a dropped attribute from an absent one. An
    /// omission with nothing to compare against is invisible, which is how every
    /// `np.bool_` attribute in an h5py file went missing without a trace (#248).
    #[test]
    fn attr_datatypes_reports_an_attribute_attrs_omits() {
        let opaque = Datatype::Opaque {
            size: 3,
            tag: b"rgb".to_vec(),
        };
        let raw = crate::attribute::AttributeMessage {
            name: "raw".into(),
            datatype: opaque.clone(),
            dataspace: Dataspace {
                space_type: crate::dataspace::DataspaceType::Scalar,
                rank: 0,
                dimensions: vec![],
                max_dimensions: None,
            },
            raw_data: vec![1, 2, 3],
            datatype_location: crate::shared_message::DatatypeLocation::Inline,
        };

        for c in attr_channels(&[("count", AttrValue::I32(1))], std::slice::from_ref(&raw)).owners {
            assert!(
                !c.values.contains_key("raw"),
                "{}: an opaque attribute has no `AttrValue`, so `attrs` omits it — \
                 if that changes, this test is measuring the wrong thing",
                c.owner
            );
            assert_eq!(
                c.datatypes.get("raw"),
                Some(&opaque),
                "{}: the datatype channel must report an attribute `attrs` omits",
                c.owner
            );
            // Specific to the one attribute: a channel that reported only the
            // undecodable one, or dropped its neighbour, would pass the above.
            assert!(
                c.values.contains_key("count") && c.datatypes.contains_key("count"),
                "{}: the attribute beside it must appear in both channels",
                c.owner
            );
        }
    }

    /// The channel must cover dense (fractal-heap) attribute storage, not only
    /// the compact form that lives in the object header.
    ///
    /// The two tests above use three attributes, which is well inside the
    /// writer's compact threshold, so on their own they leave every dense path
    /// unpinned: an implementation that walked `hdr.messages` directly and never
    /// touched the heap would pass both and report *nothing* for a real file with
    /// many attributes. Twelve is past the threshold, and the heap signature is
    /// asserted so the fixture cannot quietly revert to compact storage and take
    /// the coverage with it.
    #[test]
    fn attr_datatypes_covers_dense_attribute_storage() {
        let names: Vec<String> = (0..12).map(|i| format!("a{i:02}")).collect();
        let values: Vec<(&str, AttrValue)> = names
            .iter()
            .enumerate()
            .map(|(i, n)| {
                (
                    n.as_str(),
                    #[expect(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                    AttrValue::I32(i as i32),
                )
            })
            .collect();

        let f = attr_channels(&values, &[]);
        assert!(
            f.bytes.windows(4).any(|w| w == b"FRHP"),
            "the fixture must really use dense storage, or this test proves \
             nothing beyond the compact path the other tests already cover"
        );

        for c in f.owners {
            assert_eq!(
                c.datatypes.len(),
                names.len(),
                "{}: every attribute in the heap must be reported, got {:?}",
                c.owner,
                c.datatypes.keys().collect::<Vec<_>>()
            );
            // The two channels must agree on which attributes exist: all of these
            // decode, so neither one has anything to omit here.
            let mut from_values: Vec<&String> = c.values.keys().collect();
            let mut from_types: Vec<&String> = c.datatypes.keys().collect();
            from_values.sort();
            from_types.sort();
            assert_eq!(
                from_values, from_types,
                "{}: the channels disagree",
                c.owner
            );
            for name in &names {
                assert!(
                    matches!(
                        c.datatypes.get(name),
                        Some(Datatype::FixedPoint { size: 4, .. })
                    ),
                    "{}: {name} must keep its 4-byte width through the heap, got {:?}",
                    c.owner,
                    c.datatypes.get(name)
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Coalesced chunk reads (see `crate::chunk_span`)
    // -----------------------------------------------------------------------

    /// A [`Source`] over a file image that counts what the reader asks the file
    /// for, so a test can assert read *volume* and not only the values returned.
    struct CountingSource {
        bytes: Vec<u8>,
        reads: Arc<AtomicUsize>,
        bytes_read: Arc<AtomicUsize>,
    }

    impl Source for CountingSource {
        fn len(&self) -> u64 {
            self.bytes.len() as u64
        }

        fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            self.bytes_read.fetch_add(buf.len(), Ordering::Relaxed);
            BytesSource::new(&self.bytes).read_at(offset, buf)
        }
    }

    /// Counters for one measured read.
    #[derive(Default)]
    struct ReadCounts {
        reads: Arc<AtomicUsize>,
        bytes: Arc<AtomicUsize>,
    }

    impl ReadCounts {
        /// Run `f` and report `(reads, bytes)` it cost.
        fn measure<R>(&self, f: impl FnOnce() -> R) -> (usize, usize) {
            let r0 = self.reads.load(Ordering::Relaxed);
            let b0 = self.bytes.load(Ordering::Relaxed);
            f();
            (
                self.reads.load(Ordering::Relaxed) - r0,
                self.bytes.load(Ordering::Relaxed) - b0,
            )
        }
    }

    /// A streaming [`File`] over `bytes` whose reads are counted, built the way
    /// [`File::open_streaming_with_options`] builds one over a file handle.
    fn counting_streaming_file(
        bytes: Vec<u8>,
        counts: &ReadCounts,
        access: FileAccessProperties,
    ) -> File {
        let source: Box<dyn Source + Send + Sync> = Box::new(CountingSource {
            bytes,
            reads: Arc::clone(&counts.reads),
            bytes_read: Arc::clone(&counts.bytes),
        });
        let (superblock, addr_offset) =
            FileInner::parse_superblock_source(source.as_ref()).expect("parse superblock");
        File {
            inner: Arc::new(FileInner::from_parts(
                Backend::Streaming(source),
                superblock,
                addr_offset,
                None,
                access,
            )),
        }
    }

    /// An `n`-element f64 dataset in chunks of `chunk` elements.
    fn chunked_f64_file(n: usize, chunk: u64) -> (Vec<u8>, Vec<f64>) {
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut builder = FileBuilder::new();
        builder
            .create_dataset("d")
            .with_f64_data(&data)
            .with_shape(&[n as u64])
            .with_chunks(&[chunk]);
        (builder.finish().expect("write file"), data)
    }

    /// One chunk per row is what a writer that appends as data arrives
    /// produces; the rows land next to each other, so the streaming reader must
    /// fetch them in a few spans rather than one read each.
    #[test]
    fn a_streaming_read_coalesces_a_run_of_small_chunks() {
        let n = 1024usize;
        let (bytes, data) = chunked_f64_file(n, 1);
        let counts = ReadCounts::default();
        let file = counting_streaming_file(bytes, &counts, FileAccessProperties::new());

        let mut got = Vec::new();
        let (reads, _) = counts.measure(|| got = file.dataset("d").unwrap().read_f64().unwrap());
        assert_eq!(got, data, "the coalesced read must return the same values");

        // Reading each of the 1024 chunks on its own would cost at least that
        // many reads; the whole run plus its metadata fits in far fewer.
        assert!(
            reads < n / 8,
            "expected the {n} chunks to be coalesced into few reads, got {reads}"
        );
    }

    /// A windowed read must coalesce only the chunks its window overlaps: a
    /// plan built over the dataset's whole chunk list would put the rows on
    /// either side of the window inside a span and read them for nothing.
    #[test]
    fn a_windowed_streaming_read_fetches_only_its_own_window() {
        let rows = 4096usize;
        let (bytes, data) = chunked_f64_file(rows, 4);
        let counts = ReadCounts::default();
        let file = counting_streaming_file(bytes, &counts, FileAccessProperties::new());
        let ds = file.dataset("d").unwrap();

        // The first window walks (and caches on this handle) the chunk index,
        // so measure the second: what it reads is the window's own chunks.
        assert_eq!(ds.read_f64_rows(0, 8).unwrap(), data[0..8]);
        let (_, window_bytes) =
            counts.measure(|| assert_eq!(ds.read_f64_rows(2048, 8).unwrap(), data[2048..2056]));

        // Eight rows are two 32-byte chunks.
        assert!(
            window_bytes < 256,
            "an 8-row window read {window_bytes} bytes; it needs 64"
        );
    }

    /// The read volume of a whole-dataset read must not depend on which read it
    /// is. The chunk list comes from the handle's cached index on every read
    /// after the first, and that index is a map: it yields no address order at
    /// all. A reader holding one coalesced span re-reads a whole span each time
    /// an unordered walk crosses back, so the second read of a dataset spanning
    /// more than one span cost two orders of magnitude more than the first.
    ///
    /// A regression here is probabilistic rather than certain — the map's order
    /// is seeded per process, and a run that happened to be sorted would pass —
    /// but with hundreds of chunks over two spans the chance of that is nil.
    /// Correct code passes deterministically.
    #[test]
    fn a_second_whole_read_costs_what_the_first_did() {
        // 512 KiB of f64 in 1 KiB chunks: more than one 256 KiB span, so an
        // unordered walk has somewhere to thrash between.
        let n = 64 * 1024usize;
        let dataset_bytes = n * 8;
        let (bytes, data) = chunked_f64_file(n, 128);
        let counts = ReadCounts::default();
        let file = counting_streaming_file(bytes, &counts, FileAccessProperties::new());
        let ds = file.dataset("d").unwrap();

        counts.measure(|| assert_eq!(ds.read_f64().unwrap(), data));
        let (_, second) = counts.measure(|| assert_eq!(ds.read_f64().unwrap(), data));

        assert!(
            second <= dataset_bytes,
            "the second read fetched {second} bytes of a {dataset_bytes}-byte dataset"
        );
    }

    /// The same, for row windows: `docs/guide/streaming.md` walks a dataset in
    /// windows on one handle, so every window but the first takes its chunk
    /// list from the cached index.
    ///
    /// Each window here covers more than one span, which is what it takes to
    /// see the defect: a window whose chunks all fit a single span is served
    /// out of that one buffer whatever order it walks them in.
    #[test]
    fn a_window_loop_costs_the_dataset_once() {
        // 2 MiB of f64 in 1 KiB chunks, read in 512 KiB windows — two spans
        // each.
        let rows = 256 * 1024usize;
        let dataset_bytes = rows * 8;
        let (bytes, data) = chunked_f64_file(rows, 128);
        let counts = ReadCounts::default();
        let file = counting_streaming_file(bytes, &counts, FileAccessProperties::new());
        let ds = file.dataset("d").unwrap();

        let window = 64 * 1024;
        let (_, total) = counts.measure(|| {
            for lo in (0..rows).step_by(window) {
                let got = ds.read_f64_rows(lo as u64, window as u64).unwrap();
                assert_eq!(got, data[lo..lo + window]);
            }
        });

        assert!(
            total <= 2 * dataset_bytes,
            "a window loop over a {dataset_bytes}-byte dataset read {total} bytes"
        );
    }

    /// A span must not cover a chunk the chunk cache already holds: the read
    /// skips that chunk, so those bytes would be fetched for nothing.
    ///
    /// The cache here is given room for the whole dataset. At the default
    /// sixteen slots a dataset this size evicts its own warm chunks as it
    /// walks, so every chunk misses on the second read and the plan has nothing
    /// to leave out — a config that cannot show the difference either way.
    #[test]
    fn a_warm_chunk_cache_is_not_re_fetched() {
        // 32 KiB of f64 in 32 chunks of 1 KiB, laid end to end.
        let n = 4096usize;
        let chunk_bytes = 1024usize;
        let (bytes, data) = chunked_f64_file(n, 128);
        let counts = ReadCounts::default();
        let file = counting_streaming_file(
            bytes,
            &counts,
            FileAccessProperties::new()
                .with_chunk_cache(ChunkCacheConfig::new().with_max_slots(64)),
        );
        let ds = file.dataset("d").unwrap();

        // Warm the second half of the dataset: chunks 16..31.
        assert_eq!(ds.read_f64_rows(2048, 2048).unwrap(), data[2048..]);
        assert_eq!(
            ds.chunk_cache_stats().cached_chunks(),
            16,
            "the window's own chunks stay cached"
        );

        let (_, second) = counts.measure(|| assert_eq!(ds.read_f64().unwrap(), data));
        assert_eq!(
            second,
            16 * chunk_bytes,
            "only the cold half was needed; a span over the whole dataset would \
             have fetched {} bytes",
            32 * chunk_bytes
        );
    }

    // -----------------------------------------------------------------------
    // Group member iterators (`iter_datasets` / `iter_groups`)
    // -----------------------------------------------------------------------

    /// A root holding `datasets` datasets, `groups` subgroups (each with one
    /// dataset of its own) and one committed datatype, so a member walk has all
    /// three child kinds to sort apart.
    ///
    /// Names are not zero-padded, so lexical order and insertion order disagree
    /// past the tenth member: a walk that silently sorted would show up here.
    fn mixed_member_file(datasets: usize, groups: usize) -> Vec<u8> {
        let mut b = FileBuilder::new();
        b.commit_datatype("a_type", crate::make_i32_type());
        for i in 0..datasets {
            b.create_dataset(&format!("ds{i}"))
                .with_i32_data(&[i as i32, -(i as i32)]);
        }
        for i in 0..groups {
            let mut g = b.create_group(&format!("g{i}"));
            g.create_dataset("inner").with_i32_data(&[i as i32]);
            g.create_dataset("other").with_i32_data(&[-1]);
            b.add_group(g.finish());
        }
        b.finish().expect("write the fixture")
    }

    /// The iterator must report exactly what opening each name reports: the same
    /// members, in the same order, resolving to the same objects. Anything the
    /// two disagree on is a member a caller would see differently for having
    /// chosen the cheaper walk.
    #[test]
    fn iter_datasets_agrees_with_opening_each_name() {
        let file = File::from_bytes(mixed_member_file(12, 3)).unwrap();

        for group in [file.root(), file.group("g1").unwrap()] {
            let names = group.datasets().unwrap();
            assert!(
                !names.is_empty(),
                "the fixture must have members to compare"
            );

            let iterated: Vec<(String, Dataset)> = group.iter_datasets().unwrap().collect();
            assert_eq!(
                iterated.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>(),
                names,
                "the iterator must yield the members `datasets` lists, in that order"
            );

            for (name, ds) in &iterated {
                let opened = group.dataset(name).unwrap();
                assert_eq!(ds.header_address(), opened.header_address(), "{name}");
                assert_eq!(ds.shape().unwrap(), opened.shape().unwrap(), "{name}");
                assert_eq!(ds.read_i32().unwrap(), opened.read_i32().unwrap(), "{name}");
            }
        }
    }

    /// The subgroup counterpart, including that a handle it yields can be walked
    /// again — recursion through `iter_groups` is the shape it exists for.
    #[test]
    fn iter_groups_agrees_with_opening_each_name() {
        let file = File::from_bytes(mixed_member_file(4, 5)).unwrap();
        let root = file.root();

        let names = root.groups().unwrap();
        let iterated: Vec<(String, Group)> = root.iter_groups().unwrap().collect();
        assert_eq!(
            iterated.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>(),
            names,
            "the iterator must yield the subgroups `groups` lists, in that order"
        );
        assert_eq!(names.len(), 5);

        for (name, group) in &iterated {
            assert_eq!(
                group.header_address(),
                root.group(name).unwrap().header_address(),
                "{name}"
            );
            let mut inner = group
                .iter_datasets()
                .unwrap()
                .map(|(n, _)| n)
                .collect::<Vec<_>>();
            inner.sort();
            assert_eq!(inner, ["inner", "other"], "{name} must be walkable in turn");
        }
    }

    /// Each iterator must claim only its own kind of child. A committed datatype
    /// is the child that belongs to neither, and the one a walk asking only for
    /// datasets and groups would otherwise be free to mis-sort into either.
    #[test]
    fn the_member_iterators_sort_the_child_kinds_apart() {
        let file = File::from_bytes(mixed_member_file(3, 2)).unwrap();
        let root = file.root();

        let datasets: Vec<String> = root.iter_datasets().unwrap().map(|(n, _)| n).collect();
        let groups: Vec<String> = root.iter_groups().unwrap().map(|(n, _)| n).collect();

        assert_eq!(datasets, ["ds0", "ds1", "ds2"]);
        assert_eq!(groups, ["g0", "g1"]);
        assert_eq!(root.named_datatypes().unwrap(), ["a_type"]);
        assert!(
            !datasets.contains(&"a_type".to_string()) && !groups.contains(&"a_type".to_string()),
            "a committed datatype is neither a dataset nor a group"
        );
    }

    /// The bytes a walk of `n` members reads, by each route: opening every name,
    /// and iterating handles.
    fn member_walk_bytes(n: usize) -> (usize, usize) {
        let mut b = FileBuilder::new();
        for i in 0..n {
            b.create_dataset(&format!("ds{i}"))
                .with_i32_data(&[i as i32]);
        }
        let bytes = b.finish().expect("write the fixture");

        let counts = ReadCounts::default();
        let file = counting_streaming_file(bytes.clone(), &counts, FileAccessProperties::new());
        let (_, by_name) = counts.measure(|| {
            let root = file.root();
            for name in root.datasets().unwrap() {
                root.dataset(&name).unwrap();
            }
        });

        let counts = ReadCounts::default();
        let file = counting_streaming_file(bytes, &counts, FileAccessProperties::new());
        let (_, iterated) =
            counts.measure(|| for (_, _ds) in file.root().iter_datasets().unwrap() {});

        (by_name, iterated)
    }

    /// Opening members by name re-walks the group's link structure once per
    /// member; iterating handles walks it once. Bytes is the metric that
    /// separates them: a group this size keeps its links inline in the object
    /// header, so that header grows with the member count and re-reading it per
    /// member is quadratic, while the *number* of reads stays linear either way
    /// and would show almost nothing.
    ///
    /// Asserted as how the cost scales rather than as one fixture's byte count,
    /// since a fixed number would pass just as well on a walk that stayed
    /// quadratic with a smaller constant.
    #[test]
    fn iterating_members_enumerates_the_group_once() {
        let (_, iterated_16) = member_walk_bytes(16);
        let (by_name_64, iterated_64) = member_walk_bytes(64);

        // The rule this test exists for, and the only assertion here that is
        // about `iter_datasets` itself.
        assert!(
            iterated_64 <= 5 * iterated_16,
            "one enumeration plus one header per member is linear, so four times \
             the members must cost about four times the bytes: {iterated_16} -> \
             {iterated_64}"
        );

        // Contrast, not a property of this code: it holds because `Group::dataset`
        // re-enumerates. If a future change makes that route cheap enough to turn
        // this red, nothing here has regressed — confirm the scaling assertion
        // above still holds and then drop this one.
        assert!(
            iterated_64 < by_name_64,
            "at 64 members the one-enumeration walk should still be the cheaper: \
             {iterated_64} bytes against {by_name_64}"
        );
    }

    /// A yielded handle must carry the chunk-cache configuration the file was
    /// opened with, the same one `dataset` resolves for it.
    ///
    /// Nothing about the values read would show a handle that quietly ignored
    /// it: the cache decides how often the chunk index is re-parsed and how much
    /// decompressed data is retained, not what comes back. So the configuration
    /// has to be asserted directly, or dropping it here is a silent regression.
    #[test]
    fn a_member_handle_carries_the_files_chunk_cache_config() {
        let configured = ChunkCacheConfig::new()
            .with_max_slots(17)
            .with_max_bytes(4096)
            .with_index_cache(false);
        let file = File::from_bytes_with_options(
            mixed_member_file(3, 0),
            FileAccessProperties::new().with_chunk_cache(configured),
        )
        .unwrap();
        let root = file.root();

        let members: Vec<(String, Dataset)> = root.iter_datasets().unwrap().collect();
        assert_eq!(members.len(), 3);
        for (name, ds) in &members {
            assert_eq!(
                ds.chunk_cache_config(),
                root.dataset(name).unwrap().chunk_cache_config(),
                "{name} must resolve its cache the way `dataset` does"
            );
            assert_eq!(
                ds.chunk_cache_config(),
                configured,
                "{name} must carry the configuration the file was opened with"
            );
        }
    }

    /// A file whose root holds `inner` and `sub`, and a group `g0` holding
    /// children of those same names, plus a `refs` dataset pointing at `g0`.
    ///
    /// The duplicated names are the trap: a member handle that wrongly took a
    /// root-relative path would address a real object rather than fail, so the
    /// mistake would look like a successful write.
    fn dereferenced_group_file() -> Vec<u8> {
        let mut b = FileBuilder::new();
        b.create_dataset("inner").with_i32_data(&[0]);
        let mut root_sub = b.create_group("sub");
        root_sub.create_dataset("x").with_i32_data(&[0]);
        b.add_group(root_sub.finish());

        let mut g = b.create_group("g0");
        g.create_dataset("inner").with_i32_data(&[1]);
        let mut nested = g.create_group("sub");
        nested.create_dataset("x").with_i32_data(&[1]);
        g.add_group(nested.finish());
        b.add_group(g.finish());

        b.create_dataset("refs").with_path_references(&["g0"]);
        b.finish().expect("write the fixture")
    }

    /// A group reached by object reference has no resolvable path, so neither can
    /// its members: there is nothing for a write through one to address. The
    /// iterators must carry that `None` across rather than fall back to a
    /// root-relative path, which here would reach a different, real object.
    #[test]
    fn members_of_a_dereferenced_group_have_no_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("refs.h5");
        std::fs::write(&path, dereferenced_group_file()).unwrap();

        let file = File::open_rw(&path).unwrap();
        let mut objects = file.dataset("refs").unwrap().dereference().unwrap();
        let group = match objects.remove(0) {
            Object::Group(g) => g,
            other => panic!("expected a group, got {other:?}"),
        };

        // The file is writable and both names exist at the root, so a refusal
        // here can only come from the handle having no path to address.
        let (name, mut member) = group.iter_datasets().unwrap().next().unwrap();
        assert_eq!(name, "inner");
        assert!(
            matches!(
                member.set_attr("tag", AttrValue::I64(1)),
                Err(Error::ReadOnly)
            ),
            "a member of a path-less group must refuse a write, not address `/inner`"
        );

        let (name, subgroup) = group.iter_groups().unwrap().next().unwrap();
        assert_eq!(name, "sub");
        assert!(
            matches!(
                subgroup.set_attr("tag", AttrValue::I64(1)),
                Err(Error::ReadOnly)
            ),
            "and so must a subgroup of one, not address `/sub`"
        );
    }

    /// A yielded handle must carry the same root-relative path as one opened by
    /// name, or a write through it would address the wrong object — or no object
    /// at all. A member of a *subgroup* is the case that separates them, since
    /// its path has a prefix to get right.
    #[test]
    fn a_member_handle_resolves_to_its_own_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("members.h5");
        std::fs::write(&path, mixed_member_file(2, 2)).unwrap();

        let file = File::open_rw(&path).unwrap();
        let group = file.group("g1").unwrap();
        let (name, mut ds) = group
            .iter_datasets()
            .unwrap()
            .find(|(n, _)| n == "inner")
            .expect("g1/inner");
        assert_eq!(name, "inner");
        ds.set_attr("tag", AttrValue::I64(7)).unwrap();
        file.commit().unwrap();
        // Windows holds the write lock until the session is dropped.
        drop(ds);
        drop(group);
        drop(file);

        let file = File::open(&path).unwrap();
        assert_eq!(
            file.dataset("g1/inner")
                .unwrap()
                .attrs()
                .unwrap()
                .get("tag")
                .and_then(AttrValue::as_i64),
            Some(7),
            "the attribute must land on the member the handle came from"
        );
        assert!(
            !file
                .dataset("g0/inner")
                .unwrap()
                .attrs()
                .unwrap()
                .contains_key("tag"),
            "and on no other group's member of the same name"
        );
    }

    /// Every read-write entry point has to hand the fapl's `fsync` cadence to
    /// the session it opens, and none of them can be checked from outside: a
    /// skipped barrier writes the same bytes as an issued one (issue #263).
    ///
    /// One entry point missing the funnel is the whole failure mode, so this
    /// asserts the property at each of them rather than at the funnel.
    #[test]
    fn every_read_write_open_carries_the_fapl_sync_policy() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let fixture = |name: &str| {
            let path = dir.path().join(name);
            let mut b = FileBuilder::new();
            b.create_dataset("d")
                .with_i32_data(&[1, 2, 3, 4])
                .with_shape(&[4])
                .with_maxshape(&[u64::MAX])
                .with_chunks(&[2]);
            b.write(&path).unwrap();
            path
        };
        let props = || FileAccessProperties::new().with_sync_policy(SyncPolicy::OnClose);
        let policy_of = |file: &File| match &file.inner.backend {
            Backend::Edit(m) => m
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .sync_policy(),
            _ => panic!("a read-write open must build an editing session"),
        };

        let opened = File::open_rw_with_options(fixture("open_rw.h5"), props()).unwrap();
        assert_eq!(policy_of(&opened), SyncPolicy::OnClose, "File::open_rw");
        // Windows OS locks are mandatory: release each session before the next
        // open touches the same directory's files.
        drop(opened);

        let created = File::create_with_options(
            dir.path().join("created.h5"),
            crate::FileCreateProperties::new(),
            props(),
        )
        .unwrap();
        assert_eq!(policy_of(&created), SyncPolicy::OnClose, "File::create");
        drop(created);

        let swmr = File::open_swmr_writer_with_options(fixture("swmr.h5"), props()).unwrap();
        assert_eq!(
            policy_of(&swmr),
            SyncPolicy::OnClose,
            "File::open_swmr_writer"
        );
        drop(swmr);

        let bounded = File::open_rw_with_options(
            fixture("bounded.h5"),
            props().with_memory_strategy(MemoryStrategy::Bounded),
        )
        .unwrap();
        assert_eq!(
            policy_of(&bounded),
            SyncPolicy::OnClose,
            "File::open_rw_with_options (bounded)"
        );
    }

    /// `close` and `drop` issue their barrier under *every* policy, on both the
    /// ordinary and the SWMR branch — the four sites where this crate writes
    /// after the last point a caller could have ordered anything (issue #263).
    ///
    /// This is the half of the contract `SyncPolicy` cannot express: the two
    /// `drop` sites are unreachable by any caller discipline at all, since the
    /// handle that would have issued `File::sync` is gone by the time they run.
    /// Asserted through a counting image, because the difference between a
    /// forced barrier and a skipped one is invisible in the bytes.
    #[test]
    fn close_and_drop_force_their_barrier_under_every_policy() {
        use crate::edit::WriteEngine;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        // `swmr` picks the teardown branch; `explicit` picks `close` over `drop`.
        let teardown = |name: &str, swmr: bool, explicit: bool| -> u64 {
            let path = dir.path().join(name);
            let mut b = FileBuilder::new();
            b.create_dataset("d")
                .with_i32_data(&(0..8).collect::<Vec<_>>())
                .with_shape(&[8])
                .with_maxshape(&[u64::MAX])
                .with_chunks(&[4]);
            // Persisting, so the ordinary branch has manager re-homing to do:
            // an immediate append below leaves the on-disk managers mid-file,
            // and settling them is the write no earlier sync could cover.
            b.with_file_space_strategy(crate::FileSpaceStrategy::FsmAggr, true, 1);
            b.write(&path).unwrap();

            let syncs = Arc::new(AtomicU64::new(0));
            let session =
                WriteEngine::open_sync_counting(&path, SyncPolicy::OnClose, Arc::clone(&syncs))
                    .unwrap();
            let mut inner = FileInner::from_rw_session(
                session,
                FileAccessProperties::new().with_sync_policy(SyncPolicy::OnClose),
            )
            .unwrap();
            inner.swmr_write = swmr;
            let file = File {
                inner: Arc::new(inner),
            };
            if !swmr {
                // The SWMR branch stages nothing and appends through its own
                // path; give the ordinary branch real work to settle.
                file.dataset("d")
                    .unwrap()
                    .append(&[8i32, 9, 10, 11])
                    .unwrap();
            }
            assert_eq!(
                syncs.load(AtomicOrdering::Relaxed),
                0,
                "nothing before teardown may sync under OnClose ({name})"
            );

            if explicit {
                file.close().unwrap();
            } else {
                drop(file);
            }
            syncs.load(AtomicOrdering::Relaxed)
        };

        for (name, swmr, explicit) in [
            ("close_plain.h5", false, true),
            ("close_swmr.h5", true, true),
            ("drop_plain.h5", false, false),
            ("drop_swmr.h5", true, false),
        ] {
            assert!(
                teardown(name, swmr, explicit) > 0,
                "{name} must force its barrier: the writes it makes are past the \
                 last point a caller could have ordered them"
            );
        }
    }
}
