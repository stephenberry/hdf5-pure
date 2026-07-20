//! Reading API: File, Dataset, and Group handles for reading HDF5 files.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::edit::{AppendBuilder, SpaceAccounting, WriteEngine};
use crate::element::H5Element;
use crate::type_builders::DatasetBuilder;

use crate::attribute::{extract_attributes_full, extract_attributes_full_from_source};
use crate::chunk_cache::{ChunkCache, ChunkCacheConfig, ChunkCacheStats};
use crate::compound::CompoundType;
use crate::convert::TryToUsize;
use crate::data_layout::DataLayout;
use crate::data_read;
use crate::dataspace::Dataspace;
use crate::datatype::{Datatype, ReferenceType};
use crate::error::{Error, FormatError};
use crate::file_lock::FileLocking;
use crate::file_space_info::{FileSpaceInfo, FileSpaceStrategy};
use crate::filter_pipeline::FilterPipeline;
use crate::free_space_manager;
use crate::group_v1::GroupEntry;
use crate::group_v2;
use crate::layout_info::{Chunk, ChunkIndex, Filter, Layout};
use crate::libver::LibVer;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::signature;
use crate::source::{
    BytesSource, FileSource, MetadataCacheConfig, MetadataCachingSource, ReadSeekSource,
};
use crate::superblock::Superblock;
use crate::vl_data::{self, VlenStringReadOptions};

use crate::types::{AttrValue, DType, attrs_to_map, classify_datatype};

// ---------------------------------------------------------------------------
// File
// ---------------------------------------------------------------------------

/// Backing store for a [`File`]: either the whole file buffered in memory, or a
/// lazy [`FileSource`] that reads regions on demand (see [`File::open_streaming`]).
enum Backend {
    InMemory(Vec<u8>),
    Streaming(Box<dyn FileSource + Send + Sync>),
    /// A read-write file opened with [`File::open_rw`]: a [`WriteEngine`] (a
    /// whole-file mirror + exclusive OS lock + staged-edit queues) behind a lock,
    /// so owned handles can both read and mutate in place. Reads slice the
    /// engine's mirror; handle write methods route to the engine, and
    /// `File::commit` applies staged structural edits. Boxed to keep the
    /// `Backend` enum small (a `WriteEngine` is far larger than the other
    /// variants).
    Mirror(Box<Mutex<WriteEngine>>),
}

/// A borrowed `FileSource` view over a [`File`]'s backend, used by the
/// streaming-capable read paths so one call site serves both backends.
pub(crate) enum SourceView<'a> {
    Mem(&'a [u8]),
    Stream(&'a (dyn FileSource + Send + Sync)),
}

impl FileSource for SourceView<'_> {
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

/// A `FileSource` view shifted forward by a base address: every read at a
/// base-relative `offset` is served from `inner` at `offset + base`. Used by the
/// dataset-payload read path on a file with a userblock, where the data-layout's
/// on-disk addresses (contiguous data, chunk index, and chunk data) are stored
/// relative to the base address — presenting the reader this shifted view lets
/// those relative addresses index it directly, exactly as the in-memory path
/// slices the buffer at `base`. `len`/`read_at` shift by the base; `read_metadata_at`
/// forwards to the inner source (at the absolute offset) so its metadata cache is
/// shared, while payload reads keep the default uncached `read_exact_at`.
struct BaseOffsetSource<'a, S: FileSource + ?Sized> {
    inner: &'a S,
    base: u64,
}

impl<S: FileSource + ?Sized> FileSource for BaseOffsetSource<'_, S> {
    fn len(&self) -> u64 {
        self.inner.len().saturating_sub(self.base)
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        let abs = offset
            .checked_add(self.base)
            .ok_or(FormatError::OffsetOverflow {
                offset,
                length: buf.len() as u64,
            })?;
        self.inner.read_at(abs, buf)
    }

    /// Forward metadata reads to the inner source at the absolute offset so the
    /// inner source's metadata cache is shared (chunk-index walks on a streaming
    /// userblock file otherwise re-read every node). Payload reads keep the default
    /// `read_exact_at`, which stays uncached so user data does not evict metadata.
    fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        let abs = offset
            .checked_add(self.base)
            .ok_or(FormatError::OffsetOverflow {
                offset,
                length: len as u64,
            })?;
        self.inner.read_metadata_at(abs, len)
    }
}

/// File-access options applied when opening an HDF5 file.
///
/// This is the `hdf5-pure` analogue of the HDF5 file access property list
/// settings relevant to read-time memory usage. The metadata cache only affects
/// streaming opens; in-memory opens already have the whole file in one buffer.
/// The chunk cache is the file-wide default corresponding to HDF5
/// `H5Pset_cache`'s raw-data chunk-cache settings and applies to datasets
/// opened from either backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FileAccessOptions {
    metadata_cache: MetadataCacheConfig,
    chunk_cache: ChunkCacheConfig,
}

impl FileAccessOptions {
    /// Create options with the crate's default access behavior.
    pub const fn new() -> Self {
        Self {
            metadata_cache: MetadataCacheConfig::disabled(),
            chunk_cache: ChunkCacheConfig::new(),
        }
    }

    /// Configure the bounded streaming metadata cache.
    pub const fn with_metadata_cache(mut self, metadata_cache: MetadataCacheConfig) -> Self {
        self.metadata_cache = metadata_cache;
        self
    }

    /// Configure the per-dataset raw chunk cache used by datasets opened from
    /// this file. This is the `H5Pset_cache`-style file-wide default.
    pub const fn with_chunk_cache(mut self, chunk_cache: ChunkCacheConfig) -> Self {
        self.chunk_cache = chunk_cache;
        self
    }

    /// Return the configured streaming metadata cache.
    pub const fn metadata_cache(&self) -> MetadataCacheConfig {
        self.metadata_cache
    }

    /// Return the configured per-dataset chunk cache.
    pub const fn chunk_cache(&self) -> ChunkCacheConfig {
        self.chunk_cache
    }
}

/// Dataset-access options applied when opening a single dataset.
///
/// This is the `hdf5-pure` analogue of an HDF5 Dataset Access Property List
/// (DAPL). Its chunk cache corresponds to `H5Pset_chunk_cache`: it overrides,
/// for this one dataset, the file-wide chunk-cache default configured with
/// [`FileAccessOptions::with_chunk_cache`] (the `H5Pset_cache` analogue). When
/// left unset, the dataset inherits that file-wide default — matching the DAPL
/// default sentinels (`H5D_CHUNK_CACHE_*_DEFAULT`), which also mean "use the
/// file's setting".
///
/// [`ChunkCacheConfig`] maps `H5Pset_chunk_cache`'s `rdcc_nslots` and
/// `rdcc_nbytes`; its `rdcc_w0` preemption policy is not modeled, because this
/// read cache uses strict LRU eviction (as noted on
/// [`ChunkCacheConfig::from_h5p_cache`]).
///
/// Pass it to [`File::dataset_with_options`] or [`Group::dataset_with_options`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DatasetAccessOptions {
    chunk_cache: Option<ChunkCacheConfig>,
}

impl DatasetAccessOptions {
    /// Create options that inherit every file-wide access default.
    pub const fn new() -> Self {
        Self { chunk_cache: None }
    }

    /// Override the raw chunk cache for this one dataset, ignoring the file-wide
    /// default. This is the `H5Pset_chunk_cache` analogue.
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
    access_options: FileAccessOptions,
    /// Set by [`File::close`] to seal a read-write file: after it, a write
    /// through any surviving [`Dataset`]/[`Group`] handle or [`File`] clone
    /// returns [`Error::FileClosed`]. Reads still work. Only ever set on a
    /// `Backend::Mirror` file.
    closed: AtomicBool,
    /// True for a file opened with [`File::open_swmr_writer`]: no OS lock is held,
    /// the superblock's SWMR-write flag is raised, only immediate
    /// [`Dataset::append`] is permitted (the staged surface is refused), and the
    /// flag is cleared on [`File::close`] / `Drop`. `false` for every other file.
    swmr_write: bool,
}

impl Drop for FileInner {
    /// Best-effort clear of the SWMR-write flag for a writer dropped without an
    /// explicit [`File::close`] (mirroring `SwmrWriter::drop`). Runs only when the
    /// last `Arc<FileInner>` clone drops; a clean `close` already cleared the flag
    /// and set `closed`, so this is idempotent and skipped in that case.
    fn drop(&mut self) {
        if self.swmr_write && !self.closed.load(Ordering::Acquire) {
            if let Backend::Mirror(m) = &self.backend {
                let mut session = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                let _ = session.set_consistency_flags(0);
            }
        }
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
        Self::open_with_options(path, FileAccessOptions::new())
    }

    /// Open an HDF5 file from a filesystem path with explicit access options.
    ///
    /// Like [`open`](Self::open), this buffers the whole file in memory. Use
    /// [`open_streaming_with_options`](Self::open_streaming_with_options) when
    /// the metadata cache budget should apply to lazy metadata reads.
    pub fn open_with_options<P: AsRef<std::path::Path>>(
        path: P,
        options: FileAccessOptions,
    ) -> Result<Self, Error> {
        let bytes = std::fs::read(path.as_ref()).map_err(Error::Io)?;
        Self::from_bytes_with_options(bytes, options)
    }

    /// Open an HDF5 file for **streaming** reads, fetching regions on demand from
    /// the file instead of buffering it whole.
    ///
    /// This lets a host read a file larger than its address space — the original
    /// motivation being 32-bit targets reading multi-gigabyte files (issue #27).
    /// Metadata and dataset chunks are read through a `ReadSeekSource`, so peak
    /// memory stays close to one chunk plus the metadata being parsed.
    ///
    /// Current limits (the buffered [`File::open`] has none of these): only
    /// latest-format (v2) groups resolve — a v1 symbol-table group on the path
    /// is rejected — and attribute reading on the streaming backend is not yet
    /// supported. Dataset reads (contiguous, compact, and all chunked index
    /// types) are fully supported.
    pub fn open_streaming<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        Self::open_streaming_with_options(path, FileAccessOptions::new())
    }

    /// Open an HDF5 file for streaming reads with explicit access options.
    pub fn open_streaming_with_options<P: AsRef<std::path::Path>>(
        path: P,
        options: FileAccessOptions,
    ) -> Result<Self, Error> {
        let handle = std::fs::File::open(path.as_ref()).map_err(Error::Io)?;
        let source = ReadSeekSource::new(handle).map_err(Error::Format)?;
        let source: Box<dyn FileSource + Send + Sync> = if options.metadata_cache.is_enabled() {
            Box::new(MetadataCachingSource::new(source, options.metadata_cache))
        } else {
            Box::new(source)
        };
        let (superblock, addr_offset) = Self::parse_superblock_source(source.as_ref())?;
        Ok(Self::from_parts(
            Backend::Streaming(source),
            superblock,
            addr_offset,
            None,
            options,
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
        Self::open_swmr_with_options(path, FileAccessOptions::new())
    }

    /// Open an HDF5 file for SWMR reading with explicit access options.
    ///
    /// SWMR reads currently keep an in-memory mirror for refresh semantics, so
    /// only the per-dataset chunk-cache settings affect this backend.
    pub fn open_swmr_with_options<P: AsRef<std::path::Path>>(
        path: P,
        options: FileAccessOptions,
    ) -> Result<Self, Error> {
        let mut handle = std::fs::File::open(path.as_ref()).map_err(Error::Io)?;
        let mut data = Vec::new();
        handle.read_to_end(&mut data).map_err(Error::Io)?;
        let (superblock, addr_offset) = Self::parse_superblock(&data)?;
        Ok(Self::from_parts(
            Backend::InMemory(data),
            superblock,
            addr_offset,
            Some(handle),
            options,
        ))
    }

    /// Open an HDF5 file from an in-memory byte vector.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self, Error> {
        Self::from_bytes_with_options(data, FileAccessOptions::new())
    }

    /// Open an HDF5 file from an in-memory byte vector with explicit access options.
    pub fn from_bytes_with_options(
        data: Vec<u8>,
        options: FileAccessOptions,
    ) -> Result<Self, Error> {
        let (superblock, addr_offset) = Self::parse_superblock(&data)?;
        Ok(Self::from_parts(
            Backend::InMemory(data),
            superblock,
            addr_offset,
            None,
            options,
        ))
    }

    /// Open an existing HDF5 file for reading **and** in-place editing, taking an
    /// exclusive OS file lock held for the file's life.
    fn open_rw<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        Self::from_rw_session(WriteEngine::open(path)?)
    }

    /// Like [`open_rw`](Self::open_rw), but with an explicit file-locking policy.
    fn open_rw_with_locking<P: AsRef<std::path::Path>>(
        path: P,
        locking: FileLocking,
    ) -> Result<Self, Error> {
        Self::from_rw_session(WriteEngine::open_with_locking(path, locking)?)
    }

    /// Wrap an opened [`WriteEngine`] as a read-write [`Backend::Mirror`] file.
    fn from_rw_session(session: WriteEngine) -> Result<Self, Error> {
        let (superblock, addr_offset) = Self::parse_superblock(session.mirror_bytes())?;
        Ok(Self::from_parts(
            Backend::Mirror(Box::new(Mutex::new(session))),
            superblock,
            addr_offset,
            None,
            FileAccessOptions::new(),
        ))
    }

    /// Open for SWMR writing: no OS lock, superblock SWMR-write flag raised.
    fn open_swmr_writer<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        let mut inner = Self::from_rw_session(WriteEngine::open_swmr_writer(path)?)?;
        inner.swmr_write = true;
        Ok(inner)
    }

    /// After the caller has confirmed a [`Backend::Mirror`] backend, gate the
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

    /// A `FileSource` view over the backend, for the streaming-capable paths.
    pub(crate) fn source(&self) -> SourceView<'_> {
        match &self.backend {
            Backend::InMemory(v) => SourceView::Mem(v),
            Backend::Streaming(s) => SourceView::Stream(s.as_ref()),
            // A mirror file's bytes live behind a lock and cannot be lent out as
            // a borrowed view; its read paths take the lock directly instead, so
            // this arm is never reached.
            Backend::Mirror(_) => SourceView::Mem(&[]),
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
    fn parse_superblock_source<S: FileSource + ?Sized>(
        source: &S,
    ) -> Result<(Superblock, u64), Error> {
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
        access_options: FileAccessOptions,
    ) -> Self {
        let mut file = FileInner {
            backend,
            superblock,
            addr_offset,
            handle,
            file_space_info: None,
            access_options,
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
            Backend::Mirror(m) => {
                let core = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                let data = core.mirror_bytes();
                // A staged commit can relocate the object tree's root, so the
                // cached superblock's root address may be stale; re-parse the
                // (small, fixed) superblock from the live mirror to resolve
                // against the committed root.
                let (sb, _base) = Self::parse_superblock(data)?;
                group_v2::resolve_path_any(data, &sb, path)?
            }
        })
    }

    /// The current root-group address (base-adjusted, absolute). For a read-write
    /// [`Backend::Mirror`] file a prior relocating commit can have moved the
    /// root, so re-parse the live mirror's superblock; other backends use the
    /// cached superblock. Falls back to the cached address if the re-parse fails.
    fn mirror_root_address(&self) -> u64 {
        if let Backend::Mirror(m) = &self.backend {
            let core = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Ok((sb, _base)) = Self::parse_superblock(core.mirror_bytes()) {
                return sb.root_group_address;
            }
        }
        self.superblock.root_group_address
    }

    /// Returns the raw file bytes for an in-memory file, or an empty slice for a
    /// streaming file (which has no whole-file buffer).
    pub fn as_bytes(&self) -> &[u8] {
        match &self.backend {
            Backend::InMemory(v) => v,
            // A streaming or mirror file has no borrowable whole-file buffer.
            Backend::Streaming(_) | Backend::Mirror(_) => &[],
        }
    }

    /// Return the access options used when opening this file.
    pub const fn access_options(&self) -> FileAccessOptions {
        self.access_options
    }

    /// Returns a reference to the parsed superblock.
    pub fn superblock(&self) -> &Superblock {
        &self.superblock
    }

    /// The whole-file byte image when this file is buffered in memory
    /// ([`open`](Self::open) / [`from_bytes`](Self::from_bytes)); `None` for a
    /// streaming file ([`open_streaming`](Self::open_streaming)). Cross-file
    /// object copy ([`EditSession::copy_from`](crate::EditSession::copy_from))
    /// uses this to read source objects by absolute address.
    pub(crate) fn in_memory_image(&self) -> Option<&[u8]> {
        match &self.backend {
            Backend::InMemory(data) => Some(data),
            Backend::Streaming(_) | Backend::Mirror(_) => None,
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
            Backend::Mirror(m) => {
                let core = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                core.mirror_bytes().len() as u64
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
            Backend::Mirror(m) => {
                let core = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                ObjectHeader::parse_with_base(
                    core.mirror_bytes(),
                    address.to_usize()?,
                    os,
                    ls,
                    self.addr_offset,
                )
            }
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
            let chunk_cache =
                DatasetAccessOptions::new().resolved_chunk_cache(file.access_options.chunk_cache);
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
            Backend::Mirror(m) => {
                let core = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                group_v2::resolve_group_entries(core.mirror_bytes(), hdr, os, ls, base)
            }
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

    /// Read all attributes attached to an object header, dispatching on the
    /// backend.
    fn attrs_of(&self, hdr: &ObjectHeader) -> Result<HashMap<String, AttrValue>, Error> {
        let (os, ls, base) = (self.offset_size(), self.length_size(), self.addr_offset);
        let attr_msgs = self.attr_messages_of(hdr)?;
        match &self.backend {
            Backend::Mirror(m) => {
                let core = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                Ok(attrs_to_map(
                    &attr_msgs,
                    &BytesSource::new(core.mirror_bytes()),
                    os,
                    ls,
                    base,
                ))
            }
            _ => Ok(attrs_to_map(&attr_msgs, &self.source(), os, ls, base)),
        }
    }

    /// Names of every attribute message on `hdr`, including ones whose datatype
    /// [`attrs_of`](Self::attrs_of) cannot decode into an [`AttrValue`] (and so
    /// silently omits from its map). Repack diffs this against the decoded map to
    /// refuse rather than drop an attribute it cannot reproduce.
    pub(crate) fn attr_message_names_of(&self, hdr: &ObjectHeader) -> Result<Vec<String>, Error> {
        Ok(self
            .attr_messages_of(hdr)?
            .into_iter()
            .map(|a| a.name)
            .collect())
    }

    /// Extract every attribute message attached to an object header (compact,
    /// shared, and dense storage), dispatching on the backend.
    fn attr_messages_of(
        &self,
        hdr: &ObjectHeader,
    ) -> Result<Vec<crate::attribute::AttributeMessage>, Error> {
        let (os, ls) = (self.offset_size(), self.length_size());
        match &self.backend {
            Backend::InMemory(v) => Ok(extract_attributes_full(v, hdr, os, ls)?),
            Backend::Streaming(s) => Ok(extract_attributes_full_from_source(
                s.as_ref(),
                hdr,
                os,
                ls,
            )?),
            Backend::Mirror(m) => {
                let core = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                Ok(extract_attributes_full(core.mirror_bytes(), hdr, os, ls)?)
            }
        }
    }

    /// Read a dataset's raw bytes for the given layout, dispatching on the backend.
    fn read_dataset_raw(
        &self,
        dl: &DataLayout,
        ds: &Dataspace,
        dt: &Datatype,
        pipeline: Option<&FilterPipeline>,
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
                let frame = if base == 0 {
                    v.as_slice()
                } else {
                    let start = base.to_usize()?;
                    v.get(start..).ok_or(FormatError::UnexpectedEof {
                        expected: start,
                        available: v.len(),
                    })?
                };
                data_read::read_raw_data_cached(frame, dl, ds, dt, pipeline, os, ls, cache)
            }
            Backend::Streaming(s) if base == 0 => data_read::read_raw_data_cached_from_source(
                s.as_ref(),
                dl,
                ds,
                dt,
                pipeline,
                os,
                ls,
                cache,
            ),
            Backend::Streaming(s) => {
                let framed = BaseOffsetSource {
                    inner: s.as_ref(),
                    base,
                };
                data_read::read_raw_data_cached_from_source(
                    &framed, dl, ds, dt, pipeline, os, ls, cache,
                )
            }
            Backend::Mirror(m) => {
                let core = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                let data = core.mirror_bytes();
                let frame = if base == 0 {
                    data
                } else {
                    let start = base.to_usize()?;
                    data.get(start..).ok_or(FormatError::UnexpectedEof {
                        expected: start,
                        available: data.len(),
                    })?
                };
                data_read::read_raw_data_cached(frame, dl, ds, dt, pipeline, os, ls, cache)
            }
        }
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
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::open(path)?),
        })
    }

    /// Open an HDF5 file from a filesystem path with explicit access options.
    pub fn open_with_options<P: AsRef<std::path::Path>>(
        path: P,
        options: FileAccessOptions,
    ) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::open_with_options(path, options)?),
        })
    }

    /// Open an HDF5 file for **streaming** reads, fetching regions on demand from
    /// the file instead of buffering it whole.
    ///
    /// This lets a host read a file larger than its address space. Metadata and
    /// dataset chunks are read through a `ReadSeekSource`, so peak memory stays
    /// close to one chunk plus the metadata being parsed. Attribute reading and
    /// v1 symbol-table groups on the resolved path are not yet supported on this
    /// backend.
    pub fn open_streaming<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::open_streaming(path)?),
        })
    }

    /// Open an HDF5 file for streaming reads with explicit access options.
    pub fn open_streaming_with_options<P: AsRef<std::path::Path>>(
        path: P,
        options: FileAccessOptions,
    ) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::open_streaming_with_options(path, options)?),
        })
    }

    /// Open an HDF5 file for SWMR (single-writer/multiple-reader) reading.
    ///
    /// Like [`File::open`], but retains a live handle to the file so that
    /// [`File::refresh`] can re-read data appended by a concurrent writer.
    pub fn open_swmr<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::open_swmr(path)?),
        })
    }

    /// Open an HDF5 file for SWMR reading with explicit access options.
    pub fn open_swmr_with_options<P: AsRef<std::path::Path>>(
        path: P,
        options: FileAccessOptions,
    ) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::open_swmr_with_options(path, options)?),
        })
    }

    /// Open an HDF5 file from an in-memory byte vector.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::from_bytes(data)?),
        })
    }

    /// Open an HDF5 file from an in-memory byte vector with explicit access options.
    pub fn from_bytes_with_options(
        data: Vec<u8>,
        options: FileAccessOptions,
    ) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::from_bytes_with_options(data, options)?),
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
    pub fn open_rw<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::open_rw(path)?),
        })
    }

    /// Open an existing file for reading and in-place editing with an explicit
    /// file-locking policy — the owned-handle counterpart of HDF5's
    /// `H5Pset_file_locking`.
    ///
    /// [`open_rw`](Self::open_rw) takes an exclusive OS lock for the file's life;
    /// use this with [`FileLocking::Disabled`](crate::FileLocking) only when an
    /// external mechanism already guarantees single-writer access, or on a
    /// filesystem (such as some network mounts) where the OS lock is
    /// unavailable. Setting `HDF5_USE_FILE_LOCKING` in the environment overrides
    /// the requested policy, as in the C library.
    pub fn open_rw_with_locking<P: AsRef<std::path::Path>>(
        path: P,
        locking: FileLocking,
    ) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::open_rw_with_locking(path, locking)?),
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
    /// [`clear_swmr_flag`](Self::clear_swmr_flag).
    ///
    /// Requires a latest-format (version-2/3 superblock) file with no userblock
    /// and no persisted free-space; other files are refused with
    /// [`Error::SwmrAppendUnsupported`](crate::Error::SwmrAppendUnsupported).
    pub fn open_swmr_writer<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::open_swmr_writer(path)?),
        })
    }

    /// Clear a stale SWMR-write flag left in `path` by a writer that exited
    /// without a clean [`close`](Self::close) — the `h5clear -s` equivalent, for
    /// recovering a file the reference C library then refuses to open. A no-op if
    /// the flag is already clear.
    pub fn clear_swmr_flag<P: AsRef<std::path::Path>>(path: P) -> Result<(), Error> {
        crate::swmr_writer::SwmrWriter::clear_swmr_flag(path)
    }

    /// Create a new, empty HDF5 file at `path` and open it for reading and
    /// writing, so its contents can be built entirely through owned handles
    /// ([`Group::create_dataset`]/[`create_group`](Group::create_group), then
    /// [`commit`](Self::commit)).
    ///
    /// Overwrites any existing file at `path`. For an all-at-once write, use
    /// [`FileBuilder`](crate::FileBuilder) instead.
    pub fn create<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        let bytes = crate::writer::FileBuilder::new().finish()?;
        std::fs::write(path.as_ref(), bytes).map_err(Error::Io)?;
        Self::open_rw(path)
    }

    /// Apply all staged structural edits made through this file's handles —
    /// [`Dataset::write`]/`set_attr`/`remove_attr` and
    /// [`Group::create_group`]/`delete` — as one transaction. Immediate
    /// [`Dataset::append`]s need no commit.
    ///
    /// Requires a read-write file ([`File::open_rw`]); a read-only file returns
    /// [`Error::ReadOnly`](crate::Error::ReadOnly). A commit that relocates
    /// objects invalidates outstanding handles — re-fetch any you keep using.
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
            session.copy(&normalize_path(src), &normalize_path(dst));
            Ok(())
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
            Backend::Mirror(m) => {
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
            Backend::Mirror(m) => {
                let session = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                Ok(session.space_accounting())
            }
            _ => Err(Error::ReadOnly),
        }
    }

    /// Commit any staged edits and seal this file. The exclusive OS lock is
    /// released once the last handle derived from this file is also dropped.
    ///
    /// After `close`, a write through any surviving [`Dataset`]/[`Group`] handle
    /// or [`File`] clone returns [`Error::FileClosed`](crate::Error::FileClosed);
    /// reads still work.
    pub fn close(self) -> Result<(), Error> {
        if matches!(self.inner.backend, Backend::Mirror(_)) {
            if self.inner.swmr_write {
                // SWMR mode stages nothing (the staged surface is refused), so do
                // not commit — clear the SWMR-write flag and flush, marking the
                // file cleanly closed for any concurrent reader.
                if let Backend::Mirror(m) = &self.inner.backend {
                    let mut session = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                    session.set_consistency_flags(0)?;
                }
            } else {
                self.commit()?;
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
        let Backend::Mirror(m) = &self.inner.backend else {
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
    /// [`FileAccessOptions::with_chunk_cache`]). To override the cache for this
    /// one dataset, use [`dataset_with_options`](Self::dataset_with_options).
    pub fn dataset(&self, path: &str) -> Result<Dataset, Error> {
        self.dataset_with_options(path, DatasetAccessOptions::new())
    }

    /// Resolve a path and return an owned [`Dataset`] handle, applying per-dataset
    /// [`DatasetAccessOptions`] that override file-wide access defaults.
    ///
    /// This is the dataset-open-with-access-property-list path (HDF5's DAPL):
    /// the options' chunk cache corresponds to `H5Pset_chunk_cache` and takes
    /// precedence, for this dataset only, over the `H5Pset_cache`-style
    /// file-wide default.
    pub fn dataset_with_options(
        &self,
        path: &str,
        options: DatasetAccessOptions,
    ) -> Result<Dataset, Error> {
        let addr = self.inner.resolve_path(path)?;
        let hdr = self.inner.parse_header(addr)?;
        if !has_message(&hdr, MessageType::DataLayout) {
            return Err(Error::NotADataset(path.to_string()));
        }
        let chunk_cache = options.resolved_chunk_cache(self.inner.access_options.chunk_cache);
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

    /// Return the access options used when opening this file.
    pub fn access_options(&self) -> FileAccessOptions {
        self.inner.access_options()
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

    /// A `FileSource` view over the backend, for the streaming-capable paths.
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

    /// List the names of subgroups in this group.
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

    /// Read all attributes of this group.
    pub fn attrs(&self) -> Result<HashMap<String, AttrValue>, Error> {
        let hdr = self.file.parse_header(self.address)?;
        self.file.attrs_of(&hdr)
    }

    /// Names of every attribute on this group, including any whose datatype
    /// [`attrs`](Self::attrs) cannot represent. Used by repack to detect an
    /// attribute it would otherwise drop.
    pub(crate) fn attr_names(&self) -> Result<Vec<String>, Error> {
        let hdr = self.file.parse_header(self.address)?;
        self.file.attr_message_names_of(&hdr)
    }

    /// Get a dataset within this group by name.
    ///
    /// The dataset uses the file-wide chunk-cache default. To override the cache
    /// for this one dataset, use
    /// [`dataset_with_options`](Self::dataset_with_options).
    pub fn dataset(&self, name: &str) -> Result<Dataset, Error> {
        self.dataset_with_options(name, DatasetAccessOptions::new())
    }

    /// Get a dataset within this group by name, applying per-dataset
    /// [`DatasetAccessOptions`] that override file-wide access defaults (HDF5's
    /// DAPL; see `H5Pset_chunk_cache`).
    pub fn dataset_with_options(
        &self,
        name: &str,
        options: DatasetAccessOptions,
    ) -> Result<Dataset, Error> {
        let entries = self.children()?;
        let entry = entries
            .iter()
            .find(|e| e.name == name)
            .ok_or_else(|| Error::Format(FormatError::PathNotFound(name.to_string())))?;
        let hdr = self.file.parse_header(entry.object_header_address)?;
        if !has_message(&hdr, MessageType::DataLayout) {
            return Err(Error::NotADataset(name.to_string()));
        }
        let chunk_cache = options.resolved_chunk_cache(self.file.access_options.chunk_cache);
        Ok(Dataset {
            file: self.file.clone(),
            address: entry.object_header_address,
            header: hdr,
            chunk_cache: ChunkCache::with_config(chunk_cache),
            chunk_cache_config: chunk_cache,
            path: self.child_path(name),
        })
    }

    /// Get a subgroup within this group by name.
    pub fn group(&self, name: &str) -> Result<Group, Error> {
        let entries = self.children()?;
        let entry = entries
            .iter()
            .find(|e| e.name == name)
            .ok_or_else(|| Error::Format(FormatError::PathNotFound(name.to_string())))?;
        Ok(Group {
            file: self.file.clone(),
            address: entry.object_header_address,
            path: self.child_path(name),
        })
    }

    /// The root-relative path of a child named `name`, or `None` if this group
    /// itself has no resolvable path (reached by object reference).
    fn child_path(&self, name: &str) -> Option<String> {
        self.path.as_ref().map(|p| {
            if p.is_empty() {
                name.to_string()
            } else {
                format!("{p}/{name}")
            }
        })
    }

    /// Create a subgroup `name` within this group, staged until [`File::commit`].
    ///
    /// Requires a read-write file ([`File::open_rw`]), else
    /// [`Error::ReadOnly`](crate::Error::ReadOnly).
    pub fn create_group(&self, name: &str) -> Result<(), Error> {
        self.with_child_session(name, |session, child| {
            session.create_group(child);
            Ok(())
        })
    }

    /// Create a dataset `name` within this group, configuring it through `build`
    /// (shape, data, chunks, filters, …), staged until [`File::commit`].
    ///
    /// Requires a read-write file ([`File::open_rw`]), else
    /// [`Error::ReadOnly`](crate::Error::ReadOnly).
    pub fn create_dataset(
        &self,
        name: &str,
        build: impl FnOnce(&mut DatasetBuilder),
    ) -> Result<(), Error> {
        self.with_child_session(name, |session, child| {
            build(session.create_dataset(child));
            Ok(())
        })
    }

    /// Delete the object named `name` from this group, staged until
    /// [`File::commit`]. See [`create_group`](Self::create_group) for the
    /// file-mode rules.
    pub fn delete(&self, name: &str) -> Result<(), Error> {
        self.with_child_session(name, |session, child| {
            session.delete(child);
            Ok(())
        })
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
        self.with_own_session(|session, path| {
            session.set_group_attr(path, name, value);
            Ok(())
        })
    }

    /// Remove a compact attribute from this group, staged until [`File::commit`].
    /// See [`set_attr`](Self::set_attr) for the file-mode rules.
    pub fn remove_attr(&self, name: &str) -> Result<(), Error> {
        self.with_own_session(|session, path| {
            session.remove_group_attr(path, name);
            Ok(())
        })
    }

    /// Run `f` with the writable session and the root-relative path of child
    /// `name`. Returns [`Error::ReadOnly`](crate::Error::ReadOnly) if the file is
    /// read-only or this group has no resolvable path.
    fn with_child_session<R>(
        &self,
        name: &str,
        f: impl FnOnce(&mut WriteEngine, &str) -> Result<R, Error>,
    ) -> Result<R, Error> {
        let Backend::Mirror(m) = &self.file.backend else {
            return Err(Error::ReadOnly);
        };
        self.file.check_mutable(true)?;
        let child = self.child_path(name).ok_or(Error::ReadOnly)?;
        let mut session = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        f(&mut session, &child)
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
        let Backend::Mirror(m) = &self.file.backend else {
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
    /// The file must have been opened for writing with [`File::open_rw`]; a
    /// read-only file (or a handle reached by object reference, which has no
    /// path) returns [`Error::ReadOnly`](crate::Error::ReadOnly). The target must
    /// be a chunked, rank-1, unlimited, Extensible-Array-indexed dataset — the
    /// same contract as [`AppendWriter`](crate::AppendWriter), including filtered
    /// whole-chunk / unfiltered any-length rules — otherwise
    /// [`Error::AppendUnsupported`](crate::Error::AppendUnsupported) is returned.
    /// The append is immediate and crash-atomic (no `commit` needed).
    pub fn append<T: H5Element>(&mut self, data: &[T]) -> Result<(), Error> {
        self.with_session_mut(false, |session, path| session.append_inplace(path, data))
    }

    /// Append raw little-endian element bytes to this dataset in place. Prefer
    /// [`append`](Self::append) when the element type is known; see it for the
    /// file-mode and eligibility rules.
    pub fn append_raw(&mut self, bytes: &[u8]) -> Result<(), Error> {
        self.with_session_mut(false, |session, path| {
            session.append_inplace_raw(path, bytes)
        })
    }

    /// Overwrite this dataset's values, staged until [`File::commit`]. The new
    /// data must match the dataset's existing shape and datatype.
    ///
    /// The file must have been opened with [`File::open_rw`], else
    /// [`Error::ReadOnly`](crate::Error::ReadOnly). Unlike [`append`](Self::append)
    /// (immediate), this is a staged edit applied on [`File::commit`].
    pub fn write<T: H5Element>(&mut self, data: &[T]) -> Result<(), Error> {
        self.with_session_mut(true, |session, path| {
            let builder = session.write_dataset(path);
            T::write_into(builder, data);
            Ok(())
        })
    }

    /// Stage an append to this dataset applied on [`File::commit`] — the staged,
    /// index-rebuilding counterpart of the immediate [`append`](Self::append).
    ///
    /// Unlike [`append`](Self::append) (immediate, amortized `O(1)`,
    /// Extensible-Array only, unfiltered any-length / filtered whole-chunk), this
    /// rebuilds the chunk index on commit and so also grows **filtered** datasets
    /// by any length (a trailing partial chunk is rewritten) and datasets whose
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
    pub fn append_staged(&mut self, build: impl FnOnce(&mut AppendBuilder)) -> Result<(), Error> {
        self.with_session_mut(true, |session, path| {
            build(session.append_dataset(path));
            Ok(())
        })
    }

    /// Add or update a compact attribute on this dataset, staged until
    /// [`File::commit`]. Use [`remove_attr`](Self::remove_attr) to remove one.
    ///
    /// The file must have been opened with [`File::open_rw`], else
    /// [`Error::ReadOnly`](crate::Error::ReadOnly).
    pub fn set_attr(&mut self, name: &str, value: AttrValue) -> Result<(), Error> {
        self.with_session_mut(true, |session, path| {
            session.set_dataset_attr(path, name, value);
            Ok(())
        })
    }

    /// Remove a compact attribute from this dataset, staged until
    /// [`File::commit`]. See [`set_attr`](Self::set_attr) for the file-mode rules.
    pub fn remove_attr(&mut self, name: &str) -> Result<(), Error> {
        self.with_session_mut(true, |session, path| {
            session.remove_dataset_attr(path, name);
            Ok(())
        })
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
        let Backend::Mirror(m) = &self.file.backend else {
            return Err(Error::ReadOnly);
        };
        self.file.check_mutable(staged)?;
        let path = self.path.clone().ok_or(Error::ReadOnly)?;
        let out = {
            let mut session = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            f(&mut session, &path)?
        };
        self.header = self.file.parse_header(self.address)?;
        Ok(out)
    }

    /// The effective raw chunk-cache configuration for this dataset.
    ///
    /// This reflects the per-dataset [`DatasetAccessOptions`] override when one
    /// was supplied to [`File::dataset_with_options`] /
    /// [`Group::dataset_with_options`], otherwise the file-wide default. It is
    /// the read-side analogue of HDF5's `H5Pget_chunk_cache`.
    pub const fn chunk_cache_config(&self) -> ChunkCacheConfig {
        self.chunk_cache_config
    }

    /// A point-in-time snapshot of this dataset handle's chunk-cache occupancy.
    ///
    /// Lets callers confirm a chunk-cache configuration (set with
    /// [`FileAccessOptions::with_chunk_cache`]) is taking effect: after a
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
    /// [`EditSession::append_dataset`](crate::EditSession::append_dataset)
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
                m.msg_type, &m.data,
            )?),
            None => Ok(None),
        }
    }

    /// Read all data as `f64` values.
    pub fn read_f64(&self) -> Result<Vec<f64>, Error> {
        let raw = self.read_raw()?;
        let dt = self.datatype()?;
        Ok(data_read::read_as_f64(&raw, &dt)?)
    }

    /// Read all data as `f32` values.
    pub fn read_f32(&self) -> Result<Vec<f32>, Error> {
        let raw = self.read_raw()?;
        let dt = self.datatype()?;
        Ok(data_read::read_as_f32(&raw, &dt)?)
    }

    /// Read all data as `i32` values.
    pub fn read_i32(&self) -> Result<Vec<i32>, Error> {
        let raw = self.read_raw()?;
        let dt = self.datatype()?;
        Ok(data_read::read_as_i32(&raw, &dt)?)
    }

    /// Read all data as `i64` values.
    pub fn read_i64(&self) -> Result<Vec<i64>, Error> {
        let raw = self.read_raw()?;
        let dt = self.datatype()?;
        Ok(data_read::read_as_i64(&raw, &dt)?)
    }

    /// Read all data as `u64` values.
    pub fn read_u64(&self) -> Result<Vec<u64>, Error> {
        let raw = self.read_raw()?;
        let dt = self.datatype()?;
        Ok(data_read::read_as_u64(&raw, &dt)?)
    }

    /// Read all data as `u8` values.
    pub fn read_u8(&self) -> Result<Vec<u8>, Error> {
        self.read_raw()
    }

    /// Read all data as `i8` values.
    #[expect(
        clippy::cast_possible_wrap,
        reason = "read_i8 reinterprets each stored byte as the signed i8 the caller requested"
    )]
    pub fn read_i8(&self) -> Result<Vec<i8>, Error> {
        let raw = self.read_raw()?;
        Ok(raw.iter().map(|&b| b as i8).collect())
    }

    /// Read all data as `i16` values.
    pub fn read_i16(&self) -> Result<Vec<i16>, Error> {
        let raw = self.read_raw()?;
        let dt = self.datatype()?;
        Ok(data_read::read_as_i16(&raw, &dt)?)
    }

    /// Read all data as `u16` values.
    pub fn read_u16(&self) -> Result<Vec<u16>, Error> {
        let raw = self.read_raw()?;
        let dt = self.datatype()?;
        Ok(data_read::read_as_u16(&raw, &dt)?)
    }

    /// Read all data as `u32` values.
    pub fn read_u32(&self) -> Result<Vec<u32>, Error> {
        let raw = self.read_raw()?;
        let dt = self.datatype()?;
        Ok(data_read::read_as_u32(&raw, &dt)?)
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
        let source = self.file.source();
        Ok(vl_data::visit_vl_strings_from_source(
            &source,
            &raw,
            dataspace.num_elements(),
            self.file.offset_size(),
            self.file.length_size(),
            self.file.addr_offset,
            options,
            visitor,
        )?)
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
        let source = self.file.source();
        Ok(vl_data::read_vl_byte_objects_from_source(
            &source,
            &raw,
            dataspace.num_elements(),
            self.file.offset_size(),
            self.file.length_size(),
            self.file.addr_offset,
            1, // a VL string's base type is a single byte
            options,
        )?)
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
        let source = self.file.source();
        let objects = vl_data::read_vl_byte_objects_from_source(
            &source,
            &raw,
            dataspace.num_elements(),
            self.file.offset_size(),
            self.file.length_size(),
            self.file.addr_offset,
            element_size,
            options,
        )?;
        Ok((objects, element_size))
    }

    /// Read all attributes of this dataset.
    pub fn attrs(&self) -> Result<HashMap<String, AttrValue>, Error> {
        self.file.attrs_of(&self.header)
    }

    /// Names of every attribute on this dataset, including any whose datatype
    /// [`attrs`](Self::attrs) cannot represent. Used by repack to detect an
    /// attribute it would otherwise drop.
    pub(crate) fn attr_names(&self) -> Result<Vec<String>, Error> {
        self.file.attr_message_names_of(&self.header)
    }

    /// Returns the exact HDF5 datatype, including compound field offsets and
    /// total record size.
    pub fn datatype(&self) -> Result<Datatype, Error> {
        let msg = find_message(&self.header, MessageType::Datatype)?;
        let (dt, _) = Datatype::parse(&msg.data)?;
        Ok(dt)
    }

    pub(crate) fn dataspace(&self) -> Result<Dataspace, Error> {
        let msg = find_message(&self.header, MessageType::Dataspace)?;
        Ok(Dataspace::parse(&msg.data, self.file.length_size())?)
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
        self.header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::FilterPipeline)
            .and_then(|msg| FilterPipeline::parse(&msg.data).ok())
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
        let elem_size = self.datatype()?.type_size() as usize;
        let base = self.file.addr_offset;
        let source = self.file.source();
        // The chunk index — its root at `addr` and every internal node — stores
        // addresses relative to the base address. Walk it through a base-relative
        // view so those resolve, then shift each returned chunk address back to an
        // absolute file offset, since callers (repack) read the chunk bytes from
        // the full file source.
        if base == 0 {
            return Ok(crate::chunked_read::collect_chunks_for_layout_from_source(
                &source,
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
            inner: &source,
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
            c.address =
                c.address
                    .checked_add(base)
                    .ok_or(crate::error::FormatError::OffsetOverflow {
                        offset: c.address,
                        length: 0,
                    })?;
        }
        Ok(chunks)
    }

    /// The raw `FilterPipeline` message bytes from this dataset's object header,
    /// if it has one. Repack reuses this verbatim so that every filter — including
    /// ones this crate cannot itself apply (ZFP, SZIP, unknown) — is reproduced
    /// byte-for-byte in the repacked file's pipeline message.
    pub(crate) fn filter_pipeline_message_bytes(&self) -> Option<Vec<u8>> {
        self.header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::FilterPipeline)
            .map(|msg| msg.data.clone())
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
        Ok(self
            .file
            .read_dataset_raw(&dl, &ds, &dt, pipeline.as_ref(), &self.chunk_cache)?)
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
        let element_size = datatype.type_size().to_usize()?;
        if !matches!(datatype, Datatype::Compound { .. }) {
            return Err(FormatError::TypeMismatch {
                expected: "Compound",
                actual: "non-Compound",
            }
            .into());
        }
        let raw = self.read_raw()?;
        if element_size == 0 || !raw.len().is_multiple_of(element_size) {
            return Err(FormatError::DataSizeMismatch {
                expected: element_size,
                actual: raw.len(),
            }
            .into());
        }
        raw.chunks_exact(element_size)
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
        let stored = match attrs.get(ATTR_SHA256) {
            Some(AttrValue::String(s) | AttrValue::AsciiString(s)) => {
                s.trim_end_matches('\0').to_string()
            }
            _ => return Ok(VerifyResult::NoHash),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FileBuilder;

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
            FileAccessOptions::new().with_chunk_cache(ChunkCacheConfig::disabled()),
        )
        .unwrap();

        let ds = file
            .dataset_with_options(
                "chunked",
                DatasetAccessOptions::new().with_chunk_cache(ChunkCacheConfig::new()),
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
            FileAccessOptions::new().with_chunk_cache(ChunkCacheConfig::new()),
        )
        .unwrap();

        let ds = file
            .dataset_with_options(
                "chunked",
                DatasetAccessOptions::new().with_chunk_cache(ChunkCacheConfig::disabled()),
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
}
