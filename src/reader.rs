//! Reading API: File, Dataset, and Group handles for reading HDF5 files.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

use crate::attribute::extract_attributes_full;
use crate::chunk_cache::{ChunkCache, ChunkCacheConfig, ChunkCacheStats};
use crate::compound::CompoundType;
use crate::convert::TryToUsize;
use crate::data_layout::DataLayout;
use crate::data_read;
use crate::dataspace::Dataspace;
use crate::datatype::Datatype;
use crate::error::{Error, FormatError};
use crate::file_space_info::{FileSpaceInfo, FileSpaceStrategy};
use crate::filter_pipeline::FilterPipeline;
use crate::free_space_manager;
use crate::group_v1::GroupEntry;
use crate::group_v2;
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
}

/// A borrowed `FileSource` view over a [`File`]'s backend, used by the
/// streaming-capable read paths so one call site serves both backends.
enum SourceView<'a> {
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
pub struct File {
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
}

impl File {
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
    /// Metadata and dataset chunks are read through a [`ReadSeekSource`], so peak
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

    /// A `FileSource` view over the backend, for the streaming-capable paths.
    fn source(&self) -> SourceView<'_> {
        match &self.backend {
            Backend::InMemory(v) => SourceView::Mem(v),
            Backend::Streaming(s) => SourceView::Stream(s.as_ref()),
        }
    }

    /// Parse the superblock from `data`, returning it (with `root_group_address`
    /// normalized to an absolute offset) and the base-address offset.
    fn parse_superblock(data: &[u8]) -> Result<(Superblock, u64), Error> {
        let sig_offset = signature::find_signature(data)?;
        let mut superblock = Superblock::parse(data, sig_offset)?;
        let addr_offset = superblock.base_address;
        // Normalize root_group_address to absolute so resolve_path_any works.
        superblock.root_group_address += addr_offset;
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
        superblock.root_group_address += addr_offset;
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
        let mut file = File {
            backend,
            superblock,
            addr_offset,
            handle,
            file_space_info: None,
            access_options,
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

    /// Returns a handle to the root group.
    pub fn root(&self) -> Group<'_> {
        Group {
            file: self,
            // root_group_address was normalized to absolute in from_bytes()
            address: self.superblock.root_group_address,
        }
    }

    /// Resolve a path to an object-header address, dispatching on the backend.
    fn resolve_path(&self, path: &str) -> Result<u64, Error> {
        Ok(match &self.backend {
            Backend::InMemory(v) => group_v2::resolve_path_any(v, &self.superblock, path)?,
            Backend::Streaming(s) => {
                group_v2::resolve_path_any_from_source(s.as_ref(), &self.superblock, path)?
            }
        })
    }

    /// Resolve a path and return a `Dataset` handle.
    ///
    /// The dataset uses the file-wide chunk-cache default (configured with
    /// [`FileAccessOptions::with_chunk_cache`]). To override the cache for this
    /// one dataset, use [`dataset_with_options`](Self::dataset_with_options).
    pub fn dataset(&self, path: &str) -> Result<Dataset<'_>, Error> {
        self.dataset_with_options(path, DatasetAccessOptions::new())
    }

    /// Resolve a path and return a `Dataset` handle, applying per-dataset
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
    ) -> Result<Dataset<'_>, Error> {
        let addr = self.resolve_path(path)?;
        let hdr = self.parse_header(addr)?;
        if !has_message(&hdr, MessageType::DataLayout) {
            return Err(Error::NotADataset(path.to_string()));
        }
        let chunk_cache = options.resolved_chunk_cache(self.access_options.chunk_cache);
        Ok(Dataset {
            file: self,
            header: hdr,
            chunk_cache: ChunkCache::with_config(chunk_cache),
            chunk_cache_config: chunk_cache,
        })
    }

    /// Resolve a path and return a `Group` handle.
    pub fn group(&self, path: &str) -> Result<Group<'_>, Error> {
        let addr = self.resolve_path(path)?;
        Ok(Group {
            file: self,
            address: addr,
        })
    }

    /// Returns the raw file bytes for an in-memory file, or an empty slice for a
    /// streaming file (which has no whole-file buffer).
    pub fn as_bytes(&self) -> &[u8] {
        match &self.backend {
            Backend::InMemory(v) => v,
            Backend::Streaming(_) => &[],
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
    /// [`Superblock::eof_address`](crate::superblock::Superblock) (reachable via
    /// [`File::superblock`]) to detect appended or unaccounted tail bytes.
    pub fn file_size(&self) -> u64 {
        self.source().len()
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
                group_v2::resolve_group_entries_from_source(s.as_ref(), hdr, os, ls)
            }
        }
        .map_err(Error::Format)?;
        for entry in &mut entries {
            entry.object_header_address += base;
        }
        Ok(entries)
    }

    /// Read all attributes attached to an object header, dispatching on the
    /// backend. Attribute reading is not yet supported on the streaming backend.
    fn attrs_of(&self, hdr: &ObjectHeader) -> Result<HashMap<String, AttrValue>, Error> {
        match &self.backend {
            Backend::InMemory(v) => {
                let attr_msgs =
                    extract_attributes_full(v, hdr, self.offset_size(), self.length_size())?;
                Ok(attrs_to_map(
                    &attr_msgs,
                    v,
                    self.offset_size(),
                    self.length_size(),
                    self.addr_offset,
                ))
            }
            Backend::Streaming(_) => Err(Error::Format(FormatError::ChunkedReadError(
                "attribute reading is not yet supported on the streaming backend".into(),
            ))),
        }
    }

    /// Names of every attribute message on `hdr`, including ones whose datatype
    /// [`attrs_of`](Self::attrs_of) cannot decode into an [`AttrValue`] (and so
    /// silently omits from its map). Repack diffs this against the decoded map to
    /// refuse rather than drop an attribute it cannot reproduce.
    pub(crate) fn attr_message_names_of(&self, hdr: &ObjectHeader) -> Result<Vec<String>, Error> {
        match &self.backend {
            Backend::InMemory(v) => {
                let attr_msgs =
                    extract_attributes_full(v, hdr, self.offset_size(), self.length_size())?;
                Ok(attr_msgs.into_iter().map(|a| a.name).collect())
            }
            Backend::Streaming(_) => Err(Error::Format(FormatError::ChunkedReadError(
                "attribute reading is not yet supported on the streaming backend".into(),
            ))),
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
        match &self.backend {
            Backend::InMemory(v) => {
                data_read::read_raw_data_cached(v, dl, ds, dt, pipeline, os, ls, cache)
            }
            Backend::Streaming(s) => data_read::read_raw_data_cached_from_source(
                s.as_ref(),
                dl,
                ds,
                dt,
                pipeline,
                os,
                ls,
                cache,
            ),
        }
    }
}

impl std::fmt::Debug for File {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("File")
            .field("size", &self.source().len())
            .field("superblock_version", &self.superblock.version)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Group handle
// ---------------------------------------------------------------------------

/// A lightweight handle to an HDF5 group.
pub struct Group<'f> {
    file: &'f File,
    address: u64,
}

impl<'f> Group<'f> {
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
    pub fn dataset(&self, name: &str) -> Result<Dataset<'f>, Error> {
        self.dataset_with_options(name, DatasetAccessOptions::new())
    }

    /// Get a dataset within this group by name, applying per-dataset
    /// [`DatasetAccessOptions`] that override file-wide access defaults (HDF5's
    /// DAPL; see `H5Pset_chunk_cache`).
    pub fn dataset_with_options(
        &self,
        name: &str,
        options: DatasetAccessOptions,
    ) -> Result<Dataset<'f>, Error> {
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
            file: self.file,
            header: hdr,
            chunk_cache: ChunkCache::with_config(chunk_cache),
            chunk_cache_config: chunk_cache,
        })
    }

    /// Get a subgroup within this group by name.
    pub fn group(&self, name: &str) -> Result<Group<'f>, Error> {
        let entries = self.children()?;
        let entry = entries
            .iter()
            .find(|e| e.name == name)
            .ok_or_else(|| Error::Format(FormatError::PathNotFound(name.to_string())))?;
        Ok(Group {
            file: self.file,
            address: entry.object_header_address,
        })
    }

    fn children(&self) -> Result<Vec<GroupEntry>, Error> {
        let hdr = self.file.parse_header(self.address)?;
        self.file.group_children(&hdr)
    }
}

// ---------------------------------------------------------------------------
// Dataset handle
// ---------------------------------------------------------------------------

/// A lightweight handle to an HDF5 dataset.
pub struct Dataset<'f> {
    file: &'f File,
    header: ObjectHeader,
    // Held per-dataset: the chunk index is keyed only by chunk coordinate, so
    // a file-level cache would alias chunk addresses across datasets.
    chunk_cache: ChunkCache,
    // The effective chunk-cache config for this dataset: the file-wide default
    // or a per-dataset DAPL override. Reported by `chunk_cache_config`.
    chunk_cache_config: ChunkCacheConfig,
}

impl<'f> std::fmt::Debug for Dataset<'f> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dataset")
            .field("messages", &self.header.messages.len())
            .finish()
    }
}

impl<'f> Dataset<'f> {
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

    /// Returns the simplified datatype of the dataset.
    pub fn dtype(&self) -> Result<DType, Error> {
        let dt = self.datatype()?;
        Ok(classify_datatype(&dt))
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
    pub fn read_i8(&self) -> Result<Vec<i8>, Error> {
        let raw = self.read_raw()?;
        Ok(raw.iter().map(|&b| b as i8).collect())
    }

    /// Read all data as `i16` values.
    pub fn read_i16(&self) -> Result<Vec<i16>, Error> {
        let raw = self.read_raw()?;
        let dt = self.datatype()?;
        let vals = data_read::read_as_i32(&raw, &dt)?;
        Ok(vals.into_iter().map(|v| v as i16).collect())
    }

    /// Read all data as `u16` values.
    pub fn read_u16(&self) -> Result<Vec<u16>, Error> {
        let raw = self.read_raw()?;
        let dt = self.datatype()?;
        let vals = data_read::read_as_u64(&raw, &dt)?;
        Ok(vals.into_iter().map(|v| v as u16).collect())
    }

    /// Read all data as `u32` values.
    pub fn read_u32(&self) -> Result<Vec<u32>, Error> {
        let raw = self.read_raw()?;
        let dt = self.datatype()?;
        let vals = data_read::read_as_u64(&raw, &dt)?;
        Ok(vals.into_iter().map(|v| v as u32).collect())
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

    pub(crate) fn filter_pipeline(&self) -> Option<FilterPipeline> {
        self.header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::FilterPipeline)
            .and_then(|msg| FilterPipeline::parse(&msg.data).ok())
    }

    /// Read the dataset's exact unfiltered element bytes.
    ///
    /// For compound datasets this preserves all file padding and uses the
    /// offsets reported by [`datatype`](Self::datatype).
    pub fn read_raw(&self) -> Result<Vec<u8>, Error> {
        let dt = self.datatype()?;
        let ds = self.dataspace()?;
        let mut dl = self.data_layout()?;
        // Adjust contiguous data address by base_address offset
        if self.file.addr_offset != 0
            && let DataLayout::Contiguous {
                ref mut address, ..
            } = dl
            && let Some(addr) = address
        {
            *addr += self.file.addr_offset;
        }
        let pipeline = self.filter_pipeline();
        Ok(self
            .file
            .read_dataset_raw(&dl, &ds, &dt, pipeline.as_ref(), &self.chunk_cache)?)
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
}
