//! Error types for HDF5 format parsing.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::string::String;

#[cfg(feature = "std")]
use std::string::String;

use core::fmt;

/// Errors that can occur when parsing HDF5 binary format structures.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FormatError {
    /// The HDF5 magic signature was not found at any valid offset.
    SignatureNotFound,
    /// The superblock version is not supported.
    UnsupportedVersion(u8),
    /// Unexpected end of data.
    UnexpectedEof {
        /// Number of bytes expected.
        expected: usize,
        /// Number of bytes actually available.
        available: usize,
    },
    /// Invalid offset size (must be 2, 4, or 8).
    InvalidOffsetSize(u8),
    /// Invalid length size (must be 2, 4, or 8).
    InvalidLengthSize(u8),
    /// Invalid object header signature.
    InvalidObjectHeaderSignature,
    /// Invalid object header version.
    InvalidObjectHeaderVersion(u8),
    /// Unknown message type that is marked as must-understand.
    UnsupportedMessage(u16),
    /// Invalid datatype class.
    InvalidDatatypeClass(u8),
    /// Invalid datatype version for a given class.
    InvalidDatatypeVersion {
        /// The type class.
        class: u8,
        /// The version found.
        version: u8,
    },
    /// Invalid string padding type.
    InvalidStringPadding(u8),
    /// Invalid character set.
    InvalidCharacterSet(u8),
    /// Invalid byte order.
    InvalidByteOrder(u8),
    /// Invalid reference type.
    InvalidReferenceType(u8),
    /// Invalid file-space management strategy code in a File Space Info message.
    InvalidFileSpaceStrategy(u8),
    /// Unsupported File Space Info message version (only version 1 is handled).
    UnsupportedFileSpaceInfoVersion(u8),
    /// A free-space manager block (`FSHD`/`FSSE`) is malformed.
    InvalidFreeSpaceManager,
    /// A compound datatype has a zero total size.
    InvalidCompoundSize,
    /// A compound datatype contains no fields.
    EmptyCompoundType,
    /// A compound datatype contains the same field name more than once.
    DuplicateCompoundField(String),
    /// A compound field extends past the declared compound size.
    CompoundFieldOutOfBounds {
        /// Field name.
        name: String,
        /// Field byte offset.
        offset: u64,
        /// Field size in bytes.
        field_size: u32,
        /// Declared compound size in bytes.
        compound_size: u32,
    },
    /// Two compound fields overlap.
    CompoundFieldOverlap {
        /// Earlier field in byte order.
        first: String,
        /// Later field in byte order.
        second: String,
    },
    /// A named compound field was not present.
    CompoundFieldMissing(String),
    /// A compound field has an incompatible datatype.
    CompoundFieldTypeMismatch(String),
    /// Invalid dataspace version.
    InvalidDataspaceVersion(u8),
    /// Invalid dataspace type.
    InvalidDataspaceType(u8),
    /// Invalid data layout version.
    InvalidLayoutVersion(u8),
    /// Invalid data layout class.
    InvalidLayoutClass(u8),
    /// No data allocated for contiguous layout.
    NoDataAllocated,
    /// Type mismatch when reading data.
    TypeMismatch {
        /// Expected type description.
        expected: &'static str,
        /// Actual type description.
        actual: &'static str,
    },
    /// Data size mismatch.
    DataSizeMismatch {
        /// Expected size in bytes.
        expected: usize,
        /// Actual size in bytes.
        actual: usize,
    },
    /// Invalid local heap signature.
    InvalidLocalHeapSignature,
    /// Invalid local heap version.
    InvalidLocalHeapVersion(u8),
    /// Invalid B-tree v1 signature.
    InvalidBTreeSignature,
    /// Invalid B-tree node type.
    InvalidBTreeNodeType(u8),
    /// Invalid symbol table node signature.
    InvalidSymbolTableNodeSignature,
    /// Invalid symbol table node version.
    InvalidSymbolTableNodeVersion(u8),
    /// Path not found during group traversal.
    PathNotFound(String),
    /// Invalid Link message version.
    InvalidLinkVersion(u8),
    /// Invalid link type code.
    InvalidLinkType(u8),
    /// Invalid Link Info message version.
    InvalidLinkInfoVersion(u8),
    /// Invalid B-tree v2 signature.
    InvalidBTreeV2Signature,
    /// Invalid B-tree v2 version.
    InvalidBTreeV2Version(u8),
    /// Invalid fractal heap signature.
    InvalidFractalHeapSignature,
    /// Invalid fractal heap version.
    InvalidFractalHeapVersion(u8),
    /// Invalid heap ID type.
    InvalidHeapIdType(u8),
    /// A fractal-heap "huge" object's heap ID referenced a B-tree key that is
    /// not present in the heap's huge-objects v2 B-tree.
    HugeObjectNotFound(u64),
    /// A fractal-heap object lives in an I/O-filter-encoded heap (filtered
    /// managed or huge storage), whose filtered bytes this reader does not
    /// decode. Link and attribute heaps are never filtered, so this does not
    /// arise for them.
    UnsupportedFilteredHeapObject,
    /// A dataset uses the Virtual (VDS) data layout, which maps its elements to
    /// regions of other datasets, possibly in other files. This reader does not
    /// yet resolve virtual mappings, so such a dataset is refused rather than
    /// read as empty or wrong.
    UnsupportedVirtualLayout,
    /// Invalid attribute message version.
    InvalidAttributeVersion(u8),
    /// Invalid Attribute Info message version.
    InvalidAttributeInfoVersion(u8),
    /// Invalid shared message version.
    InvalidSharedMessageVersion(u8),
    /// Invalid global heap collection signature.
    InvalidGlobalHeapSignature,
    /// Invalid global heap version.
    InvalidGlobalHeapVersion(u8),
    /// Global heap object not found.
    GlobalHeapObjectNotFound {
        /// Address of the collection.
        collection_address: u64,
        /// Index that was not found.
        index: u16,
    },
    /// Variable-length data error.
    VlDataError(String),
    /// A variable-length read exceeded its configured element limit.
    VariableLengthElementLimitExceeded {
        /// Maximum number of elements permitted by the caller.
        limit: usize,
        /// Number of elements present in the selected data.
        actual: u64,
    },
    /// A variable-length read exceeded its configured payload-byte limit.
    VariableLengthByteLimitExceeded {
        /// Maximum number of payload bytes permitted by the caller.
        limit: usize,
        /// Number of payload bytes required by the selected data.
        required: u64,
    },
    /// Serialization error.
    SerializationError(String),
    /// Dataset is missing data.
    DatasetMissingData,
    /// Dataset is missing shape.
    DatasetMissingShape,
    /// A variable-length string dataset was requested with chunked, filtered,
    /// or resizable storage. VL element references live in the global heap,
    /// whose addresses are only known after data layout, so they cannot be
    /// patched into compressed chunks written beforehand.
    ChunkedVlenStringUnsupported,
    /// The dataset's element count implied by its shape does not match the
    /// amount of data supplied (`shape.product() * element_size != data.len()`).
    ShapeDataMismatch {
        /// Number of data bytes the shape requires (`product(shape) * element_size`).
        expected: usize,
        /// Number of data bytes actually supplied.
        actual: usize,
        /// Size in bytes of one element (the dataset's datatype size). Always
        /// non-zero; used to report the mismatch in elements as well as bytes.
        element_size: usize,
    },
    /// A chunked/filtered/extensible dataset's chunk geometry is invalid — for
    /// example chunk dimensions whose rank disagrees with the shape, a zero chunk
    /// dimension, a maximum shape whose rank disagrees with the shape or that is
    /// smaller than the current shape, or chunking requested on a scalar dataset.
    /// Reported up front so a malformed request is refused instead of panicking
    /// in the chunk splitter or producing an unreadable dataset. The payload is a
    /// human-readable reason.
    InvalidChunkGeometry(&'static str),
    /// Invalid filter pipeline version.
    InvalidFilterPipelineVersion(u8),
    /// Unsupported filter ID.
    UnsupportedFilter(u16),
    /// Filter processing error.
    FilterError(String),
    /// Decompression error.
    DecompressionError(String),
    /// Compression error.
    CompressionError(String),
    /// Fletcher32 checksum mismatch.
    Fletcher32Mismatch {
        /// Expected checksum.
        expected: u32,
        /// Computed checksum.
        computed: u32,
    },
    /// Chunked dataset read error.
    ChunkedReadError(String),
    /// Chunk assembly error.
    ChunkAssemblyError(String),
    /// CRC32C checksum mismatch.
    ChecksumMismatch {
        /// The checksum stored in the file.
        expected: u32,
        /// The checksum we computed.
        computed: u32,
    },
    /// Maximum nesting/continuation depth exceeded (malformed data protection).
    NestingDepthExceeded,
    /// Duplicate dataset name detected during parallel metadata merge.
    DuplicateDatasetName(String),
    /// ZFP filter configuration is invalid (e.g. missing element type, rank out of range).
    UnsupportedZfp(String),
    /// A file-derived 64-bit value (an offset, length, size, or element count)
    /// does not fit in the target integer type on this platform. This is the
    /// guard that replaces silent `as usize` / `as u32` truncation: on a 32-bit
    /// host, `usize` is 32 bits, so an HDF5 offset or length above `usize::MAX`
    /// would otherwise wrap and read the wrong bytes. The original value is
    /// preserved for diagnostics, and `target` names the type we tried to
    /// narrow to (e.g. `"usize"`, `"u32"`).
    ValueTooLargeForPlatform {
        /// The original 64-bit value read from the file.
        value: u64,
        /// The platform integer type the value could not fit into.
        target: &'static str,
    },
    /// Two file-derived values (typically an offset and a length) overflow `u64`
    /// when added to form a slice bound. Reported instead of wrapping so a
    /// malformed file cannot produce a wrapped or out-of-range index.
    OffsetOverflow {
        /// First operand (typically the base offset/address).
        offset: u64,
        /// Second operand (typically the length/size).
        length: u64,
    },
    /// A random-access byte source failed to
    /// supply the requested bytes. The string carries a backend-specific reason
    /// (e.g. an underlying `std::io::Error` rendered to text), so this stays
    /// `no_std`/`alloc`-friendly and free of an `std::io` dependency.
    Source(String),
    /// The library-version bounds requested via
    /// [`FileBuilder::with_libver_bounds`](crate::FileBuilder::with_libver_bounds)
    /// cannot be satisfied. This crate's writer emits exactly one on-disk format
    /// (the version 3 / HDF5 1.10 superblock), so a bound that excludes it — an
    /// upper bound older than 1.10, or a lower bound newer than 1.10 — is
    /// unsatisfiable. The fields carry the format produced and the bounds asked
    /// for, as [`LibVer::name`](crate::LibVer::name) labels.
    LibverBoundsUnsatisfiable {
        /// The library-version label of the format this crate writes.
        writes: &'static str,
        /// The requested lower bound.
        requested_low: &'static str,
        /// The requested upper bound.
        requested_high: &'static str,
    },
    /// An HDF5 object reference (`H5R_OBJECT`) could not be resolved to an
    /// object: the stored address is null or undefined (`HADDR_UNDEF`), or it
    /// does not point at a group or dataset object header. The payload is the
    /// stored (base-relative) address, preserved for diagnostics.
    InvalidObjectReference(u64),
    /// A Fill Value message (`0x0005`) has an on-disk version this crate does
    /// not recognize (only 1, 2, and 3 are defined). The payload is the version
    /// byte found.
    UnsupportedFillValueVersion(u8),
    /// A user-supplied fill value's byte width does not match the dataset's
    /// datatype element size (for example a `u8` fill value on an `i32`
    /// dataset). The fields carry the datatype element size and the fill value
    /// size, both in bytes.
    FillValueSizeMismatch {
        /// The dataset datatype's element size in bytes.
        expected: usize,
        /// The supplied fill value's size in bytes.
        actual: usize,
    },
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FormatError::SignatureNotFound => {
                write!(f, "HDF5 signature not found at any valid offset")
            }
            FormatError::UnsupportedVersion(v) => {
                write!(f, "unsupported superblock version: {v}")
            }
            FormatError::UnexpectedEof {
                expected,
                available,
            } => {
                write!(f, "unexpected EOF: need {expected} bytes, have {available}")
            }
            FormatError::InvalidOffsetSize(s) => {
                write!(f, "invalid offset size: {s} (must be 2, 4, or 8)")
            }
            FormatError::InvalidLengthSize(s) => {
                write!(f, "invalid length size: {s} (must be 2, 4, or 8)")
            }
            FormatError::InvalidObjectHeaderSignature => {
                write!(f, "invalid object header signature")
            }
            FormatError::InvalidObjectHeaderVersion(v) => {
                write!(f, "invalid object header version: {v}")
            }
            FormatError::UnsupportedMessage(id) => {
                write!(
                    f,
                    "unsupported message type {id:#06x} marked as must-understand"
                )
            }
            FormatError::InvalidDatatypeClass(c) => {
                write!(f, "invalid datatype class: {c}")
            }
            FormatError::InvalidDatatypeVersion { class, version } => {
                write!(f, "invalid datatype version {version} for class {class}")
            }
            FormatError::InvalidStringPadding(p) => {
                write!(f, "invalid string padding type: {p}")
            }
            FormatError::InvalidCharacterSet(c) => {
                write!(f, "invalid character set: {c}")
            }
            FormatError::InvalidByteOrder(b) => {
                write!(f, "invalid byte order: {b}")
            }
            FormatError::InvalidReferenceType(r) => {
                write!(f, "invalid reference type: {r}")
            }
            FormatError::InvalidFileSpaceStrategy(s) => {
                write!(f, "invalid file-space strategy code: {s}")
            }
            FormatError::UnsupportedFileSpaceInfoVersion(v) => {
                write!(f, "unsupported File Space Info message version: {v}")
            }
            FormatError::InvalidFreeSpaceManager => {
                write!(f, "malformed free-space manager block (FSHD/FSSE)")
            }
            FormatError::InvalidCompoundSize => {
                write!(f, "compound datatype size must be greater than zero")
            }
            FormatError::EmptyCompoundType => {
                write!(f, "compound datatype must contain at least one field")
            }
            FormatError::DuplicateCompoundField(name) => {
                write!(f, "duplicate compound field name: {name}")
            }
            FormatError::CompoundFieldOutOfBounds {
                name,
                offset,
                field_size,
                compound_size,
            } => {
                write!(
                    f,
                    "compound field {name:?} at offset {offset} with size {field_size} \
                     exceeds compound size {compound_size}"
                )
            }
            FormatError::CompoundFieldOverlap { first, second } => {
                write!(f, "compound fields {first:?} and {second:?} overlap")
            }
            FormatError::CompoundFieldMissing(name) => {
                write!(f, "compound field {name:?} is missing")
            }
            FormatError::CompoundFieldTypeMismatch(name) => {
                write!(f, "compound field {name:?} has an incompatible datatype")
            }
            FormatError::InvalidDataspaceVersion(v) => {
                write!(f, "invalid dataspace version: {v}")
            }
            FormatError::InvalidDataspaceType(t) => {
                write!(f, "invalid dataspace type: {t}")
            }
            FormatError::InvalidLayoutVersion(v) => {
                write!(f, "invalid data layout version: {v}")
            }
            FormatError::InvalidLayoutClass(c) => {
                write!(f, "invalid data layout class: {c}")
            }
            FormatError::NoDataAllocated => {
                write!(f, "no data allocated for contiguous layout")
            }
            FormatError::TypeMismatch { expected, actual } => {
                write!(f, "type mismatch: expected {expected}, got {actual}")
            }
            FormatError::DataSizeMismatch { expected, actual } => {
                write!(
                    f,
                    "data size mismatch: expected {expected} bytes, got {actual} bytes"
                )
            }
            FormatError::InvalidLocalHeapSignature => {
                write!(f, "invalid local heap signature")
            }
            FormatError::InvalidLocalHeapVersion(v) => {
                write!(f, "invalid local heap version: {v}")
            }
            FormatError::InvalidBTreeSignature => {
                write!(f, "invalid B-tree v1 signature")
            }
            FormatError::InvalidBTreeNodeType(t) => {
                write!(f, "invalid B-tree node type: {t}")
            }
            FormatError::InvalidSymbolTableNodeSignature => {
                write!(f, "invalid symbol table node signature")
            }
            FormatError::InvalidSymbolTableNodeVersion(v) => {
                write!(f, "invalid symbol table node version: {v}")
            }
            FormatError::PathNotFound(p) => {
                write!(f, "path not found: {p}")
            }
            FormatError::InvalidLinkVersion(v) => {
                write!(f, "invalid link message version: {v}")
            }
            FormatError::InvalidLinkType(t) => {
                write!(f, "invalid link type: {t}")
            }
            FormatError::InvalidLinkInfoVersion(v) => {
                write!(f, "invalid link info message version: {v}")
            }
            FormatError::InvalidBTreeV2Signature => {
                write!(f, "invalid B-tree v2 signature")
            }
            FormatError::InvalidBTreeV2Version(v) => {
                write!(f, "invalid B-tree v2 version: {v}")
            }
            FormatError::InvalidFractalHeapSignature => {
                write!(f, "invalid fractal heap signature")
            }
            FormatError::InvalidFractalHeapVersion(v) => {
                write!(f, "invalid fractal heap version: {v}")
            }
            FormatError::InvalidHeapIdType(t) => {
                write!(f, "invalid heap ID type: {t}")
            }
            FormatError::HugeObjectNotFound(id) => {
                write!(f, "fractal-heap huge object {id} not found in B-tree")
            }
            FormatError::UnsupportedFilteredHeapObject => {
                write!(f, "filtered fractal-heap objects are not supported")
            }
            FormatError::UnsupportedVirtualLayout => {
                write!(f, "virtual (VDS) data layout is not supported")
            }
            FormatError::InvalidAttributeVersion(v) => {
                write!(f, "invalid attribute message version: {v}")
            }
            FormatError::InvalidAttributeInfoVersion(v) => {
                write!(f, "invalid attribute info message version: {v}")
            }
            FormatError::InvalidSharedMessageVersion(v) => {
                write!(f, "invalid shared message version: {v}")
            }
            FormatError::InvalidGlobalHeapSignature => {
                write!(f, "invalid global heap collection signature")
            }
            FormatError::InvalidGlobalHeapVersion(v) => {
                write!(f, "invalid global heap version: {v}")
            }
            FormatError::GlobalHeapObjectNotFound {
                collection_address,
                index,
            } => {
                write!(
                    f,
                    "global heap object not found: collection {collection_address:#x}, index {index}"
                )
            }
            FormatError::VlDataError(msg) => {
                write!(f, "variable-length data error: {msg}")
            }
            FormatError::VariableLengthElementLimitExceeded { limit, actual } => {
                write!(
                    f,
                    "variable-length element limit exceeded: limit is {limit}, data contains {actual}"
                )
            }
            FormatError::VariableLengthByteLimitExceeded { limit, required } => {
                write!(
                    f,
                    "variable-length payload limit exceeded: limit is {limit} bytes, \
                     data requires {required} bytes"
                )
            }
            FormatError::SerializationError(msg) => {
                write!(f, "serialization error: {msg}")
            }
            FormatError::DatasetMissingData => {
                write!(f, "dataset is missing data")
            }
            FormatError::DatasetMissingShape => {
                write!(f, "dataset is missing shape")
            }
            FormatError::ChunkedVlenStringUnsupported => {
                write!(
                    f,
                    "chunked, filtered, or resizable variable-length string datasets cannot be written"
                )
            }
            FormatError::ShapeDataMismatch {
                expected,
                actual,
                element_size,
            } => {
                // `element_size` is guaranteed non-zero at construction, so the
                // element counts below are well defined.
                write!(
                    f,
                    "shape/data mismatch: shape requires {} elements ({expected} bytes), \
                     but {} elements ({actual} bytes) were supplied",
                    expected / element_size,
                    actual / element_size,
                )
            }
            FormatError::InvalidChunkGeometry(reason) => {
                write!(f, "invalid chunk geometry: {reason}")
            }
            FormatError::InvalidFilterPipelineVersion(v) => {
                write!(f, "invalid filter pipeline version: {v}")
            }
            FormatError::UnsupportedFilter(id) => {
                write!(f, "unsupported filter: {id}")
            }
            FormatError::FilterError(msg) => {
                write!(f, "filter error: {msg}")
            }
            FormatError::DecompressionError(msg) => {
                write!(f, "decompression error: {msg}")
            }
            FormatError::CompressionError(msg) => {
                write!(f, "compression error: {msg}")
            }
            FormatError::Fletcher32Mismatch { expected, computed } => {
                write!(
                    f,
                    "fletcher32 mismatch: expected {expected:#010x}, computed {computed:#010x}"
                )
            }
            FormatError::ChunkedReadError(msg) => {
                write!(f, "chunked read error: {msg}")
            }
            FormatError::ChunkAssemblyError(msg) => {
                write!(f, "chunk assembly error: {msg}")
            }
            FormatError::ChecksumMismatch { expected, computed } => {
                write!(
                    f,
                    "checksum mismatch: expected {expected:#010x}, computed {computed:#010x}"
                )
            }
            FormatError::NestingDepthExceeded => {
                write!(f, "maximum nesting/continuation depth exceeded")
            }
            FormatError::DuplicateDatasetName(name) => {
                write!(f, "duplicate dataset name during parallel merge: {name}")
            }
            FormatError::UnsupportedZfp(msg) => {
                write!(f, "unsupported ZFP configuration: {msg}")
            }
            FormatError::ValueTooLargeForPlatform { value, target } => {
                write!(
                    f,
                    "file value {value} does not fit in {target} on this platform \
                     (a 64-bit HDF5 offset/length exceeds this target's address width)"
                )
            }
            FormatError::OffsetOverflow { offset, length } => {
                write!(
                    f,
                    "offset arithmetic overflow: {offset} + {length} exceeds u64"
                )
            }
            FormatError::Source(msg) => {
                write!(f, "byte source error: {msg}")
            }
            FormatError::LibverBoundsUnsatisfiable {
                writes,
                requested_low,
                requested_high,
            } => {
                write!(
                    f,
                    "requested library-version bounds [{requested_low}, {requested_high}] \
                     cannot be satisfied: this crate writes the {writes} format"
                )
            }
            FormatError::InvalidObjectReference(addr) => {
                write!(
                    f,
                    "invalid HDF5 object reference: address {addr:#x} is null/undefined \
                     or does not point at a group or dataset"
                )
            }
            FormatError::UnsupportedFillValueVersion(v) => {
                write!(f, "unsupported fill value message version: {v}")
            }
            FormatError::FillValueSizeMismatch { expected, actual } => {
                write!(
                    f,
                    "fill value size {actual} bytes does not match the dataset datatype \
                     element size of {expected} bytes"
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for FormatError {}

// ---------------------------------------------------------------------------
// High-level Error type
// ---------------------------------------------------------------------------

/// Errors that can occur when using the high-level API.
#[cfg(feature = "std")]
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// I/O error from the filesystem.
    Io(std::io::Error),
    /// Low-level format parsing error.
    Format(FormatError),
    /// The object at the given path is not a dataset.
    NotADataset(String),
    /// A required header message was not found.
    MissingMessage(crate::message_type::MessageType),
    /// Alignment or size error for zero-copy typed access.
    AlignmentError(String),
    /// An array shape error from the `ndarray` integration: either the flat
    /// data could not be reshaped to the dataset's dimensions, or a requested
    /// static rank (e.g. `read_array::<_, Ix2>`) did not match the dataset's
    /// runtime rank. Only constructed when the `ndarray` feature is enabled.
    Shape(String),
    /// A SWMR operation (e.g. [`crate::File::refresh`]) was requested on a file
    /// that was not opened for SWMR reading via `File::open_swmr`.
    SwmrUnsupported,
    /// An operation that needs exclusive access to the open file (e.g.
    /// [`crate::File::refresh`]) was requested while owned [`crate::Dataset`] /
    /// [`crate::Group`] handles, or a clone of the [`crate::File`], are still
    /// alive. Drop them and retry.
    HandlesOutstanding,
    /// A write (e.g. [`crate::Dataset::append`]) was requested on a file opened
    /// read-only. Open it with [`crate::File::open_rw`] to modify it in place.
    ReadOnly,
    /// A write was requested through a handle whose [`crate::File`] has already
    /// been sealed by [`crate::File::close`]. Immediate and staged edits are
    /// refused; reads through surviving handles still work. Re-open the file to
    /// modify it again.
    FileClosed,
    /// A staged edit (`write` / `set_attr` / `create_*` / `delete` / `copy` /
    /// `commit`) was requested on a file opened with
    /// [`crate::File::open_swmr_writer`], which permits only immediate
    /// [`crate::Dataset::append`]. Committing a structural edit would clear the
    /// SWMR-write flag out from under a concurrent reader, so the whole staged
    /// surface is refused in SWMR-writer mode.
    SwmrStagedUnsupported,
    /// A staged edit (`write` / `set_attr` / `create_*` / `delete` / `copy` /
    /// `commit`) or another mirror-backed operation (e.g.
    /// [`crate::File::space_accounting`]) was requested on a file opened with
    /// [`crate::File::open_rw_bounded`], which keeps only bounded state in
    /// memory and supports reads plus immediate [`crate::Dataset::append`].
    /// Staged edits rebuild the object tree over a whole-file mirror; open the
    /// file with [`crate::File::open_rw`] for them.
    BoundedStagedUnsupported,
    /// The file or dataset is not a supported target for the SWMR append writer
    /// (e.g. a userblock or non-latest-format file, or a dataset that is
    /// filtered, not rank-1 with an unlimited dimension, or not
    /// Extensible-Array indexed). The payload is a human-readable reason.
    SwmrAppendUnsupported(&'static str),
    /// The dataset is not a supported target for
    /// [`EditSession::append_dataset`](crate::EditSession::append_dataset) — for
    /// example a dataset that is not chunked, not extensible along its first
    /// dimension, not indexed by an Extensible Array, higher than rank 1, uses a
    /// filter this engine cannot re-encode, has a big-endian on-disk element
    /// datatype (for a raw append), or has more than one hard link. The payload
    /// is a human-readable reason.
    AppendUnsupported(&'static str),
    /// The dataset or file is not a supported target for the fast, immediate
    /// in-place append
    /// ([`EditSession::append_inplace`](crate::EditSession::append_inplace)) — for
    /// example a userblock or non-latest-format file, a dataset whose
    /// Extensible-Array index is not yet allocated, one that is not rank-1 /
    /// unlimited / Extensible-Array indexed, one reachable through more than one
    /// hard link, or a path an uncommitted staged edit in the same session will
    /// relocate or delete. Distinct from [`AppendUnsupported`](Self::AppendUnsupported)
    /// so a caller can catch this fast-path refusal and fall back to the staged
    /// [`EditSession::append_dataset`](crate::EditSession::append_dataset). The
    /// payload is a human-readable reason.
    AppendInPlaceUnsupported(&'static str),
    /// The file or the requested object is not a supported target for the
    /// in-place editor ([`crate::EditSession`]) — for example a userblock or
    /// non-latest-format file, a group whose links are densely stored, or a
    /// dataset shape/datatype/filter combination the in-place writer cannot
    /// emit yet. The payload is a human-readable reason.
    EditUnsupported(&'static str),
    /// An object in the source file cannot be reproduced faithfully by
    /// [`repack`](crate::repack), so the repack was refused rather than write a
    /// silently degraded file — for example a variable-length, time, bitfield,
    /// or opaque datatype, a virtual/external data layout, an unsupported
    /// filter, or an object reference. The payload names the object and reason.
    RepackUnsupported(String),
    /// The file could not be opened because another process holds a conflicting
    /// OS advisory lock — for a writer ([`crate::SwmrWriter`],
    /// [`crate::EditSession`]) this means another writer or reader is active;
    /// for a plain reader it means a writer is active. The lock is released
    /// automatically when the holder's process exits, so a crashed writer does
    /// not leave a stale lock. Locking can be disabled per open with
    /// [`crate::FileLocking::Disabled`] or globally with
    /// `HDF5_USE_FILE_LOCKING=FALSE`. The payload is a human-readable reason.
    FileLocked(String),
}

#[cfg(feature = "std")]
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Format(e) => write!(f, "HDF5 format error: {e}"),
            Error::NotADataset(path) => write!(f, "not a dataset: {path}"),
            Error::MissingMessage(mt) => write!(f, "missing required message: {mt:?}"),
            Error::AlignmentError(msg) => write!(f, "alignment error: {msg}"),
            Error::Shape(msg) => write!(f, "array shape error: {msg}"),
            Error::SwmrUnsupported => write!(
                f,
                "refresh requires a file opened with File::open_swmr (live handle)"
            ),
            Error::HandlesOutstanding => write!(
                f,
                "operation needs exclusive file access: drop outstanding Dataset/Group handles and File clones first"
            ),
            Error::ReadOnly => write!(
                f,
                "cannot write to a read-only file; open it with File::open_rw"
            ),
            Error::FileClosed => write!(
                f,
                "cannot write through a handle after File::close; re-open the file to modify it"
            ),
            Error::SwmrStagedUnsupported => write!(
                f,
                "a file opened with File::open_swmr_writer allows only immediate Dataset::append, not staged edits"
            ),
            Error::BoundedStagedUnsupported => write!(
                f,
                "a file opened with File::open_rw_bounded allows reads and immediate Dataset::append only; open it with File::open_rw for staged edits"
            ),
            Error::SwmrAppendUnsupported(reason) => {
                write!(f, "unsupported SWMR append target: {reason}")
            }
            Error::AppendUnsupported(reason) => {
                write!(f, "unsupported append target: {reason}")
            }
            Error::AppendInPlaceUnsupported(reason) => {
                write!(f, "unsupported in-place append target: {reason}")
            }
            Error::EditUnsupported(reason) => {
                write!(f, "unsupported in-place edit target: {reason}")
            }
            Error::RepackUnsupported(reason) => {
                write!(f, "cannot repack faithfully: {reason}")
            }
            Error::FileLocked(reason) => write!(f, "file is locked: {reason}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::Format(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(feature = "std")]
impl From<FormatError> for Error {
    fn from(e: FormatError) -> Self {
        Error::Format(e)
    }
}

#[cfg(feature = "std")]
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
