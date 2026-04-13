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
    /// Invalid Group Info message version.
    InvalidGroupInfoVersion(u8),
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
    /// Invalid attribute message version.
    InvalidAttributeVersion(u8),
    /// Invalid Attribute Info message version.
    InvalidAttributeInfoVersion(u8),
    /// Invalid shared message version.
    InvalidSharedMessageVersion(u8),
    /// Invalid SOHM table version.
    InvalidSohmTableVersion(u8),
    /// Invalid SOHM table signature (expected "SMTB").
    InvalidSohmTableSignature,
    /// Invalid SOHM list signature (expected "SMLI").
    InvalidSohmListSignature,
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
    /// Serialization error.
    SerializationError(String),
    /// Dataset is missing data.
    DatasetMissingData,
    /// Dataset is missing shape.
    DatasetMissingShape,
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
            FormatError::InvalidGroupInfoVersion(v) => {
                write!(f, "invalid group info message version: {v}")
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
            FormatError::InvalidAttributeVersion(v) => {
                write!(f, "invalid attribute message version: {v}")
            }
            FormatError::InvalidAttributeInfoVersion(v) => {
                write!(f, "invalid attribute info message version: {v}")
            }
            FormatError::InvalidSharedMessageVersion(v) => {
                write!(f, "invalid shared message version: {v}")
            }
            FormatError::InvalidSohmTableVersion(v) => {
                write!(f, "invalid SOHM table version: {v}")
            }
            FormatError::InvalidSohmTableSignature => {
                write!(f, "invalid SOHM table signature (expected SMTB)")
            }
            FormatError::InvalidSohmListSignature => {
                write!(f, "invalid SOHM list signature (expected SMLI)")
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
                write!(f, "global heap object not found: collection {collection_address:#x}, index {index}")
            }
            FormatError::VlDataError(msg) => {
                write!(f, "variable-length data error: {msg}")
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
